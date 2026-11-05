from __future__ import annotations

import time
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from .ecc import SECDED


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


class FrozenLM:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; use configs/memory_cpu.yaml")
        dtype = cfg.get("dtype", "auto")
        if dtype == "auto":
            dtype = "bfloat16" if self.device.type == "cuda" and torch.cuda.is_bf16_supported() else "float32"
        self.dtype = getattr(torch, dtype)
        if self.device.type == "cpu" and self.dtype == torch.float16:
            raise ValueError("CPU float16 unsupported; use float32")
        common = dict(revision=cfg.get("revision", "main"), cache_dir=cfg.get("model_cache", ".cache-memory"))
        self.tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], **common)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(cfg["model_id"], torch_dtype=self.dtype,
            attn_implementation="sdpa", **common).to(self.device).eval()
        self.model.requires_grad_(False)
        self.dim = self.model.get_input_embeddings().embedding_dim
        self.revision = getattr(self.model.config, "_commit_hash", common["revision"])
        enc_common = dict(revision=cfg.get("encoder_revision", "main"), cache_dir=common["cache_dir"])
        self.encoder_tokenizer = AutoTokenizer.from_pretrained(cfg["encoder_id"], **enc_common)
        self.encoder = AutoModel.from_pretrained(cfg["encoder_id"], **enc_common).to(self.device).eval()
        self.encoder.requires_grad_(False)
        self.encoder_dim = self.encoder.config.hidden_size
        self.encoder_revision = getattr(self.encoder.config, "_commit_hash", enc_common["revision"])

    @torch.inference_mode()
    def encode(self, texts, batch_size=16):
        features = []
        tok = self.encoder_tokenizer
        previous = tok.padding_side
        tok.padding_side = "right"
        for start in range(0, len(texts), batch_size):
            batch = tok(texts[start:start + batch_size], padding=True, truncation=True,
                max_length=self.cfg.get("encoder_length", 128), return_tensors="pt").to(self.device)
            hidden = self.encoder(**batch).last_hidden_state.float()
            mask = batch.attention_mask.unsqueeze(-1)
            mean = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            # A frozen pretrained transformer, no trainable question encoder.
            features.append(F.normalize(mean, dim=-1))
        tok.padding_side = previous
        return torch.cat(features).clone() if features else torch.empty(0, self.dim, device=self.device)

    def prompt(self, question, context=""):
        messages = [dict(role="system", content=self.cfg.get("system_prompt", ("Use company memory to answer. Return only the requested values, "
            "separated by ;. If information is missing return UNKNOWN. Do not explain."))),
            dict(role="user", content=f"Company memory:\n{context or '(none)'}\nQuestion: {question}")]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def pack(self, questions, contexts, prefix=None, answers=None):
        tok = self.tokenizer
        sequences, targets = [], []
        prefix_len = 0 if prefix is None else prefix.shape[1]
        for i, (q, c) in enumerate(zip(questions, contexts)):
            prompt = tok.encode(self.prompt(q, c), add_special_tokens=False)
            max_prompt = self.cfg.get("max_prompt_tokens", 384)
            if len(prompt) > max_prompt:
                raise ValueError(f"Prompt has {len(prompt)} tokens, exceeds {max_prompt}; increase config limit")
            answer = [] if answers is None else tok.encode(answers[i], add_special_tokens=False) + [tok.eos_token_id]
            if len(answer) > self.cfg.get("max_new_tokens", 24):
                raise ValueError("Answer exceeds shared generation budget")
            sequences.append(prompt + answer)
            targets.append([-100] * (prefix_len + len(prompt)) + answer)
        width = max(map(len, sequences))
        ids, masks, labels = [], [], []
        for seq, target in zip(sequences, targets):
            pad = width - len(seq)
            ids.append([tok.pad_token_id] * pad + seq)
            masks.append([0] * pad + [1] * (prefix_len + len(seq)))
            labels.append([-100] * pad + target)
        ids = torch.tensor(ids, device=self.device)
        embeds = self.model.get_input_embeddings()(ids)
        if prefix is not None:
            # Insert before each unpadded prompt; left padding stays left of prefix.
            rows = []
            for i, seq in enumerate(sequences):
                pad = width - len(seq)
                rows.append(torch.cat((embeds[i, :pad], prefix[i].to(self.dtype), embeds[i, pad:]), dim=0))
            embeds = torch.stack(rows)
        mask = torch.tensor(masks, device=self.device)
        positions = (mask.cumsum(-1) - 1).clamp_min(0)
        return dict(inputs_embeds=embeds, attention_mask=mask, position_ids=positions), torch.tensor(labels, device=self.device)

    def answer_loss(self, questions, contexts, prefix, answers):
        inputs, labels = self.pack(questions, contexts, prefix, answers)
        # No no_grad here: frozen LM must propagate gradients to prefix.
        if self.cfg.get("answer_logits_only", False) and self.model.config.model_type == "qwen2":
            hidden = self.model.base_model(**inputs, use_cache=False).last_hidden_state
            labels_next = labels[:, 1:]
            valid = labels_next != -100
            logits = self.model.get_output_embeddings()(hidden[:, :-1][valid])
            return F.cross_entropy(logits.float(), labels_next[valid])
        result = self.model(**inputs, labels=labels, use_cache=False)
        return result.loss

    @torch.inference_mode()
    def generate(self, questions, contexts, prefix=None, max_new_tokens=None):
        pack_start = time.perf_counter()
        inputs, _ = self.pack(questions, contexts, prefix)
        sync(self.device)
        preparation = time.perf_counter() - pack_start
        start = time.perf_counter()
        extra = {"logits_to_keep": 1} if self.cfg.get("last_token_logits", False) and self.model.config.model_type == "qwen2" else {}
        output = self.model(**inputs, use_cache=True, **extra)
        next_id = output.logits[:, -1].argmax(-1)
        sync(self.device)
        prefill = time.perf_counter() - start
        generated = [next_id]
        finished = next_id == self.tokenizer.eos_token_id
        mask = inputs["attention_mask"]
        past = output.past_key_values
        decode_start = time.perf_counter()
        steps = max_new_tokens or self.cfg.get("max_new_tokens", 24)
        # Fixed budget avoids per-token CPU/GPU synchronization and matches all arms.
        for _ in range(steps - 1):
            if self.cfg.get("stop_on_eos", False) and bool(finished.all()):
                break
            mask = torch.cat((mask, torch.ones_like(mask[:, :1])), dim=1)
            pos = mask.sum(-1, keepdim=True) - 1
            output = self.model(input_ids=next_id[:, None], attention_mask=mask,
                position_ids=pos, past_key_values=past, use_cache=True)
            past = output.past_key_values
            next_id = output.logits[:, -1].argmax(-1)
            finished |= next_id == self.tokenizer.eos_token_id
            generated.append(next_id)
        sync(self.device)
        decode = time.perf_counter() - decode_start
        tokens = torch.stack(generated, dim=1).cpu().tolist()
        texts, counts = [], []
        for row in tokens:
            if self.tokenizer.eos_token_id in row:
                row = row[:row.index(self.tokenizer.eos_token_id)]
            texts.append(self.tokenizer.decode(row, skip_special_tokens=True).strip())
            counts.append(len(row))
        return texts, dict(prefill_s=prefill, decode_s=decode, preparation_s=preparation,
                           generated_tokens=sum(counts), budget_tokens=len(generated) * len(questions))


class AssociativeTrunk(nn.Module):
    def __init__(self, dim, hidden, depth):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.a = nn.Linear(dim, hidden)
        self.g = nn.Linear(dim, hidden)
        self.layers = nn.ModuleList([nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.SiLU())
            for _ in range(depth - 1)])
        self.out_norm = nn.LayerNorm(hidden)

    def forward(self, z):
        z = self.norm(z)
        h = self.a(z) * self.g(z)
        for layer in self.layers:
            h = h + layer(h)
        return self.out_norm(h)


class Memory(nn.Module):
    def __init__(self, dim, nfacts, variant, cfg, latent_dim=None):
        super().__init__()
        self.variant, self.nfacts = variant, nfacts
        self.slots, self.tokens = cfg.get("slots", 2), cfg.get("memory_tokens", 8)
        self.code_head = cfg.get("address_head", "categorical") == "code"
        self.codec = SECDED(nfacts + 1)
        hidden = cfg.get("hidden", 192)
        self.trunk = AssociativeTrunk(dim, hidden, cfg.get("depth", 2))
        self.dim = latent_dim or dim
        self.latent = nn.Linear(hidden, self.tokens * self.dim) if variant in ("latent", "hybrid") else None
        width = self.codec.width if self.code_head else nfacts + 1
        self.address = nn.Linear(hidden, self.slots * width) if variant in ("address", "hybrid") else None
        self.unknown = nn.Linear(hidden, 1) if self.address is not None else None
        if self.latent is not None:
            nn.init.normal_(self.latent.weight, std=0.001)
            nn.init.zeros_(self.latent.bias)

    def forward(self, z):
        h = self.trunk(z.float())
        prefix = self.latent(h).view(-1, self.tokens, self.dim) if self.latent is not None else None
        address = self.address(h).view(len(z), self.slots, -1) if self.address is not None else None
        unknown = self.unknown(h).squeeze(-1) if self.unknown is not None else None
        return prefix, address, unknown

    def targets(self, rows, facts, device):
        lookup = {f["fact_id"]: i for i, f in enumerate(facts)}
        targets = []
        for q in rows:
            ids = sorted(lookup[fid] for fid in q["supporting_fact_ids"])
            if len(ids) > self.slots:
                raise ValueError("More supporting facts than configured slots")
            targets.append(ids + [self.nfacts] * (self.slots - len(ids)))
        return torch.tensor(targets, device=device)

    def address_loss(self, logits, target):
        if not self.code_head:
            return F.cross_entropy(logits.flatten(0, 1), target.flatten())
        # Unknown is payload 0; fact i is payload i+1.
        payload = torch.where(target == self.nfacts, 0, target + 1)
        return F.binary_cross_entropy_with_logits(logits, self.codec.encode(payload).float())

    def addresses(self, logits, unknown, threshold=0.5):
        if self.code_head:
            payload, corrected, invalid = self.codec.decode(logits > 0)
            ids = torch.where(payload > 0, payload - 1, self.nfacts)
            ids = torch.where(invalid, self.nfacts, ids)
            statuses = dict(corrected=corrected, invalid=invalid)
        else:
            ids = logits.argmax(-1)
            statuses = {}
        ids = torch.where((unknown.sigmoid() > threshold)[:, None], self.nfacts, ids)
        return ids, statuses

    @torch.no_grad()
    def ridge_fit(self, z, targets, ridge=0.1):
        if not self.code_head or self.address is None:
            raise ValueError("Ridge initialization is for code address heads")
        h = self.trunk(z).double().cpu()
        x = torch.cat((h, torch.ones(len(h), 1, dtype=torch.float64)), dim=1)
        payload = torch.where(targets.cpu() == self.nfacts, 0, targets.cpu() + 1)
        y = (self.codec.encode(payload).double() * 2 - 1).flatten(1) * 3
        gram = x.T @ x + ridge * torch.eye(x.shape[1], dtype=torch.float64)
        solution = torch.linalg.solve(gram, x.T @ y)
        if not torch.isfinite(solution).all():
            raise FloatingPointError("Non-finite ridge solution")
        self.address.weight.copy_(solution[:-1].T.to(self.address.weight))
        self.address.bias.copy_(solution[-1].to(self.address.bias))
        return dict(ridge=ridge, gram_condition=float(torch.linalg.cond(gram)),
                    residual_mse=float(((x @ solution - y) ** 2).mean()), solver="float64 torch.linalg.solve")
