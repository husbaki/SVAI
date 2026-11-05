from __future__ import annotations

import copy
import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
import transformers
import yaml
from torch.nn import functional as F

from .data import UNKNOWN, dump, fact_text, fingerprint, read
from .models import FrozenLM, Memory, sync


def config(path):
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cfg["config_path"] = str(path)
    return cfg


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)


def environment():
    return dict(python=platform.python_version(), torch=torch.__version__, transformers=transformers.__version__,
        cuda=torch.version.cuda, cuda_available=torch.cuda.is_available(), platform=platform.platform(),
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        vram_bytes=torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0)


class Experiment:
    def __init__(self, cfg, load_llm=True):
        self.cfg = cfg
        self.root = Path(cfg["output"])
        self.data = Path(cfg["data"])
        self.root.mkdir(parents=True, exist_ok=True)
        self.facts = read(self.data / "facts.json")
        self.manifest = read(self.data / "manifest.json")
        seed_all(cfg.get("seed", 17))
        self.memories, self.artifacts = {}, {}
        start = time.perf_counter()
        self.lm = FrozenLM(cfg) if load_llm else None
        if self.lm:
            sync(self.lm.device)
        self.startup_s = time.perf_counter() - start
        self.rag_embeddings = None
        self.rag_index_s = 0.0
        self.rag_k = cfg.get("rag_k", 2)
        self.rag_threshold = cfg.get("rag_threshold", 0.0)

    def rows(self, split):
        return read(self.data / f"{split}.json")

    def features(self, rows, cache=True):
        key = fingerprint([self.cfg["encoder_id"], self.lm.encoder_revision,
            self.cfg.get("encoder_length", 128), "masked-mean-l2-v1", [q["question"] for q in rows]])
        path = self.root / "features" / f"{key}.pt"
        if cache and path.exists():
            return torch.load(path, map_location=self.lm.device, weights_only=True).clone()
        values = self.lm.encode([q["question"] for q in rows], self.cfg.get("encoder_batch", 32)).clone()
        if cache:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(values.cpu(), path)
        return values

    def make_memory(self, variant, cfg=None):
        return Memory(self.lm.encoder_dim, len(self.facts), variant, cfg or self.cfg,
                      latent_dim=self.lm.dim).to(self.lm.device)

    def checkpoint_path(self, variant):
        suffix = "_code" if variant == "address" and self.cfg.get("address_head") == "code" else ""
        return self.root / f"{variant}{suffix}.pt"

    def save(self, memory, meta):
        cfg = copy.deepcopy(self.cfg)
        cfg["revision"] = self.lm.revision
        cfg["encoder_revision"] = self.lm.encoder_revision
        artifact = dict(state={k: v.detach().cpu() for k, v in memory.state_dict().items()}, config=cfg,
            variant=memory.variant, nfacts=len(self.facts), encoder_dim=self.lm.encoder_dim, latent_dim=self.lm.dim,
            facts=self.facts if memory.variant != "latent" else [],
            corpus_version=self.manifest["version"], corpus_kind=self.manifest["kind"], meta=meta,
            environment=environment(), model_id=cfg["model_id"], tokenizer_id=cfg["model_id"])
        path = self.checkpoint_path(memory.variant)
        torch.save(artifact, path)
        dump(path.with_suffix(".json"), {k: v for k, v in artifact.items() if k not in ("state", "facts")})
        self.memories[memory.variant] = memory.eval()
        self.artifacts[memory.variant] = artifact
        return path

    def load(self, variant):
        if variant in self.memories:
            return self.memories[variant]
        start = time.perf_counter()
        path = self.checkpoint_path(variant)
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        if artifact["corpus_version"] != self.manifest["version"]:
            raise ValueError("Checkpoint belongs to a different corpus")
        if artifact["model_id"] != self.cfg["model_id"] or artifact["config"]["revision"] != self.lm.revision:
            raise ValueError("Checkpoint uses a different frozen LLM or revision")
        if artifact["config"]["encoder_id"] != self.cfg["encoder_id"] or artifact["config"]["encoder_revision"] != self.lm.encoder_revision:
            raise ValueError("Checkpoint uses a different frozen encoder or revision")
        memory = self.make_memory(variant, artifact["config"])
        memory.load_state_dict(artifact["state"])
        memory.eval()
        artifact["meta"]["checkpoint_load_s"] = time.perf_counter() - start
        self.memories[variant], self.artifacts[variant] = memory, artifact
        return memory

    def contexts(self, ids, facts=None):
        facts = self.facts if facts is None else facts
        contexts, selected, copied, sources = [], [], [], []
        for row in ids:
            valid = sorted(set(i for i in row if 0 <= i < len(facts)))
            fs = [facts[i] for i in valid]
            context_parts = []
            for f in fs:
                text = fact_text(f)
                cap = self.cfg.get("context_tokens_per_fact")
                if cap:
                    tokens = self.lm.tokenizer.encode(text, add_special_tokens=False)
                    if len(tokens) > cap:
                        half = cap // 2
                        text = self.lm.tokenizer.decode(tokens[:half]) + "\n# ... excerpt ...\n" + self.lm.tokenizer.decode(tokens[-half:])
                context_parts.append(text)
            contexts.append("\n\n".join(context_parts))
            selected.append([f["fact_id"] for f in fs])
            copied.append("; ".join(str(f["value"]) for f in fs) if fs else UNKNOWN)
            sources.append([f["source_id"] for f in fs])
        return contexts, selected, copied, sources

    @torch.inference_mode()
    def init_rag(self):
        if self.rag_embeddings is not None:
            return
        start = time.perf_counter()
        self.rag_embeddings = self.lm.encode([f.get("retrieval_text", fact_text(f)) for f in self.facts], self.cfg.get("encoder_batch", 32))
        sync(self.lm.device)
        self.rag_index_s = time.perf_counter() - start

    @torch.inference_mode()
    def tune_rag(self):
        self.init_rag()
        rows = self.rows("dev")
        if not rows:
            dump(self.root / "rag_dev.json", dict(k=self.rag_k, threshold=self.rag_threshold, reason="no dev labels"))
            return
        z = self.features(rows)
        scores = z @ self.rag_embeddings.T
        best = (-1, self.rag_k, self.rag_threshold)
        trials = []
        for k in self.cfg.get("rag_k_candidates", [1, 2, 4]):
            for threshold in self.cfg.get("rag_threshold_candidates", [0.0, 0.3, 0.5, 0.7]):
                v, ids = scores.topk(min(k, len(self.facts)), dim=-1)
                pred = [[self.facts[int(i)]["fact_id"] for value, i in zip(vrow, irow) if float(value) >= threshold]
                    for vrow, irow in zip(v, ids)]
                f1 = np.mean([set_f1(p, q["supporting_fact_ids"]) for p, q in zip(pred, rows)])
                trials.append(dict(k=k, threshold=threshold, dev_support_f1=float(f1)))
                if f1 > best[0]:
                    best = (f1, k, threshold)
        self.rag_k, self.rag_threshold = best[1:]
        dump(self.root / "rag_dev.json", dict(k=self.rag_k, threshold=self.rag_threshold,
            selection="dev supporting-fact F1 only", trials=trials, index_s=self.rag_index_s))

    def configure_rag(self):
        path = self.root / "rag_dev.json"
        if path.exists():
            c = read(path)
            self.rag_k, self.rag_threshold = c["k"], c["threshold"]
        else:
            self.tune_rag()
        self.init_rag()

    def bake(self, variant):
        train, dev = self.rows("train"), self.rows("dev")
        if not train or not dev:
            raise ValueError("Training requires independent train and dev families")
        # The test file is never loaded in this method or any tuning method.
        start = time.perf_counter()
        z, zd = self.features(train), self.features(dev)
        candidates = self.cfg.get("hybrid_loss_candidates", [[1.0, 0.25]]) if variant == "hybrid" else [[1.0, 0.25]]
        best_score, best_memory, best_meta, trials = float("inf"), None, None, []
        for alpha, beta in candidates:
            seed_all(self.cfg.get("seed", 17))
            memory = self.make_memory(variant)
            target = memory.targets(train, self.facts, z.device)
            td = memory.targets(dev, self.facts, z.device)
            unknown = torch.tensor([not q["answerable"] for q in train], device=z.device).float()
            ud = torch.tensor([not q["answerable"] for q in dev], device=z.device).float()
            ridge = memory.ridge_fit(z, target, self.cfg.get("ridge", 0.1)) if memory.code_head and memory.address is not None else None
            history = []
            optimizer = torch.optim.AdamW(memory.parameters(), lr=self.cfg.get("address_lr", 0.002), weight_decay=0.001)
            # Cheap address warmup on cached frozen features before expensive LM backprop.
            if memory.address is not None:
                for epoch in range(self.cfg.get("address_epochs", 150)):
                    memory.train()
                    _, logits, u = memory(z)
                    loss = memory.address_loss(logits, target) + beta * F.binary_cross_entropy_with_logits(u, unknown)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(memory.parameters(), 1.0)
                    optimizer.step()
                    if epoch % 50 == 0:
                        print(f"{variant} address epoch={epoch} loss={float(loss.detach()):.4f}", flush=True)
            optimizer = torch.optim.AdamW(memory.parameters(), lr=self.cfg.get("learning_rate", 0.0005), weight_decay=0.001)
            epochs = self.cfg.get("answer_epochs", 12) if variant != "address" else 1
            local_best, best_state, stale = float("inf"), None, 0
            for epoch in range(epochs):
                memory.train()
                order = torch.randperm(len(train), device=z.device)
                losses = []
                batch_size = self.cfg.get("train_batch", 4)
                if variant != "address":
                    for idx in order.split(batch_size):
                        rows = [train[i] for i in idx.cpu().tolist()]
                        prefix, logits, u = memory(z[idx])
                        contexts = [""] * len(rows)
                        loss_address = loss_unknown = torch.tensor(0.0, device=z.device)
                        if logits is not None:
                            ids, _ = memory.addresses(logits.detach(), u.detach())
                            contexts = self.contexts(ids.cpu().tolist())[0]
                            loss_address = memory.address_loss(logits, target[idx])
                            loss_unknown = F.binary_cross_entropy_with_logits(u, unknown[idx])
                        answer_loss = self.lm.answer_loss([q["question"] for q in rows], contexts, prefix,
                            [q["answer"] for q in rows])
                        loss = answer_loss + alpha * loss_address + beta * loss_unknown
                        if not torch.isfinite(loss):
                            raise FloatingPointError("Non-finite training loss")
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(memory.parameters(), 1.0)
                        optimizer.step()
                        losses.append(float(loss.detach()))
                memory.eval()
                with torch.no_grad():
                    _, ld, unkd = memory(zd)
                    address_dev = float(memory.address_loss(ld, td)) if ld is not None else 0.0
                    unknown_dev = float(F.binary_cross_entropy_with_logits(unkd, ud)) if unkd is not None else 0.0
                    dev_answer = 0.0
                    if variant != "address":
                        for start_d in range(0, len(dev), batch_size):
                            rows = dev[start_d:start_d + batch_size]
                            prefix, logits, u = memory(zd[start_d:start_d + batch_size])
                            context = [""] * len(rows)
                            if logits is not None:
                                ids, _ = memory.addresses(logits, u)
                                context = self.contexts(ids.cpu().tolist())[0]
                            loss = self.lm.answer_loss([q["question"] for q in rows], context, prefix, [q["answer"] for q in rows])
                            dev_answer += float(loss) * len(rows) / len(dev)
                score = dev_answer + address_dev + unknown_dev
                row = dict(epoch=epoch + 1, train_loss=float(np.mean(losses)) if losses else None,
                    dev_answer_ce=dev_answer, dev_address_ce=address_dev, dev_unknown_bce=unknown_dev, dev_score=score)
                history.append(row)
                print(json.dumps(dict(variant=variant, alpha=alpha, beta=beta, **row)), flush=True)
                if score < local_best:
                    local_best, stale = score, 0
                    best_state = {k: v.detach().cpu().clone() for k, v in memory.state_dict().items()}
                else:
                    stale += 1
                if stale >= self.cfg.get("patience", 4):
                    break
            memory.load_state_dict(best_state)
            memory.eval()
            # Calibrate missing-knowledge probability using dev only.
            threshold = 0.5
            if memory.address is not None:
                with torch.no_grad():
                    p = memory(zd)[2].sigmoid()
                    scored = [(float(((p > t) == ud.bool()).float().mean()), t) for t in (0.25, 0.4, 0.5, 0.6, 0.75)]
                    threshold = max(scored, key=lambda pair: (pair[0], -abs(pair[1] - .5)))[1]
            trial = dict(alpha=alpha, beta=beta, dev_score=local_best, ridge=ridge, history=history,
                         unknown_threshold=threshold)
            trials.append(trial)
            if local_best < best_score:
                best_score, best_memory, best_meta = local_best, memory, trial
            else:
                del memory
        sync(self.lm.device)
        meta = dict(**best_meta, trials=trials, baking_s=time.perf_counter() - start,
                    selection="minimum dev_answer_ce + dev_address_ce + dev_unknown_bce; no test access",
                    train_size=len(train), dev_size=len(dev))
        if best_memory.code_head:
            meta["ecc_bit_corruption"] = best_memory.codec.experiment()
        path = self.save(best_memory, meta)
        # Round trip check includes generated answers, not just tensor equality.
        before = self.run(train[:2], variant)
        del self.memories[variant]
        self.load(variant)
        after = self.run(train[:2], variant)
        if any(a["answer"] != b["answer"] or a["selected_fact_ids"] != b["selected_fact_ids"] for a, b in zip(before[0], after[0])):
            raise AssertionError("Checkpoint reload changed answers")
        meta["reload_answers_identical"] = True
        dump(path.with_suffix(".training.json"), meta)
        print(f"Saved {path}; reload answers identical", flush=True)
        return meta

    @torch.inference_mode()
    def run(self, rows, mode, structured_only=False):
        if structured_only and mode not in ("rag", "address", "hybrid"):
            raise ValueError("Structured copying requires an addressable local store")
        variant = "hybrid" if mode.startswith("hybrid") else mode
        if mode in ("rag",):
            self.configure_rag()
        memory = self.load(variant) if variant in ("latent", "address", "hybrid") else None
        sync(self.lm.device)
        start = time.perf_counter()
        z = None
        if memory is not None or mode == "rag":
            z = self.lm.encode([q["question"] for q in rows], self.cfg.get("encoder_batch", 32))
        sync(self.lm.device)
        encoder = time.perf_counter() - start
        stage = time.perf_counter()
        prefix, code_stats = None, {}
        contexts, selected, copied, sources = ([""] * len(rows), [[] for _ in rows], [UNKNOWN] * len(rows), [[] for _ in rows])
        if memory is not None:
            prefix, logits, unknown = memory(z)
            if mode == "hybrid_no_latent":
                prefix = None
            if logits is not None and mode != "hybrid_no_address":
                threshold = self.artifacts[variant]["meta"].get("unknown_threshold", 0.5)
                ids, stats = memory.addresses(logits, unknown, threshold)
                contexts, selected, copied, sources = self.contexts(ids.cpu().tolist(), self.artifacts[variant]["facts"])
                code_stats = {k: int(v.sum()) for k, v in stats.items()}
        elif mode == "rag":
            scores, ids = (z @ self.rag_embeddings.T).topk(min(self.rag_k, len(self.facts)), dim=-1)
            ids = torch.where(scores >= self.rag_threshold, ids, len(self.facts))
            contexts, selected, copied, sources = self.contexts(ids.cpu().tolist())
        elif mode == "oracle":
            mapping = {f["fact_id"]: i for i, f in enumerate(self.facts)}
            contexts, selected, copied, sources = self.contexts([[mapping[x] for x in q["supporting_fact_ids"]] for q in rows])
        elif mode != "none":
            raise ValueError(f"Unknown mode: {mode}")
        sync(self.lm.device)
        memory_s = time.perf_counter() - stage
        if structured_only:
            answers = copied
            timing = dict(preparation_s=0.0, prefill_s=0.0, decode_s=0.0, generated_tokens=0, budget_tokens=0)
        else:
            answers, timing = self.lm.generate([q["question"] for q in rows], contexts, prefix)
        sync(self.lm.device)
        total = time.perf_counter() - start
        timing.update(encoder_s=encoder, memory_s=memory_s, end_to_end_s=total,
            ttft_s=encoder + memory_s + timing["preparation_s"] + timing["prefill_s"], code_stats=code_stats)
        result = [dict(question=q["question"], answer=a, selected_fact_ids=s, copied_answer=c,
            source_ids=src) for q, a, s, c, src in zip(rows, answers, selected, copied, sources)]
        return result, timing


def set_f1(pred, gold):
    a, b = set(pred), set(gold)
    return 2 * len(a & b) / (len(a) + len(b)) if a or b else 1.0
