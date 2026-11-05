from __future__ import annotations

from collections import Counter, defaultdict
import math
import os
import re
import threading


STOP = set("a an the is are does do how what where which when why in on of to for with and or be can code function method class source repository please return tell show me this that it from as at by all".split())
TRANSLATIONS = {"подписи": "signature", "подпись": "signature", "проверяется": "verify validate", "срок": "age timestamp expire", "запрос": "request", "таймаут": "timeout", "транзакции": "transaction atomic"}


def tokenize(text, query=False):
    raw = re.sub(r"([a-z])([A-Z])", r"\1 \2", text).replace("_", " ")
    words = [w for w in re.findall(r"[a-zа-яё][a-zа-яё0-9]+", raw.lower()) if w not in STOP]
    if query:
        words += [v for w in words.copy() for v in TRANSLATIONS.get(w, "").split()]
    return words


_lexical_lock = threading.Lock()
_lexical_cache = {}


def lexical(documents, question):
    # Cached real BM25. Versions and hashes make document updates invalidate it.
    key = tuple((d["id"], d["version"], d["content_sha256"]) for d in documents)
    with _lexical_lock:
        if key not in _lexical_cache:
            postings, lengths = defaultdict(list), []
            for i, document in enumerate(documents):
                counts = Counter(tokenize(document["content"]) + tokenize(document["title"]) * 6)
                lengths.append(sum(counts.values()))
                for word, count in counts.items():
                    postings[word].append((i, count))
            _lexical_cache.clear()
            _lexical_cache[key] = postings, lengths, sum(lengths) / max(1, len(lengths))
        postings, lengths, average = _lexical_cache[key]
    scores = [0.] * len(documents)
    for term in set(tokenize(question, True)):
        found = postings.get(term, [])
        idf = math.log(1 + (len(documents) - len(found) + .5) / (len(found) + .5))
        for i, f in found:
            scores[i] += idf * f * 2.5 / (f + 1.5 * (.25 + .75 * lengths[i] / max(average, 1)))
    return scores


class LocalML:
    def __init__(self):
        self.lock = threading.RLock()
        self.model_id = os.getenv("SVAI_LLM_ID", "Qwen/Qwen2.5-0.5B-Instruct")
        self.encoder_id = os.getenv("SVAI_ENCODER_ID", "sentence-transformers/all-MiniLM-L6-v2")
        self.cache = os.getenv("SVAI_MODEL_CACHE") or None
        self.device = os.getenv("SVAI_DEVICE", "auto")
        self.loaded, self.error = False, None
        self.embedding_cache = {}

    def status(self):
        return dict(loaded=self.loaded, llm_id=self.model_id, encoder_id=self.encoder_id,
            device=self.device, error=self.error, base_models_frozen=True)

    def load(self):
        with self.lock:
            if self.loaded:
                return self.status()
            try:
                import torch
                from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
                self.torch = torch
                if self.device == "auto":
                    self.device = "cuda" if torch.cuda.is_available() else "cpu"
                dtype = torch.bfloat16 if self.device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
                opts = dict(cache_dir=self.cache)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, **opts)
                self.model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=dtype,
                    attn_implementation="sdpa", **opts).to(self.device).eval()
                self.model.requires_grad_(False)
                self.encoder_tokenizer = AutoTokenizer.from_pretrained(self.encoder_id, **opts)
                self.encoder = AutoModel.from_pretrained(self.encoder_id, **opts).to(self.device).eval()
                self.encoder.requires_grad_(False)
                self.loaded, self.error = True, None
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
                raise
            return self.status()

    def encode(self, texts):
        torch = self.torch
        with torch.inference_mode():
            values = []
            for start in range(0, len(texts), 32):
                inputs = self.encoder_tokenizer(texts[start:start + 32], padding=True, truncation=True, max_length=256, return_tensors="pt").to(self.device)
                hidden = self.encoder(**inputs).last_hidden_state.float()
                mask = inputs.attention_mask.unsqueeze(-1)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
                values.append(torch.nn.functional.normalize(pooled, dim=-1))
            return torch.cat(values) if values else torch.empty((0, self.encoder.config.hidden_size), device=self.device)

    def rank(self, documents, question, revision, lex_scores):
        with self.lock:
            self.load()
            key = (revision, tuple((d["id"], d["version"], d["content_sha256"]) for d in documents))
            if key not in self.embedding_cache:
                self.embedding_cache.clear()  # bounded resident cache, invalidated by CRUD revision
                self.embedding_cache[key] = self.encode([d["title"] + "\n" + d["content"] for d in documents])
            query = self.encode([question])
            dense = (query @ self.embedding_cache[key].T)[0].cpu().tolist()
            maximum = max(lex_scores, default=0)
            return [.6 * d + .4 * (l / maximum if maximum else 0) for d, l in zip(dense, lex_scores)]

    def answer(self, question, citations):
        with self.lock:
            self.load()
            chunks = []
            for i, citation in enumerate(citations):
                all_ids = self.tokenizer.encode(citation["snippet"], add_special_tokens=False)
                content_ids = all_ids[:550]
                citation["model_context_truncated"] = len(all_ids) > 550
                content = self.tokenizer.decode(content_ids, skip_special_tokens=True)
                chunks.append(f"[{i+1}] {citation['title']} ({citation['repository']}, document version {citation['version']})\n{content}")
            messages = [dict(role="system", content="Answer the question using only the supplied source snippets. Be concise. Cite snippet numbers such as [1]. If they do not support an answer, reply UNKNOWN. Do not invent code behavior or functions."),
                dict(role="user", content="Source snippets:\n" + "\n\n".join(chunks) + "\n\nQuestion: " + question)]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            with self.torch.inference_mode():
                output = self.model.generate(**inputs, max_new_tokens=120, do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id, use_cache=True)
            return self.tokenizer.decode(output[0, inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()
