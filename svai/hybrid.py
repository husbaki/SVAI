"""Real shared latent/address memory, validated against the current source snapshot."""
from __future__ import annotations
import os
from pathlib import Path
import threading

_CACHE = {}
_LOCK = threading.RLock()


def identity(repo, address):
    return (repo, address.get("commit"), address.get("path"), address.get("symbol"),
        address.get("start_line"), address.get("end_line"), address.get("fragment_sha256"))


class HybridAdapter:
    def __init__(self, ml):
        self.ml = ml
        self.path = Path(os.getenv("SVAI_HYBRID_CHECKPOINT", "data/checkpoints/hybrid.pt")).resolve()

    def artifact(self):
        if not self.path.is_file():
            return None
        key = (str(self.path), self.path.stat().st_mtime_ns)
        with _LOCK:
            if key not in _CACHE:
                import torch
                # Only a locally configured, trusted training artifact is loaded.
                artifact = torch.load(self.path, map_location="cpu", weights_only=False)
                if artifact.get("variant") != "hybrid" or artifact.get("corpus_kind") != "real_codebase":
                    raise ValueError("Hybrid requires a real-codebase shared-memory checkpoint")
                _CACHE.clear()
                _CACHE[key] = dict(artifact=artifact, memory=None)
            return _CACHE[key]

    def match(self, documents, artifact):
        table = {identity(f["repository"],f["address"]): (i,f) for i,f in enumerate(artifact["facts"])}
        selected, changed, unsupported = {}, [], []
        for doc in documents:
            addr = doc["metadata"].get("source_address",{})
            entry = table.get(identity(doc["repository"],addr))
            if entry is None:
                unsupported.append(doc["id"])
                continue
            index, fact = entry
            if doc["content_sha256"] != fact["source_content_sha256"]:
                changed.append(doc["id"])
            else:
                selected[index] = doc
        return selected, changed, unsupported

    def status(self, documents):
        cached = self.artifact()
        if cached is None:
            return dict(ready=False,reason="hybrid_checkpoint_not_available")
        artifact = cached["artifact"]
        selected, changed, unsupported = self.match(documents,artifact)
        repositories={d["repository"] for d in documents}
        expected=sum(f["repository"] in repositories for f in artifact["facts"])
        missing=max(0,expected-len(selected)-len(changed))
        ready = bool(selected) and not changed and not unsupported and not missing
        reason = None if ready else ("source_changed_rebake_required" if changed else "new_documents_rebake_required" if unsupported else "source_removed_or_filtered_rebake_required" if missing else "no_matching_source_records")
        return dict(ready=ready,reason=reason,corpus_version=artifact["corpus_version"],
            matched_records=len(selected),changed_records=len(changed),new_records=len(unsupported),missing_records=missing,
            checkpoint=self.path.name,latent_tokens=artifact["config"]["memory_tokens"],
            architecture="shared_trunk_latent_plus_address",training_scope="source-grounded QA subset; full candidate source table")

    def query(self, question, documents, top_k=2):
        import torch
        from corporate_memory.models import FrozenLM, Memory
        with self.ml.lock, torch.inference_mode():
            cached = self.artifact()
            artifact = cached["artifact"]
            status = self.status(documents)
            if not status["ready"]:
                raise ValueError(status["reason"])
            self.ml.load()
            cfg = artifact["config"]
            if cfg["model_id"] != self.ml.model_id or cfg["encoder_id"] != self.ml.encoder_id:
                raise ValueError("Frozen model IDs do not match the hybrid checkpoint")
            if cached["memory"] is None:
                memory = Memory(artifact["encoder_dim"],artifact["nfacts"],"hybrid",cfg,artifact["latent_dim"])
                memory.load_state_dict(artifact["state"])
                cached["memory"] = memory.to(self.ml.device).eval()
            memory = cached["memory"]
            allowed,_,_ = self.match(documents,artifact)
            prefix, logits, unknown = memory(self.ml.encode([question]))
            mask = torch.full_like(logits, -torch.inf)
            permitted = list(allowed) + [memory.nfacts]
            mask[...,permitted] = 0
            ids,_ = memory.addresses(logits+mask,unknown,artifact["meta"].get("unknown_threshold",.5))
            indices = sorted({i for i in ids[0].cpu().tolist() if i in allowed})[:top_k]
            if not indices:
                return dict(answer="UNKNOWN",document_ids=[],unknown_probability=float(unknown.sigmoid()[0]))
            contexts=[]
            for i in indices:
                value=artifact["facts"][i]["context"]
                cap=cfg.get("context_tokens_per_fact",384)
                tokens=self.ml.tokenizer.encode(value,add_special_tokens=False)
                if len(tokens)>cap:
                    half=cap//2
                    value=self.ml.tokenizer.decode(tokens[:half])+"\n# ... excerpt ...\n"+self.ml.tokenizer.decode(tokens[-half:])
                contexts.append(value)
            bridge=FrozenLM.__new__(FrozenLM)
            bridge.cfg=dict(cfg,stop_on_eos=True)
            bridge.device=torch.device(self.ml.device)
            bridge.dtype=self.ml.model.get_input_embeddings().weight.dtype
            bridge.model,bridge.tokenizer=self.ml.model,self.ml.tokenizer
            answer=bridge.generate([question],["\n\n".join(contexts)],prefix)[0][0]
            return dict(answer=answer,document_ids=[allowed[i]["id"] for i in indices],
                unknown_probability=float(unknown.sigmoid()[0]),latent_tokens=memory.tokens,
                selected_fact_ids=[artifact["facts"][i]["fact_id"] for i in indices])
