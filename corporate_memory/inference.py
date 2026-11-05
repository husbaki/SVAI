"""Standalone checkpoint inference: no raw corpus or QA files required."""
import time
from pathlib import Path
import torch
from .data import UNKNOWN, fact_text
from .experiment import seed_all
from .models import FrozenLM, Memory


def query(cfg, question, variant, structured_only=False):
    if structured_only and variant not in ("address", "hybrid", "rag"):
        raise ValueError("Structured copying requires an address head or RAG")
    seed_all(cfg.get("seed", 17))
    suffix = "_code" if variant == "address" and cfg.get("address_head") == "code" else ""
    if variant in ("none", "rag"):
        from .experiment import Experiment
        return Experiment(cfg).run([dict(question=question)], variant, structured_only=structured_only)[0][0]
    artifact = torch.load(Path(cfg["output"]) / f"{variant}{suffix}.pt", map_location="cpu", weights_only=False)
    loaded_cfg = artifact["config"]
    loaded_cfg["device"] = cfg.get("device", loaded_cfg.get("device"))
    for key in ("stop_on_eos", "last_token_logits"):
        if key in cfg:
            loaded_cfg[key] = cfg[key]
    lm = FrozenLM(loaded_cfg)
    memory = Memory(artifact["encoder_dim"], artifact["nfacts"], variant, loaded_cfg, artifact["latent_dim"]).to(lm.device)
    memory.load_state_dict(artifact["state"])
    memory.eval()
    with torch.inference_mode():
        prefix, logits, unknown = memory(lm.encode([question]))
        facts = []
        if logits is not None:
            ids, _ = memory.addresses(logits, unknown, artifact["meta"].get("unknown_threshold", 0.5))
            table = artifact["facts"]
            facts = [table[i] for i in sorted(set(ids[0].cpu().tolist())) if 0 <= i < len(table)]
        text = "\n".join(fact_text(f) for f in facts)
        answer = ("; ".join(f["value"] for f in facts) if facts else UNKNOWN) if structured_only else lm.generate([question], [text], prefix)[0][0]
    return dict(question=question, answer=answer,
        copied_answer="; ".join(f["value"] for f in facts) if facts else (UNKNOWN if variant != "latent" else None),
        selected_fact_ids=[f["fact_id"] for f in facts], source_ids=[f["source_id"] for f in facts],
        corpus_version=artifact["corpus_version"], model_id=artifact["model_id"])
