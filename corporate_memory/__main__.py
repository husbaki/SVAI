from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .data import dump, load_documents, prepare_teacher, save_corpus, synthetic
from .experiment import Experiment, config, environment
from .reporting import MODES, benchmark, benchmark_structured, compare, compile_probe, evaluate


def main():
    parser = argparse.ArgumentParser(description="Compiled corporate memory on one frozen Qwen")
    parser.add_argument("--config", default="configs/memory_quick.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--documents", nargs="+")
    p.add_argument("--teacher-model")
    p.add_argument("--products", type=int)
    p = sub.add_parser("bake")
    p.add_argument("--variant", choices=["latent", "address", "hybrid"], required=True)
    p.add_argument("--head", choices=["categorical", "code"])
    p = sub.add_parser("evaluate")
    p.add_argument("--head", choices=["categorical", "code"])
    p.add_argument("--modes", nargs="+", choices=MODES)
    p.add_argument("--split", default="test", choices=["dev", "test"])
    p = sub.add_parser("benchmark")
    p.add_argument("--head", choices=["categorical", "code"])
    p.add_argument("--modes", nargs="+", choices=MODES)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--structured", action="store_true")
    sub.add_parser("compare")
    sub.add_parser("doctor")
    sub.add_parser("run")
    p = sub.add_parser("query")
    p.add_argument("question")
    p.add_argument("--variant", default="hybrid", choices=["none", "rag", "latent", "address", "hybrid"])
    p.add_argument("--head", choices=["categorical", "code"])
    p.add_argument("--structured", action="store_true")
    p = sub.add_parser("index-codebase")
    p.add_argument("path")
    p.add_argument("--output", required=True)
    p.add_argument("--include", nargs="+")
    p = sub.add_parser("verify-codebase")
    p.add_argument("path")
    p.add_argument("--index", required=True)
    args = parser.parse_args()
    cfg = config(args.config)
    if args.command in ("index-codebase", "verify-codebase"):
        from .codebase import index_codebase, verify_codebase
        if args.command == "index-codebase":
            result = index_codebase(args.path, args.output, args.include)
            result = {k: v for k, v in result.items() if k != "files"}
        else:
            result = verify_codebase(args.path, args.index)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if result.get("valid") is False:
            raise SystemExit(1)
        return
    if getattr(args, "head", None):
        cfg["address_head"] = args.head
    if args.command == "doctor":
        info = environment()
        print(json.dumps(info, indent=2))
        dump(Path(cfg["output"]) / "environment.json", info)
        return
    if args.command in ("prepare", "run"):
        docs = getattr(args, "documents", None)
        if docs:
            facts, splits, documents = load_documents(docs)
            teacher_meta = {}
            if documents:
                if not args.teacher_model:
                    raise ValueError("Unlabeled TXT/MD/JSONL requires --teacher-model (offline cached local teacher)")
                from .models import FrozenLM
                teacher_cfg = dict(cfg, model_id=args.teacher_model, max_new_tokens=768, max_prompt_tokens=4096,
                    system_prompt="You extract grounded facts and training QA. Follow the extraction schema and output JSON only.")
                start = time.perf_counter()
                teacher = FrozenLM(teacher_cfg)
                token_counts = dict(input_tokens=0, output_tokens=0)
                def generate(text):
                    token_counts["input_tokens"] += len(teacher.tokenizer.encode(text))
                    result = teacher.generate([text], [""], max_new_tokens=768)[0][0]
                    token_counts["output_tokens"] += len(teacher.tokenizer.encode(result))
                    return result
                f, qa, rejected = prepare_teacher(documents, generate, Path(cfg["data"]) / "teacher_cache",
                    f"{args.teacher_model}@{teacher.revision}")
                facts.extend(f)
                splits["train"].extend(qa)
                teacher_meta = dict(**token_counts, wall_s=time.perf_counter() - start, model=args.teacher_model,
                    revision=teacher.revision, rejected=rejected, caution="Train-only teacher QA. Supply independent dev/test annotations.")
                del teacher
            result = save_corpus(cfg["data"], facts, splits, "real_documents", teacher_meta)
        else:
            facts, splits = synthetic(cfg.get("seed", 17), getattr(args, "products", None) or cfg.get("products", 2))
            result = save_corpus(cfg["data"], facts, splits, "synthetic")
        print(json.dumps(result, indent=2))
        if args.command == "prepare":
            return
    if args.command == "query":
        # A query has no gold facts. Inference can run from checkpoint without original documents.
        from .inference import query
        print(json.dumps(query(cfg, args.question, args.variant, args.structured), ensure_ascii=False, indent=2))
        return
    exp = Experiment(cfg, load_llm=args.command != "compare")
    if args.command == "bake":
        exp.bake(args.variant)
    elif args.command == "evaluate":
        evaluate(exp, args.modes, args.split)
    elif args.command == "benchmark":
        benchmark_structured(exp) if args.structured else benchmark(exp, args.modes)
        if args.compile:
            print(json.dumps(compile_probe(exp), indent=2))
    elif args.command == "compare":
        compare(exp)
    elif args.command == "run":
        exp.tune_rag()
        evaluate(exp, ["none", "rag", "oracle"], split="dev")
        for variant in ("latent", "address", "hybrid"):
            exp.bake(variant)
        evaluate(exp)
        benchmark(exp)
        compile_probe(exp)
        compare(exp)


if __name__ == "__main__":
    main()
