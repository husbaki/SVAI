from __future__ import annotations

import math
import re
import time
from pathlib import Path

import numpy as np
import torch

from .data import UNKNOWN, dump, read
from .experiment import set_f1
from .models import sync


MODES = ["none", "rag", "oracle", "latent", "address", "hybrid", "hybrid_no_latent", "hybrid_no_address"]


def normalize(text):
    return " ".join(str(text).strip().strip(".\n\t ").casefold().split())


def quality(rows, predictions, facts, mode):
    lookup = {f["fact_id"]: f for f in facts}
    metrics = []
    local = mode not in ("none", "latent", "hybrid_no_address")
    for q, p in zip(rows, predictions):
        pred, gold = set(p["selected_fact_ids"]), set(q["supporting_fact_ids"])
        answered = normalize(p["answer"]) != normalize(UNKNOWN)
        numbers = lambda s: re.findall(r"\b\d+(?:[./-]\d+)*\b", s)
        identifiers = lambda s: re.findall(r"\b[A-Z]+[-_][A-Za-z0-9_-]+\b", s)
        gold_sources = {lookup[i]["source_id"] for i in gold}
        row = dict(exact_match=float(normalize(p["answer"]) == normalize(q["answer"])),
            numeric_exact=float(numbers(p["answer"]) == numbers(q["answer"])) if numbers(q["answer"]) else None,
            date_exact=float(re.findall(r"\d{4}-\d{2}-\d{2}", p["answer"]) == re.findall(r"\d{4}-\d{2}-\d{2}", q["answer"])) if re.search(r"\d{4}-\d{2}-\d{2}", q["answer"]) else None,
            identifier_exact=float(identifiers(p["answer"]) == identifiers(q["answer"])) if identifiers(q["answer"]) else None,
            structured_exact=float(normalize(p["copied_answer"]) == normalize(q["answer"])) if local and q.get("copy_evaluable", True) else None,
            supporting_precision=len(pred & gold) / len(pred) if pred else float(not gold),
            supporting_recall=len(pred & gold) / len(gold) if gold else float(not pred),
            supporting_f1=set_f1(pred, gold), source_f1=set_f1(p["source_ids"], gold_sources) if local else None,
            answered=float(answered), unanswerable_error=float(answered) if not q["answerable"] else None,
            answerable=q["answerable"], tags=q.get("tags", []))
        metrics.append(row)
    result = {}
    for key in metrics[0] if metrics else []:
        if key in ("tags", "answerable"):
            continue
        values = [r[key] for r in metrics if r[key] is not None]
        result[key] = float(np.mean(values)) if values else None
    for tag in ("exceptions", "versions", "multi", "standard", "unknown", "itsdangerous", "httpx", "django", "behavior", "location", "heldout_behavior", "paraphrase"):
        sub = [m for m in metrics if tag in m["tags"]]
        result[tag + "_n"] = len(sub)
        result[tag + "_error"] = float(np.mean([1 - m["exact_match"] for m in sub])) if sub else None
        copied = [m["structured_exact"] for m in sub if m["structured_exact"] is not None]
        result[tag + "_structured_exact"] = float(np.mean(copied)) if copied else None
    result["n"] = len(rows)
    result["free_text_explanation_quality"] = "not measured; value-only generation task"
    result["source_metric_scope"] = "returned local source IDs; not free-text citation faithfulness"
    if not local:
        for key in ("supporting_precision", "supporting_recall", "supporting_f1"):
            result[key] = None
    return result


def evaluate(exp, modes=None, split="test"):
    rows = exp.rows(split)
    if not rows:
        raise ValueError(f"No {split} QA; supply independently held-out annotated questions")
    results = {}
    for mode in modes or MODES:
        predictions = []
        for start in range(0, len(rows), exp.cfg.get("eval_batch", 4)):
            predictions.extend(exp.run(rows[start:start + exp.cfg.get("eval_batch", 4)], mode)[0])
        results[mode] = quality(rows, predictions, exp.facts, mode)
        if mode == "address" and exp.cfg.get("address_head") == "code":
            memory = exp.load("address")
            with torch.inference_mode():
                z = exp.features(rows)
                _, logits, unknown = memory(z)
                targets = memory.targets(rows, exp.facts, z.device)
                payload = torch.where(targets == memory.nfacts, 0, targets + 1)
                codes = memory.codec.encode(payload)
                decoded, corrected, invalid = memory.codec.decode(logits > 0)
                book = memory.codec.encode(torch.arange(memory.nfacts + 1, device=z.device)).float()
                nearest = (logits @ (2 * book - 1).T).argmax(-1)
                results[mode]["code_diagnostics"] = dict(
                    exact_codeword_rate=float(((logits > 0) == codes.bool()).all(-1).float().mean()),
                    semantic_nearest_code_accuracy=float((nearest == payload).float().mean()),
                    decoded_payload_accuracy=float((decoded == payload).float().mean()),
                    corrected_word_rate=float(corrected.float().mean()), invalid_word_rate=float(invalid.float().mean()),
                    valid_wrong_payload_rate=float(((decoded != payload) & ~invalid).float().mean()),
                    bit_corruption_test=memory.codec.experiment())
        suffix = "_code" if exp.cfg.get("address_head") == "code" else ""
        dump(exp.root / f"predictions_{mode}_{split}{suffix}.json", [dict(gold=q, prediction=p) for q, p in zip(rows, predictions)])
        print(f"{mode}: generated EM={results[mode]['exact_match']:.3f}, copied EM={results[mode]['structured_exact']}", flush=True)
    suffix = "_code" if exp.cfg.get("address_head") == "code" else ""
    dump(exp.root / f"quality_{split}{suffix}.json", dict(corpus_kind=exp.manifest["kind"],
        corpus_version=exp.manifest["version"], split=split, results=results))
    return results


def compile_probe(exp, variant="hybrid"):
    memory = exp.load(variant).eval()
    z = exp.features(exp.rows("dev")[:8])
    results = dict(variant=variant, backend=exp.cfg.get("compile_backend", "inductor"), retained=False)
    try:
        with torch.inference_mode():
            expected = memory(z)
            for _ in range(5):
                memory(z)
            sync(exp.lm.device)
            start = time.perf_counter()
            for _ in range(40):
                memory(z)
            sync(exp.lm.device)
            eager = (time.perf_counter() - start) / 40
            compiled = torch.compile(memory, backend=results["backend"])
            start = time.perf_counter()
            actual = compiled(z)
            sync(exp.lm.device)
            overhead = time.perf_counter() - start
            for a, b in zip(expected, actual):
                if a is not None:
                    torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-4)
            start = time.perf_counter()
            for _ in range(40):
                compiled(z)
            sync(exp.lm.device)
            steady = (time.perf_counter() - start) / 40
        results.update(eager_s=eager, compiled_s=steady, first_call_s=overhead, speedup=eager / steady)
        # Microbench is insufficient evidence for an end-to-end retention decision.
        results["reason"] = "probe only; keep eager unless end-to-end improvement is confirmed"
    except Exception as exc:
        results.update(error=f"{type(exc).__name__}: {exc}", reason="compile unavailable or failed correctness check")
        # Windows CUDA often lacks Triton; still measure an available compiler backend.
        if results["backend"] == "inductor":
            previous = exp.cfg.get("compile_backend", "inductor")
            exp.cfg["compile_backend"] = "aot_eager"
            results["fallback"] = compile_probe(exp, variant)
            exp.cfg["compile_backend"] = previous
    dump(exp.root / "compile_probe.json", results)
    return results


def benchmark(exp, modes=None):
    rows = exp.rows("test")
    result = {}
    repeats = exp.cfg.get("benchmark_repeats", 8)
    tags = ("versions", "standard", "exceptions", "multi", "unknown", "dates", "identifiers")
    groups = [[q for q in rows if tag in q.get("tags", [])] for tag in tags]
    groups = [g for g in groups if g] or [rows]
    def sample_for(repetition, batch):
        return [groups[(repetition * batch + j) % len(groups)][((repetition * batch + j) // len(groups)) % len(groups[(repetition * batch + j) % len(groups)])] for j in range(batch)]
    for mode in modes or MODES:
        batches = {}
        for batch in exp.cfg.get("benchmark_batches", [1, 4, 8]):
            sample = sample_for(0, batch)
            for _ in range(exp.cfg.get("benchmark_warmup", 2)):
                exp.run(sample, mode)
            if exp.lm.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(exp.lm.device)
            timings = [exp.run(sample_for(i, batch), mode)[1] for i in range(repeats)]
            latency = np.array([t["end_to_end_s"] for t in timings])
            data = dict(batch_size=batch, samples=repeats, p50_ms=float(np.percentile(latency, 50) * 1000),
                p95_ms=float(np.percentile(latency, 95) * 1000),
                ttft_p50_ms=float(np.median([t["ttft_s"] for t in timings]) * 1000),
                throughput_queries_s=batch / float(latency.mean()), amortized_ms=float(latency.mean() * 1000 / batch),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(exp.lm.device) if exp.lm.device.type == "cuda" else 0,
                peak_reserved_bytes=torch.cuda.max_memory_reserved(exp.lm.device) if exp.lm.device.type == "cuda" else 0,
                generated_tokens_s=sum(t["generated_tokens"] for t in timings) / float(latency.sum()),
                model_budget_tokens_s=sum(t["budget_tokens"] for t in timings) / float(latency.sum()),
                stages_ms={k: float(np.mean([t[k] for t in timings]) * 1000)
                    for k in ("encoder_s", "memory_s", "preparation_s", "prefill_s", "decode_s")})
            batches[str(batch)] = data
            print(f"{mode} batch={batch}: p50={data['p50_ms']:.1f}ms, {data['throughput_queries_s']:.2f} q/s", flush=True)
        variant = "hybrid" if mode.startswith("hybrid") else mode
        memory_bytes, checkpoint_load = 0, 0
        if variant in exp.artifacts:
            path = exp.checkpoint_path(variant)
            memory_bytes = path.stat().st_size
            checkpoint_load = exp.artifacts[variant]["meta"].get("checkpoint_load_s", 0)
        elif mode == "rag":
            memory_bytes = exp.rag_embeddings.numel() * exp.rag_embeddings.element_size()
            memory_bytes += (exp.data / "facts.json").stat().st_size
        elif mode == "oracle":
            memory_bytes = (exp.data / "facts.json").stat().st_size
        result[mode] = dict(batches=batches, corpus_memory_bytes=memory_bytes,
            shared_model_startup_s=exp.startup_s, checkpoint_load_s=checkpoint_load,
            cold_start_scope="shared LLM+encoder initialization measured once; not independent process launches",
            rag_index_s=exp.rag_index_s if mode == "rag" else 0)
    output = dict(results=result, timing="synchronized host wall clock; stages include synchronization overhead",
        decode=("stop when all batch sequences reach EOS, max_new_tokens is shared cap" if exp.cfg.get("stop_on_eos")
                else "fixed max_new_tokens budget, EOS-trimmed outputs; budget includes post-EOS compute"),
        caching="resident models and precomputed corpus embeddings; query representations not cached",
        vram_scope="process peak allocated; includes all resident memory heads, shared LLM and encoder",
        startup_excludes="Python startup, imports, network download", repeats=repeats)
    output["query_mix"] = "deterministic round robin across available test tags; same questions/order in all variants; cost assumes this mix"
    suffix = "_code" if exp.cfg.get("address_head") == "code" else ""
    dump(exp.root / f"benchmark{suffix}.json", output)
    return output


def benchmark_structured(exp):
    """Supplemental exact-field path, compared with equally non-generative RAG."""
    rows = exp.rows("test")
    results = {}
    for mode in ("rag", "address", "hybrid"):
        predictions = []
        for start in range(0, len(rows), exp.cfg.get("eval_batch", 4)):
            predictions.extend(exp.run(rows[start:start + exp.cfg.get("eval_batch", 4)], mode, structured_only=True)[0])
        result = dict(exact_match=quality(rows, predictions, exp.facts, mode)["exact_match"], batches={})
        for batch in exp.cfg.get("benchmark_batches", [1, 4, 8]):
            sample = [rows[i % len(rows)] for i in range(batch)]
            for _ in range(3):
                exp.run(sample, mode, structured_only=True)
            times = [exp.run(sample, mode, structured_only=True)[1]["end_to_end_s"] for _ in range(30)]
            result["batches"][str(batch)] = dict(samples=30, p50_ms=float(np.median(times) * 1000),
                p95_ms=float(np.percentile(times, 95) * 1000), throughput_queries_s=batch / float(np.mean(times)))
        results[mode] = result
    dump(exp.root / "structured_benchmark.json", dict(results=results,
        scope="supplemental address+copy path; Qwen is resident but not executed; not part of common LLM comparison"))
    return results


def economics(cfg, benchmark_results, quality_results, artifacts, rag_index_s=0, teacher=None):
    prices = cfg.get("prices", {})
    hourly = prices.get("gpu_hour_usd", 0.35)
    teacher = teacher or {}
    teacher_cost = (teacher.get("input_tokens", 0) * prices.get("teacher_input_million_usd", 0) / 1e6
        + teacher.get("output_tokens", 0) * prices.get("teacher_output_million_usd", 0) / 1e6
        + teacher.get("wall_s", 0) / 3600 * hourly)
    shared_prepare_cost = prices.get("preparation_cpu_hours", 0) * prices.get("cpu_hour_usd", 0)
    common_cost = teacher_cost + shared_prepare_cost
    rows = {}
    rag_per_query = benchmark_results["rag"]["batches"]["1"]["amortized_ms"] / 3_600_000 * hourly
    rag_baking = rag_index_s / 3600 * hourly
    window_hours = prices.get("hours_between_updates", 720)
    for mode, perf in benchmark_results.items():
        variant = "hybrid" if mode.startswith("hybrid") else mode
        baking_s = artifacts.get(variant, {}).get("meta", {}).get("baking_s", 0)
        if mode == "rag":
            baking_s = rag_index_s
        baking = baking_s / 3600 * hourly
        per_query = perf["batches"]["1"]["amortized_ms"] / 3_600_000 * hourly
        savings = rag_per_query - per_query
        incremental = baking - rag_baking
        break_even = max(0.0, incremental / savings) if savings > 0 else None
        accuracy = quality_results[mode]["exact_match"]
        copied_accuracy = quality_results[mode].get("structured_exact")
        scenarios = []
        for queries in (10_000, 100_000, 1_000_000):
            active_hours = queries * per_query / hourly if hourly > 0 else 0
            resident_hours = max(window_hours, active_hours)
            scenarios.append(dict(queries=queries, metered_total_usd=common_cost + baking + queries * per_query,
                dedicated_total_usd=common_cost + baking + resident_hours * hourly,
                dedicated_paid_gpu_hours=resident_hours + baking_s / 3600,
                dedicated_idle_hours=max(0, resident_hours - active_hours),
                dedicated_cost_per_correct_answer=(common_cost + baking + resident_hours * hourly) / (queries * accuracy) if accuracy else None))
        rows[mode] = dict(preparation_cost_usd=common_cost, teacher_cost_usd=teacher_cost, baking_s=baking_s,
            baking_cost_usd=baking, update_cost_usd=baking + common_cost,
            update_assumption="full rebuild at same size; update duration not separately measured",
            paid_baking_gpu_hours=baking_s / 3600, cost_per_answer_usd=per_query,
            cost_per_correct_answer_usd=per_query / accuracy if accuracy else None,
            cost_per_correct_structured_answer_usd=per_query / copied_accuracy if copied_accuracy else None,
            break_even_queries=break_even, no_positive_break_even=savings <= 0, scenarios=scenarios)
    return dict(currency="USD", assumptions=prices, results=rows,
        measured="training wall seconds and synchronized batch=1 inference service seconds",
        assumed="hourly prices, billing model, update interval, repeated full-build update cost; no vendor price claims",
        break_even_scope="metered compute only; dedicated idle billing can remove savings")


def compare(exp):
    q = read(exp.root / "quality_test.json")
    b = read(exp.root / "benchmark.json")
    # Recompute deterministic metrics from stored predictions when scoring code evolves.
    for mode in q["results"]:
        path = exp.root / f"predictions_{mode}_test.json"
        if path.exists():
            saved = read(path)
            q["results"][mode] = quality([p["gold"] for p in saved], [p["prediction"] for p in saved], exp.facts, mode)
    dump(exp.root / "quality_test.json", q)
    artifacts = {}
    for variant in ("latent", "address", "hybrid"):
        path = exp.checkpoint_path(variant).with_suffix(".training.json")
        if path.exists():
            artifacts[variant] = dict(meta=read(path))
    costs = economics(exp.cfg, b["results"], q["results"], artifacts, b["results"]["rag"]["rag_index_s"], exp.manifest.get("teacher"))
    dump(exp.root / "costs.json", costs)
    lines = ["# Compiled corporate memory: measured experiment", "",
        f"Corpus: **{q['corpus_kind']}**, version `{q['corpus_version'][:12]}`. " +
        ("No results on real documents are claimed." if q["corpus_kind"] == "synthetic" else "Results refer only to this labeled document corpus."), "",
        f"Frozen LLM: `{exp.cfg['model_id']}`. Frozen encoder: `{exp.cfg['encoder_id']}`.", "",
        "| Variant | Generated EM | Copied EM | Fact F1 | p50 / p95 ms (b=1) | TTFT ms | q/s (b=1) | peak VRAM MiB | corpus MiB | USD/answer* |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for mode, quality_row in q["results"].items():
        perf = b["results"][mode]
        one = perf["batches"]["1"]
        fmt = lambda v: "n/a" if v is None else f"{v:.3f}"
        lines.append(f"| {mode} | {fmt(quality_row['exact_match'])} | {fmt(quality_row['structured_exact'])} | {fmt(quality_row['supporting_f1'])} | "
            f"{one['p50_ms']:.1f} / {one['p95_ms']:.1f} | {one['ttft_p50_ms']:.1f} | {one['throughput_queries_s']:.2f} | "
            f"{one['peak_allocated_bytes']/2**20:.1f} | {perf['corpus_memory_bytes']/2**20:.3f} | {costs['results'][mode]['cost_per_answer_usd']:.8f} |")
    best_generated = max(q["results"], key=lambda k: q["results"][k]["exact_match"])
    learned = [k for k in ("latent", "address", "hybrid") if k in q["results"]]
    best_learned = max(learned, key=lambda k: q["results"][k]["exact_match"]) if learned else "n/a"
    lines += ["", f"Highest generated EM: **{best_generated}**; highest among trained architectures: **{best_learned}**.",
        "Results establish behavior only for this corpus, split, seed and training budget. Shared training benefit must be judged from the hybrid and both ablations, not assumed.", "",
        "## Errors on held-out question families", "",
        "| Variant | Exception error | Old-version error | Multi-fact error | Unanswerable error | Answer coverage | Source F1 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for mode, row in q["results"].items():
        fmt = lambda v: "n/a" if v is None else f"{v:.3f}"
        lines.append("| " + mode + " | " + " | ".join(fmt(row[k]) for k in
            ("exceptions_error", "versions_error", "multi_error", "unanswerable_error", "answered", "source_f1")) + " |")
    lines += ["", "## Training completed", "", "| Variant | Baking seconds | Chosen alpha / beta | Best dev score | Reload answers identical |", "|---|---:|---:|---:|---|"]
    for variant, artifact in artifacts.items():
        m = artifact["meta"]
        lines.append(f"| {variant} | {m['baking_s']:.1f} | {m['alpha']} / {m['beta']} | {m['dev_score']:.4f} | {m.get('reload_answers_identical')} |")
    lines += ["", "Both hybrid coefficients and stopping checkpoints were selected on dev. Train loss alone is not a reliability result.", "",
        "## Typed output accuracy", "", "| Variant | Numbers incl. dates | ISO dates | Identifiers |", "|---|---:|---:|---:|"]
    for mode, row in q["results"].items():
        fmt = lambda v: "n/a" if v is None else f"{v:.3f}"
        lines.append("| " + mode + " | " + " | ".join(fmt(row.get(k)) for k in ("numeric_exact", "date_exact", "identifier_exact")) + " |")
    cold_path = exp.root / "cold_start.json"
    if cold_path.exists():
        cold = read(cold_path)
        lines += ["", "## Fresh process startup", "", cold["scope"] + ". One sample per variant.", "",
            "| Variant | Process start through first answer seconds |", "|---|---:|"]
        lines += [f"| {mode} | {r['process_through_first_answer_s']:.2f} |" for mode, r in cold["results"].items()]
    structured_path = exp.root / "structured_benchmark.json"
    if structured_path.exists():
        structured = read(structured_path)
        lines += ["", "## Supplemental structured copying", "", structured["scope"], "",
            "| Variant | Exact copied answer | p50 / p95 ms, batch=1 |", "|---|---:|---:|"]
        for mode, row in structured["results"].items():
            one = row["batches"]["1"]
            lines.append(f"| {mode} | {row['exact_match']:.3f} | {one['p50_ms']:.2f} / {one['p95_ms']:.2f} |")
    code_path = exp.root / "quality_test_code.json"
    if code_path.exists():
        code = read(code_path)["results"]["address"]
        lines += ["", "## Code-head experiment", "",
            f"Generated EM: {code['exact_match']:.3f}; copied EM: {code['structured_exact']:.3f}.",
            "Semantic address accuracy and code repair are separate:", "", "```json", json_string(code.get("code_diagnostics", {})), "```"]
    lines += ["", "## Economics", "",
        "*USD figures are scenarios at configurable assumed tariffs, not actual invoices. Service time is measured; tariffs are assumptions.",
        f"GPU assumption: {costs['assumptions'].get('gpu_hour_usd', .35)} USD/hour; allocated interval: {costs['assumptions'].get('hours_between_updates', 720)} hours.", "",
        "| Variant | Baking USD | Metered break-even Q vs RAG | 10k metered / dedicated USD | 100k metered / dedicated USD | 1M metered / dedicated USD |", "|---|---:|---:|---:|---:|---:|"]
    for mode, row in costs["results"].items():
        be = "none (no per-query saving)" if row["break_even_queries"] is None else f"{row['break_even_queries']:.0f}"
        scenario = " | ".join(f"{s['metered_total_usd']:.2f} / {s['dedicated_total_usd']:.2f}" for s in row["scenarios"])
        lines.append(f"| {mode} | {row['baking_cost_usd']:.4f} | {be} | {scenario} |")
    lines += ["", "## Measurement limits", "",
        "- Generated answers and exact local field copies are separate outputs. B is neural addressing plus a local fact store.",
        "- RAG uses precomputed embeddings and batched device retrieval; top-k and score threshold are chosen on dev.",
        "- Latent inference reads neither source documents nor the fact table; B/C use the fact snapshot inside their checkpoint.",
        "- Quality is evaluated on all held-out test questions. Benchmark percentiles use the sample count in benchmark.json and are exploratory.",
        "- Batch latency is request latency for the whole batch; amortized GPU service time is reported separately.",
        "- Startup measures shared model initialization once and each checkpoint load, excluding imports, process creation and model downloads.",
        "- Generation shares the same token cap and stopping policy across variants; see benchmark.json for EOS handling.",
        "- Peak allocated VRAM includes all resident heads; allocator reserved memory is recorded separately. Shared base weights are excluded from corpus disk size.",
        "- No quantization or custom kernels applied. Compile outcome is in compile_probe.json when tested.",
        "- Free-text explanation quality and real-document generalization are not measured.",
        "- SECDED corrects corrupted code bits, not semantic fact-selection mistakes. See code-head checkpoint and ECC tests.",
        "- Update costs assume a complete rebuild. GPU idle hours are billed in dedicated scenarios; metered break-even does not apply directly to them."]
    report = exp.root / "REPORT.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report, flush=True)
    return report


def json_string(value):
    import json
    return json.dumps(value, indent=2)
