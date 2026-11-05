"""HTTP adapter for the existing memory.html workspace. No simulated ML jobs."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from .hybrid import HybridAdapter
from .store import now


class WorkspaceQuery(BaseModel):
    text: str = Field(min_length=2, max_length=2000)
    variant: str | None = None
    compare: bool = False
    snapshot: str | None = None


class UploadFile(BaseModel):
    path: str = Field(min_length=1, max_length=300)
    content: str = Field(max_length=300000)


class Upload(BaseModel):
    files: list[UploadFile] = Field(min_length=1, max_length=500)


class Bake(BaseModel):
    variants: list[str] = Field(default_factory=lambda: ["rag", "hybrid"])


def create_router(store, ml, query_handler, query_type, root):
    router, root = APIRouter(), Path(root)
    with store.connect() as db:
        db.executescript("CREATE TABLE IF NOT EXISTS compat_jobs(id TEXT PRIMARY KEY,project TEXT NOT NULL,payload TEXT NOT NULL);"
            "CREATE TABLE IF NOT EXISTS compat_changes(project TEXT PRIMARY KEY,payload TEXT NOT NULL);")

    def fail(code, message, status=400):
        raise HTTPException(status, dict(code=code, message=message))

    def documents(project, allow_new=False):
        if not project.strip() or len(project) > 100:
            fail("INVALID_PROJECT", "Project name must contain 1–100 characters")
        docs, revision = store.snapshot(None if project == "all" else [project])
        if not docs and not allow_new:
            fail("PROJECT_NOT_FOUND", f"Project {project!r} has no documents", 404)
        return docs, revision

    def corpus(docs):
        payload = [(d["id"], d["version"], d["content_sha256"]) for d in docs]
        return "c_" + hashlib.sha256(json.dumps(payload).encode()).hexdigest()[:16]

    def snapshot(project, docs):
        status = HybridAdapter(ml).status(docs)
        variants = dict(rag=dict(artifact="live-dense-bm25", bake_s=None, bake_cost_usd=None,
            self_check_recall_at_1=None, reused=True, live=True))
        created = max((d["updated_at"] for d in docs), default=now())
        if status.get("ready"):
            adapter = HybridAdapter(ml)
            artifact = adapter.artifact()["artifact"]
            measured_s = artifact["meta"].get("baking_s")
            variants["hybrid"] = dict(artifact=adapter.path.name, bake_s=measured_s,
                bake_cost_usd=measured_s / 3600 * .35 if measured_s is not None else None,
                cost_assumption_usd_gpu_hour=.35, self_check_recall_at_1=None, reused=True, live=True)
            created = datetime.fromtimestamp(adapter.path.stat().st_mtime, timezone.utc).isoformat()
        cv = corpus(docs)
        return dict(snapshot_id="s_" + cv[2:], corpus_version=cv, variants=variants, created_at=created,
            source="installed_checkpoint_and_live_source_index", hybrid=status,
            trained_corpus_version=status.get("corpus_version"), config=dict(top_k=2, live_variants=list(variants)))

    def job_rows(project):
        with store.connect() as db:
            return [json.loads(r[0]) for r in db.execute("SELECT payload FROM compat_jobs WHERE project=? ORDER BY rowid DESC LIMIT 10", (project,))]

    def overview(project):
        docs, revision = documents(project)
        with store.connect() as db:
            changes = db.execute("SELECT payload FROM compat_changes WHERE project=?", (project,)).fetchone()
        snap = snapshot(project, docs)
        return dict(project=project, current_corpus=corpus(docs), facts=sum(bool(d["content"].strip()) for d in docs),
            documents=[dict(id=d["id"], title=d["title"], version=str(d["version"]), facts=int(bool(d["content"].strip()))) for d in docs[:200]],
            documents_total=len(docs), documents_truncated=len(docs) > 200,
            last_changes=json.loads(changes[0]) if changes else dict(added=[], changed=[], removed=[]),
            active_snapshot=snap, snapshots=[dict(snapshot_id=snap["snapshot_id"], corpus_version=snap["corpus_version"],
                variants=list(snap["variants"]), created_at=snap["created_at"])], jobs=job_rows(project),
            variants=["rag", "hybrid"], revision=revision, backend="live_fastapi")

    @router.get("/api/projects")
    def projects():
        names = {r["name"] for r in store.repositories()}
        order = [r for r in ("itsdangerous", "httpx", "django") if r in names]
        return dict(projects=order + sorted(names - set(order)) + (["all"] if names else []))

    @router.get("/api/projects/{project}")
    def project_overview(project: str):
        return overview(project)

    @router.get("/api/projects/{project}/corpus")
    def project_corpus(project: str):
        result = overview(project)
        return dict(project=project, corpus_version=result["current_corpus"], facts=result["facts"],
            documents=result["documents"], documents_total=result["documents_total"],
            documents_truncated=result["documents_truncated"], changes=result["last_changes"])

    @router.post("/api/projects/{project}/upload")
    def upload(project: str, payload: Upload):
        if project == "all":
            fail("INVALID_PROJECT", "Upload into a named project, not the all-sources view")
        docs, _ = documents(project, allow_new=True)
        existing = {d["metadata"].get("upload_path"): d for d in docs if d["metadata"].get("upload_path")}
        if len({f.path for f in payload.files}) != len(payload.files):
            fail("DUPLICATE_PATH", "Each uploaded file must have a distinct path")
        changes = dict(added=[], changed=[], removed=[])
        for file in payload.files:
            previous = existing.get(file.path)
            if previous is None:
                saved = store.create(file.path, file.content, project, dict(upload_path=file.path, source_type="uploaded_document"))
                changes["added"].append(saved["id"])
            elif previous["content"] != file.content:
                saved = store.update(previous["id"], dict(content=file.content, expected_version=previous["version"]))
                changes["changed"].append(saved["id"])
        with store.connect() as db:
            db.execute("INSERT OR REPLACE INTO compat_changes VALUES(?,?)", (project, json.dumps(changes)))
        result = overview(project)
        return dict(run_id="upload_" + uuid.uuid4().hex[:12], project=project,
            corpus_version=result["current_corpus"], reused=not changes["added"] and not changes["changed"],
            facts=result["facts"], documents=result["documents"], changes=changes,
            upload_semantics="incremental upsert; omitted files are retained")

    @router.get("/api/projects/{project}/document")
    def document(project: str, doc: str, corpus_version: str | None = Query(None, alias="corpus")):
        docs, _ = documents(project)
        cv = corpus(docs)
        if corpus_version and corpus_version != cv:
            fail("CORPUS_CHANGED", "The source changed after this answer. Refresh before opening it.", 409)
        record = next((d for d in docs if d["id"] == doc), None)
        if record is None:
            fail("DOCUMENT_NOT_FOUND", "Document is not in this project", 404)
        return dict(project=project, corpus_version=cv, id=record["id"], version=str(record["version"]), text=record["content"])

    def query_one(project, text, variant):
        response = query_handler(query_type(question=text, repositories=None if project == "all" else [project],
            top_k=2, mode=variant, use_ml=True))
        actual = response["actual_mode"]
        facts = []
        for c in response["citations"]:
            excerpt = c["snippet"][:1600]
            facts.append(dict(id=c["document_id"] + "#source", doc_id=c["document_id"], text=excerpt,
                heading=c["title"], start=0, end=len(excerpt.encode("utf-16-le")) // 2,
                score="address" if actual == "hybrid" else c["score"], used=bool(response.get("llm_executed")),
                source_url=c.get("source_url"), verification=c["verification"]))
        return dict(variant=actual, requested_variant=variant, actual_mode=actual, answer=response["answer"],
            status="insufficient" if response["status"] == "unknown" else "grounded",
            status_scope="source attached, not semantic correctness proof", latency_ms=response["elapsed_ms"],
            facts=facts, fallback_reason=response.get("fallback_reason"), trained_code_memory_executed=response["trained_code_memory_executed"])

    @router.post("/api/projects/{project}/query")
    def project_query(project: str, payload: WorkspaceQuery):
        docs, _ = documents(project)
        snap = snapshot(project, docs)
        if payload.snapshot and payload.snapshot != snap["snapshot_id"]:
            fail("SNAPSHOT_CHANGED", "The requested source snapshot is no longer current", 409)
        base = dict(run_id="query_" + uuid.uuid4().hex[:12], project=project, question=payload.text,
            snapshot_id=snap["snapshot_id"], corpus_version=snap["corpus_version"], config=snap["config"])
        if payload.compare:
            return dict(base, results=[query_one(project, payload.text, v) for v in snap["variants"]],
                missing_variants=[v for v in ("rag", "latent", "address", "hybrid") if v not in snap["variants"]])
        variant = payload.variant or ("hybrid" if "hybrid" in snap["variants"] else "rag")
        if variant not in ("rag", "hybrid"):
            fail("VARIANT_NOT_AVAILABLE", "Live workspace supports rag and hybrid; other variants are offline results", 422)
        return dict(base, **query_one(project, payload.text, variant))

    @router.post("/api/projects/{project}/bake")
    def bake(project: str, payload: Bake):
        started = time.perf_counter()
        docs, _ = documents(project)
        if not payload.variants or any(v not in ("rag", "hybrid") for v in payload.variants):
            fail("VARIANT_NOT_AVAILABLE", "Select rag and/or hybrid", 422)
        snap = snapshot(project, docs)
        ready = all(v in snap["variants"] for v in payload.variants)
        job = dict(job_id="validate_" + uuid.uuid4().hex[:12], project=project, kind="snapshot_validation",
            corpus_version=corpus(docs), variants=payload.variants, status="succeeded" if ready else "failed",
            stage="done" if ready else "validate", created_at=now(), progress={}, trained_now=False)
        if ready:
            job["result"] = dict(snapshot_id=snap["snapshot_id"], corpus_version=snap["corpus_version"],
                variants={v: snap["variants"][v] for v in payload.variants}, reused=True, trained_now=False,
                duration_s=time.perf_counter()-started, bake_cost_usd=0,
                note="Validated and reused the installed snapshot; no new training was executed")
        else:
            job["error"] = dict(code="REBAKE_REQUIRED", message="Current documents do not match the installed hybrid. Automatic training is not implemented; use RAG or bake offline.")
        with store.connect() as db:
            db.execute("INSERT INTO compat_jobs VALUES(?,?,?)", (job["job_id"], project, json.dumps(job)))
        return job

    @router.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        with store.connect() as db:
            row = db.execute("SELECT payload FROM compat_jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            fail("JOB_NOT_FOUND", "No such snapshot validation job", 404)
        return json.loads(row[0])

    def saved(project):
        documents(project)
        folder = root / "docs/results"
        quality = json.loads((folder / "quality_test.json").read_text(encoding="utf-8"))
        perf = json.loads((folder / "benchmark.json").read_text(encoding="utf-8"))
        training_path = root / "docs/visuals/chart_data.json"
        training = json.loads(training_path.read_text(encoding="utf-8")).get("training", {}) if training_path.is_file() else {}
        source_docs, _ = store.snapshot()
        source_lookup = {d["metadata"].get("original_fact_id"): d for d in source_docs if d["metadata"].get("original_fact_id")}
        def normalized(value):
            return " ".join(str(value).strip().strip(".\n\t ").casefold().split())
        hourly = float(os.getenv("SVAI_GPU_USD_PER_HOUR", ".35"))
        rows = []
        for variant in ("rag", "latent", "address", "hybrid"):
            q = quality["results"][variant]
            b = perf["results"][variant]["batches"]["1"]
            predictions_path = folder / f"predictions_{variant}_test.json"
            errors = []
            if predictions_path.is_file():
                for item in json.loads(predictions_path.read_text(encoding="utf-8")):
                    gold, prediction = item["gold"], item["prediction"]
                    if normalized(gold["answer"]) == normalized(prediction["answer"]):
                        continue
                    facts = []
                    for fact_id in prediction["selected_fact_ids"]:
                        doc = source_lookup.get(fact_id)
                        if doc is None:
                            facts.append(dict(doc_id=fact_id, text="Archived source record is not present in the current corpus", source_available=False))
                            continue
                        unchanged = doc["content_sha256"] == doc["metadata"].get("imported_content_sha256")
                        facts.append(dict(doc_id=doc["id"], text=doc["content"][:600] if unchanged else "Source changed after the archived experiment", source_available=unchanged))
                    errors.append(dict(question=gold["question"], expected=[gold["answer"]] if gold["answerable"] else None,
                        answer=prediction["answer"], status="insufficient" if normalized(prediction["answer"]) == "unknown" else "grounded", facts=facts))
            bake_s = perf["results"][variant].get("rag_index_s") if variant == "rag" else training.get(variant, {}).get("baking_s")
            rows.append(dict(variant=variant, accuracy=q["exact_match"], answered_rate=q["answered"], n_questions=q["n"],
                p50_ms=b["p50_ms"], p95_ms=b["p95_ms"], cost_per_1k_usd=b["amortized_ms"] / 3600 * hourly,
                bake_s=bake_s, bake_cost_usd=bake_s / 3600 * hourly if bake_s is not None else None,
                errors=errors, errors_count=len(errors) if predictions_path.is_file() else round((1-q["exact_match"]) * q["n"]),
                errors_available=predictions_path.is_file(), evaluate_run_id="saved-code-test", benchmark_run_id="saved-code-benchmark",
                snapshot_id="saved_" + quality["corpus_version"][:12], corpus_version=quality["corpus_version"]))
        return dict(run_id="saved-code-comparison", project=project, rows=rows, executed_now=False,
            scope="saved_offline_61q_all_repositories", corpus_version=quality["corpus_version"],
            pricing=dict(serve_usd_per_hour=hourly), cost_note="Saved offline results: 61 held-out questions across all three repositories, not a new test of the selected project. Costs use measured batch=1 service time and an assumed GPU tariff. Offline RAG was dense-only; live RAG combines dense + BM25.")

    @router.get("/api/projects/{project}/compare")
    def compare(project: str):
        return saved(project)

    @router.post("/api/projects/{project}/evaluate")
    def evaluate(project: str, payload: dict):
        if payload.get("suite"):
            fail("LIVE_EVALUATION_NOT_IMPLEMENTED", "Uploaded suites are not executed here. The Compare tab shows the saved 61-question offline experiment.", 501)
        result = saved(project)
        return dict(run_id="saved-code-test", kind="evaluate", project=project, executed_now=False,
            scope=result["scope"], results={r["variant"]: dict(n=r["n_questions"], accuracy=r["accuracy"],
                answered_rate=r["answered_rate"], errors=[], errors_available=False) for r in result["rows"]})

    @router.post("/api/projects/{project}/benchmark")
    def benchmark(project: str, payload: dict):
        result = saved(project)
        perf = json.loads((root / "docs/results/benchmark.json").read_text(encoding="utf-8"))
        requested = payload.get("batch_sizes", [1, 4])
        return dict(run_id="saved-code-benchmark", kind="benchmark", project=project, executed_now=False,
            scope=result["scope"], batch_sizes=[1, 4], requested_not_measured=[b for b in requested if b not in (1, 4)],
            results={v: item["batches"] for v, item in perf["results"].items() if v in ("rag", "latent", "address", "hybrid")})

    return router
