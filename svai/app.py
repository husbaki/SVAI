from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .ml import LocalML, lexical
from .store import Store, digest


ROOT = Path(__file__).resolve().parents[1]
store = Store(os.getenv("SVAI_DB", str(ROOT / "data/svai.sqlite3")))
ml = LocalML()
app = FastAPI(title="HyperWarp — Code & Document Memory", version="0.1.0",
    description="Persistent document CRUD, local MiniLM + Qwen RAG and snapshot-bound hybrid memory. "
    "Responses expose the actual execution mode and source provenance.")


class CreateDocument(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    content: str = Field(min_length=1, max_length=300000)
    repository: str = Field(default="workspace", min_length=1, max_length=100)
    metadata: dict = Field(default_factory=dict)


class UpdateDocument(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=300)
    content: str | None = Field(default=None, min_length=1, max_length=300000)
    repository: str | None = Field(default=None, min_length=1, max_length=100)
    metadata: dict | None = None
    expected_version: int | None = Field(default=None, ge=1)


class QueryRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    repositories: list[str] | None = None
    document_ids: list[str] | None = None
    top_k: int = Field(default=2, ge=1, le=5)
    use_ml: bool = True
    mode: Literal["hybrid", "rag", "evidence"] = "hybrid"


@app.get("/api/health")
@app.get("/health")
def health():
    return dict(status="ok", revision=store.revision(), documents=sum(r["documents"] for r in store.repositories()),
        model=ml.status(), persistence="sqlite", trained_code_memory_executed=False)


@app.get("/api/model/status")
def model_status():
    result = ml.status()
    result["default_mode"] = "hybrid"
    try:
        from .hybrid import HybridAdapter
        result["hybrid"] = HybridAdapter(ml).status(store.all())
    except ImportError:
        result["hybrid"] = dict(ready=False, reason="hybrid_checkpoint_not_available")
    return result


@app.post("/api/model/load")
def model_load():
    try:
        return ml.load()
    except Exception as exc:
        raise HTTPException(503, f"Local model unavailable: {type(exc).__name__}: {exc}")


@app.get("/api/repositories")
def repositories():
    return dict(repositories=store.repositories(), revision=store.revision())


@app.post("/api/documents", status_code=201)
def create_document(payload: CreateDocument):
    return store.create(**payload.model_dump())


@app.get("/api/documents")
def list_documents(repository: str | None = None, q: str | None = None,
    limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)):
    return store.list(repository, q, limit, offset)


@app.get("/api/documents/{document_id}")
def get_document(document_id: str):
    result = store.get(document_id)
    if result is None:
        raise HTTPException(404, "Document not found")
    return result


@app.put("/api/documents/{document_id}")
def update_document(document_id: str, payload: UpdateDocument):
    try:
        return store.update(document_id, payload.model_dump(exclude_unset=True))
    except KeyError:
        raise HTTPException(404, "Document not found")
    except ValueError:
        raise HTTPException(409, "Version conflict: reload the current document")


@app.delete("/api/documents/{document_id}")
def delete_document(document_id: str):
    try:
        return store.delete(document_id)
    except KeyError:
        raise HTTPException(404, "Document not found")


def citation(document, score=0):
    metadata = document["metadata"]
    original = metadata.get("imported_content_sha256")
    matched = original == document["content_sha256"] if original else None
    address = metadata.get("source_address", {})
    source_url = None
    if matched and metadata.get("upstream") and address.get("commit"):
        source_url = f"{metadata['upstream']}/blob/{address['commit']}/{address['path']}#L{address['start_line']}-L{address['end_line']}"
    return dict(document_id=document["id"], title=document["title"], repository=document["repository"],
        version=document["version"], content_sha256=document["content_sha256"], metadata=metadata,
        score=round(score, 6), snippet=document["content"], source_url=source_url,
        verification=dict(content_hash_match=digest(document["content"]) == document["content_sha256"],
            source_snapshot_match=matched, scope="stored document identity; behavioral correctness is not proved"))


@app.get("/api/citations/{document_id}")
def get_citation(document_id: str):
    return citation(get_document(document_id))


def retrieve(payload):
    documents, revision = store.snapshot(payload.repositories, payload.document_ids)
    scores = lexical(documents, payload.question)
    # No lexical evidence is an abstention, not a fabricated semantic answer.
    if not documents or not any(s > 0 for s in scores):
        return [], revision
    if payload.use_ml and payload.mode != "evidence":
        try:
            scores = ml.rank(documents, payload.question, revision, scores)
        except Exception as exc:
            raise HTTPException(503, f"Local model unavailable: {type(exc).__name__}: {exc}")
    order = sorted(range(len(documents)), key=lambda i: (-scores[i], documents[i]["id"]))[:payload.top_k]
    return [citation(documents[i], scores[i]) for i in order if scores[i] > 0], revision


@app.post("/api/search")
def search(payload: QueryRequest):
    start = time.perf_counter()
    citations, revision = retrieve(payload)
    return dict(question=payload.question, status="evidence_found" if citations else "unknown",
        citations=citations, elapsed_ms=round((time.perf_counter()-start)*1000, 3), index_revision=revision,
        model=ml.status(), trained_code_memory_executed=False,
        retrieval="hybrid_dense_bm25" if payload.use_ml and payload.mode != "evidence" else "bm25")


@app.post("/api/query")
def query(payload: QueryRequest):
    fallback_reason = None
    hybrid_status = None
    if payload.mode == "hybrid" and payload.use_ml:
        try:
            from .hybrid import HybridAdapter
            adapter = HybridAdapter(ml)
            documents, revision = store.snapshot(payload.repositories, payload.document_ids)
            status = hybrid_status = adapter.status(documents)
            if status.get("ready"):
                start = time.perf_counter()
                try:
                    response = adapter.query(payload.question, documents, payload.top_k)
                except ValueError as exc:
                    raise HTTPException(422, str(exc))
                except Exception as exc:
                    raise HTTPException(503, f"Hybrid engine unavailable: {type(exc).__name__}: {exc}")
                selected = {d["id"]: d for d in documents}
                citations = [citation(selected[key]) for key in response.get("document_ids", []) if key in selected]
                return dict(question=payload.question, answer=response["answer"], citations=citations,
                    status="evidence_found" if citations and response["answer"].strip().strip(".").upper() != "UNKNOWN" else "unknown", answer_type="hybrid", actual_mode="hybrid",
                    requested_mode=payload.mode, fallback_reason=None, index_revision=revision, model=ml.status(),
                    trained_code_memory_executed=True, hybrid=status, llm_executed=bool(citations), elapsed_ms=round((time.perf_counter()-start)*1000,3))
            fallback_reason = status.get("reason", "hybrid_snapshot_not_ready")
        except ImportError:
            fallback_reason = "hybrid_checkpoint_not_available"
    result = search(payload)
    start = time.perf_counter()
    if not result["citations"]:
        result["answer"] = "UNKNOWN"
    elif payload.use_ml and payload.mode != "evidence":
        try:
            result["answer"] = ml.answer(payload.question, result["citations"])
        except Exception as exc:
            raise HTTPException(503, f"Generation failed: {type(exc).__name__}: {exc}")
    else:
        result["answer"] = "\n".join(c["title"] + " — " + c["repository"] for c in result["citations"])
    result["answer_type"] = "qwen_rag" if payload.use_ml and payload.mode != "evidence" else "source_evidence"
    result["actual_mode"] = "rag" if payload.use_ml and payload.mode != "evidence" else "evidence"
    result["requested_mode"] = payload.mode
    result["fallback_reason"] = fallback_reason
    result["hybrid"] = hybrid_status
    result["llm_executed"] = bool(result["citations"]) and payload.use_ml and payload.mode != "evidence"
    if result["answer"].strip().strip(".").upper() == "UNKNOWN":
        result["status"] = "unknown"
    result["elapsed_ms"] += round((time.perf_counter()-start)*1000, 3)
    result["caveat"] = "Source hashes verify identity. Neural relevance and generated explanations can be wrong; no trained compiled code-memory quality is claimed."
    return result


from .legacy_api import create_router
app.include_router(create_router(store, ml, query, QueryRequest, ROOT))


static = ROOT / "static"
static.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=static), name="static")


@app.get("/", include_in_schema=False)
def home():
    if (static / "index.html").is_file():
        return FileResponse(static / "index.html")
    return dict(name="HyperWarp", api="/docs", openapi="/openapi.json", default_mode="hybrid",
        features=["document_crud", "source_provenance", "local_qwen", "minilm", "versioned_index"])
