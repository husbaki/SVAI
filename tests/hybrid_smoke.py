"""Exercise real hybrid execution, scoped addresses and stale-snapshot fallback."""
import json
from pathlib import Path
import urllib.request

BASE = "http://127.0.0.1:8011"


def call(path, payload=None, method=None):
    request = urllib.request.Request(BASE + path, data=None if payload is None else json.dumps(payload).encode(),
        method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


status = call("/api/model/status")
assert status["hybrid"]["ready"] and status["hybrid"]["matched_records"] == 12244
results = dict(initial_status=status, questions=[])
cases = [
    ("itsdangerous", "How does Signer.validate handle an invalid signature?", "False"),
    ("httpx", "What exception does Response.raise_for_status raise on an error status?", "HTTPStatusError"),
    ("django", "What does Django transaction atomic do when an exception is raised?", "rollback"),
]
for repository, question, expected in cases:
    response = call("/api/query", dict(question=question, repositories=[repository], mode="hybrid", use_ml=True, top_k=2))
    assert response["actual_mode"] == "hybrid" and response["trained_code_memory_executed"]
    assert response["hybrid"]["latent_tokens"] == 8
    assert all(c["repository"] == repository for c in response["citations"])
    results["questions"].append(dict(repository=repository, question=question, expected=expected,
        answer=response["answer"], expected_literal_present=expected.lower() in response["answer"].lower(),
        actual_mode=response["actual_mode"], citations=[c["title"] for c in response["citations"]],
        elapsed_ms=response["elapsed_ms"]))
    print(json.dumps(results["questions"][-1]), flush=True)
# Touch a copied source record, not the original corpus; its changed hash must reject hybrid.
original = call("/api/documents?repository=itsdangerous&q=Signer.validate&limit=20")["documents"]
original = next(d for d in original if d["title"] == "Signer.validate")
original = call("/api/documents/" + original["id"])
copied = call("/api/documents", dict(title="stale-snapshot-probe", repository="itsdangerous",
    content=original["content"] + "\n# Changed source snapshot for CRUD smoke\n", metadata=original["metadata"]))
try:
    fallback = call("/api/query", dict(question="What does Signer.validate return?", document_ids=[copied["id"]], mode="hybrid", use_ml=True))
    assert fallback["actual_mode"] == "rag"
    assert fallback["fallback_reason"] == "source_changed_rebake_required"
    assert not fallback["trained_code_memory_executed"]
    assert fallback["citations"][0]["verification"]["source_snapshot_match"] is False
    results["stale_snapshot"] = dict(actual_mode=fallback["actual_mode"], fallback_reason=fallback["fallback_reason"],
        changed_records=fallback["hybrid"]["changed_records"], answer=fallback["answer"])
finally:
    call("/api/documents/" + copied["id"], method="DELETE")
assert call("/api/model/status")["hybrid"]["ready"]
results["scope"] = "API integration smoke; expected_literal_present is diagnostic, not held-out benchmark accuracy"
Path("data/hybrid_api_smoke.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print(json.dumps(results["stale_snapshot"]))
