"""Live API smoke, including a real local Qwen generation and CRUD invalidation."""
import json
from pathlib import Path
import sys
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8011"


def request(path, data=None, method=None):
    body = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(BASE + path, data=body, method=method,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as response:
        return json.load(response)


results = {}
results["health"] = request("/api/health")
record = request("/api/documents", dict(title="CRUD smoke", content="quartzprobeunique version one", repository="smoke"))
key = record["id"]
try:
    found = request("/api/query", dict(question="quartzprobeunique", document_ids=[key], use_ml=False))
    assert found["citations"][0]["document_id"] == key
    updated = request("/api/documents/" + key, dict(content="cobaltprobeunique version two", expected_version=1), "PUT")
    assert updated["version"] == 2
    assert request("/api/query", dict(question="quartzprobeunique", document_ids=[key], use_ml=False))["status"] == "unknown"
    found = request("/api/query", dict(question="cobaltprobeunique", document_ids=[key], use_ml=False))
    assert found["citations"][0]["version"] == 2
    results["crud_update"] = dict(valid=True, updated_version=2, fresh_index_revision=found["index_revision"])
finally:
    request("/api/documents/" + key, method="DELETE")
try:
    request("/api/documents/" + key)
    raise AssertionError("Deleted document remained")
except urllib.error.HTTPError as exc:
    assert exc.code == 404
for repo, question in [("itsdangerous", "Signer verify signature"), ("httpx", "request timeout"), ("django", "transaction atomic rollback")]:
    response = request("/api/query", dict(question=question, repositories=[repo], use_ml=False))
    assert response["citations"] and all(c["verification"]["content_hash_match"] for c in response["citations"])
    results[repo] = dict(citations=[c["title"] for c in response["citations"]], elapsed_ms=response["elapsed_ms"])
assert request("/api/query", dict(question="zxqvblorptunknownsymbol", use_ml=False))["answer"] == "UNKNOWN"
results["unknown"] = True
results["real_ml"] = request("/api/query", dict(question="What does Signer.validate return when the signature is invalid?", repositories=["itsdangerous"], top_k=2, mode="rag", use_ml=True))
assert results["real_ml"]["model"]["loaded"] and results["real_ml"]["answer_type"] == "qwen_rag"
assert results["real_ml"]["citations"] and results["real_ml"]["answer"].strip()
Path("live_smoke_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print(json.dumps({k:v for k,v in results.items() if k != "real_ml"}, indent=2))
print("REAL QWEN ANSWER:", results["real_ml"]["answer"])
