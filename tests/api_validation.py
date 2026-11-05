"""Real HTTP validation: CRUD, concurrent edits, source identity, refreshed search.

Run with --ml only when GPU is available; it executes actual Qwen before/after update.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://127.0.0.1:8011")
parser.add_argument("--ml", action="store_true")
args = parser.parse_args()


def call(path, payload=None, method=None, expect=200):
    request = urllib.request.Request(args.url + path,
        data=None if payload is None else json.dumps(payload).encode(), method=method,
        headers={"Content-Type": "application/json"})
    try:
        response = urllib.request.urlopen(request, timeout=180)
    except urllib.error.HTTPError as exc:
        if exc.code != expect:
            raise
        return json.load(exc)
    assert response.status == expect, (path, response.status, expect)
    return json.load(response)


checks = []
content = "initialevidencemarker = 3\n"
address = dict(commit="test-snapshot", path="example.py", symbol="example", start_line=1, end_line=1)
record = call("/api/documents", dict(title="API validation source", repository="api-validation",
    content=content, metadata=dict(source_address=address, upstream="https://example.test/repo",
    imported_content_sha256=hashlib.sha256(content.encode()).hexdigest())), expect=201)
key = record["id"]
try:
    assert call("/api/documents/" + key)["content"] == content
    assert call("/api/citations/" + key)["verification"]["source_snapshot_match"] is True
    checks.append("create_read_citation")
    query = dict(question="initialevidencemarker", document_ids=[key], use_ml=False, mode="evidence")
    assert call("/api/query", query)["citations"][0]["document_id"] == key
    updated = call("/api/documents/" + key, dict(content="updatedevidencemarker = 7\n", expected_version=1), "PUT")
    assert updated["version"] == 2
    citation = call("/api/citations/" + key)
    assert citation["verification"]["content_hash_match"] is True
    assert citation["verification"]["source_snapshot_match"] is False
    assert citation["source_url"] is None
    assert call("/api/query", query)["answer"] == "UNKNOWN"
    query["question"] = "updatedevidencemarker"
    assert call("/api/query", query)["citations"][0]["version"] == 2
    checks.append("update_invalidates_index_and_original_source_claim")
    def competing_edit(word):
        request = urllib.request.Request(args.url + "/api/documents/" + key,
            data=json.dumps(dict(content=word, expected_version=2)).encode(), method="PUT",
            headers={"Content-Type": "application/json"})
        try:
            return urllib.request.urlopen(request).status
        except urllib.error.HTTPError as exc:
            return exc.code
    with ThreadPoolExecutor(2) as pool:
        assert sorted(pool.map(competing_edit, ["race-first", "race-second"])) == [200, 409]
    checks.append("concurrent_optimistic_version_conflict")
    call("/api/query", dict(question="x", top_k=0), expect=422)
    checks.append("request_validation")
finally:
    call("/api/documents/" + key, method="DELETE")
call("/api/documents/" + key, expect=404)
call("/api/documents/" + key, method="DELETE", expect=404)
checks.append("delete_and_missing_document")
sources = {}
for repo, question in [("itsdangerous", "Signer signature verify"), ("httpx", "request timeout"), ("django", "atomic transaction rollback")]:
    times = []
    for _ in range(2):
        result = call("/api/query", dict(question=question, repositories=[repo], use_ml=False, mode="evidence"))
        assert result["citations"] and all(c["verification"]["content_hash_match"] for c in result["citations"])
        times.append(result["elapsed_ms"])
    sources[repo] = dict(symbols=[c["title"] for c in result["citations"]], first_ms=times[0], warm_ms=times[1])
checks.append("all_three_real_repositories")
assert call("/api/query", dict(question="zxqvblorptunknownsymbol", use_ml=False, mode="evidence"))["answer"] == "UNKNOWN"
checks.append("unknown_abstention")
ml_results = []
if args.ml:
    record = call("/api/documents", dict(title="Retry count policy", repository="api-validation",
        content="The maximum retry count is 3."), expect=201)
    key = record["id"]
    try:
        for version, expected in [(1, "3"), (2, "7")]:
            if version == 2:
                call("/api/documents/" + key, dict(content="The maximum retry count is 7.", expected_version=1), "PUT")
            result = call("/api/query", dict(question="What is the maximum retry count?", document_ids=[key], top_k=1, mode="rag", use_ml=True))
            assert expected in result["answer"] and result["model"]["loaded"]
            assert result["citations"][0]["version"] == version
            ml_results.append(dict(version=version, expected=expected, answer=result["answer"], model=result["model"]))
        checks.append("actual_qwen_answer_changes_after_document_update")
    finally:
        call("/api/documents/" + key, method="DELETE")
output = dict(checks=checks, sources=sources, actual_ml=ml_results,
    scope="integration correctness; not a held-out code understanding benchmark")
Path("data/api_validation.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
print(json.dumps(output, indent=2))
