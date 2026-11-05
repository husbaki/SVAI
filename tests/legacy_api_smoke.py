"""Real HTTP contract checks for the existing memory.html frontend."""
import argparse
import json
from pathlib import Path
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://127.0.0.1:8011")
args = parser.parse_args()


def call(path, payload=None, method=None, expected=200):
    request = urllib.request.Request(args.url + path, data=None if payload is None else json.dumps(payload).encode(),
        method=method, headers={"Content-Type": "application/json"})
    try:
        response = urllib.request.urlopen(request, timeout=180)
    except urllib.error.HTTPError as exc:
        if exc.code != expected:
            raise
        return json.load(exc)
    assert response.status == expected
    return json.load(response)


job_ids, created_ids, checks = [], [], []
temporary_project = "compat-smoke-" + uuid.uuid4().hex[:10]
base = "/api/projects/itsdangerous"
try:
    assert "itsdangerous" in call("/api/projects")["projects"]
    overview = call(base)
    assert overview["facts"] == 79 and set(overview["active_snapshot"]["variants"]) == {"rag", "hybrid"}
    assert call("/api/projects/not-a-project", expected=404)["detail"]["code"] == "PROJECT_NOT_FOUND"
    checks.append("real_projects_overview_and_error_schema")
    answer = call(base + "/query", dict(text="How does Signer.validate handle an invalid signature?", variant="hybrid", compare=False))
    assert answer["answer"] == "False" and answer["actual_mode"] == "hybrid" and answer["facts"]
    source = call(base + "/document?" + urllib.parse.urlencode(dict(doc=answer["facts"][0]["doc_id"], corpus=answer["corpus_version"])))
    assert "validate" in source["text"]
    checks.append("actual_hybrid_query_and_source_drawer")
    comparison = call(base + "/compare")
    hybrid = next(r for r in comparison["rows"] if r["variant"] == "hybrid")
    assert comparison["executed_now"] is False and hybrid["accuracy"] == 46 / 61
    assert hybrid["errors_available"] and len(hybrid["errors"]) == hybrid["errors_count"] == 15
    assert hybrid["bake_s"] > 0
    checks.append("saved_offline_metrics_actual_errors_and_measured_bake_time")
    assert call(base + "/evaluate", {})["executed_now"] is False
    assert call(base + "/evaluate", dict(suite=[dict(question="custom", answer="x")]), expected=501)["detail"]["code"] == "LIVE_EVALUATION_NOT_IMPLEMENTED"
    benchmark = call(base + "/benchmark", dict(batch_sizes=[1, 8, 32]))
    assert benchmark["executed_now"] is False and benchmark["requested_not_measured"] == [8, 32]
    checks.append("archived_evaluate_benchmark_are_not_fresh_experiments")
    job = call(base + "/bake", dict(variants=["hybrid"]))
    job_ids.append(job["job_id"])
    assert job["status"] == "succeeded" and job["trained_now"] is False and job["result"]["reused"]
    assert call("/api/jobs/" + job["job_id"])["status"] == "succeeded"
    checks.append("existing_snapshot_validation_not_simulated_training")
    target = "/api/projects/" + temporary_project
    uploaded = call(target + "/upload", dict(files=[dict(path="policy.md", content="The maximum retry count is 3.")]))
    created_ids.extend(uploaded["changes"]["added"])
    assert len(created_ids) == 1
    failed_job = call(target + "/bake", dict(variants=["hybrid"]))
    job_ids.append(failed_job["job_id"])
    assert failed_job["status"] == "failed" and failed_job["error"]["code"] == "REBAKE_REQUIRED"
    first = call(target + "/query", dict(text="What is the maximum retry count?", variant="hybrid"))
    assert first["actual_mode"] == first["variant"] == "rag" and first["fallback_reason"] and "3" in first["answer"]
    revised = call(target + "/upload", dict(files=[dict(path="policy.md", content="The maximum retry count is 7.")]))
    assert revised["changes"]["changed"] == created_ids and revised["corpus_version"] != uploaded["corpus_version"]
    second = call(target + "/query", dict(text="What is the maximum retry count?", variant="hybrid"))
    assert second["actual_mode"] == "rag" and "7" in second["answer"]
    checks.append("persistent_upload_update_real_qwen_answer_3_to_7_and_explicit_rag_fallback")
    output = dict(checks=checks, query=dict(answer=answer["answer"], actual_mode=answer["actual_mode"], facts=len(answer["facts"])),
        saved_comparison=dict(accuracy=hybrid["accuracy"], questions=hybrid["n_questions"], error_count=hybrid["errors_count"], executed_now=False),
        changed_document=dict(before=first["answer"], after=second["answer"], actual_mode=second["actual_mode"]),
        scope="Live compatibility integration checks; archived quality remains a separate 61-question experiment")
finally:
    for key in created_ids:
        call("/api/documents/" + key, method="DELETE")
    from svai.store import Store
    store = Store("data/svai.sqlite3")
    with store.connect() as db:
        for key in job_ids:
            db.execute("DELETE FROM compat_jobs WHERE id=?", (key,))
        db.execute("DELETE FROM compat_changes WHERE project=?", (temporary_project,))
checks.append("temporary_documents_and_jobs_cleaned")
Path("docs/results/legacy_api_validation.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
print(json.dumps(output, indent=2))
