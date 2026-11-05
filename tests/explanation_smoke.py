"""Actual Qwen explanation stage with a fixed raw value, not an answering score.

Set SVAI_MODEL_CACHE to an existing Hugging Face cache before running.
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from svai.explanation import explain_answer
from svai.ml import LocalML

source = 'def has_items(items):\n    """Return False for an empty list, True otherwise."""\n    return bool(items)\n'
result = explain_answer("What does False mean when has_items is called with an empty list?", "False",
    [dict(title="has_items", snippet=source)], LocalML())
output = dict(raw_answer="False", explanation=result["text"], explanation_source=result["source"],
    scope="Real Qwen explanation stage with a fixed raw value; not a first-pass answering score")
Path("data/explanation_smoke.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
print(json.dumps(output, indent=2), flush=True)
assert result["source"] == "qwen_source_grounded"
assert "empty" in result["text"].lower() or "no items" in result["text"].lower()
assert "invalid signatures are rejected" not in result["text"].lower()
