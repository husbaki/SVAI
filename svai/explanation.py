"""Context-sensitive presentation of a raw model answer; never changes that answer."""
from __future__ import annotations

import ast
import re
import textwrap


def signature_rejection(question, raw_answer, citations):
    if str(raw_answer).strip().strip(".").lower() != "false":
        return None
    question = question.lower()
    if not any(word in question for word in ("invalid", "bad signature", "tamper", "неверн", "недейств", "поддел")):
        return None
    for citation in citations:
        if citation.get("title") not in ("Signer.validate", "TimestampSigner.validate"):
            continue
        try:
            tree = ast.parse(textwrap.dedent(citation["snippet"]))
        except (SyntaxError, KeyError):
            continue
        if len(tree.body) != 1 or not isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        function = tree.body[0]
        body = function.body
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
            body = body[1:]
        if function.name != "validate" or len(body) != 1 or not isinstance(body[0], ast.Try):
            continue
        branch = body[0]
        if branch.finalbody or branch.orelse or len(branch.body) != 2:
            continue
        call, success = branch.body
        if not (isinstance(call, ast.Expr) and isinstance(call.value, ast.Call) and
            isinstance(call.value.func, ast.Attribute) and isinstance(call.value.func.value, ast.Name) and
            call.value.func.value.id == "self" and call.value.func.attr == "unsign" and
            isinstance(success, ast.Return) and isinstance(success.value, ast.Constant) and success.value.value is True):
            continue
        for handler in branch.handlers:
            name = handler.type.id if isinstance(handler.type, ast.Name) else None
            if name == "BadSignature" and len(handler.body) == 1 and isinstance(handler.body[0], ast.Return):
                value = handler.body[0].value
                if isinstance(value, ast.Constant) and value.value is False:
                    return "Invalid signatures are rejected: the function catches BadSignature and returns False."
    return None


def explain_answer(question, raw_answer, citations, ml):
    if str(raw_answer).strip().strip(".").upper() == "UNKNOWN":
        return dict(text="I could not establish an answer from the available source snippets.", source="abstention")
    if not citations:
        return dict(text="The model returned an answer without a selected source fragment. Its meaning could not be verified against code.", source="unavailable")
    deterministic = signature_rejection(question, raw_answer, citations)
    if deterministic:
        return dict(text=deterministic, source="verified_ast_branch")
    try:
        result = ml.explain(question, raw_answer, citations).strip()
    except Exception:
        result = ""
    source_text = "\n".join(c["snippet"] for c in citations).lower()
    if re.search(r"\b(should|must|ought)\b", result.lower()) and not re.search(r"\b(should|must|ought)\b", source_text):
        # A predicate result alone cannot establish a recommendation about inputs.
        prefix = re.split(r",\s*(?:indicating|suggesting|meaning|therefore)\b", result, maxsplit=1, flags=re.IGNORECASE)
        result = prefix[0].rstrip(" .") + "." if len(prefix) == 2 and not re.search(r"\b(should|must|ought)\b", prefix[0].lower()) else ""
    if not result or result.upper().strip(".") == "UNKNOWN":
        return dict(text="The raw answer is available below, but I could not reliably explain its meaning from the selected source. Check the source before relying on it.", source="unavailable")
    return dict(text=result, source="qwen_source_grounded")
