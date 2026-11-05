"""Static Python source snapshots and verifiable addresses. No code is executed."""
from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import tokenize
from pathlib import Path

from .data import dump, fingerprint, read


def sha(data):
    return hashlib.sha256(data).hexdigest()


def source_files(root, includes=None):
    root = Path(root).resolve()
    try:
        proc = subprocess.run(["git", "ls-files", "-z", "--", "*.py"], cwd=root, capture_output=True, check=True)
        files = [root / p.decode("utf-8") for p in proc.stdout.split(b"\0") if p]
    except (subprocess.SubprocessError, FileNotFoundError):
        files = [p for p in root.rglob("*.py") if not any(x.startswith(".") or x in ("node_modules", "__pycache__") for x in p.relative_to(root).parts)]
    prefixes = [Path(p).as_posix().rstrip("/") for p in includes or []]
    return sorted(p for p in files if p.is_file() and
        (not prefixes or any(p.relative_to(root).as_posix() == x or p.relative_to(root).as_posix().startswith(x + "/") for x in prefixes)))


def definitions(text):
    records, stack = [], []
    class Visitor(ast.NodeVisitor):
        def symbol(self, node, kind):
            first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
            name = ".".join(stack + [node.name])
            records.append(dict(symbol=name, kind=kind, start_line=first, end_line=node.end_lineno))
            stack.append(node.name)
            self.generic_visit(node)
            stack.pop()

        def visit_FunctionDef(self, node):
            self.symbol(node, "function")

        def visit_AsyncFunctionDef(self, node):
            self.symbol(node, "async_function")

        def visit_ClassDef(self, node):
            self.symbol(node, "class")
    Visitor().visit(ast.parse(text))
    return records


def index_codebase(root, output, includes=None):
    root, output = Path(root).resolve(), Path(output)
    files = source_files(root, includes)
    metadata, texts, parse_errors = [], {}, []
    for path in files:
        relative = path.relative_to(root).as_posix()
        with tokenize.open(path) as stream:
            text = stream.read()
        texts[relative] = text
        metadata.append(dict(path=relative, sha256=sha(path.read_bytes()), bytes=path.stat().st_size,
                             physical_lines=len(text.splitlines())))
    snapshot = fingerprint(metadata)
    proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    commit = proc.stdout.strip() if proc.returncode == 0 else None
    records = []
    for meta in metadata:
        text = texts[meta["path"]]
        lines = text.splitlines(keepends=True)
        try:
            symbols = definitions(text)
        except SyntaxError as exc:
            parse_errors.append(dict(path=meta["path"], error=str(exc)))
            continue
        for symbol in symbols:
            snippet = "".join(lines[symbol["start_line"] - 1:symbol["end_line"]])
            address = dict(snapshot=snapshot, commit=commit, path=meta["path"], file_sha256=meta["sha256"],
                **symbol, fragment_sha256=sha(snippet.encode("utf-8")))
            record_id = "CODE-" + fingerprint(address)[:24]
            records.append(dict(fact_id=record_id, entity=symbol["symbol"], relation="source_definition",
                conditions=dict(path=meta["path"], start_line=symbol["start_line"], end_line=symbol["end_line"]),
                value=snippet, version=commit or snapshot, source_id=record_id, address=address))
    output.mkdir(parents=True, exist_ok=True)
    with (output / "code_records.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    manifest = dict(kind="codebase_source_index", root=str(root), commit=commit, snapshot=snapshot,
        includes=includes or [], files=metadata, python_files=len(metadata), records=len(records),
        physical_lines=sum(f["physical_lines"] for f in metadata), source_bytes=sum(f["bytes"] for f in metadata),
        parse_errors=parse_errors, scope="static function/class addresses, not semantic QA or executed behavior tests",
        qa_training_performed=False)
    dump(output / "code_manifest.json", manifest)
    return manifest


def verify_codebase(root, index):
    root, index = Path(root).resolve(), Path(index)
    manifest = read(index / "code_manifest.json")
    failures, cache, checked = [], {}, 0
    observed = []
    for meta in manifest["files"]:
        path = (root / meta["path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            failures.append(dict(path=meta["path"], reason="missing_or_outside_root"))
            continue
        raw = path.read_bytes()
        with tokenize.open(path) as stream:
            text = stream.read()
        digest = sha(raw)
        try:
            symbols = {(s["symbol"], s["start_line"], s["end_line"]) for s in definitions(text)}
        except SyntaxError:
            symbols = set()
        cache[meta["path"]] = (digest, text.splitlines(keepends=True), symbols)
        observed.append(dict(path=meta["path"], sha256=digest, bytes=len(raw), physical_lines=len(text.splitlines())))
    if fingerprint(observed) != manifest["snapshot"]:
        failures.append(dict(reason="source_snapshot_mismatch"))
    with (index / "code_records.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            address = record["address"]
            checked += 1
            entry = cache.get(address["path"])
            reason = None
            if entry is None:
                reason = "missing_source"
            else:
                digest, lines, symbols = entry
                snippet = "".join(lines[address["start_line"] - 1:address["end_line"]])
                if digest != address["file_sha256"] or address["snapshot"] != manifest["snapshot"]:
                    reason = "version_or_file_hash_mismatch"
                elif (address["symbol"], address["start_line"], address["end_line"]) not in symbols:
                    reason = "ast_symbol_span_mismatch"
                elif snippet != record["value"] or sha(snippet.encode("utf-8")) != address["fragment_sha256"]:
                    reason = "fragment_mismatch"
                elif record["fact_id"] != "CODE-" + fingerprint(address)[:24]:
                    reason = "record_id_mismatch"
            if reason:
                failures.append(dict(fact_id=record["fact_id"], reason=reason))
    result = dict(records_checked=checked, valid=not failures, failures=failures,
        scope="snapshot/file/span/AST identity verification only; no semantic address prediction evaluated")
    dump(index / "source_verification.json", result)
    return result
