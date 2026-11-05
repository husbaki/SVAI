"""Download pinned public repositories, statically extract source and import it.

No downloaded Python code is imported or executed. All generated data stays in data/.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import tokenize

from .store import Store


REPOSITORIES = {
    "itsdangerous": ("pallets/itsdangerous", "672971d66a2ef9f85151e53283113f33d642dabd", "src/itsdangerous"),
    "httpx": ("encode/httpx", "b5addb64f0161ff6bfe94c124ef76f6a1fba5254", "httpx"),
    "django": ("django/django", "ca3f31b255dee9d046eba41cbdb2adcfc4297588", "django"),
}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def symbols(text):
    result, stack = [], []
    class Visitor(ast.NodeVisitor):
        def symbol(self, node, kind):
            start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
            result.append(dict(symbol=".".join(stack + [node.name]), kind=kind, start_line=start, end_line=node.end_lineno))
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
    return result


def records(root, package, commit):
    names = subprocess.run(["git", "ls-files", "-z", "--", "*.py"], cwd=root, capture_output=True, check=True).stdout
    paths = sorted(root / p.decode() for p in names.split(b"\0") if p)
    paths = [p for p in paths if p.relative_to(root).as_posix().startswith(package + "/")]
    metadata, texts = [], {}
    for path in paths:
        relative = path.relative_to(root).as_posix()
        with tokenize.open(path) as stream:
            text = stream.read()
        texts[relative] = text
        metadata.append(dict(path=relative, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            bytes=path.stat().st_size, physical_lines=len(text.splitlines())))
    snapshot = fingerprint(metadata)
    for meta in metadata:
        text = texts[meta["path"]]
        lines = text.splitlines(keepends=True)
        for symbol in symbols(text):
            snippet = "".join(lines[symbol["start_line"] - 1:symbol["end_line"]])
            address = dict(snapshot=snapshot, commit=commit, path=meta["path"], file_sha256=meta["sha256"],
                **symbol, fragment_sha256=hashlib.sha256(snippet.encode()).hexdigest())
            key = "CODE-" + fingerprint(address)[:24]
            yield dict(fact_id=key, entity=symbol["symbol"], value=snippet, address=address)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repositories", nargs="+", choices=REPOSITORIES, default=list(REPOSITORIES))
    parser.add_argument("--data", type=Path, default=Path("data"))
    args = parser.parse_args()
    args.data.mkdir(parents=True, exist_ok=True)
    store = Store(args.data / "svai.sqlite3")
    for name in args.repositories:
        upstream, commit, package = REPOSITORIES[name]
        url = "https://github.com/" + upstream
        target = (args.data / "repos" / name).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            subprocess.run(["git", "-c", "core.autocrlf=false", "clone", "--depth", "1", "--no-checkout", url + ".git", str(target)], check=True)
        subprocess.run(["git", "fetch", "--depth", "1", "origin", commit], cwd=target, check=True)
        subprocess.run(["git", "-c", "core.autocrlf=false", "checkout", "--detach", commit], cwd=target, check=True)
        result = store.import_records((name, record, url) for record in records(target, package, commit))
        print(json.dumps(dict(repository=name, imported=result, commit=commit)), flush=True)
    print(json.dumps(dict(repositories=store.repositories(), revision=store.revision())))


if __name__ == "__main__":
    main()
