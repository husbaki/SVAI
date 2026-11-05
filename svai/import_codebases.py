"""Import portable source records; no source absolute paths are required at runtime."""
import argparse
import json
from pathlib import Path
import os

from .store import Store


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks", type=Path, required=True)
    parser.add_argument("--db", type=Path, default=Path(os.getenv("SVAI_DB", "data/svai.sqlite3")))
    parser.add_argument("--repositories", nargs="+")
    args = parser.parse_args()
    catalog = json.loads((args.benchmarks / "codebases.json").read_text(encoding="utf-8"))
    def records():
        for repo in catalog["repositories"]:
            if args.repositories and repo["name"] not in args.repositories:
                continue
            path = args.benchmarks / "indexes" / repo["name"] / "code_records.jsonl"
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    yield repo["name"], json.loads(line), repo["upstream"]
    store = Store(args.db)
    count = store.import_records(records())
    print(json.dumps(dict(imported=count, repositories=store.repositories(), revision=store.revision())))


if __name__ == "__main__":
    main()
