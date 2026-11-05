import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
import threading
from pathlib import Path

from svai.ml import lexical
from svai.store import Store, digest


class PersistentCRUDTests(unittest.TestCase):
    def test_crud_revision_and_search_after_update(self):
        with tempfile.TemporaryDirectory() as folder:
            db = Path(folder) / "test.sqlite3"
            store = Store(db)
            record = store.create("Memo", "oldmarker quartz", "demo")
            self.assertEqual(store.revision(), 1)
            docs, rev = store.snapshot()
            self.assertGreater(lexical(docs, "oldmarker")[0], 0)
            changed = store.update(record["id"], dict(content="newmarker cobalt", expected_version=1))
            self.assertEqual(changed["version"], 2)
            self.assertEqual(changed["content_sha256"], digest("newmarker cobalt"))
            with self.assertRaises(ValueError):
                store.update(record["id"], dict(content="stale", expected_version=1))
            docs, revision = Store(db).snapshot()
            self.assertEqual(revision, 2)
            self.assertEqual(lexical(docs, "oldmarker")[0], 0)
            self.assertGreater(lexical(docs, "newmarker")[0], 0)
            store.delete(record["id"])
            self.assertEqual(Store(db).all(), [])
            self.assertEqual(store.revision(), 3)

    def test_filters_are_bound_parameters(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(Path(folder) / "test.sqlite3")
            a = store.create("A", "alpha", "one")
            store.create("B", "beta", "two")
            self.assertEqual(len(store.all(["one"])), 1)
            self.assertEqual(store.all(["one' OR 1=1 --"]), [])
            self.assertEqual(store.all(document_ids=[a["id"]])[0]["title"], "A")

    def test_concurrent_edit_rejects_lost_update(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(Path(folder) / "test.sqlite3")
            record = store.create("Concurrent", "initial")
            barrier = threading.Barrier(2)
            def edit(value):
                barrier.wait()
                try:
                    store.update(record["id"], dict(content=value, expected_version=1))
                    return "updated"
                except ValueError:
                    return "conflict"
            with ThreadPoolExecutor(2) as pool:
                outcomes = list(pool.map(edit, ["first", "second"]))
            self.assertCountEqual(outcomes, ["updated", "conflict"])
            self.assertEqual(store.get(record["id"])["version"], 2)

    def test_snapshot_and_bm25_cache_invalidated(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(Path(folder) / "test.sqlite3")
            record = store.create("Cache", "beforemarkerunique")
            before, revision = store.snapshot()
            self.assertGreater(lexical(before, "beforemarkerunique")[0], 0)
            store.update(record["id"], dict(content="aftermarkerunique"))
            after, new_revision = store.snapshot()
            self.assertGreater(new_revision, revision)
            self.assertEqual(lexical(after, "beforemarkerunique")[0], 0)
            self.assertGreater(lexical(after, "aftermarkerunique")[0], 0)

    def test_import_is_idempotent_across_git_newline_snapshots(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(Path(folder) / "test.sqlite3")
            content = "def same():\n    return True\n"
            address = dict(commit="abc", path="same.py", symbol="same", start_line=1, end_line=2,
                fragment_sha256=digest(content), file_sha256="crlf", snapshot="windows")
            first = dict(fact_id="windows-id", entity="same", value=content, address=address)
            self.assertEqual(store.import_records([("repo", first, "https://example.test/repo")]), 1)
            second = dict(first, fact_id="linux-id", address=dict(address, file_sha256="lf", snapshot="linux"))
            self.assertEqual(store.import_records([("repo", second, "https://example.test/repo")]), 0)
            self.assertEqual(len(store.all()), 1)


if __name__ == "__main__":
    unittest.main()
