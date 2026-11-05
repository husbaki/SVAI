import unittest

from svai.hybrid import HybridAdapter


class SnapshotStatusTests(unittest.TestCase):
    def setUp(self):
        self.facts = [dict(repository="repo", address=dict(commit="abc", path="a.py", symbol=name,
            start_line=i, end_line=i, fragment_sha256=str(i)), source_content_sha256=str(i))
            for i, name in enumerate(("first", "second"), 1)]
        self.documents = [dict(id=f"doc-{i}", repository="repo", content_sha256=f["source_content_sha256"],
            metadata=dict(source_address=f["address"])) for i, f in enumerate(self.facts)]
        artifact = dict(facts=self.facts, corpus_version="fixture", config=dict(memory_tokens=8))
        self.adapter = HybridAdapter(None)
        self.adapter.artifact = lambda: dict(artifact=artifact)

    def test_complete_snapshot_ready(self):
        self.assertTrue(self.adapter.status(self.documents)["ready"])

    def test_deleted_or_filtered_record_invalidates_snapshot(self):
        status = self.adapter.status(self.documents[:1])
        self.assertFalse(status["ready"])
        self.assertEqual(status["missing_records"], 1)
        self.assertEqual(status["reason"], "source_removed_or_filtered_rebake_required")

    def test_modified_record_invalidates_snapshot(self):
        self.documents[0]["content_sha256"] = "changed"
        self.assertEqual(self.adapter.status(self.documents)["reason"], "source_changed_rebake_required")

    def test_new_record_invalidates_snapshot(self):
        self.documents.append(dict(id="new", repository="repo", content_sha256="new", metadata={}))
        self.assertEqual(self.adapter.status(self.documents)["reason"], "new_documents_rebake_required")


if __name__ == "__main__":
    unittest.main()
