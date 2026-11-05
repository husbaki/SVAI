from __future__ import annotations

from datetime import datetime, timezone
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
import uuid


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def source_identity(repository, address):
    # Git working-tree newline conventions do not change semantic source identity.
    return (repository, address.get("commit"), address.get("path"), address.get("symbol"),
            address.get("start_line"), address.get("end_line"), address.get("fragment_sha256"))


class Store:
    def __init__(self, path):
        self.path = Path(path)
        self._snapshot_cache = {}
        self._cache_lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript('''CREATE TABLE IF NOT EXISTS documents(
                id TEXT PRIMARY KEY,title TEXT NOT NULL,content TEXT NOT NULL,repository TEXT NOT NULL,
                metadata TEXT NOT NULL,content_sha256 TEXT NOT NULL,version INTEGER NOT NULL,
                created_at TEXT NOT NULL,updated_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS state(key TEXT PRIMARY KEY,value INTEGER NOT NULL);
                INSERT OR IGNORE INTO state VALUES('revision',0);
                CREATE INDEX IF NOT EXISTS document_repository ON documents(repository);''')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def row(row):
        if row is None:
            return None
        value = dict(row)
        value["metadata"] = json.loads(value["metadata"])
        return value

    def revision(self):
        with self.connect() as db:
            return db.execute("SELECT value FROM state WHERE key='revision'").fetchone()[0]

    def get(self, key):
        with self.connect() as db:
            return self.row(db.execute("SELECT * FROM documents WHERE id=?", (key,)).fetchone())

    def create(self, title, content, repository="workspace", metadata=None, key=None):
        key, timestamp = key or uuid.uuid4().hex, now()
        with self.connect() as db:
            db.execute("INSERT INTO documents VALUES(?,?,?,?,?,?,1,?,?)", (key, title, content, repository,
                json.dumps(metadata or {}, ensure_ascii=False), digest(content), timestamp, timestamp))
            db.execute("UPDATE state SET value=value+1 WHERE key='revision'")
        return self.get(key)

    def update(self, key, values):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            old = self.row(db.execute("SELECT * FROM documents WHERE id=?", (key,)).fetchone())
            if old is None:
                raise KeyError(key)
            expected = values.pop("expected_version", None)
            if expected is not None and expected != old["version"]:
                raise ValueError("version_conflict")
            for field in ("title", "content", "repository", "metadata"):
                if field in values and values[field] is not None:
                    old[field] = values[field]
            db.execute("UPDATE documents SET title=?,content=?,repository=?,metadata=?,content_sha256=?,version=version+1,updated_at=? WHERE id=?",
                (old["title"], old["content"], old["repository"], json.dumps(old["metadata"], ensure_ascii=False), digest(old["content"]), now(), key))
            db.execute("UPDATE state SET value=value+1 WHERE key='revision'")
        return self.get(key)

    def delete(self, key):
        with self.connect() as db:
            result = db.execute("DELETE FROM documents WHERE id=?", (key,))
            if not result.rowcount:
                raise KeyError(key)
            db.execute("UPDATE state SET value=value+1 WHERE key='revision'")
        return dict(deleted=True, id=key, revision=self.revision())

    def snapshot(self, repositories=None, document_ids=None):
        clauses, parameters = [], []
        for field, values in (("repository", repositories), ("id", document_ids)):
            if values is not None:
                if not values:
                    return [], self.revision()
                clauses.append(field + " IN (" + ",".join("?" for _ in values) + ")")
                parameters.extend(values)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self.connect() as db:
            db.execute("BEGIN")
            revision = db.execute("SELECT value FROM state WHERE key='revision'").fetchone()[0]
            cache_key = (revision, tuple(repositories or ()), tuple(document_ids or ()))
            with self._cache_lock:
                cached = self._snapshot_cache.get(cache_key)
            if cached is not None:
                return cached, revision
            documents = [self.row(r) for r in db.execute("SELECT * FROM documents" + where + " ORDER BY id", parameters)]
            with self._cache_lock:
                self._snapshot_cache.clear()
                self._snapshot_cache[cache_key] = documents
            return documents, revision

    def all(self, repositories=None, document_ids=None):
        return self.snapshot(repositories, document_ids)[0]

    def list(self, repository=None, q=None, limit=50, offset=0):
        where, params = [], []
        if repository:
            where.append("repository=?")
            params.append(repository)
        if q:
            where.append("(title LIKE ? OR content LIKE ?)")
            params += ["%" + q + "%"] * 2
        clause = " WHERE " + " AND ".join(where) if where else ""
        with self.connect() as db:
            total = db.execute("SELECT count(*) FROM documents" + clause, params).fetchone()[0]
            rows = [self.row(r) for r in db.execute("SELECT * FROM documents" + clause + " ORDER BY updated_at DESC,id LIMIT ? OFFSET ?", [*params, limit, offset])]
        for row in rows:
            row.pop("content")
        return dict(documents=rows, total=total, revision=self.revision())

    def repositories(self):
        with self.connect() as db:
            rows = [dict(r) for r in db.execute("SELECT repository AS name,count(*) AS documents,sum(length(content)) AS characters FROM documents GROUP BY repository ORDER BY repository")]
        return rows

    def import_records(self, records):
        timestamp, count = now(), 0
        with self.connect() as db:
            existing = set()
            for row in db.execute("SELECT repository,metadata FROM documents"):
                metadata = json.loads(row["metadata"])
                if metadata.get("source_address"):
                    existing.add(source_identity(row["repository"], metadata["source_address"]))
            for repo, record, upstream in records:
                key = repo + ":" + record["fact_id"]
                address = record["address"]
                content = record["value"]
                if digest(content) != address["fragment_sha256"]:
                    raise ValueError("Imported fragment hash does not match content")
                identity = source_identity(repo, address)
                if identity in existing:
                    continue
                metadata = dict(source_address=address, imported_content_sha256=digest(content),
                    upstream=upstream, original_fact_id=record["fact_id"], source_type="code")
                result = db.execute("INSERT OR IGNORE INTO documents VALUES(?,?,?,?,?,?,1,?,?)",
                    (key, record["entity"], content, repo, json.dumps(metadata, ensure_ascii=False), digest(content), timestamp, timestamp))
                count += result.rowcount
                existing.add(identity)
            if count:
                db.execute("UPDATE state SET value=value+1 WHERE key='revision'")
        return count
