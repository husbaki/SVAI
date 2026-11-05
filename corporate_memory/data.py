from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

UNKNOWN = "UNKNOWN"


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def fact_text(f):
    if "context" in f:
        return f["context"]
    conditions = ", ".join(f"{k}={v}" for k, v in sorted(f["conditions"].items()))
    return f'{f["entity"]}; {conditions}; {f["relation"]}; version={f["version"]}: {f["value"]} [{f["source_id"]}]'


def validate(facts, splits):
    ids = {f["fact_id"] for f in facts}
    if len(ids) != len(facts):
        raise ValueError("Duplicate fact IDs")
    seen, families = set(), {}
    for split, rows in splits.items():
        families[split] = set()
        for q in rows:
            if q["question"] in seen:
                raise ValueError("Question leakage across/within splits")
            seen.add(q["question"])
            if not set(q["supporting_fact_ids"]) <= ids:
                raise ValueError("Unknown supporting fact ID")
            if q["answerable"] != bool(q["supporting_fact_ids"]):
                raise ValueError("Inconsistent answerability")
            if not q["answerable"] and q["answer"] != UNKNOWN:
                raise ValueError("Unanswerable QA must use UNKNOWN")
            if "family" not in q:
                raise ValueError("Annotated QA must provide a split-specific family")
            families[split].add(q["family"])
    names = list(families)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if families[a] & families[b]:
                raise ValueError(f"Family leakage: {a}/{b}")


def synthetic(seed=17, products=2):
    rng = random.Random(seed)
    names = ["Atlas", "Atlas Plus", "Atlas Pro", "Atlas Prime", "Nova", "Nova Plus"]
    if not 1 <= products <= len(names):
        raise ValueError("products must be between 1 and 6")
    facts, splits = [], {s: [] for s in ("train", "dev", "test")}
    # Each split uses a different question family. Knowledge remains shared.
    templates = {
        "train": ["What is the {r} for {scope}?", "Look up {scope}: {r}.", "Give the {r} under {scope}."],
        "dev": ["Please state {scope}'s {r}."],
        "test": ["Helpdesk request: {scope}. Which {r} applies?"],
    }

    def addqa(fs, relation, scope, tags):
        fs = sorted(fs, key=lambda x: x["fact_id"])
        for split, patterns in templates.items():
            for t, pattern in enumerate(patterns):
                splits[split].append(dict(question=pattern.format(r=relation, scope=scope),
                    answer="; ".join(f["value"] for f in fs) if fs else UNKNOWN,
                    supporting_fact_ids=[f["fact_id"] for f in fs], answerable=bool(fs),
                    family=f"{split}_{t}", tags=tags))

    current = {}
    for p, name in enumerate(names[:products]):
        for country in ("Germany", "Kazakhstan"):
            for tier in ("SMB", "Enterprise"):
                local = []
                for r, unit in (("credit limit", "EUR"), ("retention period", "days")):
                    for version in ("2024", "current"):
                        value = str(rng.randrange(5, 45) * (100 if unit == "EUR" else 1)) + " " + unit
                        f = dict(fact_id=f"F{len(facts):04}", entity=name, relation=r,
                            conditions=dict(country=country, customer=tier, division="Finance" if unit == "EUR" else "Operations"),
                            value=value, version=version, source_id=f"POL-{p}-{country[:2]}-{tier}-{version}")
                        facts.append(f)
                        scope = f"{name}, {country}, {tier}, {f['conditions']['division']}, version {version}"
                        addqa([f], r, scope, ["versions"] if version != "current" else ["standard"])
                        if version == "current":
                            local.append(f)
                scope = f"{name}, {country}, {tier}, current policy"
                addqa(local, "credit limit and retention period", scope, ["multi"])
                current[name, country, tier] = local
                f = dict(fact_id=f"F{len(facts):04}", entity=name, relation="credit limit",
                    conditions=dict(country=country, customer=tier, division="Finance", exception="emergency"),
                    value=f"{rng.randrange(50, 90) * 100} EUR", version="current",
                    source_id=f"EXC-{p}-{country[:2]}-{tier}")
                facts.append(f)
                addqa([f], "credit limit", scope + ", emergency exception", ["exceptions"])
        for country in ("Japan", "Canada"):
            addqa([], "credit limit", f"{name}, {country}, SMB, current policy", ["unknown"])
        for relation, value in (("effective date", f"2025-0{p + 1}-15"), ("approval identifier", f"APR-AT-{p + 1:03}")):
            f = dict(fact_id=f"F{len(facts):04}", entity=name, relation=relation, conditions=dict(scope="global"),
                value=value, version="current", source_id=f"GLOBAL-{p}")
            facts.append(f)
            addqa([f], relation, f"{name}, global, current policy", ["identifiers"] if "identifier" in relation else ["dates"])
    validate(facts, splits)
    return facts, splits


def save_corpus(output, facts, splits, kind, teacher=None):
    if not facts:
        raise ValueError("No verified facts available; inspect teacher rejection output or provide labeled JSONL")
    validate(facts, splits)
    output = Path(output)
    dump(output / "facts.json", facts)
    for name, rows in splits.items():
        dump(output / f"{name}.json", rows)
    manifest = dict(kind=kind, version=fingerprint(facts), facts=len(facts),
                    splits={s: len(v) for s, v in splits.items()}, teacher=teacher or {})
    dump(output / "manifest.json", manifest)
    return manifest


def load_documents(paths):
    facts, splits, docs = [], {s: [] for s in ("train", "dev", "test")}, []
    expanded = []
    for p in map(Path, paths):
        expanded.extend(sorted(p.rglob("*")) if p.is_dir() else [p])
    for path in expanded:
        if path.suffix.lower() not in (".txt", ".md", ".jsonl"):
            continue
        if path.suffix.lower() != ".jsonl":
            docs.append(dict(source_id=path.name, text=path.read_text(encoding="utf-8-sig")))
            continue
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if "fact_id" in row:
                facts.append({k: row[k] for k in ("fact_id", "entity", "relation", "conditions", "value", "version", "source_id")})
            elif "question" in row:
                if "split" not in row:
                    raise ValueError("Labeled QA requires an explicit split and family")
                split = row.pop("split")
                splits[split].append(row)
            else:
                docs.append(dict(source_id=row.get("source_id", path.name), text=row["text"]))
    return facts, splits, docs


def prepare_teacher(docs, generate, cache_dir, teacher_id):
    """Offline only. Teacher never sees held-out evaluation questions."""
    facts, qa, rejected = [], [], []
    cache_dir = Path(cache_dir)
    prompt_version = "extract-v1"
    for doc in docs:
        key = fingerprint([doc, teacher_id, prompt_version])
        path = cache_dir / f"{key}.json"
        if path.exists():
            payload = read(path)
        else:
            prompt = ('Extract facts and training QA as JSON {"facts": [{"entity": str, "relation": str, '
                '"conditions": {}, "value": str, "version": str}], "qa": [{"question": str, '
                '"fact_indices": [int]}]}. Copy value verbatim from source. No markdown. '
                'Do not invent missing knowledge. Source:\n' + doc["text"])
            raw = generate(prompt)
            try:
                payload = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
            except (ValueError, json.JSONDecodeError):
                rejected.append(dict(source=doc["source_id"], reason="invalid_teacher_json", raw=raw))
                continue
            dump(path, payload)
        accepted = {}
        normalized = " ".join(doc["text"].split()).casefold()
        for i, f in enumerate(payload.get("facts", [])):
            value = str(f.get("value", ""))
            if not value.strip() or " ".join(value.split()).casefold() not in normalized:
                rejected.append(dict(source=doc["source_id"], reason="ungrounded_value", value=value))
                continue
            f = {k: f[k] for k in ("entity", "relation", "conditions", "value", "version")}
            f.update(fact_id="T" + fingerprint([doc, i, f])[:16], source_id=doc["source_id"])
            facts.append(f)
            accepted[i] = f
        for q in payload.get("qa", []):
            ids = q.get("fact_indices", [])
            if not ids or any(i not in accepted for i in ids):
                continue
            support = sorted([accepted[i] for i in ids], key=lambda f: f["fact_id"])
            qa.append(dict(question=q["question"], answer="; ".join(f["value"] for f in support),
                supporting_fact_ids=[f["fact_id"] for f in support], answerable=True,
                family="teacher_train_v1", tags=["teacher"]))
    return facts, qa, rejected
