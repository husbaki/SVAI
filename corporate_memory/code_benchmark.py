"""Executable, source-grounded QA across the three pinned repositories."""
from __future__ import annotations
import ast
import copy
import hashlib
import json
import sys
import textwrap
from pathlib import Path

from .data import dump, fingerprint, read, save_corpus, UNKNOWN


def compact_source(source):
    tree = ast.parse(textwrap.dedent(source))
    class Compact(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            node = self.generic_visit(node)
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                node.body.pop(0)
            node.body = node.body or [ast.Pass()]
            return node
        visit_AsyncFunctionDef = visit_FunctionDef
        def visit_ClassDef(self, node):
            node.body = [n for n in node.body if isinstance(n, (ast.Assign, ast.AnnAssign))] or [ast.Pass()]
            return node
    return ast.unparse(Compact().visit(tree))


def prepare(output="experiments/code_corpus"):
    base = Path("benchmarks/repos")
    for path in (base / "itsdangerous/src", base / "httpx", base / "django"):
        sys.path.insert(0, str(path.resolve()))
    import itsdangerous as it
    from itsdangerous import encoding as enc
    import httpx
    from django.conf import settings
    if not settings.configured:
        settings.configure(SECRET_KEY="benchmark-only", USE_TZ=True)
    from django.utils import html, text, encoding as de, crypto, dateparse, timezone
    from datetime import datetime, timezone as tz

    facts, lookup, dedup = [], {}, {}
    for repo in ("itsdangerous", "httpx", "django"):
        path = Path("benchmarks/indexes") / repo / "code_records.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            f = json.loads(line)
            f["repository"] = repo
            raw = f["value"]
            # Keep all indexed addresses, including overload declarations, as distractors.
            f["source_text"] = raw
            f["source_content_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
            f["value"] = f["entity"]
            try:
                code = compact_source(raw)
            except SyntaxError:
                code = raw
            f["context"] = f"# {repo}/{f['address']['path']} :: {f['entity']}\n{code}"
            f["retrieval_text"] = f"{repo} {f['entity']} {f['address']['path']}\n{raw}"
            facts.append(f)
            # The last declaration is the implementation after typing.overload stubs.
            lookup[repo, f["entity"], f["address"]["path"]] = f
    def resolve(repo, symbol):
        options = [f for (r, s, p), f in lookup.items() if r == repo and s == symbol]
        if not options:
            raise ValueError((repo, symbol))
        return options[-1]
    def exception(fn):
        try:
            fn()
        except Exception as exc:
            return type(exc).__name__
        return "NO_EXCEPTION"
    def timestamp(age, limit):
        s = it.TimestampSigner("key")
        s.get_timestamp = lambda: 1000
        signed = s.sign("x")
        s.get_timestamp = lambda: 1000 + age
        return exception(lambda: s.unsign(signed, max_age=limit))
    cases = []
    def add(repo, symbols, q, test, expected, probe, heldout=False):
        observed = str(probe())
        if observed != expected:
            raise AssertionError(f"Ground truth mismatch {q}: {observed!r} != {expected!r}")
        cases.append(dict(repository=repo, symbols=symbols, question=q, test_question=test,
            answer=expected, observed=observed, heldout=heldout,
            supporting_fact_ids=sorted(resolve(repo, s)["fact_id"] for s in symbols)))
    add("itsdangerous", ["Signer.validate"], "What does Signer.validate return for a malformed signature?", "An invalid signed token reaches the Signer validation helper. What boolean is returned?", "False", lambda:it.Signer("k").validate("broken"))
    add("itsdangerous", ["Signer.unsign"], "Which exception does Signer.unsign raise when the separator is missing?", "A Signer receives an unsigned string with no dot. Name the exception raised by unsign.", "BadSignature", lambda:exception(lambda:it.Signer("k").unsign("broken")))
    add("itsdangerous", ["Signer.__init__"], "What is the default separator character for Signer?", "Without a sep argument, which character separates a value from its signature?", ".", lambda:it.Signer("k").sep.decode())
    add("itsdangerous", ["Signer.derive_key"], "Which exception is raised for an unknown key_derivation method?", "Derive a signing key with key_derivation='invalid'. Which exception results?", "TypeError", lambda:exception(lambda:it.Signer("k",key_derivation="invalid").derive_key()))
    add("itsdangerous", ["Signer.verify_signature"], "Can a rotated Signer verify a token signed using an older allowed key?", "Does verification still accept the old key when both old and new secret keys are configured?", "True", lambda:it.Signer(["old","new"]).validate(it.Signer("old").sign("x")))
    add("itsdangerous", ["Signer.derive_key", "Signer.sign"], "Does key rotation sign new values with the newest configured key?", "Are signatures produced with keys ['old','new'] identical to using only 'new'?", "True", lambda:it.Signer(["old","new"]).sign("x")==it.Signer("new").sign("x"))
    add("itsdangerous", ["base64_encode"], "Does URL-safe base64_encode keep trailing equals padding?", "Is '=' padding preserved by the package's base64 encoding helper?", "False", lambda:enc.base64_encode(b"a").endswith(b"="))
    add("itsdangerous", ["bytes_to_int"], "What integer does bytes_to_int return for empty bytes?", "Decode an empty byte string with the package's integer helper. Give the integer.", "0", lambda:enc.bytes_to_int(b""))
    add("itsdangerous", ["TimestampSigner.unsign"], "Which exception is raised when a timestamp signature exceeds max_age?", "The signed payload is 11 seconds old and max_age is 10. What exception is raised?", "SignatureExpired", lambda:timestamp(11,10))
    add("itsdangerous", ["TimestampSigner.unsign"], "Is a signature exactly max_age seconds old rejected?", "Age equals max_age exactly. Does unsign raise an exception or succeed? Return NO_EXCEPTION if it succeeds.", "NO_EXCEPTION", lambda:timestamp(10,10), True)
    add("itsdangerous", ["TimestampSigner.unsign"], "Which exception handles a timestamp from the future?", "A signature has a negative age and max_age is enabled. Name the exception.", "SignatureExpired", lambda:timestamp(-1,10), True)
    add("itsdangerous", ["int_to_bytes"], "What is the byte order used by int_to_bytes for integer 256?", "Does encoding integer 256 put byte 1 before byte 0? Return True or False.", "True", lambda:enc.int_to_bytes(256)==b"\x01\x00", True)
    add("httpx", ["Response.is_success"], "Is HTTP status 204 considered successful by Response.is_success?", "What does the success predicate return for a 204 response?", "True", lambda:httpx.Response(204).is_success)
    add("httpx", ["Response.is_error"], "Is HTTP status 404 classified as an error by Response.is_error?", "A response status is 404. What boolean does the error property report?", "True", lambda:httpx.Response(404).is_error)
    add("httpx", ["Response.is_redirect"], "Does Response.is_redirect classify status 302 as a redirect?", "For a 302 HTTP response, is the redirect flag True or False?", "True", lambda:httpx.Response(302).is_redirect)
    add("httpx", ["Response.raise_for_status"], "Which exception does raise_for_status raise on a 404 response with a request attached?", "Calling raise_for_status on a requested URL that returned 404 raises which exception class?", "HTTPStatusError", lambda:exception(lambda:httpx.Response(404,request=httpx.Request("GET","https://example.com")).raise_for_status()))
    add("httpx", ["Response.raise_for_status"], "Which exception occurs if raise_for_status has no request attached?", "A newly constructed Response lacks its Request. Name the error from raise_for_status.", "RuntimeError", lambda:exception(lambda:httpx.Response(200).raise_for_status()))
    add("httpx", ["Headers.__getitem__"], "Are HTTPX header lookups case insensitive?", "Can Headers({'X-Test':'yes'}) be read with key 'x-test'? Return True or False.", "True", lambda:httpx.Headers({"X-Test":"yes"})["x-test"]=="yes")
    add("httpx", ["Timeout.__init__"], "What is the read timeout for Timeout(5)?", "HTTPX receives a scalar timeout of 5. Give the configured read timeout.", "5", lambda:httpx.Timeout(5).read)
    add("httpx", ["Response.json"], "What Python value is decoded from JSON true by Response.json?", "An HTTPX response body contains the JSON literal true. Give the decoded Python boolean.", "True", lambda:httpx.Response(200,content=b"true").json())
    add("httpx", ["Response.is_success"], "Does the success flag include a 299 status?", "Is the upper end of HTTPX's 2xx range, status 299, considered successful?", "True", lambda:httpx.Response(299).is_success, True)
    add("httpx", ["Response.is_error"], "Does the error flag include status 302?", "Is a redirection response also marked as an HTTP error by is_error?", "False", lambda:httpx.Response(302).is_error, True)
    add("django", ["escape"], "What does django.utils.html.escape return for '<b>'?", "HTML-escape the literal <b> using Django. Return only the resulting text.", "&lt;b&gt;", lambda:html.escape("<b>"))
    add("django", ["strip_tags"], "What remains when strip_tags processes '<b>Hello</b>'?", "Remove the HTML tags from <b>Hello</b> with Django's helper. Give the result.", "Hello", lambda:html.strip_tags("<b>Hello</b>"))
    add("django", ["slugify"], "What is slugify('Hello World') with default options?", "Convert Hello World into Django's default URL slug.", "hello-world", lambda:text.slugify("Hello World"))
    add("django", ["force_str"], "What does force_str return for UTF-8 bytes b'hello'?", "Django decodes the byte string b'hello' with force_str. What text is produced?", "hello", lambda:de.force_str(b"hello"))
    add("django", ["constant_time_compare"], "Does constant_time_compare return True for equal strings?", "Comparing 'same' with 'same' using Django's timing-safe equality helper gives which boolean?", "True", lambda:crypto.constant_time_compare("same","same"))
    add("django", ["constant_time_compare"], "Does constant_time_compare return True for different strings?", "What does Django's constant-time comparison return for 'one' and 'two'?", "False", lambda:crypto.constant_time_compare("one","two"))
    add("django", ["parse_date"], "What does parse_date return when the input is not a date?", "Pass the string 'not-a-date' to Django's date parser. Give its return value.", "None", lambda:dateparse.parse_date("not-a-date"))
    add("django", ["is_aware"], "Is an aware UTC datetime classified as aware?", "For datetime(2025,1,1,tzinfo=UTC), what boolean does Django is_aware return?", "True", lambda:timezone.is_aware(datetime(2025,1,1,tzinfo=tz.utc)))
    add("django", ["is_naive"], "Is a datetime without tzinfo classified as naive?", "What does Django is_naive report for datetime(2025,1,1) without a timezone?", "True", lambda:timezone.is_naive(datetime(2025,1,1)))
    add("django", ["slugify"], "Does slugify retain accents when allow_unicode is true?", "With allow_unicode=True, does slugifying 'café' preserve 'é'? Return True or False.", "True", lambda:text.slugify("café",allow_unicode=True)=="café", True)
    add("django", ["parse_date"], "Which exception is raised for a well-formatted but impossible date?", "Django parses '2025-02-30'. Name the exception instead of its message.", "ValueError", lambda:exception(lambda:dateparse.parse_date("2025-02-30")), True)
    add("django", ["constant_time_compare"], "Can constant_time_compare compare bytes with matching text?", "Django compares b'abc' and 'abc' with constant_time_compare. Give the resulting boolean.", "True", lambda:crypto.constant_time_compare(b"abc","abc"), True)
    splits = {s:[] for s in ("train","dev","test")}
    def qa(question, answer, supports, split, tags, case_id):
        splits[split].append(dict(question=question, answer=answer, supporting_fact_ids=supports,
            answerable=bool(supports), family=f"{split}_{tags[0]}",tags=tags,case_id=case_id,
            copy_evaluable=("location" in tags or "unknown" in tags)))
    used = set()
    for i,c in enumerate(cases):
        tags=[c["repository"],"behavior","heldout_behavior" if c["heldout"] else "paraphrase"]
        if len(c["supporting_fact_ids"])>1:tags.append("multi")
        if not c["heldout"]:
            qa(f"In {c['repository']}: {c['question']}",c["answer"],c["supporting_fact_ids"],"train",tags,f"case-{i}")
            qa(f"Code review for {c['repository']}. {c['question']} Return only the result.",c["answer"],c["supporting_fact_ids"],"train",tags,f"case-{i}")
            qa(f"Check {c['repository']}: {c['test_question']} Give a short literal answer.",c["answer"],c["supporting_fact_ids"],"dev",tags,f"case-{i}")
            used.update(c["supporting_fact_ids"])
        qa(f"Repository {c['repository']}. {c['test_question']}",c["answer"],c["supporting_fact_ids"],"test",tags,f"case-{i}")
    for f in facts:
        if f["fact_id"] not in used:continue
        for split,pattern in [("train","Which source symbol implements {symbol} in {repo}? Return the symbol name."),("train","Locate {repo}'s {symbol}. Give its qualified symbol name."),("dev","Name the qualified symbol for {symbol} in {repo}."),("test","I need {symbol} from {repo}; return its qualified name.")]:
            qa(pattern.format(symbol=f["entity"],repo=f["repository"]),f["entity"],[f["fact_id"]],split,[f["repository"],"location"],"location-"+f["fact_id"])
    for repo in ("itsdangerous","httpx","django"):
        for split,pattern in [("train","Where is {repo}'s nonexistent frobnicate_quantum_cache implemented?"),("dev","Does {repo} contain a source definition for imaginary quantum_invoice_dispatch?"),("test","Find the imaginary intergalactic_payment_engine symbol in {repo}.")]:
            qa(pattern.format(repo=repo),UNKNOWN,[],split,[repo,"unknown"],"unknown-"+repo)
    manifest=save_corpus(output,facts,splits,"real_codebase")
    dump(Path(output)/"behavior_verification.json",dict(cases=cases,passed=len(cases),
        imports={"itsdangerous":it.__file__,"httpx":httpx.__file__},scope="executed case fixtures; not exhaustive proof of all repository behavior"))
    manifest.update(repository_counts={r:sum(f["repository"]==r for f in facts) for r in ("itsdangerous","httpx","django")},
        source_index_versions={r:read(Path("benchmarks/indexes")/r/"code_manifest.json")["snapshot"] for r in ("itsdangerous","httpx","django")},
        heldout_behavior_cases=sum(c["heldout"] for c in cases),behavior_cases=len(cases),
        protocol="common pinned source corpus; independent question forms; heldout behavior intents absent from train/dev; no test tuning")
    dump(Path(output)/"manifest.json",manifest)
    return manifest
