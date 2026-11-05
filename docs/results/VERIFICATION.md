# Executed verification

## Source integrity

All 12,244 indexed source records were checked against the pinned source snapshots: 79 ItsDangerous, 533 HTTPX, and 11,632 Django symbols. Checks cover file hashes, fragment hashes, AST identity and line ranges. LF/CRLF differences are handled through normalized fragment identity in portable imports.

## Independent QA evaluation

The common source corpus contains 12,244 address candidates. Training has 103 questions, development 53, and test 61. Question families are disjoint across splits. Eight behavior intents occur only in test. The Python fixture runner executed 34 behavior cases and checked their expected values against the downloaded source versions.

| Model | Exact test answers |
|---|---:|
| Frozen Qwen, no memory | 10/61 |
| Dense local RAG | 25/61 |
| Oracle supporting context | 23/61 |
| Latent memory | 45/61 |
| Neural addressing + local store | 23/61 |
| Shared hybrid | 46/61 |

Hybrid scored 17/21 on the Django question subset and 5/8 on held-out behavior intents. All three test unanswerable questions were answered incorrectly by the hybrid. Retrieval correctness and generated-answer correctness are separate metrics in `quality_test.json`.

Oracle supplies gold source IDs but does not train a prefix. Its generated-answer score is therefore a context-selection reference, not an upper bound on the trained hybrid's answer formatting.

## Executed upstream tests

- ItsDangerous: 297 passed.
- HTTPX model, configuration and status-code suites: 865 passed.
- Django `utils_tests`: 717 ran, 49 skipped, no errors or failures with Python UTF-8 mode enabled.
- Research implementation checks: 7 passed, including frozen-model gradient propagation, answer-only masking, SECDED and checkpoint round trips.

These are the specified test suites, not an exhaustive execution of every Django or HTTPX test. The full codebases were indexed; semantic QA is sampled.

## Live API

Backend tests exercise CRUD, persistence across process restart, optimistic update conflicts, fresh retrieval after updates, scoped requests, and stale source references. The live API loads real local Qwen and MiniLM models. A matching code-trained checkpoint runs both latent and address channels; changes to source content trigger an explicit RAG fallback.

Live hybrid smoke examples correctly returned `False` for `Signer.validate` and `HTTPStatusError` for HTTPX error responses. A new Django transaction question failed: the model selected `strip_tags` and returned `SuspiciousOperation` instead of rollback. This failure is retained in the API smoke artifact; API integration success is not equated with semantic correctness.

The offline baseline uses dense cosine retrieval. The live API combines dense retrieval with BM25. Offline quality scores do not measure that different live RAG configuration.

Latency charts are exploratory measurements with seven samples per batch/mode. They do not establish production percentiles or a universal speed advantage.
