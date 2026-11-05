# Compiled corporate memory: measured experiment

Corpus: **real_codebase**, version `b507935b976d`. Results refer only to this labeled document corpus.

Frozen LLM: `Qwen/Qwen2.5-0.5B-Instruct`. Frozen encoder: `sentence-transformers/all-MiniLM-L6-v2`.

| Variant | Generated EM | Copied EM | Fact F1 | p50 / p95 ms (b=1) | TTFT ms | q/s (b=1) | peak VRAM MiB | corpus MiB | USD/answer* |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 0.164 | n/a | n/a | 642.5 / 2550.4 | 325.5 | 0.94 | 1125.1 | 0.000 | 0.00010378 |
| rag | 0.410 | 0.444 | 0.410 | 465.8 / 505.2 | 258.5 | 2.14 | 1127.7 | 56.752 | 0.00004544 |
| oracle | 0.377 | 1.000 | 1.000 | 431.9 / 460.7 | 228.8 | 2.31 | 1147.0 | 38.817 | 0.00004216 |
| latent | 0.738 | n/a | n/a | 481.8 / 799.4 | 259.8 | 1.73 | 1125.5 | 5.995 | 0.00005633 |
| address | 0.377 | 0.852 | 0.896 | 589.5 / 841.9 | 336.2 | 1.59 | 1158.4 | 53.074 | 0.00006132 |
| hybrid | 0.754 | 0.852 | 0.847 | 636.5 / 797.8 | 345.7 | 1.53 | 1159.5 | 58.492 | 0.00006352 |
| hybrid_no_latent | 0.377 | 0.852 | 0.847 | 1300.8 / 4222.4 | 963.3 | 0.46 | 1158.4 | 58.492 | 0.00021161 |
| hybrid_no_address | 0.475 | n/a | n/a | 1598.1 / 2088.7 | 1070.4 | 0.69 | 1125.6 | 58.492 | 0.00014114 |

Highest generated EM: **hybrid**; highest among trained architectures: **hybrid**.
Results establish behavior only for this corpus, split, seed and training budget. Shared training benefit must be judged from the hybrid and both ablations, not assumed.

## Errors on held-out question families

| Variant | Exception error | Old-version error | Multi-fact error | Unanswerable error | Answer coverage | Source F1 |
|---|---:|---:|---:|---:|---:|---:|
| none | n/a | n/a | 1.000 | 0.000 | 0.180 | n/a |
| rag | n/a | n/a | 1.000 | 0.000 | 0.689 | 0.410 |
| oracle | n/a | n/a | 1.000 | 0.000 | 0.639 | 1.000 |
| latent | n/a | n/a | 0.000 | 1.000 | 0.984 | n/a |
| address | n/a | n/a | 1.000 | 0.000 | 0.639 | 0.896 |
| hybrid | n/a | n/a | 0.000 | 1.000 | 1.000 | 0.847 |
| hybrid_no_latent | n/a | n/a | 1.000 | 0.000 | 0.639 | 0.847 |
| hybrid_no_address | n/a | n/a | 0.000 | 1.000 | 0.951 | n/a |

## Training completed

| Variant | Baking seconds | Chosen alpha / beta | Best dev score | Reload answers identical |
|---|---:|---:|---:|---|
| latent | 76.8 | 1.0 / 0.25 | 0.3502 | True |
| address | 2.0 | 1.0 / 0.25 | 0.4435 | True |
| hybrid | 85.9 | 1.0 / 0.5 | 0.9115 | True |

Both hybrid coefficients and stopping checkpoints were selected on dev. Train loss alone is not a reliability result.

## Typed output accuracy

| Variant | Numbers incl. dates | ISO dates | Identifiers |
|---|---:|---:|---:|
| none | 0.500 | n/a | 1.000 |
| rag | 0.500 | n/a | 1.000 |
| oracle | 0.500 | n/a | 1.000 |
| latent | 1.000 | n/a | 0.000 |
| address | 0.500 | n/a | 1.000 |
| hybrid | 1.000 | n/a | 0.000 |
| hybrid_no_latent | 0.500 | n/a | 1.000 |
| hybrid_no_address | 1.000 | n/a | 0.000 |

## Economics

*USD figures are scenarios at configurable assumed tariffs, not actual invoices. Service time is measured; tariffs are assumptions.
GPU assumption: 0.35 USD/hour; allocated interval: 720 hours.

| Variant | Baking USD | Metered break-even Q vs RAG | 10k metered / dedicated USD | 100k metered / dedicated USD | 1M metered / dedicated USD |
|---|---:|---:|---:|---:|---:|
| none | 0.0000 | none (no per-query saving) | 1.04 / 252.00 | 10.38 / 252.00 | 103.78 / 252.00 |
| rag | 0.0041 | none (no per-query saving) | 0.46 / 252.00 | 4.55 / 252.00 | 45.44 / 252.00 |
| oracle | 0.0000 | 0 | 0.42 / 252.00 | 4.22 / 252.00 | 42.16 / 252.00 |
| latent | 0.0075 | none (no per-query saving) | 0.57 / 252.01 | 5.64 / 252.01 | 56.34 / 252.01 |
| address | 0.0002 | none (no per-query saving) | 0.61 / 252.00 | 6.13 / 252.00 | 61.32 / 252.00 |
| hybrid | 0.0083 | none (no per-query saving) | 0.64 / 252.01 | 6.36 / 252.01 | 63.53 / 252.01 |
| hybrid_no_latent | 0.0083 | none (no per-query saving) | 2.12 / 252.01 | 21.17 / 252.01 | 211.62 / 252.01 |
| hybrid_no_address | 0.0083 | none (no per-query saving) | 1.42 / 252.01 | 14.12 / 252.01 | 141.15 / 252.01 |

## Measurement limits

- Generated answers and exact local field copies are separate outputs. B is neural addressing plus a local fact store.
- RAG uses precomputed embeddings and batched device retrieval; top-k and score threshold are chosen on dev.
- Latent inference reads neither source documents nor the fact table; B/C use the fact snapshot inside their checkpoint.
- Quality is evaluated on all held-out test questions. Benchmark percentiles use the sample count in benchmark.json and are exploratory.
- Batch latency is request latency for the whole batch; amortized GPU service time is reported separately.
- Startup measures shared model initialization once and each checkpoint load, excluding imports, process creation and model downloads.
- Generation shares the same token cap and stopping policy across variants; see benchmark.json for EOS handling.
- Peak allocated VRAM includes all resident heads; allocator reserved memory is recorded separately. Shared base weights are excluded from corpus disk size.
- No quantization or custom kernels applied. Compile outcome is in compile_probe.json when tested.
- Free-text explanation quality and real-document generalization are not measured.
- SECDED corrects corrupted code bits, not semantic fact-selection mistakes. See code-head checkpoint and ECC tests.
- Update costs assume a complete rebuild. GPU idle hours are billed in dedicated scenarios; metered break-even does not apply directly to them.
