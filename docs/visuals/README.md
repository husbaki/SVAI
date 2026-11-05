# HyperWarp visuals

All numerical plots are generated from saved codebase experiment JSON with Matplotlib. Both SVG and PNG are supplied, with identical data.

- `quality_overall`: generated exact match, 61 questions.
- `quality_by_repository`: separate ItsDangerous, HTTPX and Django QA samples.
- `hybrid_ablation`: same checkpoint with latent or address channel disabled.
- `quality_by_task`: behavior, location, unseen behavior and unanswerable tracks. This shows the observed abstention failure explicitly.
- `latency_and_training`: measured p50/p95 at batch=1 and measured memory-training seconds.
- `chart_data.json`: metric snapshot used to reproduce these figures.
- `generate_charts.py`: deterministic figure generator.
- `architecture-ai.png`: AI-generated educational infographic, not a numerical result.

The exact computational graph, loss, CRUD semantics and caveats are documented in `../ARCHITECTURE.md`, including Mermaid diagrams.

Image generation used the built-in `image_gen` tool. No CLI fallback and no API key were used.

## Architecture image prompt

Create a detailed wide landscape educational infographic titled "HyperWarp: RAG vs Shared Hybrid Memory". Cream-white background; navy typography; mint trainable components; blue frozen components; amber storage. Three lanes: RAG online, shared hybrid online, offline baking plus CRUD updates. RAG: question → frozen MiniLM → cosine top-k over precomputed source embeddings → selected source text → frozen Qwen2.5-0.5B → answer and references. Hybrid: question → frozen MiniLM → shared gated trunk phi(z)=(Az)*(Gz), LayerNorm → parallel latent head (8 vectors), address head (2 categorical slots), unknown head. Latent vectors enter Qwen through inputs_embeds prefix. Address IDs gather source snippets from a local exact table (commit/file/symbol/lines/hashes) and enter the same Qwen. Unknown leads to abstention. Source values remain in local storage, not all inside weights. Offline: pinned repositories → AST records and verified QA → training memory only; answer-only CE gradient passes through frozen Qwen back to prefix, not through MiniLM. Address CE and unknown BCE train auxiliary heads. CRUD → snapshot validation → matching hybrid or explicit RAG fallback/rebake. Clear readable English labels; no numerical accuracy claims.

## Targeted correction prompt

Preserve the two online lanes and CRUD row. Correct only the training row to show frozen MiniLM → trainable memory and prefix → frozen Qwen → answer-only CE. Dashed gradient goes backward through frozen Qwen to prefix/memory; Qwen weights stay fixed. Encoder has no backward arrow. Keep address CE and unknown BCE attached to trainable memory, and show "Only memory weights update".
