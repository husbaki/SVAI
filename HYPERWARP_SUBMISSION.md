## Inspiration
AI should remember your codebase, not rediscover it with every question.

## What it does
HyperWarp turns your codebase into reusable AI memory. Its hybrid engine combines learned context with precise source retrieval, giving teams answers they can trace back to real code.

## How we built it
Qwen powers generation. A shared neural memory produces both latent context and source addresses. FastAPI and SQLite connect the engine to document management and the demo workspace.

## Challenges we ran into
Keeping memory useful as code changes—and making every answer easy to check.

## Accomplishments that we're proud of
A working hybrid memory prototype, a reproducible training pipeline, and three indexed codebases: ItsDangerous, HTTPX, and Django.

## What we learned
Learned memory helps shape the answer. Exact source access makes it checkable. HyperWarp brings them together.

## What's next for HyperWarp
Automatic memory updates, larger repositories, and a shared workspace where teams can ask, verify, and act on their code.
