# Experiment Workflow

EduMind now uses a staged English-only experiment system designed to choose the best RAG stack for
the current student-study product flow.

## Goal

The experiments try to answer these product questions:

- which chunking strategy retrieves the most useful study context?
- which embedding model represents those chunks best?
- which local vector backend is the best fit?
- which retrieval strategy ranks results best?
- which local SLM answers student questions best from retrieved context?
- which full stack is the best overall tradeoff?

## Why the workflow is staged

The system does not test every part in total isolation and it does not brute-force every
combination from the start.

It uses a hybrid method:

1. change one segment at a time
2. keep the rest of the pipeline fixed
3. promote the best candidates
4. validate the best full stacks together

That gives you both:

- clear attribution for improvements
- realistic final product validation

## Benchmarks

The maintained datasets live in `data/evaluation/`:

- `synthetic_regression`
  Fast smoke and CI-friendly regression benchmark.
- `student_benchmark/dev`
  Main tuning split.
- `student_benchmark/holdout`
  Final confirmation split.
- `challenge_benchmark`
  Harder confirmation slice.

These are English-only for now.

## Stage order

1. `chunking`
2. `embedding`
3. `vectordb`
4. `retrieval`
5. `llm`
6. `final`

The full suite can be run with:

```bash
edumind-experiments --suite all --dataset student_benchmark --resume
```

Fast smoke validation:

```bash
edumind-experiments --suite smoke
```

## How to read winners

- A stage winner is the best candidate under that stage's fixed conditions.
- A final winner is the best promoted full stack after holdout and challenge confirmation.
- If two stage winners do not compose into the final winner, trust the final bakeoff.

## Outputs

All generated state goes to `artifacts/experiments/mlflow/`:

- MLflow database and artifacts
- vector store persistence
- staged resume caches
- leaderboards and promoted-candidate summaries

Each stage writes its own:

- `leaderboard.json`
- `leaderboard.csv`
- `best_candidates.json`
- `stage_summary.md`

## Current limitation

The staged benchmark code is now native end to end, but benchmark quality still depends on the
quality of the checked-in assets, snapshots, questions, and gold answers. The next improvement area
is data curation depth, not legacy experiment plumbing.
