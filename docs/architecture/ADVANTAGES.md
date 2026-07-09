# Architecture Advantages

This document explains why the modernized layout is materially stronger than the earlier mixed-layout repository.

## 1. Package-first structure

Reusable logic now lives in `src/edumind/` instead of being spread across top-level feature folders. That gives the project:

- clearer imports
- fewer path hacks
- easier testing
- cleaner boundaries between reusable code and runtime wrappers

## 2. Thin entrypoints

The repository separates product logic from delivery surfaces:

- `apps/` for Streamlit
- `services/` for FastAPI
- `experiments/` for benchmark runners

This makes it easier to change one interface without rewriting core OCR or RAG code.

## 3. One dependency source of truth

`pyproject.toml` now defines project metadata, dependency groups, and developer tooling. That removes the ambiguity of scattered requirement files and makes setup much more predictable.

## 4. Cleaner runtime state management

The repo now routes local state into `artifacts/` instead of storing databases, caches, and generated outputs next to source code. This improves:

- Git hygiene
- resetability
- reproducibility
- collaborator onboarding

## 5. Better documentation boundaries

Current operating docs now live under `docs/setup`, `docs/architecture`, and `docs/experiments`, while historical notes stay in `docs/archive`. That keeps the front-door guidance reliable without throwing away legacy material.

## 6. More testable design

The package boundaries make it easier to test:

- path and config helpers
- file handling and validation
- import smoke checks
- experiment utilities

That testing shape is already reflected in the `tests/` directory and CI workflow.

## 7. Better portfolio signal

The project now communicates stronger engineering habits:

- clear structure
- reproducible setup
- centralized tooling
- cleaner Git behavior
- optional advanced modes without making them the default
