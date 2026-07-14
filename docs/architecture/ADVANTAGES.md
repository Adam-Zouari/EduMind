# Architecture decisions

- Runtime and benchmarks share protocols, preventing a benchmark winner that production cannot reproduce.
- Immutable contracts and index manifests make model/chunker incompatibility explicit.
- Per-modality extraction selection avoids forcing one engine across unrelated source types.
- Rank fusion avoids combining incomparable dense and lexical scores.
- Hard gates plus Pareto selection preserve tradeoffs instead of hiding them in an arbitrary overall score.
- Lazy optional dependencies keep imports deterministic and give preflight ownership of downloads.
- Thin apps/services keep state, cleanup, validation, and duplicate prevention independently testable.

The main cost is deliberate complexity in provenance and manifests. That cost is justified because experimental results are intended to remain usable for later promotion decisions.
