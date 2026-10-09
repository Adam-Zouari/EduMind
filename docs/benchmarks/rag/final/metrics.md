# Final RAG and human review metrics

[Shared metric conventions](../../metrics.md) · [Benchmark methodology](methodology.md) · [Run commands](../../running.md)

Final RAG evaluates one frozen complete system. It reuses the approved
[retrieval and reranking metrics](../retrieval_reranking/metrics.md) and
[automated generation metrics](../generation/metrics.md), with the cutoffs and
operational boundaries defined in its [methodology](methodology.md).
The separate human-review contract is defined below.

## Human review metrics

For Final RAG, one blinded reviewer scores Faithfulness, Answer Correctness,
Completeness, and Citation Accuracy on the frozen `0` to `2` rubric, plus
Answerability Correctness on `0` or `1`. For the four `0–2` rubrics, `0` means
the requirement is not met, `1` partly met, and `2` fully met. For binary
Answerability Correctness, `0` means incorrect answer/refusal behavior and `1`
means correct behavior. The reviewer sees the
question, accepted answer, and supplied evidence for the anonymous frozen system.
These values are reported separately; they are never averaged into a universal
quality score. A single reviewer does not support an inter-reviewer agreement
claim. Human review is reporting-only and cannot reopen locked selection.
