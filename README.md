# EduMind

**A local study assistant and a benchmark suite for choosing the models and
strategies behind it.**

EduMind lets you ask questions about course notes, documents, and recorded
lectures. It answers using your material and cites numbered passages so you can
check the answer against the source.

The benchmarks ask a practical engineering question: **on a laptop with 4 GiB
of GPU memory, which choices improve answer quality enough to justify their
cost in latency and memory?** Suppose the retriever finds the paragraph you
need but ranks it twelfth. Can a reranker move it into the first five results,
and how much time does that extra step add?

> **Under active development.** The application and benchmark code are
> available. Dataset review, comparative runs, and final component selection
> are still in progress. The app's current defaults are provisional, not
> benchmark winners.

## How the assistant works

1. **Add your material.** EduMind extracts content from PDFs, Word documents
   (DOCX), images, audio, and video. It preserves source locations such as page
   numbers and timestamps where available.
2. **Ask a question.** The assistant searches the indexed material for passages
   relevant to your question.
3. **Check the answer.** The local language model uses those passages to answer.
   The interface shows the numbered evidence so you can inspect what was cited.

## What the benchmarks compare

| Benchmark | Question it answers |
|---|---|
| [Document extraction](docs/benchmarks/extraction/document/methodology.md) | Which parser preserves text, reading order, tables, and formulas in course materials? |
| [Audio extraction](docs/benchmarks/extraction/audio/methodology.md) | Which speech model transcribes technical lectures accurately and locates the words in time? |
| [Video extraction](docs/benchmarks/extraction/video/methodology.md) | Which frame-selection strategy captures slide changes, board writing, and screen edits without repeatedly extracting the same text while it remains visible? |
| [Chunking and embedding](docs/benchmarks/rag/chunking_embedding/methodology.md) | Which way of splitting and representing a document makes the evidence easiest to find? |
| [Retrieval and reranking](docs/benchmarks/rag/retrieval_reranking/methodology.md) | Which search method puts useful evidence near the top, and when is reranking worth the extra latency? |
| [Vector database](docs/benchmarks/vectordb/methodology.md) | On the same vectors, how do database configurations compare in search quality, speed, and storage use? |
| [Generation](docs/benchmarks/rag/generation/methodology.md) | Given the same evidence, which model gives a correct, supported answer, cites it accurately, and refuses when a question is unanswerable? |
| [Final RAG](docs/benchmarks/rag/final/methodology.md) | How well does the selected system answer held-out questions using evidence it retrieves itself? |

The stages are separate so their results are interpretable. A model that
answers correctly when given the right passage can still fail when the system
has to find that passage. The generation benchmark tests the first situation;
Final RAG tests the second. Final RAG starts from a prepared document corpus.
Raw-file extraction is evaluated separately.

## How components are chosen

Public evaluations help decide which candidates are worth testing. Local
experiments compare them on EduMind's material and hardware. Development runs
compare candidates; validation evaluates finalists on unseen data; locked tests
report the performance of choices already made. Smoke tests check that the
software runs, not how well a candidate performs.

MLflow records quality, latency, memory use, and supporting artifacts. An
engineer reviews the results and decides what to adopt; benchmark runs never
change the application's configuration.

## Getting started

- **Run the assistant:** [install the environment](docs/setup/installation.md),
  [prepare the app models](docs/setup/model-preparation.md#preparation-commands),
  then [start the application](docs/setup/running.md). No benchmark datasets are needed.
- **Understand the choices:** see [dataset selection](docs/benchmarks/dataset-selection.md)
  and [model selection](docs/benchmarks/model-selection.md).
- **Reproduce an experiment:** start with the [shared methodology](docs/benchmarks/methodology.md),
  prepare the [candidates](docs/benchmarks/model-preparation.md) and
  [datasets](docs/benchmarks/dataset-preparation.md) as applicable, then follow the
  [runbook](docs/benchmarks/running.md). The [metric reference](docs/benchmarks/metrics.md)
  explains how to interpret the results.
- **Explore the codebase:** the [architecture overview](docs/architecture/overview.md)
  shows how the app and experiments fit together. The [documentation map](docs/README.md)
  links to the individual guides.

## License

EduMind's source code and documentation are available under the
[PolyForm Strict License 1.0.0](LICENSE). Third-party dependencies, models,
and datasets retain their own licenses.
