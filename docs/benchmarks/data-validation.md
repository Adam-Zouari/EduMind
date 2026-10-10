# Benchmark data validation

[Dataset preparation](datasets.md) · [Shared methodology](methodology.md) ·
[Run commands](running.md)

Data validation establishes whether the exact inputs are ready for a benchmark.
It does not run candidates, qualify hardware, score predictions, or select models.
Automatic checks establish consistency, not annotation truth: a reviewer must
still check transcripts, blank sources, evidence, and labels against the assets.

**Implementation status:** this page defines the approved validator interface
and report-reuse contract. The standalone coordinator and reusable reports still
require implementation. Current runners perform their existing inline checks;
the commands below are planned, not available commands.

## Preparation and execution

Data preparation and benchmark startup have two separate responsibilities:

| Step | Who starts it, and when? | What does it establish? | Result |
|---|---|---|---|
| Full data validation | The engineer runs the standalone validator after preparing/reviewing a dataset, and again after relevant changes. | The inputs satisfy the annotation, media, schema, corpus, and split-isolation rules. | A successful report recording exactly which inputs and requirements passed. |
| Validated-input verification | The benchmark runner performs it automatically before every applicable profile invocation. | The inputs about to be used are the same ones that passed full validation, under the same applicable requirements. | Continue execution on a match; stop on a missing, failed, unsupported, or stale report. |

The second step checks the evidence from the first; it does not repeat the full
data-validation procedure or create a new successful validation report.

### Full data validation during preparation

The standalone validator checks schemas, assets, media properties, references,
corpus requirements, and cross-split isolation. Run it once per frozen input
version, then retain its report. Annotation truth still requires human review:
automatic checks cannot establish whether a transcript correctly describes speech.

One domain validator handles every applicable profile of its benchmark. The
coordinator invokes those same validators for one profile, all profiles, or all
benchmarks; there is no separate implementation for each profile. A batch
invocation validates the applicable inputs for each profile, not one universal
dataset that can replace their different manifests. `preflight` uses reviewed
development/stress inputs, not validation or locked content; its stress-input
requirements are checked alongside development inputs.

### Automatic validated-input verification before execution

Before starting any candidate worker, the runner:

1. Resolves the invocation's actual manifests, references, assets, and applicable
   data requirements, including any explicit input overrides.
2. Checks that the matching report has a supported version and successful status.
3. Recomputes the current manifest and referenced input-file checksums from their
   actual contents. It compares the current input fingerprint, data-relevant
   protocol requirements, validator/normalization identity, and frozen cross-split
   inventory identity with those recorded by full validation.
4. Continues only when they match. A missing file, changed content, changed
   applicable requirement, or missing/failed report stops execution and instructs
   the engineer to run full validation for the affected inputs.

Finding a report with the right filename, or trusting unchanged checksum fields
inside a manifest, is not enough. Changing an audio file without editing its
manifest must still fail verification. Hashing actual files requires I/O, but
verification does not repeat media decoding, annotation checks, or corpus review.
It does not load a model or silently run the full validator to repair a mismatch.

This check runs before each applicable `smoke`, `preflight`, `development`,
`validation`, or `locked` invocation, not before every sample or repetition.
All candidates in that invocation consume the same verified inputs, which must
remain immutable throughout execution. CPU/CUDA smoke can reuse the same data
report when their fixtures and data requirements match; qualification still has
its own hardware contract.

### Example: ASR development inputs

Assume model preparation and the other execution prerequisites are satisfied:

```text
prepare and review ASR development speech, controls, and stress inputs
-> run full data validation -> save the successful report

launch ASR development
-> automatic validated-input verification finds unchanged inputs
-> start candidate loading, warmup, and measured requests

edit a reference transcript or replace an audio file
-> launch ASR development again
-> verification detects changed content -> stop before candidate execution
-> rerun full validation -> retain the new report -> launch again
```

The validation profile later uses its own unseen speech/control inputs and their
matching report. A development report does not validate those different inputs.
Running the standalone validator for all profiles can prepare all their reports
in one batch, without letting candidate execution inspect held-out answers.

The benchmark **`validation` profile** is a third, different concept: it evaluates
engineer-selected finalists on unseen data. It is neither full data validation
nor the automatic validated-input verification that happens before it starts.
Passing input verification does not bypass model-lock, hardware-qualification,
or engineer-decision gates; those keep their separate execution requirements.

Full data validation and validated-input verification finish before resource
monitoring and candidate timing start. They contribute neither latency nor RAM/VRAM observations
to candidate metrics. Do not run validators concurrently with measured workers.
Validation can warm operating-system file caches; the cold-load contract remains
a fresh process/model load, not a promise of empty disk caches.

Cross-split checks need an inventory of source IDs, checksums, families, and
reviewed duplicate groups across the frozen manifests. A reviewer may validate
held-out annotations offline, but development workers must not inspect their
answers. Never run all-profile semantic review as part of development inference.
Incomplete inventories cannot certify authoritative split isolation. A smoke-only
report covers its declared fixtures and does not certify absent authoritative
datasets or their cross-split isolation.

## Folder and command interface

The proposed implementation reuses schema, checksum, evidence-span, and leakage
primitives in `common/datasets.py`. Domain modules exist only for real
domain-specific checks.

When suites share an identical input contract, reuse its validator rather than
adding a forwarding-only module. The tree shows domain ownership; each added
module must contain real checks beyond the shared primitives.

```text
experiments/benchmarks/
├── validate.py                    # one public coordinator
├── common/datasets.py             # shared data-validation primitives
├── extraction/
│   ├── document/validation.py
│   ├── audio/validation.py
│   └── video/validation.py
├── rag/
│   ├── chunking_embedding/validation.py
│   ├── retrieval_reranking/validation.py
│   ├── generation/validation.py
│   └── final/validation.py
└── vectordb/validation.py

artifacts/benchmarks/data-validation/
└── <benchmark>/<fingerprint>.json  # generated reports, not committed data
```

The existing generic manifest-validation helpers feed this coordinator rather
than becoming a second full validator. Sealing a manifest checksum is not proof
that its references, corpus coverage, or split isolation passed validation.

Planned commands, from the repository root:

```powershell
python -m experiments.benchmarks.validate audio --profile development
python -m experiments.benchmarks.validate audio --profile all
python -m experiments.benchmarks.validate document --profile all
python -m experiments.benchmarks.validate video --profile all
python -m experiments.benchmarks.validate chunking-embedding --profile all
python -m experiments.benchmarks.validate retrieval-reranking --profile all
python -m experiments.benchmarks.validate generation --profile all
python -m experiments.benchmarks.validate final-rag --profile locked
python -m experiments.benchmarks.validate vectordb --profile all
python -m experiments.benchmarks.validate all --profile all
```

`--profile smoke|development|validation|locked` chooses one applicable dataset;
`all` validates every applicable dataset and the cross-split inventory. Final
RAG has only locked evaluation inputs; non-locked integration fixtures are not
another Final RAG selection profile. A smoke-only preparation check can use
`python -m experiments.benchmarks.validate all --profile smoke`; suites without
that profile are explicitly reported as not applicable. Missing required
authoritative data is an error, never a silent skip or a successful all-data
report. This permits smoke checks while authoritative datasets are unpopulated.

## Report identity and reuse

A validation report contains its schema/validator version, benchmark, applicable
profile requirements, status, check results, errors, reviewed provenance,
source/group counts, and the exact input fingerprints. Its identity covers:

- manifest contents and referenced asset/reference checksums;
- data-relevant resolved protocol requirements, including preprocessing and
  normalization versions; and
- validator implementation/version and cross-split inventory identity.

A changed reference, asset, split, validation rule, or applicable requirement
requires full validation again and a new matching report. An unchanged dataset
does not need full revalidation merely because another benchmark invocation is
started: automatic verification confirms that the existing report still applies.
GPU/driver identity and model placement belong
to hardware preflight, not this data fingerprint. Candidate-dependent limits
remain execution/preflight checks; changing a model need not invalidate a data
report unless its required input contract changes. A successful report is
reusable across CPU/CUDA smoke and later invocations only when their input and
profile requirements match; success on smoke does not certify development.

Store reports locally under the path above. Benchmark provenance references the
consumed report and checksum, and MLflow receives it as an input artifact, not
as a quality metric or candidate child. Review decisions, model locks, protocols,
and data-validation reports retain their separate responsibilities.

## Shared checks and domain requirements

All applicable validators check field presence and types before filling defaults,
unique IDs, exact file checksums, declared splits, source provenance/licenses,
reference consistency, and reviewed independent-source groups. Corpus sizes,
required slices, and statistical support requirements come from the frozen
protocol and manifest review. Empty manifests and missing required annotations
are invalid; an explicitly reviewed empty annotation can be valid.

IDs are unique within their manifest. Suites may intentionally reuse the same
frozen inputs in the same split; the all-benchmark coordinator must not flag
that reuse as duplication within one corpus. Cross-split checks instead reject
the same reviewed source/duplicate group in incompatible splits, using the
declared shared holdout policy.

| Benchmark | Additional checks |
|---|---|
| Document | Claimed reference capabilities, physical page inventory, element order/identity, boxes/types/hierarchy, explicit table/formula positives and negatives, and scorer-compatible reference representations. |
| ASR | Canonical audio decoding, positive duration within 30 seconds with the frozen manifest tolerance, speech/control labels, transcripts, required timed segments, condition coverage, and separate reliability-control splits. |
| Video | Duration, transcript and visible-text annotations, nonempty timed visual units within their intervals, reviewed reappearances, and development/stress input coverage. Frozen-ASR artifacts have separate manifest/protocol integrity checks before visual execution. |
| Chunking and embedding | Canonical source text, exact evidence intervals, unique evidence IDs, answerability, evidence types, and source-group isolation. Generated chunks and native-model input lengths are checked during candidate execution. |
| Retrieval and reranking | The same reviewed RAG inputs and selected upstream data identities. Runtime pool collection, chunk-content, and dense/BM25 index checksums remain artifact-integrity gates, not data annotations. |
| Generation | Required facts, accepted answers, supplied evidence for answerable questions, unanswerable labels, and frozen context provenance. Judge calibration/readiness and predicted-output validation are separate gates. |
| Final RAG | Its untouched held-out inputs, frozen composition's input requirements, and source isolation from component selection/confirmation. Automatic checks cannot prove that a reviewer has never examined a holdout. |
| Vector database | Workload dimensions/counts, seeds, vector/metadata/query shape, finite values, filter cases, and exact-search reference inputs for synthetic or selected-real workloads. Database readiness is an environment gate. |

### Document references, including blank sources

Authoritative references declare `reference_capabilities`. Validate only claimed
annotations; smoke may infer eligibility from explicitly supplied fixture fields.
Missing fields must not become verified empty values through defaults.

- `text` with an explicit reviewed `reference_text: ""` is valid. A missing text
  annotation is invalid when text is claimed. Documents need no global positive
  prose-word quota: blank or structure-only sources are legitimate.
- Explicit complete empty element sequences are valid for negative detection or
  reading-order tasks. No eligible attribute objects makes that attribute task
  inapplicable; missing claimed element annotations are invalid.
- PDF/image references identify every physical page, including blank pages;
  an image has one page. Native DOCX does not acquire invented page geometry.
- `has_table:false` and `has_formula:false` are verified negatives. Presence
  labels and object lists must agree. A real blank-cell table still needs a
  valid structure; a claimed formula needs a nonempty expression.
- Roots explicitly record a null parent; an omitted parent field is not a root
  annotation. IDs resolve, parent graphs are acyclic, and required levels agree.
- Coordinates are finite, use the declared units/page geometry, and have positive
  area. Objects without usable comparison text need a reviewed structural or
  non-text identity; two empty prose strings cannot identify distinct objects.

Prose emptiness is evaluated after the frozen symmetric projection, even when
raw strings differ. That projection must not erase formula, code, or table-tree
identity. A valid blank source is not a missing source file. Reference/scorer
compatibility is a data requirement; installed Docker evaluators and model
components are environment requirements.

### ASR references and controls

Every transcript field must be explicitly supplied and reviewed against the
recording. An omitted annotation cannot be identified as silence automatically.
The validator cannot detect a falsely empty human annotation from its string
alone. Canonical media checks and human review therefore have different jobs.

Decode to mono, 16 kHz, signed 16-bit PCM. Check positive actual duration no
greater than 30 seconds and agreement with manifest duration within the frozen
0.1-second tolerance; that agreement tolerance does not extend the duration cap.

Assess lexical emptiness after the frozen prose projection. A legitimately
empty projected clip is allowed, including text consisting only of removed
punctuation; retain its reviewed speech/control label. Do not relabel it as
nonspeech automatically. Positive lexical speech requires complete, nonempty,
ordered reference segments with finite boundaries inside the clip. A legitimate
clip with no lexical timing task has an explicit empty segment list, not missing
annotations. Missing required timing or contradictory text/timing is invalid.

Every full authoritative speech split must contain some normalized reference
words and eligible timed segments, as well as its required condition coverage.
Reject an entirely empty speech split during preparation. This is a corpus
sanity check, not a ban on individual empty projections or an evaluation rule
reapplied to bootstrap draws. Reviewed nonspeech controls have empty projected
spoken references and cover silence, music without lyrics, background noise,
and environmental sound in each authoritative split.

## Data validity and metric applicability

Validation makes the inputs valid, not every metric applicable to every sample.
A formula-free document has no formula reconstruction task; a nonspeech clip has
no required spoken boundary to measure. Such rows retain an `inapplicable` status
and null value. A required task with missing annotation fails data validation.
Rejection produces a validation-error report, not measured candidate failures
or quality scores. It must not contribute a zero or null observation to a
benchmark aggregate that never ran.

`null` is a storage value, not a reason. Metric artifacts distinguish
`inapplicable` (no reference task), `unavailable` (a value could not be measured),
`incomplete` (required records missing), and `invalid_reference` (inputs rejected
before execution). Valid empty comparisons and observed attempt failures follow
each metric's explicit scoring convention. Data checks must not replace those
rules with a universal empty/null policy.

Dataset checks validate only the input contract. They do not reject a dataset
because its planned CI support is too small: valid small diagnostic slices can
retain point estimates without intervals. A missing required corpus allocation
or annotation is invalid; an optional slice with insufficient independent
support is a reporting limitation recorded in the validation report.
