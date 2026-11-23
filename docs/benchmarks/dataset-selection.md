# Benchmark dataset selection

[Documentation map](../README.md) · [Model selection](model-selection.md) ·
[Dataset preparation](dataset-preparation.md) · [Data validation](data-validation.md)

**Decision, 2026-11-21:** These are the source collections selected for EduMind's
English-language benchmarks. Selection is not validation: the frozen manifest,
not the size of a public release, establishes which files, references and splits
were actually evaluated.

## How sources were chosen

Start with the input a candidate will actually receive and the reference needed
to score its output. A photograph can test image parsing, but cannot test a
parser's handling of a native Word table. A subtitle can establish what was
said, but not when a line of text disappeared from a slide.

Prefer sources with original files, traceable annotations and enough independent
documents, speakers or lectures to separate development, validation and locked
testing. Add a source only for a condition the others do not cover. Keep the
original and its derivatives together, including across collections. Check
usage and redistribution terms per release and item: a repository's code license
does not grant rights to all its source material.

## Document extraction

| Input | Selected source | Why it is in the benchmark |
|---|---|---|
| Page images | [OmniDocBench](https://huggingface.co/datasets/opendatalab/OmniDocBench) | Its element boxes, types, text, reading order, table structure and formulas support several metrics on the same page. Use the English subset and the release matched to the official evaluator. The publisher limits the dataset to noncommercial research. |
| Degraded images | [PureDocBench](https://huggingface.co/datasets/zhihengli-casia/puredocbench) | Comparing matched views of the same page isolates the effect of degradation: a score change cannot be explained by a different table or a harder formula. |
| Photographed pages | [DocPTBench](https://github.com/Topdu/DocPTBench) | Real photographs test perspective and lighting on documents with known content. Match its original pages to their OmniDocBench families; a photograph is a new capture, not an independent document. |
| Multi-page PDFs | [OHR-Bench v2](https://huggingface.co/datasets/opendatalab/OHR-Bench) | Original PDFs and page-level references let the parser process a whole file and show whether content stays on the right page. A stack of page images would not test the same input path. |
| PDF structure | [OpenDataLoader Bench](https://github.com/opendataloader-project/opendataloader-bench) | The release pairs PDF files with references for headings, order and tables. Select relevant multi-page files for those capabilities; do not infer formula annotations or complete element coverage from inclusion in the release. |
| Native DOCX | **MIT OpenCourseWare editable course materials**, assembled and reviewed by EduMind | These are genuine teaching documents in the format the parser receives. [Western Music](https://ocw.mit.edu/courses/21m-011-introduction-to-western-music-fall-2024/download/) publishes some handouts in both DOCX and PDF; [Engineering Technical Communication](https://ocw.mit.edu/courses/res-16-9960-the-engineering-students-guide-to-effective-technical-communication-spring-2026/download/) provides additional editable materials. Review the Word structure itself. A companion PDF helps a reviewer read the intended content; it does not define Word pagination or prove an equation is native. |

For each split, count independent source documents with verified tables, formulas
and heading hierarchies. Fifty tables in one textbook provide useful examples
but evidence about only one textbook's layout. Sample DOCX across courses and
templates, not just across filenames.

## Audio extraction

| Condition | Selected source | Why it is in the benchmark |
|---|---|---|
| Technical lectures | [M³AV v1.0](https://github.com/Jack-ZC8/M3AV-dataset) | Tests lecture speech and specialist vocabulary. Use upstream development/test clips and verify words and timing by listening. The training transcripts use Whisper; even the development/test references combine Whisper and Microsoft STT, so they are not an independent oracle for a Whisper candidate. |
| Accented conversational English | [EdAcc](https://datashare.ed.ac.uk/items/355c07b4-500d-4e80-8f12-225e646293c9) | Natural conversations cover accent and speaking-style variation that a read-speech control does not. |
| Far-field and multi-speaker speech | [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/) | Meeting recordings and transcripts allow selection of clips with multiple speakers and distant microphones. Review the actual channel and clip before assigning either label. |
| Read-speech control | [LibriSpeech](https://www.openslr.org/12/) | Its segmented, transcribed read speech provides an interpretable control alongside the less predictable lectures and conversations. |
| Nonspeech controls | [MUSAN](https://www.openslr.org/17/) and generated digital silence | Speech-free noise and instrumental music test whether a recognizer invents speech; generated silence tests the same behavior without background sound. These controls do not enter speech WER or CER. |

Clips cut from one recording must not make a speaker or lecture appear in
multiple splits. A corpus label is not a substitute for listening to a clip.

## Video extraction

| Condition | Selected source | Why it is in the benchmark |
|---|---|---|
| Slides and presenter-plus-slides | [SlideSpeech](https://www.openslr.org/resources/144/about.html) | Its upstream development/test packages provide continuous video for 46 talks and separate annotation files. This covers slide transitions without downloading the training-video archive. |
| Blackboard and digital board | [AVLectures](https://github.com/Darshansingh11/AVLectures) | Course-level archives include original videos, subtitles and timestamped OCR. Board writing changes gradually; it tests policies that might succeed on abrupt slide changes but miss a line as it develops. |
| Screen sharing and code | **EduMind-owned screencasts** | A code edit, notebook output or short-lived UI message can change the answer without a scene cut. Record these cases directly instead of treating another slide deck as equivalent. |

For every source, reviewers verify each
scored line's visibility interval, including disappearance and reappearance.
Upstream OCR and subtitles help find events; neither is the reference for timed
visibility. Multiple excerpts from one lecture are not independent lectures.

## Chunking–embedding, retrieval–reranking and generation

Use one reviewed source corpus so a change of benchmark stage does not silently
change the domain or what counts as evidence.

| Selected source | What it contributes | What EduMind must supply |
|---|---|---|
| [EduQG](https://biblio.ugent.be/publication/01GX2YQ797G48JHY74PF38YCTN) | Expert-written textbook questions with sentence-level evidence make this a closer match to a learner's question than research papers alone. | Select questions that still have a clear answer without multiple-choice options. Keep one version of each question, exclude answer keys from searchable text and map evidence to the frozen textbook text. Split by source book family. |
| [QASPER](https://aclanthology.org/2021.naacl-main.365/) | Information-seeking questions, answers and supporting evidence anchored in whole scientific papers; includes unanswerable questions. | Preserve paper families and verify evidence spans. Recheck unanswerability against the permitted sources. Use answerable questions for retrieval quality; retain unanswerable questions for refusal evaluation. |
| **EduMind-authored questions on [OHR-Bench documents](https://github.com/opendatalab/OHR-Bench)** | Tests retrieval of a table with its headers, a complete formula, or genuinely mixed evidence. Prose questions cannot establish these capabilities. | Write and verify the questions, required facts and exact source intervals. Use the source documents, not OHR's published QA as human-authored gold: its [authors describe generating questions with GPT-4o](https://arxiv.org/html/2412.02592v3). |

Review genuinely cross-document questions separately: QASPER and EduQG source
links do not by themselves establish that two documents are necessary.
Cover table, formula and mixed evidence separately; extra prose questions
cannot make up for a missing evidence type. Generation adds reviewed answers and required factual claims to the
shared evidence. A separate, human-labeled **development-only calibration set**
qualifies the judge: a claim supported elsewhere in the context, but not by the
cited passage, must be distinguishable from a correctly cited claim. Calibration
outputs are not candidate test results.

## Vector database and Final RAG

**Vector database:** group reviewed files, queries and selected-model vectors
into realistic isolated study workspaces of different chunk counts. Evaluate
all-session search and credible user-selected subsets. Exact neighbors are
computed within each scope. Synthetic probes exercise integrity or stress in
isolation; they do not stand in for these workspaces or determine the winner.

**Final RAG:** reserve separate licensed educational document families and
questions before component selection. Evaluate the selected system once on this
locked holdout using canonical documents and live retrieval. Supplying gold
passages would remove the retrieval failures this test is meant to reveal.
Prefer privately held, authorized material; if public material is used, report
it as held out from EduMind's decisions, not necessarily unseen by pretrained
models. The holdout is a collection to prepare, not a claimed download.

## Why the list stops here

- [MPDocBench](https://github.com/Tongyi-Zhiwen/Qwen-Doc/tree/main/MPDocBench)
  has valuable multi-page labels, but part of its preparation works with page
  images. OHR-Bench and OpenDataLoader already supply PDF inputs for the chosen
  tasks without reconstructing files. Layout-only or table-only datasets would
  not replace the broader references needed here.
- [LectureMath](https://github.com/kdavila/lecturemath/blob/master/ACCESS2021_release/README.md)
  labels board keyframes, segments and connected symbols. It would still need
  line-level transcription and timing review; AVLectures already covers the
  board condition.
- [olmOCR-Bench](https://huggingface.co/datasets/allenai/olmOCR-bench) is a
  useful document-parser unit-test suite, but its property/unit-test protocol
  does not supply the complete element references required by EduMind's current
  metric contract.
- [DocLayNet](https://github.com/DS4SD/DocLayNet) is a strong layout-only corpus,
  but OmniDocBench and PureDocBench already provide layout together with text,
  table, formula, and reading-order evidence for this comparison.
- [PubTabNet](https://github.com/ibm-aur-nlp/PubTabNet) is table-specific and is
  unnecessary when the question is complete document-parser quality.
- [IAM Handwriting (NIST catalog entry)](https://tsapps.nist.gov/BDbC/Search/Details/555)
  and [HierText](https://github.com/google-research-datasets/hiertext) are useful
  text-recognition corpora, but the current benchmark asks for complete document
  parsing. OmniDocBench already supplies handwritten-note samples together with
  layout, order, table, and formula annotations.
- [TED-LIUM 3](https://www.openslr.org/51/) is a strong ASR corpus, but its
  prepared-talk role is covered more directly by M³AV's academic lectures, while
  EdAcc supplies the missing accent and natural-conversation coverage. Keeping
  all three would add source conversion work without adding another required
  condition.
- [L2-ARCTIC](https://psi.engr.tamu.edu/l2-arctic-corpus/) provides carefully
  aligned non-native read speech, but it overlaps LibriSpeech's read-speech
  control and EdAcc's accent-diversity role.
- M³AV is used for audio, not for the keyframe-policy benchmark. Its official
  packaged release provides audio, OCR images, and annotations, but not the
  stable continuous-video assets required to compare frame-selection policies.
- [How2](https://github.com/srvk/how2-dataset) is not used because its official
  repository warns that most source videos have disappeared and the downloader
  no longer produces a useful corpus.

See [dataset preparation](dataset-preparation.md) for the target counts, pinned
releases, split assignments and annotation instructions.
