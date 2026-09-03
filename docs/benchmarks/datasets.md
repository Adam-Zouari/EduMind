# Benchmark datasets

[Project overview](../../README.md) · [Documentation map](../README.md) ·
[Installation guide](../setup/installation.md) ·
[Benchmark methodology](methodology.md)

This guide defines the public source pools for the document, audio, and video
benchmarks, how to download them, and how to turn them into EduMind's frozen
manifests. Sources and links were reviewed on **2026-09-02**.

Downloading a public release does not make it a runnable EduMind dataset. A
public release is a **source pool**. An authoritative run uses a smaller,
manually reviewed manifest containing exact sample IDs, local paths, checksums,
licenses, references, and source-family-isolated splits.

## 1. Final dataset plan

### Document extraction

| Source | Required role | What it contributes | Download size |
|---|---|---|---:|
| [OmniDocBench v1.6](https://huggingface.co/datasets/opendatalab/OmniDocBench/tree/d386947f7fc3bafdcd756c8485845a2f43a19875) | Main structured-page source | Text, element boxes and types, reading order, tables, formulas, and page attributes | About 1.55 GB |
| [OHR-Bench v2](https://huggingface.co/datasets/opendatalab/OHR-Bench/tree/7f833e3eda9a571a9ea545a8f6d476fa1685033d) | Real multi-page PDF source | Original PDFs, page-level structured references, varied document domains, and page attribution | About 1.84 GB for the complete repository |
| [PureDocBench v1.0](https://huggingface.co/datasets/zhihengli-casia/puredocbench/tree/dbc6d20b49c7feba5aa43ba7a191dd56374943b6) | Controlled image-robustness source | The same pages rendered as clean, digitally degraded, and real-degraded images, with text, table, formula, and reading-order references | About 37.6 GB |
| [DocPTBench](https://huggingface.co/datasets/topdu/DocPTBench/tree/bb88a1e78588c7ab9ee7fc0dd0dc86cc7e40546c) | Photographed-document source | Matched original and photographed documents with perspective, lighting, blur, folds, and camera artifacts | About 16.9 GB |
| EduMind native-DOCX set | Native DOCX behavior | Paragraphs, headings, lists, captions, images, tables, formulas, and document order in actual `.docx` files | Depends on the selected files |

These sources answer different questions and are not interchangeable:

- OmniDocBench supplies the richest element-level references for EduMind's
  layout, table, and formula metrics. The project deliberately pins v1.6 because
  its evaluator is pinned to the matching v1.6 revision. Do not download the
  moving `main` revision for an authoritative run.
- OHR-Bench supplies real, complete, multi-page PDFs. It is used for PDF text,
  page-content, page-attribution, and whole-document behavior; it does not
  replace OmniDocBench's element-level annotations.
- PureDocBench provides matched clean/degraded views. This makes degradation
  robustness measurable without confusing harder source content with worse
  image quality.
- DocPTBench supplies the real phone-camera conditions that the other sources
  do not isolate. Use its matched original and photographed inputs, but not its
  unwarped outputs: unwarping is a separate preprocessing question and is not a
  candidate in the current experiment.
- A native DOCX set remains necessary. Image/PDF benchmarks cannot demonstrate
  how Docling handles Word headings, lists, captions, relationships, and native
  document order.

The intended authoritative corpus remains:

| Input | Development | Validation | Locked test | Total |
|---|---:|---:|---:|---:|
| Image pages | 72 | 24 | 24 | 120 |
| PDF documents | 36 | 12 | 12 | 60 |
| Native DOCX documents | 27 | 9 | 9 | 45 |

### Audio extraction

| Source | Required role | What it contributes | Download size |
|---|---|---|---:|
| [LibriSpeech test-clean and test-other](https://www.openslr.org/12/) | Read-speech control | Carefully segmented English read speech with speaker IDs and verified transcripts | About 674 MB combined |
| [M³AV v1.0](https://huggingface.co/datasets/CHHHH/M3AV_v1.0/tree/fd0ef99ec1fcf5a4ab946317e32923209b802389) | Academic-lecture speech | Technical lectures in computer science, mathematics, medicine, and biology with transcripts and word timestamps | About 45.5 GB |
| [EdAcc v1.0](https://datashare.ed.ac.uk/items/355c07b4-500d-4e80-8f12-225e646293c9/full) | Accent-diversity source | Natural remote conversations across many first- and second-language English varieties | About 5.51 GB |
| [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/) | Noisy and multi-speaker speech | Far-field and mixed meeting audio with manual orthographic transcription and timed words | Variable; only selected meetings need to be downloaded |
| [MUSAN](https://www.openslr.org/17/) | Nonspeech reliability controls | Music and environmental/background noise | About 11 GB |
| Generated silence | Nonspeech reliability control | Deterministic silence with no licensing or transcription ambiguity | Negligible |

The recommended 90-clip allocation is deliberately simple:

| Source | Development | Validation | Locked test | Total |
|---|---:|---:|---:|---:|
| LibriSpeech | 10 | 4 | 4 | 18 |
| M³AV | 14 | 5 | 5 | 24 |
| EdAcc | 14 | 5 | 5 | 24 |
| AMI | 16 | 4 | 4 | 24 |
| **Total speech clips** | **54** | **18** | **18** | **90** |

The allocation does not assign conditions automatically. A reviewer listens to
every selected clip and assigns exactly one of `clean`, `noisy`, `accented`, or
`multi_speaker`. Each split must contain all four conditions. Corpus names such
as `test-other` are not substitutes for listening and labeling.

MUSAN and generated silence are stored in a separate reliability manifest. They
never enter Corpus WER or CER.

### Video extraction

Use three complementary sources rather than forcing one lecture corpus to cover
every presentation style:

| Source | Development | Validation | Locked test | Total | What it contributes |
|---|---:|---:|---:|---:|---|
| [SlideSpeech](https://www.openslr.org/144/) | 8 | 2 | 2 | 12 | Slide-based lectures and presenter-plus-slide layouts |
| [AVLectures](https://github.com/Darshansingh11/AVLectures) | 5 | 2 | 2 | Blackboard, digital-board, and mixed lecture modes |
| EduMind-owned recordings | 5 | 2 | 2 | Screen sharing, code editors, notebooks, and gradual UI/text changes |
| **Total** | **18** | **6** | **6** | **30** | |

Public subtitles and OCR are **annotation seeds**, not EduMind ground truth. A
reviewer must watch every selected interval and correct its transcript, visible
text, appearance timestamps, duplicates, and visual-change boundaries. The
same verification rules apply to SlideSpeech, AVLectures, and owned recordings.

## 2. Prerequisites and storage layout

Activate the project virtual environment and install the benchmark dependencies
before using the Hugging Face CLI:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements/benchmarks.lock
hf --help
ffmpeg -version
```

Install FFmpeg from the [official FFmpeg download page](https://ffmpeg.org/download.html)
if `ffmpeg` is unavailable. Git Bash is useful for the split PureDocBench and
M³AV archives and the SlideSpeech fallback downloader.

Keep raw public releases under the ignored `data/benchmarks/raw` directory:

```text
data/benchmarks/raw/
├── document/
│   ├── omnidocbench-v1.6/
│   ├── ohr-bench-v2/
│   ├── puredocbench-v1.0/
│   ├── docptbench/
│   └── native-docx/
├── audio/
│   ├── librispeech/
│   ├── m3av-v1.0/
│   ├── edacc-v1.0/
│   ├── ami/
│   └── musan/
└── video/
    ├── slidespeech/
    ├── avlectures/
    └── edumind-owned/
```

Create the folders once:

```powershell
New-Item -ItemType Directory -Force `
  data/benchmarks/raw/document, `
  data/benchmarks/raw/audio, `
  data/benchmarks/raw/video | Out-Null
```

Rerunning `hf download` or `curl.exe -C -` resumes completed or partial files.
Stopping either command does not discard already downloaded bytes.

## 3. Install the document datasets

### 3.1 OmniDocBench v1.6

Official evidence:

- [Dataset at the pinned v1.6 revision](https://huggingface.co/datasets/opendatalab/OmniDocBench/tree/d386947f7fc3bafdcd756c8485845a2f43a19875)
- [Official evaluator and dataset description](https://github.com/opendatalab/OmniDocBench)
- [CVPR 2025 paper](https://openaccess.thecvf.com/content/CVPR2025/html/Ouyang_OmniDocBench_Benchmarking_Diverse_PDF_Document_Parsing_with_Comprehensive_Annotations_CVPR_2025_paper.html)

The pinned release contains 1,651 annotated pages across ten document types. It
provides 28 block-level and four span-level element categories, recognition text,
formula LaTeX, table HTML/LaTeX, element locations, and reading order.

Download the exact dataset revision:

```powershell
hf download opendatalab/OmniDocBench `
  --repo-type dataset `
  --revision d386947f7fc3bafdcd756c8485845a2f43a19875 `
  --local-dir data/benchmarks/raw/document/omnidocbench-v1.6
```

The repository code is Apache-2.0, but the official dataset statement limits the
collected PDFs to research, non-commercial use. Record that dataset restriction
in every derived manifest instead of copying the code license onto the data.

Use English pages only for the current English-first release. Select pages by
document family, not randomly by image, so pages from one source document never
cross development, validation, and locked-test boundaries.

### 3.2 OHR-Bench v2

Official evidence:

- [Pinned dataset revision](https://huggingface.co/datasets/opendatalab/OHR-Bench/tree/7f833e3eda9a571a9ea545a8f6d476fa1685033d)
- [Official repository](https://github.com/opendatalab/OHR-Bench)
- [OHR-Bench paper](https://arxiv.org/abs/2412.02592)

OHR-Bench contains more than 8,500 pages from textbooks, law, finance,
newspapers, manuals, academic documents, and administration. Its pages have
human-verified structured ground truth. The dataset is CC BY 4.0.

Download only the v2 annotations and original PDFs:

```powershell
hf download opendatalab/OHR-Bench `
  OHR-Bench_v2.parquet pdfs.zip `
  --repo-type dataset `
  --revision 7f833e3eda9a571a9ea545a8f6d476fa1685033d `
  --local-dir data/benchmarks/raw/document/ohr-bench-v2

Expand-Archive `
  data/benchmarks/raw/document/ohr-bench-v2/pdfs.zip `
  -DestinationPath data/benchmarks/raw/document/ohr-bench-v2/pdfs `
  -Force
```

Do not use OHR-Bench's synthetic OCR-error variants as extractor references.
EduMind evaluates real extractor output against the human-verified ground truth.

### 3.3 PureDocBench v1.0

Official evidence:

- [Pinned dataset revision](https://huggingface.co/datasets/zhihengli-casia/puredocbench/tree/dbc6d20b49c7feba5aa43ba7a191dd56374943b6)
- [Official repository](https://github.com/zhihengli-casia/PureDocBench)
- [Paper](https://arxiv.org/abs/2605.07492)

PureDocBench has 1,475 source-traceable pages and three aligned image tracks per
page: clean, digitally degraded, and real-degraded. It covers ten domains and 66
subcategories and is released under CC BY 4.0.

Download the exact reviewed snapshot:

```powershell
hf download zhihengli-casia/puredocbench `
  --repo-type dataset `
  --revision dbc6d20b49c7feba5aa43ba7a191dd56374943b6 `
  --local-dir data/benchmarks/raw/document/puredocbench-v1.0
```

Then open Git Bash in the downloaded directory and verify and extract the split
archive without creating a second 37 GB tar file:

```bash
cd data/benchmarks/raw/document/puredocbench-v1.0
sha256sum -c SHA256SUMS.txt
cat pdb_full.tar.part-* | tar -xf -
```

Keep all three variants of one page in the same split. They may be compared as a
matched robustness group, but they must never become independent samples split
across development, validation, and locked test.

### 3.4 DocPTBench photographed documents

Official evidence:

- [Pinned dataset revision](https://huggingface.co/datasets/topdu/DocPTBench/tree/bb88a1e78588c7ab9ee7fc0dd0dc86cc7e40546c)
- [Official repository](https://github.com/Topdu/DocPTBench)
- [Paper](https://arxiv.org/abs/2511.18434)

DocPTBench contains matched original, photographed, and unwarped document
images. The photographed track introduces perspective distortion, shadows,
motion blur, folds, lighting variation, noise, and camera artifacts. These are
real capture conditions, not another copy of the clean-page test.

Download the reviewed snapshot:

```powershell
hf download topdu/DocPTBench `
  --repo-type dataset `
  --revision bb88a1e78588c7ab9ee7fc0dd0dc86cc7e40546c `
  --local-dir data/benchmarks/raw/document/docptbench
```

Select English samples and keep each original/photographed pair in one split.
Run the original and photographed inputs so the loss caused by camera capture
can be measured on matched content. Do not run the supplied unwarped images in
this benchmark version: doing so would introduce an unwarping/preprocessing
candidate that is not part of the approved document-parser matrix.

DocPTBench replaces part of the existing 120-image allocation; it does not
increase the corpus size. Check the dataset card and the source record of every
selected item before redistribution rather than assuming the repository's code
license applies to every underlying document.

### 3.5 Build the native-DOCX set

There is no established public benchmark that simultaneously provides native
DOCX files and EduMind's required verified text, hierarchy, list, table, formula,
caption, image, and order references. Reconstructed DOCX files alone would test
the reconstruction procedure, not ordinary author-created documents.

Create 45 licensed documents under:

```text
data/benchmarks/raw/document/native-docx/
├── files/
└── references/
```

Use documents that you own or that have an explicit redistribution-compatible
license. For each document:

1. preserve the original `.docx` unchanged;
2. record its source URL, license, acquisition date, and SHA-256;
3. inspect the document in Word or LibreOffice;
4. create the ordered reference text and element list manually;
5. mark heading levels, list items, captions, images, tables, and formulas that
   are actually present;
6. have a second review pass resolve any disagreement; and
7. assign the complete document to one split by source family or author.

The set should include simple prose, nested headings, ordered and unordered
lists, captions with embedded images, native tables, native equations, headers
and footers, and mixed-content reports. Do not convert PDFs into DOCX to fill the
set.

## 4. Install the audio datasets

### 4.1 LibriSpeech controls

[OpenSLR SLR12](https://www.openslr.org/12/) publishes both archives under CC BY
4.0.

```powershell
New-Item -ItemType Directory -Force `
  data/benchmarks/raw/audio/librispeech | Out-Null

curl.exe -L -C - --retry 5 `
  -o data/benchmarks/raw/audio/librispeech/test-clean.tar.gz `
  https://www.openslr.org/resources/12/test-clean.tar.gz

curl.exe -L -C - --retry 5 `
  -o data/benchmarks/raw/audio/librispeech/test-other.tar.gz `
  https://www.openslr.org/resources/12/test-other.tar.gz

curl.exe -L --retry 5 `
  -o data/benchmarks/raw/audio/librispeech/md5sum.txt `
  https://www.openslr.org/resources/12/md5sum.txt

tar -xzf data/benchmarks/raw/audio/librispeech/test-clean.tar.gz `
  -C data/benchmarks/raw/audio/librispeech
tar -xzf data/benchmarks/raw/audio/librispeech/test-other.tar.gz `
  -C data/benchmarks/raw/audio/librispeech
```

Compare each archive's value from `Get-FileHash -Algorithm MD5` with the official
`md5sum.txt`. Preserve `reader_id` as the split-family field. To obtain several
timed segments in a representative clip, concatenate two to four consecutive
utterances from the same reader and chapter, retain every utterance boundary,
and keep the resulting clip at or below 30 seconds.

### 4.2 M³AV academic lectures

Official evidence:

- [Official project](https://jack-zc8.github.io/M3AV-dataset-page/)
- [Official repository and download instructions](https://github.com/Jack-ZC8/M3AV-dataset)
- [Pinned dataset revision](https://huggingface.co/datasets/CHHHH/M3AV_v1.0/tree/fd0ef99ec1fcf5a4ab946317e32923209b802389)
- [ACL 2024 paper](https://aclanthology.org/2024.acl-long.489/)

M³AV contributes speech from academic presentations in computer science,
mathematics, medicine, and biology. Its speech data includes written/spoken
forms and word timestamps. The complete release is CC BY-NC-SA 4.0 and the
underlying source-video copyright remains with the original owners.

Download the exact reviewed release. Both the audio archive and the smaller
`noaudio` archive are needed because the latter contains the split and annotation
metadata:

```powershell
hf download CHHHH/M3AV_v1.0 `
  --repo-type dataset `
  --revision fd0ef99ec1fcf5a4ab946317e32923209b802389 `
  --local-dir data/benchmarks/raw/audio/m3av-v1.0
```

Verify the files against the checksums in the
[official download directory](https://github.com/Jack-ZC8/M3AV-dataset/tree/main/download).
Open Git Bash and stream the split archive directly into `tar` so no second
43 GB archive is created:

```bash
cd data/benchmarks/raw/audio/m3av-v1.0
mkdir -p audio metadata
cat dataset_v1.0_onlyaudio_tar_gz/dataset_v1.0_onlyaudio.tar.gz.* | \
  tar -xzf - -C audio
tar -xzf dataset_v1.0_noaudio.tar.gz -C metadata
```

Use samples from the official development/test portions, then manually verify
the selected transcript and boundaries against the exact audio. Treat the source
lecture and speaker as the split family, and cut only complete reference segments
into clips no longer than 30 seconds.

### 4.3 EdAcc accent diversity

[EdAcc v1.0](https://datashare.ed.ac.uk/items/355c07b4-500d-4e80-8f12-225e646293c9/full)
contains almost 40 hours of dyadic remote conversations from speakers covering
many first- and second-language varieties of English. It includes speaker
linguistic profiles and sentence-level STM evaluation data and is released under
CC BY-SA 4.0.

Download the official 5.51 GB archive:

```powershell
New-Item -ItemType Directory -Force `
  data/benchmarks/raw/audio/edacc-v1.0 | Out-Null

curl.exe -L -C - --retry 5 `
  -o data/benchmarks/raw/audio/edacc-v1.0/edacc_v1.0.tar.gz `
  https://datashare.ed.ac.uk/bitstreams/819f726e-1a65-4b3c-88d2-efdf0a7021ce/download

Get-FileHash -Algorithm MD5 `
  data/benchmarks/raw/audio/edacc-v1.0/edacc_v1.0.tar.gz

tar -xzf data/benchmarks/raw/audio/edacc-v1.0/edacc_v1.0.tar.gz `
  -C data/benchmarks/raw/audio/edacc-v1.0
```

The expected archive MD5 is:

```text
146b4b8026b5d0ce9611667c708456b3
```

Select speakers across different recorded English varieties. Keep both sides of
one conversation and every clip from one speaker in a single EduMind split.
Manually verify the selected 30-second-or-shorter segments and their timestamps;
self-reported accent metadata describes coverage but is not itself a quality
label.

### 4.4 AMI Meeting Corpus

AMI provides about 100 hours of multimodal meeting recordings, manual
orthographic transcripts, and timed word annotations under CC BY 4.0. Use the
[official download chooser](https://groups.inf.ed.ac.uk/ami/download/) and the
[official ASR partitions](https://groups.inf.ed.ac.uk/ami/corpus/datasets.shtml).

Download the manual annotations first:

```powershell
New-Item -ItemType Directory -Force `
  data/benchmarks/raw/audio/ami | Out-Null

curl.exe -L -C - --retry 5 `
  -o data/benchmarks/raw/audio/ami/ami_public_manual_1.6.2.zip `
  https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip

Expand-Archive `
  data/benchmarks/raw/audio/ami/ami_public_manual_1.6.2.zip `
  -DestinationPath data/benchmarks/raw/audio/ami/annotations `
  -Force
```

In the chooser:

1. select meeting series from the official full-corpus ASR partition;
2. download the headset mix for a mixed-speaker control and microphone-array
   audio for far-field/noisy clips;
3. store the generated download commands with the dataset provenance;
4. treat the meeting series, such as `ES2008`, rather than only session suffix
   `a`, `b`, `c`, or `d`, as the source family; and
5. never place two recordings of the same spoken interval or meeting family in
   different EduMind splits.

Group the manual timed words into meaningful utterance segments. Select clips no
longer than 30 seconds and preserve overlapping speech when labeling a sample
`multi_speaker`.

### 4.5 MUSAN and deterministic silence

MUSAN is CC BY 4.0 and is used only for the nonspeech false-transcription test.

```powershell
New-Item -ItemType Directory -Force `
  data/benchmarks/raw/audio/musan | Out-Null

curl.exe -L -C - --retry 5 `
  -o data/benchmarks/raw/audio/musan/musan.tar.gz `
  https://www.openslr.org/resources/17/musan.tar.gz

tar -xzf data/benchmarks/raw/audio/musan/musan.tar.gz `
  -C data/benchmarks/raw/audio/musan
```

Choose verified music without lyrics, background noise, and environmental
sounds. Do not use MUSAN's speech directory as a nonspeech control. Generate a
deterministic silence file with FFmpeg:

```powershell
ffmpeg -f lavfi -i anullsrc=r=16000:cl=mono -t 10 `
  -c:a pcm_s16le `
  data/benchmarks/raw/audio/silence-10s.wav
```

The authoritative reliability manifest must contain silence, music without
lyrics, background noise, and environmental sound in every split. Give each
selected source file an empty spoken reference and one `nonspeech_kind` label.

## 5. Install the video datasets

### 5.1 Preferred SlideSpeech archives

Official sources:

- [OpenSLR SLR144 release](https://www.openslr.org/144/)
- [SlideSpeech project page](https://slidespeech.github.io/)
- [Official download scripts linked by the project](https://github.com/Mashiro009/slidespeech_dl)
- [ICASSP paper](https://arxiv.org/abs/2309.05396)

Download dev video, test video, and related annotations only:

```powershell
New-Item -ItemType Directory -Force `
  data/benchmarks/raw/video/slidespeech | Out-Null

curl.exe -L -C - --retry 5 `
  -o data/benchmarks/raw/video/slidespeech/dev_video.tar.gz `
  https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/SlideSpeech/dev_video.tar.gz

curl.exe -L -C - --retry 5 `
  -o data/benchmarks/raw/video/slidespeech/test_video.tar.gz `
  https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/SlideSpeech/test_video.tar.gz

curl.exe -L -C - --retry 5 `
  -o data/benchmarks/raw/video/slidespeech/related_files.tar.gz `
  https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/SlideSpeech/related_files.tar.gz
```

Verify the OpenSLR-published MD5 values:

| File | MD5 |
|---|---|
| `dev_video.tar.gz` | `779cb23bd41c697a5740b816051038fd` |
| `test_video.tar.gz` | `5211d0a6099028ce97408f8199240ceb` |
| `related_files.tar.gz` | `bfa834a7a02ba3b13e5c3f6fed82c102` |

```powershell
Get-FileHash -Algorithm MD5 data/benchmarks/raw/video/slidespeech/dev_video.tar.gz
Get-FileHash -Algorithm MD5 data/benchmarks/raw/video/slidespeech/test_video.tar.gz
Get-FileHash -Algorithm MD5 data/benchmarks/raw/video/slidespeech/related_files.tar.gz

tar -xzf data/benchmarks/raw/video/slidespeech/dev_video.tar.gz `
  -C data/benchmarks/raw/video/slidespeech
tar -xzf data/benchmarks/raw/video/slidespeech/test_video.tar.gz `
  -C data/benchmarks/raw/video/slidespeech
tar -xzf data/benchmarks/raw/video/slidespeech/related_files.tar.gz `
  -C data/benchmarks/raw/video/slidespeech
```

At the review date, the official archive host was intermittent and its dev-video
link could return 404. Do not replace it with an unknown mirror. Use the official
project-linked downloader below if an archive is unavailable.

### 5.2 Official YouTube-download fallback

Run this section in Git Bash. YouTube availability changes, so the final
benchmark must freeze only successfully downloaded video IDs and checksums.

```bash
git clone https://github.com/Mashiro009/slidespeech_dl.git \
  data/benchmarks/raw/video/slidespeech-downloader
cd data/benchmarks/raw/video/slidespeech-downloader
python -m pip install yt-dlp

curl -L -C - --retry 5 \
  -o wavid2channel.tar.gz \
  https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/SlideSpeech/wavid2channel.tar.gz
tar -xzf wavid2channel.tar.gz

python local/prepare_download_scripts.py \
  --superpath ../../../../../data/benchmarks/raw/video/slidespeech
bash data/dev/process.sh
bash data/test/process.sh
```

Record the downloader Git commit with `git rev-parse HEAD`. Some source videos
may have been removed or changed; unavailable videos are not silently replaced.

### 5.3 AVLectures

The [official AVLectures repository](https://github.com/Darshansingh11/AVLectures)
describes 2,350 lectures across 86 courses in mathematics, physics, electrical
engineering, computer science, and economics. It explicitly labels four
presentation modes: blackboard, slides, digital board, and mixed. Fifteen
courses also include temporal segmentation data.

Clone the official repository to preserve its documentation and exact revision:

```powershell
git clone https://github.com/Darshansingh11/AVLectures.git `
  data/benchmarks/raw/video/avlectures-repository

git -C data/benchmarks/raw/video/avlectures-repository rev-parse HEAD
```

Use the official repository's course access instructions to obtain only selected
course archives. Prefer courses with segmentation and choose blackboard,
digital-board, and mixed examples rather than adding more slide-only lectures.
Store them under `data/benchmarks/raw/video/avlectures/`.

AVLectures supplies `.srt` subtitles and automatic Google Cloud OCR sampled at
10 frames per second. Both are annotation seeds. They are not sufficiently
authoritative for EduMind's transcript, visible-text, or timestamp metrics until
the selected intervals have been manually checked. If an official course archive
is unavailable, record that fact and replace it with an owned recording of the
same presentation mode; do not use an unverified mirror.

### 5.4 EduMind-owned screen recordings

Create nine short English educational recordings that EduMind is allowed to keep
and evaluate. The set should include:

- a code editor or terminal where text changes gradually;
- a notebook or interactive lesson;
- a screen-shared presentation with incremental bullet reveals;
- scrolling or switching between windows;
- repeated views of the same text; and
- at least one dense technical screen.

Store the original files under:

```text
data/benchmarks/raw/video/edumind-owned/
├── files/
└── references/
```

Record the creator, recording date, license/permission, source resolution, frame
rate, and checksum. Do not create the reference by running one of the candidate
extractors; transcribe and time the visible and spoken content manually.

### 5.5 Build verified video references

The SlideSpeech license applies to its released metadata, while its underlying
videos retain their original owners' rights and terms. AVLectures course assets
must likewise be checked at the course/source level. EduMind-owned clips need
explicit permission from their creator. Review the source and license of every
selected video before retaining or redistributing a clip.

For each of the 30 selected videos:

1. choose a stable educational interval containing multiple meaningful visual
   changes;
2. preserve the original video ID, source URL, interval, and original checksum;
3. create one canonical local clip and checksum that exact clip;
4. verify the spoken transcript against the audio;
5. use the supplied OCR only as a starting point, then manually transcribe all
   educationally useful visible text;
6. record when each distinct visible-text segment first becomes available and
   when it disappears;
7. mark repeated text so duplicate extraction can be scored correctly; and
8. assign the entire source video to one split.

A practical review interval is two to five minutes: long enough to contain slide
changes and repeated scenes, but short enough for the nine development
keyframe configurations to run repeatedly. This is a selection recommendation,
not a hidden evaluator cutoff; the exact interval and duration are stored in
the manifest.

## 6. Build the frozen EduMind manifests

Create these files under `data/benchmarks/extraction`:

```text
document-development.json
document-validation.json
document-locked-test.json
audio-development.json
audio-validation.json
audio-locked-test.json
audio-reliability.json
video-development.json
video-validation.json
video-locked-test.json
```

Every sample requires:

- a stable `id`, `source_path`, and SHA-256 of the exact local asset;
- source dataset, source sample ID, source URL, exact revision/release, and
  license;
- `document_family` identifying the source document, reader/chapter, talk,
  meeting series, or source video used for split isolation;
- the frozen split and preprocessing version; and
- only human-verified reference fields, never an unreviewed model prediction.

Document samples additionally contain ordered reference text, per-page text,
and the applicable element records: kind, text, order, page, hierarchy, bounding
box, table structure, or formula representation.

Speech samples additionally contain:

- `duration_seconds` in `(0, 30]`;
- one condition: `clean`, `noisy`, `accented`, or `multi_speaker`; and
- non-empty timed `reference_segments`, all inside the clip duration.

Reliability samples contain an empty spoken reference and one of `silence`,
`music_without_lyrics`, `background_noise`, or `environmental_sound`.

Video samples additionally contain the duration, verified transcript and timed
speech segments, distinct visible-content references, the start/end interval of
each visible reference, and annotations identifying intentional reappearance.
The transcript is retained for the frozen-ASR diagnostic, but spoken and visible
tokens are not merged into a combined quality score.

Before freezing a manifest:

1. compute every local asset checksum with `Get-FileHash -Algorithm SHA256`;
2. ensure no asset checksum, sample ID, or source family occurs in two splits;
3. inspect near-duplicates, including clean/degraded variants and alternate
   encodings of the same speech or video;
4. verify every annotation against the exact canonical asset; and
5. record the selection script or reviewed ID list beside the manifests.

The strongest locked test uses privately held, licensed EduMind-specific samples
that were not public during candidate development. If only public sources are
available, the run remains useful but must be described as public-corpus
confirmation rather than proof of unseen real-world generalization.

## 7. Run the prepared benchmarks

Document extraction:

```powershell
python experiments/benchmarks/extraction/document/run.py --profile standard `
  --manifest data/benchmarks/extraction/document-development.json

python experiments/benchmarks/extraction/document/run.py --profile full `
  --manifest data/benchmarks/extraction/document-validation.json `
  --pdf-selection PDF_CONFIG_DECISION `
  --image-selection IMAGE_CONFIG_DECISION
```

Audio extraction:

```powershell
python experiments/benchmarks/extraction/audio/run.py --profile standard `
  --manifest data/benchmarks/extraction/audio-development.json `
  --device cuda

python experiments/benchmarks/extraction/audio/run.py --profile full `
  --manifest data/benchmarks/extraction/audio-validation.json `
  --shortlist AUDIO_DECISION `
  --device cuda
```

Video extraction, after freezing one document parser and one ASR profile:

```powershell
python experiments/benchmarks/extraction/video/run.py --profile standard `
  --manifest data/benchmarks/extraction/video-development.json `
  --document-selection DOCUMENT_DECISION `
  --audio-selection AUDIO_DECISION `
  --device cuda
```

The tiny committed smoke assets remain the only extraction data stored in Git.
Smoke verifies code paths only and does not replace any public or manually
verified dataset above.

## 8. Sources deliberately not required

- [olmOCR-Bench](https://huggingface.co/datasets/allenai/olmOCR-bench) is a
  useful document-parser unit-test suite, but its property/unit-test protocol
  does not supply the complete element references required by EduMind's current
  metric contract.
- [DocLayNet](https://github.com/DS4SD/DocLayNet) is a strong layout-only corpus,
  but OmniDocBench and PureDocBench already provide layout together with text,
  table, formula, and reading-order evidence for this first comparison.
- [PubTabNet](https://github.com/ibm-aur-nlp/PubTabNet) is table-specific and is
  unnecessary when the question is complete document-parser quality.
- [IAM Handwriting](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database)
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

These exclusions keep the download plan tied to a concrete metric or coverage
gap rather than collecting datasets that answer the same question.
