# Extraction dataset manifests

[Project overview](../../../README.md) · [Documentation map](../../README.md) ·
[Preparation guide](../../setup/installation.md) ·
[Extraction methodology](../methodology.md#1-document-extraction)

Only the tiny smoke manifest is committed. Standard/full source assets are intentionally not fabricated or silently downloaded: they require license review and verified references.

Each stage has `<stage>-development.json`, `<stage>-validation.json`, and
`<stage>-locked-test.json` manifests and follows the common fields in
`smoke.json`. Every media sample must additionally contain:

- `asset_sha256`, `source_path`, `source_license`, and `source_revision`;
- a verified normalized `reference`;
- `document_family` so preparation can prove family-level split isolation;
- modality annotations used by that stage, such as `reference_pages`, timestamps, `duration_seconds`, visible text, or PDF layout/oracle labels.

Audio also has a fixed `audio-reliability.json` manifest. Its rows are verified
nonspeech controls—silence, music without lyrics, background noise, or
environmental sound—with an empty spoken reference and a `nonspeech_kind` label.
They are not mixed into Corpus WER or CER; they are used only for Nonspeech
False-Transcription Rate.

Download a reviewed asset plan explicitly:

```powershell
python experiments/benchmarks/prepare.py assets --plan PLAN.json --output data/benchmarks/raw
```

An asset-plan entry requires an HTTPS URL, destination filename, exact SHA-256,
and license. The downloaded raw directory is ignored by Git. Build the frozen
manifests only after checking the reference text and annotations; the runner
rejects missing assets, absent checksums, and checksum mismatches. Dataset counts
and required modality coverage are defined once in the
[methodology](../methodology.md).
