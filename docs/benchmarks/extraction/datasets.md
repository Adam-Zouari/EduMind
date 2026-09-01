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
- one canonical audio `condition` label for authoritative ASR clips: `clean`,
  `noisy`, `accented`, or `multi_speaker`;
- modality annotations used by that stage, such as `reference_pages`, timestamps, `duration_seconds`, visible text, or PDF layout/oracle labels.

Audio also has a fixed `audio-reliability.json` manifest. Its rows are verified
nonspeech controls—silence, music without lyrics, background noise, or
environmental sound—with an empty spoken reference and a `nonspeech_kind` label.
They are not mixed into Corpus WER or CER; they are used only for Nonspeech
False-Transcription Rate.
The same reliability asset checksum cannot appear in more than one of the
development, validation, or locked-test subsets.

The committed smoke path uses `audio-reliability-smoke.json` with deterministic
silence and noise. Authoritative speech rows are limited to 30 seconds and must
contain non-empty timed reference segments; development, validation, and locked
test contain 54, 18, and 18 speech clips respectively. All three authoritative
speech manifests must exist before any authoritative audio run. The runner rejects
reused sample IDs, asset checksums, or speaker/document families across them.

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
