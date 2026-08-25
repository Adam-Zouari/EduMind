# Extraction dataset manifests

[Project overview](../../../README.md) · [Documentation map](../../README.md) ·
[Preparation guide](../../setup/installation.md) ·
[Extraction benchmark documents](../overview.md#recommended-experiment-order)

Only the tiny smoke manifest is committed. Standard/full source assets are intentionally not fabricated or silently downloaded: they require license review and verified references.

Each stage manifest is named `<stage>-validation.json` or `<stage>-locked-test.json` and follows the common fields in `smoke.json`. Every media sample must additionally contain:

- `asset_sha256`, `source_path`, `source_license`, and `source_revision`;
- a verified normalized `reference`;
- `document_family` so preparation can prove family-level split isolation;
- modality annotations used by that stage, such as `reference_pages`, timestamps, `duration_seconds`, visible text, or PDF layout/oracle labels.

Download a reviewed asset plan explicitly:

```powershell
python experiments/benchmarks/prepare.py assets --plan PLAN.json --output data/benchmarks/raw
```

An asset-plan entry requires an HTTPS URL, destination filename, exact SHA-256, and license. The downloaded raw directory is ignored by Git. Build the frozen manifests only after checking the reference text and annotations; the runner rejects missing assets, absent checksums, and checksum mismatches. Dataset counts and required modality coverage are defined in the extraction benchmark pages in this directory.
