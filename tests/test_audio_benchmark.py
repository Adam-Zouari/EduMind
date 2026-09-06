from __future__ import annotations

import pytest
import json
from pathlib import Path
from types import SimpleNamespace

from edumind.extraction.contracts import ExtractionProfile, ExtractionRequest, SourceKind
from edumind.extraction.extractors.audio import WhisperExtractor
from edumind.extraction.extractors.base import build_document
from experiments.benchmarks.common.contracts import BenchmarkPlan, DatasetManifest, SampleResult
from experiments.benchmarks.common.datasets import assert_no_split_leakage
from experiments.benchmarks.common.resources import ResourceMonitor
from experiments.benchmarks.extraction.audio.adapters import (
    ASR_PROFILES,
    QwenRuntime,
    WhisperRuntime,
)
from experiments.benchmarks.extraction.audio.evaluate import (
    METRIC_DIRECTIONS,
    aggregate,
    align_sequences,
    normalize_transcript,
    score_nonspeech,
    score_speech,
)
from experiments.benchmarks.extraction.audio.runner import (
    REQUIRED_SPEECH_CONDITIONS,
    _validate_reliability_split_isolation,
    _validate_manifest_rows,
    _worker_environment,
)
from experiments.benchmarks.common.runner import run_benchmark


def _speech(sample_id: str, reference: str, prediction: str):
    item = {
        "id": sample_id,
        "reference": reference,
        "duration_seconds": 2.0,
        "reference_segments": [{"text": reference, "start": 0.0, "end": 2.0}],
    }
    predicted = [{"text": prediction, "start": 0.1, "end": 1.9}]
    return score_speech(
        item,
        prediction,
        predicted,
        quality_latency_seconds=0.5,
        repeat_transcript_agreement=sample_id != "two",
    )


def _timings(*sample_ids: str):
    return [
        {
            "sample_id": sample_id,
            "repetition": repetition,
            "latency_seconds": 0.4 + repetition / 10,
            "duration_seconds": 2.0,
        }
        for sample_id in sample_ids
        for repetition in (1, 2, 3)
    ]


def test_asr_normalization_is_minimal_and_unicode_aware() -> None:
    assert normalize_transcript("  Straße—TEST!  ") == "strasse test"
    assert normalize_transcript("twenty-one") == "twenty one"
    assert normalize_transcript("42") == "42"


def test_whisper_preserves_untimed_text_and_benchmark_records_it(tmp_path) -> None:
    calls = []

    def fake_runtime(source: str, **options):
        calls.append((source, options))
        return {
            "text": "hello world unfinished",
            "chunks": [
                {"text": "hello", "timestamp": (0.0, 0.4)},
                {"text": "world", "timestamp": (0.5, 1.0)},
                {"text": "unfinished", "timestamp": (1.0, None)},
            ],
        }

    extractor = WhisperExtractor("revision")
    extractor._runtime = fake_runtime
    request = ExtractionRequest(
        Path("audio.wav"),
        "checksum",
        profile=ExtractionProfile("test", "whisper", "revision"),
        options={"model_path": str(tmp_path)},
    )
    document = extractor.extract(request, SourceKind.AUDIO)

    assert document.text == "hello world unfinished"
    assert [
        (segment.timestamp_start, segment.timestamp_end) for segment in document.segments
    ] == [(0.0, 0.4), (0.5, 1.0), (1.0, None)]
    assert [warning.code for warning in document.warnings] == ["incomplete_timestamp"]

    profile = ASR_PROFILES["whisper-small-en-control"]
    runtime = WhisperRuntime(profile, Path("model"), "cpu")
    runtime._runtime = fake_runtime

    transcript = runtime.transcribe(Path("audio.wav"))

    assert transcript.text == "hello world unfinished"
    assert [segment["text"] for segment in transcript.segments] == ["hello", "world"]
    assert transcript.warnings == ("1 Whisper chunk(s) lacked complete timestamp boundaries",)
    assert calls == [
        (
            "audio.wav",
            {
                "return_timestamps": "word",
                "generate_kwargs": {"do_sample": False},
            },
        ),
        (
            "audio.wav",
            {
                "return_timestamps": "word",
                "generate_kwargs": {"do_sample": False},
            },
        )
    ]

    video = build_document(
        request,
        SourceKind.VIDEO,
        request.profile,
        ["hello", "world", "slide text"],
        separators=[" ", "\n"],
    )
    assert video.text == "hello world\nslide text"


def test_word_alignment_returns_known_substitution_deletion_and_insertion_counts() -> None:
    substitution = align_sequences("a b".split(), "a x".split())
    deletion = align_sequences("a b".split(), "a".split())
    insertion = align_sequences("a".split(), "a x".split())
    assert substitution.substitutions == 1
    assert deletion.deletions == 1
    assert insertion.insertions == 1
    assert substitution.exact_matches == ((0, 0),)


def test_asr_aggregation_pools_counts_and_emits_the_exact_contract() -> None:
    samples = [
        _speech("one", "one two", "one too"),
        _speech("two", "three four five", "three five"),
        score_nonspeech(
            {"id": "silence", "duration_seconds": 2.0, "nonspeech_kind": "silence"},
            "",
            latency_seconds=0.2,
        ),
        score_nonspeech(
            {
                "id": "noise",
                "duration_seconds": 2.0,
                "nonspeech_kind": "background_noise",
            },
            "invented text",
            latency_seconds=0.2,
        ),
    ]
    metrics, intervals = aggregate(
        samples,
        _timings("one", "two"),
        cold_model_load_seconds=1.0,
        peak_process_tree_ram_mb=100.0,
        peak_vram_mb=0.0,
        resamples=100,
        seed=42,
    )
    assert set(metrics) == set(METRIC_DIRECTIONS)
    assert metrics["word_error_rate"] == pytest.approx(2 / 5)
    assert metrics["word_substitution_rate"] == pytest.approx(1 / 5)
    assert metrics["word_deletion_rate"] == pytest.approx(1 / 5)
    assert metrics["word_insertion_rate"] == 0.0
    assert metrics["nonspeech_false_transcription_rate"] == 0.5
    assert metrics["repeat_transcript_agreement_rate"] == 0.5
    assert "cold_model_load_seconds" not in intervals
    repeated = aggregate(
        samples,
        _timings("one", "two"),
        cold_model_load_seconds=1.0,
        peak_process_tree_ram_mb=100.0,
        peak_vram_mb=0.0,
        resamples=100,
        seed=42,
    )[1]
    assert intervals == repeated


def test_bootstrap_keeps_draws_without_timestamp_matches_for_defined_metrics() -> None:
    samples = [
        _speech("aligned", "alpha", "alpha"),
        _speech("unaligned", "beta", "wrong"),
        score_nonspeech(
            {"id": "silence", "duration_seconds": 2.0, "nonspeech_kind": "silence"},
            "",
            latency_seconds=0.2,
        ),
    ]

    _, intervals = aggregate(
        samples,
        _timings("aligned", "unaligned"),
        cold_model_load_seconds=1.0,
        peak_process_tree_ram_mb=100.0,
        peak_vram_mb=0.0,
        resamples=1_000,
        seed=42,
    )

    assert intervals["timestamp_alignment_coverage"]["lower"] == 0.0
    assert intervals["word_error_rate"]["upper"] == 1.0
    assert intervals["timestamp_alignment_coverage"]["resamples"] == 1_000
    assert intervals["timestamp_boundary_mae_seconds"]["resamples"] < 1_000


def test_timestamp_alignment_accepts_unequal_segment_counts() -> None:
    row = score_speech(
        {
            "id": "timed",
            "reference": "alpha beta",
            "duration_seconds": 2.0,
            "reference_segments": [
                {"text": "alpha", "start": 0.0, "end": 0.8},
                {"text": "beta", "start": 1.0, "end": 2.0},
            ],
        },
        "alpha extra beta",
        [
            {"text": "alpha", "start": 0.1, "end": 0.7},
            {"text": "extra", "start": 0.7, "end": 1.0},
            {"text": "beta", "start": 1.1, "end": 1.9},
        ],
        quality_latency_seconds=0.5,
        repeat_transcript_agreement=True,
    )
    assert row["aligned_timed_segment_count"] == 2
    assert row["timestamp_boundary_count"] == 4
    assert row["timestamp_boundary_error_seconds"] == pytest.approx(0.4)


def test_missing_timestamps_fail_but_unaligned_mae_is_null() -> None:
    with pytest.raises(ValueError, match="timestamp segments are missing"):
        score_speech(
            {
                "id": "missing",
                "reference": "alpha",
                "duration_seconds": 1.0,
                "reference_segments": [{"text": "alpha", "start": 0.0, "end": 1.0}],
            },
            "alpha",
            [],
            quality_latency_seconds=0.1,
            repeat_transcript_agreement=True,
        )
    rows = [
        score_speech(
            {
                "id": "unaligned",
                "reference": "alpha",
                "duration_seconds": 1.0,
                "reference_segments": [{"text": "alpha", "start": 0.0, "end": 1.0}],
            },
            "beta",
            [{"text": "beta", "start": 0.0, "end": 1.0}],
            quality_latency_seconds=0.1,
            repeat_transcript_agreement=True,
        ),
        score_nonspeech(
            {"id": "silence", "duration_seconds": 1.0, "nonspeech_kind": "silence"},
            "",
            latency_seconds=0.1,
        ),
    ]
    metrics, intervals = aggregate(
        rows,
        _timings("unaligned"),
        cold_model_load_seconds=1.0,
        peak_process_tree_ram_mb=1.0,
        peak_vram_mb=0.0,
        resamples=0,
        seed=42,
    )
    assert metrics["timestamp_alignment_coverage"] == 0.0
    assert metrics["timestamp_boundary_mae_seconds"] is None
    assert "timestamp_boundary_mae_seconds" not in intervals


def test_audio_registry_and_duration_limit_are_frozen() -> None:
    assert set(ASR_PROFILES) == {
        "whisper-small-en-control",
        "canary-180m",
        "parakeet-tdt-0.6b-v2",
        "moss-transcribe-diarize",
        "qwen3-asr-1.7b-aligned",
    }
    assert len(METRIC_DIRECTIONS) == 16
    speech = [
        {
            "id": "too-long",
            "source_path": "audio.wav",
            "asset_sha256": "checksum",
            "reference": "text",
            "duration_seconds": 30.1,
            "reference_segments": [{"text": "text", "start": 0.0, "end": 30.1}],
        },
        {
            "id": "second",
            "source_path": "audio.wav",
            "asset_sha256": "checksum",
            "reference": "text",
            "duration_seconds": 1.0,
            "reference_segments": [{"text": "text", "start": 0.0, "end": 1.0}],
        },
    ]
    controls = [
        {"id": "silence", "reference": "", "duration_seconds": 1.0, "nonspeech_kind": "silence"},
        {
            "id": "noise",
            "reference": "",
            "duration_seconds": 1.0,
            "nonspeech_kind": "background_noise",
        },
    ]
    with pytest.raises(ValueError, match="between 0 and 30"):
        _validate_manifest_rows(speech, controls, "smoke")

    cpu_environment = _worker_environment("cpu")
    assert cpu_environment["CUDA_VISIBLE_DEVICES"] == ""
    assert cpu_environment["NVIDIA_VISIBLE_DEVICES"] == "none"


def test_asr_runner_logs_flat_metrics_and_required_tables(tmp_path) -> None:
    operational_names = {
        "real_time_factor",
        "p50_warm_clip_latency_seconds",
        "p95_warm_clip_latency_seconds",
        "cold_model_load_seconds",
        "peak_process_tree_ram_mb",
        "peak_vram_mb",
    }
    point_metrics = {name: 0.0 for name in METRIC_DIRECTIONS}

    def evaluator(_candidate):
        operational = {name: point_metrics.pop(name) for name in operational_names}
        return (
            [SampleResult("speech", {}, 0.1), SampleResult("silence", {}, 0.1)],
            operational,
            point_metrics,
            {"device": "cpu"},
            {},
            {
                "samples": [{"sample_id": "speech"}, {"sample_id": "silence"}],
                "timings": [{"sample_id": "speech", "repetition": 1}],
            },
        )

    result = run_benchmark(
        BenchmarkPlan("extraction", "audio-smoke", "smoke", "fixture", ("asr",)),
        evaluator,
        dataset_checksum="fixture",
        directions=METRIC_DIRECTIONS,
        primary_metric="word_error_rate",
        no_mlflow=True,
        artifact_root=tmp_path,
        monitor_resources=False,
        operational_prefix="",
        paired_comparisons=False,
        candidate_artifact_name="candidate.json",
    )
    assert result.complete
    directory = result.artifact_directory / "candidates" / "asr"
    assert (directory / "samples.parquet").is_file()
    assert (directory / "timings.parquet").is_file()
    candidate = json.loads((directory / "candidate.json").read_text(encoding="utf-8"))
    assert set(candidate["artifacts"]) == {
        "samples.parquet",
        "timings.parquet",
        "candidate.json",
    }


def test_failed_asr_candidate_keeps_one_sample_artifact(tmp_path) -> None:
    def evaluator(_candidate):
        return (
            [SampleResult("speech", {}, 0.1)],
            {},
            {},
            {},
            {},
            {"samples": [{"sample_id": "speech"}]},
        )

    result = run_benchmark(
        BenchmarkPlan("extraction", "audio-smoke", "smoke", "fixture", ("asr",)),
        evaluator,
        dataset_checksum="fixture",
        directions={"required_quality": "min"},
        primary_metric="required_quality",
        no_mlflow=True,
        artifact_root=tmp_path,
        monitor_resources=False,
        candidate_artifact_name="candidate.json",
    )
    assert not result.complete
    directory = result.artifact_directory
    assert (directory / "candidates/asr/samples.parquet").is_file()
    assert not (directory / "samples/asr.parquet").exists()
    candidate = json.loads(
        (directory / "candidates/asr/candidate.json").read_text(encoding="utf-8")
    )
    assert candidate["artifacts"] == ["samples.parquet", "candidate.json"]


def test_audio_family_leakage_is_rejected() -> None:
    def manifest(split: str, sample_id: str, checksum: str, family: str):
        return DatasetManifest(
            f"audio-{split}",
            "1",
            "extraction",
            split,
            "fixture",
            "fixture",
            "fixture",
            "fixture",
            "fixture",
            42,
            (
                {
                    "kind": "audio",
                    "id": sample_id,
                    "asset_sha256": checksum,
                    "document_family": family,
                },
            ),
        )

    with pytest.raises(ValueError, match="split leakage"):
        assert_no_split_leakage(
            [
                manifest("development", "one", "a", "same-speaker"),
                manifest("validation", "two", "b", "same-speaker"),
            ]
        )


def test_audio_family_can_have_multiple_clips_inside_one_split() -> None:
    def manifest(sample_id: str, checksum: str):
        return DatasetManifest(
            f"audio-development-{sample_id}",
            "1",
            "extraction",
            "development",
            "fixture",
            "fixture",
            "fixture",
            "fixture",
            "fixture",
            42,
            (
                {
                    "kind": "audio",
                    "id": sample_id,
                    "asset_sha256": checksum,
                    "document_family": "same-speaker",
                },
            ),
        )

    assert_no_split_leakage([manifest("one", "a"), manifest("two", "b")])


def test_reliability_asset_cannot_cross_splits() -> None:
    with pytest.raises(ValueError, match="Duplicate ASR reliability asset checksum"):
        _validate_reliability_split_isolation(
            [
                {
                    "kind": "audio_reliability",
                    "id": "development-silence",
                    "split": "development",
                    "asset_sha256": "same-audio",
                },
                {
                    "kind": "audio_reliability",
                    "id": "validation-silence",
                    "split": "validation",
                    "asset_sha256": "same-audio",
                },
            ]
        )


@pytest.mark.parametrize(
    ("changed_field", "changed_value", "message"),
    (
        ("id", "development-silence", "Duplicate ASR reliability sample ID"),
        ("asset_sha256", "same-audio", "Duplicate ASR reliability asset checksum"),
    ),
)
def test_reliability_controls_cannot_repeat_inside_one_split(
    changed_field: str, changed_value: str, message: str
) -> None:
    first = {
        "kind": "audio_reliability",
        "id": "development-silence",
        "split": "development",
        "asset_sha256": "same-audio",
    }
    second = {
        "kind": "audio_reliability",
        "id": "development-noise",
        "split": "development",
        "asset_sha256": "different-audio",
        changed_field: changed_value,
    }
    with pytest.raises(ValueError, match=message):
        _validate_reliability_split_isolation([first, second])


def test_reference_transcript_must_match_timed_segment_text() -> None:
    speech = [
        {
            "id": "mismatched-reference",
            "source_path": "audio.wav",
            "asset_sha256": "checksum",
            "reference": "the verified transcript",
            "duration_seconds": 1.0,
            "reference_segments": [
                {"text": "different segment text", "start": 0.0, "end": 1.0}
            ],
        },
        {
            "id": "valid-reference",
            "source_path": "audio-2.wav",
            "asset_sha256": "checksum-2",
            "reference": "Another, VERIFIED transcript!",
            "duration_seconds": 1.0,
            "reference_segments": [
                {"text": "another verified transcript", "start": 0.0, "end": 1.0}
            ],
        },
    ]
    controls = [
        {
            "id": "silence",
            "source_path": "silence.wav",
            "asset_sha256": "silence-checksum",
            "nonspeech_kind": "silence",
            "split": "smoke",
            "reference": "",
            "duration_seconds": 1.0,
        },
        {
            "id": "noise",
            "source_path": "noise.wav",
            "asset_sha256": "noise-checksum",
            "nonspeech_kind": "background_noise",
            "split": "smoke",
            "reference": "",
            "duration_seconds": 1.0,
        },
    ]

    with pytest.raises(ValueError, match="reference does not match its timed segments"):
        _validate_manifest_rows(speech, controls, "smoke")

    valid = speech[1]
    _validate_manifest_rows(
        [
            valid,
            {
                **valid,
                "id": "valid-reference-2",
                "source_path": "audio-3.wav",
                "asset_sha256": "checksum-3",
            },
        ],
        controls,
        "smoke",
    )


def test_authoritative_audio_split_requires_all_condition_groups() -> None:
    speech = []
    for index in range(54):
        condition = "clean" if index else "noisy"
        speech.append(
            {
                "id": f"speech-{index}",
                "source_path": f"audio-{index}.wav",
                "asset_sha256": f"checksum-{index}",
                "reference": "verified speech",
                "duration_seconds": 1.0,
                "reference_segments": [
                    {"text": "verified speech", "start": 0.0, "end": 1.0}
                ],
                "source_license": "fixture",
                "source_revision": "1",
                "split": "development",
                "document_family": f"family-{index}",
                "condition": condition,
            }
        )
    controls = [
        {
            "id": kind,
            "source_path": f"{kind}.wav",
            "asset_sha256": f"{kind}-checksum",
            "nonspeech_kind": kind,
            "split": "development",
            "reference": "",
            "duration_seconds": 1.0,
            "source_license": "fixture",
            "source_revision": "1",
        }
        for kind in ("silence", "music_without_lyrics", "background_noise", "environmental_sound")
    ]

    with pytest.raises(ValueError, match="lacks required conditions"):
        _validate_manifest_rows(speech, controls, "standard")

    assert {"accented", "multi_speaker"} <= REQUIRED_SPEECH_CONDITIONS


def test_qwen_forced_aligner_uses_official_result_items(monkeypatch) -> None:
    import torch

    class Inputs(dict):
        def to(self, *_args):
            return self

    class ASRProcessor:
        def apply_transcription_request(self, **_kwargs):
            return Inputs(input_ids=torch.tensor([[1, 2]]))

        def decode(self, _tokens, *, return_format):
            assert return_format == "parsed"
            return [{"language": "English", "transcription": "hello world"}]

    class ASRModel:
        device = torch.device("cpu")
        dtype = torch.float32

        def generate(self, **_kwargs):
            return torch.tensor([[1, 2, 3]])

    class AlignerInputs(dict):
        def to(self, *_args):
            return self

    class AlignerProcessor:
        def prepare_forced_aligner_inputs(self, **_kwargs):
            return AlignerInputs(input_ids=torch.tensor([[1, 2]])), [["hello", "world"]]

        def decode_forced_alignment(self, **_kwargs):
            return [[
                SimpleNamespace(text="hello", start_time=0.0, end_time=0.4),
                SimpleNamespace(text="world", start_time=0.5, end_time=1.0),
            ]]

    class AlignerModel:
        device = torch.device("cpu")
        dtype = torch.float32
        config = SimpleNamespace(timestamp_token_id=1)

        def __call__(self, **_kwargs):
            return SimpleNamespace(logits=torch.tensor([0.0]))

    profile = ASR_PROFILES["qwen3-asr-1.7b-aligned"]
    runtime = QwenRuntime(profile, Path("asr"), "cpu", Path("aligner"))
    runtime._runtime = (ASRModel(), ASRProcessor())
    monkeypatch.setattr(
        runtime,
        "_load_aligner",
        lambda: (AlignerModel(), AlignerProcessor()),
    )

    transcript = runtime.transcribe(Path("audio.wav"))

    assert transcript.text == "hello world"
    assert transcript.segments == (
        {"text": "hello", "start": 0.0, "end": 0.4},
        {"text": "world", "start": 0.5, "end": 1.0},
    )


def test_required_cuda_monitoring_cannot_report_fabricated_zero() -> None:
    monitor = ResourceMonitor(require_vram=True, report_zero_vram=True)
    with pytest.raises(RuntimeError, match="did not capture"):
        monitor.metrics()
