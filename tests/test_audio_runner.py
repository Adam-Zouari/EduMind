"""ASR runner contract regressions without workers, telemetry, or inference."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from edumind.common.artifacts import atomic_write_json, stable_hash
from experiments.benchmarks.common.contracts import CandidateExecutionError
from experiments.benchmarks.extraction.audio import evaluate, runner


@pytest.fixture
def protocol():
    return SimpleNamespace(
        candidates=("asr-a", "asr-b", "asr-c"),
        maximum_finalists=2,
        meta=SimpleNamespace(checksum="current-audio-protocol"),
    )


@pytest.fixture
def evidence():
    speech = [
        {
            "id": "speech",
            "reference": "alpha beta",
            "duration_seconds": 1.0,
            "reference_segments": [{"text": "alpha beta", "start": 0.0, "end": 1.0}],
        }
    ]
    controls = [{"id": "control", "duration_seconds": 1.0, "nonspeech_kind": "silence"}]
    samples = [
        evaluate.score_speech(
            speech[0],
            "alpha beta",
            speech[0]["reference_segments"],
            quality_latency_seconds=0.5,
            alignment_threshold=0.5,
            timestamp_tolerance_seconds=0.1,
        ),
        evaluate.score_nonspeech(controls[0], "", latency_seconds=0.5),
    ]
    timings = []
    for row, repetitions, text in zip(samples, (3, 1), ("alpha beta", ""), strict=True):
        row.update(
            first_attempt_executed=True,
            scheduled_attempt_count=repetitions,
            executed_attempt_count=repetitions,
            failed_attempt_count=0,
            unexecuted_attempt_count=0,
            attempt_failure_rate=0.0,
            transcript_repeatability_success_rate=1.0 if repetitions > 1 else None,
            source_group_id=row["sample_id"],
        )
        for index in range(1, repetitions + 1):
            timings.append(
                {
                    "sample_id": row["sample_id"],
                    "sample_type": row["sample_type"],
                    "repetition": index,
                    "executed": True,
                    "success": True,
                    "error": None,
                    "latency_seconds": 0.4 + index / 10,
                    "duration_seconds": row["duration_seconds"],
                    "device": "cpu",
                    "projected_transcript_sha256": stable_hash(text),
                }
            )
    output = {"samples": samples, "timings": timings, "execution_error": None}
    _score(output)
    return output, speech, controls


def _score(output):
    output["metrics"], output["intervals"] = evaluate.aggregate(
        output["samples"],
        output["timings"],
        cold_model_load_seconds=1.0,
        peak_process_tree_ram_mb=100.0,
        peak_vram_mb=None,
        resamples=0,
        seed=42,
        confidence=0.95,
    )
    output["metric_statuses"] = evaluate.metric_statuses(
        output["metrics"],
        output["samples"],
        output["timings"],
        "cpu",
        output.get("execution_error"),
    )


def _validate(evidence, *, device="cpu"):
    output, speech, controls = evidence
    return runner._validate_output(output, speech, controls, 3, device)


def _reject(evidence, match):
    with pytest.raises(CandidateExecutionError, match=match) as caught:
        _validate(evidence)
    assert caught.value.artifacts["worker_evidence"] is evidence[0]


def test_validate_output_accepts_actual_aggregate_support_fields_and_is_read_only(
    evidence,
):
    output = evidence[0]
    assert output["metrics"]["word_error_rate.planned_reference_word_count"] == 2
    assert output["metrics"]["word_error_rate.contributing_source_count"] == 1
    assert set(evaluate.METRIC_DIRECTIONS) < output["metrics"].keys()
    original = deepcopy(evidence)
    assert _validate(evidence) is None
    assert evidence == original


@pytest.mark.parametrize("repetition", [True, False, 1.0, "1", None, 0, -1, 4, [], {}])
def test_validate_output_rejects_malformed_repetitions(evidence, repetition):
    evidence[0]["timings"][0]["repetition"] = repetition
    _reject(evidence, "attempt inventory")


@pytest.mark.parametrize(
    "mutation", ["missing", "duplicate", "extra", "extra-control-repeat"]
)
def test_validate_output_requires_exact_speech_and_control_attempt_inventory(
    evidence, mutation
):
    timings = evidence[0]["timings"]
    if mutation == "missing":
        timings.pop(1)
    elif mutation == "duplicate":
        timings.append(deepcopy(timings[0]))
    elif mutation == "extra":
        timings.append({**timings[0], "sample_id": "unscheduled"})
    else:
        timings.append({**timings[-1], "repetition": 2})
    _reject(evidence, "attempt inventory")


@pytest.mark.parametrize(
    "field,value",
    [
        ("sample_type", "nonspeech"),
        ("device", "cuda"),
        ("duration_seconds", 2.0),
    ],
)
def test_validate_output_requires_attempt_identity_to_match_inputs(
    evidence, field, value
):
    evidence[0]["timings"][0][field] = value
    _reject(evidence, "identity differs")


@pytest.mark.parametrize("executed", [None, 0, 1, "true", [], {}])
def test_validate_output_requires_boolean_execution_status(evidence, executed):
    evidence[0]["timings"][0]["executed"] = executed
    _reject(evidence, "explicit execution status")


@pytest.mark.parametrize("success", [None, 0, 1, "true", [], {}])
def test_validate_output_requires_boolean_measured_success(evidence, success):
    evidence[0]["timings"][0]["success"] = success
    _reject(evidence, "measured attempt")


@pytest.mark.parametrize("latency", [None, True, "0.5", -1, float("nan"), float("inf")])
def test_validate_output_rejects_invalid_measured_latency(evidence, latency):
    evidence[0]["timings"][0]["latency_seconds"] = latency
    _reject(evidence, "measured attempt")


@pytest.mark.parametrize(
    "fingerprint", [None, 64, "", "a" * 63, "a" * 65, "A" * 64, "g" * 64]
)
def test_validate_output_requires_completed_transcript_sha256(evidence, fingerprint):
    evidence[0]["timings"][0]["projected_transcript_sha256"] = fingerprint
    _reject(evidence, "valid output fingerprint")


def _failed_first(evidence, *, executed=True):
    output = evidence[0]
    first = output["timings"][0]
    first.update(
        executed=executed,
        success=False if executed else None,
        error="recorded failure" if executed else None,
        projected_transcript_sha256=None,
    )
    if not executed:
        first.update(latency_seconds=None, not_executed_reason="interrupted")
        output["execution_error"] = {"phase": "load", "error": "interrupted"}
    sample = output["samples"][0]
    for field in (
        "reference_word_count",
        "reference_character_count",
        "reference_timed_segment_count",
    ):
        sample["planned_" + field] = sample[field]
        sample[field] = None
    sample.update(
        first_attempt_executed=executed,
        first_attempt_success=first["success"],
        metric_status="unavailable" if executed else "incomplete",
        reason="first_attempt_failed" if executed else "first_attempt_not_executed",
        quality_latency_seconds=None,
        executed_attempt_count=3 if executed else 2,
        failed_attempt_count=1 if executed else 0,
        unexecuted_attempt_count=0 if executed else 1,
        attempt_failure_rate=1 / 3 if executed else None,
        transcript_repeatability_success_rate=1 / 3 if executed else None,
    )
    _score(output)


@pytest.mark.parametrize("executed", [True, False])
def test_validate_output_accepts_null_first_quality_without_selecting_later_success(
    evidence, executed
):
    _failed_first(evidence, executed=executed)
    assert evidence[0]["samples"][0]["metric_values"]["word_error_rate"] is None
    assert _validate(evidence) is None


@pytest.mark.parametrize("executed", [True, False])
def test_validate_output_rejects_first_attempt_replacement_by_later_success(
    evidence, executed
):
    _failed_first(evidence, executed=executed)
    evidence[0]["samples"][0].update(
        first_attempt_executed=True, first_attempt_success=True
    )
    _reject(evidence, "designated first attempt")


@pytest.mark.parametrize("executed", [True, False])
@pytest.mark.parametrize("name", ["word_error_rate", "timestamp_alignment_coverage"])
def test_validate_output_failed_first_cannot_publish_later_quality_with_correct_flags(
    evidence, executed, name
):
    _failed_first(evidence, executed=executed)
    sample = evidence[0]["samples"][0]
    sample["metric_values"][name] = 0.0
    sample["metric_statuses"][name] = {"status": "scored", "reason": None}
    _reject(evidence, "quality|first attempt")


@pytest.mark.parametrize("quality_latency", [None, 0.6])
def test_validate_output_successful_quality_latency_is_the_first_measured_latency(
    evidence, quality_latency
):
    evidence[0]["samples"][0]["quality_latency_seconds"] = quality_latency
    _reject(evidence, "quality latency")


@pytest.mark.parametrize(
    "error,fingerprint", [(None, None), ("", None), (True, None), ("failed", "a" * 64)]
)
def test_validate_output_rejects_fabricated_failed_attempt_evidence(
    evidence, error, fingerprint
):
    _failed_first(evidence)
    evidence[0]["timings"][0].update(
        error=error, projected_transcript_sha256=fingerprint
    )
    _reject(evidence, "failed attempt evidence")


@pytest.mark.parametrize(
    "field,value",
    [
        ("success", False),
        ("error", "invented failure"),
        ("latency_seconds", 0.0),
        ("projected_transcript_sha256", "a" * 64),
        ("not_executed_reason", None),
    ],
)
def test_validate_output_rejects_fabricated_unexecuted_observations(
    evidence, field, value
):
    _failed_first(evidence, executed=False)
    evidence[0]["timings"][0][field] = value
    _reject(evidence, "unexecuted attempt")


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "extra"])
def test_validate_output_requires_one_sample_row_per_input(evidence, mutation):
    samples = evidence[0]["samples"]
    if mutation == "missing":
        samples.pop()
    elif mutation == "duplicate":
        samples.append(deepcopy(samples[0]))
    else:
        samples.append({**samples[0], "sample_id": "unknown"})
    _reject(evidence, "sample inventory")


@pytest.mark.parametrize(
    "field,value",
    [
        ("metric_values", None),
        ("metric_statuses", []),
        ("metric_values", {"word_error_rate": True}),
    ],
)
def test_validate_output_rejects_missing_or_invalid_sample_metrics(
    evidence, field, value
):
    evidence[0]["samples"][0][field] = value
    _reject(evidence, "metric")


@pytest.mark.parametrize(
    "mutation",
    [
        "missing-core",
        "negative-support",
        "boolean-support",
        "nonfinite-support",
        "cpu-zero-vram",
    ],
)
def test_validate_output_checks_core_and_support_metric_values(evidence, mutation):
    metrics = evidence[0]["metrics"]
    if mutation == "missing-core":
        metrics.pop("word_error_rate")
    elif mutation == "cpu-zero-vram":
        metrics["peak_vram_mb"] = 0.0
    else:
        metrics["word_error_rate.contributing_count"] = {
            "negative-support": -1,
            "boolean-support": True,
            "nonfinite-support": float("nan"),
        }[mutation]
    _reject(evidence, "metric contract|CPU VRAM")


def test_validate_output_accepts_declared_cuda_profile_without_using_a_gpu(evidence):
    for row in evidence[0]["timings"]:
        row["device"] = "cuda"
    evidence[0]["metrics"]["peak_vram_mb"] = 100.0
    evidence[0]["metric_statuses"] = evaluate.metric_statuses(
        evidence[0]["metrics"], evidence[0]["samples"], evidence[0]["timings"], "cuda"
    )
    assert _validate(evidence, device="cuda") is None


@pytest.mark.parametrize(
    "mutation",
    ["error", "sample_values", "sample_statuses", "aggregate_statuses", "comparison"],
)
def test_incomplete_evidence_cannot_become_a_success_by_clearing_fields(
    evidence, mutation
):
    _failed_first(evidence, executed=False)
    output = evidence[0]
    if mutation == "error":
        output["execution_error"] = None
    elif mutation == "sample_values":
        output["samples"][0]["metric_values"] = {}
    elif mutation == "sample_statuses":
        output["samples"][0]["metric_statuses"] = {}
    elif mutation == "aggregate_statuses":
        output["metric_statuses"] = {}
    else:
        output["metric_statuses"]["word_error_rate"]["comparison_status"] = "complete"
    _reject(evidence, "ASR")


@pytest.mark.parametrize(
    "field", ["support", "comparison_reason", "execution_error_phase"]
)
def test_aggregate_status_records_require_support_and_failure_context(evidence, field):
    del evidence[0]["metric_statuses"]["word_error_rate"][field]
    _reject(evidence, "aggregate metric status")


@pytest.mark.parametrize("profile", ["smoke", "preflight", "development"])
def test_candidates_nonselection_profiles_use_the_exact_protocol_roster(
    monkeypatch, protocol, profile
):
    monkeypatch.setattr(
        runner,
        "load_engineer_decision",
        lambda *_args, **_kwargs: pytest.fail("Unexpected engineer decision"),
    )
    assert (
        runner._candidates(profile, None, protocol, "qualification")
        == protocol.candidates
    )


@pytest.mark.parametrize("profile", ["smoke", "preflight", "development"])
def test_candidates_nonselection_profiles_reject_manual_shortlists(
    protocol, profile, tmp_path
):
    with pytest.raises(ValueError, match="does not accept a manual shortlist"):
        runner._candidates(profile, tmp_path / "decision.json", protocol)


@pytest.mark.parametrize(
    "profile,exact,maximum,source",
    [
        ("validation", None, 2, ("extraction", "audio-development", "development")),
        ("locked", 1, 1, ("extraction", "audio-validation", "validation")),
    ],
)
def test_candidates_pass_exact_protocol_qualification_and_source_contract(
    monkeypatch, protocol, tmp_path, profile, exact, maximum, source
):
    path = tmp_path / "decision.json"
    selected = protocol.candidates[:1]
    captured = []

    def load(*args, **kwargs):
        captured.append((args, kwargs))
        return SimpleNamespace(selected_candidates=selected)

    monkeypatch.setattr(runner, "load_engineer_decision", load)
    assert (
        runner._candidates(profile, path, protocol, "current-qualification") == selected
    )
    assert captured == [
        (
            (path,),
            {
                "exact": exact,
                "maximum": maximum,
                "expected_protocols": {"audio": protocol.meta.checksum},
                "expected_qualification": "current-qualification",
                "expected_source": source,
            },
        )
    ]


@pytest.mark.parametrize("profile", ["validation", "locked"])
def test_candidates_require_both_decision_and_current_qualification(
    protocol, tmp_path, profile
):
    with pytest.raises(ValueError, match="requires --shortlist"):
        runner._candidates(profile, None, protocol, "qualification")
    with pytest.raises(ValueError, match="current hardware qualification"):
        runner._candidates(profile, tmp_path / "decision.json", protocol)


@pytest.mark.parametrize(
    "selected,match",
    [((), "empty"), (("asr-a", "asr-a"), "duplicates"), (("unknown",), "Unknown ASR")],
)
def test_candidates_reject_invalid_selected_rosters(
    monkeypatch, protocol, tmp_path, selected, match
):
    monkeypatch.setattr(
        runner,
        "load_engineer_decision",
        lambda *_args, **_kwargs: SimpleNamespace(selected_candidates=selected),
    )
    with pytest.raises(ValueError, match=match):
        runner._candidates(
            "validation", tmp_path / "decision.json", protocol, "qualification"
        )


def _decision(tmp_path, protocol, profile, *, mutation=None):
    source_profile = "development" if profile == "validation" else "validation"
    summary = {
        "run_id": "source-run",
        "complete": True,
        "protocols": {"audio": {"checksum": protocol.meta.checksum}},
        "plan": {
            "suite": "extraction",
            "stage": "audio-" + source_profile,
            "profile": source_profile,
            "settings": {"preflight_fingerprint": "current-qualification"},
        },
        "candidates": [
            {"candidate": candidate, "status": "success"}
            for candidate in protocol.candidates
        ],
    }
    decision = {
        "schema_version": 1,
        "source_summary": "summary.json",
        "source_run_id": "source-run",
        "selected_candidates": list(
            protocol.candidates[: 2 if profile == "validation" else 1]
        ),
        "selected_by": "engineer",
        "selected_date": "2026-10-13",
        "reason": "Reviewed evidence",
    }
    if mutation == "protocol":
        summary["protocols"]["audio"]["checksum"] = "stale-protocol"
    elif mutation == "qualification":
        summary["plan"]["settings"]["preflight_fingerprint"] = "stale-qualification"
    elif mutation == "stage":
        summary["plan"]["stage"] = "video-" + source_profile
    elif mutation == "incomplete":
        summary["complete"] = False
    elif mutation == "too-many":
        decision["selected_candidates"] = list(protocol.candidates)
    atomic_write_json(tmp_path / "summary.json", summary)
    path = tmp_path / "decision.json"
    atomic_write_json(path, decision)
    return path, tuple(decision["selected_candidates"])


@pytest.mark.parametrize("profile", ["validation", "locked"])
def test_candidates_load_complete_protocol_bound_engineer_decisions(
    protocol, tmp_path, profile
):
    path, selected = _decision(tmp_path, protocol, profile)
    assert (
        runner._candidates(profile, path, protocol, "current-qualification") == selected
    )


@pytest.mark.parametrize("profile", ["validation", "locked"])
@pytest.mark.parametrize(
    "mutation,match",
    [
        ("protocol", "different audio protocol"),
        ("qualification", "different hardware qualification"),
        ("stage", "must select from stage"),
        ("incomplete", "incomplete"),
        ("too-many", "must select"),
    ],
)
def test_candidates_reject_stale_incomplete_wrong_stage_or_oversized_decisions(
    protocol, tmp_path, profile, mutation, match
):
    path, _ = _decision(tmp_path, protocol, profile, mutation=mutation)
    with pytest.raises(ValueError, match=match):
        runner._candidates(profile, path, protocol, "current-qualification")
