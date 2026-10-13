"""ASR adapter regressions; no downloads, model weights, or GPU execution."""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from omegaconf import OmegaConf

from experiments.benchmarks.extraction.audio import adapters
from experiments.benchmarks.extraction.audio.evaluate import validate_prediction
from experiments.benchmarks.extraction.audio.protocol import (
    load_protocol,
    protocol_from_mapping,
)

PROTOCOL = load_protocol()
WHISPER = "whisper-small-en-control"
CANARY = "canary-180m"
PARAKEET = "parakeet-tdt-0.6b-v2"
MOSS = "moss-transcribe-diarize"


def runtime(alias, path=Path("snapshot"), device="cpu", protocol=PROTOCOL):
    cls = {
        WHISPER: adapters.WhisperRuntime,
        CANARY: adapters.NemoRuntime,
        PARAKEET: adapters.NemoRuntime,
        MOSS: adapters.MossRuntime,
    }[alias]
    return cls(adapters.profiles(protocol)[alias], path, device, protocol)


@pytest.fixture(autouse=True)
def no_runtime_metadata_imports(monkeypatch):
    monkeypatch.setattr(adapters, "_runtime_version", lambda _backend: "test-version")
    monkeypatch.setattr(adapters, "package_versions", lambda _packages: {})


class Model:
    def __init__(self):
        self.weights = [SimpleNamespace(device="cpu", dtype="torch.float32")]
        self.cfg = SimpleNamespace(
            decoding=OmegaConf.create(
                {
                    "strategy": "greedy",
                    "beam": {"beam_size": 7, "return_best_hypothesis": False},
                    "compute_timestamps": False,
                    "rnnt_timestamp_type": "char",
                }
            )
        )
        OmegaConf.set_struct(self.cfg.decoding, True)
        self.config = SimpleNamespace(_attn_implementation="sdpa")
        self.generation_config = SimpleNamespace(num_beams=7)
        self.events = []

    def parameters(self):
        return iter(self.weights)

    def buffers(self):
        return iter(())

    def to(self, device, dtype=None):
        self.events.append(("to", device, dtype))
        for item in self.weights:
            item.device = str(device)
            item.dtype = dtype or item.dtype
        return self

    def eval(self):
        return self

    def change_decoding_strategy(self, decoding):
        self.events.append(("decode", OmegaConf.to_container(decoding)))


def fake_torch(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            float32="torch.float32",
            float16="torch.float16",
            bfloat16="torch.bfloat16",
            device=lambda value: value,
        ),
    )


def fake_nemo(monkeypatch, model):
    fake_torch(monkeypatch)
    calls = []

    def restore_from(**kwargs):
        calls.append(kwargs)
        return model

    nemo = ModuleType("nemo")
    collections = ModuleType("nemo.collections")
    asr = ModuleType("nemo.collections.asr")
    nemo.collections = collections
    collections.asr = asr
    asr.models = SimpleNamespace(ASRModel=SimpleNamespace(restore_from=restore_from))
    for module in (nemo, collections, asr):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    return calls


def fake_moss(monkeypatch, model, output):
    fake_torch(monkeypatch)
    calls = []
    processor = object()

    def loader(path, **kwargs):
        calls.append(("model", path, kwargs))
        return model

    def processor_loader(path, **kwargs):
        calls.append(("processor", path, kwargs))
        return processor

    def generate(*args, **kwargs):
        calls.append(("generate", args, kwargs))
        return output

    package = ModuleType("moss_transcribe_diarize")
    inference = ModuleType("moss_transcribe_diarize.inference_utils")
    inference.build_transcription_messages = lambda path: [path]
    inference.generate_transcription = generate
    package.inference_utils = inference
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, inference.__name__, inference)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=loader),
            AutoProcessor=SimpleNamespace(from_pretrained=processor_loader),
        ),
    )
    return calls, processor


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("alias", [WHISPER, CANARY, PARAKEET, MOSS])
def test_roster_runtime_and_lock_identity_are_explicit(tmp_path, alias, device):
    profile = adapters.profiles(PROTOCOL)[alias]
    lock = {
        profile.model: {
            "model": profile.model,
            "revision": "pinned-sha",
            "selection_revision": "selection-sha",
            "model_path": str(tmp_path),
            "model_cache_manifest_sha256": "manifest-sha",
        }
    }
    asr = adapters.build_runtime(alias, lock, device, PROTOCOL)
    parameters = asr.parameters()
    assert parameters["dtype"] == PROTOCOL.dtype(alias, device)
    assert parameters["device"] == device
    assert parameters["model_path"] == str(tmp_path.resolve())
    assert parameters["model_revision"] == "pinned-sha"
    assert parameters["selection_revision"] == "selection-sha"
    assert parameters["model_cache_manifest_sha256"] == "manifest-sha"
    assert parameters["sample_rate_hz"] == 16000
    assert parameters["batch_size"] == 1
    assert len(adapters.profiles(PROTOCOL)) == 4
    with pytest.raises(ValueError, match="Unknown ASR"):
        adapters.build_runtime("qwen-asr", lock, device, PROTOCOL)


@pytest.mark.parametrize(
    "entry, error",
    [
        ({}, "no entry"),
        ({"model_path": "snapshot"}, "pinned revision"),
        ({"revision": " "}, "pinned revision"),
        ({"revision": "sha", "model": "wrong"}, "different model"),
    ],
)
def test_lock_rejects_missing_revision_or_wrong_identity(entry, error):
    with pytest.raises(RuntimeError, match=error):
        adapters.build_runtime(
            WHISPER, {PROTOCOL.candidate(WHISPER).model_id: entry}, "cpu", PROTOCOL
        )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_whisper_load_checks_actual_weights_and_dtype(monkeypatch, device):
    asr = runtime(WHISPER, device=device)
    model = Model().to(device, f"torch.{asr.dtype}")
    calls = []

    def load(path, device, **kwargs):
        calls.append((path, device, kwargs))
        return SimpleNamespace(model=model), asr.dtype

    monkeypatch.setattr(adapters, "load_whisper_runtime", load)
    asr.load()
    assert calls == [(Path("snapshot"), device, {"dtype": asr.dtype})]
    assert asr.placement_models() == (model,)
    model.weights.append(SimpleNamespace(device="meta", dtype=f"torch.{asr.dtype}"))
    with pytest.raises(RuntimeError, match="one requested"):
        asr.load()


def test_whisper_executes_greedy_batch_one_and_preserves_incomplete_chunks():
    asr = runtime(WHISPER)
    calls = []

    def pipeline(path, **kwargs):
        calls.append((path, kwargs))
        return {
            "text": "hello unfinished",
            "chunks": [
                {"text": "hello", "timestamp": (0.0, 0.4)},
                {"text": "unfinished", "timestamp": (0.5, None)},
            ],
        }

    asr._runtime = pipeline
    result = asr.transcribe(Path("audio.wav"))
    assert calls == [
        (
            "audio.wav",
            {
                "batch_size": 1,
                "return_timestamps": "word",
                "generate_kwargs": {"do_sample": False, "num_beams": 1},
            },
        )
    ]
    assert result.text == "hello unfinished"
    assert [item["text"] for item in result.segments] == ["hello", "unfinished"]
    assert result.segments[-1]["end"] is None
    assert result.warnings == (
        "1 Whisper chunk(s) lacked complete timestamp boundaries",
    )
    with pytest.raises((ValueError, TypeError)):
        validate_prediction(result.text, result.segments, 1.0, 0.1)


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        {"text": None},
        {"text": [], "chunks": []},
        {"text": "hello", "chunks": "bad"},
        {"text": "hello", "chunks": [None]},
        {"text": "hello", "chunks": [{"text": "hello", "timestamp": None}]},
        {"text": "hello", "chunks": [{"text": None, "timestamp": (0, 1)}]},
        {"text": "hello", "chunks": [{"text": "hello", "timestamp": (False, 1)}]},
    ],
)
def test_whisper_does_not_coerce_or_swallow_invalid_schema(payload):
    asr = runtime(WHISPER)
    asr._runtime = lambda *args, **kwargs: payload
    with pytest.raises(ValueError):
        asr.transcribe(Path("audio.wav"))


@pytest.mark.parametrize("alias", [CANARY, PARAKEET])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_nemo_decoder_is_ready_before_first_request(
    tmp_path, monkeypatch, alias, device
):
    checkpoint = tmp_path / "model.nemo"
    checkpoint.touch()
    model = Model()
    calls = fake_nemo(monkeypatch, model)
    asr = runtime(alias, tmp_path, device)
    asr.load()
    assert calls == [{"restore_path": str(checkpoint), "map_location": device}]
    assert model.events[0] == ("to", device, f"torch.{PROTOCOL.dtype(alias, device)}")
    decoding = model.events[1][1]
    if alias == CANARY:
        assert decoding["strategy"] == "beam"
        assert decoding["beam"]["beam_size"] == asr.parameters()["beam_size"] == 1
        assert decoding["beam"]["return_best_hypothesis"] is True
    else:
        assert (
            decoding["strategy"]
            == asr.parameters()["decoding_strategy"]
            == "greedy_batch"
        )
        assert decoding["compute_timestamps"] is True
        assert decoding["rnnt_timestamp_type"] == "all"
        assert asr.parameters()["timestamp_level"] == "segment_then_word"


@pytest.mark.parametrize("count", [0, 2])
def test_nemo_does_not_guess_between_checkpoints(tmp_path, monkeypatch, count):
    calls = fake_nemo(monkeypatch, Model())
    for index in range(count):
        (tmp_path / f"{index}.nemo").touch()
    with pytest.raises(FileNotFoundError, match="exactly one"):
        runtime(CANARY, tmp_path).load()
    assert calls == []


def test_native_canary_rejects_external_timestamp_aligner(tmp_path, monkeypatch):
    (tmp_path / "model.nemo").touch()
    model = Model()
    model.timestamps_asr_model = Model()
    fake_nemo(monkeypatch, model)
    with pytest.raises(RuntimeError, match="external aligner"):
        runtime(CANARY, tmp_path).load()


@pytest.mark.parametrize("alias", [CANARY, PARAKEET])
def test_nemo_preserves_seconds_and_segment_then_word_choice(alias):
    asr = runtime(alias)
    calls = []
    native = {
        "segment": [
            {
                "segment": "alpha beta",
                "start": 0.12,
                "end": 1.25,
                "start_offset": 3,
                "end_offset": 31,
            }
        ],
        "word": [{"word": "ignored", "start": 0, "end": 1}],
    }

    def transcribe(paths, **kwargs):
        calls.append((paths, kwargs))
        return [SimpleNamespace(text="alpha beta", timestamp=native)]

    asr._runtime = SimpleNamespace(transcribe=transcribe)
    result = asr.transcribe(Path("audio.wav"))
    assert result.segments == ({"text": "alpha beta", "start": 0.12, "end": 1.25},)
    kwargs = calls[0][1]
    assert kwargs["batch_size"] == 1
    assert kwargs["timestamps"] == asr.parameters()["timestamps"] is True
    assert kwargs["return_hypotheses"] is True
    if alias == CANARY:
        assert kwargs["source_lang"] == kwargs["target_lang"] == "en"
        assert kwargs["pnc"] == "True"
    native["segment"] = []
    native["word"] = [
        {"word": "alpha", "start": 0.12, "end": 0.6},
        {"word": "beta", "start": 0.7, "end": 1.25},
    ]
    words = asr.transcribe(Path("audio.wav"))
    assert [item["text"] for item in words.segments] == ["alpha", "beta"]


@pytest.mark.parametrize(
    "payload",
    [
        None,
        "",
        {"segment": None},
        {"segment": "bad"},
        {"segment": [None]},
        {"segment": [{"segment": None, "start": 0, "end": 1}]},
        {"segment": [{"segment": "hi", "start": False, "end": 1}]},
        {"segment": [{"segment": "hi", "start": float("nan"), "end": 1}]},
        {"segment": [{"segment": "hi", "start_offset": 0, "end_offset": 5}]},
    ],
)
def test_nemo_rejects_invalid_timestamps_instead_of_word_fallback(payload):
    asr = runtime(PARAKEET)
    asr._runtime = SimpleNamespace(
        transcribe=lambda *args, **kwargs: [
            SimpleNamespace(text="hi", timestamp=payload)
        ]
    )
    with pytest.raises(ValueError):
        asr.transcribe(Path("audio.wav"))


@pytest.mark.parametrize(
    "outputs", [[], ["", ""], None, "hi", [SimpleNamespace(text=None)]]
)
def test_nemo_rejects_invalid_cardinality_or_text(outputs):
    asr = runtime(PARAKEET)
    asr._runtime = SimpleNamespace(transcribe=lambda *args, **kwargs: outputs)
    with pytest.raises(ValueError):
        asr.transcribe(Path("audio.wav"))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_moss_load_is_offline_pinned_precision_without_backend_retries(
    monkeypatch, device
):
    model = Model()
    calls, processor = fake_moss(monkeypatch, model, {"text": "[0.25][S01]hello[1.25]"})
    asr = runtime(MOSS, device=device)
    asr.load()
    expected_dtype = f"torch.{PROTOCOL.dtype(MOSS, device)}"
    assert calls == [
        (
            "processor",
            "snapshot",
            {
                "trust_remote_code": True,
                "local_files_only": True,
                "fix_mistral_regex": True,
            },
        ),
        (
            "model",
            "snapshot",
            {
                "trust_remote_code": True,
                "local_files_only": True,
                "dtype": expected_dtype,
                "attn_implementation": "sdpa",
            },
        ),
    ]
    assert model.generation_config.num_beams == asr.parameters()["num_beams"] == 1
    assert asr.placement_models() == (model,)
    result = asr.transcribe(Path("audio.wav"))
    assert result == adapters.Transcript(
        "hello", ({"text": "hello", "start": 0.25, "end": 1.25},)
    )
    _, args, kwargs = calls[-1]
    assert args == (model, processor, ["audio.wav"])
    assert kwargs["max_new_tokens"] == asr.parameters()["max_new_tokens"] == 2048
    assert kwargs["do_sample"] == asr.parameters()["do_sample"] is False
    assert kwargs["device"] == device
    assert kwargs["dtype"] == expected_dtype
    assert kwargs["attention_report"]["selected"] == "sdpa"


def test_moss_does_not_accept_silently_rewritten_attention(monkeypatch):
    model = Model()
    model.config._attn_implementation = "eager"
    calls, _processor = fake_moss(monkeypatch, model, {"text": ""})
    with pytest.raises(RuntimeError, match="SDPA"):
        runtime(MOSS).load()
    assert len([call for call in calls if call[0] == "model"]) == 1


@pytest.mark.parametrize(
    "raw",
    [
        "untimed speech",
        "prefix [0][S01]hello[1]",
        "[0][S01]hello[1] trailing speech",
        "[0][S01]hello[1][1][S02]unfinished",
        "[0][S01]hello[1][broken]",
        "[0][S01]unfinished",
        "[1][S01]hello[0]",
        "[1][S01]hello[1]",
        "[0][bad]hello[1]",
        "[0][S01]hello[1][x][S02]tail[2]",
    ],
)
def test_moss_never_scores_a_tolerantly_parsed_prefix(raw):
    with pytest.raises(ValueError):
        adapters._moss_transcript(raw)


def test_moss_converts_every_segment_and_preserves_literal_brackets():
    result = adapters._moss_transcript(
        "[.25][S01]hello [literal][1.] \n[1.5][S02]world[2]"
    )
    assert result.text == "hello [literal] world"
    assert result.segments == (
        {"text": "hello [literal]", "start": 0.25, "end": 1.0},
        {"text": "world", "start": 1.5, "end": 2.0},
    )


@pytest.mark.parametrize("alias", [WHISPER, CANARY, PARAKEET, MOSS])
def test_every_adapter_accepts_explicit_empty_outputs(monkeypatch, alias):
    asr = runtime(alias)
    if alias == WHISPER:
        asr._runtime = lambda *args, **kwargs: {"text": "", "chunks": []}
    elif alias in (CANARY, PARAKEET):
        asr._runtime = SimpleNamespace(
            transcribe=lambda *args, **kwargs: [SimpleNamespace(text="", timestamp=[])]
        )
    else:
        fake_moss(monkeypatch, Model(), {"text": " \n "})
        asr.load()
    result = asr.transcribe(Path("audio.wav"))
    assert result == adapters.Transcript("", ())
    assert validate_prediction(result.text, result.segments, 1.0, 0.1) == []


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_placement_checks_all_weights_even_with_a_matching_device_map(device):
    model = Model()
    model.hf_device_map = {"": device}
    model.weights = [
        SimpleNamespace(device=device, dtype="torch.float32"),
        SimpleNamespace(device="meta", dtype="torch.float32"),
    ]
    with pytest.raises(RuntimeError, match="one requested"):
        adapters._assert_device(model, device)


def test_placement_rejects_multi_gpu_quantized_mixed_dtype_and_offloaded_models():
    model = Model()
    model.weights = [
        SimpleNamespace(device="cuda:0", dtype="torch.float16"),
        SimpleNamespace(device="cuda:1", dtype="torch.float16"),
    ]
    with pytest.raises(RuntimeError, match="one requested"):
        adapters._assert_device(model, "cuda")
    model = Model()
    model.is_quantized = True
    with pytest.raises(RuntimeError, match="quantization"):
        adapters._assert_device(model, "cpu")
    model = Model()
    with pytest.raises(RuntimeError, match="protocol dtype"):
        adapters._assert_device(model, "cpu", dtype="float16")
    model.named_modules = lambda: [
        ("decoder", SimpleNamespace(_hf_hook=SimpleNamespace(offload=True)))
    ]
    with pytest.raises(RuntimeError, match="offload hooks"):
        adapters._assert_device(model, "cpu")


@pytest.mark.parametrize("placement", ["cpu", "disk", "offload_hook", "weights_map"])
@pytest.mark.parametrize("wrong_dtype", [False, True])
def test_typed_offload_evidence_survives_cleanup_and_resource_annotation(
    placement, wrong_dtype
):
    model = Model()
    model.weights[0].device = "cpu" if placement == "cpu" else "cuda:0"
    model.weights[0].dtype = "torch.float32" if wrong_dtype else "torch.float16"
    if placement == "disk":
        model.hf_device_map = {"encoder": "disk", "decoder": 0}
    if placement in {"offload_hook", "weights_map"}:
        hook = SimpleNamespace(
            offload=placement == "offload_hook",
            weights_map={} if placement == "weights_map" else None,
        )
        model.named_modules = lambda: [("decoder", SimpleNamespace(_hf_hook=hook))]

    with pytest.raises(adapters.ASRPlacementError) as caught:
        adapters._assert_device(model, "cuda", dtype="float16")

    report = caught.value.placement_report
    assert report["status"] == "offload_detected"
    assert report["expected_device"] == "cuda"
    assert report["verified"] is True
    if placement == "cpu":
        assert report["observed_devices"] == ["cpu"]
    elif placement == "disk":
        assert report["observed_devices"] == ["cuda", "disk"]
        assert report["device_maps"] == [{"encoder": "disk", "decoder": "0"}]
    else:
        assert report["offload_hooks"] == ["decoder"]

    # Cleanup must not invalidate this snapshot. ResourceMonitor annotates a
    # separate attribute on escaping exceptions, so placement must survive that.
    snapshot = deepcopy(report)
    model.weights.clear()
    model.hf_device_map = {}
    model.named_modules = list
    caught.value.resource_evidence = {"peak_vram_mb": 3.0}
    assert caught.value.placement_report == snapshot


@pytest.mark.parametrize("failure", ["dtype", "quantization"])
def test_nonplacement_rejections_are_not_typed_or_inspected(monkeypatch, failure):
    model = Model()
    model.weights[0].device = "cuda:0"
    if failure == "quantization":
        model.is_quantized = True

    def unexpected_inspection(*args, **kwargs):
        pytest.fail("Nonplacement failures must not fabricate placement evidence")

    monkeypatch.setattr(adapters, "inspect_model_placement", unexpected_inspection)
    with pytest.raises(RuntimeError) as caught:
        adapters._assert_device(model, "cuda", dtype="float16")
    assert type(caught.value) is RuntimeError
    assert not hasattr(caught.value, "placement_report")


def test_typed_wrong_device_evidence_is_not_an_offload_exclusion():
    model = Model()
    model.weights[0].device = "mps"
    with pytest.raises(adapters.ASRPlacementError) as caught:
        adapters._assert_device(model, "cuda")
    assert caught.value.placement_report["status"] == "wrong_device"
    assert caught.value.placement_report["observed_devices"] == ["mps"]


def test_rejected_device_splitting_cannot_report_qualified():
    model = Model()
    model.weights = [
        SimpleNamespace(device="cuda:0", dtype="torch.float16"),
        SimpleNamespace(device="cuda:1", dtype="torch.float16"),
    ]
    with pytest.raises(adapters.ASRPlacementError) as caught:
        adapters._assert_device(model, "cuda", dtype="float16")
    assert caught.value.placement_report["status"] == "device_splitting_detected"
    assert caught.value.placement_report["observed_device_locations"] == [
        "cuda:0",
        "cuda:1",
    ]


def test_loader_oom_is_not_reinterpreted_as_placement(monkeypatch):
    original = RuntimeError("CUDA out of memory")

    def load(*args, **kwargs):
        raise original

    monkeypatch.setattr(adapters, "load_whisper_runtime", load)
    with pytest.raises(RuntimeError) as caught:
        runtime(WHISPER, device="cuda").load()
    assert caught.value is original
    assert not hasattr(caught.value, "placement_report")


def test_protocol_decoding_values_drive_calls_and_logging(monkeypatch):
    raw = deepcopy(PROTOCOL.meta.resolved)
    raw["decoding"][MOSS]["max_new_tokens"] = 321
    protocol = protocol_from_mapping(raw)
    calls, _processor = fake_moss(monkeypatch, Model(), {"text": ""})
    asr = runtime(MOSS, protocol=protocol)
    asr.load()
    asr.transcribe(Path("audio.wav"))
    assert calls[-1][2]["max_new_tokens"] == asr.parameters()["max_new_tokens"] == 321


@pytest.mark.parametrize(
    "raw, discarded",
    [
        ("<|0|>hello<|10|>", False),
        ("prefix <|0|>hello<|10|>", True),
        ("<|0|>hello<|10|> unfinished tail", True),
    ],
)
def test_canary_detects_lexical_loss_in_native_timestamp_processing(raw, discarded):
    asr = runtime(CANARY)
    hypothesis = SimpleNamespace(
        text="hello",
        y_sequence=[1, 2],
        timestamp={
            "segment": [{"segment": "hello", "start": 0, "end": 1}],
        },
    )
    asr._runtime = SimpleNamespace(
        transcribe=lambda *args, **kwargs: [hypothesis],
        decoding=SimpleNamespace(decode_ids_to_str=lambda token_ids: raw),
    )
    if discarded:
        with pytest.raises(ValueError, match="discarded lexical text"):
            asr.transcribe(Path("audio.wav"))
    else:
        assert asr.transcribe(Path("audio.wav")).text == "hello"


def test_canary_timestamp_only_silence_is_a_valid_empty_transcript():
    # Confirmed native CPU output: no paired timestamps, so NeMo leaves the
    # numeric timestamp control in hyp.text rather than returning empty text.
    raw = "<|0|>"
    hypothesis = SimpleNamespace(
        text=raw,
        y_sequence=SimpleNamespace(tolist=lambda: [1]),
        timestamp={"word": [], "segment": [], "char": []},
    )
    asr = runtime(CANARY)
    asr._runtime = SimpleNamespace(
        transcribe=lambda *args, **kwargs: [hypothesis],
        decoding=SimpleNamespace(decode_ids_to_str=lambda token_ids: raw),
    )

    result = asr.transcribe(Path("silence.wav"))

    assert result == adapters.Transcript("", ())
    assert validate_prediction(result.text, result.segments, 1.0, 0.1) == []


def test_device_map_alias_does_not_look_like_device_splitting():
    model = Model()
    model.weights[0].device = "cuda:0"
    model.hf_device_map = {"encoder": "cuda", "decoder": 0}
    adapters._assert_device(model, "cuda")


def test_native_transcription_errors_are_not_retried_or_swallowed():
    asr = runtime(PARAKEET)
    calls = []

    def failure(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("native decoder failure")

    asr._runtime = SimpleNamespace(transcribe=failure)
    with pytest.raises(RuntimeError, match="native decoder failure"):
        asr.transcribe(Path("audio.wav"))
    assert calls == [1]


def test_close_releases_only_after_dropping_model_references(monkeypatch):
    asr = runtime(MOSS)
    asr._runtime = (Model(), object(), {}, "cpu", "torch.float32")
    calls = []
    monkeypatch.setattr(
        adapters, "release_runtime_memory", lambda: calls.append(asr._runtime)
    )
    asr.close()
    assert calls == [None]
    assert asr.placement_models() == ()
