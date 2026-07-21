"""Entry-point helper used by each extraction stage's run.py."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.benchmarks.common.arguments import parser, resolved_candidates
from experiments.benchmarks.extraction.runner import run

ASR_SETTINGS = {
    "openai-whisper-small-en": ("openai-whisper", "small.en", "float16"),
    "distil-whisper-large-v3.5": (
        "transformers-asr",
        "distil-whisper/distil-large-v3.5",
        "float16",
    ),
    "parakeet-tdt-0.6b-v3": (
        "transformers-asr",
        "nvidia/parakeet-tdt-0.6b-v3",
        "bfloat16",
    ),
    "canary-qwen-2.5b": ("nemo-canary", "nvidia/canary-qwen-2.5b", "bfloat16"),
}


def main(stage: str, directory: Path) -> int:
    argument_parser = parser(f"Benchmark {stage} extraction")
    argument_parser.add_argument(
        "--image-summary",
        type=Path,
        help="summary.json containing exactly one approved image OCR candidate",
    )
    argument_parser.add_argument(
        "--audio-summary",
        type=Path,
        help="summary.json containing exactly one approved audio ASR candidate",
    )
    arguments = argument_parser.parse_args()
    if arguments.profile != "smoke" and stage in {"pdf", "routing", "video"}:
        if arguments.image_summary is None:
            raise ValueError(
                f"{stage} {arguments.profile} requires --image-summary so OCR is frozen"
            )
    if arguments.profile != "smoke" and stage == "video" and arguments.audio_summary is None:
        raise ValueError(
            f"video {arguments.profile} requires --audio-summary so ASR is frozen"
        )
    candidates = resolved_candidates(
        directory / "candidates.yaml", arguments.profile, arguments.shortlist
    )
    if stage == "audio" and arguments.profile == "full":
        candidates = tuple(
            expanded
            for candidate in candidates
            for expanded in (
                (candidate,)
                if "|vad=" in candidate
                else (f"{candidate}|vad=off", f"{candidate}|vad=on")
            )
        )
    result = run(
        stage,
        arguments.profile,
        candidates,
        manifest_path=arguments.manifest,
        no_mlflow=arguments.no_mlflow,
        component_options=_component_options(arguments.image_summary, arguments.audio_summary),
    )
    print(json.dumps({"run_id": result.run_id, "artifacts": str(result.artifact_directory)}, indent=2))
    return 0 if all(candidate.status == "success" for candidate in result.candidates) else 2


def _component_options(image_summary: Path | None, audio_summary: Path | None) -> dict[str, object]:
    options: dict[str, object] = {}
    if image_summary:
        image = _one_selection(image_summary)
        engine, separator, preprocessing = image.partition("|")
        options.update(
            {
                "image_engine": engine,
                "image_preprocessing": preprocessing if separator else "document",
            }
        )
    if audio_summary:
        audio_selection = _one_selection(audio_summary)
        audio, _, factor = audio_selection.partition("|")
        if factor:
            key, separator, value = factor.partition("=")
            if key != "vad" or not separator or value not in {"on", "off"}:
                raise ValueError(f"Unsupported audio factor: {factor}")
            options["vad"] = value == "on"
        if audio.startswith("faster-whisper-"):
            value = audio.removeprefix("faster-whisper-")
            model, compute_type = value.rsplit("-", 1)
            settings = (
                "faster-whisper",
                model if model == "turbo" else f"{model}.en",
                compute_type,
            )
        else:
            settings = ASR_SETTINGS.get(audio)
            if settings is None:
                raise ValueError(f"Unsupported audio selection: {audio}")
        options.update(
            {
                "audio_candidate": audio,
                "audio_engine": settings[0],
                "audio_model": settings[1],
                "audio_compute_type": settings[2],
            }
        )
    return options


def _one_selection(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("pareto_candidates")
    if not isinstance(values, list) or len(values) != 1:
        raise ValueError(f"{path} must contain exactly one explicitly approved candidate")
    return str(values[0])
