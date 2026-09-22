"""Strict primitives shared by versioned benchmark protocols."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from edumind.common.artifacts import stable_hash


@dataclass(frozen=True)
class ExecutionProfile:
    warmups: int
    repetitions: int
    bootstrap_resamples: int
    device: str
    dtype: str
    batch_size: int
    hardware_required: bool
    devices: tuple[str, ...] = ()
    device_dtypes: Mapping[str, str] | None = None

    def dtype_for(self, device: str) -> str:
        return (
            str(self.device_dtypes[device])
            if self.device_dtypes and device in self.device_dtypes
            else self.dtype
        )


@dataclass(frozen=True)
class ProtocolMetadata:
    name: str
    source_path: Path
    schema_version: int
    version: str
    checksum: str
    resolved: Mapping[str, object]
    seed: int
    profiles: Mapping[str, ExecutionProfile]

    def profile(self, name: str) -> ExecutionProfile:
        try:
            return self.profiles[name]
        except KeyError as exc:
            raise ValueError(f"{self.name} protocol has no profile {name!r}") from exc

    def artifact_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "protocol_version": self.version,
            "checksum": self.checksum,
            "resolved": plain(self.resolved),
        }

    def worker_payload(self) -> dict[str, object]:
        return self.artifact_payload()

    def validate_worker_payload(self, value: object) -> None:
        payload = mapping(value, f"{self.name} worker protocol")
        if payload.get("protocol_version") != self.version:
            raise ValueError(f"{self.name} protocol version mismatch")
        if payload.get("checksum") != self.checksum:
            raise ValueError(f"{self.name} protocol checksum mismatch")
        resolved = payload.get("resolved")
        if stable_hash(plain(resolved)) != self.checksum:
            raise ValueError(f"{self.name} resolved protocol checksum mismatch")


def load_yaml(path: Path, label: str) -> dict[str, object]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing {label} protocol: {path}") from exc
    return mapping(payload, f"{label} protocol root")


def metadata(
    name: str,
    path: Path,
    root: Mapping[str, object],
    *,
    profiles: Mapping[str, ExecutionProfile],
) -> ProtocolMetadata:
    schema = integer(root.get("schema_version"), "schema_version", minimum=1)
    if schema != 1:
        raise ValueError(f"{name} protocol must use schema_version 1")
    version = string(root.get("protocol_version"), "protocol_version")
    seed = integer(root.get("seed"), "seed", minimum=0)
    resolved = plain(root)
    return ProtocolMetadata(
        name=name,
        source_path=path.resolve(),
        schema_version=schema,
        version=version,
        checksum=stable_hash(resolved),
        resolved=resolved,
        seed=seed,
        profiles=dict(profiles),
    )


def execution_profiles(
    value: object,
    *,
    names: Sequence[str],
    label: str = "profiles",
) -> dict[str, ExecutionProfile]:
    root = strict_object(value, label, set(names))
    return {name: execution_profile(root[name], f"{label}.{name}") for name in names}


def execution_profile(value: object, label: str) -> ExecutionProfile:
    raw = mapping(value, label)
    required = {
        "warmups",
        "repetitions",
        "bootstrap_resamples",
        "device",
        "dtype",
        "batch_size",
        "hardware_required",
    }
    missing = sorted(required - set(raw))
    unknown = sorted(set(raw) - required - {"devices", "device_dtypes"})
    if missing or unknown:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unknown " + ", ".join(unknown))
        raise ValueError(f"{label}: {'; '.join(details)}")
    payload = raw
    devices = tuple(
        choice(item, f"{label}.devices[{index}]", {"cpu", "cuda"})
        for index, item in enumerate(
            sequence(payload.get("devices", []), f"{label}.devices")
        )
    )
    if devices and len(set(devices)) != len(devices):
        raise ValueError(f"{label}.devices must be unique")
    raw_dtypes = mapping(payload.get("device_dtypes", {}), f"{label}.device_dtypes")
    device_dtypes = {
        choice(device, f"{label}.device_dtypes device", {"cpu", "cuda"}): choice(
            dtype,
            f"{label}.device_dtypes.{device}",
            {"float32", "float16", "bfloat16", "auto"},
        )
        for device, dtype in raw_dtypes.items()
    }
    profile = ExecutionProfile(
        warmups=integer(payload["warmups"], f"{label}.warmups", minimum=0),
        repetitions=integer(payload["repetitions"], f"{label}.repetitions", minimum=1),
        bootstrap_resamples=integer(
            payload["bootstrap_resamples"],
            f"{label}.bootstrap_resamples",
            minimum=0,
        ),
        device=choice(payload["device"], f"{label}.device", {"cpu", "cuda"}),
        dtype=choice(
            payload["dtype"],
            f"{label}.dtype",
            {"float32", "float16", "bfloat16", "auto"},
        ),
        batch_size=integer(payload["batch_size"], f"{label}.batch_size", minimum=1),
        hardware_required=boolean(
            payload["hardware_required"], f"{label}.hardware_required"
        ),
        devices=devices,
        device_dtypes=device_dtypes or None,
    )
    if profile.hardware_required and profile.device != "cuda":
        raise ValueError(f"{label} hardware-required profile must use CUDA")
    if profile.devices and profile.device not in profile.devices:
        raise ValueError(f"{label}.device must be included in {label}.devices")
    if device_dtypes:
        if set(device_dtypes) != set(profile.devices):
            raise ValueError(
                f"{label}.device_dtypes must define every declared device exactly"
            )
        if profile.dtype_for(profile.device) != profile.dtype:
            raise ValueError(f"{label}.dtype must match the primary device dtype")
    return profile


def validate_execution(
    protocol: ProtocolMetadata,
    profile_name: str,
    *,
    seed: int,
    warmups: int,
    repetitions: int,
    bootstrap_resamples: int,
    device: str,
    dtype: str,
    batch_size: int,
) -> None:
    expected = protocol.profile(profile_name)
    observed_values = (
        seed,
        warmups,
        repetitions,
        bootstrap_resamples,
        batch_size,
    )
    expected_values = (
        protocol.seed,
        expected.warmups,
        expected.repetitions,
        expected.bootstrap_resamples,
        expected.batch_size,
    )
    if observed_values != expected_values:
        raise ValueError(f"Benchmark plan does not match the {protocol.name} protocol")
    if expected.hardware_required and (device, dtype) != (
        expected.device,
        expected.dtype,
    ):
        raise ValueError(
            f"Benchmark hardware does not match the {protocol.name} protocol"
        )


def strict_object(value: object, label: str, fields: set[str]) -> dict[str, object]:
    payload = mapping(value, label)
    missing = sorted(fields - set(payload))
    unknown = sorted(set(payload) - fields)
    if missing or unknown:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unknown " + ", ".join(unknown))
        raise ValueError(f"{label}: {'; '.join(details)}")
    return payload


def mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return {str(key): item for key, item in value.items()}


def sequence(value: object, label: str) -> list[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a list")
    return list(value)


def string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def choice(value: object, label: str, choices: set[str]) -> str:
    result = string(value, label)
    if result not in choices:
        raise ValueError(f"{label} must be one of: {', '.join(sorted(choices))}")
    return result


def boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean")
    return value


def integer(
    value: object,
    label: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{label} must be <= {maximum}")
    return value


def number(
    value: object,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_exclusive: bool = False,
    maximum_exclusive: bool = False,
) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    if minimum is not None and (
        result < minimum or (minimum_exclusive and result == minimum)
    ):
        raise ValueError(f"{label} is below its valid range")
    if maximum is not None and (
        result > maximum or (maximum_exclusive and result == maximum)
    ):
        raise ValueError(f"{label} exceeds its valid range")
    return result


def increasing_integers(
    value: object, label: str, *, minimum: int = 1
) -> tuple[int, ...]:
    values = tuple(
        integer(item, f"{label}[{index}]", minimum=minimum)
        for index, item in enumerate(sequence(value, label))
    )
    if not values or tuple(sorted(set(values))) != values:
        raise ValueError(f"{label} must be non-empty, unique, and increasing")
    return values


def plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [plain(item) for item in value]
    return value
