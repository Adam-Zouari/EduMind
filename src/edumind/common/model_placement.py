"""Framework-agnostic checks for whole-model device placement."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from os import PathLike
from types import ModuleType


def inspect_model_placement(
    *models: object,
    expected_device: str = "cuda",
) -> dict[str, object]:
    """Report whether every discoverable model weight is on one device.

    CPU tokenizers, media decoders, and ordinary preprocessing buffers are not model
    offloading and are intentionally outside this check.
    """

    devices: set[str] = set()
    device_maps: list[dict[str, str]] = []
    offload_hooks: list[str] = []
    provider_sets: list[list[str]] = []
    tensor_count = 0
    inspected_objects = 0
    for model in _walk_runtime_objects(models):
        inspected_objects += 1
        device_map = getattr(model, "hf_device_map", None)
        if isinstance(device_map, Mapping):
            normalized = {str(key): str(value) for key, value in device_map.items()}
            device_maps.append(normalized)
            devices.update(_device_type(value) for value in normalized.values())
        for tensor in _model_tensors(model):
            device = getattr(tensor, "device", None)
            if device is not None:
                devices.add(_device_type(device))
                tensor_count += 1
        named_modules = getattr(model, "named_modules", None)
        if callable(named_modules):
            try:
                for name, module in named_modules():
                    hook = getattr(module, "_hf_hook", None)
                    if hook is not None and (
                        bool(getattr(hook, "offload", False))
                        or getattr(hook, "weights_map", None) is not None
                    ):
                        offload_hooks.append(name or "<root>")
            except (RuntimeError, TypeError):
                pass
        get_providers = getattr(model, "get_providers", None)
        if callable(get_providers):
            try:
                providers = [str(value) for value in get_providers()]
            except (RuntimeError, TypeError):
                providers = []
            if providers:
                provider_sets.append(providers)
                primary = providers[0].casefold()
                devices.add("cuda" if "cuda" in primary else "cpu")
    forbidden_devices = {"disk", "meta"}
    if expected_device != "cpu":
        forbidden_devices.add("cpu")
    offloaded = bool(devices & forbidden_devices) or bool(offload_hooks)
    verified = bool(tensor_count or device_maps or provider_sets)
    valid = verified and not offloaded and devices <= {expected_device}
    status = "qualified"
    if offloaded:
        status = "offload_detected"
    elif not verified:
        status = "placement_unverifiable"
    elif not valid:
        status = "wrong_device"
    return {
        "status": status,
        "expected_device": expected_device,
        "observed_devices": sorted(devices),
        "inspected_tensor_count": tensor_count,
        "device_maps": device_maps,
        "offload_hooks": sorted(set(offload_hooks)),
        "provider_sets": provider_sets,
        "inspected_object_count": inspected_objects,
        "verified": verified,
    }


def _walk_runtime_objects(roots: Iterable[object]) -> Iterable[object]:
    """Walk loaded backend state without depending on one framework's internals."""

    pending = [(root, 0) for root in roots]
    visited: set[int] = set()
    while pending:
        value, depth = pending.pop()
        if value is None or isinstance(
            value, (str, bytes, int, float, bool, PathLike, ModuleType, type)
        ):
            continue
        identity = id(value)
        if identity in visited:
            continue
        visited.add(identity)
        yield value
        if depth >= 8 or len(visited) >= 10_000:
            continue
        if isinstance(value, Mapping):
            children = value.values()
        elif isinstance(value, (list, tuple, set, frozenset)):
            children = value
        else:
            attributes = getattr(value, "__dict__", None)
            children = attributes.values() if isinstance(attributes, Mapping) else ()
        pending.extend((child, depth + 1) for child in children)


def _model_tensors(model: object) -> Iterable[object]:
    for accessor in ("parameters", "buffers"):
        values = getattr(model, accessor, None)
        if not callable(values):
            continue
        try:
            yield from values()
        except (RuntimeError, TypeError):
            continue


def _device_type(value: object) -> str:
    text = str(value).casefold()
    if text.isdigit() or text.startswith("cuda") or "gpu" in text:
        return "cuda"
    return text.split(":", 1)[0]
