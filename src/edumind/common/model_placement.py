"""Framework-agnostic checks for whole-model device placement."""

from __future__ import annotations

from collections.abc import Iterable, Mapping


def inspect_model_placement(
    *models: object,
    expected_device: str = "cuda",
    strict: bool = True,
) -> dict[str, object]:
    """Report whether every discoverable model weight is on one device.

    CPU tokenizers, media decoders, and ordinary preprocessing buffers are not model
    offloading and are intentionally outside this check.
    """

    devices: set[str] = set()
    device_maps: list[dict[str, str]] = []
    offload_hooks: list[str] = []
    tensor_count = 0
    for root in models:
        model = root
        if not callable(getattr(model, "parameters", None)):
            model = getattr(root, "model", root)
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
    forbidden_devices = {"disk", "meta"}
    if expected_device != "cpu":
        forbidden_devices.add("cpu")
    offloaded = bool(devices & forbidden_devices) or bool(offload_hooks)
    verified = bool(tensor_count or device_maps)
    valid = verified and not offloaded and devices <= {expected_device}
    status = "qualified"
    if offloaded:
        status = "offload_detected"
    elif strict and not verified:
        status = "placement_unverifiable"
    elif strict and not valid:
        status = "wrong_device"
    return {
        "status": status,
        "expected_device": expected_device,
        "observed_devices": sorted(devices),
        "inspected_tensor_count": tensor_count,
        "device_maps": device_maps,
        "offload_hooks": sorted(set(offload_hooks)),
        "verified": verified,
    }


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
    if text.isdigit() or text.startswith("cuda"):
        return "cuda"
    return text.split(":", 1)[0]
