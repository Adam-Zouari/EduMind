"""Typed boundaries between input, generator and infrastructure failures."""


class GenerationInputError(ValueError):
    """A native input failed integrity before the model call."""

    failure_stage = "input_validation"


class GenerationInfrastructureError(RuntimeError):
    """Execution was interrupted outside the model's request contract."""

    failure_stage = "infrastructure"


def failure_evidence(exc, stage):
    return {
        "type": type(exc).__name__,
        "stage": getattr(exc, "failure_stage", stage),
        "message": str(exc),
    }
