"""Self-contained Markdown reporting without inventing a weighted score."""

from __future__ import annotations

import json
from pathlib import Path

from edumind.common.artifacts import atomic_write_text


def render_report(summary_path: Path, output_path: Path | None = None) -> Path:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    output = output_path or summary_path.with_name("report.md")
    plan = payload["plan"]
    lines = [
        f"# {plan['suite']} / {plan['stage']} benchmark",
        "",
        f"Run ID: `{payload['run_id']}`  ",
        f"Profile: `{plan['profile']}`  ",
        f"Dataset: `{plan['dataset']}`  ",
        f"Authoritative: `{payload.get('authoritative', False)}`",
        "",
        "## Pareto candidates",
        "",
    ]
    pareto = payload.get("pareto_candidates", [])
    lines.extend([f"- `{name}`" for name in pareto] or ["No candidate passed successfully."])
    gate_failures = payload.get("gate_failures", {})
    if gate_failures:
        lines.extend(["", "## Hard-gate rejections", ""])
        for candidate, failures in sorted(gate_failures.items()):
            lines.append(f"- `{candidate}`: {', '.join(failures)}")
    lines.extend(["", "## Candidate results", ""])
    for candidate in payload.get("candidates", []):
        lines.extend([f"### {candidate['candidate']}", "", f"Status: `{candidate['status']}`", ""])
        if candidate.get("error"):
            lines.append(f"Error: {candidate['error']}")
        for name, value in sorted(candidate.get("metrics", {}).items()):
            interval = candidate.get("intervals", {}).get(name, {})
            lines.append(
                f"- {name}: {value:.6f} "
                f"(95% CI {interval.get('lower', value):.6f}–"
                f"{interval.get('upper', value):.6f})"
            )
        for name, value in sorted(candidate.get("operational", {}).items()):
            lines.append(f"- operational {name}: {value:.6f}")
        lines.append("")
    lines.extend(
        [
            "## Interpretation boundary",
            "",
            "Smoke runs validate contracts and execution only. They do not support "
            "quality recommendations. Standard/full recommendations require frozen "
            "manifests, successful gates, and the documented human review.",
            "",
        ]
    )
    atomic_write_text(output, "\n".join(lines))
    return output
