from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")


def _documentation_files() -> list[Path]:
    root_pages = [ROOT / name for name in ("README.md", "CONTRIBUTING.md", "CHANGELOG.md")]
    return root_pages + sorted((ROOT / "docs").rglob("*.md"))


def test_local_markdown_links_resolve() -> None:
    broken: list[str] = []
    for document in _documentation_files():
        text = document.read_text(encoding="utf-8")
        for match in LINK.finditer(text):
            raw_target = match.group(1).strip()
            if raw_target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            path_text = unquote(raw_target.split("#", 1)[0]).strip("<>")
            if not path_text:
                continue
            target = (document.parent / path_text).resolve()
            if not target.exists():
                line = text.count("\n", 0, match.start()) + 1
                broken.append(f"{document.relative_to(ROOT)}:{line} -> {raw_target}")
    assert not broken, "Broken local Markdown links:\n" + "\n".join(broken)
