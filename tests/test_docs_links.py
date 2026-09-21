"""Ссылки в README и документах ведут на существующие файлы и заголовки (проверяющий открывает репозиторий впервые)."""
from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote

import pytest

ROOT = Path(__file__).resolve().parents[1]
MARKDOWN = [ROOT / "README.md", *sorted((ROOT / "docs").glob("0*.md"))]
LINK = re.compile(r"(?<!\!)\[[^\]]*\]\(([^)\s]+)\)|!\[[^\]]*\]\(([^)\s]+)\)")


def _slug(heading: str) -> str:
    """Якорь заголовка по правилам GitHub: строчные буквы, без знаков препинания, пробелы - дефисы."""
    text = re.sub(r"[`*_]", "", heading.strip().lower())
    return re.sub(r"[^\w\- ]", "", text).replace(" ", "-")


def _anchors(path: Path) -> set[str]:
    out = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^#{1,6}\s+(.*)$", line)
        if m:
            out.add(_slug(m.group(1)))
    return out


def _links(path: Path):
    text = re.sub(r"```.*?```", "", path.read_text(encoding="utf-8"), flags=re.S)
    for m in LINK.finditer(text):
        yield m.group(1) or m.group(2)


@pytest.mark.parametrize("path", MARKDOWN, ids=lambda p: p.name)
def test_relative_links_and_anchors_resolve(path):
    broken = []
    for target in _links(path):
        if re.match(r"^(https?:|mailto:)", target):
            continue
        file_part, _, anchor = target.partition("#")
        dest = (path.parent / unquote(file_part)).resolve() if file_part else path
        if not dest.exists():
            broken.append(target)
        elif anchor and dest.suffix == ".md" and unquote(anchor) not in _anchors(dest):
            broken.append(target)
    assert not broken, f"{path.name}: нет файла или заголовка для ссылок {broken}"


def test_readme_names_the_entry_points_a_reviewer_needs():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    for needed in ("outputs/dashboard.html", "docs/02_ОБОСНОВАНИЕ.md", "docs/03_РЕЗУЛЬТАТЫ.md", "python run.py all --data", "tools/live_viewer/server.py"):
        assert needed in text
