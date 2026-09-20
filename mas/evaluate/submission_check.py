"""Проверка полноты вычисленного комплекта сдачи."""
from __future__ import annotations

import json
from pathlib import Path

from mas.common import OUTPUTS_DIR, CACHE_DIR, CONFIG_DIR


REQUIRED_OUTPUTS = (
    CACHE_DIR / "data_summary.json",
    CACHE_DIR / "thresholds.json",
    CONFIG_DIR / "calibrated.json",
    OUTPUTS_DIR / "validation.json",
    OUTPUTS_DIR / "backtest_summary.json",
    OUTPUTS_DIR / "dashboard.html",
)


def check_submission(output_dir: Path | None = None, project_root: Path | None = None) -> dict:
    root = project_root
    required = REQUIRED_OUTPUTS if root is None else tuple(
        root / path.relative_to(Path.cwd()) for path in REQUIRED_OUTPUTS
    )
    missing = [str(path) for path in required if not path.exists()]
    demo_dir = OUTPUTS_DIR / "demo" if root is None else root / "outputs" / "demo"
    demo_files = sorted(demo_dir.glob("*.json"))
    if len(demo_files) < 10:
        missing.append(f"outputs/demo/*.json: найдено {len(demo_files)} из 10")

    report = {
        "status": "PASS" if not missing else "INCOMPLETE",
        "required_outputs": [str(path) for path in required],
        "demo_json_count": len(demo_files),
        "missing": missing,
        "next_step": (
            "Комплект готов к показу."
            if not missing
            else "Выполните `python run.py all --data <папка>` и повторите проверку."
        ),
    }
    target = output_dir or OUTPUTS_DIR
    target.mkdir(parents=True, exist_ok=True)
    (target / "submission_check.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report
