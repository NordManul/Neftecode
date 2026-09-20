"""Быстрая проверка поставки без производственных данных."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mas.common import OUTPUTS_DIR, load_cfg, load_refs, to_jsonable, v
from mas.models.blend import Component, Limits, evaluate


def run_smoke(output_dir: Path | None = None) -> dict:
    """Проверяет конфигурацию, источники и безопасный product-only цикл.

    Smoke-проверка намеренно не требует ``data/``, ``cache/`` или калибровки:
    она предназначена для первого запуска на компьютере проверяющего.
    """
    cfg = load_cfg()
    refs = load_refs()
    errors: list[str] = []

    def check_sources(node: object, path: str = "") -> None:
        if isinstance(node, dict):
            if "src" in node:
                if not node.get("ref"):
                    errors.append(f"{path}: отсутствует ref")
                else:
                    source_ids = node["ref"] if isinstance(node["ref"], list) else [node["ref"]]
                    missing = [source_id for source_id in source_ids if source_id not in refs]
                    if missing:
                        errors.append(f"{path}: неизвестные источники {missing}")
            for key, child in node.items():
                check_sources(child, f"{path}.{key}" if path else key)

    check_sources(cfg)

    mode = v(cfg["blending"]["mode"])
    main = {
        "sulfur_upper": 8.0,
        "t95_upper": 345.0,
        "density": 835.0,
        "density_age_h": 2.0,
        "cetane": 53.0,
        "cetane_age_h": 2.0,
    }
    component = Component("ГО ДТ (24-2000)", main["sulfur_upper"], main["density"],
                          main["cetane"], main["t95_upper"], -8.0, 100000.0)
    limits = Limits(v(cfg["spec"]["sulfur_max_mgkg"]), v(cfg["spec"]["t95_max_c"]),
                    v(cfg["spec"]["cetane_min"]), None, tuple(v(cfg["spec"]["d15_range"])))
    table = evaluate([component], [], np.ones((1, 1)), np.zeros((1, 0)),
                     margins={"T95": 0, "CFPP": 0, "cetane": 0},
                     batch_t=1000.0, limits=limits)
    row = table.iloc[0]
    if abs(sum(row["shares"].values()) - 1.0) > 1e-9:
        errors.append("доли product_only не составляют 100%")
    sulfur_limit = v(cfg["spec"]["sulfur_max_mgkg"])
    if row["S"] > sulfur_limit:
        errors.append("безопасный smoke-сценарий нарушает ограничение по сере")

    report = {
        "status": "PASS" if not errors else "FAIL",
        "checks": {
            "config_loaded": True,
            "source_registry_loaded": bool(refs),
            "configured_default_blending_mode": mode,
            "product_only_recipe": to_jsonable(row["shares"]),
            "hard_sulfur_limit_mgkg": sulfur_limit,
            "evaluated_sulfur_mgkg": row["S"],
        },
        "errors": errors,
    }
    target = output_dir or OUTPUTS_DIR / "smoke"
    target.mkdir(parents=True, exist_ok=True)
    (target / "smoke_report.json").write_text(
        json.dumps(to_jsonable(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if errors:
        raise RuntimeError("Smoke-проверка не пройдена: " + "; ".join(errors))
    return report
