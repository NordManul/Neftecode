"""Сбор данных для статического дашборда - один файл, светлая/тёмная тема."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from mas.common import OUTPUTS_DIR, ROOT, node_refs, to_jsonable


def _load_json(path, default=None):
    return json.load(open(path, encoding="utf-8")) if path.exists() else default


def _assumptions(cfg: dict) -> list[dict]:
    rows = []

    def walk(node, path):
        if isinstance(node, dict) and "src" in node:
            value = node["value"] if "value" in node else {k: x for k, x in node.items() if k not in ("src", "note", "ref")}
            text = "; ".join(f"{k}={x}" for k, x in value.items()) if isinstance(value, dict) else str(value)
            rows.append({"path": path, "value": text, "src": node["src"], "note": node.get("note", ""),
                         "refs": [{k: r.get(k) for k in ("kind", "short", "url")} for r in node_refs(node)]})
        elif isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{path}.{k}" if path else k)

    walk(cfg, "")
    return rows


def build_dashboard() -> str:
    from mas.common import load_cfg
    calib = _load_json(ROOT / "config" / "calibrated.json", {})
    val = _load_json(OUTPUTS_DIR / "validation.json", {})
    bt = _load_json(OUTPUTS_DIR / "backtest_summary.json", {})
    audit_path = OUTPUTS_DIR / "data_report" / "tag_audit.csv"
    audit = pd.read_csv(audit_path) if audit_path.exists() else pd.DataFrame()

    sulfur_estimators = [{"name": k, "mae": round(v_["mae"], 3) if v_["mae"] else None,
                          "auc": round(v_["auc"], 3) if v_["auc"] else None, "n": v_["n"]}
                         for k, v_ in (val.get("sulfur", {}).get("estimators", {}) or {}).items()
                         if isinstance(v_, dict)]
    failure_reasons = [{"reason": k, "count": v_} for k, v_ in (bt.get("failure_reasons", {}) or {}).items()]
    tag_audit_rows = (audit[audit["verdict"] == "противоречие"][["tag", "unit", "verdict", "problems"]]
                       .to_dict("records") if len(audit) else [])
    vak_path = OUTPUTS_DIR / "vak_regulation.csv"
    vak = pd.read_csv(vak_path) if vak_path.exists() else pd.DataFrame()

    data = {
        "calculated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "backtest": bt, "sulfur_estimators": sulfur_estimators, "failure_reasons": failure_reasons,
        "tag_audit": tag_audit_rows, "assumptions": _assumptions(load_cfg()),
        "median_nis": round(_load_json(OUTPUTS_DIR / "calibration_report.json", {}).get("sulfur", {}).get("median_nis_train", 0), 4),
        "vak_fit": int((vak["verdict"] == "пригодна").sum()) if len(vak) else 0, "vak_total": len(vak),
    }

    template_path = ROOT / "mas" / "report" / "dashboard_template.html"
    html = template_path.read_text(encoding="utf-8")
    html = html.replace("/*__DATA__*/", json.dumps(to_jsonable(data), ensure_ascii=False))
    out_path = OUTPUTS_DIR / "dashboard.html"
    out_path.write_text(html, encoding="utf-8")
    return html
