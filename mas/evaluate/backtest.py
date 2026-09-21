"""Бэктест по отложенному периоду частями, с продолжением.

Каждые `step_h` часов; части по CHUNK_MONTHS месяцев сохраняются отдельно и не пересчитываются
при повторном запуске - полный прогон дольше типичного лимита одного шага CI. Сохранённые части
используются, только если не менялись код, конфигурация, калибровка и кэш состояния (подпись).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from mas.agents.orchestrator import Orchestrator
from mas.common import CACHE_DIR, CONFIG_DIR, OUTPUTS_DIR, action_threshold, load_cfg, to_jsonable, v

CHUNK_MONTHS = 4


def _signature(step_h: int) -> str:
    """Хэш всего, от чего зависят решения: код, конфигурация, калибровка, кэш рядов состояния, шаг."""
    digest = hashlib.sha256(str(step_h).encode())
    files = sorted(Path(__file__).resolve().parents[1].rglob("*.py")) + sorted(CONFIG_DIR.glob("*.*")) + sorted(CACHE_DIR.glob("*.parquet"))
    for f in files:
        digest.update(f.name.encode())
        digest.update(f.read_bytes())
    return digest.hexdigest()


def _row_from_rec(rec) -> dict:
    chosen = rec.chosen or {}
    b4 = rec.blocks.get("4. Ожидаемый эффект", {}) or {}
    p_s_before = (b4.get("P(S>10) c/без действия") or (None, None))[1]
    p_t95_before = (b4.get("P(T95>360) c/без действия") or (None, None))[1]
    return {
        "t": rec.t, "status": rec.status, "dT5": chosen.get("dT5"), "rF9": chosen.get("rF9"),
        "dF32": chosen.get("dF32"), "P_exceed": chosen.get("P_exceed"), "P_t95": chosen.get("P_t95"),
        "P_exceed_before": p_s_before, "P_t95_before": p_t95_before,
        "P_now": (rec.blocks.get("2. Проблема / риск") or {}).get("P(S>10) сейчас"),
        "score": chosen.get("score"), "quality_margin": chosen.get("quality_margin"),
        "escalation": "ЭСКАЛАЦИЯ" in rec.status, "explanation": rec.blocks.get("7. Объяснение", ""),
    }


def _failure_reason(explanation: str) -> str | None:
    if "Пусковой режим" in explanation or "дождаться установившегося" in explanation:
        return "пусковой режим"
    if "Переходный режим" in explanation:
        return "переходный режим"
    if "вне области применимости" in explanation:
        return "вне области модели"
    if "Нет допустимого варианта" in explanation:
        return "нет допустимого варианта"
    if "Повторная проверка не пройдена" in explanation:
        return "повторная проверка"
    return None


def _episodes(action_mask: pd.Series) -> tuple[float, float]:
    grp = (~action_mask).cumsum()
    lengths = action_mask.groupby(grp).sum()
    lengths = lengths[lengths > 0]
    months = (action_mask.index.max() - action_mask.index.min()).days / 30.437
    return (len(lengths) / months if months > 0 else 0.0), (float(lengths.median()) if len(lengths) else 0.0)


DETECTION_LEADS_H = (1.0, 4.0, 7.0)


def _detection(cycles: pd.DataFrame, lab: pd.DataFrame, limit: float, threshold: float) -> list[dict]:
    """Сигнал о риске по сере на рабочем цикле (расчётная вероятность превышения выше `threshold`) против результата анализа ЛИМС
    (`lab`: `time`, `value`), отобранного не раньше чем через `lead` часов после цикла. Упреждение 1 ч - лучшее, что возможно при
    шаге цикла, 4 ч - горизонт прогноза, 7 ч - конец окна проявления эффекта действия (2-8 ч)."""
    from mas.contracts import STATUS_OBSERVE
    lab = lab.sort_values("time")
    cyc = cycles.assign(t=pd.to_datetime(cycles["t"])).sort_values("t").reset_index(drop=True)
    working = ~cyc["status"].str.replace(r" \+.*", "", regex=True).eq(STATUS_OBSERVE)
    flagged = (pd.to_numeric(cyc["P_now"], errors="coerce").fillna(0) > threshold).to_numpy()
    recommended = cyc["status"].str.startswith("КОРРЕКТИРУЮЩЕЕ").to_numpy()
    out = []
    for lead in DETECTION_LEADS_H:
        pos = cyc["t"].searchsorted(lab["time"] - pd.Timedelta(hours=lead), side="right") - 1
        keep = pos >= 0
        pos, y = pos[keep], (lab["value"].to_numpy()[keep] > limit)
        ok = working.to_numpy()[pos]
        pos, y = pos[ok], y[ok]
        al = flagged[pos]
        if not len(y):
            out.append({"lead_h": lead, "n_analyses": 0, "share_flagged": None, "precision": None, "recall": None,
                        "share_flagged_with_recommendation": None})
            continue
        out.append({"lead_h": lead, "n_analyses": int(len(y)), "share_flagged": float(al.mean()),
                    "precision": float(y[al].mean()) if al.any() else None, "recall": float(al[y].mean()) if y.any() else None,
                    "share_flagged_with_recommendation": float(recommended[pos][al].mean()) if al.any() else None})
    return out


def _test_lab_sulfur(cfg: dict) -> pd.DataFrame:
    lims = pd.read_parquet(CACHE_DIR / "lims.parquet")
    return lims[(lims["point"] == "HT2") & (lims["parameter"] == "Mg.Sulfur") & (lims["time"] >= pd.Timestamp(cfg["periods"]["train_end"]))]


def _summarize(cycles: pd.DataFrame) -> dict:
    from mas.contracts import STATUS_CORRECTIVE, STATUS_NO_ACTION, STATUS_NO_RELIABLE, STATUS_OBSERVE
    n = len(cycles)
    status_base = cycles["status"].str.replace(r" \+.*", "", regex=True)
    dist = {s: float((status_base == s).mean()) for s in
            (STATUS_NO_ACTION, STATUS_CORRECTIVE, STATUS_NO_RELIABLE, STATUS_OBSERVE)}
    working = cycles[status_base != STATUS_OBSERVE]
    # «Драйвер» - риск БЕЗ действия выше порога запуска действия (правила 4-5),
    # а не риск ПОСЛЕ выбранного варианта (тот всегда низкий - иначе вариант не прошёл бы проверку).
    cfg = load_cfg()
    driver_sulfur = float((working["P_exceed_before"].fillna(0) > action_threshold(cfg, "sulfur")).mean()) if len(working) else 0.0
    driver_t95 = float((working["P_t95_before"].fillna(0) > action_threshold(cfg, "t95")).mean()) if len(working) else 0.0

    action_mask = status_base == STATUS_CORRECTIVE
    ep_per_month, ep_median_len = _episodes(action_mask.set_axis(cycles["t"])) if n else (0.0, 0.0)

    failures = cycles.loc[status_base == STATUS_NO_RELIABLE, "explanation"].map(_failure_reason)
    failure_counts = failures.value_counts().to_dict()

    return {
        "n_cycles": n, "status_distribution": dist, "escalation_share": float(cycles["escalation"].mean()) if n else 0.0,
        "driver_share_of_working": {"сера": driver_sulfur, "T95": driver_t95},
        "episodes_per_month": ep_per_month, "episode_median_length": ep_median_len,
        "failure_reasons": failure_counts, "detection": _detection(cycles, _test_lab_sulfur(cfg), v(cfg["spec"]["sulfur_max_mgkg"]),
                                                                    action_threshold(cfg, "sulfur")),
    }


def run_backtest(step_h: int = 6) -> tuple[pd.DataFrame, dict]:
    cfg = load_cfg()
    start, end = pd.Timestamp(cfg["periods"]["train_end"]), pd.Timestamp(cfg["periods"]["test_end"])
    times = pd.date_range(start, end, freq=f"{step_h}h")

    parts_dir = OUTPUTS_DIR / "backtest_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    signature_path = parts_dir / "signature.txt"
    signature = _signature(step_h)
    if not signature_path.exists() or signature_path.read_text() != signature:
        for stale in parts_dir.glob("part_*.parquet"):
            stale.unlink()
        signature_path.write_text(signature)
    orch = Orchestrator()
    edges = pd.date_range(start, end, freq=f"{CHUNK_MONTHS}MS")
    if len(edges) == 0 or edges[-1] < end:
        edges = edges.append(pd.DatetimeIndex([end]))
    if edges[0] > start:
        edges = pd.DatetimeIndex([start]).append(edges)

    parts = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        part_path = parts_dir / f"part_{i:02d}.parquet"
        if part_path.exists():
            parts.append(pd.read_parquet(part_path))
            continue
        chunk_times = times[(times >= lo) & (times < hi)]
        rows = [_row_from_rec(orch.cycle(t, save=False)) for t in chunk_times]
        part_df = pd.DataFrame(rows)
        part_df.to_parquet(part_path)
        parts.append(part_df)

    cycles = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    cycles.to_csv(OUTPUTS_DIR / "backtest_cycles.csv", index=False)
    summary = _summarize(cycles)
    with open(OUTPUTS_DIR / "backtest_summary.json", "w", encoding="utf-8") as f:
        json.dump(to_jsonable(summary), f, ensure_ascii=False, indent=2)
    return cycles, summary
