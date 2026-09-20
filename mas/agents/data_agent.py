"""DataAgent: снимок состояния, полнота/свежесть/согласованность данных (шаги 1-2).

Только кэш и конфигурация - никаких сырых файлов.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mas.bus import MessageBus
from mas.common import CACHE_DIR, lims_age_limit_h, load_calib, load_cfg, startup_window_h, v
from mas.contracts import Issue, Snapshot
from mas.prepare.quality_flags import trailing_run_length
from mas.state.derived import load_derived


def _age_since_ok(reason: pd.Series, step_h: float) -> pd.Series:
    bad = reason != ""
    run_len = trailing_run_length(bad.astype(int))
    return (run_len * step_h).where(bad, 0.0)


class DataAgent:
    def __init__(self, bus: MessageBus) -> None:
        self.cfg, self.calib = load_cfg(), load_calib()
        self.t95_fallback_calib = {"t95_offset": self.calib["t95"]["offset"], "t95_sd_diff": self.calib["t95"]["sd_diff"]}
        self.avt, self.ho = pd.read_parquet(CACHE_DIR / "avt.parquet"), pd.read_parquet(CACHE_DIR / "ho.parquet")
        self.set_derived(load_derived())
        bus.register("DataAgent", "snapshot", lambda t: self.snapshot(t))

    def set_derived(self, d: dict[str, pd.DataFrame]) -> None:
        """Ряды, зависящие от допущений по приборам (флаги, признак работы, анализаторы, ЛИМС, состояние)."""
        self.avt_flags, self.ho_flags, self.running = d["avt_flags"], d["ho_flags"], d["running"]
        self.pak, self.q21, self.lims = d["pak_sulfur"], d["q21_sulfur"], d["lims"]
        self.lims_idx = {k: g.sort_values("available_at") for k, g in self.lims.groupby(["point", "parameter"])}
        self.state_s, self.state_treq = d["state_sulfur"], d["state_treq"]
        self.pak_age, self.q21_age = _age_since_ok(self.pak["reason"], 1 / 6), _age_since_ok(self.q21["reason"], 1 / 6)

    def _asof(self, s: pd.Series, t: pd.Timestamp):
        sl = s.loc[:t]
        return sl.iloc[-1] if len(sl) else np.nan

    def _hourly_means(self, t: pd.Timestamp) -> dict[str, float]:
        out = {}
        for df, tags in ((self.avt, ["F32", "F65", "T33", "T55", "T20", "P22"]),
                         (self.ho, ["T5", "T6", "T11", "P3", "P13", "P8", "F9", "F2"])):
            win = df.loc[t - pd.Timedelta(hours=1):t, tags].mean()
            out.update(win.to_dict())
        density = v(self.cfg["process"]["feed_density_t_m3"])
        out["GOR"] = out["F2"] / (out["F9"] / density) if out.get("F9") else np.nan
        out["F32_lag4h"] = self.avt["F32"].loc[t - pd.Timedelta(hours=5):t - pd.Timedelta(hours=4)].mean()
        return out

    def _point_values(self, t: pd.Timestamp) -> dict[str, float]:
        avt_sl, ho_sl = self.avt.loc[:t], self.ho.loc[:t]
        out = avt_sl.iloc[-1][["F32", "T33"]].to_dict() if len(avt_sl) else {}
        out.update(ho_sl.iloc[-1][["T5", "F9"]].to_dict() if len(ho_sl) else {})
        return out

    def _analyzer_status(self, t: pd.Timestamp) -> dict:
        out = {}
        for name, s, age in (("pak", self.pak, self.pak_age), ("q21", self.q21, self.q21_age)):
            row = s.loc[:t].iloc[-1] if len(s.loc[:t]) else None
            ok = bool(row is not None and row["reason"] == "")
            out[name] = {"raw": float(row["raw"]) if row is not None else np.nan,
                         "reason": str(row["reason"]) if row is not None else "нет данных",
                         "run_len_h": float(row["run_len"]) / 6 if row is not None else np.nan,
                         "age_last_ok_h": float(self._asof(age, t))}
        n_ok = sum(1 for nm in out if out[nm]["reason"] == "")
        out["n_ok"] = n_ok
        out["reason"] = "ok" if n_ok == 2 else ("частично" if n_ok == 1 else "все забракованы")
        return out

    def _sulfur_state(self, t: pd.Timestamp) -> dict[str, float]:
        row = self.state_s.loc[:t].iloc[-1] if len(self.state_s.loc[:t]) else None
        if row is None:
            return {}
        parts = {k: float(row[k]) for k in ("S_slow", "S_fast", "P_ss", "P_ff", "P_sf") if k in row.index}
        return {"S_hat": float(row["S_hat"]), "S_sigma": float(row["S_sigma"]), "bias_pak": float(row["bias_pak"]),
                "bias_q21": float(row["bias_q21"]), "dp_ratio": float(row["dp_ratio"]), **parts}

    def _lims_slice(self, point: str, param: str, t: pd.Timestamp) -> pd.DataFrame:
        g = self.lims_idx.get((point, param))
        if g is None:
            return g
        pos = g["available_at"].to_numpy().searchsorted(np.datetime64(t), side="right")
        return g.iloc[:pos]

    def _lims_field(self, point: str, param: str, t: pd.Timestamp) -> dict | None:
        sub = self._lims_slice(point, param, t)
        if sub is None or sub.empty:
            return None
        row = sub.iloc[-1]
        age_h = (t - row["time"]).total_seconds() / 3600
        return {"value": float(row["value"]), "sample_time": row["time"], "age_h": age_h}

    def _t95_samples(self, t: pd.Timestamp) -> list[dict]:
        from mas.models.t95 import feed_fallback_sample, recent_samples
        product = self.lims_idx.get(("HT2", "95%.T"))
        limit = lims_age_limit_h(self.calib, "HT2", "95%.T")
        samples = recent_samples(product, self.avt["F32"], t, limit) if product is not None else []
        if not samples:
            feed = self._lims_field("HT1", "95%.T", t)
            if feed and feed["age_h"] <= lims_age_limit_h(self.calib, "HT1", "95%.T"):
                f32_sample = self.avt["F32"].loc[feed["sample_time"] - pd.Timedelta(hours=1):feed["sample_time"]].mean()
                samples.append(feed_fallback_sample(feed["value"], feed["age_h"], float(f32_sample), self.t95_fallback_calib))
        return samples

    def _lims_latest(self, t: pd.Timestamp) -> dict[str, dict]:
        keys = [("HT2", "Mg.Sulfur"), ("HT2", "95%.T"), ("HT2", "D15"), ("HT2", "CetaneNumber"),
                ("HT2", "FlashPoint"), ("HT1", "95%.T"), ("HT1", "Mass.Sulfur")]
        out = {}
        for point, param in keys:
            f = self._lims_field(point, param, t)
            if f:
                out[f"{point}:{param}"] = f
        return out

    def _catalyst(self, t: pd.Timestamp) -> dict[str, float]:
        day = self.state_treq.loc[:t]
        if not len(day):
            return {}
        row = day.iloc[-1]
        return {"treq_7d": float(row["treq_7d"]) if pd.notna(row["treq_7d"]) else np.nan}

    def _issues(self, t, ho_running, hours_since_start, values, analyzers, sulfur_state, t95_samples) -> list[Issue]:
        issues: list[Issue] = []
        if not ho_running:
            issues.append(Issue("critical", "телеметрия", "ho_running", "Установка гидроочистки не в работе."))
            return issues
        window = startup_window_h(self.calib)
        if hours_since_start < window:
            issues.append(Issue("critical", "телеметрия", "hours_since_start",
                                 f"Пусковой режим: {hours_since_start:.1f} ч с момента пуска."))
        d_t5 = values.get("T5", np.nan) - self._point_values(t - pd.Timedelta(hours=4)).get("T5", np.nan)
        if pd.notna(d_t5) and abs(d_t5) > self.calib["reliability"]["t5_roc4h_p99"]:
            issues.append(Issue("critical", "телеметрия", "T5", f"Переходный режим: ΔT5 за 4 ч = {d_t5:.1f} °C."))
        for tag, flags in (("T5", self.ho_flags), ("F9", self.ho_flags), ("F32", self.avt_flags)):
            fl = self._asof(flags[tag], t)
            if fl != 0:
                issues.append(Issue("critical", "телеметрия", tag, f"Управляющий тег {tag} недостоверен (флаг {fl})."))
        if analyzers["n_ok"] == 0 and min(analyzers["pak"]["age_last_ok_h"], analyzers["q21"]["age_last_ok_h"]) > window:
            issues.append(Issue("critical", "ПАК/Q21", "sulfur", f"Оба анализатора серы забракованы дольше {window:.0f} ч."))
        if not t95_samples:
            issues.append(Issue("critical", "ЛИМС", "95%.T", "Нет анализа T95 продукта или сырья в пределах допустимого возраста."))
        if sulfur_state.get("S_sigma", 0) > v(self.cfg["sensor_fusion"]["max_sigma_for_decision"]):
            issues.append(Issue("warning", "модель", "S_sigma", f"Неопределённость серы высокая: σ={sulfur_state['S_sigma']:.2f}."))
        return issues

    def snapshot(self, t) -> Snapshot:
        t = pd.Timestamp(t)
        ho_running = bool(self._asof(self.running["ho_running"], t))
        avt_running = bool(self._asof(self.running["avt_running"], t))
        hours_since_start = float(self._asof(self.running["ho_hours_since_start"], t))
        values = self._hourly_means(t)
        values_4h = self._point_values(t - pd.Timedelta(hours=4))
        analyzers = self._analyzer_status(t)
        sulfur_state = self._sulfur_state(t)
        t95_samples = self._t95_samples(t)
        lims_latest = self._lims_latest(t)
        catalyst = self._catalyst(t)
        issues = self._issues(t, ho_running, hours_since_start, values, analyzers, sulfur_state, t95_samples)
        return Snapshot(t=t, ho_running=ho_running, avt_running=avt_running, hours_since_start=hours_since_start,
                         values=values, values_4h_ago=values_4h, analyzers=analyzers, sulfur_state=sulfur_state,
                         t95_samples=t95_samples, lims_latest=lims_latest, catalyst=catalyst,
                         issues=issues)
