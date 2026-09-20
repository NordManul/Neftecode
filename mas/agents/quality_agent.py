"""QualityAgent: текущее/прогнозное качество, риск нарушения спецификации (шаг 3).

Ничего не знает о ранжировании вариантов и надёжности. `evaluate` оценивает таблицу
вариантов через ансамбль 27 членов - векторно, без циклов по строкам.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from mas.bus import MessageBus
from mas.common import action_threshold, load_calib, load_cfg, startup_window_h, v
from mas.contracts import CandidateSet, QualityAssessment
from mas.models.kalman import forecast_sulfur
from mas.models.kinetics import chance_exceed, ensemble_sulfur, response_ensemble, t95_ensemble, upper_quantile_mix
from mas.models.t95 import t95_estimate

SPEC_PARAMS = [
    ("Mg.Sulfur", "sulfur_max_mgkg", "<=", "мг/кг"), ("95%.T", "t95_max_c", "<=", "°C"),
    ("FlashPoint", "flash_min_c", ">=", "°C"), ("CetaneNumber", "cetane_min", ">=", ""),
    ("D15", "d15_range", "range", "кг/м3"),
]


class QualityAgent:
    def __init__(self, bus: MessageBus) -> None:
        self.cfg, self.calib = load_cfg(), load_calib()
        self.t95_calib = {"b95": self.calib["kinetics"]["b95"], "t95_sigma": self.calib["t95"]["sigma"],
                           "t95_drift": self.calib["t95"]["drift"]}
        self.members, self.weights = response_ensemble(self.calib["kinetics"])
        bus.register("QualityAgent", "assess", lambda snap: self.assess(snap))
        bus.register("QualityAgent", "evaluate", lambda payload: self.evaluate(payload))

    def _forecast_sulfur(self, snap) -> tuple[float, float]:
        """Прогноз серы на горизонт решения по двухкомпонентной модели фильтра (без поправок на недавние действия оператора)."""
        st = snap.sulfur_state
        s_hat, s_sigma = st.get("S_hat", np.nan), st.get("S_sigma", np.nan)
        n = int(round(v(self.cfg["sensor_fusion"]["forecast_horizon_h"]) * 6))
        mean, sd = forecast_sulfur(st.get("S_slow", s_hat), st.get("S_fast", 0.0), st.get("P_ss", s_sigma ** 2),
                                   st.get("P_ff", 0.0), st.get("P_sf", 0.0), self.calib["sulfur"], n)
        return float(mean), float(sd)

    def _spec_status(self, snap) -> list[dict]:
        specs = []
        for param, cfg_key, kind, unit in SPEC_PARAMS:
            f = snap.lims_latest.get(f"HT2:{param}")
            norm_val = v(self.cfg["spec"][cfg_key])
            if param == "CetaneNumber":  # ЦЧ ДТ с гидроочистки не нормируется (норма относится к товарному ДТ, проверяется в блендинге)
                specs.append({"param": param, "value": f["value"] if f else None, "unit": unit,
                              "age_h": f["age_h"] if f else None, "norm": None, "status": "не нормируется"})
                continue
            if f is None:
                ok = None
            elif kind == "range":
                ok = norm_val[0] <= f["value"] <= norm_val[1]
            elif kind == "<=":
                ok = f["value"] <= norm_val
            else:
                ok = f["value"] >= norm_val
            specs.append({"param": param, "value": f["value"] if f else None, "unit": unit,
                          "age_h": f["age_h"] if f else None, "norm": norm_val,
                          "status": "нет данных" if ok is None else ("норма" if ok else "нарушение")})
        return specs

    def _confidence(self, snap) -> tuple[float, str]:
        """Доверие к прогнозу - среднее четырёх показателей 0-1 без весов: неопределённость оценки серы относительно
        предельной `max_sigma_for_decision`, доля исправных анализаторов серы, свежесть их показаний относительно
        окна информации (три постоянные времени серы), наличие свежей пробы T95. Метки - по третям шкалы."""
        window = startup_window_h(self.calib)
        s_sigma = snap.sulfur_state.get("S_sigma", np.inf)
        sulfur_age = min(snap.analyzers["pak"]["age_last_ok_h"], snap.analyzers["q21"]["age_last_ok_h"])
        parts = [1.0 - min(s_sigma / v(self.cfg["sensor_fusion"]["max_sigma_for_decision"]), 1.0),
                 snap.analyzers["n_ok"] / 2.0, 1.0 - min(sulfur_age / window, 1.0), 1.0 if snap.t95_samples else 0.0]
        conf = float(np.mean(parts))
        return conf, ("высокая" if conf >= 2 / 3 else ("средняя" if conf >= 1 / 3 else "низкая"))

    def assess(self, snap) -> QualityAssessment:
        s_now, s_sigma = snap.sulfur_state.get("S_hat", np.nan), snap.sulfur_state.get("S_sigma", np.nan)
        sulfur_max = v(self.cfg["spec"]["sulfur_max_mgkg"])
        p_exceed_now = float(1 - norm.cdf((sulfur_max - s_now) / s_sigma)) if s_sigma > 0 else float("nan")
        s_fore, sig_fore = self._forecast_sulfur(snap)
        p_exceed_fore = float(1 - norm.cdf((sulfur_max - s_fore) / sig_fore)) if sig_fore > 0 else float("nan")

        f32_now = snap.values.get("F32_lag4h", np.nan)
        t95_est, t95_sigma = t95_estimate(snap.t95_samples, f32_now, self.t95_calib)
        t95_max = v(self.cfg["spec"]["t95_max_c"])
        p_t95 = float(1 - norm.cdf((t95_max - t95_est) / t95_sigma)) if t95_sigma > 0 else float("nan")

        conf, label = self._confidence(snap)
        risk_alpha = v(self.cfg["spec"]["risk_alpha"])
        p_s, p_t = (p if pd.notna(p) else 0.0 for p in (p_exceed_fore, p_t95))
        if p_s > action_threshold(self.cfg, "sulfur") or p_t > action_threshold(self.cfg, "t95"):
            risk = "высокий"
        else:
            risk = "повышенный" if max(p_s, p_t) > risk_alpha else "низкий"

        warnings = []
        if conf < 2 / 3:
            warnings.append(f"Доверие к прогнозу {label} ({conf:.2f}).")
        if not snap.t95_samples:
            warnings.append("Нет доступного анализа T95 продукта или сырья в пределах допустимого возраста.")

        return QualityAssessment(
            S_now=s_now, S_sigma=s_sigma, p_exceed_now=p_exceed_now,
            horizon_h=int(v(self.cfg["sensor_fusion"]["forecast_horizon_h"])),
            S_forecast=s_fore, S_forecast_sigma=sig_fore, p_exceed_forecast=p_exceed_fore,
            t95_est=t95_est, t95_sigma=t95_sigma, p_t95_exceed=p_t95,
            specs=self._spec_status(snap), confidence=conf, confidence_label=label, warnings=warnings, risk=risk,
        )

    def evaluate(self, payload: dict) -> CandidateSet:
        qa, table = payload["qa"], payload["table"].copy()
        n = len(table)
        gamma = self.calib["kinetics"]["gamma"]
        sulfur_max, t95_max = v(self.cfg["spec"]["sulfur_max_mgkg"]), v(self.cfg["spec"]["t95_max_c"])
        risk_alpha = v(self.cfg["spec"]["risk_alpha"])

        ln_s_fore = np.full(n, np.log(qa.S_forecast))
        s_k = ensemble_sulfur(ln_s_fore, table["dT5"].to_numpy(dtype=float), table["rF9"].to_numpy(dtype=float),
                               table["dF32"].to_numpy(dtype=float), gamma, self.members, self.weights)
        sigma_rel = np.full(n, qa.S_forecast_sigma / qa.S_forecast)
        p_exceed, p_exceed_worst = chance_exceed(s_k, sigma_rel, self.weights, sulfur_max)
        s_upper_mix = upper_quantile_mix(s_k, sigma_rel, self.weights, q=1 - risk_alpha)

        b95_values = np.asarray(self.members)[:, 2]
        t95_k = t95_ensemble(np.full(n, qa.t95_est), table["dF32"].to_numpy(dtype=float), b95_values)
        p_t95_k = 1 - norm.cdf((t95_max - t95_k) / qa.t95_sigma)
        p_t95 = p_t95_k @ self.weights
        t95_mid = t95_k @ self.weights
        # смесь нормалей с общей sigma: дисперсия смеси = разброс средних + σ_т95² (закон полной дисперсии),
        # верхняя граница уровня 1 - risk_alpha - согласованная величина для проверки блендинга, а не крайний член ансамбля.
        t95_mix_var = ((t95_k - t95_mid[:, None]) ** 2) @ self.weights + qa.t95_sigma ** 2
        t95_upper_mix = t95_mid + norm.ppf(1 - risk_alpha) * np.sqrt(t95_mix_var)

        table["S_mid"], table["S_upper_mix"], table["S_upper_worst"] = s_k @ self.weights, s_upper_mix, s_k.max(axis=1)
        table["P_exceed"], table["P_exceed_worst"] = p_exceed, p_exceed_worst
        table["T95_mid"], table["T95_upper_worst"], table["T95_upper_mix"], table["P_t95"] = t95_mid, t95_k.max(axis=1), t95_upper_mix, p_t95
        table["quality_margin"] = sulfur_max - s_upper_mix
        table["q_ok_sulfur"] = p_exceed <= risk_alpha
        table["t95_risk_uncontrolled"] = qa.p_t95_exceed if pd.notna(qa.p_t95_exceed) else 0.0
        table["q_ok_t95"] = (p_t95 <= risk_alpha) | (p_t95 <= table["t95_risk_uncontrolled"])
        table["q_reason"] = np.where(~table["q_ok_sulfur"], "риск по сере выше допустимого",
                                      np.where(~table["q_ok_t95"], "риск по T95 выше, чем без изменений", ""))
        return CandidateSet(table=table, note="оценка по 27 членам ансамбля (сера + T95)")
