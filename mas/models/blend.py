"""Смешение компонентов дизельного топлива: свойства смеси и проверка ограничений (векторный расчёт).

Сера и цетановое число смешиваются по массовым долям, T95 и ПТФ - по объёмным (плотность компонента
переводит массовую долю в объёмную). Для смесей из нескольких компонентов к T95 и ПТФ добавляется, а из
цетанового числа вычитается запас на нелинейность смешения.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

PROPS = ("S", "T95", "cetane", "CFPP", "density")
TOL = 1e-9


@dataclass(frozen=True)
class Component:
    name: str
    S: float
    D15: float
    cetane: float
    T95: float
    CFPP: float
    stock_t: float
    lo: float = 0.0
    hi: float = 1.0


@dataclass(frozen=True)
class Additive:
    name: str
    param: str
    doses: tuple
    effect: float


@dataclass(frozen=True)
class Limits:
    sulfur_max: float
    t95_max: float
    cetane_min: float
    cfpp_max: float | None  # None - нормы по ПТФ нет, показатель не проверяется
    d15: tuple


def share_grid(comps: list[Component], step: float) -> np.ndarray:
    """Все сочетания долей с шагом `step`, суммой 1 и допустимыми диапазонами долей: массив (N, k)."""
    k = len(comps)
    if k == 1:
        return np.ones((1, 1))
    n = int(round(1 / step))
    grid = np.round(np.arange(n + 1) * step, 4)
    combos = np.array(list(itertools.product(grid, repeat=k - 1)))
    shares = np.column_stack([combos, np.round(1.0 - combos.sum(axis=1), 4)])
    lo, hi = np.array([c.lo for c in comps]), np.array([c.hi for c in comps])
    keep = ((shares >= -TOL) & (shares <= 1 + TOL) & (shares >= lo - TOL) & (shares <= hi + TOL)).all(axis=1)
    return shares[keep]


def dose_grid(adds: list[Additive]) -> np.ndarray:
    """Все сочетания доз присадок, кг/т: массив (M, a); без присадок - одна строка нулевой ширины."""
    if not adds:
        return np.zeros((1, 0))
    return np.array(list(itertools.product(*[a.doses for a in adds])), dtype=float)


def evaluate(comps: list[Component], adds: list[Additive], shares: np.ndarray, doses: np.ndarray, *,
             margins: dict, batch_t: float, limits: Limits) -> pd.DataFrame:
    """Свойства, проверки норм и допустимость для каждой строки (доли, дозы)."""
    col = lambda attr: np.array([getattr(c, attr) for c in comps], dtype=float)  # noqa: E731
    vol_w = shares / col("D15")
    vol = vol_w.sum(axis=1)
    vfrac = vol_w / vol[:, None]
    bonus = ((shares > TOL).sum(axis=1) > 1).astype(float)

    s = shares @ col("S")
    cetane = shares @ col("cetane") - bonus * margins["cetane"]
    t95 = vfrac @ col("T95") + bonus * margins["T95"]
    cfpp = vfrac @ col("CFPP") + bonus * margins["CFPP"]
    for j, ad in enumerate(adds):
        effect = ad.effect * doses[:, j]
        cetane = cetane + effect if ad.param == "cetane" else cetane
        cfpp = cfpp + effect if ad.param == "CFPP" else cfpp
    density = 1.0 / vol

    values = {"S": s, "T95": t95, "cetane": cetane, "CFPP": cfpp, "density": density}
    ok = {"S": s <= limits.sulfur_max, "T95": t95 <= limits.t95_max, "cetane": cetane >= limits.cetane_min,
          "density": (density >= limits.d15[0]) & (density <= limits.d15[1])}
    if limits.cfpp_max is not None:
        ok["CFPP"] = cfpp <= limits.cfpp_max
    props = [k for k in PROPS if k in ok]
    stock_ok = (shares * batch_t <= col("stock_t") + TOL).all(axis=1)
    feasible = np.logical_and.reduce(list(ok.values())) & stock_ok

    names, add_names = [c.name for c in comps], [a.name for a in adds]
    lists, oks = {k: values[k].tolist() for k in props}, {k: ok[k].tolist() for k in props}
    n = len(shares)
    return pd.DataFrame({
        "id": [f"B{i:04d}" for i in range(n)],
        "shares": [dict(zip(names, row)) for row in shares.tolist()],
        "additives": [dict(zip(add_names, row)) for row in doses.tolist()],
        **{k: lists[k] for k in props},
        "checks": [[{"check": k, "value": lists[k][i], "ok": oks[k][i]} for k in props] for i in range(n)],
        "feasible": feasible, "stock_ok": stock_ok, "dose_total": doses.sum(axis=1),
        "n_fail": (~np.array([ok[k] for k in props])).sum(axis=0) + (~stock_ok),
        "s_fail": ~ok["S"], "main_share": shares[:, 0],
    })


def best_first(table: pd.DataFrame) -> pd.DataFrame:
    """Варианты в порядке предпочтения: наибольшая доля основного компонента, затем наименьшая суммарная доза
    присадок; допустимые варианты идут перед недопустимыми, а среди недопустимых - сначала с выполненной нормой по
    сере и наименьшим числом нарушений."""
    key = (table["dose_total"].to_numpy(), -table["main_share"].to_numpy(), table["n_fail"].to_numpy(),
           table["s_fail"].to_numpy(), ~table["feasible"].to_numpy())
    order = np.lexsort(key)
    return table.iloc[order].reset_index(drop=True)
