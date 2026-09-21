"""Графики результатов в виде встроенного SVG без внешних библиотек.

Один и тот же код даёт графики для дашборда (цвета через CSS-переменные страницы) и отдельные файлы `docs/img/*.svg`
(светлый фон, цвета по умолчанию), которые показываются в README.
"""
from __future__ import annotations

import html
from typing import Sequence

import numpy as np
import pandas as pd

STATUS_KEYS = {"БЕЗ ИЗМЕНЕНИЙ": "none", "КОРРЕКТИРУЮЩЕЕ ДЕЙСТВИЕ": "action", "НАДЁЖНОЙ РЕКОМЕНДАЦИИ НЕТ": "refuse",
               "НАБЛЮДЕНИЕ (установка не в работе)": "observe"}
STATUS_LABELS = {"none": "без изменений", "action": "корректирующее действие", "refuse": "надёжной рекомендации нет",
                 "observe": "установка не в работе"}

FIG_CSS = """
.fg-bg{fill:var(--card,#fff)} .fg-axis{stroke:var(--border,#d5dbe3);stroke-width:1;fill:none}
.fg-grid{stroke:var(--border,#e6eaf0);stroke-width:1;stroke-dasharray:2 3}
.fg-t{fill:var(--muted,#5b6b7c);font:11px system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
.fg-tt{fill:var(--text,#16202b);font:600 12px system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
.fg-line{stroke:var(--accent,#1f6feb);stroke-width:1.4;fill:none;stroke-linejoin:round}
.fg-band{fill:var(--accent,#1f6feb);fill-opacity:.16;stroke:none}
.fg-limit{stroke:var(--bad,#c62828);stroke-width:1.2;stroke-dasharray:5 4;fill:none}
.fg-lab{fill:var(--muted,#5b6b7c);fill-opacity:.75} .fg-labx{fill:var(--bad,#c62828);fill-opacity:.9}
.fg-none{fill:var(--ok,#1a7f4b)} .fg-action{fill:var(--accent,#1f6feb)} .fg-refuse{fill:var(--warn,#b7791f)}
.fg-observe{fill:var(--border,#c9d1da)}
.fg-bar{fill:var(--accent,#1f6feb)} .fg-bar2{fill:var(--ok,#1a7f4b)} .fg-bar3{fill:var(--muted,#8a97a6)}
.fg-diag{stroke:var(--muted,#8a97a6);stroke-width:1;stroke-dasharray:4 4;fill:none}
.fg-dot{fill:var(--accent,#1f6feb);fill-opacity:.85;stroke:var(--card,#fff);stroke-width:1}
.fg-ref{stroke:var(--warn,#b7791f);stroke-width:1.2;stroke-dasharray:4 3;fill:none}
"""
_STYLE = f"<style>{FIG_CSS}</style>"


def _esc(text) -> str:
    return html.escape(str(text), quote=True)


def _open(width: int, height: int, title: str, standalone: bool) -> list[str]:
    head = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
            f'role="img" aria-label="{_esc(title)}" preserveAspectRatio="xMidYMid meet" style="max-width:100%;height:auto">')
    parts = [head, f"<title>{_esc(title)}</title>", _STYLE if standalone else ""]
    if standalone:
        parts.append(f'<rect class="fg-bg" x="0" y="0" width="{width}" height="{height}" rx="6"/>')
    return parts


def _runs(*arrays: np.ndarray) -> list[slice]:
    """Непрерывные участки, где все массивы конечны."""
    ok = np.ones(len(arrays[0]), dtype=bool)
    for a in arrays:
        ok &= np.isfinite(a)
    out, start = [], None
    for i, flag in enumerate(ok):
        if flag and start is None:
            start = i
        if start is not None and (not flag or i == len(ok) - 1):
            out.append(slice(start, i + 1 if flag else i))
            start = None
    return out


def _path(xs: np.ndarray, ys: np.ndarray) -> str:
    return "M" + " L".join(f"{a:.1f},{b:.1f}" for a, b in zip(xs, ys))


def _time_axis(parts: list[str], t0: pd.Timestamp, t1: pd.Timestamp, x0: float, x1: float, y_axis: float, months: int) -> None:
    span = (t1 - t0).total_seconds()
    tick = pd.Timestamp(year=t0.year, month=t0.month, day=1)
    while tick <= t1:
        if tick >= t0 and (tick.month - 1) % months == 0:
            x = x0 + (tick - t0).total_seconds() / span * (x1 - x0)
            parts.append(f'<line class="fg-grid" x1="{x:.1f}" y1="{y_axis - 1:.1f}" x2="{x:.1f}" y2="{y_axis + 4:.1f}"/>')
            parts.append(f'<text class="fg-t" x="{x:.1f}" y="{y_axis + 17:.1f}" text-anchor="middle">{tick:%Y-%m}</text>')
        tick = tick + pd.DateOffset(months=1)


def sulfur_series(state: pd.DataFrame, lab: pd.DataFrame, running: pd.Series, t0: pd.Timestamp, t1: pd.Timestamp,
                  limit: float, title: str, resample: str = "3h", markers: pd.DataFrame | None = None,
                  standalone: bool = False, width: int = 1100, height: int = 320, ymax: float = 22.0, month_step: int = 3, ymin: float = 0.0) -> str:
    """Оценка серы фильтром с 90 %-м интервалом и результаты ЛИМС; `markers` - циклы (t, status, text) в полосе под графиком."""
    s = state.loc[t0:t1, ["S_hat", "S_sigma"]]
    on = running.reindex(s.index).fillna(False).astype(bool)
    s = s.where(on).resample(resample).mean()
    xl, xr, yt, yb = 46.0, width - 14.0, 26.0, height - (58.0 if markers is not None else 34.0)
    span = (t1 - t0).total_seconds()
    fx = lambda idx: xl + (np.asarray((idx - t0).total_seconds(), dtype=float)) / span * (xr - xl)   # noqa: E731
    fy = lambda v_: yb - (np.clip(np.asarray(v_, dtype=float), ymin, ymax) - ymin) / (ymax - ymin) * (yb - yt)   # noqa: E731
    parts = _open(width, height, title, standalone)
    parts.append(f'<text class="fg-tt" x="{xl}" y="16">{_esc(title)}</text>')
    for tick in range(int(np.ceil(ymin / 2.0) * 2) if ymax - ymin <= 14 else int(np.ceil(ymin / 5.0) * 5), int(ymax) + 1, 2 if ymax - ymin <= 14 else 5):
        y = float(fy(tick))
        parts.append(f'<line class="fg-grid" x1="{xl}" y1="{y:.1f}" x2="{xr}" y2="{y:.1f}"/>')
        parts.append(f'<text class="fg-t" x="{xl - 6}" y="{y + 4:.1f}" text-anchor="end">{tick}</text>')
    parts.append(f'<text class="fg-t" x="12" y="{(yt + yb) / 2:.1f}" text-anchor="middle" transform="rotate(-90 12 {(yt + yb) / 2:.1f})">сера, мг/кг</text>')
    x = fx(s.index)
    hat, sig = s["S_hat"].to_numpy(), s["S_sigma"].to_numpy()
    lo, hi = hat - 1.645 * sig, hat + 1.645 * sig
    for run in _runs(x, hat, sig):
        if run.stop - run.start > 1:
            xs, up, dn = x[run], fy(hi[run]), fy(lo[run])
            parts.append('<path class="fg-band" d="' + _path(xs, up) + " L" + " L".join(f"{a:.1f},{b:.1f}" for a, b in zip(xs[::-1], dn[::-1])) + ' Z"/>')
            parts.append(f'<path class="fg-line" d="{_path(xs, fy(hat[run]))}"/>')
    yl = float(fy(limit))
    parts.append(f'<line class="fg-limit" x1="{xl}" y1="{yl:.1f}" x2="{xr}" y2="{yl:.1f}"/>')
    lb = lab[(lab["time"] >= t0) & (lab["time"] <= t1)]
    for t, val in zip(lb["time"], lb["value"]):
        cx, cy = float(fx(pd.DatetimeIndex([t]))[0]), float(fy(val))
        cls = "fg-labx" if val > limit else "fg-lab"
        parts.append(f'<circle class="{cls}" cx="{cx:.1f}" cy="{cy:.1f}" r="{2.6 if val > limit else 2.0}"><title>ЛИМС {t:%Y-%m-%d %H:%M}: {val:.1f} мг/кг</title></circle>')
    parts.append(f'<line class="fg-axis" x1="{xl}" y1="{yb}" x2="{xr}" y2="{yb}"/>')
    _time_axis(parts, t0, t1, xl, xr, yb, month_step)
    if markers is not None and len(markers):
        ym = yb + 30
        for t, st, txt in zip(markers["t"], markers["key"], markers["text"]):
            cx = float(fx(pd.DatetimeIndex([t]))[0])
            w = max((xr - xl) * 6 * 3600 / span, 1.2)
            parts.append(f'<rect class="fg-{st}" x="{cx - w / 2:.1f}" y="{ym}" width="{w:.1f}" height="9"><title>{_esc(txt)}</title></rect>')
        parts.append(f'<text class="fg-t" x="{xl - 6}" y="{ym + 8}" text-anchor="end">циклы</text>')
    lx = xl + 8
    for label, cls, kind in (("оценка фильтра и 90 % интервал", "fg-line", "line"), ("результат ЛИМС", "fg-lab", "dot"),
                             ("ЛИМС выше нормы", "fg-labx", "dot"), (f"норма {limit:g} мг/кг", "fg-limit", "line")):
        if kind == "line":
            parts.append(f'<line class="{cls}" x1="{lx}" y1="{yt + 8}" x2="{lx + 16}" y2="{yt + 8}"/>')
        else:
            parts.append(f'<circle class="{cls}" cx="{lx + 6}" cy="{yt + 8}" r="3"/>')
        parts.append(f'<text class="fg-t" x="{lx + 22}" y="{yt + 12}">{label}</text>')
        lx += 22 + 7 * len(label) + 14
    parts.append("</svg>")
    return "".join(parts)


def status_ribbon(cycles: pd.DataFrame, title: str, standalone: bool = False, width: int = 1100, height: int = 112) -> str:
    """Лента статусов по суткам: цвет клетки - самый значимый статус суток (действие, отказ, без изменений, не в работе)."""
    c = cycles.assign(t=pd.to_datetime(cycles["t"]))
    c["day"] = c["t"].dt.floor("D")
    rank = {"action": 3, "refuse": 2, "none": 1, "observe": 0}
    c["r"] = c["key"].map(rank)
    day = c.loc[c.groupby("day")["r"].idxmax()].set_index("day")
    days = pd.date_range(c["day"].min(), c["day"].max(), freq="D")
    xl, xr = 14.0, width - 14.0
    step = (xr - xl) / len(days)
    parts = _open(width, height, title, standalone)
    parts.append(f'<text class="fg-tt" x="{xl}" y="16">{_esc(title)}</text>')
    for i, d in enumerate(days):
        if d not in day.index:
            continue
        key = day.loc[d, "key"]
        n = int((c["day"] == d).sum())
        parts.append(f'<rect class="fg-{key}" x="{xl + i * step:.2f}" y="28" width="{max(step - 0.25, 0.6):.2f}" height="34">'
                     f'<title>{d:%Y-%m-%d}: {STATUS_LABELS[key]} (циклов за сутки: {n})</title></rect>')
    _time_axis(parts, days[0], days[-1] + pd.Timedelta(days=1), xl, xr, 62, 3)
    lx = xl
    for key in ("none", "action", "refuse", "observe"):
        parts.append(f'<rect class="fg-{key}" x="{lx}" y="92" width="10" height="10"/>')
        parts.append(f'<text class="fg-t" x="{lx + 15}" y="101">{STATUS_LABELS[key]}</text>')
        lx += 15 + 6.4 * len(STATUS_LABELS[key]) + 22
    parts.append("</svg>")
    return "".join(parts)


def stacked_bar(shares: dict[str, float], title: str, standalone: bool = False, width: int = 1100, height: int = 92) -> str:
    """Одна полоса с долями статусов бэктеста (доли в сумме 1)."""
    xl, xr = 14.0, width - 14.0
    parts = _open(width, height, title, standalone)
    parts.append(f'<text class="fg-tt" x="{xl}" y="16">{_esc(title)}</text>')
    x = xl
    for key in ("none", "action", "refuse", "observe"):
        share = shares.get(key, 0.0)
        w = share * (xr - xl)
        parts.append(f'<rect class="fg-{key}" x="{x:.1f}" y="28" width="{w:.1f}" height="28"><title>{STATUS_LABELS[key]}: {share:.1%}</title></rect>')
        if w > 44:
            parts.append(f'<text x="{x + w / 2:.1f}" y="47" text-anchor="middle" style="fill:#fff;font:600 12px system-ui,sans-serif">{share:.1%}</text>')
        x += w
    lx = xl
    for key in ("none", "action", "refuse", "observe"):
        label = f"{STATUS_LABELS[key]} {shares.get(key, 0.0):.1%}"
        parts.append(f'<rect class="fg-{key}" x="{lx}" y="70" width="10" height="10"/>')
        parts.append(f'<text class="fg-t" x="{lx + 15}" y="79">{label}</text>')
        lx += 15 + 6.4 * len(label) + 22
    parts.append("</svg>")
    return "".join(parts)


def reliability(bins: Sequence[dict], title: str, standalone: bool = False, size: int = 420) -> str:
    """Расчётная вероятность превышения и наблюдаемая частота по интервалам: точка - интервал, площадь - число анализов."""
    m, w = 44.0, float(size)
    parts = _open(size, size, title, standalone)
    parts.append(f'<text class="fg-tt" x="{m}" y="16">{_esc(title)}</text>')
    fx = lambda p: m + p * (w - m - 14)          # noqa: E731
    fy = lambda p: w - m - p * (w - m - 26)      # noqa: E731
    for tick in (0, 0.25, 0.5, 0.75, 1.0):
        parts.append(f'<line class="fg-grid" x1="{fx(0):.1f}" y1="{fy(tick):.1f}" x2="{fx(1):.1f}" y2="{fy(tick):.1f}"/>')
        parts.append(f'<line class="fg-grid" x1="{fx(tick):.1f}" y1="{fy(0):.1f}" x2="{fx(tick):.1f}" y2="{fy(1):.1f}"/>')
        parts.append(f'<text class="fg-t" x="{m - 6}" y="{fy(tick) + 4:.1f}" text-anchor="end">{tick:g}</text>')
        parts.append(f'<text class="fg-t" x="{fx(tick):.1f}" y="{w - m + 16:.1f}" text-anchor="middle">{tick:g}</text>')
    parts.append(f'<line class="fg-diag" x1="{fx(0):.1f}" y1="{fy(0):.1f}" x2="{fx(1):.1f}" y2="{fy(1):.1f}"/>')
    nmax = max((b["n"] for b in bins), default=1)
    for b in bins:
        if not b["n"] or b["frequency"] != b["frequency"]:
            continue
        r = 4 + 9 * (b["n"] / nmax) ** 0.5
        parts.append(f'<circle class="fg-dot" cx="{fx(b["mean_p"]):.1f}" cy="{fy(b["frequency"]):.1f}" r="{r:.1f}"><title>расчётная {b["mean_p"]:.2f}, '
                     f'наблюдаемая {b["frequency"]:.2f}; анализов {b["n"]}, превышений {b["n_exceeded"]}</title></circle>')
    parts.append(f'<text class="fg-t" x="{(fx(0) + fx(1)) / 2:.1f}" y="{w - 8:.1f}" text-anchor="middle">расчётная вероятность превышения</text>')
    parts.append(f'<text class="fg-t" x="12" y="{(fy(0) + fy(1)) / 2:.1f}" text-anchor="middle" transform="rotate(-90 12 {(fy(0) + fy(1)) / 2:.1f})">наблюдаемая частота</text>')
    parts.append("</svg>")
    return "".join(parts)


def hbars(items: Sequence[tuple[str, float, str]], title: str, vmax: float, fmt: str = "{:.2f}", standalone: bool = False,
          width: int = 640, ref: tuple[float, str] | None = None, label_w: int = 250) -> str:
    """Горизонтальные столбцы: (подпись, значение, класс цвета bar/bar2/bar3)."""
    row_h, top = 26, 30
    height = top + row_h * len(items) + 14
    xl, xr = float(label_w), width - 60.0
    parts = _open(width, height, title, standalone)
    parts.append(f'<text class="fg-tt" x="14" y="16">{_esc(title)}</text>')
    for i, (label, value, cls) in enumerate(items):
        y = top + i * row_h
        w = max(value / vmax, 0) * (xr - xl)
        parts.append(f'<text class="fg-t" x="{xl - 8}" y="{y + 13}" text-anchor="end">{_esc(label)}</text>')
        parts.append(f'<rect class="fg-{cls}" x="{xl}" y="{y + 2}" width="{w:.1f}" height="16" rx="2"><title>{_esc(label)}: {fmt.format(value)}</title></rect>')
        parts.append(f'<text class="fg-t" x="{xl + w + 6:.1f}" y="{y + 14}">{fmt.format(value)}</text>')
    if ref is not None:
        x = xl + ref[0] / vmax * (xr - xl)
        parts.append(f'<line class="fg-ref" x1="{x:.1f}" y1="{top - 4}" x2="{x:.1f}" y2="{height - 10}"/>')
        parts.append(f'<text class="fg-t" x="{x + 4:.1f}" y="{top - 8}">{_esc(ref[1])}</text>')
    parts.append("</svg>")
    return "".join(parts)


def grouped_bars(groups: Sequence[str], series: Sequence[tuple[str, Sequence[float], str]], title: str, base: float | None,
                 standalone: bool = False, width: int = 640, height: int = 250, ymax: float = 1.0) -> str:
    """Столбцы по группам (например, упреждение) для нескольких показателей в долях 0..1; `base` - уровень случайного выбора."""
    xl, xr, yt, yb = 46.0, width - 14.0, 40.0, height - 44.0
    parts = _open(width, height, title, standalone)
    parts.append(f'<text class="fg-tt" x="{xl}" y="16">{_esc(title)}</text>')
    step = 0.1 if ymax <= 0.5 else 0.25
    for i in range(int(round(ymax / step)) + 1):
        tick = i * step
        y = yb - tick / ymax * (yb - yt)
        parts.append(f'<line class="fg-grid" x1="{xl}" y1="{y:.1f}" x2="{xr}" y2="{y:.1f}"/>')
        parts.append(f'<text class="fg-t" x="{xl - 6}" y="{y + 4:.1f}" text-anchor="end">{tick:.0%}</text>')
    gw = (xr - xl) / len(groups)
    bw = min(34.0, gw / (len(series) + 1))
    for gi, name in enumerate(groups):
        cx = xl + gw * (gi + 0.5)
        for si, (label, vals, cls) in enumerate(series):
            v_ = vals[gi]
            x = cx + (si - (len(series) - 1) / 2) * (bw + 6) - bw / 2
            h = (v_ if v_ == v_ and v_ is not None else 0) / ymax * (yb - yt)
            parts.append(f'<rect class="fg-{cls}" x="{x:.1f}" y="{yb - h:.1f}" width="{bw:.1f}" height="{h:.1f}" rx="2"><title>{_esc(label)}, {_esc(name)}: {v_:.1%}</title></rect>')
            parts.append(f'<text class="fg-t" x="{x + bw / 2:.1f}" y="{yb - h - 4:.1f}" text-anchor="middle">{v_:.0%}</text>')
        parts.append(f'<text class="fg-t" x="{cx:.1f}" y="{yb + 16:.1f}" text-anchor="middle">{_esc(name)}</text>')
    if base is not None:
        y = yb - base / ymax * (yb - yt)
        parts.append(f'<line class="fg-ref" x1="{xl}" y1="{y:.1f}" x2="{xr}" y2="{y:.1f}"/>')
    lx = xl
    for label, _, cls in series:
        parts.append(f'<rect class="fg-{cls}" x="{lx}" y="{height - 16}" width="10" height="10"/>')
        parts.append(f'<text class="fg-t" x="{lx + 15}" y="{height - 7}">{_esc(label)}</text>')
        lx += 15 + 6.4 * len(label) + 22
    if base is not None:
        parts.append(f'<line class="fg-ref" x1="{lx}" y1="{height - 11}" x2="{lx + 16}" y2="{height - 11}"/>')
        parts.append(f'<text class="fg-t" x="{lx + 21}" y="{height - 7}">случайный выбор ({base:.1%})</text>')
    parts.append("</svg>")
    return "".join(parts)


def architecture(standalone: bool = False, width: int = 1100, height: int = 300) -> str:
    """Схема взаимодействия агентов: обмен только через шину сообщений, журнал шины входит в карточку оператора."""
    parts = _open(width, height, "Архитектура: агенты обмениваются сообщениями через шину", standalone)
    parts.append('<style>.ar-box{fill:var(--card,#fff);stroke:var(--accent,#1f6feb);stroke-width:1.4}.ar-orc{fill:var(--accent,#1f6feb);stroke:none}'
                 '.ar-src{fill:var(--card,#fff);stroke:var(--muted,#8a97a6);stroke-width:1.2;stroke-dasharray:4 3}'
                 '.ar-bus{fill:var(--accent,#1f6feb);fill-opacity:.13;stroke:var(--accent,#1f6feb);stroke-width:1}'
                 '.ar-ln{stroke:var(--muted,#8a97a6);stroke-width:1.4;fill:none;marker-end:url(#ar-arrow)}'
                 '.ar-h{fill:var(--text,#16202b);font:600 13px system-ui,-apple-system,Segoe UI,Roboto,sans-serif}'
                 '.ar-w{fill:#fff;font:600 13px system-ui,-apple-system,Segoe UI,Roboto,sans-serif}'
                 '.ar-s{fill:var(--muted,#5b6b7c);font:11px system-ui,-apple-system,Segoe UI,Roboto,sans-serif}'
                 '.ar-sw{fill:#fff;font:11px system-ui,-apple-system,Segoe UI,Roboto,sans-serif}</style>')
    parts.append('<defs><marker id="ar-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
                 '<path d="M0,0 L10,5 L0,10 z" fill="var(--muted,#8a97a6)"/></marker></defs>')
    agents = [("DataAgent", "снимок состояния,", "возраст анализов, флаги"), ("QualityAgent", "оценка серы и T95,", "риск нарушения нормы"),
              ("ReliabilityAgent", "индекс тяжести режима,", "границы модели"), ("OptimizationAgent", "105 вариантов, отбор,", "порядок предпочтения"),
              ("BlendingAgent", "доли компонентов,", "присадки, нормы смеси")]
    bw, gap, x0, y0, bh = 158, 14, 160, 34, 62
    for i, (name, l1, l2) in enumerate(agents):
        x = x0 + i * (bw + gap)
        parts.append(f'<rect class="ar-box" x="{x}" y="{y0}" width="{bw}" height="{bh}" rx="8"/>')
        parts.append(f'<text class="ar-h" x="{x + bw / 2}" y="{y0 + 22}" text-anchor="middle">{name}</text>')
        parts.append(f'<text class="ar-s" x="{x + bw / 2}" y="{y0 + 40}" text-anchor="middle">{l1}</text>')
        parts.append(f'<text class="ar-s" x="{x + bw / 2}" y="{y0 + 54}" text-anchor="middle">{l2}</text>')
        parts.append(f'<line class="ar-ln" x1="{x + bw / 2}" y1="{y0 + bh}" x2="{x + bw / 2}" y2="{y0 + bh + 34}" style="marker-end:none"/>')
    bus_y = y0 + bh + 34
    parts.append(f'<rect class="ar-bus" x="{x0 - 10}" y="{bus_y}" width="{5 * bw + 4 * gap + 20}" height="30" rx="6"/>')
    parts.append(f'<text class="ar-h" x="{x0 + (5 * bw + 4 * gap) / 2}" y="{bus_y + 19}" text-anchor="middle">шина сообщений: запрос, ответ, журнал</text>')
    ox, oy, ow, oh = x0 + (5 * bw + 4 * gap) / 2 - 170, bus_y + 64, 340, 66
    parts.append(f'<line class="ar-ln" x1="{ox + ow / 2}" y1="{oy}" x2="{ox + ow / 2}" y2="{bus_y + 30}" style="marker-end:none"/>')
    parts.append(f'<rect class="ar-orc" x="{ox}" y="{oy}" width="{ow}" height="{oh}" rx="8"/>')
    parts.append(f'<text class="ar-w" x="{ox + ow / 2}" y="{oy + 24}" text-anchor="middle">Orchestrator</text>')
    parts.append(f'<text class="ar-sw" x="{ox + ow / 2}" y="{oy + 43}" text-anchor="middle">выбор варианта, независимая повторная проверка,</text>')
    parts.append(f'<text class="ar-sw" x="{ox + ow / 2}" y="{oy + 57}" text-anchor="middle">объяснение либо отказ</text>')
    parts.append(f'<rect class="ar-src" x="8" y="{y0}" width="132" height="{bh}" rx="8"/>')
    parts.append(f'<text class="ar-h" x="74" y="{y0 + 22}" text-anchor="middle">Данные</text>')
    parts.append(f'<text class="ar-s" x="74" y="{y0 + 40}" text-anchor="middle">телеметрия, ЛИМС,</text>')
    parts.append(f'<text class="ar-s" x="74" y="{y0 + 54}" text-anchor="middle">ПАК, справочник</text>')
    parts.append(f'<line class="ar-ln" x1="140" y1="{y0 + bh / 2}" x2="{x0 - 2}" y2="{y0 + bh / 2}"/>')
    cx, cw = ox + ow + 40, 200
    parts.append(f'<line class="ar-ln" x1="{ox + ow}" y1="{oy + oh / 2}" x2="{cx - 2}" y2="{oy + oh / 2}"/>')
    parts.append(f'<rect class="ar-box" x="{cx}" y="{oy}" width="{cw}" height="{oh}" rx="8"/>')
    parts.append(f'<text class="ar-h" x="{cx + cw / 2}" y="{oy + 24}" text-anchor="middle">Карточка оператора</text>')
    parts.append(f'<text class="ar-s" x="{cx + cw / 2}" y="{oy + 43}" text-anchor="middle">8 блоков и журнал шины</text>')
    parts.append(f'<text class="ar-s" x="{cx + cw / 2}" y="{oy + 57}" text-anchor="middle">либо отказ с причиной</text>')
    parts.append("</svg>")
    return "".join(parts)
