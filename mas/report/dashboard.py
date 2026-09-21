"""Сводка результатов в одном HTML-файле: графики (встроенный SVG), выводы, демонстрационные карточки, источники допущений.

Все числа читаются из файлов `outputs/` и `config/`, поэтому страница отражает последний прогон. Файл самодостаточен:
внешних библиотек и обращений к сети нет, светлая и тёмная темы переключаются кнопкой.
"""
from __future__ import annotations

import html
import json
import re
from datetime import datetime, timezone

import pandas as pd

from mas.common import CACHE_DIR, DOCS_DIR, OUTPUTS_DIR, ROOT, node_refs, v
from mas.report import figures as fg

DEMO_NOTES = {
    1: ("Устойчивый режим", "Лишних управляющих действий система не создаёт: статус «без изменений»."),
    2: ("Риск по сере", "Оценка серы выше нормы: вариант действия с проверкой ограничений, альтернативы и объяснение."),
    3: ("Риск по T95", "Правило «не навреди» для T95 и эскалация технологу."),
    4: ("Анализатор застыл", "Показания ПАК повторяются, оценка строится по остальным источникам."),
    5: ("После пуска нет данных", "Пусковой режим и устаревшие данные: отказ с указанием причины."),
    6: ("Сера выше нормы", "Допустимого варианта в заданных границах нет: отказ и направление снижения риска."),
    7: ("Режим вне области модели", "T5 выше верхней границы коридора: рекомендация не выдаётся, эскалация технологу."),
    8: ("Простой установки", "Установка не в работе: наблюдение."),
    9: ("Запас по качеству", "Запас по сере достаточен: изменений не требуется."),
    10: ("Блендинг без данных компонентов", "Режим scenario без данных компонентов сводится к товарному ДТ = ГО ДТ с предупреждением."),
}
STATUS_CLASS = {"БЕЗ ИЗМЕНЕНИЙ": "none", "КОРРЕКТИРУЮЩЕЕ ДЕЙСТВИЕ": "action", "НАДЁЖНОЙ РЕКОМЕНДАЦИИ НЕТ": "refuse",
                "НАБЛЮДЕНИЕ (установка не в работе)": "observe"}


def _load_json(path, default=None):
    return json.load(open(path, encoding="utf-8")) if path.exists() else default


def _e(x) -> str:
    return html.escape(str(x), quote=True)


def _assumptions(cfg: dict) -> list[dict]:
    rows = []

    def walk(node, path):
        if isinstance(node, dict) and "src" in node:
            value = node["value"] if "value" in node else {k: x for k, x in node.items() if k not in ("src", "note", "ref")}
            text = "; ".join(f"{k}={x}" for k, x in value.items()) if isinstance(value, dict) else str(value)
            rows.append({"path": path, "value": text, "src": node["src"], "note": node.get("note", ""),
                         "refs": [{k: r.get(k) for k in ("kind", "short", "url", "file")} for r in node_refs(node)]})
        elif isinstance(node, dict):
            for k, x in node.items():
                walk(x, f"{path}.{k}" if path else k)

    walk(cfg, "")
    return rows


def _inline(text: str) -> str:
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", _e(text))


def card_to_html(md: str) -> str:
    """Карточка оператора из Markdown (заголовки, списки, таблицы) в HTML."""
    out, table, lst = [], [], False

    def flush_table():
        if not table:
            return
        rows = [[c.strip() for c in r.strip().strip("|").split("|")] for r in table if not re.match(r"^\|[\s\-|:]+\|$", r.strip())]
        head, body = rows[0], rows[1:]
        out.append('<div class="tw"><table><thead><tr>' + "".join(f"<th>{_inline(c)}</th>" for c in head) + "</tr></thead><tbody>" +
                   "".join("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>" for r in body) + "</tbody></table></div>")
        table.clear()

    for line in md.splitlines():
        if line.startswith("|"):
            table.append(line)
            continue
        flush_table()
        if lst and not line.startswith("- "):
            out.append("</ul>")
            lst = False
        if line.startswith("# "):
            out.append(f"<h4>{_inline(line[2:])}</h4>")
        elif line.startswith("## "):
            out.append(f"<h5>{_inline(line[3:])}</h5>")
        elif line.startswith("- "):
            if not lst:
                out.append("<ul>")
                lst = True
            out.append(f"<li>{_inline(line[2:])}</li>")
        elif line.strip():
            out.append(f"<p>{_inline(line)}</p>")
    flush_table()
    if lst:
        out.append("</ul>")
    return "".join(out)


def _pick_episode(cycles: pd.DataFrame, lab: pd.DataFrame, limit: float, days: int = 12) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Окно с наибольшим числом циклов с действием и результатов ЛИМС выше нормы."""
    best, best_score = None, -1
    act = cycles.loc[cycles["key"] == "action", "t"]
    exc = lab.loc[lab["value"] > limit, "time"]
    start = cycles["t"].min().floor("D")
    while start + pd.Timedelta(days=days) <= cycles["t"].max():
        end = start + pd.Timedelta(days=days)
        score = int(((act >= start) & (act < end)).sum()) + 2 * int(((exc >= start) & (exc < end)).sum())
        if score > best_score:
            best, best_score = start, score
        start += pd.Timedelta(days=1)
    return best, best + pd.Timedelta(days=days)


def _fmt_pct(x, digits: int = 1) -> str:
    return "-" if x is None or x != x else f"{x * 100:.{digits}f} %"


def _kpi(value: str, label: str, note: str = "", tone: str = "") -> str:
    return (f'<div class="kpi {tone}"><div class="kpi-v">{value}</div><div class="kpi-l">{_e(label)}</div>'
            f'<div class="kpi-n">{note}</div></div>')


def _table(rows: list[list], head: list[str], cls: str = "") -> str:
    return (f'<div class="tw"><table class="{cls}"><thead><tr>' + "".join(f"<th>{_e(h)}</th>" for h in head) + "</tr></thead><tbody>" +
            "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows) + "</tbody></table></div>")


def _demo_cards() -> str:
    demo_dir = OUTPUTS_DIR / "demo"
    items = []
    for js in sorted(demo_dir.glob("*.json")):
        num = int(js.stem.split("_")[0])
        payload = json.load(open(js, encoding="utf-8"))
        md = (demo_dir / f"{js.stem}.md").read_text(encoding="utf-8") if (demo_dir / f"{js.stem}.md").exists() else ""
        status = payload["status"]
        base = status.split(" +")[0]
        name, note = DEMO_NOTES.get(num, (js.stem, ""))
        escalate = "ЭСКАЛАЦИЯ" in status
        items.append(
            f'<details class="demo"><summary><span class="demo-n">{num:02d}</span><span class="demo-t"><strong>{_e(name)}</strong>'
            f'<span class="demo-d">{_e(note)}</span></span><span class="chip {STATUS_CLASS.get(base, "observe")}">{_e(base.lower())}'
            f'{" + эскалация" if escalate else ""}</span></summary><div class="card-body"><div class="card-time">Момент: {_e(str(payload["t"])[:16])}. '
            f'Файлы: <code>outputs/demo/{_e(js.stem)}.md</code>, <code>.json</code></div>{card_to_html(md)}</div></details>')
    return '<div class="demos">' + "".join(items) + "</div>"


def _sources_table(assumptions: list[dict]) -> str:
    rows = []
    for a in assumptions:
        refs = "; ".join((f'<a href="{_e(r["url"])}" target="_blank" rel="noopener">{_e(r["short"])}</a>' if r.get("url")
                          else (f'<a href="../{_e(r["file"])}">{_e(r["short"])}</a>' if (r.get("file") or "").startswith("docs/") else _e(r["short"])))
                         for r in a["refs"])
        rows.append([f"<code>{_e(a['path'])}</code>", _e(a["value"]), f'<span class="tag">{_e(a["src"])}</span>', refs, _e(a["note"])])
    return _table(rows, ["Параметр", "Значение", "Тип", "Источник", "Примечание"], "src-table")


def _effect_summary(rows: list[dict]) -> str:
    strong = [r for r in rows if "сильная" in r["ensemble"]]
    same_sign = all((r["beta"] < 0) == (rows[0]["beta"] < 0) for r in rows if r["lever"] == rows[0]["lever"])
    return (("Знак отклика по данным одинаков на обоих периодах и у обоих анализаторов. " if same_sign else "") +
            ("Сильная оценка ансамбля (литература) в интервалы не попадает; данные не исключают более сильный отклик из-за смещения к нулю." if not strong
             else "Сильная оценка ансамбля попадает в часть интервалов."))


def _effect_section(effect: dict | None) -> str:
    if not effect:
        return ""
    ev, op = effect["natural_experiments"], effect["operator_concordance"]
    rows = [[_e(r["lever"]), _e(r["source"]), _e(r["period"]), str(r["n"]), f'{r["beta"]:+.4f}', f'[{r["lo"]:+.4f}; {r["hi"]:+.4f}]', _e(r["ensemble"])]
            for r in ev["rows"]]
    groups = _table([[_e(g["group"]), str(g["n"]), f'{g["d_t5_mean"]:+.2f}', f'{g["d_f9_mean_pct"]:+.1f}'] for g in op["groups"]],
                    ["Циклы", "Циклов", "Изменение T5, °C", "Изменение загрузки F9, %"])
    d = op["difference"]
    diff = (f'После сигнала операторы поднимали T5 на {d["d_t5"]:+.1f} °C сильнее, чем в спокойных циклах (90 %-й интервал [{d["d_t5_ci"][0]:+.1f}; {d["d_t5_ci"][1]:+.1f}]); '
            f'загрузку они не снижали (разность {100 * d["d_lnf9"]:+.1f} %). Температурная часть советов согласуется с практикой операторов, снижение загрузки - нет.')
    return (f'<section id="effect"><h2>Проверка модели отклика на записанных данных</h2>'
            f'<p class="lead">Проверить эффект рекомендаций на установке нельзя. Отклик серы на изменение T5 и загрузки проверяется по ступенчатым изменениям, '
            f'которые операторы делали сами (естественные эксперименты), и по согласию советов с действиями операторов. Файл: <code>outputs/effect_check.json</code>.</p>'
            f'<div class="two"><div class="card"><h3>Отклик серы на ступенчатые изменения</h3>'
            f'{_table(rows, ["Рычаг", "Анализатор", "Период", "Событий", "Отклик", "90 %-й интервал", "Члены ансамбля в интервале"])}'
            f'<p class="note">{_e(ev["note"])} {_e(_effect_summary(ev['rows']))}</p></div>'
            f'<div class="card"><h3>Согласие с действиями операторов</h3>{groups}<p class="note">{_e(diff)} {_e(op["note"])}</p></div></div></section>')


def build_dashboard() -> str:
    from mas.common import load_calib, load_cfg
    cfg, calib = load_cfg(), load_calib()
    val = _load_json(OUTPUTS_DIR / "validation.json", {})
    bt = _load_json(OUTPUTS_DIR / "backtest_summary.json", {})
    hold = _load_json(OUTPUTS_DIR / "internal_holdout.json", {})
    report = _load_json(OUTPUTS_DIR / "calibration_report.json", {})
    effect = _load_json(OUTPUTS_DIR / "effect_check.json")
    vak_path = OUTPUTS_DIR / "vak_regulation.csv"
    vak = pd.read_csv(vak_path) if vak_path.exists() else pd.DataFrame()
    sulfur = val.get("sulfur", {})
    est = sulfur.get("estimators", {}) or {}
    dist = bt.get("status_distribution", {}) or {}
    shares = {fg.STATUS_KEYS.get(k, k): x for k, x in dist.items()}
    det = bt.get("detection", []) or []
    limit = v(cfg["spec"]["sulfur_max_mgkg"])
    t0, t1 = pd.Timestamp(cfg["periods"]["train_end"]), pd.Timestamp(cfg["periods"]["test_end"])

    state = pd.read_parquet(CACHE_DIR / "state_sulfur.parquet")
    running = pd.read_parquet(CACHE_DIR / "running.parquet")["ho_running"]
    lims = pd.read_parquet(CACHE_DIR / "lims.parquet")
    lab = lims[(lims["point"] == "HT2") & (lims["parameter"] == "Mg.Sulfur")][["time", "value"]]
    cycles = pd.read_csv(OUTPUTS_DIR / "backtest_cycles.csv", parse_dates=["t"])
    cycles["key"] = cycles["status"].str.replace(r" \+.*", "", regex=True).map(fg.STATUS_KEYS)
    cycles["text"] = [f"{t:%Y-%m-%d %H:%M}: {fg.STATUS_LABELS[k]}" + ("" if k != "action" else f" (ΔT5 {a:+.0f} °C, ΔF9 {b:+.0%}, ΔF32 {c:+.1f} т/ч)")
                      for t, k, a, b, c in zip(cycles["t"], cycles["key"], cycles["dT5"].fillna(0), cycles["rF9"].fillna(0), cycles["dF32"].fillna(0))]

    filt = est.get("Фильтр по двум приборам", {})
    gbm = est.get("Градиентный бустинг по режимным тегам", {})
    d1 = next((d for d in det if d["lead_h"] == 1.0), {})
    share_exc = sulfur.get("share_above_10")

    # --- графики
    ep0, ep1 = _pick_episode(cycles, lab, limit)
    figs = {
        "series": fg.sulfur_series(state, lab, running, t0, t1, limit, "Оценка серы фильтром и результаты ЛИМС на отложенном периоде"),
        "episode": fg.sulfur_series(state, lab, running, ep0, ep1, limit, f"Эпизод {ep0:%Y-%m-%d} … {ep1:%Y-%m-%d}: оценка, анализы и решения системы",
                                    resample="1h", markers=cycles[(cycles["t"] >= ep0) & (cycles["t"] <= ep1)], ymin=4.0, ymax=16.0, month_step=1, height=340),
        "ribbon": fg.status_ribbon(cycles, "Самый значимый статус за сутки (действие, отказ, без изменений, не в работе)"),
        "stack": fg.stacked_bar(shares, "Доли статусов, 2336 циклов с шагом 6 ч"),
        "arch": fg.architecture(),
    }
    est_items = [(k, m["auc"], "bar2" if k == "Фильтр по двум приборам" else ("bar3" if "бустинг" in k else "bar"))
                 for k, m in sorted(est.items(), key=lambda kv: -(kv[1]["auc"] or 0)) if m.get("auc")]
    figs["auc"] = fg.hbars(est_items, "Отличение превышения нормы: AUC (0.5 - уровень случайного выбора)", 1.0, "{:.3f}", ref=(0.5, "0.5"))
    pc = sulfur.get("probability_calibration", {})
    if pc:
        figs["reliability"] = fg.reliability(pc["bins"], "Расчётная вероятность и частота превышений")
    if det:
        figs["detect"] = fg.grouped_bars([f"упреждение {d['lead_h']:g} ч" for d in det],
                                         [("точность сигнала", [d["precision"] for d in det], "bar"), ("полнота сигнала", [d["recall"] for d in det], "bar2"),
                                          ("доля анализов с сигналом", [d["share_flagged"] for d in det], "bar3")],
                                         "Сигнал о риске (вероятность выше 1/2) и результат анализа ЛИМС", base=share_exc, ymax=0.4)

    # --- разделы
    kpis = "".join([
        _kpi(f"{bt.get('n_cycles', '-')}", "циклов бэктеста", f"шаг 6 ч, {t0:%Y-%m-%d} - {t1:%Y-%m-%d}"),
        _kpi(_fmt_pct(shares.get("none")), "без изменений", "лишних действий нет", "ok"),
        _kpi(_fmt_pct(shares.get("action")), "корректирующее действие", f"{_fmt_pct(d1.get('share_flagged_with_recommendation'), 0)} сигналов о риске получают рекомендацию", "accent"),
        _kpi(_fmt_pct(shares.get("refuse")), "надёжной рекомендации нет", "штатный отказ с причиной", "warn"),
        _kpi(f"{filt.get('auc', 0):.2f}", "AUC оценки серы", f"бустинг по режимным тегам {gbm.get('auc', 0):.2f}"),
        _kpi(f"{sulfur.get('brier', 0):.3f}", "Brier оценки серы", f"климатология {sulfur.get('brier_climatology', 0):.3f}"),
    ])
    holdout_est = (hold.get("out_of_sample", {}) or {}).get("estimators", {})
    hf, hm = holdout_est.get("Фильтр по двум приборам", {}), holdout_est.get("Среднее двух приборов, очищенное", {})
    fc = val.get("sulfur_forecast", {})
    conclusions = [
        ("ok", "Что работает",
         [f"Оценка серы «сейчас»: фильтр по двум анализаторам и ЛИМС даёт AUC {filt.get('auc', 0):.2f} против {gbm.get('auc', 0):.2f} у бустинга по режимным тегам "
          f"и {est.get('Последний анализ ЛИМС', {}).get('auc', 0):.2f} у «последнего анализа»; интервал накрывает {sulfur.get('coverage_90', 0):.0%} анализов.",
          f"В спокойном режиме система не создаёт действий: {_fmt_pct(shares.get('none'))} циклов без изменений.",
          "Недопустимый вариант не может быть выбран: отбор по ограничениям выполняется до сравнения, результат перепроверяется независимо; при отсутствии допустимого варианта система отказывается и называет причину.",
          "Каждое значение в конфигурации имеет источник (норматив, справочник, данные, задание); пересчёт с нуля даёт побайтово те же кэш и калибровку."]),
        ("warn", "Ограничения",
         [f"Сигнал о риске на цикле слабо информативен: точность {_fmt_pct(d1.get('precision'), 0)}, полнота {_fmt_pct(d1.get('recall'), 0)} при доле превышений "
          f"{_fmt_pct(share_exc)}; различение серы падает с ростом упреждения (прогноз на {fc.get('horizon_h', 4)} ч: AUC {fc.get('auc', 0):.2f}). Причина: замкнутый контур и погрешность лаборатории.",
          *([f"На внутренней проверке (подбор по 2023 году, оценка по 2024) фильтр равен среднему двух приборов: AUC {hf['auc']:.2f} и {hm['auc']:.2f}; преимущество, наблюдаемое на отложенном периоде, вне выборки не подтверждено."]
            if hf and hm else []),
          "Эффект рекомендованных действий получен моделью отклика; на замкнутом контуре он не проверен, число рекомендаций зависит от принятой оценки чувствительности.",
          f"Формулы виртуального анализатора не воспроизводят лабораторию (пригодных {int((vak['verdict'] == 'пригодна').sum()) if len(vak) else 0} из {len(vak)}); энергозатраты не оцениваются; блендинг по данным компонентов задаётся сценарием."]),
    ]
    concl = "".join(f'<div class="card concl {tone}"><h3>{title}</h3><ul>' + "".join(f"<li>{_e(x)}</li>" for x in items) + "</ul></div>" for tone, title, items in conclusions)

    est_rows = [[_e(k), f"{m['auc']:.3f}", f"{m['mae']:.2f}", str(m["n"])] for k, m in sorted(est.items(), key=lambda kv: -(kv[1]["auc"] or 0))]
    fail_rows = [[_e(k), str(x)] for k, x in (bt.get("failure_reasons", {}) or {}).items()]
    det_rows = [[f"{d['lead_h']:g}", str(d["n_analyses"]), _fmt_pct(d["share_flagged"]), _fmt_pct(d["precision"]), _fmt_pct(d["recall"]),
                 _fmt_pct(d["share_flagged_with_recommendation"], 0)] for d in det]
    assumptions = _assumptions(cfg)
    sources = _sources_table(assumptions)
    docs_links = [
        ("00_СОСТАВ_СИСТЕМЫ.md", "Состав системы, агенты, ограничения"), ("01_ДАННЫЕ.md", "Данные, дефекты, единицы, аудит справочника"),
        ("02_ОБОСНОВАНИЕ.md", "Обоснование решений, константы и их чувствительность"), ("03_РЕЗУЛЬТАТЫ.md", "Все числа прогона"),
        ("04_ДАЛЬНЕЙШИЕ_ШАГИ.md", "Открытые вопросы"), ("05_ИСТОЧНИКИ_ДОПУЩЕНИЙ.md", "Источники каждого допущения")]
    links = "".join(f'<li><a href="../docs/{_e(f)}">docs/{_e(f)}</a><span>{_e(d)}</span></li>' for f, d in docs_links)

    body = f"""
<section id="summary"><h2>Итоги</h2>
<div class="kpis">{kpis}</div>
<div class="two">{concl}</div></section>

<section id="architecture"><h2>Как устроена система</h2>
<p class="lead">Цепочка АВТ, гидроочистка дизельного топлива, блендинг. Пять агентов обмениваются сообщениями только через шину; оркестратор
выбирает вариант, перепроверяет его независимой проверкой и формирует карточку либо отказ. Журнал шины входит в каждую карточку.</p>
<div class="fig">{figs['arch']}</div></section>

<section id="sulfur"><h2>Оценка серы</h2>
<p class="lead">Фильтр Калмана объединяет два анализатора и лабораторию в оценку серы с интервалом неопределённости. Показания приборов
корректируются на смещение и запаздывание, лабораторный результат считается контрольным фактом.</p>
<div class="fig">{figs['series']}</div>
<div class="two"><div class="card"><div class="fig">{figs['auc']}</div></div>
<div class="card"><h3>Оценщики на отложенном периоде</h3>{_table(est_rows, ['Оценщик', 'AUC', 'MAE, мг/кг', 'Анализов'])}
<p class="note">Проверка на данных, отделённых по времени: подбор параметров только по обучающему периоду до 2025-01-01. Оценка строится по состоянию до самого анализа.</p></div></div></section>

<section id="risk"><h2>Вероятность превышения и сигнал о риске</h2>
<p class="lead">Расчётная вероятность превышения нормы отображается в наблюдаемую частоту превышений, и допустимость действия оценивается по ней.
Сигнал о риске (вероятность выше 1/2) слабо информативен на горизонте часов: различение падает с упреждением.</p>
<div class="two"><div class="card"><div class="fig">{figs.get('reliability', '')}</div></div>
<div class="card"><div class="fig">{figs.get('detect', '')}</div>{_table(det_rows, ['Упреждение, ч', 'Анализов', 'Доля с сигналом', 'Точность', 'Полнота', 'Сигналов с рекомендацией'])}</div></div></section>

<section id="backtest"><h2>Работа системы на отложенном периоде</h2>
<div class="fig">{figs['stack']}</div><div class="fig">{figs['ribbon']}</div>
<div class="two"><div class="card"><h3>Причины отказов</h3>{_table(fail_rows, ['Причина', 'Циклов'])}</div>
<div class="card"><h3>Показатели бэктеста</h3>{_table([[ 'Эскалаций технологу', _fmt_pct(bt.get('escalation_share')) ], ['Эпизодов действий в месяц', f"{bt.get('episodes_per_month', 0):.1f}"], ['Медианная длина эпизода, циклов', f"{bt.get('episode_median_length', 0):.0f}"]], ['Показатель', 'Значение'])}
<p class="note">Бэктест разомкнутый: рекомендации не применялись, поэтому по нему оценивается, что и когда система предлагает, а не результат применения.</p></div></div></section>

<section id="episode"><h2>Эпизод: оценка, анализы и решения</h2>
<p class="lead">Фрагмент отложенного периода с несколькими рекомендациями и превышениями по ЛИМС. Полоса «циклы» показывает решения системы каждые 6 ч
(зелёный - без изменений, синий - корректирующее действие, янтарный - отказ); при наведении показаны параметры действия.</p>
<div class="fig">{figs['episode']}</div></section>

{_effect_section(effect)}

<section id="demo"><h2>Демонстрационные сценарии</h2>
<p class="lead">Десять моментов из данных: спокойный режим, риск, неполные и аномальные данные, простой, блендинг. Раскройте сценарий, чтобы увидеть
карточку оператора: состояние, риск, действие, эффект, проверка ограничений, уверенность, объяснение, блендинг и журнал шины.</p>
{_demo_cards()}</section>

<section id="sources"><h2>Значения конфигурации и источники</h2>
<p class="lead">Каждое значение в <code>config/settings.yaml</code> имеет тип (<span class="tag">GOST</span>, <span class="tag">REQUIREMENT</span>, <span class="tag">TASK</span>,
<span class="tag">DATA</span>, <span class="tag">LIT</span>) и ссылку на источник; копии документов лежат в <code>docs/источники/</code>.</p>
<details class="src"><summary>Показать таблицу ({len(assumptions)} значений)</summary>
<input id="filter" type="search" placeholder="Фильтр по названию, типу или источнику" aria-label="Фильтр таблицы источников">{sources}</details></section>

<section id="repo"><h2>Где смотреть дальше</h2>
<ul class="links">{links}<li><a href="../README.md">README.md</a><span>Запуск, структура репозитория, сценарный блендинг</span></li></ul></section>
"""
    template = (ROOT / "mas" / "report" / "dashboard_template.html").read_text(encoding="utf-8")
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    page = template.replace("{{FIGCSS}}", fg.FIG_CSS).replace("{{BODY}}", body).replace("{{GENERATED}}", generated).replace("{{PERIOD}}", f"{t0:%Y-%m-%d} - {t1:%Y-%m-%d}")
    (OUTPUTS_DIR / "dashboard.html").write_text(page, encoding="utf-8")

    img_dir = DOCS_DIR / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    standalone = {
        "sulfur_series.svg": fg.sulfur_series(state, lab, running, t0, t1, limit, "Оценка серы фильтром и результаты ЛИМС на отложенном периоде", standalone=True),
        "status_ribbon.svg": fg.status_ribbon(cycles, "Самый значимый статус за сутки (действие, отказ, без изменений, не в работе)", standalone=True),
        "architecture.svg": fg.architecture(standalone=True),
        "episode.svg": fg.sulfur_series(state, lab, running, ep0, ep1, limit, f"Эпизод {ep0:%Y-%m-%d} - {ep1:%Y-%m-%d}: оценка, анализы и решения системы",
                                        resample="1h", markers=cycles[(cycles["t"] >= ep0) & (cycles["t"] <= ep1)], ymin=4.0, ymax=16.0, month_step=1, height=340, standalone=True),
    }
    if pc:
        standalone["reliability.svg"] = fg.reliability(pc["bins"], "Расчётная вероятность и частота превышений", standalone=True)
    if det:
        standalone["detection.svg"] = fg.grouped_bars([f"упреждение {d['lead_h']:g} ч" for d in det],
                                                       [("точность сигнала", [d["precision"] for d in det], "bar"), ("полнота сигнала", [d["recall"] for d in det], "bar2"),
                                                        ("доля анализов с сигналом", [d["share_flagged"] for d in det], "bar3")],
                                                       "Сигнал о риске и результат анализа ЛИМС", base=share_exc, ymax=0.4, standalone=True)
    for name, svg in standalone.items():
        (img_dir / name).write_text(svg, encoding="utf-8")
    return page
