"""CLI: prepare | calibrate | state | validate | vak | cycle | demo | backtest | docs | dashboard | all.

python run.py all --data <папка>   # полный воспроизводимый прогон
Папку данных можно также задать переменной окружения NEFTEKOD_DATA.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Callable

from mas.common import ensure_dirs, get_logger

log = get_logger("run")

def _set_data_dir(args: argparse.Namespace) -> None:
    if getattr(args, "data", None):
        os.environ["NEFTEKOD_DATA"] = args.data

def cmd_prepare(args: argparse.Namespace) -> None:
    from mas.prepare.pipeline import run_prepare
    _set_data_dir(args)
    log.info("prepare: %s", run_prepare())

def cmd_calibrate(args: argparse.Namespace) -> None:
    from mas.calibrate.pipeline import run_calibration
    if getattr(args, "global_search", False):
        from mas.calibrate.sulfur_filter import global_search
        log.info("calibrate: глобальный поиск параметров фильтра серы (десятки минут): %s", global_search())
    run_calibration()
    log.info("calibrate: config/calibrated.json записан")

def cmd_state(args: argparse.Namespace) -> None:
    from mas.common import load_calib
    from mas.state.builder import build_state
    build_state(load_calib())
    log.info("state: кэш причинных рядов состояния построен")

def cmd_validate(args: argparse.Namespace) -> None:
    from mas.evaluate.validation import validation_report
    log.info("validate: %s", validation_report().get("summary"))

def cmd_vak(args: argparse.Namespace) -> None:
    from mas.models.vak import evaluate_vak
    df = evaluate_vak()
    log.info("vak: %d сверок, пригодных %d", len(df), int((df["verdict"] == "пригодна").sum()))

def cmd_cycle(args: argparse.Namespace) -> None:
    import pandas as pd
    from mas.agents.orchestrator import Orchestrator
    rec = Orchestrator().cycle(pd.Timestamp(args.time), tag=args.tag, blend_mode=args.blend_mode)
    print(rec.blocks.get("7. Объяснение", ""))
    log.info("cycle %s: %s", args.time, rec.status)

def cmd_demo(args: argparse.Namespace) -> None:
    from mas.evaluate.demo import run_demo
    log.info("demo: %d сценариев", len(run_demo()))

def cmd_backtest(args: argparse.Namespace) -> None:
    from mas.evaluate.backtest import run_backtest
    _, summary = run_backtest(step_h=getattr(args, "step", 6))
    log.info("backtest: %s", summary)

def cmd_holdout(args: argparse.Namespace) -> None:
    from mas.evaluate.internal_holdout import run_internal_holdout
    report = run_internal_holdout(split=args.split)
    log.info("holdout: подбор до %s, проверка до %s: %s", args.split, report["check_period"]["end"],
             report["out_of_sample"]["estimators"]["Фильтр по двум приборам"])

def cmd_docs(args: argparse.Namespace) -> None:
    from mas.report.references_doc import build_references_doc
    from mas.report.results_doc import build_results_doc
    build_results_doc()
    build_references_doc()
    log.info("docs: docs/03_РЕЗУЛЬТАТЫ.md и docs/05_ИСТОЧНИКИ_ДОПУЩЕНИЙ.md обновлены")

def cmd_dashboard(args: argparse.Namespace) -> None:
    from mas.report.dashboard import build_dashboard
    build_dashboard()
    log.info("dashboard: outputs/dashboard.html записан")

def cmd_all(args: argparse.Namespace) -> None:
    for fn in (cmd_prepare, cmd_calibrate, cmd_state, cmd_validate, cmd_vak, cmd_demo):
        fn(args)
    cmd_backtest(args)
    cmd_docs(args)
    cmd_dashboard(args)

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="run.py", description="Мультиагентная система поддержки решений оператора")
    sub = p.add_subparsers(dest="command", required=True)

    def add(name: str, fn, needs_data: bool = False,
            extra: Callable[[argparse.ArgumentParser], None] | None = None) -> None:
        sp = sub.add_parser(name)
        if needs_data:
            sp.add_argument("--data", default=None, help="папка с файлами пакета данных")
        if extra:
            extra(sp)
        sp.set_defaults(func=fn)

    add("prepare", cmd_prepare, needs_data=True)
    add("calibrate", cmd_calibrate, extra=lambda sp: sp.add_argument(
        "--global-search", dest="global_search", action="store_true",
        help="заново найти стартовую точку настройки фильтра серы глобальным поиском (долго)"))
    add("state", cmd_state)
    add("validate", cmd_validate)
    add("vak", cmd_vak)
    add("cycle", cmd_cycle, extra=lambda sp: (
        sp.add_argument("--time", required=True),
        sp.add_argument("--tag", default=None),
        sp.add_argument("--blend-mode", dest="blend_mode", default=None)))
    add("demo", cmd_demo)
    add("backtest", cmd_backtest, extra=lambda sp: sp.add_argument("--step", type=int, default=6))
    add("holdout", cmd_holdout, extra=lambda sp: sp.add_argument(
        "--split", default="2024-01-01",
        help="дата разделения обучающего периода: подбор фильтра серы до неё, проверка - после (долго, не входит в `all`)"))
    add("docs", cmd_docs)
    add("dashboard", cmd_dashboard)
    add("all", cmd_all, needs_data=True)
    return p

def main(argv: list[str] | None = None) -> int:
    ensure_dirs()
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0

if __name__ == "__main__":
    sys.exit(main())
