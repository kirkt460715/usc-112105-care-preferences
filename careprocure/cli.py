"""命令行入口：检查资料、比较模型、评估候选订单、锁定与修订。

示例：

    python3 -m careprocure.cli audit --data data
    python3 -m careprocure.cli report --data data --as-of 2026-10-25 \
        --spec B --budget 515000 --reserve-per-gap 25000 --output outputs
    python3 -m careprocure.cli candidates --data data --as-of 2026-10-25 \
        --basket "o3x1,o5x2,o6x1,o7x1"
    python3 -m careprocure.cli lock --data data --as-of 2026-10-25 \
        --basket "o3x1,o5x2,o6x1,o7x1" --ledger locks/ledger.jsonl --label 初版
    python3 -m careprocure.cli revise --data data --as-of 2026-10-25 \
        --lock L001 --ledger locks/ledger.jsonl --fixed o3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .comparability import display_comparability, evidence_summary
from .demand import demand_by_segment, extrapolation_report, substitution_matrix
from .locks import Ledger, lock_basket, lock_summary, revise
from .model import SPECS, baseline_metrics, cross_validate, fit
from .procurement import (
    Assumptions,
    Basket,
    capacity_state,
    conflict_pairs,
    evaluate_basket,
    rank_baskets,
    reserve_plan,
)
from .report import build_assets, write_outputs
from .states import audit_counts, resolve_answers
from .tables import MaterialError, read_tables


def _parse_basket(text: str) -> Basket:
    packs: dict[str, int] = {}
    for part in text.replace("，", ",").split(","):
        part = part.strip()
        if not part:
            continue
        for separator in ("x", "X", "×", "*"):
            if separator in part:
                offer, count = part.split(separator, 1)
                packs[offer.strip()] = int(count)
                break
        else:
            packs[part] = 1
    return Basket(packs)


def _assumptions(args) -> Assumptions:
    return Assumptions(
        as_of=args.as_of,
        spec_key=args.spec,
        ridge=SPECS[args.spec].ridge,
        budget_cents=args.budget,
        reserve_per_gap_cents=args.reserve_per_gap,
        model_label=SPECS[args.spec].label,
    )


def _load(args):
    return read_tables(args.data)


def _emit(payload, path: str | None = None) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text + "\n", encoding="utf-8")
        print(path)
    else:
        print(text)


def cmd_audit(args) -> int:
    tables = _load(args)
    answers = resolve_answers(tables, args.as_of)
    _emit(
        {
            "as_of": args.as_of,
            "audit": audit_counts(answers),
            "evidence": evidence_summary(tables, args.as_of),
            "comparability": display_comparability(tables, args.as_of),
        },
        args.output,
    )
    return 0


def cmd_models(args) -> int:
    tables = _load(args)
    payload = {
        "baselines": baseline_metrics(tables),
        "specs": [],
    }
    for key in args.specs.split(","):
        key = key.strip()
        if not key:
            continue
        cv = cross_validate(tables, SPECS[key])
        fitted = fit(tables, SPECS[key])
        payload["specs"].append(
            {
                "spec": key,
                "label": SPECS[key].label,
                "description": SPECS[key].description,
                "ridge": fitted.ridge,
                "coefficients": fitted.coefficients(),
                "standard_errors": dict(zip(fitted.names, fitted.standard_errors or [])),
                "household": cv["household"],
                "form": cv["form"],
                "plan": cv["plan"],
            }
        )
    _emit(payload, args.output)
    return 0


def cmd_demand(args) -> int:
    tables = _load(args)
    fitted = fit(tables, SPECS[args.spec])
    payload = {
        "primary": {
            "spec": args.spec,
            "coefficients": fitted.coefficients(),
        },
        "demand": demand_by_segment(tables, fitted, draws=args.draws),
        "substitution": {
            ward: substitution_matrix(tables, fitted, ward)
            for ward in sorted({row["ward"] for row in tables.sample})
        },
        "extrapolation_warnings": extrapolation_report(tables),
    }
    _emit(payload, args.output)
    return 0


def cmd_candidates(args) -> int:
    tables = _load(args)
    assumptions = _assumptions(args)
    fitted = fit(tables, SPECS[args.spec])
    capacity = capacity_state(tables, args.as_of)
    ranking = rank_baskets(
        tables,
        fitted,
        assumptions,
        capacity,
        require_evidence=not args.allow_unevidenced,
        limit=args.limit,
    )
    payload = {
        "ranking": ranking,
        "conflicts": conflict_pairs(tables, assumptions, capacity),
        "reserve": reserve_plan(tables, assumptions),
        "submitted": None,
    }
    if args.basket:
        basket = _parse_basket(args.basket)
        evaluation = evaluate_basket(
            tables, fitted, basket, assumptions, capacity
        )
        payload["submitted"] = evaluation
    _emit(payload, args.output)
    return 0


def cmd_lock(args) -> int:
    tables = _load(args)
    assumptions = _assumptions(args)
    fitted = fit(tables, SPECS[args.spec])
    capacity = capacity_state(tables, args.as_of)
    ledger = Ledger(args.ledger)
    record = lock_basket(
        ledger,
        tables,
        fitted,
        _parse_basket(args.basket),
        assumptions,
        capacity,
        label=args.label,
        draws=args.draws,
        supersedes=args.supersedes,
        reason=args.reason,
    )
    _emit({"lock": lock_summary(record)}, args.output)
    return 0


def cmd_revise(args) -> int:
    tables = _load(args)
    assumptions = _assumptions(args)
    fitted = fit(tables, SPECS[args.spec])
    ledger = Ledger(args.ledger)
    original = ledger.find(args.lock)
    if original is None:
        print(f"台账里没有锁定 {args.lock}", file=sys.stderr)
        return 2
    revision = revise(
        tables,
        fitted,
        original,
        assumptions,
        ledger.entries(),
        fixed_offers=[part.strip() for part in args.fixed.split(",") if part.strip()],
        limit=args.limit,
    )
    payload = revision.as_dict()
    if args.relock:
        chosen = revision.alternatives[0]["basket"] if revision.alternatives else None
        if not chosen:
            print("没有可行修订组合，未写入新锁定", file=sys.stderr)
            return 1
        # 新记录替代原锁定：原锁定的占用不能被重复计入，否则新组合会被自己挤掉。
        prior = [entry for entry in ledger.entries() if entry["lock_id"] != args.lock]
        capacity = capacity_state(tables, args.as_of, prior)
        record = lock_basket(
            ledger,
            tables,
            fitted,
            Basket(chosen),
            assumptions,
            capacity,
            locked=prior,
            label=args.label or f"{args.lock} 修订",
            supersedes=args.lock,
            reason=args.reason or "容量或问卷更正后的修订",
        )
        payload["new_lock"] = lock_summary(record)
    _emit(payload, args.output)
    return 0


def cmd_report(args) -> int:
    tables = _load(args)
    assumptions = _assumptions(args)
    ledger = Ledger(args.ledger) if args.ledger else None
    entries = ledger.entries() if ledger is not None else []
    locked = entries
    assets = build_assets(
        tables,
        assumptions,
        draws=args.draws,
        ledger_entries=entries,
        locked=locked,
        limit=args.limit,
    )
    paths = write_outputs(assets, args.output)
    print(json.dumps({key: str(path) for key, path in paths.items()}, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="careprocure",
        description="家庭照护支持采购：证据、需求与组合",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(sub, needs_assumptions: bool = False):
        sub.add_argument("--data", default="data", type=Path)
        sub.add_argument("--as-of", default="2026-10-25")
        sub.add_argument("--output", default=None)
        if needs_assumptions:
            sub.add_argument("--spec", default="B", choices=sorted(SPECS))
            sub.add_argument("--budget", type=int, default=515000)
            sub.add_argument("--reserve-per-gap", type=int, default=25000)
            sub.add_argument("--draws", type=int, default=300)

    audit = subparsers.add_parser("audit", help="展示、状态与证据门槛")
    common(audit)
    audit.set_defaults(func=cmd_audit)

    models = subparsers.add_parser("models", help="模型比较与留出验证")
    common(models)
    models.add_argument("--specs", default="A,B,C")
    models.set_defaults(func=cmd_models)

    demand = subparsers.add_parser("demand", help="分群需求与替代关系")
    common(demand)
    demand.add_argument("--spec", default="B", choices=sorted(SPECS))
    demand.add_argument("--draws", type=int, default=300)
    demand.set_defaults(func=cmd_demand)

    candidates = subparsers.add_parser("candidates", help="候选订单与约束检查")
    common(candidates, needs_assumptions=True)
    candidates.add_argument("--basket", default=None)
    candidates.add_argument("--limit", type=int, default=10)
    candidates.add_argument("--allow-unevidenced", action="store_true")
    candidates.set_defaults(func=cmd_candidates)

    lock = subparsers.add_parser("lock", help="锁定一轮采购")
    common(lock, needs_assumptions=True)
    lock.add_argument("--basket", required=True)
    lock.add_argument("--ledger", default="locks/ledger.jsonl")
    lock.add_argument("--label", default="")
    lock.add_argument("--supersedes", default=None)
    lock.add_argument("--reason", default="")
    lock.set_defaults(func=cmd_lock)

    revise = subparsers.add_parser("revise", help="容量减量或问卷更正后的重排")
    common(revise, needs_assumptions=True)
    revise.add_argument("--ledger", default="locks/ledger.jsonl")
    revise.add_argument("--lock", required=True)
    revise.add_argument("--fixed", default="")
    revise.add_argument("--limit", type=int, default=5)
    revise.add_argument("--relock", action="store_true")
    revise.add_argument("--label", default="")
    revise.add_argument("--reason", default="")
    revise.set_defaults(func=cmd_revise)

    report = subparsers.add_parser("report", help="生成采购会与分区说明")
    common(report, needs_assumptions=True)
    report.add_argument("--ledger", default=None)
    report.add_argument("--limit", type=int, default=10)
    report.set_defaults(func=cmd_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (MaterialError, ValueError) as error:
        print(f"失败：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
