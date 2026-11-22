"""只追加的锁定台账。

锁定记录保存当时采用的成本、偏好模型与系数区间、证据快照、容量快照、
承诺占用与保障检查。后来重新估计不会改动原记录；修订以新记录追加，并
标明它替代了哪一条、哪些项目仍可调整、哪些家庭受影响。
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from .comparability import evidence_summary
from .model import FitResult
from .procurement import (
    Assumptions,
    Basket,
    CapacityState,
    basket_cost,
    basket_slots_by_ward,
    benefit_interval,
    check_basket,
    evaluate_basket,
    rank_baskets,
    reserve_plan,
)
from .tables import Tables

LEDGER_VERSION = 1


def _canonical(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(payload: Any) -> str:
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


@dataclass
class Ledger:
    path: Path

    def __post_init__(self):
        self.path = Path(self.path)

    def entries(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records = []
        with self.path.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"台账第 {number} 行损坏：{error}") from error
                if record.get("ledger_version") != LEDGER_VERSION:
                    raise ValueError(f"台账第 {number} 行版本不受支持")
                expected = record.get("digest")
                body = {k: v for k, v in record.items() if k != "digest"}
                if expected != _digest(body):
                    raise ValueError(
                        f"台账第 {number} 行内容与登记摘要不符，疑似被覆盖或篡改"
                    )
                records.append(record)
        return records

    def append(self, record: dict[str, Any]) -> dict[str, Any]:
        body = {
            "ledger_version": LEDGER_VERSION,
            "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **record,
        }
        body["digest"] = _digest(body)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = _canonical(body) + "\n"
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
        return body

    def latest_by_lock(self) -> dict[str, dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        for record in self.entries():
            latest[record["lock_id"]] = record
        return latest

    def find(self, lock_id: str) -> dict[str, Any] | None:
        matches = [r for r in self.entries() if r["lock_id"] == lock_id]
        return matches[-1] if matches else None

    def history(self, lock_id: str) -> list[dict[str, Any]]:
        return [r for r in self.entries() if r["lock_id"] == lock_id]


def _model_snapshot(result: FitResult, draws: int) -> dict[str, Any]:
    from .model import draw_parameters

    samples = draw_parameters(result, draws)
    ranges = {}
    if samples:
        for index, name in enumerate(result.names):
            series = sorted(sample[index] for sample in samples)
            ranges[name] = {
                "low": _percentile(series, 0.1),
                "median": _percentile(series, 0.5),
                "high": _percentile(series, 0.9),
            }
    return {
        "spec": result.spec.key,
        "label": result.spec.label,
        "ridge": result.ridge,
        "coefficients": result.coefficients(),
        "standard_errors": dict(zip(result.names, result.standard_errors or [])),
        "draw_ranges": ranges,
        "draws": draws,
        "tasks": result.n_tasks,
        "households": result.n_households,
        "loglik": result.loglik,
    }


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        return float("nan")
    position = probability * (len(values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def lock_basket(
    ledger: Ledger,
    tables: Tables,
    result: FitResult,
    basket: Basket,
    assumptions: Assumptions,
    capacity: CapacityState,
    *,
    locked: Iterable[dict[str, Any]] = (),
    label: str = "",
    draws: int = 400,
    supersedes: str | None = None,
    reason: str = "",
) -> dict[str, Any]:
    """把一次采购决定写入台账；已存在同内容锁定时仍新建一条记录。"""
    check = check_basket(tables, basket, assumptions, capacity, locked)
    if not check["feasible"]:
        raise ValueError(
            "组合未通过检查，不能锁定：" + "、".join(check["blocking"])
        )
    evaluation = evaluate_basket(
        tables, result, basket, assumptions, capacity, locked
    )
    evidence = evidence_summary(tables, assumptions.as_of)
    reserve = reserve_plan(tables, assumptions)
    lock_id = f"L{len(ledger.entries()) + 1:03d}"
    record = {
        "record_type": "lock",
        "lock_id": lock_id,
        "label": label or lock_id,
        "supersedes": supersedes,
        "reason": reason,
        "assumptions": assumptions.as_dict(),
        "basket": basket.as_dict(),
        "basket_text": str(basket),
        "cost_cents": basket_cost(tables, basket),
        "slots_by_ward": basket_slots_by_ward(tables, basket),
        "checks": check["checks"],
        "capacity_snapshot": {
            f"{provider}/{period}": {
                "cap_units": units,
                "visible_from": capacity.latest_visible.get((provider, period)),
                "promised_units": capacity.promised.get((provider, period), 0),
                "locked_units": capacity.locked.get((provider, period), 0),
            }
            for (provider, period), units in sorted(capacity.units.items())
        },
        "promises": [
            {
                "promise": row["promise"],
                "household": row["household"],
                "ward": row["ward"],
                "provider": row["provider"],
                "period": row["period"],
                "committed_units": row["committed_units"],
                "option": row["option"],
            }
            for row in tables.promises
        ],
        "model": _model_snapshot(result, draws),
        "evidence_snapshot": {
            "comparable_cell_count": evidence["comparable_cell_count"],
            "by_ward": evidence["by_ward"],
            "coverage_gaps": evidence["coverage_gaps"],
        },
        "reserve": reserve,
        "benefit_interval": benefit_interval(
            tables,
            result,
            basket,
            assumptions,
            capacity,
            locked,
            draws=min(draws, 200),
        ),
        "valuation": {
            "totals": evaluation["totals"],
            "wards": {
                ward: {
                    "provided": row["provided"],
                    "demand": row["demand"],
                    "served": row["served"],
                    "unmet": row["unmet"],
                    "benefit": row["benefit"],
                    "served_total": row["served_total"],
                    "unused_slots": row["unused_slots"],
                }
                for ward, row in evaluation["wards"].items()
            },
        },
    }
    return ledger.append(record)


@dataclass
class Revision:
    original: dict[str, Any]
    as_of: str
    capacity: CapacityState
    adjustable: list[dict[str, Any]]
    fixed: list[dict[str, Any]]
    shortfalls: list[dict[str, Any]]
    affected_households: list[dict[str, Any]]
    guarantee_gaps: list[dict[str, Any]]
    alternatives: list[dict[str, Any]]
    explanation: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "original_lock": self.original["lock_id"],
            "as_of": self.as_of,
            "capacity": {
                f"{provider}/{period}": {
                    "cap_units": units,
                    "visible_from": self.capacity.latest_visible.get((provider, period)),
                    "promised_units": self.capacity.promised.get((provider, period), 0),
                    "locked_units_other": self.capacity.locked.get((provider, period), 0),
                }
                for (provider, period), units in sorted(self.capacity.units.items())
            },
            "adjustable": self.adjustable,
            "fixed": self.fixed,
            "shortfalls": self.shortfalls,
            "affected_households": self.affected_households,
            "guarantee_gaps": self.guarantee_gaps,
            "alternatives": self.alternatives,
            "explanation": self.explanation,
        }


def revise(
    tables: Tables,
    result: FitResult,
    original: dict[str, Any],
    assumptions: Assumptions,
    locked: Iterable[dict[str, Any]] = (),
    *,
    fixed_offers: Sequence[str] = (),
    limit: int = 5,
) -> Revision:
    """容量减量或问卷更正后，重新比较尚可调整的项目。

    ``fixed_offers`` 是已经开始执行、不能再动的订单；其余按新容量重排。
    原记录不变，修订写成新记录。容量口径：as_of 可见容量，先扣承诺，再扣
    其它已锁定记录；原锁定本身不预先扣除，因为它正被重新规划。
    """
    from .procurement import capacity_state

    other_locks = [
        entry for entry in locked if entry.get("lock_id") != original.get("lock_id")
    ]
    capacity = capacity_state(tables, assumptions.as_of, other_locks)
    basket = Basket(original["basket"])
    units = dict(basket.entries)
    fixed_set = set(fixed_offers)

    shortfalls = []
    usage: dict[tuple[str, str], dict[str, Any]] = {}
    for offer_id, packs in units.items():
        offer = next((row for row in tables.offers if row["offer"] == offer_id), None)
        if offer is None:
            continue
        key = (offer["provider"], offer["period"])
        entry = usage.setdefault(
            key,
            {
                "provider": key[0],
                "period": key[1],
                "units": 0,
                "offers": [],
                "fixed_units": 0,
                "adjustable_units": 0,
            },
        )
        used = packs * offer["period_units"]
        entry["units"] += used
        entry["offers"].append(offer_id)
        if offer_id in fixed_set:
            entry["fixed_units"] += used
        else:
            entry["adjustable_units"] += used
    for key, entry in sorted(usage.items()):
        total = capacity.units.get(key, 0)
        promised = capacity.promised.get(key, 0)
        other_locked = capacity.locked.get(key, 0)
        available = total - promised - other_locked
        if entry["units"] > available:
            shortfalls.append(
                {
                    **entry,
                    "available_units": available,
                    "cap_units": total,
                    "promised_units": promised,
                    "other_locked_units": other_locked,
                    "shortfall": entry["units"] - available,
                    "visible_from": capacity.latest_visible.get(key),
                }
            )
    shortfall_keys = {(row["provider"], row["period"]) for row in shortfalls}
    adjustable = [
        {"offer": offer, "packs": packs}
        for offer, packs in units.items()
        if offer not in fixed_set
    ]
    fixed = [
        {"offer": offer, "packs": packs}
        for offer, packs in units.items()
        if offer in fixed_set
    ]
    affected_promises = []
    for promise in tables.promises:
        if (promise["provider"], promise["period"]) in shortfall_keys:
            affected_promises.append(
                {
                    "promise": promise["promise"],
                    "household": promise["household"],
                    "ward": promise["ward"],
                    "provider": promise["provider"],
                    "period": promise["period"],
                    "option": promise["option"],
                    "committed_units": promise["committed_units"],
                    "issue": "新容量下承诺与已锁定占用之和可能挤占该班次",
                }
            )

    base_packs = {item["offer"]: item["packs"] for item in fixed}
    ranked = rank_baskets(
        tables,
        result,
        assumptions,
        capacity,
        other_locks,
        require_evidence=True,
        limit=limit * 4,
    )
    alternatives = []
    for item in ranked["top"]:
        candidate = dict(base_packs)
        for offer_id, packs in item["basket"].items():
            candidate[offer_id] = candidate.get(offer_id, 0) + packs
        basket_candidate = Basket(candidate)
        check = check_basket(
            tables, basket_candidate, assumptions, capacity, other_locks
        )
        if not check["feasible"]:
            continue
        evaluation = evaluate_basket(
            tables, result, basket_candidate, assumptions, capacity, other_locks
        )
        original_eval = evaluate_basket(
            tables, result, basket, assumptions, capacity, other_locks
        )
        loss = {}
        for ward, row in evaluation["wards"].items():
            before = original_eval["wards"][ward]["served_total"]
            loss[ward] = {
                "served_before_capacity_cut": before,
                "served_now": row["served_total"],
                "served_change": row["served_total"] - before,
                "unmet_now": row["unmet_total"],
            }
        alternatives.append(
            {
                "basket": basket_candidate.as_dict(),
                "basket_text": str(basket_candidate),
                "fixed_included": base_packs,
                "cost_cents": evaluation["totals"]["cost_cents"],
                "benefit": evaluation["totals"]["benefit"],
                "served": evaluation["totals"]["served"],
                "wards": {
                    ward: {
                        "served_total": row["served_total"],
                        "benefit": row["benefit"],
                        "unmet_total": row["unmet_total"],
                    }
                    for ward, row in evaluation["wards"].items()
                },
                "served_change_by_ward": loss,
            }
        )
        if len(alternatives) >= limit:
            break

    minimum = {row["ward"]: row["minimum_families"] for row in tables.policy}
    chosen = alternatives[0] if alternatives else {
        "basket": basket.as_dict()
    }
    slots = basket_slots_by_ward(tables, Basket(chosen["basket"]))
    guarantee_gaps = [
        {
            "ward": ward,
            "minimum_families": floor,
            "planned_slots": sum(slots.get(ward, {}).values()),
            "gap": max(0, floor - sum(slots.get(ward, {}).values())),
            "note": "需另行补足保障，不能用未经证据支持的选项替代",
        }
        for ward, floor in sorted(minimum.items())
        if sum(slots.get(ward, {}).values()) < floor
    ]
    affected_households: list[dict[str, Any]] = list(affected_promises)
    if alternatives:
        base = alternatives[0]["served_change_by_ward"]
        for ward, info in base.items():
            if info["served_change"] < -1e-9:
                affected_households.append(
                    {
                        "ward": ward,
                        "household": "（未逐一列名）",
                        "issue": f"预计少获得服务的户数约 {-info['served_change']:.1f}",
                    }
                )
    explanation = (
        "原锁定记录保持原样；本次只重排可调整项目。"
        + (
            "存在容量缺口，需另行补足保障或减少可调整包数。"
            if shortfalls
            else "新容量下未发现容量缺口。"
        )
        + (
            " 没有任何替代组合能同时满足容量与保障底线，应先补足承诺服务。"
            if not alternatives
            else ""
        )
    )
    return Revision(
        original=original,
        as_of=assumptions.as_of,
        capacity=capacity,
        adjustable=adjustable,
        fixed=fixed,
        shortfalls=shortfalls,
        affected_households=affected_households,
        guarantee_gaps=guarantee_gaps,
        alternatives=alternatives,
        explanation=explanation,
    )


def lock_summary(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "lock_id": record["lock_id"],
        "label": record.get("label"),
        "recorded_at": record.get("recorded_at"),
        "supersedes": record.get("supersedes"),
        "reason": record.get("reason"),
        "basket": record["basket"],
        "basket_text": record["basket_text"],
        "cost_cents": record["cost_cents"],
        "as_of": record["assumptions"]["as_of"],
        "spec": record["model"]["spec"],
        "ridge": record["model"]["ridge"],
        "checks": [
            {"code": check["code"], "ok": check["ok"], "message": check["message"]}
            for check in record["checks"]
        ],
        "reserve_cents": record["reserve"]["reserve_cents"],
        "benefit": record["valuation"]["totals"]["benefit"],
        "served": record["valuation"]["totals"]["served"],
        "digest": record["digest"],
    }
