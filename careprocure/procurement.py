"""采购组合：整包订单、最低开办量、共用班次容量、预算与保障底线。

口径：

- 订单以“包”为单位，每包提供 ``bundles`` 个同类型家用名额，占用
  ``period_units`` 个承接点公共班次；``ties`` 同组的包开办数量必须一致。
- ``provider_caps`` 取 ``as_of`` 当日已经可见的最新版本；供给减量从
  ``seen_on`` 起生效，不能拿旧的高容量做承诺。
- 已答应家庭的过渡服务（``promises``）先占容量，任何新购不得挤占。
- 此前锁定的订单另行拨款，但同样占用班次容量。
- 预算只用于本轮新购；缺乏可比证据的选项只能列预算预留补充访问，
  不得计入已估收益。
"""
from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .comparability import evidence_cells
from .demand import (
    prepare_menu,
    scenario_gain,
    scenario_gain_by_form,
    ward_choice_sets,
)
from .model import FitResult
from .tables import Tables


@dataclass(frozen=True)
class Basket:
    """订单组合：``packs`` 以 offer 编号为键，值为开办包数。"""

    packs: Mapping[str, int]

    def normalized(self) -> "Basket":
        return Basket({key: value for key, value in sorted(self.packs.items()) if value})

    @property
    def entries(self) -> list[tuple[str, int]]:
        return sorted(self.normalized().packs.items())

    def as_dict(self) -> dict[str, int]:
        return dict(self.entries)

    def signature(self) -> tuple[tuple[str, int], ...]:
        return tuple(self.entries)

    def __str__(self) -> str:
        if not self.packs:
            return "空组合"
        return " + ".join(f"{offer}×{packs}" for offer, packs in self.entries)


@dataclass(frozen=True)
class Assumptions:
    as_of: str
    spec_key: str
    ridge: float | None
    budget_cents: int
    reserve_per_gap_cents: int = 0
    keep_promises: bool = True
    count_locked_units: bool = True
    enforce_ties: bool = True
    enforce_minimum_families: bool = True
    enforce_max_packs: bool = True
    model_label: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "as_of": self.as_of,
            "spec": self.spec_key,
            "model": self.model_label,
            "ridge": self.ridge,
            "budget_cents": self.budget_cents,
            "reserve_per_gap_cents": self.reserve_per_gap_cents,
            "keep_promises": self.keep_promises,
            "count_locked_units": self.count_locked_units,
            "enforce_ties": self.enforce_ties,
            "enforce_minimum_families": self.enforce_minimum_families,
            "enforce_max_packs": self.enforce_max_packs,
        }


@dataclass
class OfferIndex:
    offers: dict[str, dict[str, Any]]
    by_ward: dict[str, list[str]]
    by_provider: dict[tuple[str, str], list[str]]


def index_offers(tables: Tables) -> OfferIndex:
    offers = {row["offer"]: row for row in tables.offers}
    by_ward: dict[str, list[str]] = defaultdict(list)
    by_provider: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in tables.offers:
        by_ward[row["ward"]].append(row["offer"])
        by_provider[(row["provider"], row["period"])].append(row["offer"])
    return OfferIndex(
        offers=offers,
        by_ward={key: sorted(value) for key, value in by_ward.items()},
        by_provider={key: sorted(value) for key, value in by_provider.items()},
    )


@dataclass
class CapacityState:
    units: dict[tuple[str, str], int]
    source: dict[tuple[str, str], dict[str, Any]]
    promised: dict[tuple[str, str], int]
    locked: dict[tuple[str, str], int]
    latest_visible: dict[tuple[str, str], str]

    def residual(self, provider_period: tuple[str, str], pins: int) -> int:
        return self.units.get(provider_period, 0) - pins


def capacity_state(
    tables: Tables,
    as_of: str,
    locked: Iterable[dict[str, Any]] = (),
    include_promises: bool = True,
    include_locked: bool = True,
) -> CapacityState:
    """取 as_of 可见的最新容量，并扣掉承诺与已锁定订单的占用。"""
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in tables.provider_caps:
        if row["seen_on"] > as_of:
            continue
        key = (row["provider"], row["period"])
        if key not in latest or row["seen_on"] > latest[key]["seen_on"]:
            latest[key] = row
    for offer in tables.offers:
        key = (offer["provider"], offer["period"])
        if key not in latest:
            raise ValueError(
                f"承接点 {offer['provider']}/{offer['period']} 在 {as_of} 前没有登记容量"
            )
    units = {key: row["units"] for key, row in latest.items()}
    promised: dict[tuple[str, str], int] = defaultdict(int)
    if include_promises:
        for row in tables.promises:
            key = (row["provider"], row["period"])
            promised[key] += row["committed_units"]
    locked_units: dict[tuple[str, str], int] = defaultdict(int)
    if include_locked:
        # 只计仍在执行的锁定：被修订替代的旧记录不重复占容量。
        superseded = {
            entry.get("supersedes")
            for entry in locked
            if entry.get("record_type") == "lock" and entry.get("supersedes")
        }
        latest_by_lock: dict[str, dict[str, Any]] = {}
        for entry in locked:
            if entry.get("record_type") != "lock":
                continue
            latest_by_lock[entry["lock_id"]] = entry
        for lock_id, entry in latest_by_lock.items():
            if lock_id in superseded:
                continue
            basket = entry.get("basket", {})
            for offer_id, packs in basket.items():
                offer = next((o for o in tables.offers if o["offer"] == offer_id), None)
                if offer is None:
                    continue
                key = (offer["provider"], offer["period"])
                locked_units[key] += int(packs) * offer["period_units"]
    return CapacityState(
        units=units,
        source=latest,
        promised=dict(promised),
        locked=dict(locked_units),
        latest_visible={key: row["seen_on"] for key, row in latest.items()},
    )


@dataclass
class Check:
    code: str
    ok: bool
    message: str
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "ok": self.ok,
            "message": self.message,
            "detail": self.detail,
        }


def evidence_by_offer(tables: Tables, as_of: str) -> dict[str, dict[str, Any]]:
    """每个订单选项在本区的人群证据概况。"""
    cells = evidence_cells(tables, as_of)
    output: dict[str, dict[str, Any]] = {}
    for offer in tables.offers:
        relevant = [
            cell
            for cell in cells
            if cell.ward == offer["ward"] and cell.option == offer["option"]
        ]
        comparable = [cell for cell in relevant if cell.comparable]
        output[offer["offer"]] = {
            "ward": offer["ward"],
            "option": offer["option"],
            "comparable_strata": [cell.stratum for cell in comparable],
            "below_threshold_strata": [
                {
                    "stratum": cell.stratum,
                    "choice_tasks": cell.choice_tasks,
                    "evidence_min": cell.evidence_min,
                }
                for cell in relevant
                if not cell.comparable
            ],
            "chosen": sum(cell.chosen for cell in relevant),
            "choice_tasks": sum(cell.choice_tasks for cell in relevant),
            "has_any_comparable_evidence": bool(comparable),
        }
    return output


def basket_cost(tables: Tables, basket: Basket) -> int:
    offers = {row["offer"]: row for row in tables.offers}
    return sum(offers[offer]["cost_cents"] * packs for offer, packs in basket.entries)


def basket_units(tables: Tables, basket: Basket) -> dict[tuple[str, str], int]:
    offers = {row["offer"]: row for row in tables.offers}
    units: dict[tuple[str, str], int] = defaultdict(int)
    for offer_id, packs in basket.entries:
        offer = offers[offer_id]
        units[(offer["provider"], offer["period"])] += packs * offer["period_units"]
    return dict(units)


def basket_slots_by_ward(tables: Tables, basket: Basket) -> dict[str, dict[str, int]]:
    """每区每个选项的家用名额。"""
    offers = {row["offer"]: row for row in tables.offers}
    slots: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for offer_id, packs in basket.entries:
        offer = offers[offer_id]
        slots[offer["ward"]][offer["option"]] += packs * offer["bundles"]
    return {ward: dict(options) for ward, options in slots.items()}


def check_basket(
    tables: Tables,
    basket: Basket,
    assumptions: Assumptions,
    capacity: CapacityState,
    locked: Iterable[dict[str, Any]] = (),
) -> dict[str, Any]:
    """结构、容量、预算与保障检查；返回逐条可解释的结论。"""
    index = index_offers(tables)
    checks: list[Check] = []
    basket = basket.normalized()
    unknown = [offer for offer, _ in basket.entries if offer not in index.offers]
    checks.append(
        Check(
            "offer_ids",
            not unknown,
            "订单编号均可识别" if not unknown else f"未知订单：{unknown}",
            {"unknown": unknown},
        )
    )
    zero = [offer for offer, packs in basket.entries if packs == 0]
    checks.append(
        Check("nonzero", not zero, "包数均为正" if not zero else f"包数为零：{zero}", {})
    )
    over: list[dict[str, Any]] = []
    if assumptions.enforce_max_packs:
        for offer_id, packs in basket.entries:
            offer = index.offers.get(offer_id)
            if offer is None:
                continue
            if packs > offer["max_packs"]:
                over.append(
                    {
                        "offer": offer_id,
                        "packs": packs,
                        "max_packs": offer["max_packs"],
                    }
                )
    checks.append(
        Check(
            "max_packs",
            not over,
            "各订单开办量未超过上限" if not over else "有订单超过最高开办包数",
            {"violations": over},
        )
    )
    tie_violations: list[dict[str, Any]] = []
    groups: dict[str, list[str]] = defaultdict(list)
    for row in tables.offers:
        if row["ties"]:
            groups[row["ties"]].append(row["offer"])
    for tie, offer_ids in groups.items():
        counts = {offer_id: basket.as_dict().get(offer_id, 0) for offer_id in sorted(offer_ids)}
        if len(set(counts.values())) > 1:
            tie_violations.append(
                {
                    "tie": tie,
                    "offers": counts,
                    "message": "同组必须同包数；只能组合采购",
                }
            )
    if not assumptions.enforce_ties:
        tie_violations = []
    checks.append(
        Check(
            "ties",
            not tie_violations,
            "捆绑组包数一致" if not tie_violations else "捆绑组包数不一致",
            {"violations": tie_violations},
        )
    )

    units = basket_units(tables, basket)
    capacity_rows = []
    capacity_ok = True
    for key, used in sorted(units.items()):
        total = capacity.units.get(key, 0)
        promised = capacity.promised.get(key, 0)
        locked_units = capacity.locked.get(key, 0)
        available = total - promised - locked_units
        ok = used <= available
        capacity_ok = capacity_ok and ok
        capacity_rows.append(
            {
                "provider": key[0],
                "period": key[1],
                "cap_units": total,
                "promised_units": promised,
                "locked_units": locked_units,
                "available_units": available,
                "basket_units": used,
                "ok": ok,
                "cap_visible_from": capacity.latest_visible.get(key),
            }
        )
    checks.append(
        Check(
            "provider_capacity",
            capacity_ok,
            "班次容量足够，已承诺与已锁定部分已先扣除"
            if capacity_ok
            else "新购会挤占已承诺或已锁定的班次容量",
            {"rows": capacity_rows},
        )
    )

    cost = basket_cost(tables, basket)
    reserve = reserve_plan(tables, assumptions)
    # 预留与承诺只从总预算里各扣一次：净可承购预算 = 总预算 − 预留。
    # 已锁定项目另行拨款，不在此处再扣一次（它们只占容量，见上）。
    purchasable = assumptions.budget_cents - reserve["reserve_cents"]
    within_budget = cost <= purchasable
    checks.append(
        Check(
            "budget",
            within_budget,
            f"本轮新购费用 {cost} 分，在可承购预算 {purchasable} 分以内"
            if within_budget
            else f"本轮新购费用 {cost} 分，超出可承购预算 {purchasable} 分"
            f"（总预算 {assumptions.budget_cents} 分，已预留 {reserve['reserve_cents']} 分补充访问）",
            {
                "cost_cents": cost,
                "budget_cents": assumptions.budget_cents,
                "reserve_cents": reserve["reserve_cents"],
                "purchasable_budget_cents": purchasable,
            },
        )
    )

    slots = basket_slots_by_ward(tables, basket)
    minimum = {row["ward"]: row["minimum_families"] for row in tables.policy}
    minimum_rows = []
    minimum_ok = True
    for ward, floor in sorted(minimum.items()):
        got = sum(slots.get(ward, {}).values())
        ok = got >= floor or not assumptions.enforce_minimum_families
        minimum_ok = minimum_ok and ok
        minimum_rows.append(
            {
                "ward": ward,
                "minimum_families": floor,
                "purchased_slots": got,
                "ok": ok,
                "gap": max(0, floor - got),
            }
        )
    checks.append(
        Check(
            "minimum_families",
            minimum_ok,
            "各区达到最低支持户数" if minimum_ok else "有区未达到最低支持户数",
            {"rows": minimum_rows},
        )
    )

    evidence = evidence_by_offer(tables, assumptions.as_of)
    unevidenced = [
        {
            "offer": offer_id,
            "packs": packs,
            "ward": evidence[offer_id]["ward"],
            "option": evidence[offer_id]["option"],
            "reason": "本区没有任何达到门槛的可比证据；只能列为补充访问预留，不能算已估收益",
        }
        for offer_id, packs in basket.entries
        if offer_id in evidence and not evidence[offer_id]["has_any_comparable_evidence"]
    ]
    checks.append(
        Check(
            "evidence",
            not unevidenced,
            "所有新购订单在本区都有可比证据"
            if not unevidenced
            else "有订单缺少本区可比证据，不能作为已估收益的正式承购",
            {"unevidenced": unevidenced},
        )
    )

    all_ok = all(check.ok for check in checks)
    return {
        "basket": basket.as_dict(),
        "cost_cents": cost,
        "units": {f"{key[0]}/{key[1]}": value for key, value in sorted(units.items())},
        "slots_by_ward": slots,
        "checks": [check.as_dict() for check in checks],
        "feasible": all_ok,
        "blocking": [check.code for check in checks if not check.ok],
    }


def evaluate_basket(
    tables: Tables,
    result: FitResult,
    basket: Basket,
    assumptions: Assumptions,
    capacity: CapacityState,
    locked: Iterable[dict[str, Any]] = (),
    beta: Sequence[float] | None = None,
) -> dict[str, Any]:
    """在容量约束下估计各区需求、已服务与相对现状的增益。"""
    check = check_basket(tables, basket, assumptions, capacity, locked)
    evidence = evidence_by_offer(tables, assumptions.as_of)
    choice_sets = ward_choice_sets(tables, assumptions.as_of)
    slots = basket_slots_by_ward(tables, basket)
    wards: dict[str, Any] = {}
    total_benefit = 0.0
    total_served = 0.0
    total_demand = 0.0
    for ward in sorted({row["ward"] for row in tables.sample}):
        ward_slots = slots.get(ward, {})
        provided = [option for option in ward_slots if option in choice_sets[ward].options]
        excluded = sorted(set(ward_slots) - set(provided))
        menu = prepare_menu(tables, result, ward, beta=beta)
        scenario = scenario_gain_by_form(
            tables,
            result,
            ward,
            provided,
            beta=beta,
            slots=ward_slots,
            as_of=assumptions.as_of,
        )
        union_scenario = scenario_gain(
            tables, result, ward, provided, beta=beta, slots=ward_slots
        )
        unused = {
            option: max(
                0.0,
                ward_slots[option] - scenario["served"].get(option, 0.0),
            )
            for option in provided
        }
        benefit = scenario["benefit"]
        total_benefit += benefit
        total_served += scenario["served_total"]
        total_demand += scenario["demand_total"]
        stratum_rows = {}
        for stratum, info in menu.strata.items():
            probabilities = info["probabilities"]
            served_fraction = {}
            for option in provided:
                demand = info["weight"] * probabilities.get(option, 0.0)
                served_value = scenario["served"].get(option, 0.0)
                form_demand = sum(
                    form["demand"].get(option, 0.0) for form in scenario.get("forms", [])
                )
                served_fraction[option] = (
                    served_value / form_demand if form_demand else 0.0
                )
            stratum_rows[stratum] = {
                "population_estimate": info["weight"],
                "demand": {
                    option: info["weight"] * probabilities.get(option, 0.0)
                    for option in provided
                },
                "served": {
                    option: info["weight"]
                    * probabilities.get(option, 0.0)
                    * served_fraction.get(option, 0.0)
                    for option in provided
                },
            }
        wards[ward] = {
            "provided": provided,
            "excluded_no_local_evidence": excluded,
            "slots": ward_slots,
            "demand": scenario["demand"],
            "served": scenario["served"],
            "unmet": scenario["unmet"],
            "demand_total": scenario["demand_total"],
            "served_total": scenario["served_total"],
            "unmet_total": scenario["unmet_total"],
            "unused_slots": unused,
            "unused_slots_total": sum(unused.values()),
            "benefit": benefit,
            "benefit_per_served": scenario["benefit_per_served"],
            "population_estimate": scenario["population_estimate"],
            "statusquo_households": scenario["statusquo_households"],
            "valuation_mode": "per-display-form",
            "union_menu_benefit": union_scenario["benefit"],
            "union_menu_note": (
                "把全区选项并集当作一个菜单的偏乐观口径；每户实际只看到 3 个选项，"
                "报告主值采用版本菜单口径。"
            ),
            "forms": scenario.get("forms", []),
            "by_stratum": stratum_rows,
            "evidence_by_option": {
                option: {
                    "comparable_strata": sorted(
                        {
                            stratum
                            for offer_id, info in evidence.items()
                            if info["ward"] == ward
                            and info["option"] == option
                            for stratum in info["comparable_strata"]
                        }
                    ),
                    "offers": sorted(
                        offer_id
                        for offer_id, info in evidence.items()
                        if info["ward"] == ward and info["option"] == option
                    ),
                }
                for option in ward_slots
            },
            "excluded_reason": {
                option: "本区展示方案未包含此项，模型份额属外推，不计入已估收益"
                for option in excluded
            },
        }
    cost = basket_cost(tables, basket)
    return {
        "assumptions": assumptions.as_dict(),
        "basket": basket.as_dict(),
        "basket_text": str(basket),
        "check": check,
        "wards": wards,
        "totals": {
            "benefit": total_benefit,
            "served": total_served,
            "demand": total_demand,
            "unmet": total_demand - total_served,
            "cost_cents": cost,
            "benefit_per_served": total_benefit / total_served if total_served else 0.0,
            "cost_per_served_cents": cost / total_served if total_served else None,
            "population_estimate": sum(
                row["population_estimate"] for row in wards.values()
            ),
        },
    }


def enumerate_baskets(
    tables: Tables,
    assumptions: Assumptions,
    capacity: CapacityState,
    locked: Iterable[dict[str, Any]] = (),
    *,
    require_evidence: bool = True,
) -> list[Basket]:
    """枚举满足结构、容量、预算与保障底线的订单组合。

    ``require_evidence`` 为真时排除本区没有可比证据的订单——这些只能走
    补充访问预留，不能作为正式承购。
    """
    index = index_offers(tables)
    evidence = evidence_by_offer(tables, assumptions.as_of)
    eligible = []
    for row in tables.offers:
        if require_evidence and not evidence[row["offer"]]["has_any_comparable_evidence"]:
            continue
        eligible.append(row["offer"])
    ranges = [range(index.offers[offer]["max_packs"] + 1) for offer in eligible]
    baskets: list[Basket] = []
    for combination in itertools.product(*ranges):
        packs = {offer: count for offer, count in zip(eligible, combination) if count}
        # 捆绑组：未列出的同组订单按 0 处理，需显式补齐再检查
        basket = Basket(packs).normalized()
        check = check_basket(tables, basket, assumptions, capacity, locked)
        if check["feasible"]:
            baskets.append(basket)
    return baskets


def conflict_pairs(
    tables: Tables,
    assumptions: Assumptions,
    capacity: CapacityState,
    locked: Iterable[dict[str, Any]] = (),
) -> list[dict[str, Any]]:
    """单独可行、合在一起会挤占承诺容量的订单对。

    这是“新增小包单独效果很好、合起来抢走另一区已承诺夜间份额”的检测口径：
    在保留承诺的前提下，两者分别可行但不是共同可行。
    """
    index = index_offers(tables)
    evidence = evidence_by_offer(tables, assumptions.as_of)
    capacity_only = Assumptions(
        as_of=assumptions.as_of,
        spec_key=assumptions.spec_key,
        ridge=assumptions.ridge,
        budget_cents=assumptions.budget_cents,
        reserve_per_gap_cents=assumptions.reserve_per_gap_cents,
        keep_promises=assumptions.keep_promises,
        count_locked_units=assumptions.count_locked_units,
        enforce_ties=assumptions.enforce_ties,
        enforce_minimum_families=False,
        enforce_max_packs=assumptions.enforce_max_packs,
        model_label=assumptions.model_label,
    )
    single: list[str] = []
    for row in tables.offers:
        if not evidence[row["offer"]]["has_any_comparable_evidence"]:
            continue
        basket = Basket({row["offer"]: 1})
        if check_basket(tables, basket, capacity_only, capacity, locked)["feasible"]:
            single.append(row["offer"])
    conflicts: list[dict[str, Any]] = []
    for left, right in itertools.combinations(sorted(single), 2):
        basket = Basket({left: 1, right: 1})
        check = check_basket(tables, basket, capacity_only, capacity, locked)
        if check["feasible"]:
            continue
        shared = []
        left_units = basket_units(tables, Basket({left: 1}))
        right_units = basket_units(tables, Basket({right: 1}))
        for key in left_units:
            if key not in right_units:
                continue
            total = capacity.units.get(key, 0)
            promised = capacity.promised.get(key, 0)
            locked_units = capacity.locked.get(key, 0)
            shared.append(
                {
                    "provider": key[0],
                    "period": key[1],
                    "cap_units": total,
                    "promised_units": promised,
                    "locked_units": locked_units,
                    "available_units": total - promised - locked_units,
                    "combined_units": left_units[key] + right_units[key],
                    "shortfall": left_units[key]
                    + right_units[key]
                    - (total - promised - locked_units),
                }
            )
        affected_promises = [
            {
                "promise": row["promise"],
                "household": row["household"],
                "ward": row["ward"],
                "committed_units": row["committed_units"],
            }
            for row in tables.promises
            if any(
                (row["provider"], row["period"]) == (item["provider"], item["period"])
                for item in shared
                if item["shortfall"] > 0
            )
        ]
        conflicts.append(
            {
                "left": left,
                "right": right,
                "left_ward": index.offers[left]["ward"],
                "right_ward": index.offers[right]["ward"],
                "blocking": [
                    code
                    for code in check["blocking"]
                    if code in ("provider_capacity", "ties", "minimum_families")
                ],
                "shared_capacity": shared,
                "at_risk_promises": affected_promises,
                "message": "两个订单单独可行，合并会挤占已承诺的班次能力；不得一并推荐。",
            }
        )
    return conflicts


def reserve_plan(
    tables: Tables, assumptions: Assumptions
) -> dict[str, Any]:
    """缺乏可比证据的区位只能预留补充访问预算，不能算已估收益。"""
    evidence = evidence_by_offer(tables, assumptions.as_of)
    gaps = []
    for row in tables.offers:
        info = evidence[row["offer"]]
        if not info["has_any_comparable_evidence"]:
            gaps.append(
                {
                    "offer": row["offer"],
                    "ward": row["ward"],
                    "option": row["option"],
                    "reason": "本区无达到门槛的可比证据",
                }
            )
    choice_sets = ward_choice_sets(tables, assumptions.as_of)
    unshown = []
    for row in tables.offers:
        if row["option"] not in choice_sets[row["ward"]].options:
            unshown.append(
                {
                    "offer": row["offer"],
                    "ward": row["ward"],
                    "option": row["option"],
                    "reason": "本区问卷未展示此项",
                }
            )
    unique_gaps = {gap["offer"]: gap for gap in gaps + unshown}
    # 捆绑组里只要有一项缺乏可比证据，整个组都不能正式承购；
    # 同组其它本可承购的项也会被连锁锁住。
    blocked_ties = []
    for row in tables.offers:
        if not row["ties"]:
            continue
        group = [other for other in tables.offers if other["ties"] == row["ties"]]
        if any(other["offer"] in unique_gaps for other in group):
            blocked_ties.append(
                {
                    "tie": row["ties"],
                    "offers": sorted(other["offer"] for other in group),
                    "blocked_by": sorted(
                        other["offer"] for other in group if other["offer"] in unique_gaps
                    ),
                    "collateral": sorted(
                        other["offer"] for other in group if other["offer"] not in unique_gaps
                    ),
                    "note": "捆绑组要求同包数；组内一项缺少可比证据，整组本轮都不能正式承购，"
                    "需先补充访问或另议拆组。",
                }
            )
    blocked_ties = {item["tie"]: item for item in blocked_ties}
    reserve = assumptions.reserve_per_gap_cents * len(unique_gaps)
    return {
        "gaps": sorted(unique_gaps.values(), key=lambda gap: gap["offer"]),
        "gap_count": len(unique_gaps),
        "reserve_cents": reserve,
        "purchasable_budget_cents": assumptions.budget_cents - reserve,
        "blocked_tie_groups": list(blocked_ties.values()),
        "note": "预留只用于补充访问；在补齐可比证据前，相关选项不得列为已估收益的正式承购。",
    }


def benefit_interval(
    tables: Tables,
    result: FitResult,
    basket: Basket,
    assumptions: Assumptions,
    capacity: CapacityState,
    locked: Iterable[dict[str, Any]] = (),
    *,
    draws: int = 300,
    seed: int = 23,
    level: float = 0.8,
) -> dict[str, Any]:
    """同一组合在不同参数抽样下的增益区间，用来表达偏好不确定性。"""
    from .model import draw_parameters

    samples = draw_parameters(result, draws, seed=seed)
    if samples is None:
        return {"available": False, "reason": "没有可用的参数协方差"}
    values = []
    served_values = []
    for beta in samples:
        evaluation = evaluate_basket(
            tables, result, basket, assumptions, capacity, locked, beta=beta
        )
        values.append(evaluation["totals"]["benefit"])
        served_values.append(evaluation["totals"]["served"])
    values.sort()
    served_values.sort()
    alpha = (1 - level) / 2

    def quantile(series, probability):
        position = probability * (len(series) - 1)
        lower = int(position)
        upper = min(lower + 1, len(series) - 1)
        fraction = position - lower
        return series[lower] * (1 - fraction) + series[upper] * fraction

    return {
        "available": True,
        "draws": draws,
        "level": level,
        "benefit": {
            "median": quantile(values, 0.5),
            "low": quantile(values, alpha),
            "high": quantile(values, 1 - alpha),
        },
        "served": {
            "median": quantile(served_values, 0.5),
            "low": quantile(served_values, alpha),
            "high": quantile(served_values, 1 - alpha),
        },
    }


def rank_baskets(
    tables: Tables,
    result: FitResult,
    assumptions: Assumptions,
    capacity: CapacityState,
    locked: Iterable[dict[str, Any]] = (),
    *,
    require_evidence: bool = True,
    limit: int = 12,
    interval_draws: int = 0,
) -> dict[str, Any]:
    baskets = enumerate_baskets(
        tables, assumptions, capacity, locked, require_evidence=require_evidence
    )
    evaluated = []
    for basket in baskets:
        evaluation = evaluate_basket(
            tables, result, basket, assumptions, capacity, locked
        )
        if interval_draws:
            evaluation["interval"] = benefit_interval(
                tables,
                result,
                basket,
                assumptions,
                capacity,
                locked,
                draws=interval_draws,
            )
        evaluated.append(evaluation)
    evaluated.sort(
        key=lambda item: (
            -item["totals"]["benefit"],
            item["totals"]["cost_cents"],
            -item["totals"]["served"],
        )
    )
    # 标出“多买不多得”的组合：同样的估计增益与已服务户数，却更贵。
    kept: list[dict[str, Any]] = []
    for item in evaluated:
        dominated = any(
            abs(previous["totals"]["benefit"] - item["totals"]["benefit"]) <= 1e-6
            and abs(previous["totals"]["served"] - item["totals"]["served"]) <= 1e-6
            and previous["totals"]["cost_cents"] <= item["totals"]["cost_cents"]
            for previous in kept
        )
        item["dominated"] = dominated
        item["dominated_note"] = (
            "与更便宜的组合相比，估计增益与服务户数相同，多出的名额未被需求吸收；不应多买。"
            if dominated
            else None
        )
        if not dominated:
            kept.append(item)
    return {
        "assumptions": assumptions.as_dict(),
        "candidate_count": len(baskets),
        "require_evidence": require_evidence,
        "top": kept[:limit],
        "best": kept[0] if kept else None,
        "all_evaluated": evaluated,
        "all_cost_benefit": [
            {
                "basket": item["basket"],
                "benefit": item["totals"]["benefit"],
                "served": item["totals"]["served"],
                "cost_cents": item["totals"]["cost_cents"],
            }
            for item in evaluated
        ],
    }


def sensitivity(
    tables: Tables,
    primary: FitResult,
    alternatives: Mapping[str, FitResult],
    assumptions: Assumptions,
    capacity: CapacityState,
    baskets: Sequence[Basket],
    locked: Iterable[dict[str, Any]] = (),
) -> dict[str, Any]:
    """同一组候选在若干偏好口径下的排序是否改变。

    口径 A 是保守对照（合并系数、区分人群），C 是选项固定效应模型。
    若两家在增益上很接近，排序本身就不稳，报告并列写出而不是只报一个名次。
    """
    rows: dict[str, list[dict[str, Any]]] = {}
    labels = {}
    for key, result in [("primary", primary), *alternatives.items()]:
        label = ("主口径 " + key) if key == "primary" else key
        labels[key] = label
        scored = []
        for basket in baskets:
            evaluation = evaluate_basket(
                tables, result, basket, assumptions, capacity, locked
            )
            scored.append(
                {
                    "basket": basket.as_dict(),
                    "basket_text": str(basket),
                    "benefit": evaluation["totals"]["benefit"],
                    "served": evaluation["totals"]["served"],
                    "cost_cents": evaluation["totals"]["cost_cents"],
                }
            )
        scored.sort(key=lambda item: (-item["benefit"], item["cost_cents"]))
        rows[key] = scored
    ranking = {}
    order_sets = {}
    for key, scored in rows.items():
        order_sets[key] = tuple(
            tuple(item["basket"].items()) for item in scored
        )
        ranking[labels[key]] = [
            {
                "rank": index + 1,
                "basket": item["basket"],
                "basket_text": item["basket_text"],
                "benefit": item["benefit"],
            }
            for index, item in enumerate(scored)
        ]
    distinct = {order for order in order_sets.values()}
    top_choices = {
        tuple(scored[0]["basket"].items()) for scored in rows.values() if scored
    }
    stable = len(top_choices) <= 1
    leader_differences = {}
    if not stable:
        for key, scored in rows.items():
            if not scored:
                continue
            runner_up = scored[1]["benefit"] if len(scored) > 1 else None
            leader_differences[labels[key]] = {
                "leader": scored[0]["basket_text"],
                "leader_benefit": scored[0]["benefit"],
                "runner_up": scored[1]["basket_text"] if len(scored) > 1 else None,
                "runner_up_benefit": runner_up,
                "margin": scored[0]["benefit"] - runner_up if runner_up is not None else None,
            }
    return {
        "models": list(rows),
        "ranking": ranking,
        "stable": stable,
        "orders": len(distinct),
        "leader_differences": leader_differences,
        "note": (
            "排序在同组候选上一致。"
            if stable
            else "排序在不同偏好口径下不一致；应把这处差异与领先幅度写进采购会记录，"
            "而不是只报一个名次。"
        ),
    }
