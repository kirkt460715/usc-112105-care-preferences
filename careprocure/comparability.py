"""展示方案的可比性、搭桥题与证据门槛。

同名选项在不同展示版本里属性可能改变；只有属性一致，或同一批家庭同时
看到两个版本里的共同选项（搭桥），才能把两组选择放在同一个偏好尺度上。
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

from .states import ResolvedAnswer, resolve_answers
from .tables import Tables


@dataclass(frozen=True)
class OptionSignature:
    option: str
    signature: tuple[Any, ...]


def option_signatures(tables: Tables) -> dict[str, tuple[Any, ...]]:
    """属性签名；价格与陪同条件变化都会改变签名。"""
    return {
        row["option"]: (
            row["duration_hours"],
            row["time_kind"],
            row["copay_cents"],
            row["mode"],
            row["accompanying"],
        )
        for row in tables.options
    }


def display_comparability(tables: Tables, as_of: str | None = None) -> dict[str, Any]:
    """按展示方案描述每个题组的内容、暴露与可比性依据。

    ``bridge_options`` 是该方案与其它方案共同发出的新服务；``statusquo``
    在各方案里都存在，属基线，不计为搭桥。
    """
    answers = resolve_answers(tables, as_of)
    by_form: dict[str, list[ResolvedAnswer]] = defaultdict(list)
    for answer in answers:
        by_form[answer.form].append(answer)
    form_options: dict[str, set[str]] = {
        form: {option for answer in rows for option in answer.available}
        for form, rows in by_form.items()
    }
    form_plan: dict[str, str] = {}
    for answer in answers:
        form_plan.setdefault(answer.form, answer.form.split("-")[0])
    options_sold = {row["option"] for row in tables.offers}
    forms: dict[str, Any] = {}
    for form in sorted(by_form):
        rows = by_form[form]
        delivered = [r for r in rows if r.delivery == "shown"]
        household_ids = {r.household for r in rows}
        options = sorted(form_options[form])
        alternatives = [o for o in options if o != "statusquo"]
        bridges = {
            other: sorted((form_options[form] & form_options[other]) - {"statusquo"})
            for other in sorted(by_form)
            if other != form and len(form_options[form] & form_options[other]) > 1
        }
        forms[form] = {
            "plan": form_plan[form],
            "options": options,
            "alternatives": alternatives,
            "households_assigned": len(household_ids),
            "households_delivered": len({r.household for r in delivered}),
            "displays_assigned": len(rows),
            "displays_delivered": len(delivered),
            "delivered_choice_tasks": sum(1 for r in delivered if r.is_choice),
            "delivered_skips": sum(1 for r in delivered if r.state == "skipped"),
            "not_delivered": sum(1 for r in rows if r.delivery != "shown"),
            "chosen_counts": dict(
                sorted(Counter(r.choice for r in delivered if r.is_choice).items())
            ),
            "bridge_options": bridges,
            "sells_purchasable_option": sorted(set(alternatives) & options_sold),
        }
    plans: dict[str, Any] = {}
    for form, info in forms.items():
        plans.setdefault(info["plan"], {"forms": [], "options": set(), "households": set()})
        plans[info["plan"]]["forms"].append(form)
        plans[info["plan"]]["options"].update(info["options"])
    for plan, info in plans.items():
        options = info["options"]
        info["options"] = sorted(options)
        info["alternatives"] = sorted(options - {"statusquo"})
        info["households"] = len(
            {a.household for a in answers if a.form in forms and forms[a.form]["plan"] == plan}
        )
        info["has_night"] = "nightA" in options or "nightB" in options
        info["has_short"] = "short" in options
        info["has_trip"] = "trip" in options
    plan_names = sorted(plans)
    bridges: list[dict[str, Any]] = []
    for i, left in enumerate(plan_names):
        for right in plan_names[i + 1 :]:
            shared = sorted(
                set(plans[left]["options"]) & set(plans[right]["options"]) - {"statusquo"}
            )
            bridges.append(
                {
                    "left": left,
                    "right": right,
                    "shared_options": shared,
                    "same_households": 0,
                    "comparable": bool(shared),
                    "basis": (
                        "共同展示的选项编号，且 options 表中属性一致；跨方案偏好尺度靠这些选项连接"
                        if shared
                        else "无共同新服务；只能借 statusquo 做弱比较"
                    ),
                }
            )
    return {
        "forms": forms,
        "plans": plans,
        "plan_bridges": bridges,
        "option_signatures": {
            option: {
                "duration_hours": sig[0],
                "time_kind": sig[1],
                "copay_cents": sig[2],
                "mode": sig[3],
                "accompanying": sig[4],
                "signature": list(sig),
            }
            for option, sig in option_signatures(tables).items()
        },
    }


@dataclass(frozen=True)
class EvidenceCell:
    ward: str
    stratum: str
    option: str
    households_exposed: int
    delivered_displays: int
    choice_tasks: int
    chosen: int
    statusquo_tasks: int
    evidence_min: int
    comparable: bool
    note: str

    @property
    def share(self) -> float | None:
        if not self.choice_tasks:
            return None
        return self.chosen / self.choice_tasks


def evidence_cells(
    tables: Tables,
    as_of: str | None = None,
    plans: Iterable[str] | None = None,
) -> list[EvidenceCell]:
    """按区、人群与选项统计“可比较”的暴露与选择。

    ``evidence_min`` 来自 policy；同区同人群的已发题组达到门槛才标为可比。
    这里只给出计数证据，不外推。
    """
    answers = resolve_answers(tables, as_of)
    if plans is not None:
        allowed = set(plans)
        answers = [a for a in answers if a.form.split("-")[0] in allowed]
    minimum = {row["ward"]: row["evidence_min"] for row in tables.policy}
    options = sorted({row["option"] for row in tables.options} - {"statusquo"})
    strata = sorted({row["stratum"] for row in tables.sample})
    wards = sorted({row["ward"] for row in tables.sample})
    grouped: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "households": set(),
            "delivered": 0,
            "choice_tasks": 0,
            "chosen": 0,
            "statusquo": 0,
        }
    )
    for answer in answers:
        if answer.delivery != "shown":
            continue
        for option in answer.available:
            if option == "statusquo":
                continue
            cell = grouped[(answer.ward, answer.stratum, option)]
            cell["households"].add(answer.household)
            cell["delivered"] += 1
            if answer.is_choice:
                cell["choice_tasks"] += 1
                if answer.choice == option:
                    cell["chosen"] += 1
                elif answer.choice == "statusquo":
                    cell["statusquo"] += 1
    cells: list[EvidenceCell] = []
    for ward in wards:
        for stratum in strata:
            for option in options:
                data = grouped.get((ward, stratum, option))
                if data is None:
                    continue
                min_needed = minimum.get(ward, 0)
                comparable = data["choice_tasks"] >= min_needed
                if not data["delivered"]:
                    note = "没有任何已发题组"
                elif not data["choice_tasks"]:
                    note = "题组全部跳题"
                elif data["households"] == 0:
                    note = "无家庭暴露"
                elif not comparable:
                    note = f"有效题组 {data['choice_tasks']} 低于门槛 {min_needed}"
                else:
                    note = "达到门槛"
                cells.append(
                    EvidenceCell(
                        ward=ward,
                        stratum=stratum,
                        option=option,
                        households_exposed=len(data["households"]),
                        delivered_displays=data["delivered"],
                        choice_tasks=data["choice_tasks"],
                        chosen=data["chosen"],
                        statusquo_tasks=data["statusquo"],
                        evidence_min=min_needed,
                        comparable=comparable,
                        note=note,
                    )
                )
    return cells


def coverage_gaps(tables: Tables, as_of: str | None = None) -> list[dict[str, Any]]:
    """列出“本区问卷里根本没有出现”的可采购服务。"""
    answers = resolve_answers(tables, as_of)
    shown_by_ward: dict[str, set[str]] = defaultdict(set)
    for answer in answers:
        if answer.delivery == "shown":
            shown_by_ward[answer.ward].update(answer.available)
    gaps: list[dict[str, Any]] = []
    for offer in tables.offers:
        ward = offer["ward"]
        option = offer["option"]
        if option not in shown_by_ward.get(ward, set()):
            gaps.append(
                {
                    "ward": ward,
                    "option": option,
                    "offer": offer["offer"],
                    "reason": "该区展示方案未包含此项，问卷无法显示偏好",
                }
            )
    for ward in sorted({row["ward"] for row in tables.sample}):
        for option in ("nightA", "nightB", "short", "trip"):
            if option not in shown_by_ward.get(ward, set()):
                gaps.append(
                    {
                        "ward": ward,
                        "option": option,
                        "offer": None,
                        "reason": "该区问卷未展示此项",
                    }
                )
    seen: set[tuple[str, str]] = set()
    unique: list[dict[str, Any]] = []
    for gap in gaps:
        key = (gap["ward"], gap["option"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(gap)
    unique.sort(key=lambda g: (g["ward"], g["option"]))
    return unique


def evidence_summary(tables: Tables, as_of: str | None = None) -> dict[str, Any]:
    cells = evidence_cells(tables, as_of)
    comparable = [c for c in cells if c.comparable]
    gaps = coverage_gaps(tables, as_of)
    by_ward: dict[str, Any] = {}
    for ward in sorted({row["ward"] for row in tables.sample}):
        ward_cells = [c for c in cells if c.ward == ward]
        by_ward[ward] = {
            "comparable_cells": sum(1 for c in ward_cells if c.comparable),
            "cell_count": len(ward_cells),
            "options_with_comparable_evidence": sorted(
                {c.option for c in ward_cells if c.comparable}
            ),
            "options_without_any_evidence": sorted(
                {c.option for c in ward_cells if c.delivered_displays == 0}
            ),
            "unpurchasable_gaps": [g for g in gaps if g["ward"] == ward],
        }
    return {
        "cells": [
            {
                "ward": c.ward,
                "stratum": c.stratum,
                "option": c.option,
                "households_exposed": c.households_exposed,
                "delivered_displays": c.delivered_displays,
                "choice_tasks": c.choice_tasks,
                "chosen": c.chosen,
                "share": None if c.share is None else round(c.share, 4),
                "evidence_min": c.evidence_min,
                "comparable": c.comparable,
                "note": c.note,
            }
            for c in cells
        ],
        "by_ward": by_ward,
        "coverage_gaps": gaps,
        "comparable_cell_count": len(comparable),
    }
