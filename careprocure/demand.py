"""分群需求、替代关系、外推警示与不确定性区间。

一切份额都基于该区实际发出过的选项集。某区从未展示的服务即使模型能算出
一个数字，也只标为“外推”，不当作在地需求。
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

from .model import (
    FitResult,
    feature_vector,
    predict,
)
from .states import resolve_answers
from .tables import Tables

STATUSQUO = "statusquo"


@dataclass(frozen=True)
class WardChoiceSet:
    ward: str
    options: tuple[str, ...]
    delivered_displays: int
    households: int
    extrapolated: tuple[str, ...]


def ward_choice_sets(tables: Tables, as_of: str | None = None) -> dict[str, WardChoiceSet]:
    """每区实际展示过的选项集；只有 statusquo 的区为空。"""
    answers = resolve_answers(tables, as_of)
    shown: dict[str, set[str]] = defaultdict(set)
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    households: dict[str, set[str]] = defaultdict(set)
    for answer in answers:
        if answer.delivery != "shown":
            continue
        counts[answer.ward]["displays"] += 1
        households[answer.ward].add(answer.household)
        for option in answer.available:
            shown[answer.ward].add(option)
            counts[answer.ward][option] += 1
    result: dict[str, WardChoiceSet] = {}
    for ward in sorted({row["ward"] for row in tables.sample}):
        options = tuple(sorted(shown.get(ward, {"statusquo"})))
        result[ward] = WardChoiceSet(
            ward=ward,
            options=options,
            delivered_displays=counts[ward]["displays"],
            households=len(households.get(ward, ())),
            extrapolated=(),
        )
    return result


def union_menu_shares(
    tables: Tables,
    result: FitResult,
    ward: str,
    options: Sequence[str] | None = None,
    beta: Sequence[float] | None = None,
) -> dict[str, Any]:
    """把该区所有展示过的选项并成一个菜单的偏乐观口径。

    没有任何家庭真的看到过这个并集，因此这份结果只用于对照：报告主值用
    :func:`segment_shares_by_form`。不要用它宣称在地需求。
    """
    choice_set = list(options) if options is not None else list(
        ward_choice_sets(tables)[ward].options
    )
    households = [row for row in tables.sample if row["ward"] == ward]
    by_stratum: dict[str, Any] = {}
    strata = sorted({row["stratum"] for row in households})
    for stratum in strata:
        rows = [row for row in households if row["stratum"] == stratum]
        total_weight = sum(1.0 / row["sampling_probability"] for row in rows)
        shares = {option: 0.0 for option in choice_set}
        for row in rows:
            probabilities = predict(
                tables, result, choice_set, stratum, beta=beta
            )
            weight = 1.0 / row["sampling_probability"]
            for option, value in probabilities.items():
                shares[option] += weight * value
        by_stratum[stratum] = {
            "population_estimate": total_weight,
            "shares": {option: value / total_weight for option, value in shares.items()},
            "household_counts": {
                option: value for option, value in shares.items()
            },
        }
    total_weight = sum(1.0 / row["sampling_probability"] for row in households)
    overall = {option: 0.0 for option in choice_set}
    for stratum in strata:
        weight = by_stratum[stratum]["population_estimate"]
        for option, value in by_stratum[stratum]["shares"].items():
            overall[option] += weight * value
    overall = {option: value / total_weight for option, value in overall.items()}
    return {
        "ward": ward,
        "options": choice_set,
        "population_estimate": total_weight,
        "overall_shares": overall,
        "overall_counts": {option: value * total_weight for option, value in overall.items()},
        "by_stratum": by_stratum,
    }


def share_interval(
    tables: Tables,
    result: FitResult,
    ward: str,
    options: Sequence[str] | None = None,
    draws: int = 400,
    seed: int = 11,
    level: float = 0.8,
) -> dict[str, Any]:
    """用聚类稳健协方差抽样，给出份额的区间与方案排序稳定性。"""
    from .model import draw_parameters

    samples = draw_parameters(result, draws, seed=seed)
    if samples is None:
        return {"available": False, "reason": "没有可用的参数协方差"}
    values: dict[str, list[float]] = defaultdict(list)
    for beta in samples:
        shares = segment_shares_by_form(tables, result, ward, beta=beta)["overall_shares"]
        for option, value in shares.items():
            values[option].append(value)
    alpha = (1 - level) / 2
    summary = {}
    for option, series in values.items():
        series.sort()
        summary[option] = {
            "median": _quantile(series, 0.5),
            "low": _quantile(series, alpha),
            "high": _quantile(series, 1 - alpha),
            "mean": sum(series) / len(series),
        }
    return {"available": True, "draws": draws, "level": level, "options": summary}


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        return float("nan")
    position = probability * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def _form_base_shares(
    tables: Tables,
    result: FitResult,
    ward: str,
    beta: Sequence[float] | None = None,
) -> dict[str, float]:
    """每户在它实际看到的版本菜单内的基准份额（statusquo 含在内）。"""
    menus = prepare_form_menus(tables, result, ward, beta=beta)
    options = sorted({option for menu in menus for option in menu["options"]})
    counts = {option: 0.0 for option in options}
    population = 0.0
    for menu in menus:
        for info in menu["strata"].values():
            population += info["weight"]
            for option in options:
                probability = (
                    info["statusquo_probability"]
                    if option == STATUSQUO
                    else info["probabilities"].get(option, 0.0)
                )
                counts[option] += info["weight"] * probability
    return {option: (value / population if population else 0.0) for option, value in counts.items()}


def diversion(
    tables: Tables,
    result: FitResult,
    ward: str,
    removed: str,
    options: Sequence[str] | None = None,
    beta: Sequence[float] | None = None,
) -> dict[str, Any]:
    """移除某项后，原份额流向哪里；是“替代关系”的核心读数。

    在每户实际看到的版本菜单内移除该项，再按抽样权汇总；只在该区展示过的
    选项之间讨论替代。
    """
    choice_set = list(options) if options is not None else list(
        ward_choice_sets(tables)[ward].options
    )
    if removed not in choice_set:
        raise ValueError(f"{ward} 的选项集里没有 {removed}")
    menus = prepare_form_menus(tables, result, ward, beta=beta)
    base = _form_base_shares(tables, result, ward, beta=beta)
    reduced_counts = {
        option: 0.0 for option in choice_set if option != removed
    }
    population = 0.0
    for menu in menus:
        reduced = [option for option in menu["options"] if option != removed]
        for info in menu["strata"].values():
            population += info["weight"]
            utilities = {option: info["utilities"][option] for option in reduced}
            top = max(utilities.values())
            exponentials = {
                option: math.exp(value - top) for option, value in utilities.items()
            }
            denominator = sum(exponentials.values())
            for option in reduced:
                reduced_counts[option] += (
                    info["weight"] * exponentials[option] / denominator
                )
    reduced_shares = {
        option: (value / population if population else 0.0)
        for option, value in reduced_counts.items()
    }
    removed_share = base.get(removed, 0.0)
    flows = {}
    for option, value in reduced_shares.items():
        change = value - base.get(option, 0.0)
        flows[option] = {
            "share_change": change,
            "diversion_ratio": change / removed_share if removed_share else None,
        }
    return {
        "ward": ward,
        "removed": removed,
        "removed_share": removed_share,
        "flows": flows,
        "self_statusquo": reduced_shares.get(STATUSQUO, 0.0)
        - base.get(STATUSQUO, 0.0),
    }


def substitution_matrix(
    tables: Tables,
    result: FitResult,
    ward: str,
    options: Sequence[str] | None = None,
) -> dict[str, Any]:
    choice_set = list(options) if options is not None else list(
        ward_choice_sets(tables)[ward].options
    )
    matrix = {}
    for removed in choice_set:
        if removed == STATUSQUO:
            continue
        matrix[removed] = diversion(tables, result, ward, removed, choice_set)
    return {"ward": ward, "options": choice_set, "matrix": matrix}


def demand_by_segment(
    tables: Tables,
    result: FitResult,
    draws: int = 400,
    seed: int = 11,
) -> dict[str, Any]:
    """全区×人群的需求与区间；对未展示选项明确标出外推。"""
    choice_sets = ward_choice_sets(tables)
    output: dict[str, Any] = {}
    for ward, choice_set in choice_sets.items():
        shares = segment_shares_by_form(tables, result, ward)
        interval = share_interval(tables, result, ward, draws=draws, seed=seed)
        sells = {
            row["option"] for row in tables.offers if row["ward"] == ward
        }
        missing = sorted(sells - set(choice_set.options))
        output[ward] = {
            "options": shares["options"],
            "menus": shares["menus"],
            "delivered_displays": choice_set.delivered_displays,
            "households": choice_set.households,
            "population_estimate": shares["population_estimate"],
            "overall_shares": shares["overall_shares"],
            "overall_counts": shares["overall_counts"],
            "by_stratum": shares["by_stratum"],
            "intervals": interval,
            "not_shown_purchasable": missing,
            "extrapolation_warning": [
                {
                    "option": option,
                    "text": "本区问卷未展示该服务；任何模型份额都只是外推，不能当作在地需求，需先补充访问。",
                }
                for option in missing
            ],
        }
    return output


@dataclass
class MenuDemand:
    """按区、人群预先算好的效用与需求，供采购组合快速评估。"""

    ward: str
    options: tuple[str, ...]
    strata: dict[str, dict[str, Any]]
    population_estimate: float

    def demand(self, option: str) -> float:
        return sum(
            info["weight"] * info["probabilities"].get(option, 0.0)
            for info in self.strata.values()
        )

    def demand_counts(self) -> dict[str, float]:
        return {option: self.demand(option) for option in self.options}

    def availability_gain(self, fractions: dict[str, float]) -> float:
        """可用概率加权 logsum 与维持现状之差（效用单位，按抽样权汇总）。"""
        total = 0.0
        for info in self.strata.values():
            inside = math.exp(info["u_statusquo"])
            for option, value in info["exp_utility"].items():
                fraction = fractions.get(option, 0.0)
                if fraction > 0:
                    inside += fraction * value
            total += info["weight"] * (math.log(inside) - info["u_statusquo"])
        return total


def prepare_form_menus(
    tables: Tables,
    result: FitResult,
    ward: str,
    as_of: str | None = None,
    beta: Sequence[float] | None = None,
) -> list[dict[str, Any]]:
    """把该区每个实际展示版本当作一个菜单，并按抽样权汇总家庭数。

    没有任何家庭看到过全区的选项并集；用版本菜单做反事实，比把并集当成
    一个菜单更贴近问卷实际。
    """
    answers = resolve_answers(tables, as_of)
    forms: dict[str, set[str]] = defaultdict(set)
    households_by_form: dict[str, set[str]] = defaultdict(set)
    forms_by_household: dict[str, set[str]] = defaultdict(set)
    for answer in answers:
        if answer.ward == ward and answer.delivery == "shown":
            forms[answer.form].update(answer.available)
            households_by_form[answer.form].add(answer.household)
            forms_by_household[answer.household].add(answer.form)
    vector = result.beta if beta is None else beta
    population = {row["household"]: row for row in tables.sample if row["ward"] == ward}
    menus = []
    for form in sorted(forms):
        options = tuple(sorted(forms[form]))
        strata: dict[str, dict[str, Any]] = {}
        for stratum in sorted({row["stratum"] for row in tables.sample}):
            rows = [
                row
                for row in population.values()
                if row["stratum"] == stratum and row["household"] in households_by_form[form]
            ]
            if not rows:
                continue
            # 同一户看过多个版本；每户的总权 1/p 平均分到它看过的版本，避免重复计数。
            weight = sum(
                1.0
                / row["sampling_probability"]
                / len(forms_by_household[row["household"]])
                for row in rows
            )
            utilities = {
                option: sum(
                    x * b
                    for x, b in zip(
                        feature_vector(tables, result.spec, option, stratum), vector
                    )
                )
                for option in options
            }
            u_statusquo = utilities[STATUSQUO]
            exponentials = {
                option: math.exp(utilities[option] - u_statusquo)
                for option in options
                if option != STATUSQUO
            }
            denominator = 1.0 + sum(exponentials.values())
            strata[stratum] = {
                "weight": weight,
                "u_statusquo": u_statusquo,
                "utilities": utilities,
                "exp_utility": exponentials,
                "probabilities": {
                    option: value / denominator
                    for option, value in exponentials.items()
                },
                "statusquo_probability": 1.0 / denominator,
            }
        menus.append(
            {
                "form": form,
                "options": options,
                "strata": strata,
                "population_estimate": sum(
                    info["weight"] for info in strata.values()
                ),
            }
        )
    return menus


def scenario_gain_by_form(
    tables: Tables,
    result: FitResult,
    ward: str,
    provided: Sequence[str],
    beta: Sequence[float] | None = None,
    slots: dict[str, int] | None = None,
    as_of: str | None = None,
) -> dict[str, Any]:
    """在每户实际看到的版本菜单内评估，并按版本份额分配名额。

    名额先在各版本间按需求比例分配，再在各版本内按 ``名额/需求`` 配给；
    因此被削减的家庭会转向该版本菜单内仍可选的其它服务或维持现状。
    """
    menus = prepare_form_menus(tables, result, ward, as_of, beta=beta)
    provided = list(provided)
    per_form = []
    for menu in menus:
        available = [option for option in provided if option in menu["options"]]
        form_demand = {
            option: sum(
                info["weight"] * info["probabilities"].get(option, 0.0)
                for info in menu["strata"].values()
            )
            for option in available
        }
        per_form.append(
            {
                "form": menu["form"],
                "options": list(menu["options"]),
                "provided": available,
                "demand": form_demand,
                "menu": menu,
            }
        )
    total_demand = {
        option: sum(row["demand"].get(option, 0.0) for row in per_form)
        for option in provided
    }
    # 名额按需求比例分到各版本
    allocation: dict[str, dict[str, float]] = {}
    for option in provided:
        capacity = None if slots is None else slots.get(option)
        total = total_demand.get(option, 0.0)
        allocation[option] = {}
        for row in per_form:
            demand = row["demand"].get(option, 0.0)
            if capacity is None:
                allocation[option][row["form"]] = demand
            elif total <= 0:
                allocation[option][row["form"]] = demand
            else:
                allocation[option][row["form"]] = min(
                    demand, capacity * demand / total
                )
    total_benefit = 0.0
    served = {option: 0.0 for option in provided}
    statusquo_households = 0.0
    population = 0.0
    for row in per_form:
        menu = row["menu"]
        fractions = {}
        for option in row["provided"]:
            demand = row["demand"][option]
            allocated = allocation[option][row["form"]]
            fractions[option] = 1.0 if demand <= 0 else min(1.0, allocated / demand)
            served[option] += demand * fractions[option]
        inside_total = 0.0
        for info in menu["strata"].values():
            inside = math.exp(info["u_statusquo"])
            for option in row["provided"]:
                inside += fractions[option] * info["exp_utility"][option]
            inside_total += info["weight"] * (
                math.log(inside) - info["u_statusquo"]
            )
            population += info["weight"]
            statusquo_households += info["weight"] * info["statusquo_probability"]
        total_benefit += inside_total
    demand = {option: total_demand.get(option, 0.0) for option in provided}
    unmet = {option: demand[option] - served[option] for option in provided}
    return {
        "ward": ward,
        "mode": "form_menus",
        "provided": provided,
        "demand": demand,
        "served": served,
        "unmet": unmet,
        "demand_total": sum(demand.values()),
        "served_total": sum(served.values()),
        "unmet_total": sum(unmet.values()),
        "benefit": total_benefit,
        "benefit_per_served": total_benefit / sum(served.values())
        if sum(served.values())
        else 0.0,
        "population_estimate": population,
        "statusquo_households": statusquo_households,
        "forms": [
            {
                "form": row["form"],
                "options": row["options"],
                "provided": row["provided"],
                "demand": row["demand"],
                "allocated": {
                    option: allocation[option][row["form"]]
                    for option in row["provided"]
                },
            }
            for row in per_form
        ],
    }


def prepare_menu(
    tables: Tables,
    result: FitResult,
    ward: str,
    options: Sequence[str] | None = None,
    beta: Sequence[float] | None = None,
) -> MenuDemand:
    choice_set = tuple(
        options if options is not None else ward_choice_sets(tables)[ward].options
    )
    vector = result.beta if beta is None else beta
    strata: dict[str, dict[str, Any]] = {}
    for stratum in sorted({row["stratum"] for row in tables.sample}):
        rows = [
            row
            for row in tables.sample
            if row["ward"] == ward and row["stratum"] == stratum
        ]
        if not rows:
            continue
        weight = sum(1.0 / row["sampling_probability"] for row in rows)
        utilities = {
            option: sum(
                x * b
                for x, b in zip(
                    feature_vector(tables, result.spec, option, stratum), vector
                )
            )
            for option in choice_set
        }
        u_statusquo = utilities[STATUSQUO]
        exponentials = {
            option: math.exp(utilities[option] - u_statusquo)
            for option in choice_set
            if option != STATUSQUO
        }
        denominator = 1.0 + sum(exponentials.values())
        strata[stratum] = {
            "weight": weight,
            "u_statusquo": u_statusquo,
            "utilities": utilities,
            "exp_utility": exponentials,
            "probabilities": {
                option: value / denominator for option, value in exponentials.items()
            },
            "statusquo_probability": 1.0 / denominator,
        }
    return MenuDemand(
        ward=ward,
        options=choice_set,
        strata=strata,
        population_estimate=sum(info["weight"] for info in strata.values()),
    )


def segment_shares_by_form(
    tables: Tables,
    result: FitResult,
    ward: str,
    beta: Sequence[float] | None = None,
    as_of: str | None = None,
) -> dict[str, Any]:
    """按每户实际看到的版本菜单求期望份额。

    每户总权 ``1/p`` 平均分到它看过的版本，因此份额之和（含 statusquo）为 1，
    含义是“在已发出的展示条件下，一个家庭选各服务的期望概率”。
    """
    menus = prepare_form_menus(tables, result, ward, as_of, beta=beta)
    options = sorted({option for menu in menus for option in menu["options"]})
    overall = {option: 0.0 for option in options}
    by_stratum: dict[str, dict[str, Any]] = {}
    population = 0.0
    for menu in menus:
        for stratum, info in menu["strata"].items():
            entry = by_stratum.setdefault(
                stratum, {"population_estimate": 0.0, "counts": {o: 0.0 for o in options}}
            )
            entry["population_estimate"] += info["weight"]
            population += info["weight"]
            for option in options:
                probability = info["probabilities"].get(option, 0.0)
                if option == STATUSQUO:
                    probability = info["statusquo_probability"]
                entry["counts"][option] += info["weight"] * probability
                overall[option] += info["weight"] * probability
    shares = {
        option: (value / population if population else 0.0)
        for option, value in overall.items()
    }
    for entry in by_stratum.values():
        total = entry["population_estimate"]
        entry["shares"] = {
            option: (value / total if total else 0.0)
            for option, value in entry["counts"].items()
        }
        entry["counts"] = {
            option: value for option, value in entry["counts"].items()
        }
    return {
        "ward": ward,
        "options": options,
        "population_estimate": population,
        "overall_shares": shares,
        "overall_counts": overall,
        "by_stratum": by_stratum,
        "menus": [
            {
                "form": menu["form"],
                "options": list(menu["options"]),
                "population_estimate": menu["population_estimate"],
            }
            for menu in menus
        ],
    }


def scenario_gain(
    tables: Tables,
    result: FitResult,
    ward: str,
    provided: Sequence[str],
    beta: Sequence[float] | None = None,
    slots: dict[str, int] | None = None,
) -> dict[str, Any]:
    """相对于维持现状的预期增益，并按名额做比例配给。

    配给口径：若需求超过名额，按 ``名额 / 需求`` 的比例削减该项的可得概率，
    再以可用概率加权 logsum 计算增益——被削减的家庭会转向仍可选的其它服务
    或维持现状，而不是简单消失。
    """
    menu = prepare_menu(tables, result, ward, beta=beta)
    provided = [option for option in provided if option in menu.options]
    full_demand = menu.demand_counts()
    fractions: dict[str, float] = {}
    for option in provided:
        demand = full_demand.get(option, 0.0)
        capacity = None if slots is None else slots.get(option)
        if capacity is None:
            fractions[option] = 1.0
        elif demand <= 0:
            fractions[option] = 1.0
        else:
            fractions[option] = min(1.0, capacity / demand)
    gain = menu.availability_gain(fractions)
    demand = {option: full_demand.get(option, 0.0) for option in provided}
    served = {option: demand[option] * fractions[option] for option in provided}
    unmet = {option: demand[option] - served[option] for option in provided}
    statusquo_households = sum(
        info["weight"] * info["statusquo_probability"]
        for info in menu.strata.values()
    )
    served_probability: dict[str, float] = {}
    for stratum, info in menu.strata.items():
        for row in tables.sample:
            if row["ward"] != ward or row["stratum"] != stratum:
                continue
            weight = 1.0 / row["sampling_probability"] / info["weight"]
            served_probability[row["household"]] = weight * sum(
                info["probabilities"].get(option, 0.0) * fractions[option]
                for option in provided
            )
    return {
        "ward": ward,
        "provided": provided,
        "options_considered": list(menu.options),
        "demand": demand,
        "served": served,
        "unmet": unmet,
        "fractions": fractions,
        "demand_total": sum(demand.values()),
        "served_total": sum(served.values()),
        "unmet_total": sum(unmet.values()),
        "benefit": gain,
        "benefit_per_served": gain / sum(served.values()) if sum(served.values()) else 0.0,
        "population_estimate": menu.population_estimate,
        "statusquo_households": statusquo_households,
        "household_served_probability": served_probability,
    }


def extrapolation_report(tables: Tables) -> list[dict[str, Any]]:
    """把“本区没有展示机会”的服务逐条列出，供报告与采购警示使用。"""
    choice_sets = ward_choice_sets(tables)
    sells: dict[str, set[str]] = defaultdict(set)
    for row in tables.offers:
        sells[row["ward"]].add(row["option"])
    warnings: list[dict[str, Any]] = []
    for ward, options in sorted(sells.items()):
        for option in sorted(options - set(choice_sets[ward].options)):
            warnings.append(
                {
                    "ward": ward,
                    "option": option,
                    "warning": "本区问卷未提供共同展示机会",
                    "consequence": "模型份额只能是外推；不得当作在地需求或已估收益",
                }
            )
    return warnings
