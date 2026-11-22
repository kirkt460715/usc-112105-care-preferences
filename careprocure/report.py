"""生成采购会与分区说明用的报告（数据 + Markdown）。

报告中的每个数字都能追溯到来源表、展示编号、估计口径与锁定记录。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .comparability import display_comparability, evidence_summary
from .demand import (
    demand_by_segment,
    extrapolation_report,
    substitution_matrix,
    ward_choice_sets,
)
from .locks import lock_summary
from .model import (
    SPECS,
    FitResult,
    baseline_metrics,
    cross_validate,
    fit,
    fit_latent_class,
)
from .procurement import (
    Assumptions,
    Basket,
    CapacityState,
    capacity_state,
    conflict_pairs,
    rank_baskets,
    reserve_plan,
    sensitivity,
)
from .states import audit_counts, resolve_answers
from .tables import Tables


def _pct(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "—"
    scaled = value * 100
    if abs(scaled) < 0.05:
        scaled = 0.0
    return f"{scaled:.{digits}f}%"


def _money(cents: int | float | None) -> str:
    if cents is None:
        return "—"
    return f"{cents / 100:,.0f} 元"


def _num(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "—"
    return f"{value:,.{digits}f}"


@dataclass
class ReportAssets:
    tables: Tables
    assumptions: Assumptions
    primary: FitResult
    alternatives: dict[str, FitResult]
    capacity: CapacityState
    ranking: dict[str, Any]
    conflicts: list[dict[str, Any]]
    reserve: dict[str, Any]
    demand: dict[str, Any]
    sensitivity: dict[str, Any]
    model_comparison: list[dict[str, Any]]
    baselines: dict[str, Any]
    latent_class: dict[str, Any]
    ledger_entries: list[dict[str, Any]]


def build_assets(
    tables: Tables,
    assumptions: Assumptions,
    *,
    spec_keys: Sequence[str] = ("A", "B", "C"),
    draws: int = 300,
    ledger_entries: Iterable[dict[str, Any]] = (),
    locked: Iterable[dict[str, Any]] = (),
    limit: int = 10,
) -> ReportAssets:
    fits = {key: fit(tables, SPECS[key]) for key in spec_keys}
    primary = fits[assumptions.spec_key]
    alternatives = {key: value for key, value in fits.items() if key != assumptions.spec_key}
    capacity = capacity_state(tables, assumptions.as_of, locked)
    ranking = rank_baskets(
        tables,
        primary,
        assumptions,
        capacity,
        locked,
        limit=limit,
        interval_draws=draws,
    )
    conflicts = conflict_pairs(tables, assumptions, capacity, locked)
    reserve = reserve_plan(tables, assumptions)
    demand = demand_by_segment(tables, primary, draws=draws)
    candidates = [Basket(item["basket"]) for item in ranking["top"][:6]]
    sens = sensitivity(
        tables, primary, alternatives, assumptions, capacity, candidates, locked
    )
    comparison = []
    baselines = baseline_metrics(tables)
    latent = fit_latent_class(
        tables, classes=2, ridge=1.0, seed=7, em_iterations=40, inner_iterations=20
    )
    for key in spec_keys:
        cv = cross_validate(tables, SPECS[key])
        comparison.append(
            {
                "spec": key,
                "label": SPECS[key].label,
                "description": SPECS[key].description,
                "ridge": SPECS[key].ridge,
                "household": cv["household"],
                "form": cv["form"],
                "plan": cv["plan"],
            }
        )
    return ReportAssets(
        tables=tables,
        assumptions=assumptions,
        primary=primary,
        alternatives=alternatives,
        capacity=capacity,
        ranking=ranking,
        conflicts=conflicts,
        reserve=reserve,
        demand=demand,
        sensitivity=sens,
        model_comparison=comparison,
        baselines=baselines,
        latent_class={
            "classes": latent.classes,
            "class_probability": latent.class_probability,
            "loglik": latent.loglik,
            "seed": latent.seed,
            "coefficients": [
                model.coefficients() for model in latent.models
            ],
        },
        ledger_entries=list(ledger_entries),
    )


def explain_ward(assets: ReportAssets, ward: str) -> dict[str, Any]:
    """面向该区的说明：看到了什么、证据多强、买了什么、为什么。"""
    tables = assets.tables
    answers = resolve_answers(tables, assets.assumptions.as_of)
    ward_answers = [a for a in answers if a.ward == ward]
    delivered = [a for a in ward_answers if a.delivery == "shown"]
    choice_tasks = [a for a in delivered if a.is_choice]
    options = ward_choice_sets(tables, assets.assumptions.as_of)[ward].options
    evidence = evidence_summary(tables, assets.assumptions.as_of)["by_ward"][ward]
    demand = assets.demand[ward]
    purchased = {}
    if assets.ranking["best"]:
        purchased = assets.ranking["best"]["wards"][ward]
    conflicts = [
        conflict
        for conflict in assets.conflicts
        if conflict["left_ward"] == ward or conflict["right_ward"] == ward
    ]
    return {
        "ward": ward,
        "survey": {
            "households_sampled": len({a.household for a in ward_answers}),
            "displays_delivered": len(delivered),
            "choice_tasks": len(choice_tasks),
            "statusquo_choices": sum(1 for a in choice_tasks if a.choice == "statusquo"),
            "skips": sum(1 for a in delivered if a.state == "skipped"),
            "not_delivered": sum(1 for a in ward_answers if a.delivery != "shown"),
            "options_shown": list(options),
        },
        "evidence": evidence,
        "demand": {
            "shares": demand["overall_shares"],
            "counts": demand["overall_counts"],
            "intervals": demand.get("intervals", {}).get("options", {}),
            "not_shown_purchasable": demand["not_shown_purchasable"],
        },
        "proposed_purchase": purchased,
        "conflicts": conflicts,
        "explanation": _ward_narrative(ward, options, evidence, purchased, demand),
    }


def _ward_narrative(
    ward: str,
    options: Sequence[str],
    evidence: dict[str, Any],
    purchased: dict[str, Any],
    demand: dict[str, Any],
) -> str:
    offered = "、".join(options)
    comparable = "、".join(evidence["options_with_comparable_evidence"]) or "无"
    gaps = demand["not_shown_purchasable"]
    parts = [
        f"{ward} 区问卷实际展示的选项为 {offered}；达到证据门槛的选项为 {comparable}。"
    ]
    if gaps:
        parts.append(
            "本区没有展示 " + "、".join(gaps) + "，因此这些服务无法从本区问卷得到可信偏好；"
            "我们不会把模型外推当成在地需求，只会先安排补充访问。"
        )
    if purchased:
        provided = purchased.get("provided", [])
        benefit = purchased.get("benefit", 0.0)
        served = purchased.get("served_total", 0.0)
        parts.append(
            "本轮在该区购买 "
            + ("、".join(provided) if provided else "无")
            + f"，估计可服务约 {served:.1f} 户，相对维持现状的偏好增益约 {benefit:.1f} 效用单位；"
            "区间与替代关系见报告附表。"
        )
    else:
        parts.append("本轮没有在该区安排新购；维持现状仍是默认。")
    return "".join(parts)


def to_payload(assets: ReportAssets, wards: Sequence[str] | None = None) -> dict[str, Any]:
    tables = assets.tables
    wards = list(wards or sorted({row["ward"] for row in tables.sample}))
    audit = audit_counts(resolve_answers(tables, assets.assumptions.as_of))
    comparability = display_comparability(tables, assets.assumptions.as_of)
    evidence = evidence_summary(tables, assets.assumptions.as_of)
    substitutions = {
        ward: substitution_matrix(tables, assets.primary, ward)
        for ward in wards
        if len(ward_choice_sets(tables, assets.assumptions.as_of)[ward].options) > 1
    }
    return {
        "assumptions": assets.assumptions.as_dict(),
        "audit": audit,
        "comparability": {
            "plans": comparability["plans"],
            "plan_bridges": comparability["plan_bridges"],
            "forms": comparability["forms"],
        },
        "evidence": evidence,
        "models": {
            "comparison": assets.model_comparison,
            "baselines": assets.baselines,
            "latent_class": assets.latent_class,
            "primary": {
                "spec": assets.primary.spec.key,
                "label": assets.primary.spec.label,
                "coefficients": assets.primary.coefficients(),
                "standard_errors": dict(
                    zip(assets.primary.names, assets.primary.standard_errors or [])
                ),
            },
        },
        "demand": assets.demand,
        "substitution": substitutions,
        "extrapolation_warnings": extrapolation_report(tables),
        "procurement": {
            "ranking": assets.ranking,
            "conflicts": assets.conflicts,
            "reserve": assets.reserve,
            "capacity": {
                f"{provider}/{period}": {
                    "cap_units": units,
                    "visible_from": assets.capacity.latest_visible.get((provider, period)),
                    "promised_units": assets.capacity.promised.get((provider, period), 0),
                    "locked_units": assets.capacity.locked.get((provider, period), 0),
                }
                for (provider, period), units in sorted(assets.capacity.units.items())
            },
        },
        "sensitivity": assets.sensitivity,
        "locks": [lock_summary(entry) for entry in assets.ledger_entries],
        "ward_explanations": {ward: explain_ward(assets, ward) for ward in wards},
    }


def render_markdown(assets: ReportAssets, wards: Sequence[str] | None = None) -> str:
    payload = to_payload(assets, wards)
    tables = assets.tables
    lines: list[str] = []
    add = lines.append
    add("# 家庭照护支持采购：证据、需求与组合建议")
    add("")
    add(
        f"资料截至 **{assets.assumptions.as_of}**；主偏好口径 **{assets.primary.spec.label}**"
        f"（规格 {assets.primary.spec.key}，岭惩罚 {assets.primary.ridge:g}）。"
        "本报告中的份额与增益都是模型估计，不是问卷票数。"
    )
    add("")
    add("## 一、问卷状态与证据边界")
    add("")
    audit = payload["audit"]
    add("| 记录 | 数量 | 处理 |")
    add("| --- | ---: | --- |")
    add(f"| 发出展示 | {audit['displays']} | 全部登记 |")
    add(f"| 实际展示 | {audit['delivered_displays']} | 进入暴露与选择统计 |")
    add(
        f"| 主动选择新服务 | {audit['chose_alternative']} | 进入选择模型 |"
    )
    add(
        f"| 主动选择维持现状 | {audit['chose_statusquo']} | 进入选择模型，**不是**未选 |"
    )
    add(f"| 跳题 | {audit['skipped']} | 不当作不喜欢，不进似然 |")
    add(f"| 漏发 | {audit['not_delivered']} | 不算“没看到后不喜欢”，不进似然 |")
    add(f"| 未应答 | {audit['missing']} | 不进似然 |")
    add(
        f"| 补正题组 | {audit['corrected_displays']} | 取最晚登记答卷；被替代答卷保留在报告中 |"
    )
    add("")
    for correction in audit["corrections"]:
        superseded = "；".join(
            f"{old['seen_on']} 选 {old['choice']}"
            for old in correction["superseded"]
        )
        add(
            f"> 补正：{correction['display']}（{correction['household']}）采用 "
            f"{correction['kept_seen_on']} 的“{correction['kept_choice']}”，"
            f"覆盖此前 {superseded}。"
        )
    add("")
    add("### 展示方案与搭桥")
    add("")
    add("| 方案 | 展示选项 | 家庭数 | 是否有夜间 | 是否有短托 |")
    add("| --- | --- | ---: | :---: | :---: |")
    for plan, info in sorted(payload["comparability"]["plans"].items()):
        add(
            f"| {plan} | {'、'.join(info['alternatives'])} | {info['households']} |"
            f" {'是' if info['has_night'] else '否'} | {'是' if info['has_short'] else '否'} |"
        )
    add("")
    add("同一内容（选项编号及其属性）在多个方案里出现，就构成连接不同展示版本的搭桥；同名但属性改变过的不算。没有共同新服务的组别只能借 statusquo 弱比较，不能当作同一尺度。")
    add("")
    add("| 组别 A | 组别 B | 共同新服务 | 可比性 |")
    add("| --- | --- | --- | --- |")
    for bridge in payload["comparability"]["plan_bridges"]:
        add(
            f"| {bridge['left']} | {bridge['right']} |"
            f" {'、'.join(bridge['shared_options']) or '无'} |"
            f" {bridge['basis']} |"
        )
    add("")
    add("## 二、模型比较与留出表现")
    add("")
    add(
        "留出检查按家庭分组（同户多个题组不跨折）与按展示题组（整版留出）；"
        "另做“每个方案留一题组”的跨方案检查。对数损失越低越好，均匀基准为 ln(3)≈1.099。"
    )
    add("")
    add("| 规格 | 口径 | 家庭留出 对数损失 | 家庭留出 准确率 | 题组留出 对数损失 | 题组留出 准确率 | 跨方案留出 对数损失 |")
    add("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for row in payload["models"]["comparison"]:
        household = row["household"] or {}
        form = row["form"] or {}
        plan = row["plan"] or {}
        add(
            f"| {row['spec']} | {row['label']} | {_num(household.get('mean_logloss'), 4)} |"
            f" {_pct(household.get('mean_accuracy'))} | {_num(form.get('mean_logloss'), 4)} |"
            f" {_pct(form.get('mean_accuracy'))} | {_num(plan.get('logloss'), 4)} |"
        )
    baseline = payload["models"]["baselines"]
    add(
        f"| 基准 | 题内均匀 | {_num(baseline['uniform']['logloss'], 4)} |"
        f" {_pct(baseline['uniform']['accuracy'])} | — | — | — |"
    )
    add(
        f"| 基准 | 题组历史频率 | {_num(baseline['form_frequency']['logloss'], 4)} |"
        f" {_pct(baseline['form_frequency']['accuracy'])} | — | — | — |"
    )
    add("")
    add(
        "> 读法：规格 B 在家庭留出上更好，但在整版留出上会把没见过的题组组合外推得过远；"
        "因此主口径用 B 出分群结果，同时用 A 做保守对照、用 C 检查结论是否只依赖价格函数形式。"
        "若三者在候选排序上不一致，报告并列写出，而不是只报一个名次。"
    )
    add("")
    add("### 主口径系数（效用单位，非金额）")
    add("")
    add("| 项 | 系数 | 聚类稳健标准误 |")
    add("| --- | ---: | ---: |")
    for name, value in payload["models"]["primary"]["coefficients"].items():
        se = payload["models"]["primary"]["standard_errors"].get(name)
        add(f"| {name} | {_num(value, 3)} | {_num(se, 3)} |")
    add("")
    add("### 未观测异质性的稳健性（两类别潜类模型）")
    add("")
    latent = payload["models"]["latent_class"]
    add(
        f"固定种子 {latent['seed']} 的两类别潜类模型类别权重为 "
        + "、".join(f"{weight:.2f}" for weight in latent["class_probability"])
        + "。它只用于检查合并模型是否掩盖了两类家庭，不用于分群点估计——"
        "类别归属无法在采购里直接识别。"
    )
    add("")
    add("| 类别 | asc_sq | hours | copay | night | weekend | site | accompanying |")
    add("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for index, coefficients in enumerate(latent["coefficients"]):
        values = [
            coefficients.get(name, 0.0)
            for name in (
                "asc_sq",
                "hours",
                "copay",
                "night",
                "weekend",
                "site",
                "accompanying",
            )
        ]
        formatted = " | ".join(_num(value, 2) for value in values)
        add(f"| {index + 1} | {formatted} |")
    add("")

    add("## 三、分区与分群的需求、替代关系与区间")
    add("")
    add(
        "份额按每户实际看到的版本菜单计算：同户看过 3 个版本，其抽样权平均分到 3 个菜单，"
        "因此不重复计数，statusquo 份额也与实际选择率相符。未在本区展示的服务单独列为外推警示，"
        "不进入份额。区间来自参数抽样（80% 区间）。"
    )
    add("")
    for ward in sorted(payload["demand"]):
        demand = payload["demand"][ward]
        add(f"### {ward}")
        add("")
        add(
            f"发出展示 {demand['delivered_displays']} 个，家庭 {demand['households']} 户，"
            f"估计人口 {demand['population_estimate']:.1f} 户；共 {len(demand.get('menus', []))} 个展示版本。"
            "各版本出现过的选项："
            + "、".join(demand["options"])
            + "。"
        )
        add("")
        add("| 选项 | 总份额 | 80% 区间 | 估计户数 | 全职 | 夜班 | 受限 |")
        add("| --- | ---: | --- | ---: | ---: | ---: | ---: |")
        for option in demand["options"]:
            interval = demand.get("intervals", {}).get("options", {}).get(option, {})
            by_stratum = demand["by_stratum"]
            cells = []
            for stratum in ("fulltime", "nightshift", "limited"):
                info = by_stratum.get(stratum)
                cells.append(_pct(info["shares"].get(option)) if info else "—")
            add(
                f"| {option} | {_pct(demand['overall_shares'][option])} |"
                f" {_pct(interval.get('low'))} – {_pct(interval.get('high'))} |"
                f" {demand['overall_counts'][option]:.1f} | {' | '.join(cells)} |"
            )
        add("")
        warning = demand.get("extrapolation_warning") or []
        if warning:
            for item in warning:
                add(f"- **外推警示**：{item['option']}——{item['text']}")
            add("")
        substitution = payload["substitution"].get(ward)
        if substitution:
            add("替代关系（移除一项后，原份额流向别处的比例）：")
            add("")
            add("| 移除 | 转向夜间 | 转向周末上门 | 转向维持现状 | 主要流向 |")
            add("| --- | ---: | ---: | ---: | --- |")
            for removed, info in substitution["matrix"].items():
                night = sum(
                    value["diversion_ratio"]
                    for option, value in info["flows"].items()
                    if option.startswith("night")
                )
                visit = sum(
                    value["diversion_ratio"]
                    for option, value in info["flows"].items()
                    if option.startswith("visit")
                )
                top = max(
                    info["flows"].items(),
                    key=lambda item: item[1]["diversion_ratio"] or 0.0,
                )
                add(
                    f"| {removed} | {_pct(night)} | {_pct(visit)} |"
                    f" {_pct(info['self_statusquo'] / info['removed_share'] if info['removed_share'] else None)} |"
                    f" {top[0]}（{_pct(top[1]['diversion_ratio'])}） |"
                )
            add("")
    add("## 四、候选组合与预算")
    add("")
    procurement = payload["procurement"]
    add("| 容量点 | 最新容量 | 可见日期 | 已承诺 | 已锁定 |")
    add("| --- | ---: | --- | ---: | ---: |")
    for key, info in procurement["capacity"].items():
        add(
            f"| {key} | {info['cap_units']} | {info['visible_from']} |"
            f" {info['promised_units']} | {info['locked_units']} |"
        )
    add("")
    add(
        "订单为整包时段，同组捆绑必须同包数；班次容量先扣已答应家庭的过渡服务，"
        "再扣此前锁定订单。预算仅用于本轮新购。"
    )
    add("")
    add(
        "增益按每户实际看到的版本菜单计算：购买名额先按各版本的需求比例分配，"
        "再在版本内按“名额/需求”配给；被削减的家庭转向本版本仍可选的其它服务或维持现状。"
        "增益是效用单位，不是金额；跨区可相加是因为共用同一偏好尺度。"
    )
    add("")
    add(
        "下表已剔除“多买不多得”的组合：与更便宜的方案相比，估计增益和服务户数相同、"
        "却多买了名额——多出的名额没有被需求吸收。"
    )
    add("")
    add("| 排名 | 组合 | 新购费用 | 估计增益 | 增益 80% 区间 | 估计服务户数 | 每户成本 |")
    add("| ---: | --- | ---: | ---: | --- | ---: | ---: |")
    for index, item in enumerate(payload["procurement"]["ranking"]["top"], 1):
        interval = item.get("interval") or {}
        benefit_interval = interval.get("benefit") or {}
        add(
            f"| {index} | {item['basket_text']} | {_money(item['totals']['cost_cents'])} |"
            f" {_num(item['totals']['benefit'], 1)} |"
            f" {_num(benefit_interval.get('low'), 1)} – {_num(benefit_interval.get('high'), 1)} |"
            f" {item['totals']['served']:.1f} |"
            f" {_money(item['totals']['cost_per_served_cents'])} |"
        )
    add("")
    add("### 假设敏感性：换一个偏好口径，排序会变吗")
    add("")
    sensitivity = payload["sensitivity"]
    add(
        f"同一组候选组合在 {len(sensitivity['models'])} 个偏好口径下，"
        + ("首选组合一致" if sensitivity["stable"] else "首选组合不一致")
        + (
            "；但第二、第三名的次序会随口径变化，采购会应按首选组合讨论，"
            "并把次优差异写进记录。"
            if sensitivity.get("orders", 1) > 1
            else "。"
        )
    )
    add("")
    if not sensitivity["stable"] and sensitivity.get("leader_differences"):
        add("各口径下的领先者与领先幅度：")
        add("")
        add("| 口径 | 领先组合 | 增益 | 次优组合 | 领先幅度 |")
        add("| --- | --- | ---: | --- | ---: |")
        for label, info in sensitivity["leader_differences"].items():
            add(
                f"| {label} | {info['leader']} | {_num(info['leader_benefit'], 1)} |"
                f" {info['runner_up'] or '—'} | {_num(info['margin'], 1)} |"
            )
        add("")
    add("| 口径 | 第一 | 第二 | 第三 |")
    add("| --- | --- | --- | --- |")
    for model, rows in sensitivity["ranking"].items():
        cells = []
        for row in rows[:3]:
            cells.append(f"{row.get('basket_text') or row['basket']}（{row['benefit']:.1f}）")
        while len(cells) < 3:
            cells.append("—")
        add(f"| {model} | {' | '.join(cells)} |")
    add("")
    if payload["procurement"]["conflicts"]:
        add("### 不能一并推荐的组合")
        add("")
        for conflict in payload["procurement"]["conflicts"]:
            shared = conflict["shared_capacity"][0] if conflict["shared_capacity"] else {}
            add(
                f"- **{conflict['left']} + {conflict['right']}**：分别可行，合并后在 "
                f"{shared.get('provider')}/{shared.get('period')} 需要 "
                f"{shared.get('combined_units')} 个班次单位，可用仅 "
                f"{shared.get('available_units')}；会挤占 "
                + ("、".join(p["promise"] for p in conflict["at_risk_promises"]) or "其它承诺")
                + "。"
            )
        add("")
    reserve = payload["procurement"]["reserve"]
    add("### 证据缺口与预算预留")
    add("")
    add(
        f"共有 {reserve['gap_count']} 个订单区位缺乏可比证据，按每处 "
        f"{_money(assets.assumptions.reserve_per_gap_cents)} 预留 "
        f"{_money(reserve['reserve_cents'])} 用于补充访问；"
        f"可用于正式承购的预算为 {_money(reserve['purchasable_budget_cents'])}。"
    )
    add("")
    for gap in reserve["gaps"]:
        add(f"- {gap['ward']} / {gap['option']}（{gap['offer']}）：{gap['reason']}。")
    add("")
    if reserve.get("blocked_tie_groups"):
        add("捆绑组的连锁影响：")
        add("")
        for group in reserve["blocked_tie_groups"]:
            add(
                f"- 捆绑组 {group['tie']}（{'、'.join(group['offers'])}）中 "
                f"{'、'.join(group['blocked_by'])} 缺少可比证据；因同组必须同包数，"
                f"{'、'.join(group['collateral']) or '同组其它项'}本轮也不能正式承购；"
                "需先补充访问或另议拆组。"
            )
        add("")
    add("## 五、分区解释")
    add("")
    for ward, info in payload["ward_explanations"].items():
        add(f"### {ward}")
        add("")
        add(info["explanation"])
        add("")
        survey = info["survey"]
        add(
            f"- 抽样 {survey['households_sampled']} 户，实际展示 {survey['displays_delivered']} 个题组，"
            f"有效选择 {survey['choice_tasks']} 个，主动维持现状 {survey['statusquo_choices']} 个，"
            f"跳题 {survey['skips']} 个，漏发 {survey['not_delivered']} 个。"
        )
        add(
            f"- 有可比证据的选项：{'、'.join(info['evidence']['options_with_comparable_evidence']) or '无'}。"
        )
        add("")
        add("证据范围（有效题组数；括号内为证据门槛，达到门槛记 ✓）：")
        add("")
        cells = [
            cell
            for cell in payload["evidence"]["cells"]
            if cell["ward"] == ward and cell["delivered_displays"] > 0
        ]
        options = sorted({cell["option"] for cell in cells})
        strata = sorted({cell["stratum"] for cell in cells})
        if cells:
            add("| 选项 | " + " | ".join(strata) + " |")
            add("| --- | " + " | ".join("---:" for _ in strata) + " |")
            lookup = {(cell["option"], cell["stratum"]): cell for cell in cells}
            for option in options:
                row = []
                for stratum in strata:
                    cell = lookup.get((option, stratum))
                    if cell is None:
                        row.append("—")
                    else:
                        mark = "✓" if cell["comparable"] else "✗"
                        row.append(f"{cell['choice_tasks']}（{cell['evidence_min']}）{mark}")
                add(f"| {option} | " + " | ".join(row) + " |")
            add("")
        if info["proposed_purchase"]:
            provided = info["proposed_purchase"].get("provided") or []
            unmet = info["proposed_purchase"].get("unmet") or {}
            add(
                f"- 建议购买：{'、'.join(provided) or '无'}；未满足需求约 "
                f"{sum(unmet.values()):.1f} 户。"
            )
        add("")
    add("## 六、锁定记录")
    add("")
    if payload["locks"]:
        add("| 锁定 | 记录时间 | 组合 | 费用 | 采用的 as_of | 规格 | 增益 | 替代 | 摘要 |")
        add("| --- | --- | --- | ---: | --- | --- | ---: | --- | --- |")
        for lock in payload["locks"]:
            add(
                f"| {lock['lock_id']} | {lock['recorded_at']} | {lock['basket_text']} |"
                f" {_money(lock['cost_cents'])} | {lock['as_of']} | {lock['spec']} |"
                f" {_num(lock['benefit'], 1)} | {lock['supersedes'] or '—'} |"
                f" {lock['digest'][:12]}… |"
            )
        add("")
        add(
            "台账只追加；修订是新记录，标明替代关系。原锁定采用的成本、偏好范围、"
            "保障检查与容量快照均保留，不会被后来的估计覆盖。"
        )
        revised_from = {
            lock["supersedes"]
            for lock in payload["locks"]
            if lock.get("supersedes")
        }
        superseded = [lock for lock in payload["locks"] if lock["lock_id"] in revised_from]
        if superseded:
            add("")
            add("被修订的锁定仍保留原记录；当时采用的成本与容量口径如下：")
            add("")
            for lock in superseded:
                record = next(
                    entry
                    for entry in assets.ledger_entries
                    if entry["lock_id"] == lock["lock_id"]
                )
                snapshot = record["capacity_snapshot"]
                caps = "；".join(
                    f"{point} {info['cap_units']}（{info['visible_from']}）"
                    for point, info in snapshot.items()
                )
                add(
                    f"- {lock['lock_id']}（{lock['as_of']}）：{lock['basket_text']}，"
                    f"费用 {_money(lock['cost_cents'])}，容量 {caps}；其替代记录见上表。"
                )
            add("")
    else:
        add("尚无锁定记录。")
    add("")
    add("## 七、资料与复现")
    add("")
    add(
        "分析只使用项目内来源表（sample / displays / responses / options / offers / "
        "provider_caps / policy / promises）。编号原样保留；所有中间结果可由 "
        "`python3 -m careprocure.cli` 重新生成。"
    )
    add("")
    return "\n".join(lines)


def write_outputs(
    assets: ReportAssets,
    directory: str | Path,
    wards: Sequence[str] | None = None,
) -> dict[str, Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = to_payload(assets, wards)
    json_path = directory / "report.json"
    markdown_path = directory / "report.md"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(assets, wards), encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}
