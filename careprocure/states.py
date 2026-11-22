"""把展示、答卷与选项展开成可审计的记录状态。

业务上必须区分：

- ``choice_alternative``：主动选了某个新服务；
- ``choice_statusquo``：主动选择维持现状；
- ``skipped``：看到了题组但跳过；
- ``missing``：展示已发出但没有答卷；
- ``not_delivered``：展示没有发到家庭；
- ``response_on_undelivered``：登记了答卷但展示状态为未发出（资料冲突）。

另有一层“未选到某项”的暴露状态，见 :func:`exposure_matrix`：
``not_offered``（展示方案里根本没有该项）与
``not_delivered_only``（有该项的题组漏发）都不是“不喜欢”。
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

from .tables import Tables

CHOICE_ALTERNATIVE = "choice_alternative"
CHOICE_STATUSQUO = "choice_statusquo"
SKIPPED = "skipped"
MISSING = "missing"
NOT_DELIVERED = "not_delivered"
RESPONSE_ON_UNDELIVERED = "response_on_undelivered"


@dataclass(frozen=True)
class ResolvedAnswer:
    display: str
    household: str
    ward: str
    stratum: str
    form: str
    order: int
    delivery: str
    available: tuple[str, ...]
    answer: str | None
    choice: str | None
    state: str
    seen_on: str | None
    superseded: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    @property
    def is_choice(self) -> bool:
        return self.state in (CHOICE_ALTERNATIVE, CHOICE_STATUSQUO)


def _latest_by_display(responses: Iterable[dict[str, Any]], as_of: str | None):
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in responses:
        if as_of is not None and row["seen_on"] > as_of:
            continue
        grouped[row["display"]].append(row)
    latest: dict[str, dict[str, Any]] = {}
    superseded: dict[str, tuple[dict[str, Any], ...]] = {}
    for display, rows in grouped.items():
        rows.sort(key=lambda r: (r["seen_on"], r["answer"]))
        latest[display] = rows[-1]
        superseded[display] = tuple(rows[:-1])
    return latest, superseded


def resolve_answers(tables: Tables, as_of: str | None = None) -> list[ResolvedAnswer]:
    """按展示逐条给出最终状态；补正取同一题组最晚登记的答卷。

    未发出的展示即使有答卷也单独标成冲突，不进入选择模型。
    """
    households = {row["household"]: row for row in tables.sample}
    latest, superseded = _latest_by_display(tables.responses, as_of)
    resolved: list[ResolvedAnswer] = []
    for display in tables.displays:
        row = latest.get(display["display"])
        household = households[display["household"]]
        if display["delivery"] != "shown":
            state = RESPONSE_ON_UNDELIVERED if row is not None else NOT_DELIVERED
            answer = row["answer"] if row else None
            choice = row["choice"] if row else None
            seen_on = row["seen_on"] if row else None
        elif row is None:
            state = MISSING
            answer = choice = seen_on = None
        elif row["state"] == SKIPPED:
            state = SKIPPED
            answer, choice, seen_on = row["answer"], None, row["seen_on"]
        elif row["choice"] == "statusquo":
            state = CHOICE_STATUSQUO
            answer, choice, seen_on = row["answer"], "statusquo", row["seen_on"]
        else:
            state = CHOICE_ALTERNATIVE
            answer, choice, seen_on = row["answer"], row["choice"], row["seen_on"]
        resolved.append(
            ResolvedAnswer(
                display=display["display"],
                household=display["household"],
                ward=household["ward"],
                stratum=household["stratum"],
                form=display["form"],
                order=display["order"],
                delivery=display["delivery"],
                available=tuple(display["available"]),
                answer=answer,
                choice=choice,
                state=state,
                seen_on=seen_on,
                superseded=superseded.get(display["display"], ()),
            )
        )
    resolved.sort(key=lambda r: (r.household, r.order, r.display))
    return resolved


def audit_counts(answers: list[ResolvedAnswer]) -> dict[str, Any]:
    state_counts = Counter(a.state for a in answers)
    deliveries = Counter(a.delivery for a in answers)
    corrections = [a for a in answers if a.superseded]
    return {
        "displays": len(answers),
        "delivery": dict(sorted(deliveries.items())),
        "states": dict(sorted(state_counts.items())),
        "delivered_displays": deliveries.get("shown", 0),
        "choice_tasks": state_counts.get(CHOICE_ALTERNATIVE, 0)
        + state_counts.get(CHOICE_STATUSQUO, 0),
        "chose_statusquo": state_counts.get(CHOICE_STATUSQUO, 0),
        "chose_alternative": state_counts.get(CHOICE_ALTERNATIVE, 0),
        "skipped": state_counts.get(SKIPPED, 0),
        "not_delivered": state_counts.get(NOT_DELIVERED, 0)
        + state_counts.get(RESPONSE_ON_UNDELIVERED, 0),
        "missing": state_counts.get(MISSING, 0),
        "corrected_displays": len(corrections),
        "superseded_answers": sum(len(a.superseded) for a in corrections),
        "corrections": [
            {
                "display": a.display,
                "household": a.household,
                "kept_answer": a.answer,
                "kept_choice": a.choice,
                "kept_seen_on": a.seen_on,
                "superseded": [
                    {
                        "answer": old["answer"],
                        "choice": old["choice"],
                        "state": old["state"],
                        "seen_on": old["seen_on"],
                    }
                    for old in a.superseded
                ],
            }
            for a in corrections
        ],
    }


def exposure_matrix(tables: Tables, as_of: str | None = None) -> dict[str, dict[str, dict[str, Any]]]:
    """每个家庭对每个选项的暴露与选择。

    返回 ``matrix[household][option]``，其中：

    - ``seen``：已发出的题组中包含该选项的次数；
    - ``not_delivered``：该选项所在的漏发题组次数；
    - ``chosen``：最终选择该选项的次数；
    - ``skipped``：看到该选项但跳过的次数；
    - ``state``：``chosen`` / ``seen_not_chosen`` / ``not_delivered_only`` / ``not_offered``。
    """
    answers = resolve_answers(tables, as_of)
    options = [row["option"] for row in tables.options]
    matrix: dict[str, dict[str, dict[str, Any]]] = {
        row["household"]: {
            option: {
                "seen": 0,
                "not_delivered": 0,
                "chosen": 0,
                "skipped": 0,
                "state": "not_offered",
            }
            for option in options
        }
        for row in tables.sample
    }
    for answer in answers:
        for option in answer.available:
            cell = matrix[answer.household][option]
            if answer.delivery == "shown":
                cell["seen"] += 1
                if answer.state == SKIPPED:
                    cell["skipped"] += 1
                elif answer.choice == option:
                    cell["chosen"] += 1
            else:
                cell["not_delivered"] += 1
    for household, cells in matrix.items():
        for option, cell in cells.items():
            if cell["chosen"]:
                cell["state"] = "chosen"
            elif cell["seen"]:
                cell["state"] = "seen_not_chosen"
            elif cell["not_delivered"]:
                cell["state"] = "not_delivered_only"
            else:
                cell["state"] = "not_offered"
    return matrix


def state_by_household(answers: list[ResolvedAnswer]) -> dict[str, list[ResolvedAnswer]]:
    grouped: dict[str, list[ResolvedAnswer]] = defaultdict(list)
    for answer in answers:
        grouped[answer.household].append(answer)
    return dict(grouped)
