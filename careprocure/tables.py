"""读入来源表并做结构校验。

资料只读；任何分析都从这里的原始记录出发，编号原样保留。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

TABLE_NAMES = (
    "sample",
    "displays",
    "responses",
    "options",
    "offers",
    "provider_caps",
    "policy",
    "promises",
)


class MaterialError(ValueError):
    """来源资料与登记的字段或关联口径不符。"""


@dataclass(frozen=True)
class Tables:
    sample: list[dict[str, Any]]
    displays: list[dict[str, Any]]
    responses: list[dict[str, Any]]
    options: list[dict[str, Any]]
    offers: list[dict[str, Any]]
    provider_caps: list[dict[str, Any]]
    policy: list[dict[str, Any]]
    promises: list[dict[str, Any]]

    def by(self, name: str) -> list[dict[str, Any]]:
        return list(getattr(self, name))


def _read_table(directory: Path, name: str, columns: Iterable[str]) -> list[dict[str, Any]]:
    expected = set(columns)
    records: list[dict[str, Any]] = []
    with (directory / f"{name}.jsonl").open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise MaterialError(f"{name} 第 {number} 行不是有效 JSON：{error}") from error
            if not isinstance(record, dict) or record.keys() != expected:
                raise MaterialError(f"{name} 第 {number} 行的字段与登记格式不符")
            records.append(record)
    if not records:
        raise MaterialError(f"{name} 没有任何记录")
    return records


def read_tables(directory: str | Path) -> Tables:
    """读取一个资料目录；字段名与 ``manifest.json`` 完全一致才接受。"""
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MaterialError(f"无法读取 manifest.json：{error}") from error
    missing = [name for name in TABLE_NAMES if name not in manifest]
    if missing:
        raise MaterialError(f"manifest.json 缺少表：{', '.join(missing)}")
    payload = {name: _read_table(directory, name, manifest[name]) for name in TABLE_NAMES}
    tables = Tables(**payload)
    validate_links(tables)
    return tables


def validate_links(tables: Tables) -> None:
    """检查跨表编号能否对应；不在这里解释业务状态。"""
    households = {row["household"] for row in tables.sample}
    if len(households) != len(tables.sample):
        raise MaterialError("sample 中家庭编号重复")
    display_ids = {row["display"] for row in tables.displays}
    if len(display_ids) != len(tables.displays):
        raise MaterialError("displays 中展示编号重复")
    for row in tables.displays:
        if row["household"] not in households:
            raise MaterialError(f"展示 {row['display']} 指向未登记家庭 {row['household']}")
        if not row["available"] or len(set(row["available"])) != len(row["available"]):
            raise MaterialError(f"展示 {row['display']} 的选项集为空或有重复")
    display_to_household = {row["display"]: row["household"] for row in tables.displays}
    answer_ids: set[str] = set()
    for row in tables.responses:
        if row["answer"] in answer_ids:
            raise MaterialError(f"答卷编号 {row['answer']} 重复")
        answer_ids.add(row["answer"])
        if row["display"] not in display_to_household:
            raise MaterialError(f"答卷 {row['answer']} 指向未登记展示 {row['display']}")
        if row["state"] not in ("choice", "skipped"):
            raise MaterialError(f"答卷 {row['answer']} 的 state 不在约定范围内")
        if row["state"] == "choice" and row["choice"] is None:
            raise MaterialError(f"答卷 {row['answer']} 标记为选择却没有选项")
        if row["state"] == "skipped" and row["choice"] is not None:
            raise MaterialError(f"答卷 {row['answer']} 标记为跳题却带有选项")
        if row["choice"] is not None:
            display = next(d for d in tables.displays if d["display"] == row["display"])
            if row["choice"] not in display["available"]:
                raise MaterialError(
                    f"答卷 {row['answer']} 选择了展示中不存在的选项 {row['choice']}"
                )
    options = {row["option"] for row in tables.options}
    if len(options) != len(tables.options):
        raise MaterialError("options 中选项编号重复")
    if "statusquo" not in options:
        raise MaterialError("options 缺少 statusquo 基线")
    for row in tables.displays:
        unknown = set(row["available"]) - options
        if unknown:
            raise MaterialError(f"展示 {row['display']} 含未登记选项 {sorted(unknown)}")
    for row in tables.offers:
        if row["option"] not in options - {"statusquo"}:
            raise MaterialError(f"订单 {row['offer']} 指向不可采购选项 {row['option']}")
        if row["period_units"] <= 0 or row["cost_cents"] <= 0:
            raise MaterialError(f"订单 {row['offer']} 的班次占用或费用非正")
    promise_ids = {row["promise"] for row in tables.promises}
    if len(promise_ids) != len(tables.promises):
        raise MaterialError("promises 中承诺编号重复")
    for row in tables.promises:
        if row["household"] not in households:
            raise MaterialError(f"承诺 {row['promise']} 指向未登记家庭")
    caps = {(row["provider"], row["period"], row["seen_on"]) for row in tables.provider_caps}
    if len(caps) != len(tables.provider_caps):
        raise MaterialError("provider_caps 同一承接点、班次与可见日期重复")
