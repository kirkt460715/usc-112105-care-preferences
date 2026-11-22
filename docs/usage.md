# 使用说明：从问卷证据到采购组合

本目录把项目资料变成可复核的分析与可审计的锁定记录。所有编号（家庭、展示、选项、订单、承诺）与来源表一致，不另行编号。

## 快速开始

```sh
# 1. 看清记录状态与证据边界
python3 -m careprocure.cli audit --data data

# 2. 比较偏好模型与留出表现
python3 -m careprocure.cli models --data data

# 3. 分区需求、替代关系与外推警示
python3 -m careprocure.cli demand --data data

# 4. 候选订单：哪些能按同一组假设落实、哪些人群有得益证据
python3 -m careprocure.cli candidates --data data --as-of 2026-10-25

# 5. 提交一组的检查（含容量、捆绑、预算、保障与证据）
python3 -m careprocure.cli candidates --data data \
    --basket "o3x1,o5x2,o6x1,o7x1"

# 6. 生成采购会与分区说明
python3 -m careprocure.cli report --data data --output outputs
```

## 默认口径

- **资料截止日 `--as-of 2026-10-25`**：只看该日及之前可见的容量与答卷。`provider_caps` 的减量从 `seen_on` 起生效；用 `--as-of 2026-09-25` 可复原减量前的容量。
- **主偏好口径 `--spec B`**：属性×人群 MNL，人群差异用岭惩罚收缩。`A` 是合并属性模型（保守对照），`C` 是选项固定效应模型（不依赖价格函数形式）。
- **预算 `--budget 515000`（5,150 元）**：全部分区共用的本轮新购预算，不含已另行拨款的锁定订单。
- **补充访问预留 `--reserve-per-gap 25000`（250 元/处）**：每个缺乏可比证据的订单区位预留一份，从预算中先扣除；正式承购只能用扣后的余额。
- **抽样权**：家庭权重 `1 / sampling_probability`；同一户看过多个展示版本时，权重平均分到各版本，避免重复计数。

## 记录状态的口径

| 状态 | 含义 | 是否进入偏好模型 |
| --- | --- | --- |
| `choice_alternative` | 主动选择某个新服务 | 是 |
| `choice_statusquo` | 主动选择维持现状 | 是（不是“未选”） |
| `skipped` | 看到题组但跳过 | 否 |
| `not_delivered` | 展示未发出 | 否 |
| `missing` | 已发出但没有答卷 | 否 |

补正取同一题组最晚登记的答卷，被替代的答卷只作为历史保留在报告里。同一题组若登记了答卷但展示状态是未发出，会标成 `response_on_undelivered` 冲突，不进入模型。

## 证据与可比性

- 每个订单选项在“区 × 人群”上单独检查有效题组数是否达到 `policy.evidence_min`。
- 展示方案之间靠共同展示的选项搭桥；`ridge` 的 `remote` 方案从未展示任何夜间服务，因此 `ridge` 的夜间份额只能是**外推**，不会进入需求、增益或正式承购。
- 缺乏可比证据或从未展示的订单，只能列入**补充访问预留**，不能算已估收益。

## 采购组合检查

`check_basket` 逐条给出：

- `offer_ids` / `nonzero` / `max_packs`：订单编号、包数与开办上限；
- `ties`：同组捆绑订单必须同包数；
- `provider_capacity`：最新可见容量 − 已答应家庭的承诺 − 其它仍在执行的锁定；
- `budget`：扣除补充访问预留后的可承购预算；
- `minimum_families`：各区最低支持户数；
- `evidence`：本区没有达到门槛的可比证据时拦截。

排名会剔除“多买不多得”的组合（与更便宜的组合相比，增益与服务户数相同），并给出增益的 80% 区间。若单个小包效果很好、合并会挤占另一区已承诺的夜间班次，会列入“不能一并推荐的组合”。

## 锁定与修订

```sh
# 锁定一轮采购（只追加台账；用于采购会后留档）
python3 -m careprocure.cli lock --data data --as-of 2026-10-25 \
    --basket "o3x1,o5x2,o6x1,o7x1" \
    --ledger locks/ledger.jsonl --label "2027 首轮"

# 容量减量或问卷更正后，重排尚可调整的项目
python3 -m careprocure.cli revise --data data --as-of 2026-10-25 \
    --lock L001 --ledger locks/ledger.jsonl \
    --fixed o3 --limit 5

# 确认修订后再写一条新锁定（原记录不变）
python3 -m careprocure.cli revise --data data --as-of 2026-10-25 \
    --lock L001 --ledger locks/ledger.jsonl --relock \
    --label "容量减量修订" --reason "clinic/night 由 14 降至 10"
```

锁定记录保存：当时的成本、采用的偏好模型与抽取区间、证据快照、容量快照、承诺占用、各项检查与预算预留。台账每行带内容摘要，手工改一行会被检出。修订是新记录，`supersedes` 指向被替代的锁定；被替代的记录仍可查，不会被新估计覆盖。

修订报告会列出：哪些项目仍可调整、哪些已固定、容量缺口、受影响的承诺家庭与预计少服务的户数、能同时满足容量与保障底线的替代组合。

## 输出文件

- `outputs/report.md`：采购会主报告与分区解释。
- `outputs/report.json`：全部中间结果，便于复核或接入其它系统。
- `examples/ledger.jsonl`、`examples/lock-L001.json`、`examples/revision-L001.json`：
  一条“九月锁定 → 十月夜班容量减量 → 修订”的示例记录，用于核对台账格式与修订报告。
  它不是正式承诺，正式台账请另指路径（如 `--ledger locks/ledger.jsonl`）。

## 测试

```sh
python3 -m unittest discover -s tests -v
```

测试覆盖：漏发/跳题/主动现状/补正的区分、补正取最晚答卷、未展示选项不得当作需求、同户多版本不重复计数、份额与区间、捆绑同包数、容量减量可见性、承诺先扣、跨区冲突、证据缺口预留、锁定不可覆盖与台账防篡改。
