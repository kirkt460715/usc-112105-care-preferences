# 家庭照护支持采购

资料模拟属性选择问卷、采购价目和承接容量。不包含实际采购或个人资料。

项目使用 Python 3.11.2 标准库。资料按来源表以 UTF-8 JSON Lines 保存，每行是一条记录。缺失值写作 `null`，其业务含义见资料约定。列名见 `data/manifest.json`，编号由各资料表原样保留。

[资料约定](docs/materials.md)说明各表的来源、时间及关联语义。[使用说明](docs/usage.md)说明从问卷证据到采购组合的完整口径与命令。

## 这套分析解决什么

- 不让“漏发”“跳题”“本区没有夜间服务”被当成“不喜欢”，主动选现状与未选严格分开；
- 只把属性一致或有搭桥支持的展示版本放在同一比较里；对没有共同展示机会的人群保留不确定性，模型外推不当作在地需求；
- 比较合理的偏好模型与估计口径，在留出的家庭与题组上检查预测表现，给出分群替代关系、证据范围与变化区间；
- 按整包订单、捆绑同包数、承接点共用班次、承诺先扣、预算与保障底线检查候选组合；证据不足处只列补充访问预留；
- 锁定记录保留当时采用的成本、偏好范围、保障检查与容量快照，后续估计或容量减量不会覆盖原承诺。

## 常用命令

```sh
python3 inspect_inputs.py
python3 -m unittest discover -s tests -v

python3 -m careprocure.cli audit    --data data
python3 -m careprocure.cli models   --data data
python3 -m careprocure.cli demand   --data data
python3 -m careprocure.cli candidates --data data --basket "o3x1,o5x2,o6x1,o7x1"
python3 -m careprocure.cli report   --data data --output outputs
```

采购会报告见 `outputs/report.md`（`outputs/report.json` 为可复核的完整结果）。
