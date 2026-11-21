# 家庭照护支持采购资料约定

资料模拟属性选择问卷、采购价目和承接容量。不包含实际采购或个人资料。

## sample

登记分区、家庭人群、抽样概率、展示版本和保障家庭标识。每家可能回答多个题组。

字段 `household`, `ward`, `stratum`, `sampling_probability`, `display_plan`, `guaranteed`。

## displays

实际发出的选项集和交付状态，版本之间选项同名不改变属性编号。

字段 `display`, `household`, `form`, `order`, `available`, `delivery`。

## responses

未选、未发与主动选 statusquo 不同。补正用同一题组新登记时间识别。

字段 `answer`, `display`, `choice`, `state`, `seen_on`。

## options

价格和同伴条件等属性，每个 option 编号唯一地指向一组属性。

字段 `option`, `duration_hours`, `time_kind`, `copay_cents`, `mode`, `accompanying`。

## offers

并非所有选项都可按单场订购。每包包含 bundles 个同类型家用名额。cost 为整包价格，period_units 是每包占机构公共班次。ties 同组的包开办数量必须一致。

字段 `offer`, `ward`, `provider`, `period`, `option`, `bundles`, `period_units`, `cost_cents`, `ties`, `max_packs`。

## provider_caps

承接点同一班次类别服务量，减掉已承诺的容量以后方能订新的包。供给减量从 seen_on 起可见。

字段 `update`, `provider`, `period`, `units`, `seen_on`。

## policy

分区最低支持户数，预算为全部分区共用。evidence_min 是数量可比较的最少家庭；原定方案与备选均需覆盖保障底线。

字段 `ward`, `minimum_families`, `evidence_min`, `global_budget_cents`。

## promises

采购前已向家庭锁定的支持资格，占用承接点容量。

字段 `promise`, `household`, `ward`, `provider`, `period`, `committed_units`, `option`。
