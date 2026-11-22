"""竞争性偏好模型：拟合、验证、不确定性与潜在类别稳健性。

估计口径（全部只用已发出且获得有效选择的题组）：

- 每条题组是一个选择集，选项集就是家庭实际看到的 ``available``；
- 漏发、跳题、未应答不进入似然，也不被当作“没选”；
- 家庭权重为 ``1 / sampling_probability``，使各区的抽样设计反映到汇总；
- 补正取同一题组最晚登记的答卷。

模型规格：

- ``A``：属性合并 MNL（价格、时长、时段、场所、陪同）；
- ``B``：属性 × 人群交互 MNL（人群差异用岭惩罚收缩）；
- ``C``：选项固定效应 × 人群交互（不依赖价格符号，但无法外推新属性）；
- ``L``：两类别潜类模型（未观测异质性稳健性，不用于分群点估计）。
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .states import CHOICE_ALTERNATIVE, CHOICE_STATUSQUO, resolve_answers
from .tables import Tables

STATUSQUO = "statusquo"
ATTRIBUTE_NAMES = ("asc_sq", "hours", "copay", "night", "weekend", "site", "accompanying")
OPTION_ORDER = ("nightA", "nightB", "visitA", "visitB", "satA", "trip", "short")
SEGMENTS = ("fulltime", "nightshift", "limited")


@dataclass(frozen=True)
class ChoiceTask:
    display: str
    household: str
    ward: str
    stratum: str
    form: str
    options: tuple[str, ...]
    choice: str
    weight: float

    @property
    def choice_index(self) -> int:
        return self.options.index(self.choice)


@dataclass(frozen=True)
class Spec:
    key: str
    label: str
    kind: str  # "attribute" | "option"
    segment_interactions: bool
    ridge: float = 0.0
    description: str = ""


SPECS: dict[str, Spec] = {
    "A": Spec(
        key="A",
        label="属性合并 MNL",
        kind="attribute",
        segment_interactions=False,
        description="七项属性合并估计；跨区跨人群共用一套系数。",
    ),
    "B": Spec(
        key="B",
        label="属性×人群 MNL",
        kind="attribute",
        segment_interactions=True,
        ridge=3.0,
        description="以 fulltime 为基准，nightshift/limited 的差异用岭惩罚收缩。",
    ),
    "C": Spec(
        key="C",
        label="选项×人群 MNL",
        kind="option",
        segment_interactions=True,
        ridge=3.0,
        description="七个选项固定效应加人群交互；不解释价格系数，适合现有菜单的需求预测。",
    ),
}


def build_tasks(tables: Tables, as_of: str | None = None) -> list[ChoiceTask]:
    answers = resolve_answers(tables, as_of)
    return [
        ChoiceTask(
            display=a.display,
            household=a.household,
            ward=a.ward,
            stratum=a.stratum,
            form=a.form,
            options=a.available,
            choice=a.choice or "",
            weight=1.0 / _sampling_probability(tables, a.household),
        )
        for a in answers
        if a.state in (CHOICE_ALTERNATIVE, CHOICE_STATUSQUO)
    ]


def _sampling_probability(tables: Tables, household: str) -> float:
    for row in tables.sample:
        if row["household"] == household:
            probability = float(row["sampling_probability"])
            if not 0 < probability <= 1:
                raise ValueError(f"家庭 {household} 的抽样概率不在 (0, 1]")
            return probability
    raise KeyError(household)


def _option_attributes(tables: Tables) -> dict[str, dict[str, Any]]:
    return {row["option"]: row for row in tables.options}


def feature_names(spec: Spec) -> list[str]:
    base = (
        list(ATTRIBUTE_NAMES)
        if spec.kind == "attribute"
        else [f"opt_{option}" for option in OPTION_ORDER]
    )
    if not spec.segment_interactions:
        return base
    names = list(base)
    for segment in SEGMENTS[1:]:
        names.extend(f"{name}×{segment}" for name in base)
    return names


def feature_vector(
    tables: Tables, spec: Spec, option: str, stratum: str
) -> list[float]:
    attributes = _option_attributes(tables)
    if spec.kind == "attribute":
        row = attributes[option]
        base = [
            1.0 if option == STATUSQUO else 0.0,
            float(row["duration_hours"]),
            float(row["copay_cents"]) / 100.0,
            1.0 if row["time_kind"] == "night" else 0.0,
            1.0 if row["time_kind"] == "weekend" else 0.0,
            1.0 if row["mode"] == "site" else 0.0,
            1.0 if row["accompanying"] else 0.0,
        ]
    else:
        base = [1.0 if option == name else 0.0 for name in OPTION_ORDER]
    if not spec.segment_interactions:
        return base
    vector = list(base)
    for segment in SEGMENTS[1:]:
        indicator = 1.0 if stratum == segment else 0.0
        vector.extend(indicator * value for value in base)
    return vector


def penalty_vector(spec: Spec, ridge: float | None = None) -> list[float]:
    names = feature_names(spec)
    strength = spec.ridge if ridge is None else ridge
    base_count = len(ATTRIBUTE_NAMES) if spec.kind == "attribute" else len(OPTION_ORDER)
    penalties = []
    for index, _ in enumerate(names):
        if index < base_count:
            penalties.append(1e-7)
        else:
            penalties.append(strength)
    return penalties


@dataclass
class FitResult:
    spec: Spec
    ridge: float
    beta: list[float]
    names: list[str]
    loglik: float
    iterations: int
    converged: bool
    covariance: list[list[float]] | None = None
    n_tasks: int = 0
    n_households: int = 0

    def coefficients(self) -> dict[str, float]:
        return {name: value for name, value in zip(self.names, self.beta)}

    @property
    def standard_errors(self) -> list[float] | None:
        if self.covariance is None:
            return None
        return [math.sqrt(max(self.covariance[i][i], 0.0)) for i in range(len(self.beta))]


def _probabilities(beta: Sequence[float], rows: Sequence[Sequence[float]]) -> list[float]:
    utilities = [sum(x * b for x, b in zip(row, beta)) for row in rows]
    top = max(utilities)
    exponentials = [math.exp(value - top) for value in utilities]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def _task_rows(
    tables: Tables, spec: Spec, tasks: Iterable[ChoiceTask]
) -> list[tuple[list[list[float]], int, float, ChoiceTask]]:
    prepared = []
    for task in tasks:
        rows = [feature_vector(tables, spec, option, task.stratum) for option in task.options]
        prepared.append((rows, task.choice_index, task.weight, task))
    return prepared


def _weighted_loglik(beta, prepared, penalties) -> float:
    total = 0.0
    for rows, y, weight, _ in prepared:
        utilities = [sum(x * b for x, b in zip(row, beta)) for row in rows]
        top = max(utilities)
        logsum = top + math.log(sum(math.exp(u - top) for u in utilities))
        total += weight * (utilities[y] - logsum)
    total -= 0.5 * sum(p * b * b for p, b in zip(penalties, beta))
    return total


def _gradient(beta, prepared, penalties):
    p = len(beta)
    gradient = [0.0] * p
    for rows, y, weight, _ in prepared:
        probabilities = _probabilities(beta, rows)
        for j, row in enumerate(rows):
            factor = weight * ((1.0 if j == y else 0.0) - probabilities[j])
            for k in range(p):
                gradient[k] += factor * row[k]
    for k in range(p):
        gradient[k] -= penalties[k] * beta[k]
    return gradient


def _hessian(beta, prepared, penalties):
    p = len(beta)
    hessian = [[0.0] * p for _ in range(p)]
    for rows, _, weight, _ in prepared:
        probabilities = _probabilities(beta, rows)
        means = [
            sum(probabilities[j] * rows[j][k] for j in range(len(rows))) for k in range(p)
        ]
        for j, row in enumerate(rows):
            for k in range(p):
                left = row[k] - means[k]
                for l in range(k, p):
                    hessian[k][l] -= weight * probabilities[j] * left * (row[l] - means[l])
    for k in range(p):
        hessian[k][k] -= penalties[k]
        for l in range(k):
            hessian[k][l] = hessian[l][k]
    # 对数似然的二阶导（负定）。牛顿步解 H·step = -gradient。
    return hessian


def _solve(matrix, vector):
    n = len(vector)
    augmented = [list(matrix[i]) + [vector[i]] for i in range(n)]
    for column in range(n):
        pivot = max(range(column, n), key=lambda i: abs(augmented[i][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            return None
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        value = augmented[column][column]
        for j in range(column, n + 1):
            augmented[column][j] /= value
        for i in range(n):
            if i == column:
                continue
            factor = augmented[i][column]
            if factor == 0.0:
                continue
            for j in range(column, n + 1):
                augmented[i][j] -= factor * augmented[column][j]
    return [augmented[i][n] for i in range(n)]


def _invert(matrix):
    n = len(matrix)
    augmented = [
        list(matrix[i]) + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)
    ]
    for column in range(n):
        pivot = max(range(column, n), key=lambda i: abs(augmented[i][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            return None
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        value = augmented[column][column]
        for j in range(2 * n):
            augmented[column][j] /= value
        for i in range(n):
            if i == column:
                continue
            factor = augmented[i][column]
            if factor == 0.0:
                continue
            for j in range(2 * n):
                augmented[i][j] -= factor * augmented[column][j]
    return [row[n:] for row in augmented]


def fit(
    tables: Tables,
    spec: Spec,
    tasks: Sequence[ChoiceTask] | None = None,
    *,
    ridge: float | None = None,
    compute_covariance: bool = True,
    max_iterations: int = 100,
) -> FitResult:
    tasks = list(tasks if tasks is not None else build_tasks(tables))
    if not tasks:
        raise ValueError("没有可用于估计的选择题组")
    prepared = _task_rows(tables, spec, tasks)
    penalties = penalty_vector(spec, ridge)
    p = len(penalties)
    beta = [0.0] * p
    converged = False
    iteration = 0
    for iteration in range(1, max_iterations + 1):
        gradient = _gradient(beta, prepared, penalties)
        hessian = _hessian(beta, prepared, penalties)
        for k in range(p):
            hessian[k][k] += 1e-9
        step = _solve(hessian, [-g for g in gradient])
        if step is None:
            break
        current = _weighted_loglik(beta, prepared, penalties)
        scale = 1.0
        accepted = False
        while scale > 1e-12:
            candidate = [b + scale * s for b, s in zip(beta, step)]
            if _weighted_loglik(candidate, prepared, penalties) >= current - 1e-12:
                accepted = True
                break
            scale *= 0.5
        if not accepted:
            break
        beta = candidate
        if max(abs(scale * s) for s in step) < 1e-9:
            converged = True
            break
    result = FitResult(
        spec=spec,
        ridge=spec.ridge if ridge is None else ridge,
        beta=beta,
        names=feature_names(spec),
        loglik=_weighted_loglik(beta, prepared, penalties),
        iterations=iteration,
        converged=converged,
        n_tasks=len(tasks),
        n_households=len({t.household for t in tasks}),
    )
    if compute_covariance:
        result.covariance = cluster_covariance(tables, spec, tasks, result, penalties)
    return result


def cluster_covariance(
    tables: Tables,
    spec: Spec,
    tasks: Sequence[ChoiceTask],
    result: FitResult,
    penalties: Sequence[float] | None = None,
) -> list[list[float]] | None:
    """按家庭聚类的稳健协方差（加权拟似然）。"""
    prepared = _task_rows(tables, spec, tasks)
    penalties = list(penalties) if penalties is not None else penalty_vector(spec, result.ridge)
    p = len(result.beta)
    meat = [[0.0] * p for _ in range(p)]
    by_household: dict[str, list[float]] = {}
    for rows, y, weight, task in prepared:
        probabilities = _probabilities(result.beta, rows)
        score = [0.0] * p
        for j, row in enumerate(rows):
            factor = weight * ((1.0 if j == y else 0.0) - probabilities[j])
            for k in range(p):
                score[k] += factor * row[k]
        household_score = by_household.setdefault(task.household, [0.0] * p)
        for k in range(p):
            household_score[k] += score[k]
    for score in by_household.values():
        for i in range(p):
            for j in range(i, p):
                meat[i][j] += score[i] * score[j]
    for i in range(p):
        for j in range(i):
            meat[i][j] = meat[j][i]
    # _hessian 返回对数似然的二阶导（负定）；信息矩阵取其相反数。
    information = [[-value for value in row] for row in _hessian(result.beta, prepared, penalties)]
    bread = _invert(information)
    if bread is None:
        return None
    temp = [[sum(bread[i][k] * meat[k][j] for k in range(p)) for j in range(p)] for i in range(p)]
    covariance = [
        [sum(temp[i][k] * bread[k][j] for k in range(p)) for j in range(p)]
        for i in range(p)
    ]
    groups = len(by_household)
    correction = groups / (groups - 1) if groups > 1 else 1.0
    return [
        [covariance[i][j] * correction for j in range(p)] for i in range(p)
    ]


def _cholesky(matrix):
    n = len(matrix)
    lower = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            value = matrix[i][j] - sum(lower[i][k] * lower[j][k] for k in range(j))
            if i == j:
                if value <= 0:
                    return None
                lower[i][j] = math.sqrt(value)
            else:
                lower[i][j] = value / lower[j][j]
    return lower


def draw_parameters(
    result: FitResult, draws: int, seed: int = 7
) -> list[list[float]] | None:
    """从聚类稳健协方差中抽取参数；用于份额与增益的变化区间。"""
    if result.covariance is None:
        return None
    lower = _cholesky(result.covariance)
    if lower is None:
        # 协方差非正定：退回对角近似，仍给出可复现区间。
        lower = [[0.0] * len(result.beta) for _ in range(len(result.beta))]
        for i in range(len(result.beta)):
            lower[i][i] = math.sqrt(max(result.covariance[i][i], 0.0))
    rng = random.Random(seed)
    p = len(result.beta)
    samples: list[list[float]] = []
    for _ in range(draws):
        z = [rng.gauss(0.0, 1.0) for _ in range(p)]
        samples.append(
            [
                result.beta[i] + sum(lower[i][k] * z[k] for k in range(i + 1))
                for i in range(p)
            ]
        )
    return samples


def predict(
    tables: Tables,
    result: FitResult,
    options: Sequence[str],
    stratum: str,
    beta: Sequence[float] | None = None,
) -> dict[str, float]:
    rows = [feature_vector(tables, result.spec, option, stratum) for option in options]
    probabilities = _probabilities(result.beta if beta is None else beta, rows)
    return dict(zip(options, probabilities))


def logsum(
    tables: Tables,
    result: FitResult,
    options: Sequence[str],
    stratum: str,
    beta: Sequence[float] | None = None,
) -> float:
    vector = result.beta if beta is None else beta
    utilities = [
        sum(
            x * b
            for x, b in zip(feature_vector(tables, result.spec, option, stratum), vector)
        )
        for option in options
    ]
    top = max(utilities)
    return top + math.log(sum(math.exp(value - top) for value in utilities))


# ---------------------------------------------------------------------------
# 验证
# ---------------------------------------------------------------------------


def _folds(tables: Tables, tasks: Sequence[ChoiceTask], count: int, seed: int = 42):
    households = sorted({task.household for task in tasks})
    population = {row["household"]: row for row in tables.sample}
    cells: dict[tuple[str, str], list[str]] = {}
    for household in households:
        key = (population[household]["ward"], population[household]["stratum"])
        cells.setdefault(key, []).append(household)
    rng = random.Random(seed)
    fold_of: dict[str, int] = {}
    for values in cells.values():
        rng.shuffle(values)
        for index, household in enumerate(values):
            fold_of[household] = index % count
    return fold_of


def _metrics(
    tables: Tables,
    spec: Spec,
    tasks: Sequence[ChoiceTask],
    beta: Sequence[float],
) -> dict[str, float]:
    logloss = 0.0
    correct = 0
    for task in tasks:
        rows = [feature_vector(tables, spec, option, task.stratum) for option in task.options]
        probabilities = _probabilities(beta, rows)
        logloss -= math.log(max(probabilities[task.choice_index], 1e-15))
        if max(range(len(probabilities)), key=lambda i: probabilities[i]) == task.choice_index:
            correct += 1
    return {
        "logloss": logloss / len(tasks),
        "accuracy": correct / len(tasks),
        "tasks": len(tasks),
    }


def _inner_ridge(
    tables: Tables,
    spec: Spec,
    tasks: Sequence[ChoiceTask],
    candidates: Sequence[float],
    seed: int,
) -> float:
    if len(candidates) == 1:
        return candidates[0]
    fold_of = _folds(tables, tasks, 3, seed)
    best = None
    for candidate in candidates:
        scores = []
        for fold in range(3):
            train = [task for task in tasks if fold_of[task.household] != fold]
            test = [task for task in tasks if fold_of[task.household] == fold]
            if not train or not test:
                continue
            result = fit(
                tables,
                spec,
                train,
                ridge=candidate,
                compute_covariance=False,
                max_iterations=60,
            )
            scores.append(_metrics(tables, spec, test, result.beta)["logloss"])
        score = sum(scores) / len(scores) if scores else float("inf")
        if best is None or score < best[0]:
            best = (score, candidate)
    return best[1]


RIDGE_GRID = (0.3, 1.0, 3.0, 10.0, 30.0)


def cross_validate(
    tables: Tables,
    spec: Spec,
    tasks: Sequence[ChoiceTask] | None = None,
    *,
    ridge_candidates: Sequence[float] = RIDGE_GRID,
    select_ridge: bool = True,
    seed: int = 42,
) -> dict[str, Any]:
    """三类留出：家庭、题组（展示版本）、每种方案留一题组。"""
    tasks = list(tasks if tasks is not None else build_tasks(tables))
    if not spec.segment_interactions:
        ridge_candidates = (0.0,)
    results: dict[str, Any] = {"household": [], "form": [], "plan": []}

    fold_of = _folds(tables, tasks, 5, seed)
    for fold in range(5):
        train = [t for t in tasks if fold_of[t.household] != fold]
        test = [t for t in tasks if fold_of[t.household] == fold]
        if not train or not test:
            continue
        ridge = (
            _inner_ridge(tables, spec, train, ridge_candidates, seed + fold)
            if select_ridge
            else spec.ridge
        )
        fitted = fit(tables, spec, train, ridge=ridge, compute_covariance=False)
        metrics = _metrics(tables, spec, test, fitted.beta)
        metrics.update({"fold": fold, "ridge": ridge, "train_tasks": len(train)})
        results["household"].append(metrics)

    forms = sorted({task.form for task in tasks})
    for form in forms:
        train = [t for t in tasks if t.form != form]
        test = [t for t in tasks if t.form == form]
        if not train or not test:
            continue
        ridge = (
            _inner_ridge(tables, spec, train, ridge_candidates, seed + len(form))
            if select_ridge
            else spec.ridge
        )
        fitted = fit(tables, spec, train, ridge=ridge, compute_covariance=False)
        metrics = _metrics(tables, spec, test, fitted.beta)
        metrics.update({"form": form, "ridge": ridge, "train_tasks": len(train)})
        results["form"].append(metrics)

    plan_forms: dict[str, list[str]] = {}
    for task in tasks:
        plan = task.form.split("-")[0]
        plan_forms.setdefault(plan, [])
        if task.form not in plan_forms[plan]:
            plan_forms[plan].append(task.form)
    held_out = {sorted(values)[1] for values in plan_forms.values() if len(values) > 1}
    train = [t for t in tasks if t.form not in held_out]
    test = [t for t in tasks if t.form in held_out]
    if train and test:
        ridge = (
            _inner_ridge(tables, spec, train, ridge_candidates, seed + 99)
            if select_ridge
            else spec.ridge
        )
        fitted = fit(tables, spec, train, ridge=ridge, compute_covariance=False)
        metrics = _metrics(tables, spec, test, fitted.beta)
        metrics.update({"held_out_forms": sorted(held_out), "ridge": ridge})
        results["plan"] = metrics

    def summarise(rows):
        if not rows:
            return None
        return {
            "mean_logloss": sum(r["logloss"] for r in rows) / len(rows),
            "mean_accuracy": sum(r["accuracy"] for r in rows) / len(rows),
            "folds": len(rows),
            "detail": rows,
        }

    return {
        "spec": spec.key,
        "label": spec.label,
        "household": summarise(results["household"]),
        "form": summarise(results["form"]),
        "plan": results["plan"] or None,
    }


def baseline_metrics(tables: Tables, tasks: Sequence[ChoiceTask] | None = None) -> dict[str, Any]:
    """两个不依赖偏好模型的基准：题内均匀、同题组历史频率。"""
    tasks = list(tasks if tasks is not None else build_tasks(tables))
    uniform_logloss = sum(math.log(len(t.options)) for t in tasks) / len(tasks)
    uniform_accuracy = sum(1 / len(t.options) for t in tasks) / len(tasks)
    by_form_household = _folds(tables, tasks, 5, 42)
    loglosses = []
    accuracies = []
    for fold in range(5):
        train = [t for t in tasks if by_form_household[t.household] != fold]
        test = [t for t in tasks if by_form_household[t.household] == fold]
        if not train or not test:
            continue
        ll = 0.0
        correct = 0
        for task in test:
            counts: dict[str, int] = {}
            for other in train:
                if other.form == task.form:
                    counts[other.choice] = counts.get(other.choice, 0) + 1
            if not counts:
                for other in train:
                    counts[other.choice] = counts.get(other.choice, 0) + 1
            total = sum(counts.values())
            probabilities = [counts.get(option, 0) / total for option in task.options]
            normaliser = sum(probabilities)
            if normaliser <= 0:
                probabilities = [1 / len(task.options)] * len(task.options)
            else:
                probabilities = [p / normaliser for p in probabilities]
            ll -= math.log(max(probabilities[task.choice_index], 1e-15))
            if max(range(len(probabilities)), key=lambda i: probabilities[i]) == task.choice_index:
                correct += 1
        loglosses.append(ll / len(test))
        accuracies.append(correct / len(test))
    return {
        "uniform": {"logloss": uniform_logloss, "accuracy": uniform_accuracy},
        "form_frequency": {
            "logloss": sum(loglosses) / len(loglosses),
            "accuracy": sum(accuracies) / len(accuracies),
        },
        "tasks": len(tasks),
    }


# ---------------------------------------------------------------------------
# 潜类稳健性
# ---------------------------------------------------------------------------


@dataclass
class LatentClassFit:
    classes: int
    class_probability: list[float]
    models: list[FitResult]
    loglik: float
    iterations: int
    seed: int

    def probabilities(self, tables: Tables, options: Sequence[str], stratum: str) -> dict[str, float]:
        total = {option: 0.0 for option in options}
        for weight, model in zip(self.class_probability, self.models):
            shares = predict(tables, model, options, stratum)
            for option in options:
                total[option] += weight * shares[option]
        return total


def fit_latent_class(
    tables: Tables,
    tasks: Sequence[ChoiceTask] | None = None,
    *,
    classes: int = 2,
    ridge: float = 1.0,
    seed: int = 7,
    em_iterations: int = 60,
    inner_iterations: int = 25,
) -> LatentClassFit:
    """K 类潜类 MNL；固定随机种子，报告收敛值而非挑选最优重启。"""
    tasks = list(tasks if tasks is not None else build_tasks(tables))
    spec = SPECS["A"]
    prepared = _task_rows(tables, spec, tasks)
    penalties = penalty_vector(spec, ridge)
    p = len(penalties)
    rng = random.Random(seed)
    betas = [
        [0.0 if k == 0 else rng.gauss(0.0, 0.2) for _ in range(p)] for k in range(classes)
    ]
    weights = [1.0 / classes] * classes
    loglik = float("-inf")
    iteration = 0
    for iteration in range(1, em_iterations + 1):
        responsibilities: list[list[float]] = []
        loglik = 0.0
        for rows, y, weight, _ in prepared:
            likelihoods = []
            for k in range(classes):
                probability = _probabilities(betas[k], rows)[y]
                likelihoods.append(max(weights[k] * probability, 1e-300))
            total = sum(likelihoods)
            responsibilities.append([value / total for value in likelihoods])
            loglik += weight * math.log(total)
        new_betas = []
        for k in range(classes):
            weighted = [
                (rows, y, weight * responsibilities[i][k], task)
                for i, (rows, y, weight, task) in enumerate(prepared)
            ]
            weighted = [row for row in weighted if row[2] > 1e-8]
            if not weighted:
                new_betas.append(betas[k])
                continue
            model = _fit_prepared(
                spec, weighted, penalties, max_iterations=inner_iterations, warm=betas[k]
            )
            new_betas.append(model)
        betas = new_betas
        class_totals = [sum(row[k] for row in responsibilities) for k in range(classes)]
        overall = sum(class_totals) or 1.0
        weights = [max(value / overall, 1e-6) for value in class_totals]
        normaliser = sum(weights)
        weights = [value / normaliser for value in weights]
    models = []
    for k in range(classes):
        models.append(
            FitResult(
                spec=spec,
                ridge=ridge,
                beta=betas[k],
                names=feature_names(spec),
                loglik=_weighted_loglik(betas[k], prepared, penalties),
                iterations=inner_iterations,
                converged=True,
                n_tasks=len(tasks),
                n_households=len({t.household for t in tasks}),
            )
        )
    return LatentClassFit(
        classes=classes,
        class_probability=weights,
        models=models,
        loglik=loglik,
        iterations=iteration,
        seed=seed,
    )


def _fit_prepared(spec, prepared, penalties, max_iterations, warm):
    p = len(penalties)
    beta = list(warm)
    for _ in range(max_iterations):
        gradient = _gradient(beta, prepared, penalties)
        hessian = _hessian(beta, prepared, penalties)
        for k in range(p):
            hessian[k][k] += 1e-9
        step = _solve(hessian, [-g for g in gradient])
        if step is None:
            break
        current = _weighted_loglik(beta, prepared, penalties)
        scale = 1.0
        accepted = False
        while scale > 1e-10:
            candidate = [b + scale * s for b, s in zip(beta, step)]
            if _weighted_loglik(candidate, prepared, penalties) >= current - 1e-12:
                accepted = True
                break
            scale *= 0.5
        if not accepted:
            break
        beta = candidate
        if max(abs(scale * s) for s in step) < 1e-8:
            break
    return beta
