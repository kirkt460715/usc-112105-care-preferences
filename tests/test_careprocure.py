import json
import tempfile
import unittest
from pathlib import Path

from careprocure.comparability import (
    coverage_gaps,
    display_comparability,
    evidence_cells,
    evidence_summary,
)
from careprocure.demand import (
    demand_by_segment,
    diversion,
    extrapolation_report,
    prepare_form_menus,
    scenario_gain_by_form,
    ward_choice_sets,
)
from careprocure.locks import (
    Ledger,
    lock_basket,
    lock_summary,
    revise,
)
from careprocure.model import SPECS, build_tasks, fit, predict
from careprocure.procurement import (
    Assumptions,
    Basket,
    basket_slots_by_ward,
    benefit_interval,
    capacity_state,
    check_basket,
    conflict_pairs,
    evaluate_basket,
    rank_baskets,
    reserve_plan,
)
from careprocure.states import (
    CHOICE_ALTERNATIVE,
    CHOICE_STATUSQUO,
    MISSING,
    NOT_DELIVERED,
    SKIPPED,
    audit_counts,
    exposure_matrix,
    resolve_answers,
)
from careprocure.tables import MaterialError, read_tables

DATA = Path(__file__).resolve().parents[1] / "data"


class Fixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tables = read_tables(DATA)
        cls.assumptions = Assumptions(
            as_of="2026-10-25",
            spec_key="B",
            ridge=SPECS["B"].ridge,
            budget_cents=515000,
            reserve_per_gap_cents=25000,
            model_label=SPECS["B"].label,
        )
        cls.fit_b = fit(cls.tables, SPECS["B"])
        cls.capacity = capacity_state(cls.tables, cls.assumptions.as_of)


class RecordStateTests(Fixture):
    def test_undelivered_displays_are_not_choices(self):
        answers = resolve_answers(self.tables, self.assumptions.as_of)
        not_delivered = [a for a in answers if a.state == NOT_DELIVERED]
        self.assertEqual(len(not_delivered), 14)
        self.assertTrue(all(a.choice is None for a in not_delivered))
        self.assertTrue(all(a.delivery == "not_delivered" for a in not_delivered))

    def test_statusquo_skip_and_missing_are_distinct(self):
        answers = resolve_answers(self.tables, self.assumptions.as_of)
        counts = audit_counts(answers)
        self.assertEqual(counts["chose_statusquo"], 55)
        self.assertEqual(counts["skipped"], 33)
        self.assertEqual(counts["missing"], 0)
        self.assertEqual(counts["not_delivered"], 14)
        self.assertEqual(
            counts["choice_tasks"],
            counts["chose_statusquo"] + counts["chose_alternative"],
        )
        self.assertTrue(any(a.state == CHOICE_STATUSQUO for a in answers))
        self.assertTrue(any(a.state == SKIPPED for a in answers))

    def test_correction_uses_latest_registration_and_keeps_history(self):
        answers = resolve_answers(self.tables, self.assumptions.as_of)
        corrected = [a for a in answers if a.superseded]
        self.assertEqual(len(corrected), 1)
        answer = corrected[0]
        self.assertEqual(answer.display, "F006-2")
        self.assertEqual(answer.choice, "statusquo")
        self.assertEqual(answer.seen_on, "2026-09-30")
        self.assertEqual(answer.superseded[0]["choice"], "trip")
        audit = audit_counts(answers)
        self.assertEqual(audit["corrected_displays"], 1)
        # 旧答卷只以历史形式保留，最新答卷进入似然
        tasks = build_tasks(self.tables, self.assumptions.as_of)
        self.assertNotIn("a-F006-2", {task.display for task in tasks})
        task = [task for task in tasks if task.display == "F006-2"][0]
        self.assertEqual(task.choice, "statusquo")

    def test_as_of_before_correction_uses_old_answer(self):
        answers = resolve_answers(self.tables, "2026-09-25")
        corrected = [a for a in answers if a.display == "F006-2"][0]
        self.assertEqual(corrected.choice, "trip")
        self.assertFalse(corrected.superseded)

    def test_exposure_separates_not_offered_from_not_delivered(self):
        matrix = exposure_matrix(self.tables, self.assumptions.as_of)
        # ridge 问卷里从未出现 nightA/nightB
        for household, cells in matrix.items():
            if household == "F000":
                self.assertEqual(cells["nightA"]["state"], "not_offered")
                self.assertEqual(cells["nightA"]["seen"], 0)
        # F027 的 remote-2 漏发，但该户还从别的题组看到过 visitB
        self.assertIn(matrix["F027"]["visitB"]["state"], {"seen_not_chosen", "chosen"})
        # F057 的 remote-1 漏发且没有其它题组包含 visitA? remote-2/3 不含 visitA
        self.assertEqual(matrix["F057"]["visitA"]["state"], "not_delivered_only")

    def test_delivered_displays_all_have_answers(self):
        answers = resolve_answers(self.tables, self.assumptions.as_of)
        self.assertFalse(any(a.state == MISSING for a in answers))


class ComparabilityTests(Fixture):
    def test_plan_option_sets_and_bridges(self):
        payload = display_comparability(self.tables, self.assumptions.as_of)
        self.assertFalse(payload["plans"]["remote"]["has_night"])
        self.assertTrue(payload["plans"]["early"]["has_night"])
        bridges = {
            (row["left"], row["right"]): row["shared_options"]
            for row in payload["plan_bridges"]
        }
        self.assertIn("satA", bridges[("early", "later")])
        self.assertIn("nightA", bridges[("cross", "early")])

    def test_option_attributes_are_shared_across_display_versions(self):
        payload = display_comparability(self.tables, self.assumptions.as_of)
        signatures = payload["option_signatures"]
        # 同名选项只有一组属性；版本变化不改属性编号
        forms = payload["forms"]
        for form, info in forms.items():
            for option in info["options"]:
                self.assertIn(option, signatures)

    def test_ridge_has_no_local_evidence_for_night(self):
        gaps = {(gap["ward"], gap["option"]) for gap in coverage_gaps(self.tables)}
        self.assertIn(("ridge", "nightA"), gaps)
        self.assertIn(("ridge", "nightB"), gaps)

    def test_evidence_threshold_marks_small_cells(self):
        cells = {
            (c.ward, c.stratum, c.option): c
            for c in evidence_cells(self.tables, self.assumptions.as_of)
        }
        # harbor 夜班全职有一定的有效题组
        cell = cells[("harbor", "nightshift", "nightB")]
        self.assertFalse(cell.comparable)
        self.assertEqual(cell.evidence_min, 12)
        summary = evidence_summary(self.tables, self.assumptions.as_of)
        self.assertIn("ridge", summary["by_ward"])


class ModelTests(Fixture):
    def test_fit_is_reproducible_and_reaches_optimum(self):
        again = fit(self.tables, SPECS["B"])
        self.assertTrue(again.converged)
        for left, right in zip(self.fit_b.beta, again.beta):
            self.assertAlmostEqual(left, right, places=9)
        self.assertLess(again.loglik, 0)

    def test_tasks_exclude_non_choices(self):
        tasks = build_tasks(self.tables, self.assumptions.as_of)
        self.assertEqual(len(tasks), 385)
        self.assertTrue(all(task.choice in task.options for task in tasks))

    def test_probabilities_sum_to_one(self):
        shares = predict(
            self.tables,
            self.fit_b,
            ("statusquo", "nightA", "visitB"),
            "nightshift",
        )
        self.assertAlmostEqual(sum(shares.values()), 1.0, places=12)

    def test_unseen_option_is_extrapolation_only(self):
        # 模型能给 ridge 的 nightB 打分，但 demand 层必须标成外推
        report = extrapolation_report(self.tables)
        self.assertTrue(
            any(item["ward"] == "ridge" and item["option"] == "nightB" for item in report)
        )
        demand = demand_by_segment(self.tables, self.fit_b, draws=20)
        self.assertIn("nightB", demand["ridge"]["not_shown_purchasable"])
        self.assertNotIn("nightB", demand["ridge"]["options"])


class DemandTests(Fixture):
    def test_form_menus_do_not_double_count_households(self):
        menus = prepare_form_menus(
            self.tables, self.fit_b, "harbor", self.assumptions.as_of
        )
        total = sum(menu["population_estimate"] for menu in menus)
        expected = sum(
            1.0 / row["sampling_probability"]
            for row in self.tables.sample
            if row["ward"] == "harbor"
        )
        self.assertAlmostEqual(total, expected, places=9)

    def test_shares_include_statusquo_and_sum_to_one(self):
        demand = demand_by_segment(self.tables, self.fit_b, draws=20)
        for ward, info in demand.items():
            self.assertAlmostEqual(sum(info["overall_shares"].values()), 1.0, places=9)
            self.assertIn("statusquo", info["overall_shares"])

    def test_ridge_intervals_cover_point_estimate(self):
        demand = demand_by_segment(self.tables, self.fit_b, draws=60)
        intervals = demand["ridge"]["intervals"]["options"]
        for option, bounds in intervals.items():
            point = demand["ridge"]["overall_shares"][option]
            self.assertLessEqual(bounds["low"], point + 1e-9)
            self.assertGreaterEqual(bounds["high"], point - 1e-9)

    def test_diversion_uses_only_locally_shown_options(self):
        diversion_map = diversion(self.tables, self.fit_b, "ridge", "short")
        self.assertNotIn("nightB", diversion_map["flows"])
        self.assertAlmostEqual(
            sum(value["diversion_ratio"] for value in diversion_map["flows"].values()),
            1.0,
            places=6,
        )

    def test_ward_choice_sets(self):
        choice_sets = ward_choice_sets(self.tables, self.assumptions.as_of)
        self.assertNotIn("nightA", choice_sets["ridge"].options)
        self.assertIn("nightA", choice_sets["water"].options)


class ProcurementTests(Fixture):
    def test_tied_offers_must_move_together(self):
        broken = Basket({"o1": 1, "o2": 2})
        check = check_basket(self.tables, broken, self.assumptions, self.capacity)
        self.assertFalse(check["feasible"])
        self.assertIn("ties", check["blocking"])
        paired = Basket({"o1": 2, "o2": 2})
        check = check_basket(self.tables, paired, self.assumptions, self.capacity)
        self.assertNotIn("ties", check["blocking"])

    def test_capacity_uses_latest_visible_reduction(self):
        early = capacity_state(self.tables, "2026-09-25")
        late = capacity_state(self.tables, "2026-10-25")
        self.assertEqual(early.units[("clinic", "night")], 14)
        self.assertEqual(late.units[("clinic", "night")], 10)

    def test_promises_are_deducted_before_new_purchases(self):
        self.assertEqual(self.capacity.promised[("clinic", "night")], 3)
        # 夜班剩 7 单位；两包 o3 已超
        check = check_basket(
            self.tables, Basket({"o3": 2}), self.assumptions, self.capacity
        )
        self.assertIn("provider_capacity", check["blocking"])

    def test_tie_group_is_blocked_when_one_member_lacks_evidence(self):
        reserve = reserve_plan(self.tables, self.assumptions)
        groups = {item["tie"]: item for item in reserve["blocked_tie_groups"]}
        self.assertIn("paired", groups)
        self.assertEqual(groups["paired"]["blocked_by"], ["o1"])
        self.assertEqual(groups["paired"]["collateral"], ["o2"])

    def test_missing_evidence_is_blocked_or_reserved(self):
        basket = Basket({"o1": 1, "o2": 1})
        check = check_basket(self.tables, basket, self.assumptions, self.capacity)
        self.assertIn("evidence", check["blocking"])
        reserve = reserve_plan(self.tables, self.assumptions)
        self.assertGreaterEqual(reserve["gap_count"], 2)
        self.assertEqual(
            reserve["purchasable_budget_cents"],
            self.assumptions.budget_cents - reserve["reserve_cents"],
        )

    def test_minimum_families_per_ward(self):
        check = check_basket(
            self.tables, Basket({"o3": 1}), self.assumptions, self.capacity
        )
        self.assertIn("minimum_families", check["blocking"])

    def test_conflict_pair_between_two_wards(self):
        conflicts = conflict_pairs(self.tables, self.assumptions, self.capacity)
        pairs = {(row["left"], row["right"]) for row in conflicts}
        self.assertIn(("o3", "o4"), pairs)
        conflict = [row for row in conflicts if (row["left"], row["right"]) == ("o3", "o4")][0]
        self.assertIn("provider_capacity", conflict["blocking"])
        self.assertTrue(conflict["at_risk_promises"])

    def test_ranking_excludes_dominated_and_unseen(self):
        ranking = rank_baskets(
            self.tables, self.fit_b, self.assumptions, self.capacity, interval_draws=30
        )
        self.assertGreater(ranking["candidate_count"], 0)
        for item in ranking["top"]:
            self.assertFalse(item["dominated"])
            self.assertTrue(item["check"]["feasible"])
            self.assertNotIn("o1", item["basket"])
            self.assertNotIn("o8", item["basket"])

    def test_each_purchased_option_has_local_evidence(self):
        ranking = rank_baskets(
            self.tables, self.fit_b, self.assumptions, self.capacity, limit=1
        )
        best = ranking["best"]
        for ward, info in best["wards"].items():
            for option in info["provided"]:
                self.assertTrue(info["evidence_by_option"][option]["comparable_strata"])

    def test_benefit_interval_brackets_point_estimate(self):
        basket = Basket({"o3": 1, "o5": 2, "o6": 1, "o7": 1})
        evaluation = evaluate_basket(
            self.tables, self.fit_b, basket, self.assumptions, self.capacity
        )
        interval = benefit_interval(
            self.tables,
            self.fit_b,
            basket,
            self.assumptions,
            self.capacity,
            draws=40,
        )
        self.assertLessEqual(interval["benefit"]["low"], interval["benefit"]["high"])
        self.assertGreaterEqual(interval["benefit"]["high"], 0.0)
        self.assertGreater(evaluation["totals"]["served"], 0)

    def test_slots_by_ward(self):
        slots = basket_slots_by_ward(self.tables, Basket({"o3": 2, "o5": 1}))
        self.assertEqual(slots["harbor"]["nightA"], 18)
        self.assertEqual(slots["water"]["visitB"], 6)


class LockTests(Fixture):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.ledger = Ledger(Path(self.temp.name) / "ledger.jsonl")
        self.basket = Basket({"o3": 1, "o5": 2, "o6": 1, "o7": 1})

    def tearDown(self):
        self.temp.cleanup()

    def test_lock_records_cost_preferences_and_checks(self):
        record = lock_basket(
            self.ledger,
            self.tables,
            self.fit_b,
            self.basket,
            self.assumptions,
            self.capacity,
            label="初版",
            draws=20,
        )
        self.assertEqual(record["lock_id"], "L001")
        self.assertEqual(record["cost_cents"], 387000)
        self.assertEqual(record["model"]["spec"], "B")
        self.assertIn("coefficients", record["model"])
        self.assertIn("draw_ranges", record["model"])
        self.assertTrue(all(check["ok"] for check in record["checks"]))
        self.assertEqual(record["capacity_snapshot"]["clinic/night"]["cap_units"], 10)
        self.assertEqual(len(record["promises"]), 3)
        self.assertTrue(record["evidence_snapshot"]["by_ward"])

    def test_infeasible_basket_cannot_be_locked(self):
        with self.assertRaisesRegex(ValueError, "不能锁定"):
            lock_basket(
                self.ledger,
                self.tables,
                self.fit_b,
                Basket({"o3": 2}),
                self.assumptions,
                self.capacity,
            )

    def test_original_lock_is_not_overwritten_by_later_estimates(self):
        early = Assumptions(
            as_of="2026-09-25",
            spec_key="B",
            ridge=SPECS["B"].ridge,
            budget_cents=515000,
            reserve_per_gap_cents=25000,
        )
        old_capacity = capacity_state(self.tables, "2026-09-25")
        first = lock_basket(
            self.ledger,
            self.tables,
            self.fit_b,
            self.basket,
            early,
            old_capacity,
            label="九月容量",
            draws=20,
        )
        revision = revise(
            self.tables, self.fit_b, first, self.assumptions, self.ledger.entries()
        )
        self.assertEqual(revision.original["lock_id"], "L001")
        self.assertTrue(revision.alternatives)
        new_basket = Basket(revision.alternatives[0]["basket"])
        second = lock_basket(
            self.ledger,
            self.tables,
            self.fit_b,
            new_basket,
            self.assumptions,
            capacity_state(self.tables, "2026-10-25"),
            label="修订",
            supersedes="L001",
            reason="夜班容量减量",
            draws=20,
        )
        self.assertEqual(second["supersedes"], "L001")
        original = self.ledger.find("L001")
        self.assertEqual(original["cost_cents"], first["cost_cents"])
        self.assertEqual(original["assumptions"]["as_of"], "2026-09-25")
        self.assertEqual(original["capacity_snapshot"]["clinic/night"]["cap_units"], 14)
        self.assertEqual(len(self.ledger.history("L001")), 1)

    def test_revision_reports_shortfall_and_promises(self):
        early = Assumptions(
            as_of="2026-09-25",
            spec_key="B",
            ridge=SPECS["B"].ridge,
            budget_cents=515000,
            reserve_per_gap_cents=25000,
        )
        old_capacity = capacity_state(self.tables, "2026-09-25")
        # 九月容量下能把夜班两区一起买满（新容量下不行）
        basket = Basket({"o3": 1, "o4": 1, "o5": 1, "o6": 1, "o7": 1})
        first = lock_basket(
            self.ledger,
            self.tables,
            self.fit_b,
            basket,
            early,
            old_capacity,
            draws=20,
        )
        revision = revise(
            self.tables, self.fit_b, first, self.assumptions, self.ledger.entries()
        )
        self.assertTrue(revision.shortfalls)
        shortfall = revision.shortfalls[0]
        self.assertEqual((shortfall["provider"], shortfall["period"]), ("clinic", "night"))
        self.assertGreater(shortfall["shortfall"], 0)
        promised = {row["promise"] for row in revision.affected_households if "promise" in row}
        self.assertIn("g1", promised)
        self.assertTrue(revision.alternatives)
        self.assertTrue(all(alt["benefit"] >= 0 for alt in revision.alternatives))

    def test_ledger_detects_tampering(self):
        lock_basket(
            self.ledger,
            self.tables,
            self.fit_b,
            self.basket,
            self.assumptions,
            self.capacity,
            draws=20,
        )
        text = self.ledger.path.read_text(encoding="utf-8")
        self.ledger.path.write_text(text.replace('"o3":1', '"o3":2', 1), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "摘要"):
            self.ledger.entries()

    def test_summary_exposes_snapshot(self):
        record = lock_basket(
            self.ledger,
            self.tables,
            self.fit_b,
            self.basket,
            self.assumptions,
            self.capacity,
            draws=20,
        )
        summary = lock_summary(record)
        self.assertEqual(summary["basket"], self.basket.as_dict())
        self.assertEqual(summary["spec"], "B")
        self.assertTrue(summary["digest"])


class MaterialValidationTests(unittest.TestCase):
    def test_rejects_choice_outside_display(self):
        tables = read_tables(DATA)
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            for name in (
                "sample",
                "displays",
                "responses",
                "options",
                "offers",
                "provider_caps",
                "policy",
                "promises",
                "manifest",
            ):
                suffix = ".json" if name == "manifest" else ".jsonl"
                (directory / f"{name}{suffix}").write_bytes((DATA / f"{name}{suffix}").read_bytes())
            responses = [
                json.loads(line)
                for line in (directory / "responses.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            responses[0]["choice"] = "nightB"  # F000-1 未展示 nightB
            (directory / "responses.jsonl").write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in responses) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(MaterialError, "不存在"):
                read_tables(directory)

    def test_rejects_duplicate_answer_id(self):
        tables = read_tables(DATA)
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            for name in (
                "sample",
                "displays",
                "responses",
                "options",
                "offers",
                "provider_caps",
                "policy",
                "promises",
                "manifest",
            ):
                suffix = ".json" if name == "manifest" else ".jsonl"
                (directory / f"{name}{suffix}").write_bytes((DATA / f"{name}{suffix}").read_bytes())
            lines = (directory / "responses.jsonl").read_text(encoding="utf-8").splitlines()
            first = json.loads(lines[0])
            first["display"] = json.loads(lines[1])["display"]
            lines.append(json.dumps(first, ensure_ascii=False))
            (directory / "responses.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(MaterialError, "重复"):
                read_tables(directory)


if __name__ == "__main__":
    unittest.main()
