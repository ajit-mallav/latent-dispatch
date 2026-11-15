"""No weights, tokenizers, GPU calls, or network. Run with unittest to avoid pytest's shared model fixtures."""
import copy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from latent_dispatch import prompts, text_controls as tc
from scripts.run_text_controls import check_saved_answers, pin_manifest


def question():
    # Deliberately BA: selecting gold must not silently reorder the facts into AB.
    return {"id": "example:BA:s0", "question": "Who is Ajit's friend's favorite artist?",
            "workers": [["Unrelated fact.", "Josh's favorite artist is Pat."],
                        ["Ajit's friend is Josh.", "Another unrelated fact."]],
            "gold": [{"hop": 0, "worker": 1, "index": 0}, {"hop": 1, "worker": 0, "index": 1}],
            "answers": ["Pat"], "source_id": "example", "dataset": "toy_d2",
            "qtype": "hop1", "seed": 0, "placement": "BA", "hops": 2, "crossings": 1, "bridges": ["Josh"]}


def report_row(q):
    return {"id": q["id"], "model": "Qwen/Qwen3-4B", "dtype": "float16", "report_tokens": 1024,
            "reports": ["Josh's favorite artist is Pat.", "Ajit's friend is Josh."],
            "worker_tokens": [9, 8], "truncated": [False, False], "question_sha256": tc.question_digest(q)}


class FakeModel:
    """Capture the prompt while exercising the shared answer function, without a real model."""
    fmt = prompts.FORMATS["qwen3"]

    def ids(self, text):
        return list(text)

    def prefill(self, ids):
        self.context = "".join(ids)
        return ids, None

    def greedy(self, ids, cache=None, pos=None, max_new=None, think_budget=None):
        self.tail = "".join(ids)
        text = "</think>\\boxed{Pat}" if think_budget is not None else "\\boxed{Pat}"
        return SimpleNamespace(text=text, tokens=8, think_tokens=2 if think_budget else 0,
                               cut=False, think_cut=False, think_eos=False)


class TextControlTests(unittest.TestCase):
    def test_cleaning_keeps_facts_and_records_exact_removed_spans(self):
        text = "Ajit's friend is Josh. The document does not provide the artist's name. Josh likes Pat."
        clean, edits = tc.strip_abstentions(text)
        self.assertEqual(clean, "Ajit's friend is Josh. Josh likes Pat.")
        self.assertEqual(len(edits), 1)
        self.assertEqual(text[edits[0]["start"]:edits[0]["end"]], edits[0]["text"])

    def test_cleaning_preserves_world_negation_and_mixed_clauses(self):
        for text in ("Josh is not the mayor.",
                     "The document does not name the mayor, but Josh lives in Kastrel.",
                     "The document does not name the mayor; Josh lives in Kastrel.",
                     "I cannot determine this because Josh's favorite artist is Pat."):
            self.assertEqual(tc.strip_abstentions(text), (text, []))

    def test_unchanged_report_is_byte_identical(self):
        text = "Facts:\n  Ajit's friend is Josh.\n\nJosh likes Pat.  "
        self.assertEqual(tc.strip_abstentions(text), (text, []))

    def test_empty_and_all_abstention_reports_are_retained_as_cases(self):
        self.assertEqual(tc.strip_abstentions(""), ("", []))
        cleaned, edits = tc.strip_abstentions("I cannot determine the answer.")
        self.assertEqual(cleaned, "")
        self.assertEqual(len(edits), 1)

    def test_gold_paragraphs_preserve_ba_and_exclude_distractors(self):
        q = question()
        self.assertEqual(tc.gold_paragraphs(q), [["Josh's favorite artist is Pat."], ["Ajit's friend is Josh."]])
        reports, ra = tc.evidence_blocks(q, ["", ""], "gold_reports")
        joint, ja = tc.evidence_blocks(q, ["", ""], "gold_joint")
        self.assertEqual(ra["gold_paragraphs"], ja["gold_paragraphs"])
        self.assertEqual(ra["gold_unit"], "paragraph")
        self.assertEqual(reports, [prompts.report_block(i, r) for i, r in enumerate(ra["presented_reports"])])
        self.assertEqual(joint, [prompts.block([p for w in ra["gold_paragraphs"] for p in w])])
        self.assertLess(joint[0].index("Josh's"), joint[0].index("Ajit's"))

    def test_colocated_gold_keeps_empty_worker_and_document_order(self):
        q = question()
        q["workers"] = [["Distractor."], ["Second hop.", "First hop."]]
        q["gold"] = [{"hop": 0, "worker": 1, "index": 1}, {"hop": 1, "worker": 1, "index": 0}]
        self.assertEqual(tc.gold_paragraphs(q), [[], ["Second hop.", "First hop."]])

    def test_invalid_evidence_and_report_shapes_fail(self):
        q = question()
        with self.assertRaises(ValueError):
            tc.evidence_blocks(q, ["only one worker"], "reports")
        with self.assertRaises(ValueError):
            tc.evidence_blocks(q, ["", ""], "unknown")
        q["gold"][0]["index"] = 99
        with self.assertRaises(ValueError):
            tc.gold_paragraphs(q)

    def test_report_identity_budget_and_question_are_checked(self):
        q, row = question(), report_row(question())
        tc.validate_report(row, q, row["model"], row["dtype"], 1024)
        for change in ({"dtype": "bfloat16"}, {"model": "Qwen/Qwen3-8B"}, {"report_tokens": 256},
                       {"id": "other"}, {"question_sha256": "wrong"}):
            with self.assertRaises(ValueError):
                tc.validate_report({**row, **change}, q, row["model"], row["dtype"], 1024)

    def test_original_t1_export_is_accepted(self):
        q, row = question(), report_row(question())
        row["options"] = {"report_tokens": row.pop("report_tokens"), "thinking": "forced"}
        row.pop("question_sha256")
        tc.validate_report(row, q, row["model"], row["dtype"], 1024)

    def test_unclosed_thinking_never_becomes_a_final_answer(self):
        self.assertFalse(tc.final_output_flags("I think \\boxed{Pat}", "forced")["final_answer_present"])
        self.assertFalse(tc.final_output_flags("</think>", "forced")["final_answer_present"])
        self.assertFalse(tc.final_output_flags("</think>\\boxed{Pat", "forced")["final_boxed"])
        self.assertTrue(tc.final_output_flags("</think>\\boxed{\\text{Pat}}", "forced")["final_boxed"])

    def test_original_report_context_and_alternate_prompt(self):
        q, row, model = question(), report_row(question()), FakeModel()
        before = copy.deepcopy(row)
        for condition in ("reports", "compose_prompt"):
            tc.evaluate(model, q, row, condition, thinking="off")
            self.assertEqual(model.context, prompts.prefix(model.fmt, q["question"]) +
                             "".join(prompts.report_block(i, r) for i, r in enumerate(row["reports"])))
            self.assertEqual(model.tail, prompts.tail(model.fmt, "off", tc.COMPOSE if condition == "compose_prompt" else ""))
        self.assertEqual(row, before)

    def test_joint_context_matches_existing_b1_layout(self):
        q, row, model = question(), report_row(question()), FakeModel()
        out = tc.evaluate(model, q, row, "joint", thinking="forced", think_budget=1024)
        self.assertEqual(model.context, prompts.prefix(model.fmt, q["question"]) +
                         "".join(prompts.block(w) for w in q["workers"] if w))
        self.assertTrue(out["final_answer_present"])
        self.assertEqual(out["input_tokens"], len(model.context) + len(model.tail))

    def test_resume_rejects_changed_or_lost_reports(self):
        q, bank = question(), {question()["id"]: report_row(question())}
        saved = {q["id"]: {"reports_sha256": tc.digest(bank[q["id"]]["reports"]),
                            "question_sha256": tc.question_digest(q)}}
        check_saved_answers(saved, bank, [q])
        with self.assertRaises(ValueError):
            check_saved_answers(saved, {}, [q])
        bank[q["id"]]["reports"][0] = "Different report."
        with self.assertRaises(ValueError):
            check_saved_answers(saved, bank, [q])

    def test_manifest_allows_only_matching_resume(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            pin_manifest(path, {"question_ids": ["one"]}, "commit1")
            pin_manifest(path, {"question_ids": ["one"]}, "commit2")
            with self.assertRaises(ValueError):
                pin_manifest(path, {"question_ids": ["two"]}, "commit2")


if __name__ == "__main__":
    unittest.main()
