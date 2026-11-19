import json
import tempfile
import unittest
from pathlib import Path

import prospectus_review as review


class ReviewTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.row = dict(question_id="q01", question="Question", method="tfidf",
                        rank="1", chunk_id="c1", file_name="a.htm", text="Evidence")
        self.available = [self.row]

    def save_passage(self, label, expected):
        review.save(self.root, "passages", self.row, {"relevance_label": label}, expected,
                    lambda: self.available, {"a.htm"})

    def test_save_reload_conflict_revision_and_preservation(self):
        self.save_passage("2", None)
        identity = review.key("passages", self.row)
        before = review.inputs(self.root, "passages")[identity]
        self.assertEqual(before["relevance_label"], "2")
        with self.assertRaisesRegex(ValueError, "Another submission"):
            self.save_passage("0", None)
        self.save_passage("1", before)
        self.assertEqual(review.inputs(self.root, "passages")[identity]["relevance_label"], "1")
        receipts = [json.loads(file.read_text()) for file in (self.root / "revisions").glob("*.json")]
        self.assertEqual(len(receipts), 2)
        self.assertIn(before, [receipt["before"] for receipt in receipts])

    def test_evidence_change_is_rejected(self):
        self.available = [self.row | {"text": "Different evidence"}]
        with self.assertRaisesRegex(ValueError, "item changed"):
            self.save_passage("2", None)
        self.assertFalse((self.root / "passages.csv").exists())

    def test_filing_scope_validation(self):
        filing = dict(question_id="s06", question="Question", proposed_files="a.htm")
        for value in ("other.htm", ""):
            with self.assertRaises(ValueError):
                review.save(self.root, "filing_truth", filing,
                            dict(verified_relevant_files=value, complete_verified="yes"), None,
                            lambda: [filing], {"a.htm"})

    def test_answer_version_and_scoring_require_generated_answer(self):
        pending = dict(question_id="q19", case_id="exact", question="Question",
                       status="awaiting_generation", answer="", evidence_json="[]")
        with self.assertRaisesRegex(ValueError, "awaiting generation"):
            review.save(self.root, "answers", pending, {}, None, lambda: [pending], set())
        ready = pending | {"status": "ready_for_review", "answer": "Answer"}
        values = dict.fromkeys(review.FIELDS["answers"], "PASS")
        review.save(self.root, "answers", ready, values, None, lambda: [ready], set())
        with self.assertRaisesRegex(ValueError, "different evidence version"):
            review.check_input_versions(self.root, "answers", [ready | {"answer": "Other answer"}])

    def test_heldout_export_requires_approval(self):
        heldout = dict(question_id="h01", case_id="proposal", question="Question", evidence=["c1"])
        with self.assertRaisesRegex(ValueError, "need human approval"):
            review.approved_heldout(self.root, "v1", [heldout], {}, {})
        review.save(self.root / "human", "heldout", heldout, {"decision": "approved"}, None,
                    lambda: [heldout], {"a.htm"})
        path = review.approved_heldout(self.root, "v1", [heldout], {}, {"c1": {"file_name": "a.htm"}})
        self.assertEqual(json.loads(path.read_text())["cases"][0]["approved_candidate_chunks"], ["c1"])
