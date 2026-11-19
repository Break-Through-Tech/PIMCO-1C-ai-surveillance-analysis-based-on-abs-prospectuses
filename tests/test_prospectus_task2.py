import tempfile
import unittest

from prospectus_task2 import aggregate, run_semantic_plan, score


class Task2Tests(unittest.TestCase):
    def test_scoring_requires_reviewed_labels_and_complete_filing_truth(self):
        cases = [dict(id="q1", file=None, abstention=False)]
        review = [dict(question_id="q1", method="tfidf", rank=n, primary=True,
                       file_name="first", filing_top5="first; second", chunk_id=str(n),
                       relevance_label=label)
                  for n, label in enumerate(["2", "1", "0", "2", ""], 1)]
        allowed = {"first", "second", "third"}
        self.assertIsNone(aggregate(score(cases, review, [], allowed))["precision_at_5"]["value"])
        review[-1]["relevance_label"] = "1"
        truth = [dict(question_id="q1", complete_verified="yes",
                      verified_relevant_files="second; third")]
        metrics = aggregate(score(cases, review, truth, allowed))
        self.assertEqual(metrics["precision_at_5"]["value"], 0.4)
        self.assertEqual(metrics["filing_hit_at_5"]["value"], 1)
        self.assertEqual(metrics["filing_recall_at_5"]["value"], 0.5)

    def test_live_semantic_requires_consent_and_reuses_exact_cache(self):
        cases = [dict(id="q1", file=None, question="Question?")]
        chunk = dict(chunk_id="c", file_name="filing", doc_url="url", text="text")
        calls = []

        def retrieve(question, file):
            calls.append(question)
            return [chunk]

        with tempfile.TemporaryDirectory() as directory:
            args = (cases, "index", directory, retrieve, {"c": chunk}, {"filing"})
            with self.assertRaises(PermissionError):
                run_semantic_plan(*args, question_ids=("q1",), max_calls=1)
            with self.assertRaises(ValueError):
                run_semantic_plan(*args, approved=True, question_ids=("q1",), max_calls=0)
            for _ in range(2):
                run_semantic_plan(*args, approved=True, question_ids=("q1",), max_calls=1)
            self.assertEqual(calls, ["Question?"])
