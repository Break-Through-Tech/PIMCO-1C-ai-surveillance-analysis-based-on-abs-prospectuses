import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import prospectus_evaluation as evaluation
from prospectus_backend import hybrid_search as shared_hybrid_search
from prospectus_provenance import validate_manifest


def chunk(number, file="deal.htm"):
    return dict(chunk_id=f"{file}_{number}", file_name=file, deal_name="Deal",
                filing_date="2026-04-08", doc_url="https://www.sec.gov/example",
                text="Source passage")


class ProspectusEvaluationTests(unittest.TestCase):
    def test_true_sale_only_changes_filing_scoped_true_sale_queries(self):
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(["Payment date", "Sale pledge assign", "Prepayment"])
        ranks = [0, 2, 1]
        eligible = np.array([0, 1, 2])
        for question, file in [("How are collections handled?", "toyota.htm"),
                               ("What is a true sale?", None)]:
            self.assertEqual(evaluation.true_sale_order(
                question, file, ranks, vectorizer, matrix, eligible), ranks)
        self.assertEqual(evaluation.true_sale_order(
            "Is this a true-sale?", "toyota.htm", ranks, vectorizer, matrix, eligible), [1, 0, 2])

    def test_notebook_hybrid_returns_lexical_only_passages(self):
        texts = ["cash payment" for _ in range(51)] + ["sell and assign pledge"]
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(texts)
        chunks = [chunk(i) | {"text": item} for i, item in enumerate(texts)]
        vectors = np.ones((len(texts), 1))
        vectors[-1, 0] = 0  # Outside the top 50 semantic candidates.
        client = SimpleNamespace(embeddings=SimpleNamespace(
            create=lambda **kwargs: SimpleNamespace(data=[SimpleNamespace(embedding=[1])])
        ))
        env = dict(np=np, pd=pd, indexed_chunks=chunks, complete_files={"deal.htm"},
                   indexed_vectors=vectors, global_vectorizer=vectorizer,
                   global_tfidf=matrix, EMBEDDING_MODEL="mock",
                   require_api_client=lambda: client,
                   shared_hybrid_search=shared_hybrid_search)
        nb_path = Path(__file__).resolve().parents[1] / "notebooks" / "btt_ai_fall_studio_pimco_1c.ipynb"
        notebook = json.loads(nb_path.read_text())
        source = next("".join(cell["source"]) for cell in notebook["cells"]
                      if cell.get("id") == "394d87c8")
        exec(compile(source, str(nb_path), "exec"), env)
        result = env["hybrid_search"]("Would this be a true sale?", file_name="deal.htm", top_k=1)
        self.assertEqual(result.iloc[0]["chunk_id"], chunks[-1]["chunk_id"])
        self.assertEqual(result.iloc[0]["hybrid_score"], 0.0)

    def test_precision_requires_five_valid_human_labels(self):
        rows = [dict(question_id="q18", rank=i, chunk_id=str(i), relevance_label=label)
                for i, label in enumerate(("2", "0", "1", "2", "2"), 1)]
        self.assertEqual(evaluation.reviewed_precision(rows), {"q18": .6})
        self.assertEqual(evaluation.reviewed_precision(rows[:4]), {})
        rows[1]["relevance_label"] = ""
        self.assertEqual(evaluation.reviewed_precision(rows), {})

    def test_coverage_metrics_and_label_keys(self):
        self.assertEqual(evaluation._metrics([], set(), set(), str), (None, None))
        self.assertEqual(evaluation._metrics([], {"X"}, set(), str), (0, 0))
        row = {"question_id": "q18", "chunk_id": "C1", "relevance_label": ""}
        keys = ("question_id", "chunk_id")
        labels = ("relevance_label",)
        entry = row | {"relevance_label": "2"}
        self.assertEqual(evaluation.merge_review_labels([row], [entry], keys, labels), [entry])
        self.assertEqual(row["relevance_label"], "")
        with self.assertRaises(ValueError):
            evaluation.merge_review_labels([row], [entry | {"chunk_id": "stale"}], keys, labels)

    def test_answer_cache_identity_and_abstention_plan(self):
        metrics = [{"question_id": "q19", "question": "Which deals concern student loans?"},
                   {"question_id": "q03", "question": "Which deals concern motorcycles?"}]
        plans = evaluation.answer_plan(metrics, {}, [chunk(1)], "index", "model", "prompt", "system")
        self.assertEqual(set(plans), {"q19"})
        with tempfile.TemporaryDirectory() as temp:
            calls = []
            def generate(plan):
                calls.append(plan["cache_id"])
                return {"question": plan["question"], "model": plan["model"],
                        "prompt_version": plan["prompt_version"],
                        "answer": "The supplied evidence is insufficient.", "sources": []}
            self.assertEqual(evaluation.evaluate_answers(plans, metrics, temp, generate)[0]["status"],
                             "awaiting_generation")
            with self.assertRaises(ValueError):
                evaluation.evaluate_answers(plans, metrics, temp, generate, question_ids=("q19",))
            evaluation.evaluate_answers(plans, metrics, temp, generate, question_ids=("q19",), max_calls=1)
            self.assertEqual(evaluation.evaluate_answers(plans, metrics, temp, generate)[0]["status"],
                             "ready_for_review")
            self.assertEqual(len(calls), 1)
            changed = evaluation.answer_plan(metrics, {}, [chunk(1)], "different index", "model", "prompt", "system")
            self.assertEqual(evaluation.evaluate_answers(changed, metrics, temp, generate)[0]["status"],
                             "awaiting_generation")

    def test_immutable_exports_and_smoke_request_cap(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "record.json"
            evaluation.write_once(target, "first")
            evaluation.write_once(target, "first")
            with self.assertRaises(RuntimeError):
                evaluation.write_once(target, "second")
            self.assertEqual(target.read_text(), "first")
            calls = []
            scopes = [("One", "deal.htm"), ("All", None)]
            def retrieve(question, file):
                calls.append(("embed", file))
                return [chunk(1)]
            def generate(question, file, evidence):
                calls.append(("chat", file))
                source = {k: v for k, v in evidence[0].items() if k not in {"source_ref", "text"}}
                return {"answer": "Evidence [S1]", "sources": [source | {"reference": "S1"}]}
            with self.assertRaises(PermissionError):
                evaluation.smoke_test(temp, "index", scopes, retrieve, generate)
            self.assertEqual(calls, [])
            outputs = evaluation.smoke_test(temp, "index", scopes, retrieve, generate, approved=True)
            self.assertEqual([kind for kind, _ in calls], ["embed", "chat", "embed", "chat"])
            self.assertTrue(all(json.loads(path.read_text())["citation_checks"]["all_sources_match"]
                                for path in outputs))

    def test_manifest_collision_requires_evidence(self):
        rows = pd.DataFrame([
            dict(file_name="filing.htm", primary_doc=primary, accession=accession, doc_url=url)
            for primary, accession, url in [("one.htm", "one", "url1"), ("two.htm", "two", "url2")]
        ])
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp) / "filing.htm"
            file.write_text("There is no recognizable primary document here.")
            selected, collisions = validate_manifest(rows, temp)
            self.assertTrue(selected.empty)
            self.assertEqual(collisions[0]["resolution"], "unresolved")
            file.write_text("<title>two.htm</title>")
            selected, _ = validate_manifest(rows, temp)
            self.assertEqual(selected["accession"].tolist(), ["two"])


if __name__ == "__main__":
    unittest.main()
