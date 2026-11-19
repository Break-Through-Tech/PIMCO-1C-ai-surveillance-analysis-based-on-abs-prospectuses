import json
import tempfile
import unittest
from pathlib import Path
from threading import Lock
from types import SimpleNamespace

import numpy as np

from prospectus_backend import (
    MissingCredentials, ProspectusBackend, QueryError, TOP_K,
    build_keyword_index, question_cache_path, sec_url,
)


class BackendTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        backend = self.backend = ProspectusBackend.__new__(ProspectusBackend)
        backend.root = backend.processed = backend.cache_dir = Path(folder.name)
        backend.lock = Lock()
        backend.index = SimpleNamespace(ntotal=3)
        backend.order_hash, backend.source_hash = "order", "archive"
        backend.complete_files = {"first.htm", "second.htm"}
        backend.chunks = [
            dict(chunk_id=f"c{i}", file_name=name, deal_name=name,
                 filing_date="2026-04-08", filing_year=2026,
                 doc_url=f"https://www.sec.gov/Archives/{name}", text="collections payment")
            for i, name in enumerate(("first.htm", "second.htm", "partial.htm"))
        ]
        backend.lookup = {chunk["chunk_id"]: chunk for chunk in backend.chunks}
        backend.vectors = np.array([[1, 0], [0, 1], [1, 0]], dtype="float32")
        backend.vectorizer, backend.matrix = build_keyword_index(backend.chunks)
        self.calls = []

        def embed(**kwargs):
            self.calls.append(("embedding", kwargs))
            return SimpleNamespace(data=[SimpleNamespace(embedding=[1, 0])])

        def complete(**kwargs):
            self.calls.append(("completion", kwargs))
            message = SimpleNamespace(content="The evidence supports this finding. [S1]")
            return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")], usage=None)

        backend._client = SimpleNamespace(
            embeddings=SimpleNamespace(create=embed),
            chat=SimpleNamespace(completions=SimpleNamespace(create=complete)),
        )

    def test_scopes_cache_and_session_limit(self):
        session = {}
        for question, file, expected in (
            ("First question", "first.htm", {"first.htm"}),
            ("All filings", None, self.backend.complete_files),
            ("Third question", "second.htm", {"second.htm"}),
        ):
            answer, origin = self.backend.analyze(question, file, True, session)
            self.assertEqual(origin, "Live analysis")
            self.assertEqual({source["file_name"] for source in answer["sources"]}, expected)
            self.assertEqual(answer["citations_used"], ["S1"])
            self.assertTrue(all(self.backend.resolve_source(source) for source in answer["sources"]))
            cached, origin = self.backend.analyze(question, file, False, session)
            self.assertEqual((cached, origin), (answer, "Cached answer"))
        self.assertEqual([kind for kind, _ in self.calls], ["embedding", "completion"] * 3)
        with self.assertRaisesRegex(QueryError, "three-query"):
            self.backend.analyze("One more", None, True, session)
        self.assertEqual(len(self.calls), 6)

    def test_early_validation_and_existing_cache(self):
        with self.assertRaisesRegex(QueryError, "Enter"):
            self.backend.analyze(" ", None, True, {})
        with self.assertRaisesRegex(QueryError, "fully indexed"):
            self.backend.analyze("question", "partial.htm", True, {})
        with self.assertRaisesRegex(QueryError, "no saved answer"):
            self.backend.analyze("question", None, False, {})
        path = question_cache_path("question", None, TOP_K, cache_dir=self.backend.cache_dir,
                                   order_hash="order", source_hash="archive", indexed_count=3)
        path.write_text(json.dumps({"answer": "Previously saved", "sources": []}))
        result, origin = self.backend.analyze("question", None, False, {})
        self.assertEqual((result["answer"], origin), ("Previously saved", "Cached answer"))
        self.assertEqual(self.calls, [])

    def test_no_results_failure_accounting_and_links(self):
        self.backend.search = lambda *args, **kwargs: __import__("pandas").DataFrame()
        answer, _ = self.backend.analyze("missing results", None, True, {})
        self.assertEqual(answer["sources"], [])
        self.assertEqual(self.calls, [])
        self.backend.require_client = lambda: (_ for _ in ()).throw(MissingCredentials())
        budget = {}
        with self.assertRaisesRegex(QueryError, "requires API credentials"):
            self.backend.analyze("missing credentials", None, True, budget)
        self.assertEqual(budget, {})
        self.assertIsNone(sec_url("https://www.sec.gov.example/Archives/filing"))
        self.assertIsNone(sec_url("javascript:alert(1)"))
        self.assertEqual(sec_url("https://www.sec.gov/Archives/example"), "https://www.sec.gov/Archives/example")
