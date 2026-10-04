"""Lightweight unit tests for retrieval control flow; no models/downloads required."""
import sys
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# The retrieval methods tested here are pure Python. Stub heavy runtime packages
# so these regression tests can run before models and the Chroma index are set up.
_STUBS = {
    "sentence_transformers": {"CrossEncoder": type("CrossEncoder", (), {})},
    "chromadb": {},
    "langchain_huggingface": {"HuggingFaceEmbeddings": type("HuggingFaceEmbeddings", (), {})},
    "rank_bm25": {"BM25Okapi": type("BM25Okapi", (), {})},
    "ollama": {},
}
for _name, _attrs in _STUBS.items():
    if _name not in sys.modules:
        _module = types.ModuleType(_name)
        for _key, _value in _attrs.items():
            setattr(_module, _key, _value)
        sys.modules[_name] = _module

from rag_engine import LegalRAGEngine


class RetrievalLogicTests(unittest.TestCase):
    def setUp(self):
        self.engine = LegalRAGEngine.__new__(LegalRAGEngine)

    def test_article_14_reference_and_domain_routing_without_classifier(self):
        query = (
            "What does Article 14 of the Constitution of India say about "
            "equality before the law?"
        )

        self.assertEqual(
            self.engine.extract_legal_reference(query),
            ("article", "14"),
        )
        self.assertEqual(
            self.engine.classify_query_domain(query, [0.0]),
            "constitution",
        )

    def test_exact_article_14_ignores_other_sources_and_contents(self):
        self.engine.bm25_docs = [
            "14. Equality before law. Wrong statute decoy.",
            "CONTENTS\n14. Equality before law. Contents decoy.",
            (
                "14. Equality before law. The State shall not deny to any "
                "person equality before the law or the equal protection of "
                "the laws within the territory of India."
            ),
        ]
        self.engine.bm25_metadatas = [
            {
                "source": "other_statute.pdf",
                "page": 14,
                "citation": "Article 14",
            },
            {
                "source": "constitution_of_india.pdf",
                "page": 4,
                "citation": "Contents: Article 14",
            },
            {
                "source": "constitution_of_india.pdf",
                "page": 37,
                "citation": "Article 14",
            },
        ]

        hits = self.engine.exact_provision_search(
            "article", "14", "constitution"
        )

        self.assertEqual(len(hits), 1)
        self.assertEqual(
            hits[0]["metadata"]["source"], "constitution_of_india.pdf"
        )
        self.assertEqual(hits[0]["metadata"]["page"], 37)
        self.assertEqual(hits[0]["metadata"]["citation"], "Article 14")
        self.assertIn(
            "equality before the law",
            hits[0]["text"].lower(),
        )
        self.assertIn(
            "equal protection of the laws",
            hits[0]["text"].lower(),
        )

    def test_exact_article_14_selection_does_not_depend_on_fixture_order(self):
        valid_passage = (
            "14.\nEquality before law - The State shall not deny to any "
            "person equality before the law, or the equal protection of "
            "the laws within the territory of India."
        )
        misleading_passage = (
            "INDEX\n14. Equality before law. An index entry without the "
            "operative provision."
        )
        valid_metadata = {
            "source": "constitution_of_india.pdf",
            "page": 37,
            "citation": "Article 14",
        }
        misleading_metadata = {
            "source": "constitution_of_india.pdf",
            "page": 290,
            "citation": "Article 14",
        }

        for documents, metadatas in (
            (
                [misleading_passage, valid_passage],
                [misleading_metadata, valid_metadata],
            ),
            (
                [valid_passage, misleading_passage],
                [valid_metadata, misleading_metadata],
            ),
        ):
            with self.subTest(first_passage=documents[0]):
                self.engine.bm25_docs = documents
                self.engine.bm25_metadatas = metadatas

                hits = self.engine.exact_provision_search(
                    "article", "14", "constitution"
                )

                self.assertEqual(len(hits), 1)
                with self.subTest(first_passage=documents[0], field="source"):
                    self.assertEqual(
                        hits[0]["metadata"]["source"],
                        "constitution_of_india.pdf",
                    )
                with self.subTest(first_passage=documents[0], field="passage"):
                    self.assertEqual(hits[0]["text"], valid_passage)
                with self.subTest(first_passage=documents[0], field="page"):
                    self.assertEqual(hits[0]["metadata"]["page"], 37)
                with self.subTest(first_passage=documents[0], field="citation"):
                    self.assertEqual(
                        hits[0]["metadata"]["citation"], "Article 14"
                    )

    def test_exact_article_14_rejects_wrong_citation(self):
        self.engine.bm25_docs = [
            (
                "14. Equality before law. The State shall not deny to any "
                "person equality before the law or the equal protection of "
                "the laws."
            )
        ]
        self.engine.bm25_metadatas = [
            {
                "source": "constitution_of_india.pdf",
                "page": 37,
                "citation": "Article 15",
            }
        ]

        self.assertEqual(
            self.engine.exact_provision_search(
                "article", "14", "constitution"
            ),
            [],
        )

    def test_exact_article_14_accepts_page_citation_and_missing_or_blank_citation(self):
        valid_passage = (
            "14. Equality before law. The State shall not deny to any "
            "person equality before the law or the equal protection of "
            "the laws."
        )

        for metadata in (
            {
                "source": "constitution_of_india.pdf",
                "page": 37,
                "citation": "Page 37",
            },
            {"source": "constitution_of_india.pdf", "page": 37},
            {
                "source": "constitution_of_india.pdf",
                "page": 37,
                "citation": "   ",
            },
        ):
            with self.subTest(metadata=metadata):
                self.engine.bm25_docs = [valid_passage]
                self.engine.bm25_metadatas = [metadata]

                hits = self.engine.exact_provision_search(
                    "article", "14", "constitution"
                )

                self.assertEqual(len(hits), 1)
                self.assertEqual(hits[0]["metadata"], metadata)

    def test_exact_article_14_accepts_explicit_matching_citation(self):
        valid_passage = (
            "14. Equality before law. The State shall not deny to any "
            "person equality before the law or the equal protection of "
            "the laws."
        )
        metadata = {
            "source": "constitution_of_india.pdf",
            "page": 37,
            "citation": "Article 14",
        }
        self.engine.bm25_docs = [valid_passage]
        self.engine.bm25_metadatas = [metadata]

        hits = self.engine.exact_provision_search(
            "article", "14", "constitution"
        )

        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["metadata"], metadata)

    def test_exact_article_14_rejects_wrong_heading(self):
        self.engine.bm25_docs = [
            (
                "14. Index note. The State shall not deny to any person "
                "equality before the law or the equal protection of the laws."
            )
        ]
        self.engine.bm25_metadatas = [
            {
                "source": "constitution_of_india.pdf",
                "page": 37,
                "citation": "Article 14",
            }
        ]

        self.assertEqual(
            self.engine.exact_provision_search(
                "article", "14", "constitution"
            ),
            [],
        )

    def test_exact_article_14_returns_no_hit_for_ambiguous_or_unverified_candidates(self):
        valid_passage = (
            "14. Equality before law. The State shall not deny to any "
            "person equality before the law or the equal protection of "
            "the laws."
        )
        valid_metadata = {
            "source": "constitution_of_india.pdf",
            "page": 37,
            "citation": "Article 14",
        }

        self.engine.bm25_docs = [valid_passage, valid_passage]
        self.engine.bm25_metadatas = [valid_metadata, dict(valid_metadata)]
        self.assertEqual(
            self.engine.exact_provision_search(
                "article", "14", "constitution"
            ),
            [],
        )

        self.engine.bm25_docs = [
            (
                "14. Index note. No equality before the law entry in the "
                "contents of the laws."
            )
        ]
        self.engine.bm25_metadatas = [valid_metadata]
        self.assertEqual(
            self.engine.exact_provision_search(
                "article", "14", "constitution"
            ),
            [],
        )

    def test_exact_article_skips_contents_and_stops_at_next_article(self):
        self.engine.bm25_docs = [
            "CONTENTS\n12. Definition.\n13. Laws inconsistent with rights.\n",
            "12. In this Part, unless the context otherwise requires, the State includes the Government and Parliament of India.\n"
            "13. Laws inconsistent with or in derogation of the fundamental rights.\n",
            "12. United Nations Organisation.\n",
        ]
        self.engine.bm25_metadatas = [
            {"source": "constitution_of_india.pdf", "page": 5},
            {"source": "constitution_of_india.pdf", "page": 10},
            {"source": "constitution_of_india.pdf", "page": 300},
        ]
        hits = self.engine.exact_provision_search("article", "12", "constitution")
        self.assertEqual(len(hits), 1)
        self.assertTrue(hits[0]["text"].startswith("12. In this Part"))
        self.assertNotIn("13. Laws", hits[0]["text"])

    def test_exact_section_supports_subsection_opening_and_skips_contents(self):
        self.engine.bm25_docs = [
            "TABLE OF CONTENTS\n1. Short title.\n2. Definitions.\n",
            "1. (1) This Act may be called the Consumer Protection Act, 2019.\n"
            "(2) It extends to the whole of India.\n",
        ]
        self.engine.bm25_metadatas = [
            {"source": "consumer_protection_act_2019.pdf", "page": 1},
            {"source": "consumer_protection_act_2019.pdf", "page": 2},
        ]
        hits = self.engine.exact_provision_search("section", "1", "consumer_protection")
        self.assertEqual(len(hits), 1)
        self.assertTrue(hits[0]["text"].startswith("1. (1) This Act"))

    def test_sparse_search_preserves_metadata_for_duplicate_text(self):
        class FakeBM25:
            def get_scores(self, tokens):
                return [1.0, 3.0, 2.0]

        self.engine.bm25 = FakeBM25()
        self.engine.bm25_docs = ["same chunk", "same chunk", "other chunk"]
        self.engine.bm25_metadatas = [{"page": 1}, {"page": 2}, {"page": 3}]
        hits = self.engine.sparse_search("legal question", k=3)
        self.assertEqual([h["metadata"]["page"] for h in hits], [2, 3, 1])

    def test_explicit_section_routes_to_indexed_consumer_act(self):
        self.engine.query_classifier = type(
            "Classifier", (), {"predict": lambda self, values: ["out_of_domain"]}
        )()
        self.assertEqual(
            self.engine.classify_query_domain("What does Section 2 say?", [0.0]),
            "consumer_protection",
        )

    def test_overlap_metrics_do_not_mislabel_threshold_as_accuracy(self):
        answer_metrics = self.engine.compute_answer_metrics(
            "Article 12 defines the State.",
            [{"text": "Article 12 defines the State under the Constitution."}],
        )
        qa_metrics = self.engine.compute_qa_metrics(
            "Article 12 defines the State.", "Article 12 defines the State."
        )
        self.assertIn("overlap_threshold_pass", answer_metrics)
        self.assertIn("overlap_threshold_pass", qa_metrics)
        self.assertNotIn("accuracy", answer_metrics)
        self.assertNotIn("accuracy", qa_metrics)

    def test_exact_hit_bypasses_dense_distance_guardrail(self):
        self.engine.embedding_fn = type("Emb", (), {"embed_query": lambda self, query: [0.1]})()
        self.engine.classify_query_domain = lambda query, vector: "constitution"
        self.engine.extract_legal_reference = lambda query: ("article", "12")
        exact = {
            "text": "12. In this Part...",
            "metadata": {"source": "constitution_of_india.pdf", "page": 10},
            "exact_provision_match": True,
        }
        self.engine.exact_provision_search = lambda *args: [exact]
        self.engine.dense_search = lambda query, k: ([], 2.0)
        self.engine.sparse_search = lambda query, k: []
        self.engine.reciprocal_rank_fusion = lambda *args, **kwargs: []
        self.engine.rerank = lambda *args, **kwargs: []
        contexts, confident, distance, domain = self.engine.retrieve("What does Article 12 say?")
        self.assertTrue(confident)
        self.assertEqual(distance, 2.0)
        self.assertEqual(domain, "constitution")
        self.assertTrue(contexts[0]["exact_provision_match"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
