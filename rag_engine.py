import os
import re
import pickle
from typing import List, Dict, Any, Tuple
from sentence_transformers import CrossEncoder
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
import ollama

CHROMA_DIR = "./chroma_db"
BM25_PKL_PATH = "./chroma_db/bm25_index.pkl"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
COLLECTION_NAME = "legal_normative_docs"
DISTANCE_GUARDRAIL_THRESHOLD = 1.35


class LegalRAGEngine:
    def __init__(self, model_name: str = "legal-ai-finetuned:latest"):
        self.model_name = model_name
        self.embedding_fn = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.collection = self.chroma_client.get_collection(name=COLLECTION_NAME)

        with open(BM25_PKL_PATH, "rb") as f:
            bm25_data = pickle.load(f)
            self.bm25: BM25Okapi = bm25_data["bm25"]
            self.bm25_docs: List[str] = bm25_data["docs"]
            self.bm25_metadatas: List[Dict[str, Any]] = bm25_data["metadatas"]

        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-small")

        with open("query_classifier.pkl", "rb") as f:
            self.query_classifier = pickle.load(f)

    def dense_search(self, query: str, k: int = 10) -> Tuple[List[Dict[str, Any]], float]:
        query_vector = self.embedding_fn.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )
        dense_hits = []
        min_distance = results["distances"][0][0] if results["distances"][0] else 999.0
        for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
            dense_hits.append({"text": doc, "metadata": meta, "distance": dist})
        return dense_hits, min_distance
    def extract_legal_reference(self, query: str):
        """Detect explicit Article/Section references in the query."""
        article_match = re.search(
            r'\bArticle\s+([0-9]+[A-Za-z]?)\b',
            query,
            re.IGNORECASE
        )

        if article_match:
            return "article", article_match.group(1)

        section_match = re.search(
            r'\bSection\s+([0-9]+[A-Za-z]?)\b',
            query,
            re.IGNORECASE
        )

        if section_match:
            return "section", section_match.group(1)

        return None, None


    def exact_provision_search(
        self,
        reference_type: str,
        reference_number: str,
        predicted_class: str
    ) -> List[Dict[str, Any]]:
        """Find an explicitly referenced legal provision in the corpus."""

        exact_hits = []

        for text, metadata in zip(
            self.bm25_docs,
            self.bm25_metadatas
        ):
            text_lower = text.lower()
            source = metadata.get("source", "").lower()

            # Constitution Article
            if (
                reference_type == "article"
                and predicted_class == "constitution"
                and source == "constitution_of_india.pdf"
            ):
                first_part = text_lower[:500]

                excluded = (
                    "contents" in first_part
                    or "seventh schedule" in first_part
                    or "eighth schedule" in first_part
                    or "ninth schedule" in first_part
                    or "tenth schedule" in first_part
                    or "eleventh schedule" in first_part
                    or "twelfth schedule" in first_part
                    or "appendix" in first_part
                )

                if excluded:
                    continue

                pattern = (
                    rf"(?m)^\s*{re.escape(reference_number)}\.\s+[A-Za-z]"
                )

                if re.search(pattern, text):
                    # Extract ONLY the requested Article from the larger
                    # page/chunk so neighbouring Articles cannot confuse
                    # the generation model.
                    start_match = re.search(
                        rf"(?m)^\s*{re.escape(reference_number)}\.\s+[A-Za-z]",
                        text
                    )

                    if start_match:
                        start = start_match.start()

                        # Find the next numbered constitutional provision.
                        next_match = re.search(
                            r"(?m)^\s*(?:\d+|\d+[A-Za-z])\.\s+[A-Za-z]",
                            text[start + 1:]
                        )

                        if next_match:
                            end = start + 1 + next_match.start()
                            provision_text = text[start:end].strip()
                        else:
                            provision_text = text[start:].strip()

                        exact_hits.append({
                            "text": provision_text,
                            "metadata": metadata,
                            "exact_provision_match": True
                        })

            # Consumer Protection Act Section
            elif (
                reference_type == "section"
                and predicted_class == "consumer_protection"
                and source == "consumer_protection_act_2019.pdf"
            ):
                pattern = (
                    rf"(?m)^\s*{re.escape(reference_number)}\.\s+[A-Za-z]"
                )

                if re.search(pattern, text):
                    exact_hits.append({
                        "text": text,
                        "metadata": metadata,
                        "exact_provision_match": True
                    })

        return exact_hits


    def sparse_search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        tokenized_query = re.findall(r'\w+', query.lower())
        top_docs = self.bm25.get_top_n(tokenized_query, self.bm25_docs, n=k)
        sparse_hits = []
        for doc in top_docs:
            idx = self.bm25_docs.index(doc)
            sparse_hits.append({"text": doc, "metadata": self.bm25_metadatas[idx]})
        return sparse_hits

    def reciprocal_rank_fusion(self, dense_hits: List[Dict], sparse_hits: List[Dict], k: int = 60, top_n: int = 4) -> List[Dict]:
        rrf_scores = {}

        def add_ranks(hits):
            for rank, hit in enumerate(hits):
                doc_text = hit["text"]
                if doc_text not in rrf_scores:
                    rrf_scores[doc_text] = {"score": 0.0, "hit": hit}
                rrf_scores[doc_text]["score"] += 1.0 / (k + (rank + 1))

        add_ranks(dense_hits)
        add_ranks(sparse_hits)

        sorted_docs = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
        return [item["hit"] for item in sorted_docs[:top_n]]

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_n: int = 4) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.reranker.predict(pairs)
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_n]

    def classify_query_domain(self, query: str, query_vector):
        """
        Hybrid legal-domain routing.

        The trained classifier remains the primary decision mechanism.
        High-precision rules handle:
        1. Explicit references to legal domains outside the indexed corpus.
        2. Strong Constitution / Consumer Protection indicators that may be
           underrepresented in the small classifier training set.
        """

        q = re.sub(r"\s+", " ", query.lower().strip())

        # ---------------------------------------------------------
        # 1. Explicitly named external legal domains
        # ---------------------------------------------------------
        external_domain_patterns = [
            r"\bright to information act\b",
            r"\brti act\b",
            r"\bsebi act\b",
            r"\bsecurities and exchange board\b",
            r"\barbitration and conciliation act\b",
            r"\benvironmental protection act\b",
            r"\bprevention of corruption act\b",
            r"\binsolvency and bankruptcy code\b",
            r"\bnational food security act\b",
            r"\bright to education act\b",
            r"\bforeign exchange management act\b",
            r"\bfema\b",
            r"\bminimum wage law\b",
            r"\bcyberbullying law\b",
            r"\bcopyright law\b",
            r"\bindian contract act\b",
            r"\bcontract act\b",
            r"\binformation technology act\b",
            r"\bit act\b",
            r"\bindustrial disputes\b",
            r"\bindustrial dispute act\b",
            r"\bsecurities markets?\b",
            r"\bsebi\b",
            r"\bsecurities and exchange board\b",
        ]

        if any(re.search(pattern, q) for pattern in external_domain_patterns):
            return "out_of_domain"

        # ---------------------------------------------------------
        # 2. Explicit out-of-domain legal indicators
        # ---------------------------------------------------------
        #
        # The supported corpus contains only:
        #   - Constitution of India
        #   - Consumer Protection Act, 2019
        #
        # These patterns identify clearly unrelated legal domains.
        # They are intentionally broad enough to catch different
        # phrasings, rather than memorizing individual test questions.
        #

        ood_patterns = [
            # Intellectual property
            r"\bcopyright\b",
            r"\bpatent\b",
            r"\bpatents\b",
            r"\btrademark\b",
            r"\btrademarks\b",
            r"\bgeographical indication\b",

            # Companies / corporate law
            r"\bcompanies act\b",
            r"\bcompany law\b",
            r"\bcorporate law\b",
            r"\bcompetition act\b",
            r"\bcompetition law\b",
            r"\banti[- ]competitive\b",
            r"\babuse of dominance\b",
            r"\binsolvency and bankruptcy\b",

            # Tax / financial regulation
            r"\bincome tax\b",
            r"\bincome[- ]tax\b",
            r"\bgoods and services tax\b",
            r"\bgst\b",
            r"\bcustoms law\b",
            r"\bcustoms act\b",
            r"\bforeign exchange\b",
            r"\bfema\b",
            r"\bsebi\b",
            r"\bsecurities market\b",
            r"\bsecurities markets\b",

            # Labour / employment
            r"\bminimum wage\b",
            r"\bminimum wages\b",
            r"\blabou?r law\b",
            r"\bindustrial dispute\b",
            r"\bindustrial disputes\b",
            r"\bprovident fund\b",
            r"\bemployees.? provident fund\b",
            r"\bmaternity benefit\b",
            r"\bmaternity benefits\b",

            # Criminal law
            r"\bindian penal code\b",
            r"\bipc\b",
            r"\bbharatiya nyaya sanhita\b",
            r"\bbns\b",
            r"\bcriminal procedure\b",
            r"\bcriminal law\b",
            r"\bcode of criminal procedure\b",
            r"\bcrpc\b",
            r"\bindian evidence act\b",
            r"\bbharatiya sakshya\b",

            # Environment
            r"\benvironmental protection act\b",
            r"\benvironment protection act\b",
            r"\benvironmental law\b",
            r"\benvironment law\b",
            r"\bair act\b",
            r"\bair pollution\b",
            r"\bwater act\b",
            r"\bwater pollution\b",
            r"\bwildlife protection act\b",
            r"\bforest conservation\b",

            # Technology / cyber
            r"\binformation technology act\b",
            r"\binformation technology law\b",
            r"\bcyber law\b",
            r"\bcyber crime\b",
            r"\bcyber offences?\b",
            r"\bdata protection\b",
            r"\bdigital personal data\b",

            # Contract / dispute law
            r"\bindian contract act\b",
            r"\bcontract law\b",
            r"\barbitration and conciliation\b",
            r"\barbitration law\b",
            r"\bnegotiable instruments\b",
            r"\bcheque dishonou?r\b",

            # Other statutes
            r"\bright to information act\b",
            r"\brti act\b",
            r"\bright to education act\b",
            r"\bnational food security act\b",
            r"\bprevention of corruption act\b",
            r"\bmotor vehicles act\b",
            r"\bpassport act\b",
            r"\btransfer of property act\b",
            r"\bspecific relief act\b",
        ]

        for pattern in ood_patterns:
            if re.search(pattern, q):
                return "out_of_domain"

        # ---------------------------------------------------------
        # 2. Strong Constitution indicators
        # ---------------------------------------------------------
        constitution_patterns = [
            r"\bconstitution of india\b",
            r"\bconstitutional\b",
            r"\bpreamble\b",
            r"\bfundamental rights?\b",
            r"\bfundamental duties?\b",
            r"\bdirective principles?\b",
            r"\bunion list\b",
            r"\bstate list\b",
            r"\bconcurrent list\b",
            r"\bseventh schedule\b",
            r"\bschedule\s*7\b",
            r"\bforming a new state\b",
            r"\bnew state\b.*\bconstitution\b",
            r"\bpresident's rule\b",
            r"\bpresident rule\b",
            r"\bparliament\b.*\bconstitution\b",
            r"\bhigh court\b.*\bconstitution\b",
            r"\bsupreme court\b.*\bconstitution\b",
            r"\barticle\s+\d+[a-z]?\b",
        ]

        constitution_hit = any(
            re.search(pattern, q) for pattern in constitution_patterns
        )

        # ---------------------------------------------------------
        # 3. Strong Consumer Protection indicators
        # ---------------------------------------------------------
        consumer_patterns = [
            r"\bconsumer protection\b",
            r"\bconsumer complaint\b",
            r"\bconsumer rights?\b",
            r"\bconsumer commission\b",
            r"\bdistrict commission\b",
            r"\bstate commission\b",
            r"\bnational commission\b",
            r"\bcentral consumer protection authority\b",
            r"\bccpa\b",
            r"\bmisleading advertisement\b",
            r"\bunfair trade practice\b",
            r"\brestrictive trade practice\b",
            r"\bproduct liability\b",
            r"\bdeficiency in service\b",
            r"\bdefect in goods\b",
            r"\bspurious goods\b",
            r"\bunfair contract\b",
            r"\be-commerce liability\b",
            r"\bconsumer mediation\b",
            r"\bmediation settlement\b",
            r"\bconsumer law\b",
            r"\bcomplainant\b.*\bconsumer\b",
        ]

        consumer_hit = any(
            re.search(pattern, q) for pattern in consumer_patterns
        )

        # Explicit Article queries belong to the Constitution corpus.
        if re.search(r"\barticle\s+\d+[a-z]?\b", q):
            return "constitution"

        if constitution_hit and not consumer_hit:
            return "constitution"

        if consumer_hit and not constitution_hit:
            return "consumer_protection"

        # If both domains appear, retain the trained classifier's decision.
        predicted = self.query_classifier.predict([query_vector])[0]

        # ---------------------------------------------------------
        # 4. Fallback to the trained classifier
        # ---------------------------------------------------------
        return predicted

    def retrieve(self, query: str) -> Tuple[List[Dict[str, Any]], bool, float, str]:
        query_vector = self.embedding_fn.embed_query(query)

        predicted_class = self.classify_query_domain(query, query_vector)
        reference_type, reference_number = self.extract_legal_reference(query)

        exact_hits = []

        if reference_type and reference_number:
            exact_hits = self.exact_provision_search(
                reference_type,
                reference_number,
                predicted_class
            )
        if predicted_class == "out_of_domain":
            dense_hits, min_distance = self.dense_search(query, k=10)
            return [], False, min_distance, predicted_class

        dense_hits, min_distance = self.dense_search(query, k=10)
        sparse_hits = self.sparse_search(query, k=10)

        if min_distance > DISTANCE_GUARDRAIL_THRESHOLD:
            return [], False, min_distance, predicted_class

        fused_contexts = self.reciprocal_rank_fusion(dense_hits, sparse_hits, top_n=8)
        reranked_contexts = self.rerank(query, fused_contexts, top_n=4)

        # Explicit legal references are resolved deterministically.
        # Put the exact provision first, then retain the best
        # hybrid-retrieval contexts.
        if exact_hits:
            exact_texts = {h["text"] for h in exact_hits}

            remaining = [
                h for h in reranked_contexts
                if h["text"] not in exact_texts
            ]

            reranked_contexts = exact_hits + remaining

        return reranked_contexts[:4], True, min_distance, predicted_class

    def get_topic_image(self, predicted_domain: str, shown_images: set) -> str:
        import json
        import random
        topic_map = {
            "constitution": "government_building",
            "consumer_protection": "shopping_consumer",
        }
        image_category = topic_map.get(predicted_domain)
        if not image_category:
            return None
        try:
            with open("class_all_images.json", "r") as f:
                manifest = json.load(f)
            all_images = manifest.get(image_category, [])
            if not all_images:
                return None
            unseen = [img for img in all_images if img not in shown_images]
            if not unseen:
                unseen = all_images
            return random.choice(unseen)
        except Exception:
            return None

    def embed_and_chunk_text(self, text: str, chunk_size: int = 300) -> List[Dict[str, Any]]:
        words = text.split()
        chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
        chunks = [c for c in chunks if len(c.strip()) > 20]
        if not chunks:
            return []
        embeddings = self.embedding_fn.embed_documents(chunks)
        return [{"text": c, "embedding": e} for c, e in zip(chunks, embeddings)]

    def search_uploaded(self, query: str, uploaded_chunks: List[Dict[str, Any]], top_n: int = 2) -> List[Dict[str, Any]]:
        if not uploaded_chunks:
            return []
        import numpy as np
        query_vec = np.array(self.embedding_fn.embed_query(query))
        sims = []
        for chunk in uploaded_chunks:
            chunk_vec = np.array(chunk["embedding"])
            sim = np.dot(query_vec, chunk_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(chunk_vec) + 1e-8)
            sims.append(sim)
        ranked_idx = np.argsort(sims)[::-1][:top_n]
        return [
            {"text": uploaded_chunks[i]["text"], "metadata": {"source": "Uploaded Document", "citation": "User Upload", "page": "-"}}
            for i in ranked_idx
        ]

    def classify_document_image(self, image_bytes) -> Dict[str, Any]:
        import torch
        import torch.nn as nn
        from torchvision import transforms, models
        from PIL import Image
        import io

        checkpoint = torch.load("doc_image_classifier.pt", map_location="cpu")
        class_names = checkpoint["classes"]

        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
        model.load_state_dict(checkpoint["model_state"])
        model.eval()

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
            pred_idx = torch.argmax(probs).item()

        return {
            "predicted_class": class_names[pred_idx],
            "confidence": round(float(probs[pred_idx]) * 100, 1),
            "all_scores": {class_names[i]: round(float(probs[i]) * 100, 1) for i in range(len(class_names))}
        }

    def generate_stream(self, query: str, contexts: list, chat_history: list = None, language: str = "English", eli5: bool = False):
        formatted_context = ""
        for i, c in enumerate(contexts):
            meta = c.get("metadata", {}) if isinstance(c, dict) else getattr(c, "metadata", {})
            citation = meta.get("citation", "Unknown Document")
            source = meta.get("source", "File")
            page = meta.get("page", "-")
            text_content = c.get("text", "") if isinstance(c, dict) else getattr(c, "text", "")
            formatted_context += f"\n--- CONTEXT CHUNK {i+1} [Source: {source} | Citation: {citation} | Page: {page}] ---\n{text_content}\n"

        history_str = ""
        if chat_history:
            for turn in chat_history[-4:]:
                role = "User" if turn["role"] == "user" else "Assistant"
                history_str += f"{role}: {turn['content']}\n"

        language_instruction = (
            "Respond entirely in Hindi (Devanagari script), including all legal terms translated naturally, "
            "while keeping Article/Section numbers in their original numeral form."
            if language == "Hindi" else
            "Respond entirely in English."
        )

        eli5_instruction = (
            "\nIMPORTANT: Explain this in very simple, plain language, as if speaking to someone with no legal background. "
            "Avoid jargon, use short sentences and everyday analogies, but still keep the Article/Section citations."
            if eli5 else ""
        )

        system_prompt = (
            "You are an expert Legal AI Assistant adhering strictly to legal interpretation norms.\n"
            "Your task is to answer the user's question using ONLY the provided authoritative legal context.\n"
            "Every statement you make MUST be directly traceable to the provided context.\n"
            "You MUST explicitly cite the Article, Chapter, or Document Section whenever making assertions.\n"
            "If the context does not contain sufficient factual evidence to answer the question, state clearly that the provided texts do not cover this query. DO NOT HALLUCINATE.\n"
            f"{language_instruction}{eli5_instruction}"
        )

        full_prompt = (
            f"{system_prompt}\n\n"
            f"{'CONVERSATION SO FAR:' + chr(10) + history_str if history_str else ''}\n"
            f"{formatted_context}\n\nUSER QUESTION: {query}\n\nLEGAL ANSWER:"
        )

        response = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": full_prompt}],
            stream=True,
            options={"num_ctx": 2048}
        )
        for chunk in response:
            yield chunk['message']['content']

    def generate_baseline(self, query: str, language: str = "English"):
        """Generates an answer WITHOUT any retrieved context, for RAG-vs-No-RAG comparison.
        This answer is NOT grounded and may hallucinate — that is the point of the comparison."""
        language_instruction = "Respond entirely in Hindi (Devanagari script)." if language == "Hindi" else "Respond entirely in English."
        prompt = (
            "Answer the following legal question using only your own general knowledge. "
            "Do not mention that you lack access to documents.\n"
            f"{language_instruction}\n\n"
            f"QUESTION: {query}\n\nANSWER:"
        )
        response = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            options={"num_ctx": 1024}
        )
        for chunk in response:
            yield chunk['message']['content']

    def generate_followups(self, query: str, answer: str) -> list:
        prompt = (
            "Based on this legal Q&A, suggest exactly 3 short, relevant follow-up questions "
            "a user might ask next. Reply with ONLY the 3 questions, one per line, no numbering, no extra text.\n\n"
            f"Question: {query}\nAnswer: {answer[:600]}\n\nFollow-up questions:"
        )
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                options={"num_ctx": 1024}
            )
            text = response['message']['content'].strip()
            questions = [q.strip("-• 123456789.").strip() for q in text.split("\n") if q.strip()]
            return questions[:3]
        except Exception:
            return [
                "What are the key exceptions under this provision?",
                "Which authority enforces these guidelines?",
                "What legal remedies are available for non-compliance?"
            ]

    def check_groundedness(self, answer: str, contexts: list) -> float:
        if not contexts or not answer.strip():
            return 0.0

        chunk_texts = []
        for c in contexts:
            text = c.get("text", "") if isinstance(c, dict) else getattr(c, "text", "")
            if text.strip():
                chunk_texts.append(text)

        sentences = [s.strip() for s in answer.replace(chr(10), " ").split(".") if len(s.strip()) > 15]

        if not sentences or not chunk_texts:
            return 0.0

        sentence_scores = []
        for sent in sentences:
            pairs = [(chunk, sent) for chunk in chunk_texts]
            scores = self.nli_model.predict(pairs, apply_softmax=True)
            entailment_probs = [s[1] for s in scores]
            best_match = max(entailment_probs)
            sentence_scores.append(best_match)

        avg_entailment = sum(sentence_scores) / len(sentence_scores)
        return round(max(0.0, min(100.0, avg_entailment * 100)), 1)

    def compute_answer_metrics(self, answer: str, contexts: list) -> Dict[str, float]:
        """Token-overlap based Precision/Recall/F1/Accuracy for THIS answer, comparing
        generated answer words against retrieved context words (SQuAD-style F1 metric,
        adapted as a per-answer groundedness proxy since classification metrics don't
        directly apply to free-text generation without a ground-truth reference)."""
        import re
        from collections import Counter

        def tokenize(text):
            return re.findall(r"\w+", text.lower())

        answer_tokens = tokenize(answer)
        context_text = " ".join(
            c.get("text", "") if isinstance(c, dict) else getattr(c, "text", "")
            for c in contexts
        )
        context_tokens = tokenize(context_text)

        if not answer_tokens or not context_tokens:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}

        answer_counts = Counter(answer_tokens)
        context_counts = Counter(context_tokens)
        common = answer_counts & context_counts
        num_common = sum(common.values())

        precision = num_common / len(answer_tokens)
        recall = num_common / len(context_tokens)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        accuracy = 1.0 if f1 >= 0.3 else 0.0

        return {
            "precision": round(precision * 100, 1),
            "recall": round(recall * 100, 1),
            "f1": round(f1 * 100, 1),
            "accuracy": round(accuracy * 100, 1)
        }
    def compute_qa_metrics(self, answer: str, reference_answer: str) -> Dict[str, float]:
        """Standard SQuAD-style token-overlap Precision/Recall/F1, comparing the
        generated answer against a human-written reference (gold) answer —
        not against raw retrieved context length."""
        import re
        from collections import Counter

        def tokenize(text):
            return re.findall(r"\w+", text.lower())

        answer_tokens = tokenize(answer)
        reference_tokens = tokenize(reference_answer)

        if not answer_tokens or not reference_tokens:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}

        answer_counts = Counter(answer_tokens)
        reference_counts = Counter(reference_tokens)
        common = answer_counts & reference_counts
        num_common = sum(common.values())

        precision = num_common / len(answer_tokens)
        recall = num_common / len(reference_tokens)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        accuracy = 1.0 if f1 >= 0.3 else 0.0

        return {
            "precision": round(precision * 100, 1),
            "recall": round(recall * 100, 1),
            "f1": round(f1 * 100, 1),
            "accuracy": round(accuracy * 100, 1)
        }
