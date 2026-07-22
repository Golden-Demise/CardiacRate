"""
knowledge_retrieval.py

Lightweight TF-IDF retrieval over a small, curated patient-education
knowledge base (patient_education_knowledge.json).

This is the "retrieval" half of the system: case-specific facts are still
looked up directly by case_id (a deterministic lookup, not retrieval), but
general medical-term explanations are retrieved by matching the user's
question against this knowledge base with TF-IDF + cosine similarity.

Kept dependency-light on purpose (scikit-learn only, already used by the
ACDC Random Forest classifier) so it works fully offline with no model
download, which matters for a live demo.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class KnowledgeBase:
    def __init__(self, entries: List[Dict[str, Any]]):
        self.entries = entries or []
        self.vectorizer = None
        self.matrix = None
        if self.entries:
            corpus = [f"{e.get('topic', '')}. {e.get('text', '')}" for e in self.entries]
            self.vectorizer = TfidfVectorizer(stop_words="english")
            self.matrix = self.vectorizer.fit_transform(corpus)

    def retrieve(self, query: str, top_k: int = 2, min_score: float = 0.08) -> List[Dict[str, Any]]:
        """Return up to top_k knowledge entries relevant to query.

        Returns an empty list when nothing in the knowledge base is
        relevant enough (score below min_score) rather than forcing an
        unrelated snippet into the prompt.
        """
        if not self.entries or not query or not query.strip():
            return []

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix)[0]
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results = []
        for i in ranked_idx[:top_k]:
            if scores[i] >= min_score:
                entry = dict(self.entries[i])
                entry["score"] = float(scores[i])
                results.append(entry)
        return results


def load_knowledge_base(path: str) -> KnowledgeBase:
    p = Path(path)
    if not p.exists():
        print(f"[knowledge_retrieval] WARNING: knowledge base not found at {p}, "
              f"running without general-knowledge retrieval.")
        return KnowledgeBase([])
    with p.open("r", encoding="utf-8") as f:
        entries = json.load(f)
    print(f"[knowledge_retrieval] Loaded {len(entries)} knowledge entries from {p}")
    return KnowledgeBase(entries)


def format_knowledge_block(entries: List[Dict[str, Any]]) -> str:
    """Render retrieved entries into a compact block for the LLM prompt.

    Returns "" when entries is empty so callers can omit the section
    entirely rather than showing an empty GENERAL_KNOWLEDGE block.
    """
    if not entries:
        return ""
    lines = []
    for e in entries:
        source = e.get("source_title") or "general medical reference"
        lines.append(f"- {e.get('topic', '')}: {e.get('text', '')} (source: {source})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Standalone test/demo entry point.
#
# This lets you sanity-check retrieval quality without loading the LLM or
# touching the GPU. Two ways to use it:
#
#   python knowledge_retrieval.py
#       -> runs a fixed set of sample questions and prints top matches
#
#   python knowledge_retrieval.py --query "What does aortic valve calcification mean?"
#       -> runs retrieval for your own question
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test TF-IDF retrieval over the patient education knowledge base.")
    parser.add_argument("--knowledge_json", default="patient_education_knowledge.json",
                         help="Path to the knowledge base JSON (default: patient_education_knowledge.json in the current directory)")
    parser.add_argument("--query", default=None, help="A single question to test. If omitted, runs sample questions instead.")
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--min_score", type=float, default=0.08)
    args = parser.parse_args()

    kb = load_knowledge_base(args.knowledge_json)

    if args.query:
        queries = [args.query]
    else:
        queries = [
            "What does aortic valve calcification mean?",
            "Do I need an echocardiogram?",
            "What is the Agatston score?",
            "What is my blood pressure today?",
            "Is this result serious?",
            "What can this system not tell me?",
        ]

    for q in queries:
        hits = kb.retrieve(q, top_k=args.top_k, min_score=args.min_score)
        print(f"\nQ: {q}")
        if not hits:
            print("  -> no relevant knowledge entry (GENERAL_KNOWLEDGE block would be empty)")
        for h in hits:
            print(f"  -> [{h['score']:.3f}] {h['topic']}")
        block = format_knowledge_block(hits)
        if block:
            print("  --- rendered block ---")
            for line in block.splitlines():
                print("  " + line)
