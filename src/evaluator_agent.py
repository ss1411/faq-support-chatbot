#!/usr/bin/env python3
"""
evaluator_agent.py

Evaluator agent for RAG responses.

Function:
    evaluate_response(user_question: str, system_answer: str, chunks_related: list) -> dict

The evaluator returns a score (0-10) and a structured reason explaining:
 - support_score (0-4): proportion of answer sentences supported by retrieved chunks
 - citation_score (0-2): whether the answer cites chunk IDs and if those IDs are present
 - completeness_score (0-3): how well the answer covers important keywords present in the retrieved chunks
 - clarity_score (0-1): basic fluency/clarity heuristic

It also returns a human-readable explanation and component breakdown.

Chunks_related is expected to be a list of dicts with:
    - "document": chunk text
    - "metadata": may contain "chunk_id"

"""

import re
from collections import Counter
from typing import List, Dict, Any
import json

# small stopword set for lightweight keyword extraction
STOPWORDS = {
    "the","and","is","in","to","of","a","for","with","that","on","as","by","an","be","are","this","it","or","from",
    "at","which","have","has","was","were","but","not","their","they","we","our","you","your","will","can","may","such"
}

def tokenize(text: str):
    """Simple tokenization."""
    words = re.findall(r"\b\w+\b", text.lower())
    return [w for w in words if len(w) >= 2]

def sentence_split(text: str):
    """Naive sentence split."""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]

def extract_top_keywords_from_chunks(chunks_related: List[Dict[str,Any]], top_k: int = 20):
    """Extract top keywords from retrieved chunks."""
    all_words = []
    for c in chunks_related:
        all_words.extend(tokenize(c.get("document", "")))
    filtered = [w for w in all_words if w not in STOPWORDS and len(w) > 3]
    counts = Counter(filtered)
    return [w for w,_ in counts.most_common(top_k)]

def sentence_supported_by_chunks(sentence: str, chunks_related: List[Dict[str,Any]], overlap_threshold: int = 3):
    """Check if a sentence is supported by chunks based on token overlap."""
    sent_tokens = [w for w in tokenize(sentence) if w not in STOPWORDS and len(w) > 3]
    if not sent_tokens:
        return False, 0

    chunk_words = set()
    for c in chunks_related:
        chunk_words.update([w for w in tokenize(c.get("document","")) if w not in STOPWORDS and len(w) > 3])

    match_count = sum(1 for w in set(sent_tokens) if w in chunk_words)
    return (match_count >= overlap_threshold, match_count)

def extract_cited_chunk_ids(answer_text: str):
    """Find chunk references such as [chunk_0001] or chunk_0001."""
    ids = set(re.findall(r"chunk[_\-]?\d{3,6}", answer_text.lower()))
    return list(ids)

def evaluate_response(user_question: str, system_answer: str, chunks_related: List[Dict[str,Any]]):
    """Main evaluator logic."""
    system_answer = system_answer or ""
    user_question = user_question or ""
    chunks_related = chunks_related or []

    # --- SUPPORT SCORE (0–4) ---
    sentences = sentence_split(system_answer)
    if not sentences:
        support_score = 0.0
        supported_ratio = 0.0
        match_counts = []
        support_matches = []
    else:
        support_matches = []
        match_counts = []
        for s in sentences:
            supported, match_count = sentence_supported_by_chunks(s, chunks_related, overlap_threshold=2)
            support_matches.append(supported)
            match_counts.append(match_count)

        supported_ratio = sum(1 for m in support_matches if m) / len(support_matches)
        support_score = round(supported_ratio * 4, 2)

    # --- CITATION SCORE (0–2) ---
    cited_ids = extract_cited_chunk_ids(system_answer)

    available_ids = set()
    for c in chunks_related:
        meta = c.get("metadata") or {}
        cid = meta.get("chunk_id") or meta.get("id") or ""
        if cid:
            available_ids.add(cid.lower())

    valid_cite_count = sum(1 for cid in cited_ids if cid in available_ids)

    if not cited_ids:
        citation_score = 0.0
    else:
        citation_score = min(2.0, 2.0 * (valid_cite_count / len(cited_ids)))

    # --- COMPLETENESS SCORE (0–3) ---
    top_keywords = extract_top_keywords_from_chunks(chunks_related, top_k=20)
    if not top_keywords:
        completeness_score = 0.0
        covered_kw_ratio = 0.0
    else:
        ans_tokens = set(tokenize(system_answer))
        covered = sum(1 for kw in top_keywords if kw in ans_tokens)
        covered_kw_ratio = covered / len(top_keywords)
        completeness_score = round(covered_kw_ratio * 3, 2)

    # --- CLARITY SCORE (0–1) ---
    avg_sent_len = 0.0
    if sentences:
        avg_sent_len = sum(len(tokenize(s)) for s in sentences) / len(sentences)

    if avg_sent_len == 0:
        clarity_score = 0.0
    elif 6 <= avg_sent_len <= 40:
        clarity_score = 1.0
    elif 3 <= avg_sent_len < 6 or 40 < avg_sent_len <= 60:
        clarity_score = 0.5
    else:
        clarity_score = 0.2

    # --- FINAL SCORE (0–10) ---
    raw_score = support_score + citation_score + completeness_score + clarity_score
    final_score = max(0.0, min(10.0, round(raw_score, 2)))

    explanation_parts = [
        f"Support: {support_score:.2f}/4 — {supported_ratio*100:.1f}% of sentences supported.",
        f"Citations: {citation_score:.2f}/2 — found {cited_ids}, valid: {valid_cite_count}.",
        f"Completeness: {completeness_score:.2f}/3 — keyword coverage: {covered_kw_ratio*100:.1f}%.",
        f"Clarity: {clarity_score:.2f}/1 — avg sentence length: {avg_sent_len:.1f} tokens."
    ]

    return {
        "final_score": final_score,
        "components": {
            "support_score": support_score,
            "citation_score": citation_score,
            "completeness_score": completeness_score,
            "clarity_score": clarity_score
        },
        "metadata": {
            "sentences": len(sentences),
            "supported_sentences": sum(1 for m in support_matches if m) if sentences else 0,
            "cited_ids": cited_ids,
            "valid_cited_ids": valid_cite_count,
            "available_chunk_ids": list(available_ids),
            "top_keywords": top_keywords
        },
        "explanation": " ".join(explanation_parts)
    }

# Demo if executed directly
if __name__ == "__main__":
    sample_question = "What are the onboarding steps?"
    sample_answer = "Onboarding includes Digital Offer Letter, Document Collection, IT provisioning, and orientation. See [chunk_0001]."
    sample_chunks = [
        {"document": "Digital Offer Letter and Acceptance. Document collection for ID proofs, tax forms.", "metadata": {"chunk_id": "chunk_0001"}},
        {"document": "IT provisioning includes email, laptop, and access permissions.", "metadata": {"chunk_id": "chunk_0002"}}
    ]
    print(json.dumps(
        evaluate_response(sample_question, sample_answer, sample_chunks),
        indent=2
    ))
