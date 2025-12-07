# services/scoring.py
from __future__ import annotations

from typing import List, Dict, Any, Tuple
from extensions import sentence_model, kw_model, logger
from sklearn.metrics.pairwise import cosine_similarity

# ====== CONFIGURABLE THRESHOLDS ====== #
# These assume a final score in [0, 1]
COSINE_ONLY_QUALIFIED_THRESHOLD = 0.6
COMBINED_FULL_QUALIFIED_THRESHOLD = 0.7
COMBINED_PARTIAL_QUALIFIED_THRESHOLD = 0.5

# Weights for the combined score
COSINE_WEIGHT = 0.6
KEYWORD_WEIGHT = 0.25
LENGTH_WEIGHT = 0.15


def _normalize_weights() -> Tuple[float, float, float]:
    """
    Ensure COSINE_WEIGHT + KEYWORD_WEIGHT + LENGTH_WEIGHT = 1.0 internally.
    """
    total = COSINE_WEIGHT + KEYWORD_WEIGHT + LENGTH_WEIGHT
    if total == 0:
        return 1.0, 0.0, 0.0
    return COSINE_WEIGHT / total, KEYWORD_WEIGHT / total, LENGTH_WEIGHT / total


def _encode(text: str):
    """
    Encodes text using the global sentence_model.
    Raises RuntimeError if model not loaded.
    """
    if not sentence_model:
        raise RuntimeError("SentenceTransformer model not loaded.")
    # SentenceTransformer usually expects a list of strings
    return sentence_model.encode([text])[0]


def _safe_cosine_score(q_emb, a_emb) -> float:
    """
    Compute cosine similarity and map it to [0, 1].
    SentenceTransformers can return [-1, 1]; we normalize to [0, 1]
    so our thresholds are more intuitive.
    """
    raw = cosine_similarity([q_emb], [a_emb])[0][0]
    # map [-1, 1] -> [0, 1]
    normalized = (raw + 1.0) / 2.0
    # clamp for safety
    return float(max(0.0, min(1.0, normalized)))


def _length_score(answer: str) -> float:
    """
    Very simple heuristic score based on answer length.
    Short answers are penalized because they typically lack detail.
    Returns a value in [0, 1].
    """
    words = len(answer.split())
    if words == 0:
        return 0.0
    if words < 5:
        return 0.1
    if words < 15:
        return 0.4
    if words < 40:
        return 0.8
    # long enough to be detailed; we don't reward too much extra length
    return 1.0


def explain_score(score: float) -> str:
    """
    Maps a numeric score to a short human-readable explanation.
    """
    if score > 0.8:
        return "Excellent and highly relevant answer."
    if score > 0.6:
        return "Good answer with relevant content."
    if score > 0.4:
        return "Partially relevant. Add more detail and examples."
    return "Answer lacks relevance. Try to address the question more directly."


def compute_answer_score(question: str, answer: str) -> float:
    """
    Simple cosine similarity between question and answer (normalized to [0, 1]).
    Used by score_answer_single.
    """
    if not question.strip() or not answer.strip():
        return 0.0

    try:
        q_emb = _encode(question.strip())
        a_emb = _encode(answer.strip())
        score = _safe_cosine_score(q_emb, a_emb)
        return round(score, 2)
    except Exception as e:
        logger.error(f"❌ Error in compute_answer_score: {e}")
        return 0.0


def _extract_keywords(text: str, top_n: int = 5) -> List[str]:
    """
    Extracts keywords from a text using kw_model, if available.
    Returns a list of lowercase keyword strings.
    """
    if not kw_model:
        return []

    try:
        kw_pairs = kw_model.extract_keywords(text, top_n=top_n)
        # kw_pairs is usually like [('keyword', score), ...]
        return [k[0].lower() for k in kw_pairs if k and k[0]]
    except Exception as e:
        logger.error(f"❌ Error extracting keywords: {e}")
        return []


def _compute_keyword_overlap(
    question: str, answer: str, top_n: int = 5
) -> Tuple[List[str], List[str], float]:
    """
    Given question and answer, returns:
    - matched_keywords: list[str]
    - total_keywords: list[str]
    - keyword_score: float in [0, 1]

    Uses substring matching so multi-word keywords (e.g. "project management")
    can still be counted if they appear in the answer.
    """
    keywords = _extract_keywords(question, top_n=top_n)
    if not keywords:
        return [], [], 0.0

    answer_text = answer.lower()
    matched = [kw for kw in keywords if kw in answer_text]

    keyword_score = len(matched) / len(keywords) if keywords else 0.0
    return matched, keywords, float(keyword_score)


def score_answer_single(question: str, answer: str) -> Dict[str, Any]:
    """
    Scores an answer for a single question using cosine similarity only.
    Suitable for quick, per-question feedback.
    """
    question = question.strip()
    answer = answer.strip()

    score = compute_answer_score(question, answer)
    feedback = explain_score(score)
    status = (
        "Qualified"
        if score >= COSINE_ONLY_QUALIFIED_THRESHOLD
        else "Not Qualified"
    )

    return {
        "score": score,
        "qualification_status": status,
        "feedback": feedback,
    }


def score_answer_combined(question: str, answer: str) -> Dict[str, Any]:
    """
    Combined scoring:
    - cosine similarity between question and answer
    - keyword overlap between question keywords and answer
    - answer length (penalize very short answers)

    Returns a dict with:
    - score (0–1)
    - qualification_status
    - matched_keywords
    - total_keywords
    - cosine_score
    - keyword_score
    - length_score
    """
    question = question.strip()
    answer = answer.strip()

    if not question or not answer:
        return {
            "score": 0.0,
            "qualification_status": "Not Qualified",
            "matched_keywords": [],
            "total_keywords": [],
            "cosine_score": 0.0,
            "keyword_score": 0.0,
            "length_score": 0.0,
        }

    try:
        # cosine similarity
        q_emb = _encode(question)
        a_emb = _encode(answer)
        cosine_score = _safe_cosine_score(q_emb, a_emb)

        # keyword overlap
        matched, keywords, keyword_score = _compute_keyword_overlap(question, answer)

        # length factor
        length_score = _length_score(answer)

        w_cos, w_kw, w_len = _normalize_weights()
        raw_final = (cosine_score * w_cos) + (keyword_score * w_kw) + (length_score * w_len)
        final_score = round(float(max(0.0, min(1.0, raw_final))), 2)

        if final_score >= COMBINED_FULL_QUALIFIED_THRESHOLD:
            status = "Qualified"
        elif final_score >= COMBINED_PARTIAL_QUALIFIED_THRESHOLD:
            status = "Partially Qualified"
        else:
            status = "Not Qualified"

        return {
            "score": final_score,
            "qualification_status": status,
            "matched_keywords": matched,
            "total_keywords": keywords,
            "cosine_score": round(cosine_score, 2),
            "keyword_score": round(keyword_score, 2),
            "length_score": round(length_score, 2),
        }

    except Exception as e:
        logger.error(f"❌ Error in score_answer_combined: {e}")
        return {
            "score": 0.0,
            "qualification_status": "Error",
            "matched_keywords": [],
            "total_keywords": [],
            "cosine_score": 0.0,
            "keyword_score": 0.0,
            "length_score": 0.0,
        }


def score_many(qa_pairs: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Scores a list of {'question': str, 'answer': str} dictionaries.
    Returns:
    - average_score
    - qualification_status
    - answers: list of per-answer score dicts
    """
    total = 0.0
    results: List[Dict[str, Any]] = []

    for pair in qa_pairs:
        q = pair.get("question", "") or ""
        a = pair.get("answer", "") or ""
        res = score_answer_combined(q, a)
        results.append(res)
        total += res.get("score", 0.0)

    avg = round(total / len(results), 2) if results else 0.0

    if avg >= COMBINED_FULL_QUALIFIED_THRESHOLD:
        status = "Qualified"
    elif avg >= COMBINED_PARTIAL_QUALIFIED_THRESHOLD:
        status = "Partially Qualified"
    else:
        status = "Not Qualified"

    return {
        "average_score": avg,
        "qualification_status": status,
        "answers": results,
    }


def generate_detailed_advice(
    qa_results: List[Dict[str, Any]], avg_score: float
) -> str:
    """
    Generates a short, human-readable advice message based on:
    - overall average score
    - weakest answer inside qa_results
    - breakdown of cosine / keyword / length scores (when available)
    """
    if not qa_results:
        return "No advice available."

    # Overall performance message
    if avg_score >= COMBINED_FULL_QUALIFIED_THRESHOLD:
        base = "You performed very well overall. Your answers were relevant and detailed."
    elif avg_score >= COMBINED_PARTIAL_QUALIFIED_THRESHOLD:
        base = "Your interview performance is decent, but there is room to improve some answers."
    else:
        base = "Your answers suggest you should prepare more before the next interview."

    # Find lowest-scoring answer to tailor advice
    lowest = min(qa_results, key=lambda r: r.get("score", 0.0))
    lowest_score = lowest.get("score", 0.0)
    kw_score = lowest.get("keyword_score", 0.0)
    cos_score = lowest.get("cosine_score", 0.0)
    len_score = lowest.get("length_score", 0.0)

    # Heuristic, more specific advice depending on what seems weakest
    if lowest_score < 0.4:
        # Really weak overall – see which component is worst
        if len_score < 0.4:
            detail = "Try to give longer and more structured answers. Explain the situation, what you did, and the result."
        elif kw_score < 0.4:
            detail = "Try to use more of the key terms from the question in your answer, and clearly connect your experience to those topics."
        elif cos_score < 0.4:
            detail = "Make sure you are directly answering what is being asked, not going off-topic. Read the question carefully before responding."
        else:
            detail = "Try to give more concrete examples and clearly connect your experiences to the question."
    else:
        # Not terrible, but can improve
        if len_score < 0.6:
            detail = "Your answers are on the right track, but a bit short. Add more detail, context, and specific results."
        else:
            detail = "Keep refining your examples and add more specific details where possible."

    return f"{base} Focus especially on your weaker answers. {detail}"
