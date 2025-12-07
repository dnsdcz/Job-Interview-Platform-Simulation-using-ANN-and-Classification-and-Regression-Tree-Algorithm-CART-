# services/scoring.py
from typing import List, Dict, Any, Tuple
from extensions import sentence_model, kw_model, logger
from sklearn.metrics.pairwise import cosine_similarity

# ====== CONFIGURABLE THRESHOLDS ====== #
COSINE_ONLY_QUALIFIED_THRESHOLD = 0.6
COMBINED_FULL_QUALIFIED_THRESHOLD = 0.7
COMBINED_PARTIAL_QUALIFIED_THRESHOLD = 0.5


def _encode(text: str):
    """
    Encodes text using the global sentence_model.
    Raises RuntimeError if model not loaded.
    """
    if not sentence_model:
        raise RuntimeError("SentenceTransformer model not loaded.")
    return sentence_model.encode([text])[0]


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
    Simple cosine similarity between question and answer.
    Used by score_answer_single.
    """
    if not question.strip() or not answer.strip():
        return 0.0

    try:
        q_emb = _encode(question)
        a_emb = _encode(answer)
        score = cosine_similarity([q_emb], [a_emb])[0][0]
        return round(float(score), 2)
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
    """
    keywords = _extract_keywords(question, top_n=top_n)
    if not keywords:
        return [], [], 0.0

    # basic normalization: split by whitespace, lowercase
    answer_words = set(answer.lower().split())
    matched = [kw for kw in keywords if kw in answer_words]

    keyword_score = len(matched) / len(keywords) if keywords else 0.0
    return matched, keywords, keyword_score


def score_answer_single(question: str, answer: str) -> Dict[str, Any]:
    """
    Scores an answer for a single question using cosine similarity only.
    Suitable for quick, per-question feedback.
    """
    score = compute_answer_score(question, answer)
    feedback = explain_score(score)
    status = "Qualified" if score >= COSINE_ONLY_QUALIFIED_THRESHOLD else "Not Qualified"

    return {
        "score": score,
        "qualification_status": status,
        "feedback": feedback,
    }


def score_answer_combined(question: str, answer: str) -> Dict[str, Any]:
    """
    Combined scoring:
    - cosine similarity between question and answer (70%)
    - keyword overlap between question keywords and answer (30%)
    Returns a dict with:
    - score
    - qualification_status
    - matched_keywords
    - total_keywords
    - cosine_score
    - keyword_score
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
        }

    try:
        # cosine similarity
        q_emb = _encode(question)
        a_emb = _encode(answer)
        cosine_score = cosine_similarity([q_emb], [a_emb])[0][0]

        # keyword overlap
        matched, keywords, keyword_score = _compute_keyword_overlap(question, answer)

        final_score = round((cosine_score * 0.7) + (keyword_score * 0.3), 2)

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
        q = pair.get("question", "")
        a = pair.get("answer", "")
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
    """
    if not qa_results:
        return "No advice available."

    if avg_score >= COMBINED_FULL_QUALIFIED_THRESHOLD:
        base = "You performed very well overall. Your answers were relevant and detailed."
    elif avg_score >= COMBINED_PARTIAL_QUALIFIED_THRESHOLD:
        base = "Your interview performance is decent, but there is room to improve some answers."
    else:
        base = "Your answers suggest you should prepare more before the next interview."

    # find lowest score question for targeted advice
    lowest = min(qa_results, key=lambda r: r.get("score", 0.0))
    lowest_score = lowest.get("score", 0.0)

    # You could expand this later to give more specific advice based on the score
    if lowest_score < 0.4:
        suggestion = (
            "Try to give more concrete examples and clearly connect your experiences to the question."
        )
    else:
        suggestion = "Keep refining your examples and add more specific details where possible."

    return f"{base} Focus especially on questions where your score was low. {suggestion}"
