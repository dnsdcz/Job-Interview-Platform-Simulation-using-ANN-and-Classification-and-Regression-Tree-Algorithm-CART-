# services/scoring.py
from typing import List, Dict, Any
from extensions import sentence_model, kw_model, logger
from sklearn.metrics.pairwise import cosine_similarity


def _encode(text: str):
    if not sentence_model:
        raise RuntimeError("SentenceTransformer model not loaded.")
    return sentence_model.encode([text])[0]


def explain_score(score: float) -> str:
    if score > 0.8:
        return "Excellent and highly relevant answer."
    if score > 0.6:
        return "Good answer with relevant content."
    if score > 0.4:
        return "Partially relevant. Add more detail and examples."
    return "Answer lacks relevance. Try to address the question more directly."


def compute_answer_score(question: str, answer: str) -> float:
    """Simple cosine similarity between question and answer."""
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


def score_answer_single(question: str, answer: str) -> Dict[str, Any]:
    score = compute_answer_score(question, answer)
    feedback = explain_score(score)
    status = "Qualified" if score >= 0.6 else "Not Qualified"
    return {
        "score": score,
        "qualification_status": status,
        "feedback": feedback,
    }


def score_answer_combined(question: str, answer: str) -> Dict[str, Any]:
    """More advanced: cosine + keyword overlap."""
    if not question.strip() or not answer.strip():
        return {
            "score": 0.0,
            "qualification_status": "Not Qualified",
            "matched_keywords": [],
            "total_keywords": [],
            "cosine_score": 0.0,
            "keyword_score": 0.0,
        }

    try:
        q_emb = _encode(question)
        a_emb = _encode(answer)
        cosine_score = cosine_similarity([q_emb], [a_emb])[0][0]

        if kw_model:
            kw_pairs = kw_model.extract_keywords(question, top_n=5)
            keywords = [k[0].lower() for k in kw_pairs]
        else:
            keywords = []

        answer_words = set(answer.lower().split())
        matched = [kw for kw in keywords if kw in answer_words]
        keyword_score = len(matched) / len(keywords) if keywords else 0.0

        final_score = round((cosine_score * 0.7) + (keyword_score * 0.3), 2)

        status = (
            "Qualified"
            if final_score >= 0.7
            else "Partially Qualified"
            if final_score >= 0.5
            else "Not Qualified"
        )

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
    total = 0.0
    results = []

    for pair in qa_pairs:
        q = pair.get("question", "")
        a = pair.get("answer", "")
        res = score_answer_combined(q, a)
        results.append(res)
        total += res["score"]

    avg = round(total / len(results), 2) if results else 0.0
    status = (
        "Qualified" if avg >= 0.7 else
        "Partially Qualified" if avg >= 0.5 else
        "Not Qualified"
    )

    return {
        "average_score": avg,
        "qualification_status": status,
        "answers": results,
    }


def generate_detailed_advice(qa_results: List[Dict[str, Any]], avg_score: float) -> str:
    if not qa_results:
        return "No advice available."

    if avg_score >= 0.7:
        base = "You performed very well overall. Your answers were relevant and detailed."
    elif avg_score >= 0.5:
        base = "Your interview performance is decent, but there is room to improve some answers."
    else:
        base = "Your answers suggest you should prepare more before the next interview."

    # find lowest score question for targeted advice
    lowest = min(qa_results, key=lambda r: r.get("score", 0))
    suggestion = (
        "Try to give more concrete examples and connect them clearly to the question."
    )

    return f"{base} Focus especially on questions where your score was low. {suggestion}"
