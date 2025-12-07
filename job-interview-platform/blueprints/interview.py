# blueprints/interview.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

from flask import (
    Blueprint,
    render_template,
    request,
    jsonify,
    session,
    redirect,
    url_for,
    flash,
)
import uuid

from extensions import mysql, logger, limiter
from services.questions import get_questions_for
from services.scoring import (
    score_answer_single,
    score_answer_combined,
    score_many,
    generate_detailed_advice,
)

interview_bp = Blueprint("interview", __name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_current_user() -> Optional[tuple[str, str, str]]:
    """
    Fetch the currently logged-in user's (email, username, contact_number).

    Returns:
        (email, username, contact_number) or None if not found / error.
    """
    user_id = session.get("user_id")
    if not user_id:
        return None

    cur = None
    try:
        cur = mysql.connection.cursor()
        cur.execute(
            "SELECT email, username, contact_number FROM users WHERE id = %s",
            (user_id,),
        )
        user = cur.fetchone()
        if not user:
            return None
        email, username, contact = user
        return email, username, contact
    except Exception as e:
        logger.error(f"❌ Error fetching user in _get_current_user: {e}")
        return None
    finally:
        if cur:
            cur.close()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@interview_bp.route("/chatapp")
def chat_app():
    """
    Main chat-app page that shows:
    - user info
    - latest chatbot result (if any)
    """
    if "user_id" not in session:
        return redirect(url_for("auth.login"))

    user_id = session["user_id"]

    user_info = _get_current_user()
    if not user_info:
        flash("User not found.", "error")
        return redirect(url_for("auth.login"))

    email, username, contact = user_info

    cur = None
    result = None
    try:
        cur = mysql.connection.cursor()
        cur.execute(
            """
            SELECT user_name,
                   position,
                   experience,
                   qualification_status,
                   confidence,
                   average_score,
                   created_at
            FROM chatbot
            WHERE user_id = %s
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (user_id,),
        )
        result = cur.fetchone()
    except Exception as e:
        logger.error(f"❌ Error fetching chatbot record in /chatapp: {e}")
    finally:
        if cur:
            cur.close()

    if result:
        (
            name,
            position,
            experience,
            qualified,
            confidence,
            average_score,
            created_at,
        ) = result
        chatbot_needed = False
    else:
        # fallback from session (probably from application form)
        name = session.get("name")
        position = session.get("position")
        experience = session.get("experience")
        qualified = confidence = average_score = created_at = None
        chatbot_needed = True

    return render_template(
        "chat-app.html",
        name=name,
        email=email,
        contact=contact,
        username=username,
        position=position,
        experience=experience,
        qualified=qualified,
        confidence=confidence,
        average_score=average_score,
        created_at=created_at,
        chatbot_needed=chatbot_needed,
    )


@interview_bp.route("/chat")
def chatbot_page():
    """
    Chatbot interview page.
    Prevents user from taking the chat twice if a chatbot record exists.
    """
    if "user_id" not in session:
        flash("Please submit your application first.", "error")
        return redirect(url_for("applicants.dashboard"))

    user_id = session["user_id"]
    name = session.get("name")
    experience = session.get("experience", 0)
    position = session.get("position", "Business Analyst")

    cur = None
    existing = None
    try:
        cur = mysql.connection.cursor()
        cur.execute(
            "SELECT id FROM chatbot WHERE user_id = %s ORDER BY id DESC LIMIT 1",
            (user_id,),
        )
        existing = cur.fetchone()
    except Exception as e:
        logger.error(f"❌ Error checking existing chatbot in /chat: {e}")
    finally:
        if cur:
            cur.close()

    if existing:
        flash("You have already completed the chat interview.", "info")
        return redirect(url_for("summary.summary_report"))

    return render_template(
        "chatbot.html",
        name=name,
        experience=experience,
        position=position,
        user_id=user_id,
    )


@interview_bp.route("/get_questions")
def get_questions_route():
    """
    Fetch questions for a given position and experience.
    Used by frontend (e.g. preview or manual fetch).
    """
    position = request.args.get("position") or session.get(
        "position", "Business Analyst"
    )

    try:
        exp = int(request.args.get("experience") or session.get("experience", 0))
    except (ValueError, TypeError):
        exp = 0

    questions = get_questions_for(position, exp)
    if not questions:
        return jsonify({"error": "No questions available"}), 404

    return jsonify({"questions": questions})


@interview_bp.route("/start-interview", methods=["POST"])
def start_interview():
    """
    Initializes a new interview session:
    - Generates a session_id
    - Loads questions based on position & years_of_experience
    - Resets question index & history
    """
    data: Dict[str, Any] = request.get_json(force=True) or {}

    position = data.get("position") or session.get("position", "Business Analyst")
    try:
        years = int(data.get("years_of_experience") or session.get("experience", 0))
    except (ValueError, TypeError):
        years = 0

    questions = get_questions_for(position, years)
    if not questions:
        return jsonify({"error": "No questions found for this role."}), 400

    session_id = str(uuid.uuid4())
    session["session_id"] = session_id
    session["questions"] = questions
    session["question_index"] = 0
    session["answers_history"] = []
    session["position"] = position
    session["experience"] = years

    first_question = questions[0]
    return jsonify({"session_id": session_id, "question": first_question})

@interview_bp.route("/next_question", methods=["POST"])
def next_question():
    data = request.get_json(force=True) or {}
    answer = (data.get("answer") or "").strip()
    current_question = (data.get("question") or "").strip()

    # Use questions stored in session (set in /start-interview)
    questions = session.get("questions") or []
    idx = int(session.get("question_index", 0))

    # Fallback: if questions got lost from session, recompute
    if not questions:
        position = session.get("position", "Business Analyst")
        try:
            years = int(session.get("experience", 0))
        except (ValueError, TypeError):
            years = 0
        questions = get_questions_for(position, years)
        session["questions"] = questions

    if not questions:
        return jsonify({"error": "Interview session not initialized properly."}), 400

    # Save and score current answer
    if answer and current_question:
        results = score_answer_combined(current_question, answer)
        history = session.get("answers_history", [])
        history.append(
            {
                "question": current_question,
                "answer": answer,
                **results,
            }
        )
        session["answers_history"] = history

        idx += 1
        session["question_index"] = idx

    # ---------- FINISHED ALL QUESTIONS ---------- #
    if idx >= len(questions):
        answers_history = session.get("answers_history", [])

        qa_pairs = [
            {"question": h["question"], "answer": h["answer"]}
            for h in answers_history
        ]
        summary = score_many(qa_pairs)
        advice = generate_detailed_advice(
            summary["answers"], summary["average_score"]
        )

        # compute "match score" and "confidence" from qualified answers
        qualified_count = sum(
            1
            for ans in summary.get("answers", [])
            if ans.get("qualification_status") == "Qualified"
        )
        total_answers = len(summary.get("answers", [])) or 1
        match_score_pct = round(100.0 * qualified_count / total_answers, 2)
        confidence_pct = match_score_pct  # or tweak if you want a different logic

        # store for later usage (e.g., summary report, pdf)
        session["qualification_status"] = summary["qualification_status"]

        # ---------- SAVE TO DATABASE ---------- #
        try:
            user_id = session.get("user_id")
            name = session.get("name")
            position = session.get("position")
            experience = session.get("experience")

            cur = mysql.connection.cursor()
            cur.execute(
                """
                INSERT INTO chatbot
                    (user_id, user_name, position, experience,
                     qualification_status, confidence, average_score)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    user_id,
                    name,
                    position,
                    experience,
                    summary["qualification_status"],
                    confidence_pct,
                    summary["average_score"],
                ),
            )
            mysql.connection.commit()
            cur.close()
        except Exception as e:
            logger.error(f"❌ Failed to save chatbot summary: {e}")

        return jsonify(
            {
                "finished": True,
                "summary": {
                    **summary,
                    "match_score_pct": match_score_pct,
                    "qualified_count": qualified_count,
                    "total_answers": total_answers,
                },
                "advice": advice,
                "answers_history": answers_history,
            }
        )

    # ---------- NOT FINISHED → NEXT QUESTION ---------- #
    next_q = questions[idx]
    return jsonify(
        {
            "finished": False,
            "next_question": next_q,
            "feedback": "Thank you for your answer!",
            "qualification_status": "In Progress",
        }
    )


@interview_bp.route("/score_answer", methods=["POST"])
@limiter.limit("10/minute")
def score_answer_route():
    """
    Scores a single question-answer pair.
    Used for instant feedback per answer.
    """
    data: Dict[str, Any] = request.get_json(force=True) or {}
    question = (data.get("question") or "").strip()
    answer = (data.get("answer") or "").strip()

    if not question or not answer:
        return jsonify({"error": "Missing question or answer."}), 400

    try:
        result = score_answer_single(question, answer)
    except Exception as e:
        logger.error(f"❌ Error in /score_answer: {e}")
        return jsonify(
            {
                "error": "Failed to score the answer.",
                "details": "Internal scoring error.",
            }
        ), 500

    return jsonify(result)


@interview_bp.route("/score", methods=["POST"])
def score_many_route():
    """
    Scores multiple question-answer pairs sent from the frontend.
    Expects:
        {
            "questions": [...],
            "answers": [...]
        }
    """
    data: Dict[str, Any] = request.get_json(force=True) or {}
    questions = data.get("questions", [])
    answers = data.get("answers", [])

    if len(questions) != len(answers):
        return jsonify({"error": "Mismatched questions and answers."}), 400

    qa_pairs = [
        {"question": q, "answer": a}
        for q, a in zip(questions, answers)
    ]

    try:
        summary = score_many(qa_pairs)
    except Exception as e:
        logger.error(f"❌ Error in /score: {e}")
        return jsonify({"error": "Failed to score answers."}), 500

    return jsonify(summary)


@interview_bp.route("/submit_interview", methods=["POST"])
def submit_interview():
    """
    Endpoint to submit the full interview from frontend, score all answers,
    and return overall feedback.

    Expects:
        {
            "answers": [
                {"question": "...", "answer": "..."},
                ...
            ]
        }
    """
    data: Dict[str, Any] = request.get_json(force=True) or {}
    answers: List[Dict[str, Any]] = data.get("answers", [])

    if not answers:
        return jsonify({"error": "No answers submitted."}), 400

    try:
        summary = score_many(answers)
        advice = generate_detailed_advice(
            summary["answers"], summary["average_score"]
        )
    except Exception as e:
        logger.error(f"❌ Error in /submit_interview: {e}")
        return jsonify({"error": "Failed to process interview."}), 500

    return jsonify(
        {
            "status": "success",
            "average_score": summary["average_score"],
            "feedback": advice,
            "summary": summary,
        }
    )

@interview_bp.route("/get_interview_summary", methods=["GET"])
def get_interview_summary():
    """
    Returns the interview summary based on answers stored in the session.
    """
    answers_history: List[Dict[str, Any]] = session.get("answers_history", [])
    if not answers_history:
        return jsonify({"error": "No answers submitted yet"}), 400

    qa_pairs = [
        {"question": h["question"], "answer": h["answer"]}
        for h in answers_history
    ]

    try:
        summary = score_many(qa_pairs)

        # Get overall status (Qualified / Partially Qualified / Not Qualified)
        status = summary.get("qualification_status", "Not Qualified")

        # --- Custom advice text based on status ---
        if status == "Qualified":
            advice = (
                "Congratulations! You just have passed. "
                "Wait for further announcement or call from us."
            )
        elif status == "Partially Qualified":
            advice = (
                "Thank you for completing the interview. "
                "You met some of the expectations for this role, "
                "and we will review your application further."
            )
        else:  # Not Qualified
            advice = (
                "Thank you for completing the interview. "
                "At this time you did not fully meet the requirements for this role."
            )

    except Exception as e:
        logger.error(f"❌ Error in /get_interview_summary: {e}")
        return jsonify({"error": "Failed to build interview summary."}), 500

    return jsonify(
        {
            "summary": summary,
            "advice": advice,
            "answers": answers_history,
        }
    )
