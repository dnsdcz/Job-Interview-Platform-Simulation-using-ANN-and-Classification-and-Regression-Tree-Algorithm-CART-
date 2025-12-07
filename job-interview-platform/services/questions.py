# services/questions.py
from __future__ import annotations

import random
from typing import List, Dict, Iterable


# ---------------------------------------------------------------------------
# Question bank
# ---------------------------------------------------------------------------

# Your original question bank (slightly formatted but unchanged in content)
role_questions: Dict[str, Dict[str, List[str]]] = {
    "business_analyst": {
        "junior": [
            "Tell me about yourself.",
            "Why are you leaving your current job?",
            "How do you handle criticism?",
            "Why should we hire you?",
            "Tell me about a time you had to solve a difficult problem at work.",
            "Give an example of a time you supported a teammate under pressure.",
            "Have you ever taken the lead on a project? What happened?",
            "Give an example of a mistake you made and how you handled it.",
            "How do you prioritize tasks when handling multiple small projects?",
            "How do you handle conflicts between team members?",
            "What is the difference between a project and a program in IT?",
            "How do you stay organized when working on multiple deliverables?",
        ],
        "mid": [
            "Tell me about yourself.",
            "Why are you leaving your current job?",
            "How do you handle criticism?",
            "Why should we hire you?",
            "Tell me about a time you had to solve a difficult problem at work.",
            "Give an example of a time you supported a teammate under pressure.",
            "Have you ever taken the lead on a project? What happened?",
            "Give an example of a mistake you made and how you handled it.",
            "How do you balance quality, time, and cost in a constrained project?",
            "How do you ensure proper communication between developers, QA, and business stakeholders?",
            "What methods do you use to manage project risks?",
            "How do you ensure cross-functional teams are aligned on goals?",
        ],
        "senior": [
            "Tell me about yourself.",
            "Why are you leaving your current job?",
            "How do you handle criticism?",
            "Why should we hire you?",
            "Tell me about a time you had to solve a difficult problem at work.",
            "Give an example of a time you supported a teammate under pressure.",
            "Have you ever taken the lead on a project? What happened?",
            "Give an example of a mistake you made and how you handled it.",
            "Describe a time when you had to make a difficult decision that impacted the entire team.",
            "Tell me about a program that failed and how you responded.",
            "What’s your approach to resource allocation across multiple high-priority programs?",
            "How do you evaluate whether a program should be continued, pivoted, or stopped?",
        ],
        "special": [
            "Tell me about yourself.",
            "Why are you leaving your current job?",
            "How do you handle criticism?",
            "Why should we hire you?",
            "Tell me about a time you had to solve a difficult problem at work.",
            "Give an example of a time you supported a teammate under pressure.",
            "Have you ever taken the lead on a project? What happened?",
            "Give an example of a mistake you made and how you handled it.",
            "Describe a time you coached or mentored other program/project managers.",
            "Describe a scenario where your technical understanding of IT architecture helped resolve a program issue.",
            "How do you forecast risk and opportunity over multi-year IT programs?",
            "What innovations have you introduced to improve program delivery or stakeholder engagement?",
        ],
    },
    "project_manager": {
        "junior": [
            "How do you prioritize tasks when handling multiple small projects?",
            "How do you handle conflicts between team members?",
            "What is the difference between a project and a program in IT?",
            "How do you stay organized when working on multiple deliverables?",
        ],
        "mid": [
            "How do you address situations where project requirements change during the development phase?",
            "Could you provide an example of a project where you had to manage multiple stakeholders?",
            "If you had the opportunity to enhance one aspect of the Business Analysis process, what would you focus on and why?",
            "What metrics do you track during a project, and how do you assess whether the project is on the right path toward success?",
        ],
        "senior": [
            "How do you align business analysis with organizational strategy?",
            "Can you describe a time when your analysis influenced the direction or outcome of a project?",
            "How do you assess the effectiveness of a newly implemented business process or change?",
            "What is your experience with Agile methodologies, and how do you adjust your business analysis approach to fit within Agile frameworks?",
        ],
        "special": [
            "What advanced business analysis methodologies or techniques do you employ to manage complex, large-scale projects?",
            "Could you share an example where you successfully led a team of business analysts on a high-profile project?",
            "If you were tasked with creating a new methodology for business analysis, what would it look like and why?",
            "How do you handle conflicting or contradictory data when making critical business recommendations?",
        ],
    },
    "java_developer": {
        "junior": [
            "Explain the difference between int[] arr = new int[5]; and int[] arr = {1, 2, 3, 4, 5};",
            "Can you explain the concept of inheritance and give a simple example?",
            "How would you create and use an ArrayList in Java?",
            "Can you describe a small Java program you’ve written and what it did?",
        ],
        "mid": [
            "What are the main principles of OOP and how does Java implement them?",
            "Explain the differences between ArrayList, LinkedList, and HashMap.",
            "How does garbage collection work in Java?",
            "Explain how you would connect a Java application to a database (JDBC or ORM).",
        ],
        "senior": [
            "How do you approach managing multi-threading in Java? Can you provide examples of situations where multi-threading was necessary?",
            "Tell us about a time when you mentored junior developers. What strategies did you use to help them improve their skills?",
            "How would you go about designing a scalable Java application? What potential challenges would you anticipate, and how would you address them?",
            "What is the difference between a synchronized block and a synchronized method in Java?",
        ],
        "special": [
            "How do you ensure high availability and fault tolerance in a distributed Java system?",
            "Describe your experience with designing enterprise-level Java applications. What were the most critical design decisions you made?",
            "If Java were to be replaced by a new language tomorrow, what would your transition strategy be?",
            "What is your approach to optimizing Java performance in high-load applications?",
        ],
    },
}


# ---------------------------------------------------------------------------
# Experience level / role normalization
# ---------------------------------------------------------------------------

def get_experience_level(years: int) -> str:
    """
    Map years of experience to an experience level bucket.

    Args:
        years: Total years of relevant experience.

    Returns:
        One of: "junior", "mid", "senior", "special".
    """
    if years <= 2:
        return "junior"
    if 3 <= years <= 5:
        return "mid"
    if 6 <= years <= 8:
        return "senior"
    return "special"


def normalize_position(position: str) -> str:
    """
    Map a free-text position label from the UI to an internal role key.

    This keeps the rest of the system using consistent keys (the dict keys
    in `role_questions`), while the UI can show more user-friendly titles.

    Args:
        position: The raw position string, e.g. "Business Analyst",
                  "Senior Java Developer", etc.

    Returns:
        A normalized role key such as "business_analyst", "project_manager",
        or "java_developer". Defaults to "business_analyst" if unsure.
    """
    if not position:
        return "business_analyst"

    p = position.lower()
    if "business" in p and "analyst" in p:
        return "business_analyst"
    if "project" in p:
        return "project_manager"
    if "java" in p or "developer" in p:
        return "java_developer"

    # default fallback
    return "business_analyst"


def get_questions_for(position: str, years_experience: int) -> List[str]:
    """
    Get the static question list for a given role and experience level.

    This is backwards-compatible with your original implementation and
    can be safely used anywhere in the codebase.

    Args:
        position: UI label or raw position text.
        years_experience: Candidate's years of experience.

    Returns:
        A list of questions for that role/level. Returns an empty list
        if no questions are configured.
    """
    role_key = normalize_position(position)
    level = get_experience_level(years_experience)
    return role_questions.get(role_key, {}).get(level, [])


# ---------------------------------------------------------------------------
# Optional keyword-based dynamic follow-up templates
# ---------------------------------------------------------------------------

keyword_templates: Dict[str, str] = {
    "python": "Tell me about your experience with Python.",
    "django": "Have you used Django in any of your projects?",
    "team": "Describe your role in a team project.",
    "management": "How do you manage responsibilities?",
    "machine learning": "What ML projects have you done?",
    "communication": "How do you ensure good team communication?",
    "sql": "Tell me about your experience with SQL.",
}


def infer_followup_questions(
    user_text: str,
    max_followups: int | None = 2,
) -> List[str]:
    """
    Infer dynamic follow-up questions based on keywords in the user's text.

    Example usage in your chat flow:
        followups = infer_followup_questions(last_answer)
        for q in followups:
            ask(q)

    Args:
        user_text: The candidate's free-text answer or profile.
        max_followups: Optional maximum number of follow-up questions to return.
                       Use None for no limit.

    Returns:
        A list of follow-up question strings. May be empty.
    """
    text = (user_text or "").lower()
    found: List[str] = []

    for keyword, question in keyword_templates.items():
        if keyword in text:
            found.append(question)

    # De-duplicate while preserving order
    seen = set()
    unique_found: List[str] = []
    for q in found:
        if q not in seen:
            unique_found.append(q)
            seen.add(q)

    if max_followups is not None:
        unique_found = unique_found[:max_followups]

    return unique_found


# ---------------------------------------------------------------------------
# Higher-level helpers for the chatbot / UI
# ---------------------------------------------------------------------------

def generate_interview_questions(
    position: str,
    years_experience: int,
    max_questions: int | None = None,
    shuffle: bool = True,
) -> List[str]:
    """
    Generate the main list of interview questions for a candidate.

    This wraps `get_questions_for` and adds:
    - optional shuffling (to avoid same fixed order every time)
    - optional maximum number of questions

    Args:
        position: Raw position label from the UI.
        years_experience: Candidate's years of experience.
        max_questions: Optionally limit number of questions. If None,
                       returns all questions.
        shuffle: Whether to randomize the order of questions.

    Returns:
        A list of questions ready to be used in the interview flow.
    """
    questions = list(get_questions_for(position, years_experience))

    if shuffle:
        random.shuffle(questions)

    if max_questions is not None:
        questions = questions[:max_questions]

    return questions


def extend_with_dynamic_followups(
    base_questions: Iterable[str],
    user_profile_text: str,
    max_followups: int | None = 2,
) -> List[str]:
    """
    Convenience helper: take a base question list, and extend it with
    dynamic follow-up questions inferred from some user text (e.g., CV,
    self-introduction, or previous answers).

    Args:
        base_questions: Existing list/iterable of questions.
        user_profile_text: Text to scan for keywords.
        max_followups: Cap on number of dynamic follow-ups to append.

    Returns:
        A new list containing the original questions plus dynamic follow-ups.
    """
    questions = list(base_questions)
    followups = infer_followup_questions(user_profile_text, max_followups=max_followups)

    return questions + followups
