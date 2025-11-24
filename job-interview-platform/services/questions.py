# services/questions.py
from typing import List, Dict


# your original question bank (unchanged)
role_questions: Dict[str, Dict[str, list]] = {
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


def get_experience_level(years: int) -> str:
    if years <= 2:
        return "junior"
    if 3 <= years <= 5:
        return "mid"
    if 6 <= years <= 8:
        return "senior"
    return "special"


def normalize_position(position: str) -> str:
    """Map UI labels to role_questions keys."""
    p = position.lower()
    if "business" in p and "analyst" in p:
        return "business_analyst"
    if "project" in p:
        return "project_manager"
    if "java" in p or "developer" in p:
        return "java_developer"
    # default
    return "business_analyst"


def get_questions_for(position: str, years_experience: int) -> List[str]:
    role_key = normalize_position(position)
    level = get_experience_level(years_experience)
    return role_questions.get(role_key, {}).get(level, [])


# optional keyword-based templates (for dynamic follow-up)
keyword_templates = {
    "python": "Tell me about your experience with Python.",
    "django": "Have you used Django in any of your projects?",
    "team": "Describe your role in a team project.",
    "management": "How do you manage responsibilities?",
    "machine learning": "What ML projects have you done?",
    "communication": "How do you ensure good team communication?",
    "sql": "Tell me about your experience with SQL.",
}
