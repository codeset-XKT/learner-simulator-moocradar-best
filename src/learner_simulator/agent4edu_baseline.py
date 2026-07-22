from __future__ import annotations

import re
from typing import Any


def build_agent4edu_profile_prompt(profile: dict[str, Any]) -> str:
    activity = "high" if float(profile.get("activity_ratio", 0.0)) > 0.0017 else "low"
    diversity = "high" if float(profile.get("diversity_ratio", 0.0)) > 0.055 else "low"
    success_rate = float(profile.get("success_rate", 0.5))
    success = "good" if success_rate > 0.6 else "common" if success_rate > 0.3 else "poor"
    ability_value = float(profile.get("effective_ability", 0.5))
    ability = "good" if ability_value > 0.5 else "common" if ability_value > 0.4 else "poor"
    preference = profile.get("preference_route") or profile.get("preference_cid") or "unknown"
    return (
        "You are simulating a student doing exercises on an online education platform. "
        f"During online study, you exhibit {activity} activity. "
        f"You have {diversity} knowledge diversity. "
        f"The knowledge concept you practice most often is {preference}. "
        f"Your success rate is {success}. "
        f"You possess {ability} analytical and problem-solving skills.\n"
        "The information above is your # profile #."
    )


def build_agent4edu_action_prompt(
    question: dict[str, Any],
    short_memory: list[dict[str, Any]],
    long_memory: dict[str, Any],
    concept_options: list[str],
    proficiency: dict[str, Any],
) -> str:
    chunks: list[str] = []
    if short_memory:
        chunks.append("I will give you some exercise records as # Fact # below:")
        chunks.extend(_format_records(short_memory))
        chunks.append("The information above is your # short-term memory #.")

    significant = list(long_memory.get("significant_facts", []))
    if significant:
        chunks.append("I will give you some important exercise records as # Fact # below:")
        chunks.extend(_format_records(significant))

    chunks.append("Your current # Knowledge Proficiency # is:")
    chunks.append(f"- {proficiency['concept']}: {proficiency['level']}")
    status = str(long_memory.get("latest_learning_status", "")).strip()
    if status:
        chunks.append("Your current # Learning Status # is:")
        chunks.append(status)
    chunks.append("The information above is your # long-term memory #.")

    chunks.append(
        "Currently, you start to answer the recommended exercise. Its content information is as follows:\n"
        f"# Textual Content #: {question.get('content', '')}\n"
        f"# Options #: {question.get('options', '')}\n"
        f"# Reference Answer #: {question.get('answer', '')}\n"
        f"# Analysis #: {question.get('analysis', '')}"
    )
    chunks.append(
        "Please complete the following tasks based on the provided exercise and your profile and memory:\n"
        "Task1: Would you like to attempt to answer the exercise? Answer yes or no.\n"
        "Task2: Which knowledge concept does the exercise test? Select only one of the following options:\n"
        + "\n".join(f"- {concept}" for concept in concept_options)
        + "\nTask3: Please provide the detailed solution process for the exercise.\n"
        "Task4: Can you correctly answer the exercise? Answer yes or no."
    )
    chunks.append(
        "Output exactly in this format:\n"
        "Task1: <Yes or No>\n"
        "Task2: <one provided concept>\n"
        "Task3: <solution process>\n"
        "Task4: <Yes or No>"
    )
    return "\n\n".join(chunks)


def build_agent4edu_reflection_prompt(
    question: dict[str, Any],
    parsed_action: dict[str, Any],
    real_response: int,
    true_concept: str,
    short_memory: list[dict[str, Any]],
    long_memory: dict[str, Any],
) -> str:
    score = "correct" if real_response == 1 else "incorrect"
    corrective = ""
    predicted_concept = str(parsed_action.get("task2", "")).strip()
    if predicted_concept != true_concept:
        corrective += (
            f"The knowledge tested by this question is {true_concept}, but you wrongly think "
            f"the knowledge is {predicted_concept or 'unknown knowledge'}.\n"
        )
    predicted_correct = int(parsed_action.get("simulated_correct", 0))
    if predicted_correct == 0 and real_response == 1:
        corrective += (
            "You thought you could not solve this problem correctly, but in fact, "
            "you will solve it correctly.\n"
        )
    if predicted_correct == 1 and real_response == 0:
        corrective += (
            "You thought you could solve this problem correctly, but in fact, "
            "you do not answer it correctly.\n"
        )
    memory_context: list[str] = []
    if short_memory:
        memory_context.append("# short-term memory #")
        memory_context.extend(_format_records(short_memory))
    significant = list(long_memory.get("significant_facts", []))
    if significant:
        memory_context.append("# long-term memory #")
        memory_context.extend(_format_records(significant))
    status = str(long_memory.get("latest_learning_status", "")).strip()
    if status:
        memory_context.append("# previous Learning Status #")
        memory_context.append(status)
    return (
        "\n".join(memory_context)
        + ("\n" if memory_context else "")
        + corrective
        + f"You have just answered an exercise. Your answer was {score}.\n"
        f"# Exercise #: {question.get('content', '')}\n"
        f"# Reference Answer #: {question.get('answer', '')}\n"
        f"# Analysis #: {question.get('analysis', '')}\n"
        f"# Your Solution #: {parsed_action.get('task3', '')}\n"
        f"# Correct Knowledge Concept #: {true_concept}\n"
        "You should directly output your reflection and summarize your # Learning Status # within 500 words "
        "based on your # profile #, # short-term memory #, # long-term memory # and previous "
        "# Learning Status #. Do not output any other information."
    )


def parse_agent4edu_action(raw: str | None) -> dict[str, Any] | None:
    if not raw:
        return None
    matches: dict[str, str] = {}
    labels = ["Task1", "Task2", "Task3", "Task4"]
    for index, label in enumerate(labels):
        next_label = labels[index + 1] if index + 1 < len(labels) else None
        pattern = (
            rf"{label}\s*:\s*(.*?)(?={next_label}\s*:)"
            if next_label
            else rf"{label}\s*:\s*(.*)$"
        )
        match = re.search(pattern, raw, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            return None
        matches[label.lower()] = match.group(1).strip()
    return {
        **matches,
        "attempt": _yes_no(matches["task1"]),
        "identified_concept": matches["task2"],
        "solution_process": matches["task3"],
        "simulated_correct": int(_yes_no(matches["task4"]) == "yes"),
        "response_format": "agent4edu_tasks",
    }


def _format_records(records: list[dict[str, Any]]) -> list[str]:
    formatted: list[str] = []
    for index, item in enumerate(records, start=1):
        result = "rightly" if int(item.get("simulated_response", 0)) == 1 else "wrongly"
        routes = item.get("kc_routes") or []
        concept = routes[0] if routes else item.get("cid", "unknown")
        formatted.append(
            f"Fact{index}: You {result} answered an exercise.\n"
            f"- # Textual Content #: {item.get('content_preview', '')}\n"
            f"- # Knowledge Concept #: {concept}"
        )
    return formatted


def _yes_no(value: Any) -> str:
    text = str(value).strip().lower()
    return "yes" if text.startswith(("yes", "y")) else "no"
