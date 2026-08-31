from __future__ import annotations

import re
from typing import Any


# Values used by the official Agent4Edu Profile implementation.  They are
# global training-set means, not thresholds estimated on an evaluation cohort.
AGENT4EDU_ACTIVITY_MEAN = 0.039885658914728686
AGENT4EDU_DIVERSITY_MEAN = 0.06271615720524018


def build_agent4edu_profile_prompt(profile: dict[str, Any]) -> str:
    activity = "high" if float(profile.get("activity_ratio", 0.0)) > AGENT4EDU_ACTIVITY_MEAN else "low"
    diversity = "high" if float(profile.get("diversity_ratio", 0.0)) > AGENT4EDU_DIVERSITY_MEAN else "low"
    success_rate = float(profile.get("success_rate", 0.5))
    success = "high" if success_rate > 0.6 else "medium" if success_rate > 0.3 else "low"
    ability_value = float(profile.get("effective_ability", 0.5))
    ability = "good" if ability_value > 0.5 else "common" if ability_value > 0.4 else "poor"
    preference = profile.get("preference_route") or profile.get("preference_cid") or "unknown"
    activity_tip = (
        "you maintain a high level of online exercise activity and practice frequently"
        if activity == "high"
        else "you practice less regularly and with lower enthusiasm"
    )
    diversity_tip = (
        "you explore diverse knowledge categories"
        if diversity == "high"
        else "you focus on limited knowledge categories"
    )
    return (
        "You are a high school student engaging in self-directed exercising on an online learning platform. "
        f"During online study, you exhibit {activity} activity, which means {activity_tip}. "
        f"You have {diversity} knowledge diversity, which means {diversity_tip}. "
        f"The knowledge concept you practice most often is: {preference}. "
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
    """Build the official Agent4Edu four-task prompt over adapted data fields."""

    chunks: list[str] = []
    if short_memory:
        chunks.append("I will give you recent practice records as # Recent Facts # below.")
        chunks.extend(_format_records(short_memory))
        chunks.append("The information above is your # short-term memory #.")

    significant = list(long_memory.get("significant_facts", []))
    if significant:
        chunks.append("I will give you important reinforced records as # Reinforced Facts # below.")
        chunks.extend(_format_records(significant))

    if proficiency:
        chunks.append("Your current # Knowledge Proficiency # is:")
        value = proficiency.get("value")
        chunks.append(
            f"- {proficiency['concept']}: {proficiency['level']}"
            + (f" ({float(value):.4f})" if value is not None else "")
        )
    status = str(long_memory.get("latest_learning_status", "")).strip()
    if status:
        chunks.append("Your current # Learning Status # is:")
        chunks.append(status)
    if significant or proficiency or status:
        chunks.append("The information above is your # long-term memory #.")

    chunks.append(
        "Currently, you start to answer the recommended exercise. Its content information is as follows:\n"
        f"# Textual Content #: {question.get('content', '')}\n"
        f"# Options #: {question.get('options', '')}\n"
        f"# Reference Answer #: {question.get('answer', '')}\n"
        f"# Analysis #: {question.get('analysis', '')}"
    )
    chunks.append("To answer this exercise, please complete the following four tasks in sequence:")
    chunks.append(
        "Task 1 is to decide whether to attempt the recommended problem based on your ability in Profile "
        "and knowledge proficiency in Long-term Memory. If you consider the problem too difficult, output \"No\"; otherwise output \"Yes\". "
        "Regardless of your choice, the subsequent tasks will still be executed."
    )
    chunks.append("Task 2 is to choose one knowledge concept tested by this exercise from the following three options:")
    chunks.extend(f"- {concept}" for concept in concept_options)
    chunks.append("Only output the knowledge concept and do not output any other information for Task 2.")
    chunks.append(
        "Task 3 is to design a short problem-solving idea for this question based on your profile, memory and learning status, "
        "and then give a final answer. Your response should align with your profile, memory, and past performance."
    )
    chunks.append(
        "Task 4 is to estimate whether you can correctly solve this problem based on your profile, learning records, "
        "learning status, and problem-solving idea. If you can correctly solve it, answer \"Yes\"; otherwise answer \"No\"."
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
    """Official reflection call: post-action outcome feedback only.

    The official implementation supplies the observed practice score after the
    action and asks for a new long-term learning-status summary.  It does not
    inject an extra target-item solution prompt at this stage.
    """

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
    return (
        corrective
        + f"You have just answered an exercise. Your answer was {score}.\n"
        "You should directly output your reflection and summarize your # Learning Status # within 500 words "
        "based on your # profile #, # short-term memory #, # long-term memory # and previous "
        "# Learning Status #. Do not output any other information."
    )


def parse_agent4edu_action(raw: str | None) -> dict[str, Any] | None:
    if not raw:
        return None
    raw = re.sub(r"(?i)task\s*1\s*:", "Task1:", raw)
    raw = re.sub(r"(?i)task\s*2\s*:", "Task2:", raw)
    raw = re.sub(r"(?i)task\s*3\s*:", "Task3:", raw)
    raw = re.sub(r"(?i)task\s*4\s*:", "Task4:", raw)
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
