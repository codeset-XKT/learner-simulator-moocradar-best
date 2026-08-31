from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from learner_simulator.agent4edu_prompt import (  # noqa: E402
    build_action_prompt,
    build_profile_system_prompt,
    proficiency_context,
)
from learner_simulator.behavior import apply_behavior_adjustment, non_cognitive_factors  # noqa: E402
from learner_simulator.data import (  # noqa: E402
    AGENT4EDU_HISTORY_STEPS,
    AGENT4EDU_TARGET_STEPS,
    clean_sequence,
    load_questions,
    split_rows_agent4edu,
    take_sequence_rows,
)
from learner_simulator.llm import call_openai_compatible_chat, load_json  # noqa: E402
from learner_simulator.simulators import LLMLearnerSimulator  # noqa: E402

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render or call the autonomous LLM learner prompt.")
    parser.add_argument("--dataset-root", default="E:/yyx/KT数据集/XES3G5M/XES3G5M")
    parser.add_argument("--config", default="configs/llm.example.json")
    parser.add_argument("--source-rows", type=int, default=1000)
    parser.add_argument("--learner-index", type=int, default=0)
    parser.add_argument("--step-index", type=int, default=0)
    parser.add_argument("--call-api", action="store_true")
    parser.add_argument("--output", default="outputs/llm_prompt_preview.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    questions = load_questions(dataset_root / "metadata" / "questions.json")
    source_rows = take_sequence_rows(
        dataset_root / "kc_level" / "train_valid_sequences.csv",
        args.source_rows,
    )
    train_rows, test_rows = split_rows_agent4edu(
        source_rows,
        history_steps=AGENT4EDU_HISTORY_STEPS,
        target_steps=AGENT4EDU_TARGET_STEPS,
        max_users=args.learner_index + 1,
    )

    simulator = LLMLearnerSimulator(seed=42)
    simulator.fit(train_rows, questions=questions)

    row = test_rows[args.learner_index]
    uid = row["uid"]
    steps = clean_sequence(row)
    step = steps[args.step_index]
    qid = step["qid"]
    cid = step["cid"]
    question = questions[str(qid)]

    history_by_uid = {item["uid"]: item for item in train_rows}
    state, memory = simulator.initialize_from_history(
        uid,
        history_by_uid.get(uid),
        questions,
    )
    profile = simulator.get_profile(uid)
    profile_context = profile.to_context()
    p_cognitive = simulator.predict_probability(uid, qid, cid, state)
    factors = non_cognitive_factors(profile_context, int(step.get("position") or 0), simulator.random)
    p_correct = apply_behavior_adjustment(p_cognitive, factors)
    concept_default = simulator.concept_rates[cid].value(simulator.global_rate.value(0.5))
    mastery = state.get_mastery(cid, concept_default)
    memory_context = memory.snapshot(cid, question.get("kc_routes", []), mastery)

    context = {
        "uid": uid,
        "qid": qid,
        "history_interactions": AGENT4EDU_HISTORY_STEPS,
        "target_interactions": AGENT4EDU_TARGET_STEPS,
        "mastery": round(mastery, 4),
        "p_cognitive_baseline": round(p_cognitive, 4),
        "p_correct_baseline": round(p_correct, 4),
        "non_cognitive_factors": factors,
        "profile": profile_context,
        "short_memory": memory_context.get("short_memory", []),
        "long_memory": memory_context.get("long_memory", {}),
        "question_type": question.get("type", ""),
        "kc_routes": question.get("kc_routes", []),
        "content": question.get("content", ""),
        "options": question.get("options", {}),
    }
    routes = question.get("kc_routes", [])
    true_concept = str(routes[0]) if routes else str(cid)
    concept_options = simulator.concept_options(true_concept, seed=42 + qid + args.step_index)
    system_prompt = build_profile_system_prompt(profile_context)
    prompt = build_action_prompt(
        question=question,
        short_memory=memory_context.get("short_memory", []),
        long_memory=memory_context.get("long_memory", {}),
        concept_options=concept_options,
        proficiency=proficiency_context(true_concept, mastery),
        behavior_factors=factors,
    )
    result = None

    if args.call_api:
        config = load_json(ROOT / args.config)
        result = call_openai_compatible_chat(config, prompt, system_prompt=system_prompt)

    report = {
        "called_api": args.call_api,
        "context": context,
        "system_prompt": system_prompt,
        "prompt": prompt,
        "llm_result": result,
    }

    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "ok": True,
                "called_api": args.call_api,
                "output": str(output_path),
                "qid": qid,
                "p_cognitive_baseline": round(p_cognitive, 4),
                "p_correct_baseline": round(p_correct, 4),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
