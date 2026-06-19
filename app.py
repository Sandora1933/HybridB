import json
import os
import torch

from llm import load_llm, generate_answer
from prompts import (
    build_baseline_system_message,
    build_json_augmented_system_message,
    build_baseline_prompt,
    build_json_augmented_prompt
)
from evaluate import evaluate_model_answer

RESULTS_DIR = "results"

def load_json_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_to_file(data, output_file):
    output_dir = os.path.dirname(output_file)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json_records(file_paths):
    records = []

    for file_path in file_paths:
        records.append(load_json_file(file_path))

    return records


def load_query(query_id):
    queries_data = load_json_file("data/queries.json")

    for query in queries_data:
        if query.get("query_id") == query_id:
            return query

    raise ValueError(f"Query with id '{query_id}' not found.")


def load_golden_answers(query_id):
    queries_data = load_json_file("data/queries.json")

    for query in queries_data:
        if query.get("query_id") == query_id:
            return query.get("gold_battles")

    raise ValueError(f"Golden answers for query with id '{query_id}' not found.")


def run_baseline_experiment(
    llm,
    query,
    all_battles,
    gold_battles,
    output_file,
):
    system_message = build_baseline_system_message()

    prompt = build_baseline_prompt(
        query=query,
        all_battles=all_battles
    )

    answer = generate_answer(
        llm=llm,
        prompt=prompt,
        system_message=system_message
    )

    evaluation = evaluate_model_answer(
        answer=answer,
        gold_battles=gold_battles
    )

    result = {
        "experiment_type": "baseline",
        "query": query,
        "all_battles": all_battles,
        "gold_battles": gold_battles,
        "system_message": system_message,
        "prompt": prompt,
        "answer": answer,
        "model_battles": evaluation["model_battles"],
        "metrics": {
            "precision": evaluation["precision"],
            "recall": evaluation["recall"],
            "f1": evaluation["f1"],
            "exact_match": evaluation["exact_match"]
        }
    }

    save_json_to_file(result, output_file)

    return result


def run_json_augmented_experiment(
    llm,
    query,
    all_battles,
    gold_battles,
    json_file_paths,
    output_file,
):
    json_records = load_json_records(json_file_paths)

    json_data = json.dumps(
        json_records,
        indent=2,
        ensure_ascii=False
    )

    system_message = build_json_augmented_system_message()

    prompt = build_json_augmented_prompt(
        query=query,
        all_battles=all_battles,
        json_data=json_data
    )

    answer = generate_answer(
        llm=llm,
        prompt=prompt,
        system_message=system_message
    )

    evaluation = evaluate_model_answer(
        answer=answer,
        gold_battles=gold_battles
    )

    result = {
        "experiment_type": "json_augmented",
        "query": query,
        "all_battles": all_battles,
        "gold_battles": gold_battles,
        "json_file_paths": json_file_paths,
        "system_message": system_message,
        "prompt": prompt,
        "answer": answer,
        "model_battles": evaluation["model_battles"],
        "metrics": {
            "precision": evaluation["precision"],
            "recall": evaluation["recall"],
            "f1": evaluation["f1"],
            "exact_match": evaluation["exact_match"]
        }
    }

    save_json_to_file(result, output_file)

    return result


def print_experiment_result(result):
    print(f"Experiment Type: {result['experiment_type']}\n")
    print(f"System Message: {result['system_message']}\n")
    print(f"Prompt: {result['prompt']}\n")
    print(f"Query: {result['query']}")
    print(f"All Battles: {result['all_battles']}")
    print(f"Gold Battles: {result['gold_battles']}")
    print(f"Model Answer: {result['answer']}\n")

    print("Evaluation:")
    metrics = result["metrics"]

    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Exact Match: {metrics['exact_match']}")


if __name__ == "__main__":
    all_battles = [
        "Battle of Leipzig",
        "Battle of Cannae",
        "Battle of Hastings",
        "Battle of Vienna",
        "Battle of Zama"
    ]

    json_file_paths = [
        "data/battle_of_leipzig.json",
        "data/battle_of_cannae.json",
        "data/battle_of_hastings.json",
        "data/battle_of_vienna.json",
        "data/battle_of_zama.json"
    ]

    query_id = "q17"
    query_data = load_query(query_id)

    query_id = query_data["query_id"]
    query = query_data["query"]
    gold_battles = query_data["gold_battles"]

    print("Loading LLM...")
    llm = load_llm()

    # print("Running baseline experiment...\n")
    # baseline_result = run_baseline_experiment(
    #     llm=llm,
    #     query=query,
    #     all_battles=all_battles,
    #     gold_battles=gold_battles,
    #     output_file=f"results/{query_id}_baseline_answer.json"
    # )

    # print_experiment_result(baseline_result)

    print("Running JSON-augmented experiment...")
    json_augmented_result = run_json_augmented_experiment(
        llm=llm,
        query=query,
        all_battles=all_battles,
        gold_battles=gold_battles,
        json_file_paths=json_file_paths,
        output_file=f"results/{query_id}_json_augmented_answer.json"
    )

    print_experiment_result(json_augmented_result)
