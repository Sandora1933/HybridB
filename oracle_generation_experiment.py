##############################
#   The task: given json records (both golden battles and negatives) from KB, can the JSON-augmeneted LLM 
#   perform better in choosing the correct battles from the provided list of battles and explain why they
#   are correct based on the JSON evidence? 
#   
#   Configuration: Baseline LLM vs JSON-augmented LLM vs Raw-text-augmented LLM
##############################

import json
import os
from pathlib import Path

from llm import load_llm, generate_answer
from evaluate_closed_set_metrics import evaluate_model_answer


QUERIES_FILE = "data/queries/queries_rare_demo.json" # "data/queries/queries_all.json"

RESULTS_DIR = "results/oracle_generation"
DETAILS_DIR = os.path.join(RESULTS_DIR, "details")
FINAL_REPORTS_DIR = os.path.join(RESULTS_DIR, "final_reports")

JSON_KB_DIR = "data/json_kb_v1"
RAW_TEXT_DIR = "data/extracted_wikipedia_raw"

MAX_NEW_TOKENS = 1720


# -------------------------
# Basic file utilities
# -------------------------

def load_json_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_to_file(data, output_file):
    output_dir = os.path.dirname(output_file)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


# -------------------------
# Query helpers
# -------------------------

def battle_dict_to_items(battle_dict):
    """
    Convert this query format:

    {
      "Q1113659": "Battle of Fulford",
      "Q203225": "Battle of Stamford Bridge"
    }

    into:

    [
      {"qid": "Q1113659", "name": "Battle of Fulford"},
      {"qid": "Q203225", "name": "Battle of Stamford Bridge"}
    ]
    """

    if battle_dict is None:
        return []

    if not isinstance(battle_dict, dict):
        raise ValueError(
            "Expected battle collection to be a dictionary: "
            "{qid: battle_name}."
        )

    return [
        {
            "qid": qid,
            "name": name
        }
        for qid, name in battle_dict.items()
    ]


def get_gold_battle_items(query_data):
    return battle_dict_to_items(query_data.get("gold_battles", {}))


def get_negative_battle_items(query_data):
    return battle_dict_to_items(query_data.get("closed_set_negatives", {}))


def get_all_battle_items(query_data):
    """
    Oracle closed set = gold battles + closed-set negatives.
    """

    items = []
    seen_qids = set()

    for item in get_gold_battle_items(query_data) + get_negative_battle_items(query_data):
        qid = item["qid"]

        if qid not in seen_qids:
            items.append(item)
            seen_qids.add(qid)

    return items


def get_battle_names(battle_items):
    return [item["name"] for item in battle_items]


def get_gold_battle_names(query_data):
    return get_battle_names(get_gold_battle_items(query_data))


def get_all_battle_names(query_data):
    return get_battle_names(get_all_battle_items(query_data))


def get_json_path(qid):
    file_path = Path(JSON_KB_DIR) / f"{qid}.json"

    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found for {qid}: {file_path}")

    return str(file_path)


def get_raw_text_path(qid):
    file_path = Path(RAW_TEXT_DIR) / f"{qid}.txt"

    if not file_path.exists():
        raise FileNotFoundError(f"Raw text file not found for {qid}: {file_path}")

    return str(file_path)


def load_json_records_for_query(query_data):
    records = []
    file_paths = {}

    for item in get_all_battle_items(query_data):
        qid = item["qid"]
        battle_name = item["name"]
        json_path = get_json_path(qid)

        records.append(load_json_file(json_path))

        file_paths[qid] = {
            "battle_name": battle_name,
            "json_path": json_path
        }

    return records, file_paths


def load_raw_texts_for_query(query_data):
    records = []
    file_paths = {}

    for item in get_all_battle_items(query_data):
        qid = item["qid"]
        battle_name = item["name"]
        raw_text_path = get_raw_text_path(qid)

        records.append({
            "qid": qid,
            "battle": battle_name,
            "source_file": raw_text_path,
            "text": load_text_file(raw_text_path)
        })

        file_paths[qid] = {
            "battle_name": battle_name,
            "raw_text_path": raw_text_path
        }

    return records, file_paths


# -------------------------
# Prompt builders
# -------------------------

def build_baseline_system_message():
    return (
        "You are a historical battle question-answering assistant.\n\n"
        "Rules:\n"
        "- Use your own general knowledge.\n"
        "- Return only battles from the given battle set.\n"
        "- Include a battle only if you are confident that the battle matches the question.\n"
        "- If you are uncertain whether a battle matches, exclude it.\n"
        "- If none of the battles from the set match the question, return an empty battles list.\n"
        "- Do not invent details that you are not confident about.\n\n"
        "Output format:\n"
        "Return only valid JSON. Do not include markdown, code fences, or extra text.\n"
        "The JSON must have exactly this structure:\n"
        "{\n"
        '  "battles": ["Battle name 1", "Battle name 2"],\n'
        '  "explanations": {\n'
        '    "Battle name 1": "one short explanation",\n'
        '    "Battle name 2": "one short explanation"\n'
        "  }\n"
        "}\n\n"
        "If there are no matching battles, return:\n"
        "{\n"
        '  "battles": [],\n'
        '  "explanations": {}\n'
        "}"
    )


def build_json_augmented_system_message():
    return (
        "You are a historical battle question-answering assistant.\n\n"
        "Rules:\n"
        "- Use only the provided JSON battle records as evidence.\n"
        "- Do not use your internal knowledge or outside knowledge.\n"
        "- Return only battles from the given battle set.\n"
        "- Include a battle only if the provided JSON explicitly supports that it matches the question.\n"
        "- If you are uncertain whether the JSON supports a battle, exclude it.\n"
        "- If the JSON data does not support an answer, return an empty battles list.\n"
        "- Do not invent missing details.\n"
        "- In each explanation, mention the relevant JSON field if possible.\n\n"
        "Output format:\n"
        "Return only valid JSON. Do not include markdown, code fences, or extra text.\n"
        "The JSON must have exactly this structure:\n"
        "{\n"
        '  "battles": ["Battle name 1", "Battle name 2"],\n'
        '  "explanations": {\n'
        '    "Battle name 1": "one short explanation based on JSON evidence",\n'
        '    "Battle name 2": "one short explanation based on JSON evidence"\n'
        "  }\n"
        "}\n\n"
        "If there are no matching battles or the answer is not supported by the JSON, return:\n"
        "{\n"
        '  "battles": [],\n'
        '  "explanations": {}\n'
        "}"
    )


def build_raw_text_augmented_system_message():
    return (
        "You are a historical battle question-answering assistant.\n\n"
        "Rules:\n"
        "- Use only the provided raw battle texts as evidence.\n"
        "- Do not use your internal knowledge or outside knowledge.\n"
        "- Return only battles from the given battle set.\n"
        "- Include a battle only if the provided raw text explicitly supports that it matches the question.\n"
        "- If you are uncertain whether the raw text supports a battle, exclude it.\n"
        "- If the raw text data does not support an answer, return an empty battles list.\n"
        "- Do not invent missing details.\n"
        "- In each explanation, briefly mention the supporting textual evidence.\n\n"
        "Output format:\n"
        "Return only valid JSON. Do not include markdown, code fences, or extra text.\n"
        "The JSON must have exactly this structure:\n"
        "{\n"
        '  "battles": ["Battle name 1", "Battle name 2"],\n'
        '  "explanations": {\n'
        '    "Battle name 1": "one short explanation based on raw-text evidence",\n'
        '    "Battle name 2": "one short explanation based on raw-text evidence"\n'
        "  }\n"
        "}\n\n"
        "If there are no matching battles or the answer is not supported by the raw text, return:\n"
        "{\n"
        '  "battles": [],\n'
        '  "explanations": {}\n'
        "}"
    )


def build_baseline_prompt(query, all_battles):
    battle_list = "\n".join(f"- {name}" for name in all_battles)

    return (
        "Battle set:\n"
        f"{battle_list}\n\n"
        "Question:\n"
        f"{query}"
    )


def build_json_augmented_prompt(query, all_battles, json_records):
    battle_list = "\n".join(f"- {name}" for name in all_battles)

    json_data = json.dumps(
        json_records,
        indent=2,
        ensure_ascii=False
    )

    return (
        "Battle set:\n"
        f"{battle_list}\n\n"
        "Question:\n"
        f"{query}\n\n"
        "JSON battle records:\n"
        f"{json_data}"
    )


def build_raw_text_augmented_prompt(query, all_battles, raw_text_records):
    battle_list = "\n".join(f"- {name}" for name in all_battles)

    context_parts = []

    for record in raw_text_records:
        context_parts.append(
            "Battle:\n"
            f"{record['battle']}\n\n"
            "Raw text:\n"
            f"{record['text']}"
        )

    raw_text_context = "\n\n---\n\n".join(context_parts)

    return (
        "Battle set:\n"
        f"{battle_list}\n\n"
        "Question:\n"
        f"{query}\n\n"
        "Raw battle texts:\n"
        f"{raw_text_context}"
    )


# -------------------------
# Evaluation helpers
# -------------------------

def evaluate_answer(answer, gold_battles):
    try:
        return evaluate_model_answer(
            answer=answer,
            gold_battles=gold_battles
        )
    except Exception as e:
        return {
            "model_battles": [],
            "gold_battles": gold_battles,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "exact_match": False,
            "parse_error": str(e)
        }


# -------------------------
# Experiment execution
# -------------------------

def run_single_experiment(
    llm,
    experiment_type,
    query_data,
    output_file
):
    query_id = query_data["query_id"]
    query = query_data["generation_query"]

    all_battles = get_all_battle_names(query_data)
    gold_battles = get_gold_battle_names(query_data)

    if experiment_type == "baseline":
        system_message = build_baseline_system_message()
        prompt = build_baseline_prompt(
            query=query,
            all_battles=all_battles
        )
        context_file_paths = None

    elif experiment_type == "json_augmented":
        json_records, context_file_paths = load_json_records_for_query(query_data)
        system_message = build_json_augmented_system_message()
        prompt = build_json_augmented_prompt(
            query=query,
            all_battles=all_battles,
            json_records=json_records
        )

    elif experiment_type == "raw_text_augmented":
        raw_text_records, context_file_paths = load_raw_texts_for_query(query_data)
        system_message = build_raw_text_augmented_system_message()
        prompt = build_raw_text_augmented_prompt(
            query=query,
            all_battles=all_battles,
            raw_text_records=raw_text_records
        )

    else:
        raise ValueError(f"Unknown experiment_type: {experiment_type}")

    answer = generate_answer(
        llm=llm,
        prompt=prompt,
        system_message=system_message,
        max_new_tokens=MAX_NEW_TOKENS
    )

    evaluation = evaluate_answer(
        answer=answer,
        gold_battles=gold_battles
    )

    result = {
        "experiment_type": experiment_type,
        "query_id": query_id,
        "query": query,
        "query_type": query_data.get("query_type"),
        "query_structure": query_data.get("query_structure"),
        "query_difficulty": query_data.get("query_difficulty"),
        "all_battles": all_battles,
        "gold_battles": gold_battles,
        "gold_battle_qids": list(query_data.get("gold_battles", {}).keys()),
        "closed_set_negatives": list(query_data.get("closed_set_negatives", {}).values()),
        "closed_set_negative_qids": list(query_data.get("closed_set_negatives", {}).keys()),
        "context_file_paths": context_file_paths,
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

    if "parse_error" in evaluation:
        result["metrics"]["parse_error"] = evaluation["parse_error"]

    save_json_to_file(result, output_file)

    return result


def build_run_error_result(experiment_type, query_data, error):
    return {
        "experiment_type": experiment_type,
        "query_id": query_data.get("query_id"),
        "query": query_data.get("generation_query"),
        "query_type": query_data.get("query_type"),
        "query_structure": query_data.get("query_structure"),
        "query_difficulty": query_data.get("query_difficulty"),
        "all_battles": get_all_battle_names(query_data),
        "gold_battles": get_gold_battle_names(query_data),
        "gold_battle_qids": list(query_data.get("gold_battles", {}).keys()),
        "closed_set_negatives": list(query_data.get("closed_set_negatives", {}).values()),
        "closed_set_negative_qids": list(query_data.get("closed_set_negatives", {}).keys()),
        "answer": None,
        "model_battles": [],
        "metrics": {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "exact_match": False,
            "run_error": str(error)
        }
    }


# -------------------------
# Reports
# -------------------------

def build_final_report(results):
    report = {}

    metric_names = ["precision", "recall", "f1", "exact_match"]

    for result in results:
        query_id = result["query_id"]
        metrics = result["metrics"]

        report[query_id] = {
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "exact_match": metrics["exact_match"],
            "model_battles": result["model_battles"],
            "gold_battles": result["gold_battles"]
        }

        if metrics.get("parse_error") is not None:
            report[query_id]["parse_error"] = metrics["parse_error"]

        if metrics.get("run_error") is not None:
            report[query_id]["run_error"] = metrics["run_error"]

    final_results = {}

    for metric_name in metric_names:
        values = [
            result["metrics"][metric_name]
            for result in results
        ]

        final_results[metric_name] = (
            sum(values) / len(values)
            if values
            else 0.0
        )

    parse_error_count = sum(
        1
        for result in results
        if result["metrics"].get("parse_error") is not None
    )

    run_error_count = sum(
        1
        for result in results
        if result["metrics"].get("run_error") is not None
    )

    final_results["num_queries"] = len(results)
    final_results["parse_error_count"] = parse_error_count
    final_results["run_error_count"] = run_error_count

    report["Final_results"] = final_results

    return report


def build_comparison_report(final_reports):
    comparison = {
        "configurations": {},
        "best_by_metric": {}
    }

    metric_names = ["precision", "recall", "f1", "exact_match"]

    for experiment_type, report in final_reports.items():
        comparison["configurations"][experiment_type] = report["Final_results"]

    for metric_name in metric_names:
        best_config = None
        best_score = -1.0

        for experiment_type, report in final_reports.items():
            score = report["Final_results"][metric_name]

            if score > best_score:
                best_score = score
                best_config = experiment_type

        comparison["best_by_metric"][metric_name] = {
            "configuration": best_config,
            "score": best_score
        }

    return comparison


def print_final_report(experiment_type, report):
    final_results = report["Final_results"]

    print(f"\nFinal report: {experiment_type}")
    print(f"Queries: {final_results['num_queries']}")
    print(f"Precision: {final_results['precision']:.4f}")
    print(f"Recall: {final_results['recall']:.4f}")
    print(f"F1: {final_results['f1']:.4f}")
    print(f"Exact Match: {final_results['exact_match']:.4f}")
    print(f"Parse Errors: {final_results['parse_error_count']}")
    print(f"Run Errors: {final_results['run_error_count']}")


# -------------------------
# Main
# -------------------------

def main():
    os.makedirs(DETAILS_DIR, exist_ok=True)
    os.makedirs(FINAL_REPORTS_DIR, exist_ok=True)

    if not Path(QUERIES_FILE).exists():
        raise FileNotFoundError(f"Queries file not found: {QUERIES_FILE}")

    if not Path(JSON_KB_DIR).exists():
        raise FileNotFoundError(f"JSON KB directory not found: {JSON_KB_DIR}")

    if not Path(RAW_TEXT_DIR).exists():
        raise FileNotFoundError(f"Raw text directory not found: {RAW_TEXT_DIR}")

    queries = load_json_file(QUERIES_FILE)

    print(f"Queries loaded: {len(queries)}")
    print(f"JSON KB directory: {JSON_KB_DIR}")
    print(f"Raw text directory: {RAW_TEXT_DIR}")

    print("Loading LLM...")
    llm = load_llm()

    experiment_types = [
        "baseline",
        "json_augmented",
        "raw_text_augmented"
    ]

    results_by_experiment = {
        experiment_type: []
        for experiment_type in experiment_types
    }

    for query_data in queries:
        query_id = query_data["query_id"]
        all_battles = get_all_battle_names(query_data)

        print(f"\nRunning query {query_id}")
        print(f"Candidate battles: {len(all_battles)}")

        for experiment_type in experiment_types:
            print(f"  Running {experiment_type}...")

            output_file = os.path.join(
                DETAILS_DIR,
                f"{query_id}_{experiment_type}_answer.json"
            )

            try:
                result = run_single_experiment(
                    llm=llm,
                    experiment_type=experiment_type,
                    query_data=query_data,
                    output_file=output_file
                )

            except Exception as e:
                result = build_run_error_result(
                    experiment_type=experiment_type,
                    query_data=query_data,
                    error=e
                )

                save_json_to_file(result, output_file)

            results_by_experiment[experiment_type].append(result)

            metrics = result["metrics"]
            print(
                f"    P={metrics['precision']:.4f} "
                f"R={metrics['recall']:.4f} "
                f"F1={metrics['f1']:.4f} "
                f"EM={metrics['exact_match']}"
            )

            if metrics.get("run_error") is not None:
                print(f"    Run error: {metrics['run_error']}")

            if metrics.get("parse_error") is not None:
                print(f"    Parse error: {metrics['parse_error']}")

    final_reports = {}

    for experiment_type, results in results_by_experiment.items():
        report = build_final_report(results)
        final_reports[experiment_type] = report

        report_file = os.path.join(
            FINAL_REPORTS_DIR,
            f"{experiment_type}_final_report.json"
        )

        save_json_to_file(report, report_file)
        print_final_report(experiment_type, report)

    comparison_report = build_comparison_report(final_reports)

    save_json_to_file(
        comparison_report,
        os.path.join(FINAL_REPORTS_DIR, "comparison_final_report.json")
    )

    print("\nDone.")
    print(f"Detailed answers saved to: {DETAILS_DIR}")
    print(f"Final reports saved to: {FINAL_REPORTS_DIR}")


if __name__ == "__main__":
    main()
