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

from typing import Any, Literal
from matplotlib import pyplot as plt
import numpy as np

from llm import load_llm, generate_answer


QUERIES_FILE = "data/queries/queries_all.json"

RESULTS_DIR = "results/oracle_generation"
DETAILS_DIR_KB1 = os.path.join(RESULTS_DIR, "details_kb1")
DETAILS_DIR_KB2 = os.path.join(RESULTS_DIR, "details_kb2")
FINAL_REPORTS_DIR_KB1 = os.path.join(RESULTS_DIR, "final_reports_kb1")
FINAL_REPORTS_DIR_KB2 = os.path.join(RESULTS_DIR, "final_reports_kb2")
KB1_KB2_RESULTS_DIR = os.path.join(RESULTS_DIR, "kb1_vs_kb2")
COMMON_RARE_RESULTS_DIR = os.path.join(RESULTS_DIR, "common_vs_rare")

JSON_KB1_DIR = "data/json_kb_v1"
JSON_KB2_DIR = "data/json_kb_v2"
RAW_TEXT_DIR = "data/extracted_wikipedia_raw"

MAX_NEW_TOKENS = 1720

# -------------------------
# Query filtering helpers
# -------------------------

def _filter_queries_by_rarity(
    queries: list[dict[str, Any]],
    rarity: Literal["common", "rare"],
) -> list[dict[str, Any]]:
    return [
        query
        for query in queries
        if query.get("context_rarity") == rarity
    ]


def _filter_queries_by_difficulty(
    queries: list[dict[str, Any]],
    difficulty: Literal["easy", "medium", "hard"],
) -> list[dict[str, Any]]:
    """Filter queries by difficulty: 'easy', 'medium', or 'hard'."""
    return [
        query
        for query in queries
        if query.get("query_difficulty") == difficulty
    ]

def _filter_queries_by_type(
    queries: list[dict[str, Any]],
    query_type: Literal[
        "Participant-related",
        "Similarity",
        "Narrative / Tactical",
        "Temporal",
        "Spatial / Terrain",
        "Outcome-related",
        "Weapon / Unit-related",
    ],
) -> list[dict[str, Any]]:
    """Filter queries by query type."""
    return [
        query
        for query in queries
        if query.get("query_type") == query_type
    ]



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


def format_battle_set(battle_items):
    return "\n".join(
        f"- {item['qid']}: {item['name']}"
        for item in battle_items
    )


def get_json_path(qid, json_dir=JSON_KB1_DIR):
    file_path = Path(json_dir) / f"{qid}.json"

    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found for {qid}: {file_path}")

    return str(file_path)


def get_raw_text_path(qid):
    file_path = Path(RAW_TEXT_DIR) / f"{qid}.txt"

    if not file_path.exists():
        raise FileNotFoundError(f"Raw text file not found for {qid}: {file_path}")

    return str(file_path)


def load_json_records_for_query(query_data, json_dir=JSON_KB1_DIR):
    records = []
    file_paths = {}

    for item in get_all_battle_items(query_data):
        qid = item["qid"]
        battle_name = item["name"]
        json_path = get_json_path(qid, json_dir=json_dir)

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
        "- Return only battle QIDs from the given battle set.\n"
        "- Include a battle only if you are confident that the battle matches the question.\n"
        "- If you are uncertain whether a battle matches, exclude it.\n"
        "- If none of the battles from the set match the question, return an empty battles list.\n"
        "- Do not invent details that you are not confident about.\n\n"
        "Output format:\n"
        "Return only valid JSON. Do not include markdown, code fences, or extra text.\n"
        "The JSON must have exactly this structure:\n"
        "{\n"
        '  "battles": ["QID1", "QID2"],\n'
        '  "battle_names": {\n'
        '    "QID1": "Battle name 1",\n'
        '    "QID2": "Battle name 2"\n'
        "  },\n"
        '  "explanations": {\n'
        '    "QID1": "one short explanation",\n'
        '    "QID2": "one short explanation"\n'
        "  }\n"
        "}\n\n"
        "If there are no matching battles, return:\n"
        "{\n"
        '  "battles": [],\n'
        '  "battle_names": {},\n'
        '  "explanations": {}\n'
        "}"
    )


def build_json_augmented_system_message():
    return (
        "You are a historical battle question-answering assistant.\n\n"
        "Rules:\n"
        "- Use only the provided JSON battle records as evidence.\n"
        "- Do not use your internal knowledge or outside knowledge.\n"
        "- Return only battle QIDs from the given battle set.\n"
        "- Include a battle only if the provided JSON explicitly supports that it matches the question.\n"
        "- If you are uncertain whether the JSON supports a battle, exclude it.\n"
        "- If the JSON data does not support an answer, return an empty battles list.\n"
        "- Do not invent missing details.\n"
        "- In each explanation, mention the relevant JSON field if possible.\n\n"
        "Output format:\n"
        "Return only valid JSON. Do not include markdown, code fences, or extra text.\n"
        "The JSON must have exactly this structure:\n"
        "{\n"
        '  "battles": ["QID1", "QID2"],\n'
        '  "battle_names": {\n'
        '    "QID1": "Battle name 1",\n'
        '    "QID2": "Battle name 2"\n'
        "  },\n"
        '  "explanations": {\n'
        '    "QID1": "one short explanation based on JSON evidence",\n'
        '    "QID2": "one short explanation based on JSON evidence"\n'
        "  }\n"
        "}\n\n"
        "If there are no matching battles or the answer is not supported by the JSON, return:\n"
        "{\n"
        '  "battles": [],\n'
        '  "battle_names": {},\n'
        '  "explanations": {}\n'
        "}"
    )


def build_raw_text_augmented_system_message():
    return (
        "You are a historical battle question-answering assistant.\n\n"
        "Rules:\n"
        "- Use only the provided raw battle texts as evidence.\n"
        "- Do not use your internal knowledge or outside knowledge.\n"
        "- Return only battle QIDs from the given battle set.\n"
        "- Include a battle only if the provided raw text explicitly supports that it matches the question.\n"
        "- If you are uncertain whether the raw text supports a battle, exclude it.\n"
        "- If the raw text data does not support an answer, return an empty battles list.\n"
        "- Do not invent missing details.\n"
        "- In each explanation, briefly mention the supporting textual evidence.\n\n"
        "Output format:\n"
        "Return only valid JSON. Do not include markdown, code fences, or extra text.\n"
        "The JSON must have exactly this structure:\n"
        "{\n"
        '  "battles": ["QID1", "QID2"],\n'
        '  "battle_names": {\n'
        '    "QID1": "Battle name 1",\n'
        '    "QID2": "Battle name 2"\n'
        "  },\n"
        '  "explanations": {\n'
        '    "QID1": "one short explanation based on raw-text evidence",\n'
        '    "QID2": "one short explanation based on raw-text evidence"\n'
        "  }\n"
        "}\n\n"
        "If there are no matching battles or the answer is not supported by the raw text, return:\n"
        "{\n"
        '  "battles": [],\n'
        '  "battle_names": {},\n'
        '  "explanations": {}\n'
        "}"
    )


def build_baseline_prompt(query, all_battle_items):
    battle_list = format_battle_set(all_battle_items)

    return (
        "Battle set:\n"
        f"{battle_list}\n\n"
        "Question:\n"
        f"{query}"
    )


def build_json_augmented_prompt(query, all_battle_items, json_records):
    battle_list = format_battle_set(all_battle_items)

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


def build_raw_text_augmented_prompt(query, all_battle_items, raw_text_records):
    battle_list = format_battle_set(all_battle_items)

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

def parse_model_battle_qids(answer):
    """Parse the model response and return its ranked list of battle QIDs."""
    if isinstance(answer, dict):
        parsed_answer = answer
    elif isinstance(answer, str):
        answer_text = answer.strip()

        if answer_text.startswith("```"):
            lines = answer_text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            answer_text = "\n".join(lines).strip()

        parsed_answer = json.loads(answer_text)
    else:
        raise ValueError(
            "Model answer must be a JSON string or a dictionary."
        )

    if not isinstance(parsed_answer, dict):
        raise ValueError("Model answer JSON must be an object.")

    battles = parsed_answer.get("battles")

    if not isinstance(battles, list):
        raise ValueError(
            "Model answer must contain a 'battles' list of QIDs."
        )

    battle_qids = []
    seen_qids = set()

    for item in battles:
        # The documented output format uses QID strings. Accepting a {qid: ...}
        # object as well makes parsing resilient without falling back to names.
        if isinstance(item, str):
            qid = item.strip()
        elif isinstance(item, dict) and isinstance(item.get("qid"), str):
            qid = item["qid"].strip()
        else:
            raise ValueError(
                "Each item in 'battles' must be a QID string or an object "
                "containing a string 'qid' field."
            )

        if not qid.startswith("Q") or not qid[1:].isdigit():
            raise ValueError(f"Invalid battle QID in model answer: {qid!r}")

        if qid not in seen_qids:
            battle_qids.append(qid)
            seen_qids.add(qid)

    return battle_qids


def evaluate_answer(answer, gold_battle_qids):
    """Calculate set-based classification metrics using QIDs only."""
    try:
        model_battle_qids = parse_model_battle_qids(answer)
        model_set = set(model_battle_qids)
        gold_set = {
            qid.strip()
            for qid in gold_battle_qids
            if isinstance(qid, str)
        }

        true_positives = len(model_set & gold_set)
        precision = (
            true_positives / len(model_set)
            if model_set
            else 0.0
        )
        recall = (
            true_positives / len(gold_set)
            if gold_set
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        return {
            "model_battle_qids": model_battle_qids,
            "gold_battle_qids": sorted(gold_set),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "exact_match": model_set == gold_set,
        }
    except Exception as e:
        return {
            "model_battle_qids": [],
            "gold_battle_qids": list(gold_battle_qids),
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
    output_file,
    json_dir=None
):
    query_id = query_data["query_id"]
    query = query_data["generation_query"]

    all_battle_items = get_all_battle_items(query_data)
    all_battles = get_all_battle_names(query_data)
    gold_battles = get_gold_battle_names(query_data)
    gold_battle_qids = list(query_data.get("gold_battles", {}).keys())
    qid_to_name = {
        item["qid"]: item["name"]
        for item in all_battle_items
    }

    if experiment_type == "baseline":
        system_message = build_baseline_system_message()
        prompt = build_baseline_prompt(
            query=query,
            all_battle_items=all_battle_items
        )
        context_file_paths = None

    elif experiment_type in {"json_augmented", "json_kb1", "json_kb2"}:
        if json_dir is None:
            json_dir = JSON_KB1_DIR

        json_records, context_file_paths = load_json_records_for_query(
            query_data,
            json_dir=json_dir
        )
        system_message = build_json_augmented_system_message()
        prompt = build_json_augmented_prompt(
            query=query,
            all_battle_items=all_battle_items,
            json_records=json_records
        )

    elif experiment_type == "raw_text_augmented":
        raw_text_records, context_file_paths = load_raw_texts_for_query(query_data)
        system_message = build_raw_text_augmented_system_message()
        prompt = build_raw_text_augmented_prompt(
            query=query,
            all_battle_items=all_battle_items,
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
        gold_battle_qids=gold_battle_qids
    )

    model_battle_qids = evaluation["model_battle_qids"]
    model_battles = [
        qid_to_name.get(qid, qid)
        for qid in model_battle_qids
    ]

    result = {
        "experiment_type": experiment_type,
        "query_id": query_id,
        "query": query,
        "query_type": query_data.get("query_type"),
        "query_structure": query_data.get("query_structure"),
        "query_difficulty": query_data.get("query_difficulty"),
        "all_battles": all_battles,
        "gold_battles": gold_battles,
        "gold_battle_qids": gold_battle_qids,
        "closed_set_negatives": list(query_data.get("closed_set_negatives", {}).values()),
        "closed_set_negative_qids": list(query_data.get("closed_set_negatives", {}).keys()),
        "context_file_paths": context_file_paths,
        "system_message": system_message,
        "prompt": prompt,
        "answer": answer,
        "model_battle_qids": model_battle_qids,
        "model_battles": model_battles,
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
        "model_battle_qids": [],
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
            "model_battle_qids": result["model_battle_qids"],
            "model_battles": result["model_battles"],
            "gold_battle_qids": result["gold_battle_qids"],
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

    if "json_kb1" in final_reports and "json_kb2" in final_reports:
        comparison["kb2_minus_kb1"] = {
            metric_name: (
                final_reports["json_kb2"]["Final_results"][metric_name]
                - final_reports["json_kb1"]["Final_results"][metric_name]
            )
            for metric_name in metric_names
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


def plot_metric_kb1_kb2(
    metric_name: str,
    file: Path = Path(KB1_KB2_RESULTS_DIR) / "final_reports" / "comparison_final_report.json",
) -> None:
    """
    Plot one oracle-generation metric for all four configurations.

    The input is the comparison report produced by
    ``run_oracle_experiment_kb1_kb2()``.
    """
    file = Path(file)

    with file.open("r", encoding="utf-8") as f:
        report = json.load(f)

    report_configurations = report.get("configurations")

    if not isinstance(report_configurations, dict):
        raise ValueError(
            f"Missing 'configurations' object in report: {file}"
        )

    configurations = [
        ("baseline", "Baseline", "green"),
        ("raw_text_augmented", "Raw text", "blue"),
        ("json_kb1", "JSON KB1", "orange"),
        ("json_kb2", "JSON KB2", "darkred"),
    ]
    labels = []
    values = []
    colors = []

    for configuration_key, display_name, color in configurations:
        if configuration_key not in report_configurations:
            raise ValueError(
                f"Missing configuration {configuration_key!r} in report: "
                f"{file}"
            )

        configuration_results = report_configurations[configuration_key]

        if "Final_results" in configuration_results:
            configuration_results = configuration_results["Final_results"]

        if metric_name not in configuration_results:
            raise ValueError(
                f"Metric {metric_name!r} is missing for "
                f"{configuration_key!r} in report: {file}"
            )

        labels.append(display_name)
        values.append(float(configuration_results[metric_name]))
        colors.append(color)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    bars = ax.bar(labels, values, color=colors, width=0.65)
    ax.bar_label(
        bars,
        fmt="%.3f",
        padding=3,
        fontsize=9,
    )

    ax.set_ylabel(metric_name)
    ax.set_title(f"Oracle-context generation: {metric_name}")

    if metric_name.startswith("exact"):
        ax.set_ylim(0.3, 0.9)
    else:
        ax.set_ylim(0.4, 1.05)

    ax.grid(axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout()
    plt.show()


def plot_metric_common_rare(
    metric_name: str = "f1",
    common_file: Path = (
        Path(COMMON_RARE_RESULTS_DIR)
        / "oracle_generation_average_metrics_common.json"
    ),
    rare_file: Path = (
        Path(COMMON_RARE_RESULTS_DIR)
        / "oracle_generation_average_metrics_rare.json"
    ),
) -> None:
    """
    Plot one metric for common and rare oracle-generation queries.

    Each x-axis group contains four bars: baseline, raw-text augmentation,
    JSON KB1, and JSON KB2.
    """
    common_file = Path(common_file)
    rare_file = Path(rare_file)

    with common_file.open("r", encoding="utf-8") as f:
        common_results = json.load(f)

    with rare_file.open("r", encoding="utf-8") as f:
        rare_results = json.load(f)

    labels = ["Common", "Rare"]
    reports = [common_results, rare_results]
    report_files = [common_file, rare_file]
    configurations = [
        ("baseline", "Baseline", "green"),
        ("raw_text_augmented", "Raw text", "blue"),
        ("json_kb1", "JSON KB1", "orange"),
        ("json_kb2", "JSON KB2", "darkred"),
    ]

    x = np.arange(len(labels))
    width = 0.15

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for i, (configuration_key, display_name, color) in enumerate(configurations):
        values = []

        for report, report_file in zip(reports, report_files):
            report_configurations = report.get("configurations")

            if not isinstance(report_configurations, dict):
                raise ValueError(
                    f"Missing 'configurations' object in report: {report_file}"
                )

            if configuration_key not in report_configurations:
                raise ValueError(
                    f"Missing configuration {configuration_key!r} in report: "
                    f"{report_file}"
                )

            configuration_results = report_configurations[configuration_key]

            if metric_name not in configuration_results:
                raise ValueError(
                    f"Metric {metric_name!r} is missing for "
                    f"{configuration_key!r} in report: {report_file}"
                )

            values.append(float(configuration_results[metric_name]))

        offset = (i - (len(configurations) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=display_name,
            color=color,
        )
        ax.bar_label(
            bars,
            fmt="%.3f",
            padding=3,
            fontsize=8,
        )

    ax.set_ylabel(metric_name)
    ax.set_title(f"Common vs rare queries: {metric_name}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    if metric_name.startswith("exact"):
        ylim_start = 0.2
        ylim_end = 0.8
    else:
        ylim_start = 0.4
        ylim_end = 1.05

    ax.set_ylim(ylim_start, ylim_end)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()

    fig.tight_layout()
    plt.show()


def plot_metric_difficulty(
    metric_name: str = "f1",
    easy_file: Path = (
        Path(RESULTS_DIR)
        / "difficulty"
        / "oracle_generation_average_metrics_easy.json"
    ),
    medium_file: Path = (
        Path(RESULTS_DIR)
        / "difficulty"
        / "oracle_generation_average_metrics_medium.json"
    ),
    hard_file: Path = (
        Path(RESULTS_DIR)
        / "difficulty"
        / "oracle_generation_average_metrics_hard.json"
    ),
) -> None:
    """
    Plot one metric for easy, medium, and hard queries.

    Each difficulty group contains four bars:
    baseline, raw-text augmentation, JSON KB1, and JSON KB2.
    """
    report_files = [
        Path(easy_file),
        Path(medium_file),
        Path(hard_file),
    ]

    reports = []

    for report_file in report_files:
        with report_file.open(
            "r",
            encoding="utf-8",
        ) as f:
            reports.append(json.load(f))

    labels = [
        "Easy",
        "Medium",
        "Hard",
    ]

    configurations = [
        ("baseline", "Baseline", "green"),
        (
            "raw_text_augmented",
            "Raw text",
            "blue",
        ),
        ("json_kb1", "JSON KB1", "orange"),
        ("json_kb2", "JSON KB2", "darkred"),
    ]

    x = np.arange(len(labels))
    width = 0.18

    fig, ax = plt.subplots(
        figsize=(10, 5.5)
    )

    for i, (
        configuration_key,
        display_name,
        color,
    ) in enumerate(configurations):
        values = []

        for report, report_file in zip(
            reports,
            report_files,
        ):
            report_configurations = report.get(
                "configurations"
            )

            if not isinstance(
                report_configurations,
                dict,
            ):
                raise ValueError(
                    "Missing 'configurations' object "
                    f"in report: {report_file}"
                )

            if (
                configuration_key
                not in report_configurations
            ):
                raise ValueError(
                    f"Missing configuration "
                    f"{configuration_key!r} in report: "
                    f"{report_file}"
                )

            configuration_results = (
                report_configurations[
                    configuration_key
                ]
            )

            if metric_name not in configuration_results:
                raise ValueError(
                    f"Metric {metric_name!r} is missing "
                    f"for {configuration_key!r} "
                    f"in report: {report_file}"
                )

            values.append(
                float(
                    configuration_results[
                        metric_name
                    ]
                )
            )

        offset = (
            i - (len(configurations) - 1) / 2
        ) * width

        bars = ax.bar(
            x + offset,
            values,
            width,
            label=display_name,
            color=color,
        )

        ax.bar_label(
            bars,
            fmt="%.3f",
            padding=3,
            fontsize=8,
        )

    ax.set_ylabel(metric_name)
    ax.set_title(
        f"Performance by query difficulty: "
        f"{metric_name}"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.3, 1.05)

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.35,
    )

    ax.legend()

    fig.tight_layout()
    plt.show()


def plot_metric_query_type(
    metric_name: str = "f1",
    labels: list[str] | None = None,
    query_type_results_dir: Path = (
        Path(RESULTS_DIR) / "query_type"
    ),
) -> None:
    """
    Plot one metric for configurable query types.

    Each query-type group contains four bars:

    - baseline
    - raw_text_augmented
    - json_kb1
    - json_kb2
    """
    if labels is None:
        labels = [
            "Participant-related",
            "Similarity",
            "Narrative / Tactical",
            "Temporal",
            "Spatial / Terrain",
            "Outcome-related",
            "Weapon / Unit-related",
        ]

    # Remove duplicates while preserving order.
    labels = list(dict.fromkeys(labels))

    if not labels:
        raise ValueError(
            "At least one query-type label must be provided."
        )

    query_type_results_dir = Path(
        query_type_results_dir
    )

    reports = []
    report_files = []

    for query_type in labels:
        query_type_slug = (
            query_type
            .strip()
            .lower()
            .replace(" / ", "_")
            .replace("/", "_")
            .replace(" ", "_")
            .replace("-", "_")
        )

        while "__" in query_type_slug:
            query_type_slug = (
                query_type_slug.replace(
                    "__",
                    "_",
                )
            )

        report_file = (
            query_type_results_dir
            / (
                "oracle_generation_average_metrics_"
                f"{query_type_slug}.json"
            )
        )

        if not report_file.exists():
            raise FileNotFoundError(
                f"Average-metrics report not found for "
                f"{query_type!r}: {report_file}"
            )

        with report_file.open(
            "r",
            encoding="utf-8",
        ) as f:
            reports.append(json.load(f))

        report_files.append(report_file)

    configurations = [
        ("baseline", "Baseline", "green"),
        (
            "raw_text_augmented",
            "Raw text",
            "blue",
        ),
        ("json_kb1", "JSON KB1", "orange"),
        ("json_kb2", "JSON KB2", "darkred"),
    ]

    x = np.arange(len(labels))
    width = 0.18

    figure_width = max(
        8,
        len(labels) * 2.2,
    )

    fig, ax = plt.subplots(
        figsize=(figure_width, 5.5)
    )

    for i, (
        configuration_key,
        display_name,
        color,
    ) in enumerate(configurations):
        values = []

        for report, report_file in zip(
            reports,
            report_files,
        ):
            report_configurations = report.get(
                "configurations"
            )

            if not isinstance(
                report_configurations,
                dict,
            ):
                raise ValueError(
                    "Missing 'configurations' object "
                    f"in report: {report_file}"
                )

            if (
                configuration_key
                not in report_configurations
            ):
                raise ValueError(
                    f"Missing configuration "
                    f"{configuration_key!r} in report: "
                    f"{report_file}"
                )

            configuration_results = (
                report_configurations[
                    configuration_key
                ]
            )

            if metric_name not in configuration_results:
                raise ValueError(
                    f"Metric {metric_name!r} is missing "
                    f"for {configuration_key!r} "
                    f"in report: {report_file}"
                )

            values.append(
                float(
                    configuration_results[
                        metric_name
                    ]
                )
            )

        offset = (
            i - (len(configurations) - 1) / 2
        ) * width

        bars = ax.bar(
            x + offset,
            values,
            width,
            label=display_name,
            color=color,
        )

        ax.bar_label(
            bars,
            fmt="%.3f",
            padding=3,
            fontsize=8,
        )

    ax.set_ylabel(metric_name)

    ax.set_title(
        f"Performance by query type: {metric_name}"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        labels,
        rotation=15 if len(labels) > 3 else 0,
        ha="right" if len(labels) > 3 else "center",
    )

    if metric_name.startswith("exact"):
        ax.set_ylim(0.1, 1.05)
    else:
        ax.set_ylim(0.5, 1.05)

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.35,
    )

    ax.legend()

    fig.tight_layout()
    plt.show()


def run_oracle_experiment_kb1_kb2():
    """
    Compare baseline, raw-text augmentation, JSON KB1, and JSON KB2.

    Every configuration receives the same query and closed candidate set.
    The two JSON configurations differ only in the directory from which their
    battle records are loaded. Detailed per-query results, one final report per
    configuration, and one comparison report are written below
    ``results/oracle_generation/kb1_vs_kb2``.
    """
    details_dir = os.path.join(KB1_KB2_RESULTS_DIR, "details")
    final_reports_dir = os.path.join(KB1_KB2_RESULTS_DIR, "final_reports")

    os.makedirs(details_dir, exist_ok=True)
    os.makedirs(final_reports_dir, exist_ok=True)

    required_paths = {
        "Queries file": Path(QUERIES_FILE),
        "JSON KB1 directory": Path(JSON_KB1_DIR),
        "JSON KB2 directory": Path(JSON_KB2_DIR),
        "Raw text directory": Path(RAW_TEXT_DIR),
    }

    for label, path in required_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    queries = load_json_file(QUERIES_FILE)

    # Optional filters:
    # queries = _filter_queries_by_rarity(queries, rarity="common")
    # queries = _filter_queries_by_difficulty(queries, difficulty="hard")
    # queries = _filter_queries_by_type(
    #     queries,
    #     query_type="Narrative / Tactical"
    # )

    print(f"Queries loaded: {len(queries)}")
    print(f"JSON KB1 directory: {JSON_KB1_DIR}")
    print(f"JSON KB2 directory: {JSON_KB2_DIR}")
    print(f"Raw text directory: {RAW_TEXT_DIR}")
    print("Loading LLM...")

    llm = load_llm()

    experiment_types = [
        "baseline",
        "raw_text_augmented",
        "json_kb1",
        "json_kb2",
    ]
    json_dirs = {
        "json_kb1": JSON_KB1_DIR,
        "json_kb2": JSON_KB2_DIR,
    }
    results_by_experiment = {
        experiment_type: []
        for experiment_type in experiment_types
    }

    for query_data in queries:
        query_id = query_data["query_id"]
        all_battles = get_all_battle_names(query_data)

        print("\n" + "=" * 80)
        print(f"Running query {query_id}")
        print(f"Candidate battles: {len(all_battles)}")
        print("Candidate battle names:")
        for battle in all_battles:
            print(f"  - {battle}")

        for experiment_type in experiment_types:
            print(f"\n  Running {experiment_type}...")

            output_file = os.path.join(
                details_dir,
                f"{query_id}_{experiment_type}_answer.json"
            )

            try:
                result = run_single_experiment(
                    llm=llm,
                    experiment_type=experiment_type,
                    query_data=query_data,
                    output_file=output_file,
                    json_dir=json_dirs.get(experiment_type)
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
            final_reports_dir,
            f"{experiment_type}_final_report.json"
        )
        save_json_to_file(report, report_file)
        print_final_report(experiment_type, report)

    comparison_report = build_comparison_report(final_reports)
    save_json_to_file(
        comparison_report,
        os.path.join(final_reports_dir, "comparison_final_report.json")
    )

    print("\n" + "=" * 80)
    print("KB1 vs KB2 ORACLE EXPERIMENT FINISHED")
    print("=" * 80)
    print(
        f"\n{'Configuration':<22}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
        f"{'Exact Match':>14}"
    )
    print("-" * 72)

    for experiment_type in experiment_types:
        final_results = final_reports[experiment_type]["Final_results"]
        print(
            f"{experiment_type:<22}"
            f"{final_results['precision']:>12.4f}"
            f"{final_results['recall']:>12.4f}"
            f"{final_results['f1']:>12.4f}"
            f"{final_results['exact_match']:>14.4f}"
        )

    print("\nDone.")
    print(f"Detailed answers saved to: {details_dir}")
    print(f"Final reports saved to: {final_reports_dir}")

    return comparison_report


def run_oracle_experiment_common_rare():
    """
    Compare oracle-generation performance on common and rare queries.

    The function loads ``QUERIES_FILE`` once, filters it with
    ``_filter_queries_by_rarity()``, and evaluates the same four configurations
    for both subsets: baseline, raw text, JSON KB1, and JSON KB2.

    For each rarity, it saves:
      - one aggregate details file containing all per-query results;
      - one average-metrics file containing the four final reports and their
        comparison.

    Individual query/configuration answers are also retained in a details
    subdirectory so failed or surprising answers can be inspected directly.
    """
    required_paths = {
        "Queries file": Path(QUERIES_FILE),
        "JSON KB1 directory": Path(JSON_KB1_DIR),
        "JSON KB2 directory": Path(JSON_KB2_DIR),
        "Raw text directory": Path(RAW_TEXT_DIR),
    }

    for label, path in required_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    os.makedirs(COMMON_RARE_RESULTS_DIR, exist_ok=True)

    all_queries = load_json_file(QUERIES_FILE)
    experiment_types = [
        "baseline",
        "raw_text_augmented",
        "json_kb1",
        "json_kb2",
    ]
    json_dirs = {
        "json_kb1": JSON_KB1_DIR,
        "json_kb2": JSON_KB2_DIR,
    }

    print(f"Queries loaded: {len(all_queries)}")
    print(f"JSON KB1 directory: {JSON_KB1_DIR}")
    print(f"JSON KB2 directory: {JSON_KB2_DIR}")
    print(f"Raw text directory: {RAW_TEXT_DIR}")
    print("Loading LLM...")

    llm = load_llm()
    output_files = {}

    for rarity in ("common", "rare"):
        queries = _filter_queries_by_rarity(
            all_queries,
            rarity=rarity,
        )

        if not queries:
            raise ValueError(
                f"No queries with context_rarity={rarity!r} were found in "
                f"{QUERIES_FILE}."
            )

        per_query_details_dir = os.path.join(
            COMMON_RARE_RESULTS_DIR,
            rarity,
            "details",
        )
        os.makedirs(per_query_details_dir, exist_ok=True)

        details_file = os.path.join(
            COMMON_RARE_RESULTS_DIR,
            f"oracle_generation_details_{rarity}.json",
        )
        average_metrics_file = os.path.join(
            COMMON_RARE_RESULTS_DIR,
            f"oracle_generation_average_metrics_{rarity}.json",
        )

        print("\n" + "=" * 80)
        print(f"RUNNING {rarity.upper()} QUERIES: {len(queries)}")
        print("=" * 80)

        results_by_experiment = {
            experiment_type: []
            for experiment_type in experiment_types
        }

        for query_data in queries:
            query_id = query_data["query_id"]
            all_battles = get_all_battle_names(query_data)

            print(f"\nRunning query {query_id} ({rarity})")
            print(f"Candidate battles: {len(all_battles)}")

            for experiment_type in experiment_types:
                print(f"  Running {experiment_type}...")

                output_file = os.path.join(
                    per_query_details_dir,
                    f"{query_id}_{experiment_type}_answer.json",
                )

                try:
                    result = run_single_experiment(
                        llm=llm,
                        experiment_type=experiment_type,
                        query_data=query_data,
                        output_file=output_file,
                        json_dir=json_dirs.get(experiment_type),
                    )
                except Exception as e:
                    result = build_run_error_result(
                        experiment_type=experiment_type,
                        query_data=query_data,
                        error=e,
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

        details_report = {
            "rarity": rarity,
            "num_queries": len(queries),
            "results_by_experiment": results_by_experiment,
        }
        save_json_to_file(details_report, details_file)

        final_reports = {
            experiment_type: build_final_report(results)
            for experiment_type, results in results_by_experiment.items()
        }
        average_metrics_report = build_comparison_report(final_reports)
        average_metrics_report["rarity"] = rarity
        average_metrics_report["num_queries"] = len(queries)
        save_json_to_file(
            average_metrics_report,
            average_metrics_file,
        )

        for experiment_type in experiment_types:
            print_final_report(
                experiment_type,
                final_reports[experiment_type],
            )

        output_files[rarity] = {
            "details_file": details_file,
            "average_metrics_file": average_metrics_file,
            "per_query_details_dir": per_query_details_dir,
        }

        print(f"\n{rarity.capitalize()} details saved to: {details_file}")
        print(
            f"{rarity.capitalize()} average metrics saved to: "
            f"{average_metrics_file}"
        )

    print("\nCOMMON VS RARE ORACLE EXPERIMENT FINISHED")
    return output_files


def run_oracle_experiment_difficulty():
    """
    Compare oracle-generation performance across easy, medium, and hard queries.

    The function evaluates four configurations for every difficulty level:

    - baseline
    - raw_text_augmented
    - json_kb1
    - json_kb2

    For each difficulty level, it saves:

    - one aggregate details file;
    - one average-metrics comparison file;
    - individual query/configuration result files.

    Returns a dictionary containing all generated output paths.
    """
    difficulty_results_dir = os.path.join(
        RESULTS_DIR,
        "difficulty",
    )

    required_paths = {
        "Queries file": Path(QUERIES_FILE),
        "JSON KB1 directory": Path(JSON_KB1_DIR),
        "JSON KB2 directory": Path(JSON_KB2_DIR),
        "Raw text directory": Path(RAW_TEXT_DIR),
    }

    for label, path in required_paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"{label} not found: {path}"
            )

    os.makedirs(
        difficulty_results_dir,
        exist_ok=True,
    )

    all_queries = load_json_file(QUERIES_FILE)

    difficulty_levels = [
        "easy",
        "medium",
        "hard",
    ]

    experiment_types = [
        "baseline",
        "raw_text_augmented",
        "json_kb1",
        "json_kb2",
    ]

    json_dirs = {
        "json_kb1": JSON_KB1_DIR,
        "json_kb2": JSON_KB2_DIR,
    }

    print(f"Queries loaded: {len(all_queries)}")
    print(f"JSON KB1 directory: {JSON_KB1_DIR}")
    print(f"JSON KB2 directory: {JSON_KB2_DIR}")
    print(f"Raw text directory: {RAW_TEXT_DIR}")
    print("Loading LLM...")

    llm = load_llm()
    output_files = {}

    for difficulty in difficulty_levels:
        queries = _filter_queries_by_difficulty(
            all_queries,
            difficulty=difficulty,
        )

        if not queries:
            raise ValueError(
                f"No queries with query_difficulty={difficulty!r} "
                f"were found in {QUERIES_FILE}."
            )

        per_query_details_dir = os.path.join(
            difficulty_results_dir,
            difficulty,
            "details",
        )

        os.makedirs(
            per_query_details_dir,
            exist_ok=True,
        )

        details_file = os.path.join(
            difficulty_results_dir,
            f"oracle_generation_details_{difficulty}.json",
        )

        average_metrics_file = os.path.join(
            difficulty_results_dir,
            f"oracle_generation_average_metrics_{difficulty}.json",
        )

        print("\n" + "=" * 80)
        print(
            f"RUNNING {difficulty.upper()} QUERIES: "
            f"{len(queries)}"
        )
        print("=" * 80)

        results_by_experiment = {
            experiment_type: []
            for experiment_type in experiment_types
        }

        for query_data in queries:
            query_id = query_data["query_id"]
            all_battles = get_all_battle_names(query_data)

            print(
                f"\nRunning query {query_id} "
                f"({difficulty})"
            )
            print(
                f"Candidate battles: {len(all_battles)}"
            )

            for experiment_type in experiment_types:
                print(
                    f"  Running {experiment_type}..."
                )

                output_file = os.path.join(
                    per_query_details_dir,
                    f"{query_id}_{experiment_type}_answer.json",
                )

                try:
                    result = run_single_experiment(
                        llm=llm,
                        experiment_type=experiment_type,
                        query_data=query_data,
                        output_file=output_file,
                        json_dir=json_dirs.get(
                            experiment_type
                        ),
                    )

                except Exception as e:
                    result = build_run_error_result(
                        experiment_type=experiment_type,
                        query_data=query_data,
                        error=e,
                    )

                    save_json_to_file(
                        result,
                        output_file,
                    )

                results_by_experiment[
                    experiment_type
                ].append(result)

                metrics = result["metrics"]

                print(
                    f"    "
                    f"P={metrics['precision']:.4f} "
                    f"R={metrics['recall']:.4f} "
                    f"F1={metrics['f1']:.4f} "
                    f"EM={metrics['exact_match']}"
                )

                if metrics.get("run_error") is not None:
                    print(
                        f"    Run error: "
                        f"{metrics['run_error']}"
                    )

                if metrics.get("parse_error") is not None:
                    print(
                        f"    Parse error: "
                        f"{metrics['parse_error']}"
                    )

        # Save all detailed results for this difficulty.
        details_report = {
            "difficulty": difficulty,
            "num_queries": len(queries),
            "results_by_experiment": (
                results_by_experiment
            ),
        }

        save_json_to_file(
            details_report,
            details_file,
        )

        # Calculate average metrics for every configuration.
        final_reports = {
            experiment_type: build_final_report(
                results
            )
            for experiment_type, results
            in results_by_experiment.items()
        }

        average_metrics_report = (
            build_comparison_report(final_reports)
        )

        average_metrics_report[
            "difficulty"
        ] = difficulty

        average_metrics_report[
            "num_queries"
        ] = len(queries)

        save_json_to_file(
            average_metrics_report,
            average_metrics_file,
        )

        for experiment_type in experiment_types:
            print_final_report(
                experiment_type,
                final_reports[experiment_type],
            )

        output_files[difficulty] = {
            "details_file": details_file,
            "average_metrics_file": (
                average_metrics_file
            ),
            "per_query_details_dir": (
                per_query_details_dir
            ),
        }

        print(
            f"\n{difficulty.capitalize()} details "
            f"saved to: {details_file}"
        )

        print(
            f"{difficulty.capitalize()} average metrics "
            f"saved to: {average_metrics_file}"
        )

    print(
        "\nDIFFICULTY ORACLE EXPERIMENT FINISHED"
    )

    return output_files


def run_oracle_experiment_query_type(
    labels: list[str] | None = None,
    similarity_queries_file: Path = Path(
        "data/queries/queries_similarity_only.json"
    ),
):
    """
    Compare oracle-generation performance across configurable query types.

    Similarity queries are loaded from ``similarity_queries_file``.
    All other query types are loaded from ``QUERIES_FILE``.
    """
    if labels is None:
        labels = [
            "Participant-related",
            "Similarity",
            "Narrative / Tactical",
            "Temporal",
            "Spatial / Terrain",
            "Outcome-related",
            "Weapon / Unit-related",
        ]

    # Remove duplicate labels while preserving order.
    labels = list(dict.fromkeys(labels))

    if not labels:
        raise ValueError(
            "At least one query-type label must be provided."
        )

    query_type_results_dir = os.path.join(
        RESULTS_DIR,
        "query_type",
    )

    required_paths = {
        "Main queries file": Path(QUERIES_FILE),
        "JSON KB1 directory": Path(JSON_KB1_DIR),
        "JSON KB2 directory": Path(JSON_KB2_DIR),
        "Raw text directory": Path(RAW_TEXT_DIR),
    }

    # The separate similarity file is required only when
    # Similarity is included in the selected labels.
    if "Similarity" in labels:
        required_paths[
            "Similarity queries file"
        ] = similarity_queries_file

    for path_label, path in required_paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"{path_label} not found: {path}"
            )

    os.makedirs(
        query_type_results_dir,
        exist_ok=True,
    )

    main_queries = load_json_file(
        QUERIES_FILE
    )

    similarity_queries = []

    if "Similarity" in labels:
        similarity_queries = load_json_file(
            similarity_queries_file
        )

    experiment_types = [
        "baseline",
        "raw_text_augmented",
        "json_kb1",
        "json_kb2",
    ]

    json_dirs = {
        "json_kb1": JSON_KB1_DIR,
        "json_kb2": JSON_KB2_DIR,
    }

    print(
        f"Main queries loaded: "
        f"{len(main_queries)}"
    )

    if "Similarity" in labels:
        print(
            f"Similarity queries loaded: "
            f"{len(similarity_queries)}"
        )
        print(
            f"Similarity queries file: "
            f"{similarity_queries_file}"
        )

    print(f"Selected query types: {labels}")
    print(f"JSON KB1 directory: {JSON_KB1_DIR}")
    print(f"JSON KB2 directory: {JSON_KB2_DIR}")
    print(f"Raw text directory: {RAW_TEXT_DIR}")
    print("Loading LLM...")

    llm = load_llm()
    output_files = {}

    for query_type in labels:
        if query_type == "Similarity":
            query_source = similarity_queries
        else:
            query_source = main_queries

        queries = _filter_queries_by_type(
            query_source,
            query_type=query_type,
        )

        if not queries:
            source_file = (
                similarity_queries_file
                if query_type == "Similarity"
                else QUERIES_FILE
            )

            raise ValueError(
                f"No queries with query_type={query_type!r} "
                f"were found in {source_file}."
            )

        query_type_slug = (
            query_type
            .strip()
            .lower()
            .replace(" / ", "_")
            .replace("/", "_")
            .replace(" ", "_")
            .replace("-", "_")
        )

        while "__" in query_type_slug:
            query_type_slug = (
                query_type_slug.replace(
                    "__",
                    "_",
                )
            )

        per_query_details_dir = os.path.join(
            query_type_results_dir,
            query_type_slug,
            "details",
        )

        os.makedirs(
            per_query_details_dir,
            exist_ok=True,
        )

        details_file = os.path.join(
            query_type_results_dir,
            (
                "oracle_generation_details_"
                f"{query_type_slug}.json"
            ),
        )

        average_metrics_file = os.path.join(
            query_type_results_dir,
            (
                "oracle_generation_average_metrics_"
                f"{query_type_slug}.json"
            ),
        )

        print("\n" + "=" * 80)
        print(
            f"RUNNING QUERY TYPE: {query_type} "
            f"({len(queries)} queries)"
        )
        print("=" * 80)

        results_by_experiment = {
            experiment_type: []
            for experiment_type in experiment_types
        }

        for query_data in queries:
            query_id = query_data["query_id"]

            all_battles = get_all_battle_names(
                query_data
            )

            print(
                f"\nRunning query {query_id} "
                f"({query_type})"
            )
            print(
                f"Candidate battles: "
                f"{len(all_battles)}"
            )

            for experiment_type in experiment_types:
                print(
                    f"  Running {experiment_type}..."
                )

                output_file = os.path.join(
                    per_query_details_dir,
                    (
                        f"{query_id}_"
                        f"{experiment_type}_answer.json"
                    ),
                )

                try:
                    result = run_single_experiment(
                        llm=llm,
                        experiment_type=experiment_type,
                        query_data=query_data,
                        output_file=output_file,
                        json_dir=json_dirs.get(
                            experiment_type
                        ),
                    )

                except Exception as e:
                    result = build_run_error_result(
                        experiment_type=experiment_type,
                        query_data=query_data,
                        error=e,
                    )

                    save_json_to_file(
                        result,
                        output_file,
                    )

                results_by_experiment[
                    experiment_type
                ].append(result)

                metrics = result["metrics"]

                print(
                    f"    "
                    f"P={metrics['precision']:.4f} "
                    f"R={metrics['recall']:.4f} "
                    f"F1={metrics['f1']:.4f} "
                    f"EM={metrics['exact_match']}"
                )

                if metrics.get(
                    "run_error"
                ) is not None:
                    print(
                        f"    Run error: "
                        f"{metrics['run_error']}"
                    )

                if metrics.get(
                    "parse_error"
                ) is not None:
                    print(
                        f"    Parse error: "
                        f"{metrics['parse_error']}"
                    )

        details_report = {
            "query_type": query_type,
            "query_type_slug": query_type_slug,
            "num_queries": len(queries),
            "results_by_experiment": (
                results_by_experiment
            ),
        }

        save_json_to_file(
            details_report,
            details_file,
        )

        final_reports = {
            experiment_type: build_final_report(
                results
            )
            for experiment_type, results
            in results_by_experiment.items()
        }

        average_metrics_report = (
            build_comparison_report(
                final_reports
            )
        )

        average_metrics_report[
            "query_type"
        ] = query_type

        average_metrics_report[
            "query_type_slug"
        ] = query_type_slug

        average_metrics_report[
            "num_queries"
        ] = len(queries)

        save_json_to_file(
            average_metrics_report,
            average_metrics_file,
        )

        for experiment_type in experiment_types:
            print_final_report(
                experiment_type,
                final_reports[experiment_type],
            )

        output_files[query_type] = {
            "query_type_slug": query_type_slug,
            "details_file": details_file,
            "average_metrics_file": (
                average_metrics_file
            ),
            "per_query_details_dir": (
                per_query_details_dir
            ),
        }

        print(
            f"\n{query_type} details saved to: "
            f"{details_file}"
        )

        print(
            f"{query_type} average metrics saved to: "
            f"{average_metrics_file}"
        )

    print(
        "\nQUERY-TYPE ORACLE EXPERIMENT FINISHED"
    )

    return output_files

# -------------------------
# Main
# -------------------------

def main():
    use_kb_2 = False

    if use_kb_2:
        details_dir = DETAILS_DIR_KB2
        final_reports_dir = FINAL_REPORTS_DIR_KB2
        json_dir = JSON_KB2_DIR
    else:
        details_dir = DETAILS_DIR_KB1
        final_reports_dir = FINAL_REPORTS_DIR_KB1
        json_dir = JSON_KB1_DIR

    os.makedirs(details_dir, exist_ok=True)
    os.makedirs(final_reports_dir, exist_ok=True)

    if not Path(QUERIES_FILE).exists():
        raise FileNotFoundError(f"Queries file not found: {QUERIES_FILE}")

    if not Path(json_dir).exists():
        raise FileNotFoundError(f"JSON KB directory not found: {json_dir}")

    if not Path(RAW_TEXT_DIR).exists():
        raise FileNotFoundError(f"Raw text directory not found: {RAW_TEXT_DIR}")

    queries = load_json_file(QUERIES_FILE)
    #queries = _filter_queries_by_rarity(queries, rarity="common")

    print(f"Queries loaded: {len(queries)}")
    print(f"Raw text directory: {RAW_TEXT_DIR}")
    if use_kb_2:
        print(f"Using JSON KB v2 directory: {JSON_KB2_DIR}")
    else:
        print(f"Using JSON KB v1 directory: {JSON_KB1_DIR}")
    
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
        print("Candidate battle names:")
        for battle in all_battles:
            print(f"  - {battle}")

        for experiment_type in experiment_types:
            print(f"  Running {experiment_type}...")

            output_file = os.path.join(
                details_dir,
                f"{query_id}_{experiment_type}_answer.json"
            )

            try:
                result = run_single_experiment(
                    llm=llm,
                    experiment_type=experiment_type,
                    query_data=query_data,
                    output_file=output_file,
                    json_dir=json_dir
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
            final_reports_dir,
            f"{experiment_type}_final_report.json"
        )

        save_json_to_file(report, report_file)
        print_final_report(experiment_type, report)

    comparison_report = build_comparison_report(final_reports)

    save_json_to_file(
        comparison_report,
        os.path.join(final_reports_dir, "comparison_final_report.json")
    )

    print("\nDone.")
    print(f"Detailed answers saved to: {details_dir}")
    print(f"Final reports saved to: {final_reports_dir}")


if __name__ == "__main__":
    #run_oracle_experiment_kb1_kb2()
    #plot_metric_kb1_kb2(metric_name="precision")
    #run_oracle_experiment_common_rare()
    #plot_metric_common_rare(metric_name="precision")
    #run_oracle_experiment_difficulty()
    #plot_metric_difficulty(metric_name="precision")
    #run_oracle_experiment_query_type()
    plot_metric_query_type(metric_name="precision", labels=[
        "Weapon / Unit-related",
        "Narrative / Tactical",
        "Similarity",
        "Outcome-related"])
    plot_metric_query_type(metric_name="recall", labels=[
        "Weapon / Unit-related",
        "Narrative / Tactical",
        "Similarity",
        "Outcome-related"])
    plot_metric_query_type(metric_name="f1", labels=[
        "Weapon / Unit-related",
        "Narrative / Tactical",
        "Similarity",
        "Outcome-related"])
    plot_metric_query_type(metric_name="exact_match", labels=[
        "Weapon / Unit-related",
        "Narrative / Tactical",
        "Similarity",
        "Outcome-related"])
