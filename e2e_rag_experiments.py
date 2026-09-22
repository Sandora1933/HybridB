
import json
import os
import unicodedata
from pathlib import Path
from typing import Any, Literal

from llm import generate_answer, load_llm

QUERIES_FILE = "data/queries/queries_all.json"
RETRIEVED_DATA_DIR = "data/e2e_retrieved_data"

RETRIEVAL_REPORT_KB2 = os.path.join(
    RETRIEVED_DATA_DIR,
    "retrieval_experiment_report_kb2.json",
)

JSON_KB2_DIR = "data/json_kb_v2"
RAW_TEXT_DIR = "data/extracted_wikipedia_raw"

RESULTS_DIR = "results/e2e_rag"
TOP_K_RESULTS_DIR = os.path.join(RESULTS_DIR, "top_k")

MAX_NEW_TOKENS = 4000

E2EExperimentType = Literal["baseline", "raw-text", "json"]


def _load_json_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json_file(data, output_file):
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _format_retrieved_battles_for_report(
    retrieved_battles: dict[str, str],
) -> list[dict[str, Any]]:
    """Convert an ordered QID-to-name mapping into ranked report rows."""
    return [
        {
            "rank": rank,
            "qid": qid,
            "battle_name": battle_name,
        }
        for rank, (qid, battle_name) in enumerate(
            retrieved_battles.items(),
            start=1,
        )
    ]


def convert_ranked_battles_to_dict(
    ranked_battles: list[str],
) -> dict[str, str]:
    """
    Convert ranked battle strings into a QID-to-name dictionary.

    Example:
        "Battle of Hastings (Q83224)"

    becomes:
        {"Q83224": "Battle of Hastings"}
    """
    battles = {}

    for ranked_battle in ranked_battles:
        if not isinstance(ranked_battle, str):
            raise ValueError(
                "Every ranked battle must be a string."
            )

        # Use the final "(Q...)" because a battle name may contain
        # another parenthetical expression, such as "(1211)".
        qid_start = ranked_battle.rfind(" (Q")

        if qid_start == -1 or not ranked_battle.endswith(")"):
            raise ValueError(
                f"Could not extract battle QID from: "
                f"{ranked_battle!r}"
            )

        battle_name = ranked_battle[:qid_start].strip()
        battle_qid = ranked_battle[
            qid_start + 2:-1
        ].strip()

        if (
            not battle_qid.startswith("Q")
            or not battle_qid[1:].isdigit()
        ):
            raise ValueError(
                f"Invalid battle QID in ranking: "
                f"{ranked_battle!r}"
            )

        if not battle_name:
            raise ValueError(
                f"Missing battle name in ranking: "
                f"{ranked_battle!r}"
            )

        battles[battle_qid] = battle_name

    return battles


def retrieve_hybrid_battles(
    report_file_path: str | Path | list[dict[str, Any]],
    query_id: str,
    top_k: int,
) -> dict[str, str]:
    """
    Retrieve the top-k hybrid-ranked battles for a query.

    Returns:
        A dictionary in the following format:

        {
            "QID": "Battle name"
        }
    """
    if (
        not isinstance(top_k, int)
        or isinstance(top_k, bool)
        or top_k <= 0
    ):
        raise ValueError(
            "top_k must be a positive integer."
        )

    if isinstance(report_file_path, list):
        report = report_file_path
        report_source = "loaded retrieval report"
    else:
        report_path = Path(report_file_path)

        if not report_path.exists():
            raise FileNotFoundError(
                f"Report file not found: {report_path}"
            )

        with report_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            report = json.load(file)

        report_source = str(report_path)

    if not isinstance(report, list):
        raise ValueError(
            "The report must contain a list of query results."
        )

    query_report = next(
        (
            item
            for item in report
            if (
                isinstance(item, dict)
                and item.get("query_id") == query_id
            )
        ),
        None,
    )

    if query_report is None:
        raise KeyError(
            f"Query ID {query_id!r} was not found "
            f"in {report_source}."
        )

    results = query_report.get("results")

    if not isinstance(results, list):
        raise ValueError(
            f"Query {query_id!r} does not contain "
            "a valid 'results' list."
        )

    hybrid_result = next(
        (
            result
            for result in results
            if (
                isinstance(result, dict)
                and result.get("algorithm") == "hybrid"
            )
        ),
        None,
    )

    if hybrid_result is None:
        raise KeyError(
            f"No hybrid retrieval result was found "
            f"for query {query_id!r}."
        )

    ranking = hybrid_result.get("ranking")

    if not isinstance(ranking, list):
        raise ValueError(
            f"The hybrid result for query {query_id!r} "
            "does not contain a valid 'ranking' list."
        )

    top_ranked_battles = ranking[:top_k]

    return convert_ranked_battles_to_dict(
        top_ranked_battles
    )


def _load_json_context(
    retrieved_battles: dict[str, str],
    json_dir: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    records = []
    file_paths = {}

    for rank, (qid, battle_name) in enumerate(
        retrieved_battles.items(),
        start=1,
    ):
        json_path = Path(json_dir) / f"{qid}.json"

        if not json_path.exists():
            raise FileNotFoundError(
                f"JSON file not found for {qid}: {json_path}"
            )

        records.append({
            "retrieval_rank": rank,
            "qid": qid,
            "battle_name": battle_name,
            "record": _load_json_file(json_path),
        })
        file_paths[qid] = {
            "battle_name": battle_name,
            "path": str(json_path),
        }

    return records, file_paths


def _load_raw_text_context(
    retrieved_battles: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    records = []
    file_paths = {}

    for rank, (qid, battle_name) in enumerate(
        retrieved_battles.items(),
        start=1,
    ):
        raw_text_path = Path(RAW_TEXT_DIR) / f"{qid}.txt"

        if not raw_text_path.exists():
            raise FileNotFoundError(
                f"Raw-text file not found for {qid}: {raw_text_path}"
            )

        text = raw_text_path.read_text(encoding="utf-8")
        records.append({
            "retrieval_rank": rank,
            "qid": qid,
            "battle_name": battle_name,
            "text": text,
        })
        file_paths[qid] = {
            "battle_name": battle_name,
            "path": str(raw_text_path),
        }

    return records, file_paths


def build_baseline_system_message() -> str:
    return (
        "You are a historical battle question-answering assistant.\n\n"
        "Rules:\n"
        "- Use your own general knowledge.\n"
        "- Return the names of battles that match the question.\n"
        "- There is no predefined battle set; identify the matching battles yourself.\n"
        "- Include a battle only if you are confident that it matches the question.\n"
        "- Do not return Wikidata QIDs or any other identifiers.\n"
        "- If you are uncertain whether a battle matches, exclude it.\n"
        "- If no battles match the question, return an empty battles list.\n"
        "- Do not invent details.\n\n"
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


def build_json_augmented_system_message() -> str:
    return (
        "You are a historical battle question-answering assistant.\n\n"
        "Rules:\n"
        "- Use only the provided retrieved JSON battle records as evidence.\n"
        "- Do not use your internal knowledge or outside knowledge.\n"
        "- Return only battle QIDs that occur in the provided JSON records.\n"
        "- The records are retrieval results and may contain irrelevant battles.\n"
        "- Include a battle only if its JSON record explicitly supports that it matches the question.\n"
        "- If you are uncertain whether the JSON supports a battle, exclude it.\n"
        "- If the JSON data does not support an answer, return an empty battles list.\n"
        "- Do not invent missing details or QIDs.\n"
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


def build_raw_text_augmented_system_message() -> str:
    return (
        "You are a historical battle question-answering assistant.\n\n"
        "Rules:\n"
        "- Use only the provided retrieved raw battle texts as evidence.\n"
        "- Do not use your internal knowledge or outside knowledge.\n"
        "- Return only battle QIDs that occur in the provided raw-text records.\n"
        "- The records are retrieval results and may contain irrelevant battles.\n"
        "- Include a battle only if its raw text explicitly supports that it matches the question.\n"
        "- If you are uncertain whether the raw text supports a battle, exclude it.\n"
        "- If the raw-text data does not support an answer, return an empty battles list.\n"
        "- Do not invent missing details or QIDs.\n"
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


def _build_baseline_prompt(query: str) -> str:
    return f"Question:\n{query}"


def _build_json_prompt(
    query: str,
    json_records: list[dict[str, Any]],
) -> str:
    return (
        f"Question:\n{query}\n\n"
        "Retrieved JSON battle records:\n"
        + json.dumps(json_records, indent=2, ensure_ascii=False)
    )


def _build_raw_text_prompt(
    query: str,
    raw_text_records: list[dict[str, Any]],
) -> str:
    context_parts = []

    for record in raw_text_records:
        context_parts.append(
            f"Retrieval rank: {record['retrieval_rank']}\n"
            f"QID: {record['qid']}\n"
            f"Battle: {record['battle_name']}\n\n"
            f"Raw text:\n{record['text']}"
        )

    return (
        f"Question:\n{query}\n\n"
        "Retrieved raw-text battle records:\n"
        + "\n\n---\n\n".join(context_parts)
    )


def _normalize_battle_name(name: str) -> str:
    decomposed = unicodedata.normalize("NFKD", name.casefold())
    without_accents = "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    )
    alphanumeric = "".join(
        character if character.isalnum() else " "
        for character in without_accents
    )
    return " ".join(alphanumeric.split())


def _build_battle_name_index(
    json_dir: str | Path,
) -> dict[str, dict[str, set[str]]]:
    """Build separate primary- and alternative-name indexes."""
    json_directory = Path(json_dir)
    json_files = sorted(json_directory.glob("*.json"))

    if not json_files:
        raise ValueError(
            f"No JSON battle records found in: {json_directory}"
        )

    primary_name_index: dict[str, set[str]] = {}
    alternative_name_index: dict[str, set[str]] = {}

    for json_path in json_files:
        record = _load_json_file(json_path)
        record_qid = record.get("battle_id")
        qid = (
            record_qid.strip()
            if isinstance(record_qid, str)
            else json_path.stem
        )

        if not qid.startswith("Q") or not qid[1:].isdigit():
            raise ValueError(
                f"Could not determine a valid battle QID for: {json_path}"
            )

        identification = record.get("identification", {})
        if isinstance(identification, dict):
            primary_name = identification.get("name")
            alternative_names = identification.get(
                "alternative_names",
                [],
            )

            if isinstance(primary_name, str):
                normalized_primary_name = _normalize_battle_name(
                    primary_name
                )

                if normalized_primary_name:
                    primary_name_index.setdefault(
                        normalized_primary_name,
                        set(),
                    ).add(qid)

            if isinstance(alternative_names, list):
                for alternative_name in alternative_names:
                    if not isinstance(alternative_name, str):
                        continue

                    normalized_alternative_name = (
                        _normalize_battle_name(alternative_name)
                    )

                    if normalized_alternative_name:
                        alternative_name_index.setdefault(
                            normalized_alternative_name,
                            set(),
                        ).add(qid)

    if not primary_name_index and not alternative_name_index:
        raise ValueError(
            f"No battle names could be indexed from: {json_directory}"
        )

    return {
        "primary": primary_name_index,
        "alternative": alternative_name_index,
    }


def _parse_model_battles(answer: Any) -> list[dict[str, str | None]]:
    if isinstance(answer, dict):
        parsed_answer = answer
    elif isinstance(answer, str):
        answer_text = answer.strip()

        if answer_text.startswith("```"):
            lines = answer_text.splitlines()
            lines = lines[1:] if lines else lines

            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            answer_text = "\n".join(lines).strip()

        parsed_answer = json.loads(answer_text)
    else:
        raise ValueError(
            "Model answer must be a JSON string or dictionary."
        )

    if not isinstance(parsed_answer, dict):
        raise ValueError("Model answer JSON must be an object.")

    battles = parsed_answer.get("battles")

    if not isinstance(battles, list):
        raise ValueError("Model answer must contain a 'battles' list.")

    parsed_battles = []

    for item in battles:
        if isinstance(item, str):
            if item.startswith("Q") and item[1:].isdigit():
                qid = item
                name = None
            else:
                qid = None
                name = item.strip()
        elif isinstance(item, dict):
            qid = item.get("qid")
            name = item.get("name")

            if qid is not None and not isinstance(qid, str):
                raise ValueError("Battle 'qid' must be a string or null.")

            if name is not None and not isinstance(name, str):
                raise ValueError("Battle 'name' must be a string or null.")

            qid = qid.strip() if isinstance(qid, str) else None
            name = name.strip() if isinstance(name, str) else None
        else:
            raise ValueError(
                "Every item in 'battles' must be a string or object."
            )

        if qid is not None:
            if not qid.startswith("Q") or not qid[1:].isdigit():
                raise ValueError(f"Invalid battle QID: {qid!r}")

        if not qid and not name:
            raise ValueError(
                "Every predicted battle must contain a QID or name."
            )

        parsed_battles.append({
            "qid": qid,
            "name": name,
        })

    return parsed_battles


def _evaluate_baseline_answer(
    answer: Any,
    gold_battles: dict[str, str],
    battle_name_index: dict[str, dict[str, set[str]]],
) -> dict[str, Any]:
    """Resolve baseline battle names to QIDs before computing metrics."""
    try:
        parsed_battles = _parse_model_battles(answer)
        gold_ids = {qid.strip() for qid in gold_battles}
        gold_name_index: dict[str, set[str]] = {}

        for gold_qid, gold_name in gold_battles.items():
            normalized_gold_name = _normalize_battle_name(gold_name)

            if normalized_gold_name:
                gold_name_index.setdefault(
                    normalized_gold_name,
                    set(),
                ).add(gold_qid.strip())

        predicted_names = []
        resolved_qids = []
        name_resolution = []
        prediction_keys = []
        seen_keys = set()

        for battle in parsed_battles:
            if battle["qid"] is not None:
                raise ValueError(
                    "Baseline must return battle names, not Wikidata QIDs."
                )

            battle_name = (battle["name"] or "").strip()
            normalized_name = _normalize_battle_name(battle_name)

            if not normalized_name:
                raise ValueError(
                    "Every baseline prediction must contain a battle name."
                )

            gold_matches = gold_name_index.get(
                normalized_name,
                set(),
            )
            primary_matches = battle_name_index["primary"].get(
                normalized_name,
                set(),
            )
            alternative_matches = battle_name_index["alternative"].get(
                normalized_name,
                set(),
            )

            if gold_matches:
                matching_qids = gold_matches
                match_type = "gold"
            elif primary_matches:
                matching_qids = primary_matches
                match_type = "primary"
            else:
                matching_qids = alternative_matches
                match_type = (
                    "alternative"
                    if alternative_matches
                    else None
                )

            resolved_qid = (
                next(iter(matching_qids))
                if len(matching_qids) == 1
                else None
            )

            if resolved_qid is not None:
                resolution_status = "resolved"
                prediction_key = resolved_qid
            elif matching_qids:
                resolution_status = "ambiguous"
                prediction_key = f"NAME::{normalized_name}"
            else:
                resolution_status = "not_found"
                prediction_key = f"NAME::{normalized_name}"

            if prediction_key in seen_keys:
                continue

            seen_keys.add(prediction_key)
            prediction_keys.append(prediction_key)
            predicted_names.append(battle_name)

            if resolved_qid is not None:
                resolved_qids.append(resolved_qid)

            name_resolution.append({
                "battle_name": battle_name,
                "resolved_qid": resolved_qid,
                "status": resolution_status,
                "match_type": match_type,
                "candidate_qids": sorted(matching_qids),
            })

        predicted_set = set(prediction_keys)
        true_positives = len(predicted_set & gold_ids)
        precision = (
            true_positives / len(predicted_set)
            if predicted_set
            else 0.0
        )
        recall = (
            true_positives / len(gold_ids)
            if gold_ids
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        return {
            "predicted_battles": parsed_battles,
            "predicted_battle_names": predicted_names,
            "resolved_prediction_qids": resolved_qids,
            "name_resolution": name_resolution,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "exact_match": predicted_set == gold_ids,
        }
    except Exception as error:
        return {
            "predicted_battles": [],
            "predicted_battle_names": [],
            "resolved_prediction_qids": [],
            "name_resolution": [],
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "exact_match": False,
            "parse_error": str(error),
        }


def _evaluate_answer(
    answer: Any,
    gold_battles: dict[str, str],
) -> dict[str, Any]:
    try:
        predicted_battles = _parse_model_battles(answer)
        gold_ids = {qid.strip() for qid in gold_battles}
        gold_name_to_qid = {
            _normalize_battle_name(name): qid.strip()
            for qid, name in gold_battles.items()
        }

        prediction_keys = []
        resolved_qids = []
        seen_keys = set()

        for battle in predicted_battles:
            qid = battle["qid"]
            name = battle["name"]

            if qid:
                key = qid
                resolved_qid = qid
            else:
                normalized_name = _normalize_battle_name(name or "")
                resolved_qid = gold_name_to_qid.get(normalized_name)
                key = resolved_qid or f"NAME::{normalized_name}"

            if key not in seen_keys:
                prediction_keys.append(key)
                resolved_qids.append(resolved_qid)
                seen_keys.add(key)

        predicted_set = set(prediction_keys)
        true_positives = len(predicted_set & gold_ids)
        precision = (
            true_positives / len(predicted_set)
            if predicted_set
            else 0.0
        )
        recall = (
            true_positives / len(gold_ids)
            if gold_ids
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        return {
            "predicted_battles": predicted_battles,
            "resolved_prediction_qids": resolved_qids,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "exact_match": predicted_set == gold_ids,
        }
    except Exception as error:
        return {
            "predicted_battles": [],
            "resolved_prediction_qids": [],
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "exact_match": False,
            "parse_error": str(error),
        }


def _run_single_e2e_experiment(
    llm,
    experiment_type: str,
    query_data: dict[str, Any],
    output_file: str,
    retrieved_battles: dict[str, str] | None = None,
    json_dir: str | None = None,
    retrieval_source: str | None = None,
    battle_name_index: dict[str, dict[str, set[str]]] | None = None,
) -> dict[str, Any]:
    original_query = query_data["query"]
    context_query = query_data.get("generation_query", original_query)
    context_file_paths = None

    if experiment_type == "baseline":
        system_message = build_baseline_system_message()
        prompt = _build_baseline_prompt(original_query)
    elif experiment_type == "json":
        if retrieved_battles is None or json_dir is None:
            raise ValueError(
                "json requires retrieved battles and json_dir."
            )

        json_records, context_file_paths = _load_json_context(
            retrieved_battles,
            json_dir,
        )
        system_message = build_json_augmented_system_message()
        prompt = _build_json_prompt(context_query, json_records)
    elif experiment_type == "raw-text":
        if retrieved_battles is None:
            raise ValueError(
                "raw-text requires retrieved battles."
            )

        raw_records, context_file_paths = _load_raw_text_context(
            retrieved_battles
        )
        system_message = build_raw_text_augmented_system_message()
        prompt = _build_raw_text_prompt(context_query, raw_records)
    else:
        raise ValueError(f"Unknown experiment_type: {experiment_type}")

    answer = generate_answer(
        llm=llm,
        prompt=prompt,
        system_message=system_message,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    if experiment_type == "baseline":
        if battle_name_index is None:
            raise ValueError(
                "baseline requires a battle name index for QID resolution."
            )

        evaluation = _evaluate_baseline_answer(
            answer,
            query_data.get("gold_battles", {}),
            battle_name_index,
        )
    else:
        evaluation = _evaluate_answer(
            answer,
            query_data.get("gold_battles", {}),
        )

    result = {
        "experiment_type": experiment_type,
        "query_id": query_data["query_id"],
        "query": original_query,
        "generation_query": context_query,
        "query_type": query_data.get("query_type"),
        "query_structure": query_data.get("query_structure"),
        "query_difficulty": query_data.get("query_difficulty"),
        "context_rarity": query_data.get("context_rarity"),
        "gold_battles": query_data.get("gold_battles", {}),
        "retrieval_source": retrieval_source,
        "retrieved_battles": retrieved_battles,
        "context_file_paths": context_file_paths,
        "system_message": system_message,
        "prompt": prompt,
        "answer": answer,
        "predicted_battles": evaluation["predicted_battles"],
        "predicted_battle_names": evaluation.get(
            "predicted_battle_names",
            [],
        ),
        "resolved_prediction_qids": evaluation[
            "resolved_prediction_qids"
        ],
        "name_resolution": evaluation.get("name_resolution", []),
        "metrics": {
            "precision": evaluation["precision"],
            "recall": evaluation["recall"],
            "f1": evaluation["f1"],
            "exact_match": evaluation["exact_match"],
        },
    }

    if "parse_error" in evaluation:
        result["metrics"]["parse_error"] = evaluation["parse_error"]

    _save_json_file(result, output_file)
    return result


def _build_run_error_result(
    experiment_type: str,
    query_data: dict[str, Any],
    error: Exception,
    retrieved_battles: dict[str, str] | None,
    retrieval_source: str | None,
) -> dict[str, Any]:
    return {
        "experiment_type": experiment_type,
        "query_id": query_data.get("query_id"),
        "query": query_data.get("query"),
        "generation_query": query_data.get("generation_query"),
        "query_type": query_data.get("query_type"),
        "query_structure": query_data.get("query_structure"),
        "query_difficulty": query_data.get("query_difficulty"),
        "context_rarity": query_data.get("context_rarity"),
        "gold_battles": query_data.get("gold_battles", {}),
        "retrieval_source": retrieval_source,
        "retrieved_battles": retrieved_battles,
        "answer": None,
        "predicted_battles": [],
        "predicted_battle_names": [],
        "resolved_prediction_qids": [],
        "name_resolution": [],
        "metrics": {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "exact_match": False,
            "run_error": str(error),
        },
    }


def _build_final_report(results: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = ["precision", "recall", "f1", "exact_match"]
    query_results = {}

    for result in results:
        metrics = result["metrics"]
        query_result = {
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "exact_match": metrics["exact_match"],
            "predicted_battles": result["predicted_battles"],
            "predicted_battle_names": result.get(
                "predicted_battle_names",
                [],
            ),
            "resolved_prediction_qids": result.get(
                "resolved_prediction_qids",
                [],
            ),
            "name_resolution": result.get("name_resolution", []),
            "gold_battles": result["gold_battles"],
        }

        if metrics.get("parse_error") is not None:
            query_result["parse_error"] = metrics["parse_error"]

        if metrics.get("run_error") is not None:
            query_result["run_error"] = metrics["run_error"]
            query_result["excluded_from_averages"] = True

        query_results[result["query_id"]] = query_result

    # A run error means that no valid model prediction was produced. Keep such
    # queries in the detailed report for inspection, but do not treat them as
    # zero-score predictions when calculating aggregate metrics. This includes
    # generations stopped because the maximum token limit was reached.
    evaluated_results = [
        result
        for result in results
        if result["metrics"].get("run_error") is None
    ]

    averages = {
        metric_name: (
            sum(
                result["metrics"][metric_name]
                for result in evaluated_results
            )
            / len(evaluated_results)
            if evaluated_results
            else 0.0
        )
        for metric_name in metric_names
    }
    averages.update({
        "num_queries": len(results),
        "num_evaluated_queries": len(evaluated_results),
        "num_excluded_queries": len(results) - len(evaluated_results),
        "parse_error_count": sum(
            result["metrics"].get("parse_error") is not None
            for result in results
        ),
        "run_error_count": sum(
            result["metrics"].get("run_error") is not None
            for result in results
        ),
    })

    return {
        "queries": query_results,
        "Final_results": averages,
    }


def run_e2e_experiment_top_k(
    experiment_type: E2EExperimentType,
    top_k: int | None = None,
) -> dict[str, Any]:
    """
    Run one end-to-end experiment configuration.

    Args:
        experiment_type:
            One of "baseline", "raw-text", or "json".
        top_k:
            Number of KB2 hybrid retrieval results used as context. Required
            for "raw-text" and "json". Ignored for "baseline", because the
            baseline does not use external context.

    Returns:
        The final report containing per-query and average metrics.

    Output layout:
        results/e2e_rag/top_k/baseline/
        results/e2e_rag/top_k/raw_text/top_<k>/
        results/e2e_rag/top_k/json/top_<k>/
    """
    valid_experiment_types = {"baseline", "raw-text", "json"}

    if experiment_type not in valid_experiment_types:
        raise ValueError(
            f"Unknown experiment_type {experiment_type!r}. "
            'Choose "baseline", "raw-text", or "json".'
        )

    uses_context = experiment_type != "baseline"

    if uses_context and (
        not isinstance(top_k, int)
        or isinstance(top_k, bool)
        or top_k <= 0
    ):
        raise ValueError(
            'top_k must be a positive integer for "raw-text" and "json".'
        )

    effective_top_k = top_k if uses_context else None
    retrieval_source = "kb2_hybrid" if uses_context else None

    required_paths = {
        "Queries file": Path(QUERIES_FILE),
    }

    if experiment_type == "baseline":
        required_paths.update({
            "JSON KB2 directory for name resolution": Path(JSON_KB2_DIR),
        })
    elif experiment_type == "raw-text":
        required_paths.update({
            "KB2 retrieval report": Path(RETRIEVAL_REPORT_KB2),
            "Raw-text directory": Path(RAW_TEXT_DIR),
        })
    elif experiment_type == "json":
        required_paths.update({
            "KB2 retrieval report": Path(RETRIEVAL_REPORT_KB2),
            "JSON KB2 directory": Path(JSON_KB2_DIR),
        })

    for label, path in required_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    experiment_folder = (
        "raw_text"
        if experiment_type == "raw-text"
        else experiment_type
    )
    experiment_dir = Path(TOP_K_RESULTS_DIR) / experiment_folder

    if uses_context:
        experiment_dir = experiment_dir / f"top_{effective_top_k}"

    details_dir = experiment_dir / "details"
    final_report_file = experiment_dir / "final_report.json"
    average_metrics_file = experiment_dir / "average_metrics.json"
    retrieved_battles_file = (
        experiment_dir / "retrieved_battles.json"
        if uses_context
        else None
    )
    details_dir.mkdir(parents=True, exist_ok=True)

    queries = _load_json_file(QUERIES_FILE)
    retrieval_report = (
        _load_json_file(RETRIEVAL_REPORT_KB2)
        if uses_context
        else None
    )
    battle_name_index = (
        _build_battle_name_index(JSON_KB2_DIR)
        if experiment_type == "baseline"
        else None
    )

    print(f"Queries loaded: {len(queries)}")
    print(f"Experiment type: {experiment_type}")

    if uses_context:
        print(f"Retrieval source: {RETRIEVAL_REPORT_KB2}")
        print(f"Context size: top-{effective_top_k}")
    elif battle_name_index is not None:
        indexed_names = set(battle_name_index["primary"])
        indexed_names.update(battle_name_index["alternative"])
        indexed_qids = {
            qid
            for index in battle_name_index.values()
            for matching_qids in index.values()
            for qid in matching_qids
        }
        print(
            "Baseline name-resolution index: "
            f"{len(indexed_names)} names for "
            f"{len(indexed_qids)} battles"
        )
    elif top_k is not None:
        print(
            f"Ignoring top_k={top_k}: baseline does not use "
            "retrieved context."
        )

    print("Loading LLM...")
    llm = load_llm()

    results = []
    retrieved_battles_by_query = []

    for query_data in queries:
        query_id = query_data["query_id"]
        retrieved_battles = None

        if uses_context:
            retrieved_battles = retrieve_hybrid_battles(
                retrieval_report,
                query_id,
                effective_top_k,
            )
            retrieved_battles_by_query.append({
                "query_id": query_id,
                "query": query_data.get("query"),
                "generation_query": query_data.get("generation_query"),
                "gold_battles": query_data.get("gold_battles", {}),
                "retrieved_battles": (
                    _format_retrieved_battles_for_report(
                        retrieved_battles
                    )
                ),
            })

            # Save progressively so completed retrieval results remain
            # available if a later model call is interrupted.
            _save_json_file(
                {
                    "experiment_type": experiment_type,
                    "top_k": effective_top_k,
                    "retrieval_source": retrieval_source,
                    "num_queries": len(retrieved_battles_by_query),
                    "queries": retrieved_battles_by_query,
                },
                retrieved_battles_file,
            )

        output_file = details_dir / f"{query_id}_answer.json"

        print("\n" + "=" * 80)
        print(f"Running {experiment_type} for query {query_id}")

        if uses_context:
            print(f"Retrieved battles: {len(retrieved_battles)}")

        try:
            result = _run_single_e2e_experiment(
                llm=llm,
                experiment_type=experiment_type,
                query_data=query_data,
                output_file=str(output_file),
                retrieved_battles=retrieved_battles,
                json_dir=(
                    JSON_KB2_DIR
                    if experiment_type == "json"
                    else None
                ),
                retrieval_source=retrieval_source,
                battle_name_index=battle_name_index,
            )
        except Exception as error:
            result = _build_run_error_result(
                experiment_type=experiment_type,
                query_data=query_data,
                error=error,
                retrieved_battles=retrieved_battles,
                retrieval_source=retrieval_source,
            )
            _save_json_file(result, output_file)

        results.append(result)
        metrics = result["metrics"]
        print(
            f"  P={metrics['precision']:.4f} "
            f"R={metrics['recall']:.4f} "
            f"F1={metrics['f1']:.4f} "
            f"EM={metrics['exact_match']}"
        )

        if metrics.get("parse_error") is not None:
            print(f"  Parse error: {metrics['parse_error']}")

        if metrics.get("run_error") is not None:
            print(f"  Run error: {metrics['run_error']}")

    final_report = {
        "experiment_type": experiment_type,
        "top_k": effective_top_k,
        "retrieval_source": retrieval_source,
        "name_resolution_source": (
            JSON_KB2_DIR
            if experiment_type == "baseline"
            else None
        ),
        **_build_final_report(results),
    }
    _save_json_file(final_report, final_report_file)
    _save_json_file(
        {"Final_results": final_report["Final_results"]},
        average_metrics_file,
    )

    print("\n" + "=" * 80)
    print("END-TO-END EXPERIMENT FINISHED")
    print("=" * 80)

    final_metrics = final_report["Final_results"]
    print(
        f"P={final_metrics['precision']:.4f} "
        f"R={final_metrics['recall']:.4f} "
        f"F1={final_metrics['f1']:.4f} "
        f"EM={final_metrics['exact_match']:.4f}"
    )
    print(f"Detailed responses: {details_dir}")
    print(f"Final report: {final_report_file}")
    print(f"Average metrics: {average_metrics_file}")

    if retrieved_battles_file is not None:
        print(f"Retrieved battles: {retrieved_battles_file}")

    return final_report


if __name__ == "__main__":
    run_e2e_experiment_top_k("baseline")
    run_e2e_experiment_top_k("raw-text", top_k=20)
    run_e2e_experiment_top_k("json", top_k=20)
