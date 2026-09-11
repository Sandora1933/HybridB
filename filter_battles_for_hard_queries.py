import json
from pathlib import Path


QUERIES_FILE = "data/queries/queries_all.json"
JSON_KB_DIR = "data/json_kb_v1"
OUTPUT_QIDS_FILE = "data/filtered_battles/filtered_battle_qids.json"

MIN_RETRIEVAL_DOC_CHARS = 300


def load_json_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_to_file(data, output_file):
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_nested_value(data, path, default=None):
    current = data

    for key in path:
        if not isinstance(current, dict):
            return default

        current = current.get(key)

        if current is None:
            return default

    return current


def has_non_empty_text(value):
    return isinstance(value, str) and value.strip() != ""


def has_non_empty_list(value):
    return isinstance(value, list) and any(str(item).strip() for item in value)


def has_name(record):
    name = get_nested_value(record, ["identification", "name"])
    return has_non_empty_text(name)


def has_year(record):
    """
    Supports both schema variants:
    1. metadata.time_period.year
    2. metadata.time_period.start_date.year

    Your current schema stores year inside start_date, for example:
    metadata.time_period.start_date.year = 1515
    """
    direct_year = get_nested_value(record, ["metadata", "time_period", "year"])
    start_year = get_nested_value(record, ["metadata", "time_period", "start_date", "year"])
    end_year = get_nested_value(record, ["metadata", "time_period", "end_date", "year"])

    return (
        isinstance(direct_year, int)
        or isinstance(start_year, int)
        or isinstance(end_year, int)
    )


def get_year(record):
    direct_year = get_nested_value(record, ["metadata", "time_period", "year"])
    start_year = get_nested_value(record, ["metadata", "time_period", "start_date", "year"])
    end_year = get_nested_value(record, ["metadata", "time_period", "end_date", "year"])

    for year in [direct_year, start_year, end_year]:
        if isinstance(year, int):
            return year

    return None


def participant_has_content(participant):
    if not isinstance(participant, dict):
        return False

    if has_non_empty_text(participant.get("side")):
        return True

    if has_non_empty_list(participant.get("commanders")):
        return True

    strength = participant.get("strength")
    if isinstance(strength, dict):
        if strength.get("total") is not None:
            return True
        if has_non_empty_text(strength.get("label")):
            return True

    force_composition = participant.get("force_composition")
    if isinstance(force_composition, list) and len(force_composition) > 0:
        return True

    equipment = participant.get("equipment")
    if isinstance(equipment, list) and len(equipment) > 0:
        return True

    return False


def has_participants(record):
    participants = record.get("participants")

    if not isinstance(participants, list):
        return False

    return any(participant_has_content(participant) for participant in participants)


def has_outcome(record):
    outcome = record.get("outcome")

    if not isinstance(outcome, dict):
        return False

    return (
        has_non_empty_text(outcome.get("winner"))
        or has_non_empty_text(outcome.get("result"))
        or has_non_empty_text(outcome.get("result_type"))
    )


def has_military_or_narrative_content(record):
    military_facets = record.get("military_facets", {})
    narrative_patterns = record.get("narrative_patterns", {})
    retrieval_texts = record.get("retrieval_texts", {})

    if not isinstance(military_facets, dict):
        military_facets = {}

    if not isinstance(narrative_patterns, dict):
        narrative_patterns = {}

    if not isinstance(retrieval_texts, dict):
        retrieval_texts = {}

    return (
        has_non_empty_list(military_facets.get("tactics"))
        or has_non_empty_list(military_facets.get("terrain"))
        or has_non_empty_list(military_facets.get("weapons_or_units"))
        or has_non_empty_list(narrative_patterns.get("patterns"))
        or has_non_empty_list(narrative_patterns.get("action_sequence"))
        or has_non_empty_list(narrative_patterns.get("similarity_tags"))
        or has_non_empty_text(retrieval_texts.get("narrative_summary"))
        or has_non_empty_text(retrieval_texts.get("tactical_summary"))
        or has_non_empty_text(retrieval_texts.get("action_sequence_summary"))
        or has_non_empty_text(retrieval_texts.get("similarity_query_text"))
    )


def build_retrieval_doc_text(record):
    parts = []

    name = get_nested_value(record, ["identification", "name"])
    year = get_year(record)
    conflict = get_nested_value(record, ["metadata", "conflict"])
    country_modern = get_nested_value(record, ["metadata", "location", "country_modern"])

    if has_non_empty_text(name):
        parts.append(f"Battle: {name}")

    if year is not None:
        parts.append(f"Year: {year}")

    if has_non_empty_text(conflict):
        parts.append(f"Conflict: {conflict}")

    if has_non_empty_text(country_modern):
        parts.append(f"Modern country: {country_modern}")

    outcome = record.get("outcome", {})
    if isinstance(outcome, dict):
        for field in ["winner", "result", "result_type"]:
            value = outcome.get(field)
            if has_non_empty_text(value):
                parts.append(f"{field}: {value}")

    military_facets = record.get("military_facets", {})
    if isinstance(military_facets, dict):
        for field in [
            "battle_type",
            "domain",
            "terrain",
            "weapons_or_units",
            "tactics",
            "special_features",
        ]:
            values = military_facets.get(field)
            if has_non_empty_list(values):
                clean_values = [str(v).strip() for v in values if str(v).strip()]
                parts.append(f"{field}: {'; '.join(clean_values)}")

    narrative_patterns = record.get("narrative_patterns", {})
    if isinstance(narrative_patterns, dict):
        for field in ["patterns", "action_sequence", "similarity_tags"]:
            values = narrative_patterns.get(field)
            if has_non_empty_list(values):
                clean_values = [str(v).strip() for v in values if str(v).strip()]
                parts.append(f"{field}: {'; '.join(clean_values)}")

    retrieval_texts = record.get("retrieval_texts", {})
    if isinstance(retrieval_texts, dict):
        for field in [
            "short_summary",
            "tactical_summary",
            "narrative_summary",
            "action_sequence_summary",
            "similarity_query_text",
        ]:
            value = retrieval_texts.get(field)
            if has_non_empty_text(value):
                parts.append(f"{field}: {value}")

    return "\n".join(parts)


def has_sufficient_retrieval_doc_length(record):
    retrieval_doc_text = build_retrieval_doc_text(record)
    return len(retrieval_doc_text.strip()) >= MIN_RETRIEVAL_DOC_CHARS


def get_qids_used_in_queries(queries_file):
    queries = load_json_file(queries_file)
    used_qids = set()

    for query in queries:
        gold_battles = query.get("gold_battles", {})
        closed_set_negatives = query.get("closed_set_negatives", {})

        if isinstance(gold_battles, dict):
            used_qids.update(gold_battles.keys())

        if isinstance(closed_set_negatives, dict):
            used_qids.update(closed_set_negatives.keys())

    return used_qids


def passes_filters(record):
    return (
        has_name(record)
        and has_year(record)
        and has_participants(record)
        and has_outcome(record)
        and has_military_or_narrative_content(record)
        and has_sufficient_retrieval_doc_length(record)
    )


def filter_battles():
    queries_path = Path(QUERIES_FILE)
    json_kb_path = Path(JSON_KB_DIR)

    if not queries_path.exists():
        raise FileNotFoundError(f"Queries file not found: {QUERIES_FILE}")

    if not json_kb_path.exists():
        raise FileNotFoundError(f"JSON KB directory not found: {JSON_KB_DIR}")

    json_files = sorted(json_kb_path.rglob("*.json"))
    used_qids = get_qids_used_in_queries(QUERIES_FILE)
    filtered_qids = []

    invalid_json_count = 0
    excluded_used_qids_count = 0
    failed_filters_count = 0

    print(f"Total JSON files in knowledge base: {len(json_files)}")
    print(f"QIDs used in queries_all.json: {len(used_qids)}")
    print(f"Minimum retrieval doc length: {MIN_RETRIEVAL_DOC_CHARS} characters")

    for json_file in json_files:
        qid = json_file.stem

        if qid in used_qids:
            excluded_used_qids_count += 1
            continue

        try:
            record = load_json_file(json_file)
        except Exception:
            invalid_json_count += 1
            continue

        if passes_filters(record):
            filtered_qids.append(qid)
        else:
            failed_filters_count += 1

    save_json_to_file(filtered_qids, OUTPUT_QIDS_FILE)

    print(f"Excluded because already used in queries: {excluded_used_qids_count}")
    print(f"Invalid JSON files skipped: {invalid_json_count}")
    print(f"Failed quality filters: {failed_filters_count}")
    print(f"Filtered battles: {len(filtered_qids)}")
    print(f"Saved QIDs to: {OUTPUT_QIDS_FILE}")

    return filtered_qids


if __name__ == "__main__":
    filter_battles()