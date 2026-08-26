from __future__ import annotations

import json
from pathlib import Path
from typing import Any


BATTLE_JSON_DIR = Path("data/json_kb_v1")
OUTPUT_FILE = Path("data/retrieval_docs/retrieval_docs_v1.jsonl")

NUMBER_OF_BATTLES = None

# Keep retrieval docs compact enough for indexing, but rich enough for event-centric queries.
MAX_EVENT_ITEMS = 8
MAX_FIELD_CHARS = 2500


def load_json(file_path: Path) -> dict[str, Any]:
    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def clean_string(value: Any) -> str | None:
    """
    Normalize a scalar value for retrieval text.
    Returns None for empty / null values.
    """

    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    return text


def clean_list(values: Any) -> list[str]:
    """
    Return only non-empty scalar values from a list.
    This intentionally avoids flattening dictionaries.
    """

    if not isinstance(values, list):
        return []

    result: list[str] = []

    for value in values:
        if isinstance(value, (dict, list)):
            continue

        text = clean_string(value)

        if text:
            result.append(text)

    return result


def deduplicate(values: list[str]) -> list[str]:
    """
    Deduplicate strings while preserving order.
    """

    seen: set[str] = set()
    result: list[str] = []

    for value in values:
        normalized = value.casefold()

        if normalized in seen:
            continue

        seen.add(normalized)
        result.append(value)

    return result


def join_values(values: list[str]) -> str | None:
    """
    Convert a list into a semicolon-separated retrieval string.
    """

    values = deduplicate([
        value.strip()
        for value in values
        if value and value.strip()
    ])

    if not values:
        return None

    return "; ".join(values)


def trim_text(text: str, max_chars: int = MAX_FIELD_CHARS) -> str:
    """
    Keep single retrieval fields from becoming too long.
    """

    if len(text) <= max_chars:
        return text

    return text[: max_chars - 3].rstrip() + "..."


def get_nested(data: dict[str, Any], path: list[str]) -> Any:
    """
    Safe nested dictionary access.
    """

    current: Any = data

    for key in path:
        if not isinstance(current, dict):
            return None

        current = current.get(key)

    return current


def get_year(battle: dict[str, Any]) -> int | None:
    """
    Extract battle start year from either nested or older flat date schema.
    """

    start_date = get_nested(
        battle,
        ["metadata", "time_period", "start_date"],
    )

    if isinstance(start_date, dict):
        year = start_date.get("year")

        if isinstance(year, int):
            return year

        try:
            return int(year)
        except (TypeError, ValueError):
            return None

    # Fallback for old schema where year is directly under time_period.
    year = get_nested(
        battle,
        ["metadata", "time_period", "year"],
    )

    if isinstance(year, int):
        return year

    try:
        return int(year)
    except (TypeError, ValueError):
        return None


def get_location_text(battle: dict[str, Any]) -> str | None:
    """
    Build location string with historical and modern country separated.
    """

    location = battle.get("metadata", {}).get("location", {})

    if not isinstance(location, dict):
        return None

    historical_parts = [
        clean_string(location.get("name")),
        clean_string(location.get("region")),
        clean_string(location.get("country")),
    ]

    historical_parts = deduplicate([
        value
        for value in historical_parts
        if value
    ])

    modern_country = clean_string(location.get("country_modern"))

    if not historical_parts and not modern_country:
        return None

    text = ", ".join(historical_parts)

    if modern_country:
        if text:
            text += f"; modern country: {modern_country}"
        else:
            text = f"modern country: {modern_country}"

    return text


def add_text_field(
    lines: list[str],
    label: str,
    value: Any,
) -> None:
    """
    Add non-empty field to retrieval text.
    """

    if value is None:
        return

    if isinstance(value, list):
        value = join_values(clean_list(value))
    else:
        value = clean_string(value)

    if not value:
        return

    lines.append(f"{label}: {trim_text(value)}")


def get_participants(battle: dict[str, Any]) -> list[str]:
    """
    Extract strict top-level participant side names for metadata.
    """

    participants = battle.get("participants", [])

    if not isinstance(participants, list):
        return []

    result: list[str] = []

    for participant in participants:
        if not isinstance(participant, dict):
            continue

        side = clean_string(participant.get("side"))

        if side:
            result.append(side)

    return deduplicate(result)


def get_participants_text(battle: dict[str, Any]) -> list[str]:
    """
    Build richer participant descriptions for retrieval text.
    Keeps metadata participants strict, but text can include contingents/allies.
    """

    participants = battle.get("participants", [])

    if not isinstance(participants, list):
        return []

    result: list[str] = []

    for participant in participants:
        if not isinstance(participant, dict):
            continue

        side = clean_string(participant.get("side"))

        if not side:
            continue

        contingents: list[str] = []

        force_composition = participant.get("force_composition", [])

        if isinstance(force_composition, list):
            for component in force_composition:
                if not isinstance(component, dict):
                    continue

                contingent = clean_string(component.get("contingent"))

                if contingent:
                    contingents.append(contingent)

        contingents = deduplicate(contingents)

        if contingents:
            result.append(f"{side} with {', '.join(contingents)}")
        else:
            result.append(side)

    return result


def get_commanders_text(battle: dict[str, Any]) -> list[str]:
    """
    Build side-specific commander text for retrieval.
    """

    participants = battle.get("participants", [])

    if not isinstance(participants, list):
        return []

    result: list[str] = []

    for participant in participants:
        if not isinstance(participant, dict):
            continue

        side = clean_string(participant.get("side"))
        commanders = clean_list(participant.get("commanders", []))

        if side and commanders:
            result.append(f"{side}: {', '.join(commanders)}")

    return result


def format_strength_or_casualties(
    participant: dict[str, Any],
    field_name: str,
) -> str | None:
    """
    Format strength/casualty object for retrieval text.
    """

    field = participant.get(field_name)

    if not isinstance(field, dict):
        return None

    total = clean_string(field.get("total"))
    label = clean_string(field.get("label"))
    note = clean_string(field.get("note"))

    parts: list[str] = []

    if total:
        parts.append(total)

    if label:
        parts.append(label)

    text = " ".join(parts).strip()

    if note:
        if text:
            text += f" ({note})"
        else:
            text = note

    return text or None


def get_strengths_text(battle: dict[str, Any]) -> list[str]:
    """
    Build side-specific strength text.
    """

    participants = battle.get("participants", [])

    if not isinstance(participants, list):
        return []

    result: list[str] = []

    for participant in participants:
        if not isinstance(participant, dict):
            continue

        side = clean_string(participant.get("side"))
        strength = format_strength_or_casualties(participant, "strength")

        if side and strength:
            result.append(f"{side}: {strength}")

    return result


def get_casualties_text(battle: dict[str, Any]) -> list[str]:
    """
    Build side-specific casualty text.
    """

    participants = battle.get("participants", [])

    if not isinstance(participants, list):
        return []

    result: list[str] = []

    for participant in participants:
        if not isinstance(participant, dict):
            continue

        side = clean_string(participant.get("side"))
        casualties = format_strength_or_casualties(participant, "casualties")

        if side and casualties:
            result.append(f"{side}: {casualties}")

    return result


def get_force_composition_text(battle: dict[str, Any]) -> list[str]:
    """
    Build concise force composition text from participant force_composition.
    """

    participants = battle.get("participants", [])

    if not isinstance(participants, list):
        return []

    result: list[str] = []

    for participant in participants:
        if not isinstance(participant, dict):
            continue

        side = clean_string(participant.get("side"))
        force_composition = participant.get("force_composition", [])

        if not side or not isinstance(force_composition, list):
            continue

        components: list[str] = []

        for component in force_composition:
            if not isinstance(component, dict):
                continue

            contingent = clean_string(component.get("contingent"))
            size = clean_string(component.get("size"))

            unit_types: list[str] = []
            units = component.get("units", [])

            if isinstance(units, list):
                for unit in units:
                    if not isinstance(unit, dict):
                        continue

                    unit_type = clean_string(unit.get("unit_type"))

                    if unit_type:
                        unit_types.append(unit_type)

            unit_types = deduplicate(unit_types)

            part = contingent

            if part and size:
                part = f"{part} ({size})"
            elif size:
                part = size

            if part and unit_types:
                part = f"{part}: {', '.join(unit_types)}"

            if part:
                components.append(part)

        if components:
            result.append(f"{side}: {', '.join(components)}")

    return result


def get_equipment_text(battle: dict[str, Any]) -> list[str]:
    """
    Build side-specific equipment text from participant equipment arrays.
    """

    participants = battle.get("participants", [])

    if not isinstance(participants, list):
        return []

    result: list[str] = []

    for participant in participants:
        if not isinstance(participant, dict):
            continue

        side = clean_string(participant.get("side"))
        equipment_items = participant.get("equipment", [])

        if not side or not isinstance(equipment_items, list):
            continue

        formatted_items: list[str] = []

        for item in equipment_items:
            if not isinstance(item, dict):
                continue

            equipment_type = clean_string(item.get("equipment_type"))
            size = clean_string(item.get("size"))
            role = clean_string(item.get("role"))

            if not equipment_type:
                continue

            text = equipment_type

            if size:
                text += f" ({size})"

            if role:
                text += f" - {role}"

            formatted_items.append(text)

        if formatted_items:
            result.append(f"{side}: {', '.join(formatted_items)}")

    return result


def get_turning_points(battle: dict[str, Any]) -> list[str]:
    """
    Extract turning point descriptions.
    """

    turning_points = battle.get("narrative", {}).get("turning_points", [])

    if not isinstance(turning_points, list):
        return []

    result: list[str] = []

    for item in turning_points:
        if not isinstance(item, dict):
            continue

        description = clean_string(item.get("description"))

        if description:
            result.append(description)

    return result


def get_consequences(battle: dict[str, Any]) -> list[str]:
    """
    Extract consequences.
    """

    return clean_list(
        battle.get("narrative", {}).get("consequences", [])
    )


def get_event_sequence(battle: dict[str, Any]) -> list[str]:
    """
    Build compact event descriptions for event-centric retrieval.
    """

    events = battle.get("narrative", {}).get("events", [])

    if not isinstance(events, list):
        return []

    result: list[str] = []

    for event in events[:MAX_EVENT_ITEMS]:
        if not isinstance(event, dict):
            continue

        phase = clean_string(event.get("phase"))
        event_type = clean_string(event.get("event_type"))
        description = clean_string(event.get("description"))
        outcome = clean_string(event.get("outcome"))

        prefix_parts = [
            value
            for value in [phase, event_type]
            if value
        ]

        prefix = " / ".join(prefix_parts)

        body_parts = [
            value
            for value in [description, outcome]
            if value
        ]

        if not body_parts:
            continue

        body = " -> ".join(body_parts)

        if prefix:
            result.append(f"{prefix}: {body}")
        else:
            result.append(body)

    return result


def get_numerical_advantage_text(battle: dict[str, Any]) -> str | None:
    """
    Format numerical advantage, including null-side caveats.
    """

    numerical_advantage = get_nested(
        battle,
        ["military_facets", "numerical_advantage"],
    )

    if not isinstance(numerical_advantage, dict):
        return None

    side = clean_string(numerical_advantage.get("side"))
    note = clean_string(numerical_advantage.get("note"))

    if side and note:
        return f"{side} ({note})"

    if side:
        return side

    if note:
        return note

    return None


def build_retrieval_document(
    battle: dict[str, Any],
    battle_id: str,
) -> dict[str, Any]:
    """
    Convert one structured battle JSON into one retrieval document.
    """

    identification = battle.get("identification", {})
    metadata = battle.get("metadata", {})
    outcome = battle.get("outcome", {})
    military_facets = battle.get("military_facets", {})
    narrative = battle.get("narrative", {})
    narrative_patterns = battle.get("narrative_patterns", {})
    retrieval_texts = battle.get("retrieval_texts", {})

    if not isinstance(identification, dict):
        identification = {}
    if not isinstance(metadata, dict):
        metadata = {}
    if not isinstance(outcome, dict):
        outcome = {}
    if not isinstance(military_facets, dict):
        military_facets = {}
    if not isinstance(narrative, dict):
        narrative = {}
    if not isinstance(narrative_patterns, dict):
        narrative_patterns = {}
    if not isinstance(retrieval_texts, dict):
        retrieval_texts = {}

    name = clean_string(identification.get("name"))
    alternative_names = clean_list(identification.get("alternative_names", []))

    participants = get_participants(battle)
    participants_text = get_participants_text(battle)

    year = get_year(battle)
    location_text = get_location_text(battle)
    conflict = clean_string(metadata.get("conflict"))

    location = metadata.get("location", {})
    country_modern = None

    if isinstance(location, dict):
        country_modern = clean_string(location.get("country_modern"))

    winner = clean_string(outcome.get("winner"))
    result = clean_string(outcome.get("result"))
    result_type = clean_string(outcome.get("result_type"))

    battle_type = clean_list(military_facets.get("battle_type", []))
    domain = clean_list(military_facets.get("domain", []))
    terrain = clean_list(military_facets.get("terrain", []))
    weapons_or_units = clean_list(military_facets.get("weapons_or_units", []))
    tactics = clean_list(military_facets.get("tactics", []))
    special_features = clean_list(military_facets.get("special_features", []))

    patterns = clean_list(narrative_patterns.get("patterns", []))
    similarity_tags = clean_list(narrative_patterns.get("similarity_tags", []))
    action_sequence = clean_list(narrative_patterns.get("action_sequence", []))

    lines: list[str] = []

    add_text_field(lines, "Battle", name)
    add_text_field(lines, "Alternative names", alternative_names)
    add_text_field(lines, "Year", year)
    add_text_field(lines, "Conflict", conflict)
    add_text_field(lines, "Location", location_text)

    # Richer than metadata participants; improves recall for allies/contingents.
    add_text_field(lines, "Participants", participants_text or participants)
    add_text_field(lines, "Commanders", get_commanders_text(battle))
    add_text_field(lines, "Strengths", get_strengths_text(battle))
    add_text_field(lines, "Force composition", get_force_composition_text(battle))

    add_text_field(lines, "Winner", winner)
    add_text_field(lines, "Result", result)
    add_text_field(lines, "Result type", result_type)

    add_text_field(lines, "Battle type", battle_type)
    add_text_field(lines, "Domain", domain)
    add_text_field(lines, "Terrain", terrain)
    add_text_field(lines, "Weapons or units", weapons_or_units)
    add_text_field(lines, "Equipment", get_equipment_text(battle))
    add_text_field(lines, "Tactics", tactics)
    add_text_field(lines, "Special features", special_features)
    add_text_field(lines, "Numerical advantage", get_numerical_advantage_text(battle))

    add_text_field(lines, "Narrative patterns", patterns)
    add_text_field(lines, "Similarity tags", similarity_tags)
    add_text_field(lines, "Action sequence", action_sequence)

    add_text_field(lines, "Background", narrative.get("background"))
    add_text_field(lines, "Initial situation", narrative.get("initial_situation"))
    add_text_field(lines, "Key events", get_event_sequence(battle))
    add_text_field(lines, "Turning points", get_turning_points(battle))
    add_text_field(lines, "Consequences", get_consequences(battle))
    add_text_field(lines, "Historical significance", narrative.get("historical_significance"))

    # Do not include source_summary if it is manually filled / unchecked.
    add_text_field(lines, "Summary", retrieval_texts.get("short_summary"))
    add_text_field(lines, "Tactical summary", retrieval_texts.get("tactical_summary"))
    add_text_field(lines, "Narrative summary", retrieval_texts.get("narrative_summary"))
    add_text_field(lines, "Action sequence summary", retrieval_texts.get("action_sequence_summary"))
    add_text_field(lines, "Similarity query", retrieval_texts.get("similarity_query_text"))

    retrieval_text = "\n".join(lines)

    quality_status = "auto_extracted"

    retrieval_metadata = {
        "year": year,
        "alternative_names": alternative_names,
        "conflict": conflict,
        "country_modern": country_modern,
        "battle_type": battle_type,
        "domain": domain,
        "participants": participants,
        "winner": winner,
        "terrain": terrain,
        "weapons_or_units": weapons_or_units,
        "tactics": tactics,
        "patterns": patterns,
        "quality_status": quality_status,
    }

    return {
        "battle_id": battle_id,
        "name": name,
        "text": retrieval_text,
        "metadata": retrieval_metadata,
    }


def main() -> None:
    if not BATTLE_JSON_DIR.exists():
        raise FileNotFoundError(
            f"Battle JSON directory not found: {BATTLE_JSON_DIR.resolve()}"
        )

    json_files = sorted(
        BATTLE_JSON_DIR.glob("Q*.json"),
        key=lambda path: int(path.stem[1:]),
    )

    json_files = json_files[:NUMBER_OF_BATTLES]

    print(f"Battle JSON files selected: {len(json_files)}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    successful = 0
    failed = 0

    with OUTPUT_FILE.open("w", encoding="utf-8") as output_file:
        for index, json_file in enumerate(json_files, start=1):
            print(f"[{index}/{len(json_files)}] Processing {json_file.stem}")

            try:
                battle = load_json(json_file)
                battle_id = json_file.stem

                retrieval_document = build_retrieval_document(
                    battle=battle,
                    battle_id=battle_id,
                )

                output_file.write(
                    json.dumps(
                        retrieval_document,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                successful += 1

            except (
                json.JSONDecodeError,
                OSError,
                TypeError,
                ValueError,
            ) as error:
                failed += 1

                print(
                    f"Failed {json_file.name}: "
                    f"{type(error).__name__}: {error}"
                )

    print()
    print("Retrieval document generation completed.")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Output: {OUTPUT_FILE.resolve()}")


def validate_retrieval_docs(
    battle_json_dir: Path,
    retrieval_docs_file: Path,
    min_text_length: int = 300,
) -> dict[str, Any]:
    """
    Validate generated retrieval documents and print a summary report.

    Checks:
    - retrieval doc count vs battle JSON count
    - battle_id exists
    - name exists
    - text is non-empty
    - year exists where possible
    - participants is a list
    - winner exists where possible
    - duplicate battle IDs
    - extremely short retrieval texts
    - useful event/narrative signal presence
    """

    battle_json_files = list(battle_json_dir.glob("Q*.json"))
    source_battle_count = len(battle_json_files)

    if not retrieval_docs_file.exists():
        raise FileNotFoundError(
            f"Retrieval docs file not found: {retrieval_docs_file.resolve()}"
        )

    total_docs = 0
    valid_docs = 0

    missing_battle_id = 0
    missing_name = 0
    missing_text = 0
    missing_year = 0
    missing_participants = 0
    invalid_participants = 0
    missing_winner = 0
    short_texts = 0
    invalid_json_lines = 0
    missing_event_signal = 0

    battle_ids: set[str] = set()
    duplicate_ids: set[str] = set()

    text_lengths: list[int] = []

    with retrieval_docs_file.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                invalid_json_lines += 1
                print(f"Invalid JSON at line {line_number}")
                continue

            total_docs += 1
            doc_valid = True

            battle_id = doc.get("battle_id")

            if not battle_id:
                missing_battle_id += 1
                doc_valid = False
            elif battle_id in battle_ids:
                duplicate_ids.add(str(battle_id))
                doc_valid = False
            else:
                battle_ids.add(str(battle_id))

            name = doc.get("name")

            if not name:
                missing_name += 1
                doc_valid = False

            text = doc.get("text")

            if not isinstance(text, str) or not text.strip():
                missing_text += 1
                doc_valid = False
            else:
                text_length = len(text.strip())
                text_lengths.append(text_length)

                if text_length < min_text_length:
                    short_texts += 1
                    doc_valid = False

                if not any(
                    marker in text
                    for marker in [
                        "Key events:",
                        "Action sequence:",
                        "Narrative summary:",
                        "Turning points:",
                    ]
                ):
                    missing_event_signal += 1

            metadata = doc.get("metadata")

            if not isinstance(metadata, dict):
                metadata = {}
                doc_valid = False

            if metadata.get("year") is None:
                missing_year += 1

            participants = metadata.get("participants")

            if participants is None:
                missing_participants += 1
                doc_valid = False
            elif not isinstance(participants, list):
                invalid_participants += 1
                doc_valid = False
            elif len(participants) == 0:
                missing_participants += 1

            if metadata.get("winner") is None:
                missing_winner += 1

            if doc_valid:
                valid_docs += 1

    average_text_length = (
        sum(text_lengths) / len(text_lengths)
        if text_lengths
        else 0
    )

    count_difference = total_docs - source_battle_count

    report = {
        "source_battle_jsons": source_battle_count,
        "total_docs": total_docs,
        "valid_docs": valid_docs,
        "count_difference": count_difference,
        "missing_battle_id": missing_battle_id,
        "missing_name": missing_name,
        "missing_text": missing_text,
        "missing_year": missing_year,
        "missing_participants": missing_participants,
        "invalid_participants": invalid_participants,
        "missing_winner": missing_winner,
        "duplicate_ids": len(duplicate_ids),
        "short_texts": short_texts,
        "invalid_json_lines": invalid_json_lines,
        "missing_event_signal": missing_event_signal,
        "average_text_length": average_text_length,
    }

    print()
    print("=" * 50)
    print("RETRIEVAL DOCUMENT VALIDATION REPORT")
    print("=" * 50)

    print(f"Source battle JSONs:      {source_battle_count}")
    print(f"Total retrieval docs:     {total_docs}")
    print(f"Valid docs:               {valid_docs}")
    print(f"Count difference:         {count_difference}")

    print()
    print(f"Missing battle_id:        {missing_battle_id}")
    print(f"Missing name:             {missing_name}")
    print(f"Missing text:             {missing_text}")
    print(f"Missing year:             {missing_year}")
    print(f"Missing participants:     {missing_participants}")
    print(f"Invalid participants:     {invalid_participants}")
    print(f"Missing winner:           {missing_winner}")
    print(f"Duplicate IDs:            {len(duplicate_ids)}")
    print(f"Extremely short docs:     {short_texts}")
    print(f"Missing event signal:     {missing_event_signal}")
    print(f"Invalid JSON lines:       {invalid_json_lines}")

    print()
    print(f"Average text length:      {average_text_length:.2f} characters")

    if duplicate_ids:
        print()
        print("Duplicate battle IDs:")

        for battle_id in sorted(duplicate_ids):
            print(f"  - {battle_id}")

    print("=" * 50)

    return report


if __name__ == "__main__":
    main()

    validate_retrieval_docs(
        battle_json_dir=BATTLE_JSON_DIR,
        retrieval_docs_file=OUTPUT_FILE,
    )
