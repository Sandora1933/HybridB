from __future__ import annotations

import json
from pathlib import Path
from typing import Any


BATTLE_JSON_DIR = Path("data/json_kb_v1")
OUTPUT_FILE = Path("data/retrieval_docs/retrieval_docs_v1.jsonl")

NUMBER_OF_BATTLES = None


def load_json(file_path: Path) -> dict[str, Any]:
    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def clean_list(values: Any) -> list[str]:
    """
    Return only non-empty string values from a list.
    """

    if not isinstance(values, list):
        return []

    return [
        str(value).strip()
        for value in values
        if value is not None and str(value).strip()
    ]


def join_values(values: list[str]) -> str | None:
    """
    Convert a list into a semicolon-separated retrieval string.
    """

    if not values:
        return None

    return "; ".join(values)


def get_participants(
    battle: dict[str, Any],
) -> list[str]:
    """
    Extract participant side names.
    """

    participants = battle.get("participants", [])

    return [
        str(participant["side"]).strip()
        for participant in participants
        if isinstance(participant, dict)
        and participant.get("side")
    ]


def get_year(
    battle: dict[str, Any],
) -> int | None:
    """
    Extract battle start year.
    """

    try:
        return battle["metadata"]["time_period"]["start_date"]["year"]
    except (KeyError, TypeError):
        return None


def get_location_text(
    battle: dict[str, Any],
) -> str | None:
    """
    Build location string with historical and modern country.
    """

    location = (
        battle.get("metadata", {})
        .get("location", {})
    )

    historical_parts = [
        location.get("name"),
        location.get("region"),
        location.get("country"),
    ]

    historical_parts = [
        str(value).strip()
        for value in historical_parts
        if value
    ]

    historical_parts = list(
        dict.fromkeys(historical_parts)
    )

    modern_country = location.get("country_modern")

    if not historical_parts and not modern_country:
        return None

    text = ", ".join(historical_parts)

    if modern_country:
        if text:
            text += f"; modern country: {modern_country}"
        else:
            text = f"Modern country: {modern_country}"

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

    if not value:
        return

    lines.append(f"{label}: {value}")


def build_retrieval_document(
    battle: dict[str, Any],
    battle_id: str,
) -> dict[str, Any]:
    """
    Convert one structured battle JSON into one retrieval document.
    """

    identification = battle.get(
        "identification",
        {},
    )

    metadata = battle.get(
        "metadata",
        {},
    )

    outcome = battle.get(
        "outcome",
        {},
    )

    military_facets = battle.get(
        "military_facets",
        {},
    )

    narrative_patterns = battle.get(
        "narrative_patterns",
        {},
    )

    retrieval_texts = battle.get(
        "retrieval_texts",
        {},
    )

    extraction_metadata = battle.get(
        "extraction_metadata",
        {},
    )

    name = identification.get("name")

    alternative_names = clean_list(
        identification.get(
            "alternative_names",
            [],
        )
    )

    participants = get_participants(battle)

    year = get_year(battle)

    location_text = get_location_text(battle)

    conflict = metadata.get("conflict")

    location = metadata.get(
        "location",
        {},
    )

    country_modern = location.get(
        "country_modern"
    )

    winner = outcome.get("winner")
    result = outcome.get("result")

    battle_type = clean_list(
        military_facets.get(
            "battle_type",
            [],
        )
    )

    domain = clean_list(
        military_facets.get(
            "domain",
            [],
        )
    )

    terrain = clean_list(
        military_facets.get(
            "terrain",
            [],
        )
    )

    weapons_or_units = clean_list(
        military_facets.get(
            "weapons_or_units",
            [],
        )
    )

    tactics = clean_list(
        military_facets.get(
            "tactics",
            [],
        )
    )

    special_features = clean_list(
        military_facets.get(
            "special_features",
            [],
        )
    )

    patterns = clean_list(
        narrative_patterns.get(
            "patterns",
            [],
        )
    )

    similarity_tags = clean_list(
        narrative_patterns.get(
            "similarity_tags",
            [],
        )
    )

    action_sequence = clean_list(
        narrative_patterns.get(
            "action_sequence",
            [],
        )
    )

    lines: list[str] = []

    add_text_field(
        lines,
        "Battle",
        name,
    )

    add_text_field(
        lines,
        "Alternative names",
        alternative_names,
    )

    add_text_field(
        lines,
        "Year",
        year,
    )

    add_text_field(
        lines,
        "Conflict",
        conflict,
    )

    add_text_field(
        lines,
        "Location",
        location_text,
    )

    add_text_field(
        lines,
        "Participants",
        participants,
    )

    add_text_field(
        lines,
        "Winner",
        winner,
    )

    add_text_field(
        lines,
        "Result",
        result,
    )

    add_text_field(
        lines,
        "Battle type",
        battle_type,
    )

    add_text_field(
        lines,
        "Domain",
        domain,
    )

    add_text_field(
        lines,
        "Terrain",
        terrain,
    )

    add_text_field(
        lines,
        "Weapons or units",
        weapons_or_units,
    )

    add_text_field(
        lines,
        "Tactics",
        tactics,
    )

    add_text_field(
        lines,
        "Special features",
        special_features,
    )

    add_text_field(
        lines,
        "Narrative patterns",
        patterns,
    )

    add_text_field(
        lines,
        "Similarity tags",
        similarity_tags,
    )

    add_text_field(
        lines,
        "Summary",
        retrieval_texts.get(
            "short_summary"
        ),
    )

    add_text_field(
        lines,
        "Tactical summary",
        retrieval_texts.get(
            "tactical_summary"
        ),
    )

    add_text_field(
        lines,
        "Narrative summary",
        retrieval_texts.get(
            "narrative_summary"
        ),
    )

    add_text_field(
        lines,
        "Action sequence",
        action_sequence,
    )

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
            f"Battle JSON directory not found: "
            f"{BATTLE_JSON_DIR.resolve()}"
        )

    json_files = sorted(
        BATTLE_JSON_DIR.glob("Q*.json"),
        key=lambda path: int(path.stem[1:]),
    )

    json_files = json_files[:NUMBER_OF_BATTLES]

    print(
        f"Battle JSON files selected: {len(json_files)}"
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    successful = 0
    failed = 0

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as output_file:

        for index, json_file in enumerate(
            json_files,
            start=1,
        ):
            print(
                f"[{index}/{len(json_files)}] "
                f"Processing {json_file.stem}"
            )

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
                    f"{type(error).__name__}: "
                    f"{error}"
                )

    print()
    print("Retrieval document generation completed.")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Output: {OUTPUT_FILE.resolve()}")


def validate_retrieval_docs(
    battle_json_dir: Path,
    retrieval_docs_file: Path,
    min_text_length: int = 100,
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
    """

    battle_json_files = list(
        battle_json_dir.glob("Q*.json")
    )

    source_battle_count = len(battle_json_files)

    if not retrieval_docs_file.exists():
        raise FileNotFoundError(
            f"Retrieval docs file not found: "
            f"{retrieval_docs_file.resolve()}"
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

    battle_ids: set[str] = set()
    duplicate_ids: set[str] = set()

    text_lengths: list[int] = []

    with retrieval_docs_file.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                doc = json.loads(line)

            except json.JSONDecodeError:
                invalid_json_lines += 1

                print(
                    f"Invalid JSON at line {line_number}"
                )
                continue

            total_docs += 1

            doc_valid = True

            # battle_id
            battle_id = doc.get("battle_id")

            if not battle_id:
                missing_battle_id += 1
                doc_valid = False
            else:
                if battle_id in battle_ids:
                    duplicate_ids.add(battle_id)
                    doc_valid = False
                else:
                    battle_ids.add(battle_id)

            # name
            name = doc.get("name")

            if not name:
                missing_name += 1
                doc_valid = False

            # text
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

            # metadata
            metadata = doc.get("metadata")

            if not isinstance(metadata, dict):
                metadata = {}
                doc_valid = False

            # year
            if metadata.get("year") is None:
                missing_year += 1

            # participants
            participants = metadata.get("participants")

            if participants is None:
                missing_participants += 1
                doc_valid = False

            elif not isinstance(participants, list):
                invalid_participants += 1
                doc_valid = False

            elif len(participants) == 0:
                missing_participants += 1

            # winner
            if metadata.get("winner") is None:
                missing_winner += 1

            if doc_valid:
                valid_docs += 1

    average_text_length = (
        sum(text_lengths) / len(text_lengths)
        if text_lengths
        else 0
    )

    count_difference = (
        total_docs - source_battle_count
    )

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
    print(f"Invalid JSON lines:       {invalid_json_lines}")

    print()
    print(
        f"Average text length:      "
        f"{average_text_length:.2f} characters"
    )

    if duplicate_ids:
        print()
        print("Duplicate battle IDs:")

        for battle_id in sorted(duplicate_ids):
            print(f"  - {battle_id}")

    print("=" * 50)

    return report


if __name__ == "__main__":
    #main()
    validate_retrieval_docs(
        battle_json_dir=BATTLE_JSON_DIR,
        retrieval_docs_file=OUTPUT_FILE,
    )