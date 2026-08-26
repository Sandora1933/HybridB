from __future__ import annotations

import csv
import json
import re
import time
from pathlib import Path
from typing import Any

from llm import load_llm, generate_answer


# ============================================================
# Paths
# ============================================================

RAW_TEXT_DIR = Path("data/extracted_wikipedia_raw")
INPUT_JSON_DIR = Path("data/json_kb_v1")

# Keep corrected files separate from the original knowledge base.
OUTPUT_JSON_DIR = Path("data/json_kb_v2")

# Optional: save verifier reports for later inspection/evaluation.
VERIFIER_REPORT_DIR = Path("data/verification_reports")

BATTLES_IDS_CSV = Path("data/wikidata_battles.csv")

FAILED_LOG_FILE = Path(
    "data/json_verification_failures.jsonl"
)


# ============================================================
# Run configuration
# ============================================================

START_FILE_INDEX = 0
NUMBER_OF_FILES: int | None = 800

REQUEST_DELAY_SECONDS = 0.7
MAX_RETRIES = 3

# If True, save the verifier's correction report for every battle.
SAVE_VERIFIER_REPORTS = True

# If True, files already present in json_kb_v2 are skipped.
SKIP_EXISTING = True


# ============================================================
# Model
# ============================================================

LLM = load_llm()


# ============================================================
# File utilities
# ============================================================

def load_text_file(file_path: Path) -> str:
    return file_path.read_text(
        encoding="utf-8"
    ).strip()


def load_json_file(
    file_path: Path,
) -> dict[str, Any]:
    with file_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(
            f"JSON root must be an object: {file_path}"
        )

    return data


def save_json_file(
    data: dict[str, Any],
    output_file: Path,
) -> None:
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# JSON parsing
# ============================================================

def extract_json_from_response(
    response_text: str,
) -> dict[str, Any]:
    """
    Parse a JSON object from the model response.

    Supports:
    - plain JSON;
    - JSON inside Markdown code fences;
    - limited surrounding explanatory text.
    """

    response_text = response_text.strip()

    fenced_match = re.search(
        r"```(?:json)?\s*(\{.*\})\s*```",
        response_text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    if fenced_match:
        response_text = fenced_match.group(1).strip()

    try:
        parsed = json.loads(response_text)

    except json.JSONDecodeError:
        first_brace = response_text.find("{")
        last_brace = response_text.rfind("}")

        if first_brace == -1 or last_brace == -1:
            raise ValueError(
                "The model response does not contain "
                "a JSON object."
            )

        json_candidate = response_text[
            first_brace:last_brace + 1
        ]

        try:
            parsed = json.loads(json_candidate)

        except json.JSONDecodeError as error:
            raise ValueError(
                f"The model returned invalid JSON: "
                f"{error}"
            ) from error

    if not isinstance(parsed, dict):
        raise ValueError(
            "The model response must be a JSON object."
        )

    return parsed


# ============================================================
# Model calls
# ============================================================

def call_verifier_model(
    prompt: str,
) -> str:
    return generate_answer(
        llm=LLM,
        prompt=prompt,
        system_message=(
            "You are a verification system for structured "
            "historical battle data. "
            "Compare the generated JSON against the supplied "
            "source text. "
            "Use only the source text and identify important "
            "factual, semantic, scope, and completeness errors."
        ),
        max_new_tokens=4096,
        do_sample=False,
    )


def call_regeneration_model(
    prompt: str,
) -> str:
    return generate_answer(
        llm=LLM,
        prompt=prompt,
        system_message=(
            "You are a structured information correction system. "
            "Correct the supplied historical battle JSON using "
            "only the source text and the verifier report. "
            "Preserve valid existing information and return only "
            "the corrected JSON."
        ),
        max_new_tokens=8192,
        do_sample=False,
    )


# ============================================================
# Verifier prompt
# ============================================================

def build_verifier_prompt(
    qid: str,
    battle_text: str,
    generated_json: dict[str, Any],
) -> str:

    json_text = json.dumps(
        generated_json,
        ensure_ascii=False,
        indent=2,
    )

    return f"""
You are reviewing a structured JSON extraction of one historical battle.

TARGET BATTLE:
{qid}

TASK:
Compare the generated JSON against the supplied source text and identify
important corrections.

Use ONLY the source text. Do not use external or prior historical knowledge.

Correctness has priority over completeness.

CHECK FOR:

1. Unsupported information:
   values or claims that are not supported by the source.

2. Incorrect information:
   values that contradict or misrepresent the source.

3. Scope errors:
   partial, subunit, phase-specific, approximate, or incomparable numbers
   incorrectly represented as whole-side totals or general battle facts.

4. Schema-semantic errors:
   information placed in an inappropriate field, such as personnel or units
   represented as equipment, or formations assigned to unsupported parent
   organizations.

5. Important omissions:
   clearly supported information whose absence materially reduces the quality
   of participants, force composition, casualties, military facets, major
   events, turning points, narrative patterns, or retrieval summaries.

RULES:

- Preserve uncertainty, ranges, conflicting estimates, and event-specific scope.
- Do not infer numerical advantage unless opposing strength figures are
  reasonably comparable.
- Do not combine personnel, animals, equipment, non-combatants, reinforcements,
  or forces from different phases into one total unless the source clearly
  supports that total.
- Do not introduce external geographical normalization, equipment functions,
  historical rankings, causal claims, comparisons, or broader significance.
- Do not report trivial stylistic issues.
- Do not request changes to fields that are already reasonable and supported.
- Do not rewrite the whole JSON.
- Report only corrections that materially improve factual accuracy,
  semantic correctness, or useful source-supported coverage.

Return exactly one JSON object with this structure:

{{
  "verdict": "PASS or REVISE",
  "corrections": [
    {{
      "field": "JSON path or field description",
      "problem": "brief explanation",
      "action": "replace, remove, add, or revise",
      "suggested_correction": "source-grounded correction, or null"
    }}
  ]
}}

If there are no important corrections, return:

{{
  "verdict": "PASS",
  "corrections": []
}}

SOURCE TEXT:

{battle_text}

GENERATED JSON:

{json_text}

Return only the verification JSON.
""".strip()


# ============================================================
# Regeneration prompt
# ============================================================

def build_regeneration_prompt(
    qid: str,
    battle_text: str,
    current_json: dict[str, Any],
    verifier_report: dict[str, Any],
) -> str:

    current_json_text = json.dumps(
        current_json,
        ensure_ascii=False,
        indent=2,
    )

    verifier_report_text = json.dumps(
        verifier_report,
        ensure_ascii=False,
        indent=2,
    )

    return f"""
You are correcting a structured JSON extraction of one historical battle.

TARGET BATTLE:
{qid}

TASK:
Revise the existing JSON using the verifier report and the supplied source
text.

Use ONLY the source text.

Preserve all existing information that is already source-supported and was not
identified as problematic.

RULES:

1. Apply every verifier correction that is supported by the source.

2. Do not blindly copy suggested corrections if they conflict with the source.
   The source text is authoritative.

3. Correctness has priority over completeness.

4. Do not introduce external or prior historical knowledge.

5. Preserve the exact existing JSON structure, field names, nesting, and data
   types.

6. Preserve uncertainty, ranges, conflicting estimates, and event-specific
   numerical scope.

7. Do not convert partial or incomparable figures into whole-side totals.

8. Unsupported scalar fields must be null.
   Unsupported arrays must be [].

9. Do not remove valid source-supported information merely because it was not
   mentioned by the verifier.

10. Set overall_confidence, every field_confidence value, and every event or
    turning-point confidence field to null.

11. Return exactly one valid JSON object with no Markdown, explanation,
    comments, or surrounding text.

SOURCE TEXT:

{battle_text}

CURRENT JSON:

{current_json_text}

VERIFIER REPORT:

{verifier_report_text}

Return the corrected JSON object.
""".strip()


# ============================================================
# Validation
# ============================================================

def validate_verifier_report(
    report: dict[str, Any],
) -> None:

    verdict = report.get("verdict")
    corrections = report.get("corrections")

    if verdict not in {"PASS", "REVISE"}:
        raise ValueError(
            "Verifier report must contain verdict "
            "'PASS' or 'REVISE'."
        )

    if not isinstance(corrections, list):
        raise ValueError(
            "Verifier report 'corrections' must be a list."
        )

    if verdict == "PASS" and corrections:
        raise ValueError(
            "PASS verifier report must have an empty "
            "corrections list."
        )


def validate_corrected_json(
    qid: str,
    original_json: dict[str, Any],
    corrected_json: dict[str, Any],
) -> None:
    """
    Lightweight safety validation.

    This does not judge historical correctness.
    It only catches obvious structural failures.
    """

    if corrected_json.get("battle_id") != original_json.get(
        "battle_id"
    ):
        raise ValueError(
            "Corrected battle_id differs from original battle_id."
        )

    if (
        original_json.get("identification", {}).get("name")
        and corrected_json.get("identification", {}).get("name")
        is None
    ):
        raise ValueError(
            "Corrected JSON unexpectedly removed battle name."
        )

    metadata = corrected_json.get(
        "extraction_metadata",
        {},
    )

    if metadata.get("overall_confidence") is not None:
        raise ValueError(
            "overall_confidence must be null."
        )

    field_confidence = metadata.get(
        "field_confidence",
        {},
    )

    if isinstance(field_confidence, dict):
        for field_name, value in field_confidence.items():
            if value is not None:
                raise ValueError(
                    f"field_confidence.{field_name} "
                    "must be null."
                )

    narrative = corrected_json.get(
        "narrative",
        {},
    )

    for event in narrative.get("events", []):
        if (
            isinstance(event, dict)
            and event.get("confidence") is not None
        ):
            raise ValueError(
                "Event confidence must be null."
            )

    for turning_point in narrative.get(
        "turning_points",
        [],
    ):
        if (
            isinstance(turning_point, dict)
            and turning_point.get("confidence")
            is not None
        ):
            raise ValueError(
                "Turning-point confidence must be null."
            )


# ============================================================
# Retry logic
# ============================================================

def verify_json(
    qid: str,
    battle_text: str,
    generated_json: dict[str, Any],
) -> dict[str, Any]:

    prompt = build_verifier_prompt(
        qid=qid,
        battle_text=battle_text,
        generated_json=generated_json,
    )

    response = call_verifier_model(prompt)

    report = extract_json_from_response(response)

    validate_verifier_report(report)

    return report


def verify_json_with_retries(
    qid: str,
    battle_text: str,
    generated_json: dict[str, Any],
    max_retries: int = MAX_RETRIES,
) -> dict[str, Any]:

    last_error: Exception | None = None

    for attempt in range(
        1,
        max_retries + 1,
    ):
        try:
            return verify_json(
                qid=qid,
                battle_text=battle_text,
                generated_json=generated_json,
            )

        except Exception as error:
            last_error = error

            print(
                f"Verifier attempt "
                f"{attempt}/{max_retries} failed "
                f"for {qid}: "
                f"{type(error).__name__}: {error}"
            )

            if attempt < max_retries:
                wait_seconds = 2 ** attempt

                print(
                    f"Retrying verifier in "
                    f"{wait_seconds} seconds..."
                )

                time.sleep(wait_seconds)

    assert last_error is not None
    raise last_error


def regenerate_json(
    qid: str,
    battle_text: str,
    current_json: dict[str, Any],
    verifier_report: dict[str, Any],
) -> dict[str, Any]:

    prompt = build_regeneration_prompt(
        qid=qid,
        battle_text=battle_text,
        current_json=current_json,
        verifier_report=verifier_report,
    )

    response = call_regeneration_model(prompt)

    corrected_json = extract_json_from_response(
        response
    )

    validate_corrected_json(
        qid=qid,
        original_json=current_json,
        corrected_json=corrected_json,
    )

    return corrected_json


def regenerate_json_with_retries(
    qid: str,
    battle_text: str,
    current_json: dict[str, Any],
    verifier_report: dict[str, Any],
    max_retries: int = MAX_RETRIES,
) -> dict[str, Any]:

    last_error: Exception | None = None

    for attempt in range(
        1,
        max_retries + 1,
    ):
        try:
            return regenerate_json(
                qid=qid,
                battle_text=battle_text,
                current_json=current_json,
                verifier_report=verifier_report,
            )

        except Exception as error:
            last_error = error

            print(
                f"Regeneration attempt "
                f"{attempt}/{max_retries} failed "
                f"for {qid}: "
                f"{type(error).__name__}: {error}"
            )

            if attempt < max_retries:
                wait_seconds = 2 ** attempt

                print(
                    f"Retrying regeneration in "
                    f"{wait_seconds} seconds..."
                )

                time.sleep(wait_seconds)

    assert last_error is not None
    raise last_error


# ============================================================
# Failure logging
# ============================================================

def log_failure(
    qid: str,
    stage: str,
    error: Exception,
) -> None:

    FAILED_LOG_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with FAILED_LOG_FILE.open(
        "a",
        encoding="utf-8",
    ) as file:
        file.write(
            json.dumps(
                {
                    "qid": qid,
                    "stage": stage,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                ensure_ascii=False,
            )
            + "\n"
        )


# ============================================================
# Battle processing
# ============================================================

def process_battle(
    qid: str,
    json_file: Path,
    raw_file: Path,
) -> tuple[str, dict[str, Any] | None]:

    battle_text = load_text_file(raw_file)

    if not battle_text:
        raise ValueError(
            f"Raw text is empty: {raw_file}"
        )

    generated_json = load_json_file(
        json_file
    )

    print(
        f"Running verifier for {qid}..."
    )

    verifier_report = verify_json_with_retries(
        qid=qid,
        battle_text=battle_text,
        generated_json=generated_json,
    )

    if SAVE_VERIFIER_REPORTS:
        report_file = (
            VERIFIER_REPORT_DIR
            / f"{qid}.json"
        )

        save_json_file(
            verifier_report,
            report_file,
        )

    verdict = verifier_report["verdict"]
    corrections = verifier_report["corrections"]

    print(
        f"Verifier verdict: {verdict}"
    )
    print(
        f"Corrections found: {len(corrections)}"
    )

    # If verifier found no important problem,
    # preserve the original JSON as the accepted v2 record.
    if verdict == "PASS":
        return "pass", generated_json

    print(
        f"Regenerating {qid} using "
        f"{len(corrections)} corrections..."
    )

    corrected_json = regenerate_json_with_retries(
        qid=qid,
        battle_text=battle_text,
        current_json=generated_json,
        verifier_report=verifier_report,
    )

    return "revised", corrected_json


# ============================================================
# Main
# ============================================================

def main() -> None:

    if not RAW_TEXT_DIR.exists():
        raise FileNotFoundError(
            f"Raw-text directory not found: "
            f"{RAW_TEXT_DIR.resolve()}"
        )

    if not INPUT_JSON_DIR.exists():
        raise FileNotFoundError(
            f"Input JSON directory not found: "
            f"{INPUT_JSON_DIR.resolve()}"
        )

    if not BATTLES_IDS_CSV.exists():
        raise FileNotFoundError(
            f"Battle ID CSV not found: "
            f"{BATTLES_IDS_CSV.resolve()}"
        )

    OUTPUT_JSON_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if SAVE_VERIFIER_REPORTS:
        VERIFIER_REPORT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

    battle_records: list[tuple[str, Path, Path]] = []

    with BATTLES_IDS_CSV.open(
        newline="",
        encoding="utf-8",
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            qid = row["qid"]

            if not re.fullmatch(
                r"Q[1-9]\d*",
                qid,
            ):
                continue

            json_file = (
                INPUT_JSON_DIR
                / f"{qid}.json"
            )

            raw_file = (
                RAW_TEXT_DIR
                / f"{qid}.txt"
            )

            if (
                json_file.exists()
                and raw_file.exists()
            ):
                battle_records.append(
                    (
                        qid,
                        json_file,
                        raw_file,
                    )
                )

    if NUMBER_OF_FILES is None:
        selected_records = battle_records[
            START_FILE_INDEX:
        ]

    else:
        selected_records = battle_records[
            START_FILE_INDEX:
            START_FILE_INDEX + NUMBER_OF_FILES
        ]

    total = len(selected_records)

    passed = 0
    revised = 0
    failed = 0
    skipped = 0

    print(
        f"Battles with both JSON and raw text: "
        f"{len(battle_records)}"
    )

    print(
        f"Battles selected for this run: "
        f"{total}"
    )

    for current_index, (
        qid,
        json_file,
        raw_file,
    ) in enumerate(
        selected_records,
        start=1,
    ):

        output_file = (
            OUTPUT_JSON_DIR
            / f"{qid}.json"
        )

        print(
            f"\n[{current_index}/{total}] "
            f"Processing {qid}..."
        )

        if (
            SKIP_EXISTING
            and output_file.exists()
        ):
            skipped += 1

            print(
                f"Skipped: "
                f"{output_file.name} "
                f"already exists."
            )

            continue

        try:
            status, final_json = process_battle(
                qid=qid,
                json_file=json_file,
                raw_file=raw_file,
            )

            assert final_json is not None

            save_json_file(
                data=final_json,
                output_file=output_file,
            )

            if status == "pass":
                passed += 1

                print(
                    f"PASS: copied accepted JSON "
                    f"to {output_file}"
                )

            else:
                revised += 1

                print(
                    f"REVISED: saved corrected JSON "
                    f"to {output_file}"
                )

        except Exception as error:
            failed += 1

            print(
                f"Failed for {qid}: "
                f"{type(error).__name__}: "
                f"{error}"
            )

            log_failure(
                qid=qid,
                stage="verification_pipeline",
                error=error,
            )

        time.sleep(
            REQUEST_DELAY_SECONDS
        )

    print("\nVerification pipeline completed.")
    print(f"Passed without revision: {passed}")
    print(f"Revised: {revised}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    start = time.perf_counter()

    main()

    end = time.perf_counter()

    print(
        f"\nExecution time: "
        f"{end - start:.2f} seconds"
    )