from __future__ import annotations

import csv
import json
import re
import time
from pathlib import Path
from typing import Any
from llm import load_llm, generate_answer
from prompts import build_json_generation_prompt


RAW_TEXT_DIR = Path("data/extracted_wikipedia_raw")
OUTPUT_JSON_DIR = Path("data/json_battles")
EMPTY_SCHEMA_FILE = Path("data/empty_schema.json")
BATTLES_IDS_CSV = Path("data/wikidata_battles.csv")
FAILED_LOG_FILE = Path("data/json_generation_failures.jsonl") 

START_FILE_INDEX = 0
NUMBER_OF_FILES: int | None = 11561

REQUEST_DELAY_SECONDS = 0.7
MAX_RETRIES = 3

LLM = load_llm()

def load_text_file(file_path: Path) -> str:

    return file_path.read_text(encoding="utf-8").strip()


def call_model(prompt: str) -> str:
    return generate_answer(
        llm=LLM,
        prompt=prompt,
        system_message=(
            "You are a structured information extraction system. "
            "Your task is to populate the provided JSON template using only information "
            "contained in the source text about the target historical battle. "
        ),
        max_new_tokens=8192,
        do_sample=False,
    )


def extract_json_from_response(response_text: str) -> dict[str, Any]:
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
                "The model response does not contain a JSON object."
            )

        json_candidate = response_text[first_brace:last_brace + 1]

        try:
            parsed = json.loads(json_candidate)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"The model returned invalid JSON: {error}"
            ) from error

    if not isinstance(parsed, dict):
        raise ValueError(
            "The model response must be a JSON object."
        )

    return parsed


def save_json_file(
    data: dict[str, Any],
    output_file: Path,
) -> None:
    """Save JSON using readable UTF-8 formatting."""

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


def log_failure(
    qid: str,
    raw_file: Path,
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
                    "raw_file": str(raw_file),
                    "error": str(error),
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def generate_json(
    raw_text_file: Path,
    empty_schema: dict[str, Any],
) -> dict[str, Any]:
    qid = raw_text_file.stem

    if not re.fullmatch(r"Q[1-9]\d*", qid):
        raise ValueError(
            f"Raw filename does not contain a valid QID: "
            f"{raw_text_file.name}"
        )

    battle_text = load_text_file(raw_text_file)

    if not battle_text:
        raise ValueError(
            f"Raw text file is empty: {raw_text_file}"
        )

    prompt = build_json_generation_prompt(
        qid=qid,
        battle_text=battle_text,
        empty_schema=empty_schema,
    )
    print(f"Prompt for {qid}:\n{prompt}\n")
    print(f"End prompt for {qid}\n")

    model_response = call_model(prompt)

    return extract_json_from_response(model_response)


def generate_json_with_retries(
    raw_text_file: Path,
    empty_schema: dict[str, Any],
    max_retries: int = MAX_RETRIES,
) -> dict[str, Any]:
    """Generate one JSON record with retry logic."""

    qid = raw_text_file.stem

    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            return generate_json(
                raw_text_file=raw_text_file,
                empty_schema=empty_schema,
            )

        except Exception as error:
            last_error = error

            print(
                f"Attempt {attempt}/{max_retries} failed "
                f"for {qid}: {type(error).__name__}: {error}"
            )

            if attempt < max_retries:
                wait_seconds = 2 ** attempt

                print(
                    f"Retrying in {wait_seconds} seconds..."
                )

                time.sleep(wait_seconds)

    assert last_error is not None
    raise last_error


def main() -> None:
    if not RAW_TEXT_DIR.exists():
        raise FileNotFoundError(
            f"Raw-text directory was not found: "
            f"{RAW_TEXT_DIR.resolve()}"
        )

    if not EMPTY_SCHEMA_FILE.exists():
        raise FileNotFoundError(
            f"Empty schema file was not found: "
            f"{EMPTY_SCHEMA_FILE.resolve()}"
        )

    with EMPTY_SCHEMA_FILE.open("r", encoding="utf-8") as file:
        empty_schema = json.load(file)

    # Load the list of raw text files corresponding to Wikidata QIDs
    raw_files = []

    with open(BATTLES_IDS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            qid = row["qid"]              
            file_path = RAW_TEXT_DIR / f"{qid}.txt"

            if file_path.exists():
                raw_files.append(file_path)

    # Select the subset of files contrained by NUMBER_OF_FILES
    if NUMBER_OF_FILES is None:
        selected_files = raw_files[START_FILE_INDEX:]
    else:
        selected_files = raw_files[START_FILE_INDEX : START_FILE_INDEX + NUMBER_OF_FILES]

    successful = 0
    failed = 0
    skipped = 0

    total = len(selected_files)

    print(f"Raw battle files found: {len(raw_files)}")
    print(f"Files selected for this run: {total}")
    print("Chosen files:")
    for file in selected_files:
        print(f"  {file.name}")

    for current_index, raw_file in enumerate(
        selected_files,
        start=1,
    ):
        qid = raw_file.stem
        output_file = OUTPUT_JSON_DIR / f"{qid}.json"

        print(
            f"\n[{current_index}/{total}] "
            f"Processing {qid}..."
        )

        # Skip potential overwrites of existing JSON files
        if output_file.exists():
            skipped += 1
            print(
                f"Skipped: {output_file.name} already exists."
            )
            continue

        # generate json for current raw text file
        try:
            generated_json = generate_json_with_retries(
                raw_text_file=raw_file,
                empty_schema=empty_schema,
            )

            save_json_file(
                data=generated_json,
                output_file=output_file,
            )

            successful += 1

            print(
                f"Saved to: {output_file}"
            )

        except Exception as error:
            failed += 1

            print(
                f"Generation failed for {qid}: "
                f"{type(error).__name__}: {error}"
            )

            log_failure(
                qid=qid,
                raw_file=raw_file,
                error=error,
            )

        time.sleep(REQUEST_DELAY_SECONDS)

    print(f"\n JSON generation completed.")
    print(f"Successful: {successful}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    start = time.perf_counter()
    main()
    end = time.perf_counter()
    print(f"\nExecution time: {end - start:.2f} seconds")