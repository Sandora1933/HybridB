from __future__ import annotations
import csv
from time import perf_counter

import re
from pathlib import Path
from urllib.parse import unquote

import wikipediaapi


USER_AGENT = (
    "BattleRetriever/1.0 "
    "(Master thesis; vladyslav.romanov@tu-dresden.de)"
)

BATTLES_CSV = Path("data/wikidata_battles.csv")
DEFAULT_OUTPUT_DIR = Path("data/extracted_wikipedia_raw")

EXCLUDED_SECTION_TITLES = {
    "references",
    "external links",
    "notes",
    "explanatory notes",
    "citations",
    "sources",
    "bibliography",
    "further reading",
    "see also",
}


def extract_page_title(wikipedia_url: str) -> str:
    pattern = r"^https?://[a-z-]+\.wikipedia\.org/wiki/([^?#]+)"
    match = re.match(pattern, wikipedia_url.strip())

    if not match:
        raise ValueError(
            f"Invalid Wikipedia article URL: {wikipedia_url}"
        )

    return unquote(match.group(1))


def normalize_section_title(title: str) -> str:
    return re.sub(r"\s+", " ", title.strip().lower())


def should_exclude_section(title: str) -> bool:
    normalized_title = normalize_section_title(title)

    return normalized_title in EXCLUDED_SECTION_TITLES


def collect_section_text(
    section: wikipediaapi.WikipediaPageSection,
    depth: int = 2,
) -> list[str]:
    """
    Recursively collect section text while preserving heading hierarchy.

    depth=2 produces:
        ## Top-level section
        ### Subsection
        #### Deeper subsection
    """

    if should_exclude_section(section.title):
        return []

    parts: list[str] = []

    heading_level = min(depth, 6)
    heading_prefix = "#" * heading_level

    parts.append(f"{heading_prefix} {section.title}")

    section_text = section.text.strip()

    if section_text:
        parts.append(section_text)

    for subsection in section.sections:
        parts.extend(
            collect_section_text(
                subsection,
                depth=depth + 1,
            )
        )

    return parts


def build_clean_article_text(
    page: wikipediaapi.WikipediaPage,
) -> str:
    """
    Build cleaned article text while preserving section hierarchy.
    """

    parts: list[str] = []

    summary = page.summary.strip()

    if summary:
        parts.append(summary)

    for section in page.sections:
        parts.extend(
            collect_section_text(
                section,
                depth=2,
            )
        )

    return "\n\n".join(parts).strip()


def save_text_to_file(
    text: str,
    output_file: Path,
) -> None:
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file.write_text(
        text,
        encoding="utf-8",
    )


def retrieve_wikipedia_page(
    wikipedia_url: str,
    qid: str,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, object]:
    if not re.fullmatch(r"Q[1-9]\d*", qid):
        raise ValueError(f"Invalid Wikidata QID: {qid}")

    page_title = extract_page_title(wikipedia_url)

    wiki = wikipediaapi.Wikipedia(
        user_agent=USER_AGENT,
        language="en",
    )

    page = wiki.page(page_title)

    if not page.exists():
        raise ValueError(
            f"Wikipedia page '{page_title}' does not exist."
        )

    cleaned_text = build_clean_article_text(page)

    if not cleaned_text:
        raise ValueError(
            f"Wikipedia page '{page_title}' contains no usable text."
        )

    output_file = output_dir / f"{qid}.txt"

    save_text_to_file(
        text=cleaned_text,
        output_file=output_file,
    )

    return {
        "qid": qid,
        "title": page.title,
        "url": page.fullurl,
        "output_file": str(output_file),
        "character_count": len(cleaned_text),
    }


def main() -> None:
    battle_limit = 10

    if not BATTLES_CSV.exists():
        raise FileNotFoundError(
            f"Input CSV file was not found: {BATTLES_CSV.resolve()}"
        )

    successful_extractions = 0
    failed_extractions = 0

    with BATTLES_CSV.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        reader = csv.DictReader(csv_file)

        required_columns = {
            "qid",
            "wikipedia_url",
        }

        available_columns = set(reader.fieldnames or [])

        missing_columns = required_columns - available_columns

        if missing_columns:
            raise ValueError(
                "CSV file is missing required columns: "
                + ", ".join(sorted(missing_columns))
            )

        for row_number, row in enumerate(reader, start=1):
            if row_number > battle_limit:
                break

            qid = row["qid"].strip()
            wikipedia_url = row["wikipedia_url"].strip()

            print()
            print(
                f"[{row_number}/{battle_limit}] "
                f"Extracting {qid}"
            )
            print(f"Source: {wikipedia_url}")

            if not qid or not wikipedia_url:
                print("Extraction skipped: missing QID or Wikipedia URL.")
                failed_extractions += 1
                continue

            try:
                page_data = retrieve_wikipedia_page(
                    wikipedia_url=wikipedia_url,
                    qid=qid,
                )

                successful_extractions += 1

                print(f"Title: {page_data['title']}")
                print(f"URL: {page_data['url']}")
                print(
                    f"Characters: "
                    f"{page_data['character_count']}"
                )
                print(f"Saved to: {page_data['output_file']}")

            except (
                ValueError,
                OSError,
            ) as error:
                failed_extractions += 1
                print(f"Extraction failed for {qid}: {error}")

    print()
    print("Extraction completed.")
    print(f"Successful: {successful_extractions}")
    print(f"Failed: {failed_extractions}")


if __name__ == "__main__":
    start = perf_counter()

    try:
        main()
    finally:
        end = perf_counter()
        print(f"Execution time: {end - start:.6f} seconds")
        