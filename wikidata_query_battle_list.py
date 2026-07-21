from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import requests


SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
OUTPUT_CSV = Path("data/wikidata_battles.csv")

HEADERS = {
    "User-Agent": (
        "HistoricalBattlesThesis/1.0 "
        "(Master thesis; contact: vladyslav.romanov@tu-dresden.de)"
    ),
    "Accept": "application/sparql-results+json",
}

QUERY = """
SELECT DISTINCT ?item ?wikipediaArticle WHERE {
  ?item wdt:P31 wd:Q178561.

  ?wikipediaArticle schema:about ?item;
                    schema:isPartOf <https://en.wikipedia.org/>.
}
"""

MAX_RETRIES = 5


def execute_query(
    session: requests.Session,
    query: str,
) -> dict[str, Any]:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(
                SPARQL_ENDPOINT,
                params={
                    "query": query,
                    "format": "json",
                },
                headers=HEADERS,
                timeout=90,
            )

            if response.status_code == 429:
                retry_after = int(
                    response.headers.get("Retry-After", "10")
                )

                print(
                    f"Rate limited. Waiting {retry_after} seconds..."
                )

                time.sleep(retry_after)
                continue

            response.raise_for_status()
            return response.json()

        except (
            requests.Timeout,
            requests.ConnectionError,
            requests.HTTPError,
        ) as error:
            if attempt == MAX_RETRIES:
                raise RuntimeError(
                    "Wikidata query failed after all retries."
                ) from error

            wait_seconds = 2 ** attempt

            print(f"Attempt {attempt} failed: {error}")
            print(f"Retrying in {wait_seconds} seconds...")

            time.sleep(wait_seconds)

    raise RuntimeError("Unexpected end of retry loop.")


def extract_qid(wikidata_uri: str) -> str:
    return wikidata_uri.rsplit("/", maxsplit=1)[-1]


def parse_results(
    result: dict[str, Any],
) -> list[dict[str, str]]:
    records_by_qid: dict[str, dict[str, str]] = {}

    for binding in result["results"]["bindings"]:
        qid = extract_qid(binding["item"]["value"])

        records_by_qid[qid] = {
            "qid": qid,
            "wikipedia_url": binding["wikipediaArticle"]["value"],
        }

    return sorted(
        records_by_qid.values(),
        key=lambda record: int(record["qid"][1:]),
    )


def save_csv(
    records: list[dict[str, str]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "qid",
                "wikipedia_url",
            ],
        )

        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    print("Querying direct instances of battle...")

    with requests.Session() as session:
        result = execute_query(session, QUERY)

    records = parse_results(result)
    save_csv(records, OUTPUT_CSV)

    print(f"Unique items retrieved: {len(records)}")
    print(f"CSV saved to: {OUTPUT_CSV.resolve()}")


if __name__ == "__main__":
    main()