from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi


JSON_KB_DIR = Path(
    "data/json_kb_v1"
)


def json_to_text(
    battle: dict[str, Any],
) -> str:
    """
    Serialize a full battle JSON object directly to text for BM25.

    No intermediate retrieval-document projection is created.
    Compact JSON is used to reduce unnecessary whitespace while
    preserving all keys and values.
    """

    return json.dumps(
        battle,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def load_json_battles(
    directory: Path = JSON_KB_DIR,
) -> list[dict[str, Any]]:
    """
    Load all Q*.json battle instances directly from the JSON knowledge base.
    """

    if not directory.exists():
        raise FileNotFoundError(
            f"JSON knowledge-base directory not found: "
            f"{directory.resolve()}"
        )

    json_files = sorted(
        directory.glob("Q*.json"),
        key=lambda path: int(path.stem[1:]),
    )

    documents: list[dict[str, Any]] = []

    for file_path in json_files:
        try:
            with file_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                battle = json.load(file)

        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid JSON file: {file_path} "
                f"({error})"
            ) from error

        if not isinstance(battle, dict):
            raise ValueError(
                f"Expected JSON object in {file_path}, "
                f"got {type(battle).__name__}."
            )

        battle_id = file_path.stem

        identification = battle.get(
            "identification",
            {},
        )

        if not isinstance(identification, dict):
            identification = {}

        name = identification.get("name")

        documents.append(
            {
                "battle_id": battle_id,
                "name": name,
                "json": battle,
                "text": json_to_text(battle),
            }
        )

    return documents


def tokenize_text(text: str) -> list[str]:
    """
    Simple tokenizer for BM25.

    Converts text to lowercase and extracts alphanumeric tokens.
    """

    return re.findall(
        r"\b\w+\b",
        text.lower(),
        flags=re.UNICODE,
    )


def build_bm25_index(
    documents: list[dict[str, Any]],
) -> tuple[BM25Okapi, list[list[str]]]:
    """
    Build a BM25 index directly over complete serialized battle JSON objects.
    """

    tokenized_corpus: list[list[str]] = []

    for document in documents:
        text = document.get(
            "text",
            "",
        )

        if not isinstance(text, str):
            text = ""

        tokenized_corpus.append(
            tokenize_text(text)
        )

    bm25 = BM25Okapi(
        tokenized_corpus
    )

    return bm25, tokenized_corpus


def retrieve_bm25(
    query: str,
    bm25: BM25Okapi,
    documents: list[dict[str, Any]],
    candidate_k: int,
) -> list[dict[str, Any]]:
    """
    Retrieve candidate-k battle JSON instances using BM25.
    """

    if not query.strip():
        raise ValueError(
            "Query must not be empty."
        )

    if candidate_k <= 0:
        raise ValueError(
            "candidate_k must be greater than 0."
        )

    candidate_k = min(
        candidate_k,
        len(documents),
    )

    tokenized_query = tokenize_text(
        query
    )

    scores = bm25.get_scores(
        tokenized_query
    )

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )[:candidate_k]

    results: list[dict[str, Any]] = []

    for rank, index_position in enumerate(
        ranked_indices,
        start=1,
    ):
        document = documents[
            index_position
        ]

        results.append(
            {
                "rank": rank,
                "score": float(
                    scores[index_position]
                ),
                "battle_id": document.get(
                    "battle_id"
                ),
                "name": document.get(
                    "name"
                ),
                "json": document.get(
                    "json"
                ),
            }
        )

    return results


def test_bm25(
    candidate_k: int = 10,
    query: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """
    Load the JSON knowledge base, build BM25 directly over full JSON records,
    run a test query, print results, and return the ranked result list.

    Set limit=10 for a quick test.
    Leave limit=None to use the full knowledge base.
    """

    documents = load_json_battles()

    if limit is not None:
        documents = documents[:limit]

    print(
        f"Loaded JSON battle instances: "
        f"{len(documents)}"
    )

    bm25, _ = build_bm25_index(
        documents
    )

    results = retrieve_bm25(
        query=query,
        bm25=bm25,
        documents=documents,
        candidate_k=candidate_k,
    )

    print()
    print(f"Query: {query}")
    print()

    for result in results:
        print(
            f"{result['rank']}. "
            f"{result['name']} "
            f"({result['battle_id']})"
        )

        print(
            f"   BM25 score: "
            f"{result['score']:.4f}"
        )

        print()

    return results


if __name__ == "__main__":
    query = (
        "Find battles involving war elephants "
        "where the elephants were used in the main battle "
        "but did not secure victory."
    )

    test_bm25(
        candidate_k=30,
        query=query,
    )
