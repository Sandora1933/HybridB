from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi


RETRIEVAL_DOCS_FILE = Path(
    "data/retrieval_docs/retrieval_docs_v1.jsonl"
)


def load_retrieval_docs(
    file_path: Path = RETRIEVAL_DOCS_FILE,
) -> list[dict[str, Any]]:
    """
    Load retrieval documents from a JSONL file.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Retrieval docs file not found: "
            f"{file_path.resolve()}"
        )

    documents: list[dict[str, Any]] = []

    with file_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                document = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at line {line_number}: {error}"
                ) from error

            documents.append(document)

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
    Build a BM25 index over the retrieval document text fields.
    """

    tokenized_corpus = []

    for document in documents:
        text = document.get("text", "")

        if not isinstance(text, str):
            text = ""

        tokenized_corpus.append(
            tokenize_text(text)
        )

    # Build the BM25 index for my retrieval documents
    bm25 = BM25Okapi(tokenized_corpus)

    return bm25, tokenized_corpus


def retrieve_bm25(
    query: str,
    bm25: BM25Okapi,
    documents: list[dict[str, Any]],
    candidate_k: int,
) -> list[dict[str, Any]]:
    """
    Retrieve the candidate-k battle documents using BM25.
    """

    if not query.strip():
        raise ValueError("Query must not be empty.")

    if candidate_k <= 0:
        raise ValueError("candidate_k must be greater than 0.")

    tokenized_query = tokenize_text(query)

    scores = bm25.get_scores(tokenized_query)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )[:candidate_k]

    results = []

    for rank, index in enumerate(
        ranked_indices,
        start=1,
    ):
        document = documents[index]

        results.append(
            {
                "rank": rank,
                "score": float(scores[index]),
                "battle_id": document.get("battle_id"),
                "name": document.get("name"),
                "text": document.get("text"),
                "metadata": document.get("metadata"),
            }
        )

    return results


def test_bm25(candidate_k: int = 10) -> None:
    # collections of retrieval documents as jsons
    documents = load_retrieval_docs()

    print(
        f"Loaded retrieval documents: {len(documents)}"
    )

    bm25, _ = build_bm25_index(documents)
    query = "Find battles involving war elephants where the elephants were used in the main battle but did not secure victory."

    results = retrieve_bm25(
        query=query,
        bm25=bm25,
        documents=documents,
        candidate_k=candidate_k,
    )

    print(f"\nQuery: {query}\n")

    for result in results:
        print(
            f"{result['rank']}. "
            f"{result['name']} "
            f"({result['battle_id']})"
        )
        print(f"   BM25 score: {result['score']:.4f}\n")


if __name__ == "__main__":
    test_bm25(candidate_k=30)