from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi


JSON_KB1_DIR = Path(
    "data/json_kb_v1"
)

JSON_KB2_DIR = Path(
    "data/json_kb_v2"
)


# ============================================================
# Document loading and serialization
# ============================================================


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
    use_kb_2: bool = False,
) -> list[dict[str, Any]]:
    """
    Load all Q*.json battle instances directly from the JSON knowledge base.
    """

    if use_kb_2:
        directory = JSON_KB2_DIR
    else:
        directory = JSON_KB1_DIR

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


# ============================================================
# Tokenization and BM25 index
# ============================================================


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


# ============================================================
# Retrieval
# ============================================================


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
                # Keep the corpus index so the result can be explained later.
                "document_index": index_position,
            }
        )

    return results


# ============================================================
# BM25 explanation helpers
# ============================================================


def explain_bm25_match(
    query: str,
    bm25: BM25Okapi,
    document_index: int,
) -> list[dict[str, Any]]:
    """
    Explain how much each matched query token contributes to the BM25 score
    for one document.

    The contribution is calculated using the same BM25 term-scoring formula
    and parameters stored in the BM25Okapi object.
    """

    query_tokens = tokenize_text(query)

    doc_freqs = bm25.doc_freqs[
        document_index
    ]
    doc_len = bm25.doc_len[
        document_index
    ]

    explanations: list[dict[str, Any]] = []

    for token in query_tokens:
        if token not in doc_freqs:
            continue

        term_frequency = doc_freqs[token]
        idf = bm25.idf.get(token, 0.0)

        denominator = (
            term_frequency
            + bm25.k1
            * (
                1
                - bm25.b
                + bm25.b
                * doc_len
                / bm25.avgdl
            )
        )

        contribution = (
            idf
            * term_frequency
            * (bm25.k1 + 1)
            / denominator
        )

        explanations.append(
            {
                "token": token,
                "term_frequency": term_frequency,
                "idf": idf,
                "contribution": contribution,
            }
        )

    explanations.sort(
        key=lambda item: item["contribution"],
        reverse=True,
    )

    return explanations


def find_token_in_json(
    data: Any,
    token: str,
    path: str = "",
) -> list[dict[str, str]]:
    """
    Find string values in a JSON object that contain a token.

    This is only for interpretation/visualization. BM25 itself still works on
    the flat serialized JSON text and does not use JSON fields explicitly.
    """

    matches: list[dict[str, str]] = []

    if isinstance(data, dict):
        for key, value in data.items():
            child_path = (
                f"{path}.{key}"
                if path
                else key
            )

            matches.extend(
                find_token_in_json(
                    data=value,
                    token=token,
                    path=child_path,
                )
            )

    elif isinstance(data, list):
        for index, value in enumerate(data):
            child_path = f"{path}[{index}]"

            matches.extend(
                find_token_in_json(
                    data=value,
                    token=token,
                    path=child_path,
                )
            )

    elif isinstance(data, str):
        if token in tokenize_text(data):
            matches.append(
                {
                    "path": path,
                    "value": data,
                }
            )

    return matches


def print_bm25_explanation(
    query: str,
    bm25: BM25Okapi,
    result: dict[str, Any],
    max_terms: int | None = 10,
    max_field_matches_per_term: int = 5,
) -> None:
    """
    Print a human-readable explanation for one ranked BM25 result.

    For each matching query token, prints:
    - term frequency in the battle document;
    - IDF;
    - BM25 contribution;
    - JSON fields in which that token appears.
    """

    document_index = result["document_index"]

    explanation = explain_bm25_match(
        query=query,
        bm25=bm25,
        document_index=document_index,
    )

    if max_terms is not None:
        explanation = explanation[:max_terms]

    print()
    print("=" * 80)
    print(
        f"BM25 EXPLANATION FOR RANK {result['rank']}: "
        f"{result['name']} ({result['battle_id']})"
    )
    print(
        f"Total BM25 score: {result['score']:.4f}"
    )
    print("=" * 80)

    if not explanation:
        print("No query tokens matched this document.")
        return

    print()
    print("Matched query terms:")
    print()

    for item in explanation:
        print(
            f"{item['token']:<20} "
            f"TF={item['term_frequency']:<4} "
            f"IDF={item['idf']:.4f} "
            f"Contribution={item['contribution']:.4f}"
        )

        field_matches = find_token_in_json(
            data=result["json"],
            token=item["token"],
        )

        if field_matches:
            for match in field_matches[
                :max_field_matches_per_term
            ]:
                value = match["value"].replace(
                    "\n",
                    " ",
                )

                if len(value) > 180:
                    value = value[:177] + "..."

                print(
                    f"    {match['path']} -> {value}"
                )
        else:
            # This can happen when the matching token comes from a JSON key
            # rather than a string value, because full serialized JSON is indexed.
            print(
                "    Match may come from a JSON field name/key "
                "rather than a string value."
            )

        print()


# ============================================================
# Manual test
# ============================================================


def test_bm25(
    candidate_k: int = 10,
    query: str | None = None,
    limit: int | None = None,
    use_kb_2: bool = False,
    explain_rank: int | None = 1,
    explanation_max_terms: int | None = 10,
) -> list[dict[str, Any]]:
    """
    Load the JSON knowledge base, build BM25 over full JSON records,
    run a test query, print ranked results, and optionally explain one result.

    Parameters
    ----------
    candidate_k:
        Number of ranked candidates to return.

    query:
        Natural-language retrieval query.

    limit:
        Optional number of first documents to load for a quick test.
        Leave as None to use the complete knowledge base.

    use_kb_2:
        If True, load data/json_kb_v2 instead of data/json_kb_v1.

    explain_rank:
        Rank to explain after retrieval, e.g. 1 for the top result,
        3 for the third result. Set to None to disable explanation.

    explanation_max_terms:
        Maximum number of matched query terms to print in the explanation.
        Set to None to print all matching terms.
    """

    if query is None:
        raise ValueError(
            "A query must be provided."
        )

    documents = load_json_battles(
        use_kb_2=use_kb_2
    )

    if limit is not None:
        documents = documents[:limit]

    print(
        f"Loaded JSON battle instances: "
        f"{len(documents)}"
    )

    print(
        f"Knowledge base: "
        f"{'KB2' if use_kb_2 else 'KB1'}"
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

    if explain_rank is not None:
        if explain_rank < 1:
            raise ValueError(
                "explain_rank must be at least 1 or None."
            )

        if explain_rank > len(results):
            raise ValueError(
                f"Cannot explain rank {explain_rank}; "
                f"only {len(results)} results were returned."
            )

        selected_result = results[
            explain_rank - 1
        ]

        print_bm25_explanation(
            query=query,
            bm25=bm25,
            result=selected_result,
            max_terms=explanation_max_terms,
        )

    return results


if __name__ == "__main__":
    query = (
        "Find battles where war elephants participated in a major attacking charge but the charge was stopped by disciplined infantry, artillery, or reserve forces and failed to secure victory."
    )

    test_bm25(
        candidate_k=10,
        query=query,
        use_kb_2=True,
        explain_rank=3,
        explanation_max_terms=10,
    )
    
