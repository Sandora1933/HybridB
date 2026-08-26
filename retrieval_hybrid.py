from __future__ import annotations

from typing import Any

from retrieval_bm25 import (
    load_retrieval_docs,
    build_bm25_index,
    retrieve_bm25,
)

from retrieval_dense import (
    load_embedding_model,
    load_dense_index,
    dense_search,
)


RRF_K = 60


def reciprocal_rank_fusion(
    bm25_results: list[dict[str, Any]],
    dense_results: list[dict[str, Any]],
    rrf_k: int = RRF_K,
) -> list[dict[str, Any]]:
    """
    Combine BM25 and dense rankings using
    Reciprocal Rank Fusion.

    RRF(d) = sum(1 / (k + rank(d)))
    """

    fused: dict[str, dict[str, Any]] = {}

    # Add BM25 results
    for rank, result in enumerate(
        bm25_results,
        start=1,
    ):
        battle_id = result["battle_id"]

        if battle_id not in fused:
            fused[battle_id] = {
                "battle_id": battle_id,
                "name": result.get("name"),
                "text": result.get("text"),
                "metadata": result.get(
                    "metadata",
                    {},
                ),
                "rrf_score": 0.0,
                "bm25_rank": None,
                "dense_rank": None,
                "bm25_score": None,
                "dense_score": None,
            }

        fused[battle_id][
            "rrf_score"
        ] += 1.0 / (
            rrf_k + rank
        )

        fused[battle_id][
            "bm25_rank"
        ] = rank

        fused[battle_id][
            "bm25_score"
        ] = result["score"]

    # Add dense results
    for rank, result in enumerate(
        dense_results,
        start=1,
    ):
        battle_id = result["battle_id"]

        if battle_id not in fused:
            fused[battle_id] = {
                "battle_id": battle_id,
                "name": result.get("name"),
                "text": result.get("text"),
                "metadata": result.get(
                    "metadata",
                    {},
                ),
                "rrf_score": 0.0,
                "bm25_rank": None,
                "dense_rank": None,
                "bm25_score": None,
                "dense_score": None,
            }

        fused[battle_id][
            "rrf_score"
        ] += 1.0 / (
            rrf_k + rank
        )

        fused[battle_id][
            "dense_rank"
        ] = rank

        fused[battle_id][
            "dense_score"
        ] = result["score"]

    ranked_results = sorted(
        fused.values(),
        key=lambda result: result["rrf_score"],
        reverse=True,
    )

    for rank, result in enumerate(
        ranked_results,
        start=1,
    ):
        result["rank"] = rank

    return ranked_results


def retrieve_hybrid(
    query: str,
    bm25,
    bm25_documents: list[dict[str, Any]],
    embedding_model,
    dense_index,
    dense_documents: list[dict[str, Any]],
    top_k: int,
    candidate_k: int,
) -> list[dict[str, Any]]:
    """
    Hybrid retrieval pipeline:

    BM25 top candidate_k
            +
    Dense top candidate_k
            ↓
            RRF
            ↓
        final top_k
    """

    if not query.strip():
        raise ValueError(
            "Query must not be empty."
        )

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than 0."
        )

    if candidate_k < top_k:
        raise ValueError(
            "candidate_k must be >= top_k."
        )

    bm25_results = retrieve_bm25(
        query=query,
        bm25=bm25,
        documents=bm25_documents,
        candidate_k=candidate_k,
    )

    dense_results = dense_search(
        query=query,
        model=embedding_model,
        index=dense_index,
        docs=dense_documents,
        candidate_k=candidate_k,
    )

    fused_results = reciprocal_rank_fusion(
        bm25_results=bm25_results,
        dense_results=dense_results,
    )

    return fused_results[:top_k]


def test_hybrid(
    query: str,
    top_k: int = 10,
    candidate_k: int = 50,
) -> None:
    """
    Load BM25 and dense retrieval components
    and run a hybrid test query.
    """

    # -------------------------
    # Load BM25
    # -------------------------

    bm25_documents = load_retrieval_docs()

    print(
        f"Loaded BM25 documents: "
        f"{len(bm25_documents)}"
    )

    bm25, _ = build_bm25_index(
        bm25_documents
    )

    print(
        "BM25 index created."
    )

    # -------------------------
    # Load dense retrieval
    # -------------------------

    embedding_model = load_embedding_model()

    dense_index, dense_documents = (
        load_dense_index()
    )

    print(
        f"Loaded dense vectors: "
        f"{dense_index.ntotal}"
    )

    # -------------------------
    # Hybrid search
    # -------------------------

    results = retrieve_hybrid(
        query=query,
        bm25=bm25,
        bm25_documents=bm25_documents,
        embedding_model=embedding_model,
        dense_index=dense_index,
        dense_documents=dense_documents,
        top_k=top_k,
        candidate_k=candidate_k,
    )

    print()
    print(
        f"Query: {query}"
    )
    print()

    for result in results:

        print(
            f"{result['rank']}. "
            f"{result['name']} "
            f"({result['battle_id']})"
        )

        print(
            f"   RRF score: "
            f"{result['rrf_score']:.6f}"
        )

        print(
            f"   BM25 rank: "
            f"{result['bm25_rank']}"
        )

        print(
            f"   Dense rank: "
            f"{result['dense_rank']}"
        )

        if result["bm25_score"] is not None:
            print(
                f"   BM25 score: "
                f"{result['bm25_score']:.4f}"
            )

        if result["dense_score"] is not None:
            print(
                f"   Dense score: "
                f"{result['dense_score']:.4f}"
            )

        print()


if __name__ == "__main__":

    query = (
        "Find battles involving Napoleon where "
        "the French initially planned offensive action "
        "but were eventually encircled or forced into retreat."
    )

    test_hybrid(
        query=query,
        top_k=10,
        candidate_k=50,
    )