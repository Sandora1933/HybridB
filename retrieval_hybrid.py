from __future__ import annotations

from typing import Any


RRF_K = 60


def reciprocal_rank_fusion(
    bm25_results: list[dict[str, Any]],
    dense_results: list[dict[str, Any]],
    rrf_k: int = RRF_K,
) -> list[dict[str, Any]]:
    """
    Combine BM25 and dense rankings using Reciprocal Rank Fusion.

    RRF(d) = sum(1 / (k + rank(d)))
    """

    fused: dict[str, dict[str, Any]] = {}

    # BM25 contribution
    for rank, result in enumerate(
        bm25_results,
        start=1,
    ):
        battle_id = result["battle_id"]

        if battle_id not in fused:
            fused[battle_id] = {
                "battle_id": battle_id,
                "name": result.get("name"),
                "rrf_score": 0.0,
                "bm25_rank": None,
                "dense_rank": None,
                "bm25_score": None,
                "dense_score": None,
            }

        fused[battle_id]["rrf_score"] += (
            1.0 / (rrf_k + rank)
        )

        fused[battle_id]["bm25_rank"] = rank
        fused[battle_id]["bm25_score"] = result.get(
            "score"
        )

    # Dense contribution
    for rank, result in enumerate(
        dense_results,
        start=1,
    ):
        battle_id = result["battle_id"]

        if battle_id not in fused:
            fused[battle_id] = {
                "battle_id": battle_id,
                "name": result.get("name"),
                "rrf_score": 0.0,
                "bm25_rank": None,
                "dense_rank": None,
                "bm25_score": None,
                "dense_score": None,
            }

        fused[battle_id]["rrf_score"] += (
            1.0 / (rrf_k + rank)
        )

        fused[battle_id]["dense_rank"] = rank
        fused[battle_id]["dense_score"] = result.get(
            "score"
        )

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


def test_hybrid(
    bm25_results: list[dict[str, Any]],
    dense_results: list[dict[str, Any]],
    top_k: int = 10,
    rrf_k: int = RRF_K,
) -> list[dict[str, Any]]:
    """
    Fuse already retrieved BM25 and dense result lists.

    No retrieval is performed here.
    """

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than 0."
        )

    fused_results = reciprocal_rank_fusion(
        bm25_results=bm25_results,
        dense_results=dense_results,
        rrf_k=rrf_k,
    )

    results = fused_results[:top_k]

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

    return results