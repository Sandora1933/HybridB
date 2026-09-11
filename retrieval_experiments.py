from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Literal

from matplotlib import pyplot as plt
import numpy as np

from retrieval_bm25 import (
    load_json_battles,
    build_bm25_index,
    retrieve_bm25,
)

from retrieval_dense import (
    load_embedding_model,
    load_dense_index,
    dense_search,
)

from retrieval_hybrid import reciprocal_rank_fusion 

QUERIES_FILE = Path(
    "data/queries/queries_all.json"
)

REPORT_FILE_KB1 = Path(
    "data/evaluation/retrieval_experiment_report_kb1.json"
)

REPORT_FILE_KB2 = Path(
    "data/evaluation/retrieval_experiment_report_kb2.json"
)

AVERAGE_METRICS_FILE_KB1 = Path(
    "data/evaluation/retrieval_experiment_average_metrics_kb1.json"
)

AVERAGE_METRICS_FILE_KB2 = Path(
    "data/evaluation/retrieval_experiment_average_metrics_kb2.json"
)


CANDIDATE_K = 100
HYBRID_TOP_K = 50


def _filter_queries_by_rarity(
    queries: list[dict[str, Any]],
    rarity: Literal["common", "rare"],
) -> list[dict[str, Any]]:
    return [
        query
        for query in queries
        if query.get("context_rarity") == rarity
    ]


def _filter_queries_by_difficulty(
    queries: list[dict[str, Any]],
    difficulty: Literal["easy", "medium", "hard"],
) -> list[dict[str, Any]]:
    """Filter queries by difficulty: 'easy', 'medium', or 'hard'."""
    return [
        query
        for query in queries
        if query.get("query_difficulty") == difficulty
    ]

def _filter_queries_by_type(
    queries: list[dict[str, Any]],
    query_type: Literal[
        "Participant-related",
        "Similarity",
        "Narrative / Tactical",
        "Temporal",
        "Spatial / Terrain",
        "Outcome-related",
        "Weapon / Unit-related",
    ],
) -> list[dict[str, Any]]:
    """Filter queries by query type."""
    return [
        query
        for query in queries
        if query.get("query_type") == query_type
    ]

def _extract_battle_ids(
    retrieved_battles: list[Any],
) -> list[str]:
    """
    Extract battle QIDs from ranked retrieval results.
    """

    battle_ids: list[str] = []

    for item in retrieved_battles:
        if isinstance(item, dict):
            battle_id = item.get("battle_id")

            if isinstance(battle_id, str):
                battle_ids.append(
                    battle_id.strip()
                )

    return battle_ids


def _prepare_sets(
    retrieved_battles: list[Any],
    golden_battles: dict[str, str],
) -> tuple[list[str], set[str]]:
    """
    Prepare retrieved QID ranking and gold QID set.

    Evaluation is performed by QID rather than battle name.
    """

    retrieved_ids = _extract_battle_ids(
        retrieved_battles
    )

    gold_ids = set(
        golden_battles.keys()
    )

    return retrieved_ids, gold_ids

def eval_recall_at_k(
    algorithm_retrieved_battles_list: list[Any],
    golden_battles_list: dict[str, str],
    k: int,
) -> float:
    """
    Recall@k.
    """

    if not golden_battles_list:
        return 0.0

    retrieved, gold = _prepare_sets(
        algorithm_retrieved_battles_list,
        golden_battles_list,
    )

    retrieved_top_k = set(
        retrieved[:k]
    )

    relevant_retrieved = len(
        retrieved_top_k & gold
    )

    return relevant_retrieved / len(gold)


def eval_recall_10(
    algorithm_retrieved_battles_list: list[Any],
    golden_battles_list: dict[str, str],
) -> float:
    return eval_recall_at_k(
        algorithm_retrieved_battles_list,
        golden_battles_list,
        k=10,
    )


def eval_recall_20(
    algorithm_retrieved_battles_list: list[Any],
    golden_battles_list: dict[str, str],
) -> float:
    return eval_recall_at_k(
        algorithm_retrieved_battles_list,
        golden_battles_list,
        k=20,
    )


def eval_recall_50(
    algorithm_retrieved_battles_list: list[Any],
    golden_battles_list: dict[str, str],
) -> float:
    return eval_recall_at_k(
        algorithm_retrieved_battles_list,
        golden_battles_list,
        k=50,
    )


def eval_precision_10(
    algorithm_retrieved_battles_list: list[Any],
    golden_battles_list: dict[str, str],
) -> float:
    """
    Precision@10.

    Since experiments retrieve at least 50 candidates,
    the denominator is always 10.
    """

    retrieved, gold = _prepare_sets(
        algorithm_retrieved_battles_list,
        golden_battles_list,
    )

    retrieved_top_10 = retrieved[:10]

    relevant_retrieved = sum(
        1
        for battle in retrieved_top_10
        if battle in gold
    )

    return relevant_retrieved / 10.0


def eval_mrr(
    algorithm_retrieved_battles_list: list[Any],
    golden_battles_list: dict[str, str],
) -> float:
    """
    Reciprocal Rank for one query.

    Averaging this value across all queries gives MRR.
    """

    retrieved, gold = _prepare_sets(
        algorithm_retrieved_battles_list,
        golden_battles_list,
    )

    for rank, battle in enumerate(
        retrieved,
        start=1,
    ):
        if battle in gold:
            return 1.0 / rank

    return 0.0


def eval_map_10(
    algorithm_retrieved_battles_list: list[Any],
    golden_battles_list: dict[str, str],
) -> float:
    """
    Average Precision@10 for one query.

    Averaging this value across all queries gives MAP@10.
    """

    if not golden_battles_list:
        return 0.0

    retrieved, gold = _prepare_sets(
        algorithm_retrieved_battles_list,
        golden_battles_list,
    )

    retrieved_top_10 = retrieved[:10]

    relevant_seen = 0
    precision_sum = 0.0

    for rank, battle in enumerate(
        retrieved_top_10,
        start=1,
    ):
        if battle in gold:
            relevant_seen += 1
            precision_sum += (
                relevant_seen / rank
            )

    denominator = min(
        len(gold),
        10,
    )

    if denominator == 0:
        return 0.0

    return precision_sum / denominator


def eval_ndcg_10(
    algorithm_retrieved_battles_list: list[Any],
    golden_battles_list: dict[str, str],
) -> float:
    """
    Binary nDCG@10.
    """

    if not golden_battles_list:
        return 0.0

    retrieved, gold = _prepare_sets(
        algorithm_retrieved_battles_list,
        golden_battles_list,
    )

    retrieved_top_10 = retrieved[:10]

    dcg = 0.0

    for rank, battle in enumerate(
        retrieved_top_10,
        start=1,
    ):
        if battle in gold:
            dcg += (
                1.0
                / math.log2(rank + 1)
            )

    ideal_relevant_count = min(
        len(gold),
        10,
    )

    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(
            1,
            ideal_relevant_count + 1,
        )
    )

    if idcg == 0:
        return 0.0

    return dcg / idcg


def evaluate_query(
    algorithm_retrieved_battles_list: list[Any],
    golden_battles_list: dict[str, str],
) -> dict[str, float]:
    """
    Calculate all retrieval metrics for one query.
    """

    return {
        "Recall@10": eval_recall_10(
            algorithm_retrieved_battles_list,
            golden_battles_list,
        ),
        "Recall@20": eval_recall_20(
            algorithm_retrieved_battles_list,
            golden_battles_list,
        ),
        "Recall@50": eval_recall_50(
            algorithm_retrieved_battles_list,
            golden_battles_list,
        ),
        "MRR": eval_mrr(
            algorithm_retrieved_battles_list,
            golden_battles_list,
        ),
        "MAP@10": eval_map_10(
            algorithm_retrieved_battles_list,
            golden_battles_list,
        ),
        "nDCG@10": eval_ndcg_10(
            algorithm_retrieved_battles_list,
            golden_battles_list,
        ),
    }


def format_ranking(
    results: list[dict[str, Any]],
) -> list[str]:
    """
    Convert ranked output to:
        ["Battle Name (QID)", ...]

    List position corresponds to rank.
    """

    return [
        f"{result.get('name')} "
        f"({result.get('battle_id')})"
        for result in results
    ]


def single_query_retrieval_report(
    query_item: dict[str, Any],
    bm25_results: list[dict[str, Any]],
    dense_results: list[dict[str, Any]],
    hybrid_results: list[dict[str, Any]],
    bm25_metrics: dict[str, float],
    dense_metrics: dict[str, float],
    hybrid_metrics: dict[str, float],
) -> dict[str, Any]:
    """
    Build one complete query report.
    """

    return {
        "query_id": query_item.get("query_id"),
        "query": query_item.get("query"),
        "generation_query": query_item.get("generation_query"),
        "query_type": query_item.get("query_type"),
        "query_structure": query_item.get("query_structure"),
        "query_difficulty": query_item.get("query_difficulty"),
        "context_rarity": query_item.get("context_rarity"),
        "gold_battles": query_item.get(
            "gold_battles",
            {},
        ),
        "closed_set_negatives": query_item.get(
            "closed_set_negatives",
            {},
        ),
        "explanation": query_item.get("explanation"),
        "results": [
            {
                "algorithm": "bm25",
                "ranking": format_ranking(
                    bm25_results
                ),
                "metrics": bm25_metrics,
            },
            {
                "algorithm": "dense",
                "ranking": format_ranking(
                    dense_results
                ),
                "metrics": dense_metrics,
            },
            {
                "algorithm": "hybrid",
                "ranking": format_ranking(
                    hybrid_results
                ),
                "metrics": hybrid_metrics,
            },
        ],
    }


def save_retrieval_report(
    reports: list[dict[str, Any]],
    output_file: Path = REPORT_FILE_KB1,
) -> None:
    """
    Save all completed query reports.
    """

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            reports,
            file,
            ensure_ascii=False,
            indent=2,
        )


def load_queries(
    file_path: Path = QUERIES_FILE,
) -> list[dict[str, Any]]:
    """
    Load retrieval evaluation queries.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Queries file not found: "
            f"{file_path.resolve()}"
        )

    with file_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        queries = json.load(file)

    if not isinstance(queries, list):
        raise ValueError(
            "Queries file must contain a JSON list."
        )

    return queries


def run_single_query_experiment(
    query_item: dict[str, Any],
    bm25,
    bm25_documents: list[dict[str, Any]],
    embedding_model,
    dense_index,
    dense_documents: list[dict[str, Any]],
    candidate_k: int = CANDIDATE_K,
    hybrid_top_k: int = HYBRID_TOP_K,
) -> dict[str, Any]:
    """
    Run BM25, dense, and hybrid retrieval for one query.

    BM25 and dense retrieval each return candidate_k results.
    RRF fuses those rankings and keeps only hybrid_top_k results.

    BM25 index, dense model, FAISS index, and document mappings
    are provided by main() and reused across all queries.
    """

    if candidate_k < hybrid_top_k:
        raise ValueError(
            "candidate_k must be greater than or equal to "
            "hybrid_top_k."
        )

    if hybrid_top_k < 50:
        raise ValueError(
            "hybrid_top_k must be at least 50 because "
            "Recall@50 is evaluated for the hybrid strategy."
        )

    query = query_item["query"]
    golden_battles = query_item["gold_battles"]

    # BM25 search only. No BM25 index rebuild.
    bm25_results = retrieve_bm25(
        query=query,
        bm25=bm25,
        documents=bm25_documents,
        candidate_k=candidate_k,
    )

    # Dense query search only. Existing FAISS index is reused.
    # Only the query itself is embedded through the API.
    dense_results = dense_search(
        query=query,
        model=embedding_model,
        index=dense_index,
        docs=dense_documents,
        candidate_k=candidate_k,
    )

    # Hybrid RRF over the already-computed rankings.
    hybrid_results = reciprocal_rank_fusion(
        bm25_results=bm25_results,
        dense_results=dense_results,
    )[:hybrid_top_k]

    bm25_metrics = evaluate_query(
        algorithm_retrieved_battles_list=bm25_results,
        golden_battles_list=golden_battles,
    )

    dense_metrics = evaluate_query(
        algorithm_retrieved_battles_list=dense_results,
        golden_battles_list=golden_battles,
    )

    hybrid_metrics = evaluate_query(
        algorithm_retrieved_battles_list=hybrid_results,
        golden_battles_list=golden_battles,
    )

    return single_query_retrieval_report(
        query_item=query_item,
        bm25_results=bm25_results,
        dense_results=dense_results,
        hybrid_results=hybrid_results,
        bm25_metrics=bm25_metrics,
        dense_metrics=dense_metrics,
        hybrid_metrics=hybrid_metrics,
    )


def calculate_average_metrics(
    reports: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """
    Calculate mean metric values across all completed queries.
    """

    if not reports:
        return {}

    totals: dict[str, dict[str, float]] = {}

    for report in reports:
        for algorithm_result in report["results"]:
            algorithm = algorithm_result["algorithm"]
            metrics = algorithm_result["metrics"]

            if algorithm not in totals:
                totals[algorithm] = {
                    metric_name: 0.0
                    for metric_name in metrics
                }

            for metric_name, value in metrics.items():
                totals[algorithm][metric_name] += value

    number_of_queries = len(reports)

    return {
        algorithm: {
            metric_name: value / number_of_queries
            for metric_name, value in metrics.items()
        }
        for algorithm, metrics in totals.items()
    }


def plot_metric(
    metric_name: str,
    kb1_file: Path = AVERAGE_METRICS_FILE_KB1,
    #kb2_file: Path = AVERAGE_METRICS_FILE_KB2,
) -> None:
    """Plot a selected retrieval metric for BM25, dense, and hybrid."""

    with kb1_file.open("r", encoding="utf-8") as f:
        kb1_results = json.load(f)

    #with kb2_file.open("r", encoding="utf-8") as f:
    #    kb2_results = json.load(f)

    labels = ["KB1", "KB2"]
    methods = ["bm25", "dense", "hybrid"]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, method in enumerate(methods):
        values = [
            kb1_results[method][metric_name],
            #kb2_results[method][metric_name],
        ]

        ax.bar(
            x + (i - 1) * width,
            values,
            width,
            label=method,
        )

    ax.set_ylabel(metric_name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.legend()

    fig.tight_layout()
    plt.show()


def main() -> None:
    """
    Run all retrieval experiments.

    Initialization is done once:
        1. Load JSON KB and build BM25 once.
        2. Load Qwen embedding client once.
        3. Load the existing FAISS direct-JSON index once.
        4. Reuse all of them for every query.

    Retrieval depths:
        BM25 candidate_k = 100
        Dense candidate_k = 100
        Hybrid RRF top_k = 50
    """

    queries = load_queries()
    queries = _filter_queries_by_rarity(
        queries=queries,
        rarity="common", # or rare
    )
    
    use_kb_2 = False

    print(
        f"Loaded evaluation queries: "
        f"{len(queries)}"
    )

    # -------------------------------------------------
    # Initialize BM25 once
    # -------------------------------------------------

    print("\nLoading JSON knowledge base for BM25...")

    bm25_documents = load_json_battles(use_kb_2=use_kb_2)

    print(
        f"Loaded BM25 documents: "
        f"{len(bm25_documents)}"
    )

    print("Building BM25 index once...")

    bm25, _ = build_bm25_index(
        bm25_documents
    )

    # -------------------------------------------------
    # Load existing dense index once
    # -------------------------------------------------

    print("\nLoading embedding model/client...")

    embedding_model = load_embedding_model()

    print("Loading existing FAISS dense index...")

    dense_index, dense_documents = (
        load_dense_index(use_kb_2=use_kb_2)
    )

    print(
        f"Loaded dense vectors: "
        f"{dense_index.ntotal}"
    )

    if dense_index.ntotal != len(
        dense_documents
    ):
        raise ValueError(
            "Dense index/document mapping mismatch: "
            f"{dense_index.ntotal} vectors vs "
            f"{len(dense_documents)} documents."
        )

    # -------------------------------------------------
    # Run all queries
    # -------------------------------------------------

    all_reports: list[dict[str, Any]] = []

    for index, query_item in enumerate(
        queries,
        start=1,
    ):
        query_id = query_item.get(
            "query_id",
            f"q{index:03d}",
        )

        print()
        print("=" * 80)
        print(
            f"Running query "
            f"{index}/{len(queries)} "
            f"({query_id})"
        )
        print(query_item["query"])
        print("=" * 80)

        query_report = run_single_query_experiment(
            query_item=query_item,
            bm25=bm25,
            bm25_documents=bm25_documents,
            embedding_model=embedding_model,
            dense_index=dense_index,
            dense_documents=dense_documents,
            candidate_k=CANDIDATE_K,
            hybrid_top_k=HYBRID_TOP_K,
        )

        all_reports.append(
            query_report
        )

        if use_kb_2:
            report_file = REPORT_FILE_KB2
            average_metrics_file = AVERAGE_METRICS_FILE_KB2
        else:
            report_file = REPORT_FILE_KB1
            average_metrics_file = AVERAGE_METRICS_FILE_KB1

        # Save after each completed query.
        save_retrieval_report(all_reports, output_file=report_file)

        print(
            f"Saved {len(all_reports)} "
            f"completed query reports."
        )

    # -------------------------------------------------
    # Print overall averages
    # -------------------------------------------------

    average_metrics = calculate_average_metrics(
        all_reports
    )

    print()
    print("=" * 80)
    print("Average metrics")
    print("=" * 80)

    for algorithm, metrics in average_metrics.items():
        print(f"\n{algorithm}:")

        for metric_name, value in metrics.items():
            print(
                f"  {metric_name}: "
                f"{value:.4f}"
            )

    print()
    print(
        f"Final report saved to: "
        f"{report_file.resolve()}"
    )

    # Save average metrics to JSON
    average_metrics_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with average_metrics_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            average_metrics,
            file,
            ensure_ascii=False,
            indent=4,
        )


if __name__ == "__main__":
    #main()
    plot_metric("Recall@10")
