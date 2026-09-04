from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from llm import load_llm


JSON_KB_DIR = Path(
    "data/json_kb_v1"
)

FAISS_INDEX_FILE = Path(
    "data/dense_index/qwen3_embedding_4b_direct.index"
)

DOC_MAPPING_FILE = Path(
    "data/dense_index/doc_mapping_direct.json"
)

MODEL_NAME = "Qwen/Qwen3-Embedding-4B"

BATCH_SIZE = 8

QUERY_INSTRUCTION = (
    "Given a query about historical battles, "
    "retrieve relevant battle records."
)


def json_to_text(
    battle: dict[str, Any],
) -> str:
    """
    Serialize a full battle JSON object directly as text for embedding.

    No intermediate retrieval-document projection is created.
    Compact JSON reduces unnecessary whitespace while preserving
    all keys and values.
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

    docs: list[dict[str, Any]] = []

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

        docs.append(
            {
                "battle_id": battle_id,
                "name": name,
                "json": battle,
                "text": json_to_text(battle),
            }
        )

    return docs


def load_embedding_model():
    """
    Load the API client configured for Qwen3-Embedding-4B.
    """

    print(
        f"Loading embedding model: {MODEL_NAME}"
    )

    return load_llm(
        model_name=MODEL_NAME
    )


def normalize_embeddings(
    embeddings: np.ndarray,
) -> np.ndarray:
    """
    L2-normalize embeddings so inner product corresponds
    to cosine similarity.
    """

    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True,
    )

    norms[norms == 0] = 1.0

    return embeddings / norms


def embed_texts(
    model,
    texts: list[str],
) -> np.ndarray:
    """
    Generate embeddings through the ScaDS OpenAI-compatible API.
    """

    response = model.client.embeddings.create(
        model=model.model_name,
        input=texts,
    )

    embeddings = np.asarray(
        [
            item.embedding
            for item in response.data
        ],
        dtype="float32",
    )

    return normalize_embeddings(
        embeddings
    )


def build_dense_index(
    docs: list[dict[str, Any]],
    model,
) -> faiss.IndexFlatIP:
    """
    Embed complete battle JSON objects and build an exact FAISS index.
    """

    texts = [
        doc["text"]
        for doc in docs
    ]

    print(
        f"Embedding {len(texts)} JSON battle instances..."
    )

    embedding_batches: list[np.ndarray] = []

    total_batches = (
        len(texts) + BATCH_SIZE - 1
    ) // BATCH_SIZE

    for start_index in range(
        0,
        len(texts),
        BATCH_SIZE,
    ):
        end_index = min(
            start_index + BATCH_SIZE,
            len(texts),
        )

        batch_number = (
            start_index // BATCH_SIZE
        ) + 1

        print(
            f"Embedding batch "
            f"{batch_number}/{total_batches} "
            f"({start_index + 1}-{end_index})"
        )

        batch_texts = texts[
            start_index:end_index
        ]

        batch_embeddings = embed_texts(
            model=model,
            texts=batch_texts,
        )

        embedding_batches.append(
            batch_embeddings
        )

    if not embedding_batches:
        raise ValueError(
            "No battle JSON files were loaded."
        )

    doc_embeddings = np.vstack(
        embedding_batches
    ).astype("float32")

    dimension = doc_embeddings.shape[1]

    print(
        f"Embedding dimension: {dimension}"
    )

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        doc_embeddings
    )

    print(
        f"Vectors added to FAISS: {index.ntotal}"
    )

    return index


def save_dense_index(
    index: faiss.Index,
    docs: list[dict[str, Any]],
) -> None:
    """
    Save the direct-JSON FAISS index and document mapping.
    """

    FAISS_INDEX_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    faiss.write_index(
        index,
        str(FAISS_INDEX_FILE),
    )

    with DOC_MAPPING_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            docs,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"FAISS index saved to: "
        f"{FAISS_INDEX_FILE.resolve()}"
    )

    print(
        f"Document mapping saved to: "
        f"{DOC_MAPPING_FILE.resolve()}"
    )


def load_dense_index(
) -> tuple[
    faiss.Index,
    list[dict[str, Any]],
]:
    """
    Load the saved direct-JSON FAISS index and mapping.
    """

    if not FAISS_INDEX_FILE.exists():
        raise FileNotFoundError(
            f"FAISS index not found: "
            f"{FAISS_INDEX_FILE.resolve()}"
        )

    if not DOC_MAPPING_FILE.exists():
        raise FileNotFoundError(
            f"Document mapping not found: "
            f"{DOC_MAPPING_FILE.resolve()}"
        )

    index = faiss.read_index(
        str(FAISS_INDEX_FILE)
    )

    with DOC_MAPPING_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        docs = json.load(file)

    if index.ntotal != len(docs):
        raise ValueError(
            f"Index/mapping mismatch: "
            f"{index.ntotal} vectors but "
            f"{len(docs)} mapped documents."
        )

    return index, docs


def build_query_text(
    query: str,
) -> str:
    """
    Add a fixed retrieval instruction for Qwen3 embeddings.
    """

    return (
        f"Instruct: {QUERY_INSTRUCTION}\n"
        f"Query: {query}"
    )


def dense_search(
    query: str,
    model,
    index: faiss.Index,
    docs: list[dict[str, Any]],
    candidate_k: int,
) -> list[dict[str, Any]]:
    """
    Retrieve candidate-k battles from the directly embedded JSON KB.
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
        len(docs),
    )

    query_text = build_query_text(
        query
    )

    query_embedding = embed_texts(
        model=model,
        texts=[query_text],
    )

    scores, indices = index.search(
        query_embedding,
        candidate_k,
    )

    results: list[dict[str, Any]] = []

    for rank, index_position in enumerate(
        indices[0],
        start=1,
    ):
        if index_position < 0:
            continue

        doc = docs[index_position]

        results.append(
            {
                "rank": rank,
                "battle_id": doc["battle_id"],
                "name": doc.get("name"),
                "score": float(
                    scores[0][rank - 1]
                ),
                "json": doc["json"],
            }
        )

    return results


def create_index(
    limit: int | None = None,
) -> None:
    """
    Build and save the dense index directly from battle JSON files.

    Set limit=10 for a quick test.
    Leave limit=None for the full knowledge base.
    """

    docs = load_json_battles()

    if limit is not None:
        docs = docs[:limit]

    print(
        f"Loaded JSON battle instances: {len(docs)}"
    )

    model = load_embedding_model()

    index = build_dense_index(
        docs=docs,
        model=model,
    )

    save_dense_index(
        index=index,
        docs=docs,
    )


def test_search(
    candidate_k: int = 10,
) -> None:
    """
    Load the direct-JSON index and run a test query.
    """

    model = load_embedding_model()

    index, docs = load_dense_index()

    print(
        f"Loaded vectors: {index.ntotal}"
    )

    query = "Find battles involving a large coalition of several states where allied forces combined against a common enemy."

    results = dense_search(
        query=query,
        model=model,
        index=index,
        docs=docs,
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
            f"   Dense score: "
            f"{result['score']:.4f}"
        )

        print()


if __name__ == "__main__":
    # Quick test:
    #create_index(limit=10)

    # Full direct-JSON index:
    #create_index()

    # After the index is created:
    test_search(candidate_k=15)
