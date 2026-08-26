from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from llm import load_llm


RETRIEVAL_DOCS_FILE = Path(
    "data/retrieval_docs/retrieval_docs.jsonl"
)

FAISS_INDEX_FILE = Path(
    "data/dense_index/qwen3_embedding_4b.index"
)

DOC_MAPPING_FILE = Path(
    "data/dense_index/doc_mapping.json"
)

MODEL_NAME = "Qwen/Qwen3-Embedding-4B"

BATCH_SIZE = 32

QUERY_INSTRUCTION = (
    "Given a query about historical battles, "
    "retrieve relevant battle descriptions."
)


def load_retrieval_docs(
    file_path: Path = RETRIEVAL_DOCS_FILE,
) -> list[dict[str, Any]]:
    """
    Load retrieval documents from JSONL.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Retrieval docs file not found: "
            f"{file_path.resolve()}"
        )

    docs: list[dict[str, Any]] = []

    with file_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)

            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at line {line_number}: "
                    f"{error}"
                ) from error

            text = item.get("text")

            if not isinstance(text, str) or not text.strip():
                continue

            docs.append(
                {
                    "battle_id": item.get("battle_id"),
                    "name": item.get("name"),
                    "text": text,
                    "metadata": item.get(
                        "metadata",
                        {},
                    ),
                }
            )

    return docs


def load_embedding_model():
    """
    Load the API client configured for Qwen3-Embedding-4B.
    """

    print(f"Loading embedding model: {MODEL_NAME}")

    return load_llm(
        model_name=MODEL_NAME
    )


def normalize_embeddings(
    embeddings: np.ndarray,
) -> np.ndarray:
    """
    L2-normalize embeddings so that inner product
    corresponds to cosine similarity.
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

    return normalize_embeddings(embeddings)


def build_dense_index(
    docs: list[dict[str, Any]],
    model,
) -> faiss.IndexFlatIP:
    """
    Embed all retrieval documents and build an exact FAISS index.

    Normalized vectors + inner product = cosine similarity.
    """

    texts = [
        doc["text"]
        for doc in docs
    ]

    print(
        f"Embedding {len(texts)} retrieval documents..."
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
    Save the FAISS index and its document mapping.
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
    Load the saved FAISS index and document mapping.
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
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """
    Retrieve the top-k battles using dense retrieval.
    """

    if not query.strip():
        raise ValueError(
            "Query must not be empty."
        )

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than 0."
        )

    top_k = min(
        top_k,
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
        top_k,
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
                "name": doc["name"],
                "score": float(
                    scores[0][rank - 1]
                ),
                "text": doc["text"],
                "metadata": doc["metadata"],
            }
        )

    return results


def create_index() -> None:
    """
    Build and save the dense retrieval index.
    """

    docs = load_retrieval_docs()

    print(
        f"Loaded retrieval documents: {len(docs)}"
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


def test_search() -> None:
    """
    Load the existing index and run a test query.
    """

    model = load_embedding_model()

    index, docs = load_dense_index()

    print(
        f"Loaded vectors: {index.ntotal}"
    )

    query = "military defeats that caused the collapse of a state"

    results = dense_search(
        query=query,
        model=model,
        index=index,
        docs=docs,
        top_k=10,
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
    #create_index()

    # After the index is created:
    #
    test_search()