"""Vector store operations for the RAG pipeline."""

from __future__ import annotations

import logging
import os
import pickle
import uuid
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
from chromadb.config import Settings
from rank_bm25 import BM25Okapi

from edumind.common.config import load_yaml_config
from edumind.common.paths import PROJECT_ROOT

logger = logging.getLogger(__name__)


class VectorStore:
    """Manage ChromaDB persistence and a lightweight BM25 index."""

    def __init__(self, config_path: str | None = None):
        self.config = load_yaml_config(config_path)

        vectordb_config = self.config["vectordb"]
        self.collection_name = vectordb_config["collection_name"]
        self.distance_metric = vectordb_config["distance_metric"]
        persist_directory = Path(vectordb_config["persist_directory"])
        if not persist_directory.is_absolute():
            persist_directory = (PROJECT_ROOT / persist_directory).resolve()
        persist_directory.mkdir(parents=True, exist_ok=True)
        self.persist_directory = persist_directory

        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": self.distance_metric},
        )

        self.bm25_path = self.persist_directory / "bm25_index.pkl"
        self.bm25: BM25Okapi | None = None
        self.bm25_corpus: list[list[str]] = []
        self.doc_map: list[str] = []
        self._load_bm25()

    def _load_bm25(self) -> None:
        if not self.bm25_path.exists():
            return
        try:
            with self.bm25_path.open("rb") as handle:
                data = pickle.load(handle)
            self.bm25 = data["index"]
            self.bm25_corpus = data["corpus"]
            self.doc_map = data["doc_map"]
        except Exception as exc:
            logger.error(f"Failed to load BM25 index: {exc}")

    def _save_bm25(self) -> None:
        if self.bm25 is None:
            return
        with self.bm25_path.open("wb") as handle:
            pickle.dump(
                {
                    "index": self.bm25,
                    "corpus": self.bm25_corpus,
                    "doc_map": self.doc_map,
                },
                handle,
            )

    def add_documents(self, chunks: list[dict[str, Any]]) -> None:
        if not chunks:
            return

        batch_size = 5000
        for index in range(0, len(chunks), batch_size):
            self._add_batch(chunks[index : index + batch_size])

    def _add_batch(self, chunks: list[dict[str, Any]]) -> None:
        ids: list[str] = []
        embeddings: list[list[float]] = []
        documents: list[str] = []
        metadatas: list[dict[str, str]] = []
        new_corpus: list[list[str]] = []

        for chunk in chunks:
            chunk_id = str(uuid.uuid4())
            ids.append(chunk_id)
            self.doc_map.append(chunk_id)

            embedding = chunk.get("embedding", [])
            embeddings.append(embedding if isinstance(embedding, list) else embedding.tolist())

            text = chunk.get("text", "")
            documents.append(text)
            new_corpus.append(text.lower().split())

            metadata = {k: v for k, v in chunk.items() if k not in ["text", "embedding"]}
            metadatas.append({k: str(v) for k, v in metadata.items()})

        self.collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        self.bm25_corpus.extend(new_corpus)
        self.bm25 = BM25Okapi(self.bm25_corpus)
        self._save_bm25()

    def query_hybrid(self, query_text: str, query_embedding: list[float], top_k: int = 5, alpha: float = 0.3) -> list[dict[str, Any]]:
        dense_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k * 2,
            include=["documents", "metadatas", "distances"],
        )

        dense_hits: dict[str, dict[str, Any]] = {}
        if dense_results["ids"] and dense_results["ids"][0]:
            ids = dense_results["ids"][0]
            distances = dense_results["distances"][0]
            docs = dense_results["documents"][0]
            metas = dense_results["metadatas"][0]
            for index, doc_id in enumerate(ids):
                dense_hits[doc_id] = {
                    "score": 1 - distances[index],
                    "doc": docs[index],
                    "meta": metas[index],
                }

        bm25_hits: dict[str, float] = {}
        if self.bm25 is not None:
            doc_scores = self.bm25.get_scores(query_text.lower().split())
            top_indices = np.argsort(doc_scores)[::-1][: top_k * 2]
            max_score = np.max(doc_scores) if len(doc_scores) > 0 else 1
            normalized_scores = (doc_scores / max_score) if max_score > 0 else doc_scores
            for idx in top_indices:
                if idx < len(self.doc_map):
                    bm25_hits[self.doc_map[idx]] = normalized_scores[idx]

        all_ids = set(dense_hits) | set(bm25_hits)
        combined_results: list[dict[str, Any]] = []
        for doc_id in all_ids:
            dense_score = dense_hits.get(doc_id, {}).get("score", 0.0)
            bm25_score = bm25_hits.get(doc_id, 0.0)
            final_score = ((1 - alpha) * dense_score) + (alpha * bm25_score)

            if doc_id in dense_hits:
                combined_results.append(
                    {
                        "id": doc_id,
                        "document": dense_hits[doc_id]["doc"],
                        "metadata": dense_hits[doc_id]["meta"],
                        "score": final_score,
                    }
                )
            else:
                try:
                    fetched = self.collection.get(ids=[doc_id])
                    if fetched["documents"]:
                        combined_results.append(
                            {
                                "id": doc_id,
                                "document": fetched["documents"][0],
                                "metadata": fetched["metadatas"][0],
                                "score": final_score,
                            }
                        )
                except Exception as exc:
                    logger.warning(f"Could not fetch doc {doc_id}: {exc}")

        combined_results.sort(key=lambda item: item["score"], reverse=True)
        return combined_results[:top_k]

    def query_by_text(self, query_text: str, embedder, top_k: int = 5, filter_metadata: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        _ = filter_metadata
        query_embedding = embedder.embed_text(query_text)
        hybrid_results = self.query_hybrid(query_text, query_embedding.tolist(), top_k=top_k)
        return [
            {
                "id": result["id"],
                "document": result["document"],
                "metadata": result["metadata"],
                "distance": 1 - result["score"],
            }
            for result in hybrid_results
        ]

    def get_collection_count(self) -> int:
        return self.collection.count()

    def delete_collection(self) -> None:
        self.client.delete_collection(name=self.collection_name)
        if self.bm25_path.exists():
            os.remove(self.bm25_path)
        self.bm25 = None
        self.bm25_corpus = []
        self.doc_map = []

    def reset_collection(self) -> None:
        self.delete_collection()
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": self.distance_metric},
        )
