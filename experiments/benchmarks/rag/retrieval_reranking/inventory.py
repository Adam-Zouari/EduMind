"""Complete frozen chunk evidence for model-free retrieval rescoring."""

from dataclasses import asdict

from edumind.common.artifacts import stable_hash
from experiments.benchmarks.common.datasets import answerable_questions
from experiments.benchmarks.rag.evaluation import Chunk, OutputIntegrityError

ARTIFACT = "chunk_inventory.json"


def chunk_rows(chunks):
    return [
        {
            "corpus_order": order,
            **{
                key: value
                for key, value in asdict(chunk).items()
                if key != "model_tokens"
            },
        }
        for order, chunk in enumerate(chunks)
    ]


def _references(manifest):
    return {
        "manifest_checksum": manifest.checksum,
        "manifest_fingerprint": manifest.fingerprint,
        "documents": [
            {"id": str(row["id"]), "record_sha256": stable_hash(row)}
            for row in manifest.samples
            if row.get("kind") == "document"
        ],
        "questions": [
            {"id": str(row["id"]), "record_sha256": stable_hash(row)}
            for row in answerable_questions(manifest)
        ],
    }


def canonical_inventory(index, manifest):
    content = {
        "schema_version": 1,
        "references": _references(manifest),
        "evaluation_tokenizer": index.tokenizer.name,
        "chunking_fingerprint": index.chunking_fingerprint,
        "chunking_contract": dict(index.chunking_contract),
        "chunks": chunk_rows(index.chunks),
    }
    return {**content, "sha256": stable_hash(content)}


def inventory_reference(inventory):
    return {
        "owner": "comparison-parent",
        "artifact": ARTIFACT,
        "sha256": inventory["sha256"],
        "chunk_count": len(inventory["chunks"]),
    }


def chunks_from_inventory(inventory, manifest, *, expected_sha256):
    """Verify saved boundaries and reference identity without rebuilding chunks."""
    content = {key: value for key, value in inventory.items() if key != "sha256"}
    if (
        type(content.get("schema_version")) is not int
        or content["schema_version"] != 1
        or inventory.get("sha256") != expected_sha256
        or stable_hash(content) != expected_sha256
        or content.get("references") != _references(manifest)
    ):
        raise OutputIntegrityError("Stale or changed canonical chunk inventory")
    rows = content.get("chunks", [])
    if not rows or [row.get("corpus_order") for row in rows] != list(range(len(rows))):
        raise OutputIntegrityError("Incomplete canonical chunk inventory/order")
    chunks = [
        Chunk(**{key: value for key, value in row.items() if key != "corpus_order"})
        for row in rows
    ]
    if len({chunk.identifier for chunk in chunks}) != len(chunks):
        raise OutputIntegrityError("Duplicate canonical chunk identity")
    return chunks
