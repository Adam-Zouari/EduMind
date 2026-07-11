"""Quick helper to inspect clinical psychology chunk duplication."""

from __future__ import annotations

import json

from experiments.mlflow.mlflow_config import EVALUATION_DIR

with open(EVALUATION_DIR / "ground_truth.json", encoding="utf-8") as handle:
    ground_truth = json.load(handle)

clinical_chunks = [(chunk_id, chunk) for chunk_id, chunk in ground_truth.items() if "clinical_psychology" in chunk_id]

print(f"Total clinical psychology chunks: {len(clinical_chunks)}")
print("\nFirst 5 chunks:")
for index, (chunk_id, chunk) in enumerate(clinical_chunks[:5], start=1):
    print(f"\n{index}. ID: {chunk_id}")
    print(f"   Variant: {chunk.get('variant', 'N/A')}")
