"""Deterministic two-dimensional embedder for offline retrieval tests."""

import numpy as np


class Embedder:
    def __init__(self, spec, **_):
        self.spec = spec
        self.batch_size = 1

    def prepare(self):
        pass

    def input_token_counts(self, texts, *, role, **_):
        return [len(text.split()) + 2 for text in texts]

    def input_configuration(self, role):
        return {"role": role}

    def embed_texts(self, texts):
        return np.tile([1.0, 0.0], (len(texts), 1))

    def embed_query(self, _):
        return np.array([1.0, 0.0])
