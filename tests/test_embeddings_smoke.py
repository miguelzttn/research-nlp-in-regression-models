import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT_DIR / "research" / "3_embeddings" / "my_embeddings.py"


def _load_embeddings_module():
    spec = importlib.util.spec_from_file_location("my_embeddings", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeVectors:
    def __init__(self, vectors):
        self._vectors = vectors
        self.vector_size = len(next(iter(vectors.values())))

    def __contains__(self, key):
        return key in self._vectors

    def __getitem__(self, key):
        return np.array(self._vectors[key], dtype=np.float32)


class TestEmbeddingsSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.emb = _load_embeddings_module()

    @staticmethod
    def _simple_tokenize_by_strategy(text, strategy, idiom="english"):
        del strategy, idiom
        return str(text).lower().split()

    def test_bow_smoke(self):
        texts = ["Machine learning rocks", "Learning embeddings rocks"]
        with patch.object(self.emb, "_tokenize_by_strategy", side_effect=self._simple_tokenize_by_strategy):
            result = self.emb.get_embeddings(strategy="document", text_inputs=texts, type="BoW")

        self.assertEqual(result.shape[0], 2)
        self.assertGreater(result.shape[1], 0)
        self.assertGreater(result.nnz, 0)

    def test_tfidf_smoke(self):
        texts = ["deep learning model", "deep model embeddings"]
        with patch.object(self.emb, "_tokenize_by_strategy", side_effect=self._simple_tokenize_by_strategy):
            result = self.emb.get_embeddings(strategy="document", text_inputs=texts, type="TF-IDF")

        self.assertEqual(result.shape[0], 2)
        self.assertGreater(result.shape[1], 0)
        self.assertGreater(result.nnz, 0)

    def test_word2vec_smoke(self):
        texts = ["deep learning", "unknown terms"]
        fake_model = _FakeVectors(
            {
                "deep": [1.0, 0.0, 0.0],
                "learning": [0.0, 1.0, 0.0],
            }
        )

        with patch.object(self.emb, "_tokenize_by_strategy", side_effect=self._simple_tokenize_by_strategy), patch.object(
            self.emb, "_load_w2v_model", return_value=fake_model
        ):
            result = self.emb.get_embeddings(strategy="document", text_inputs=texts, type="Word2Vec")

        dense = result.toarray()
        self.assertEqual(dense.shape, (2, 3))
        self.assertTrue(np.isfinite(dense).all())

    def test_glove_smoke(self):
        texts = ["vector space", "missing token"]
        fake_model = _FakeVectors(
            {
                "vector": [0.2, 0.4, 0.6, 0.8],
                "space": [0.1, 0.3, 0.5, 0.7],
            }
        )

        with patch.object(self.emb, "_tokenize_by_strategy", side_effect=self._simple_tokenize_by_strategy), patch.object(
            self.emb, "_load_glove_model", return_value=fake_model
        ):
            result = self.emb.get_embeddings(strategy="document", text_inputs=texts, type="GloVe")

        dense = result.toarray()
        self.assertEqual(dense.shape, (2, 4))
        self.assertTrue(np.isfinite(dense).all())


if __name__ == "__main__":
    unittest.main()