import numpy as np
import pytest

from ticketrag.store import VectorStore


def _unit(n, dim=8, seed=0):
    v = np.random.default_rng(seed).normal(size=(n, dim)).astype("float32")
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_search_returns_nearest_first():
    s, v = VectorStore(8), _unit(5)
    s.upsert([10, 11, 12, 13, 14], v)
    top = s.search(v[2:3], k=3)[0]
    assert top[0][0] == 12 and top[0][1] == pytest.approx(1.0, abs=1e-5)
    assert len(top) == 3


def test_upsert_replaces_and_remove_deletes():
    s, v = VectorStore(8), _unit(3)
    s.upsert([1, 2, 3], v)
    s.upsert([2], v[0:1])  # id 2 now points at vector 0
    assert len(s) == 3
    assert s.search(v[0:1], k=3)[0][0][1] == pytest.approx(1.0, abs=1e-5)
    s.remove([2])
    assert len(s) == 2 and 2 not in [i for i, _ in s.search(v[0:1], k=3)[0]]


def test_save_load_roundtrip(tmp_path):
    s, v = VectorStore(8, meta={"model_name": "m"}), _unit(4)
    s.upsert([1, 2, 3, 4], v)
    s.save(tmp_path)
    loaded = VectorStore.load(tmp_path)
    assert len(loaded) == 4 and loaded.meta["model_name"] == "m" and loaded.dim == 8
    assert loaded.search(v[3:4], k=1)[0][0][0] == 4


def test_upsert_rejects_wrong_shape():
    with pytest.raises(ValueError):
        VectorStore(8).upsert([1], _unit(1, dim=4))
