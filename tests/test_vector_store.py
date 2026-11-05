"""Vector store lab of labs/M10/vector_store.py (T34): the in-memory reference and the stores that must match it."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("m10_vector_store", SOURCE / "labs/M10/vector_store.py")
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)


def small_set(n=300, queries=20):
    return lab.synthetic(n=n, dimension=16, clusters=8, queries=queries, groups=5, seed=3)


def test_reference_orders_by_cosine_distance_and_breaks_ties_by_id():
    vectors = np.array([[1, 0], [0, 1], [2, 0], [1, 1]], dtype=np.float32)  # rows 0 and 2 point the same way
    result = lab.exact_topk(vectors, np.array([[1, 0]], dtype=np.float32), k=3)[0]
    assert [i for i, _ in result] == [0, 2, 3]
    assert result[0][1] == pytest.approx(0) and result[2][1] == pytest.approx(1 - 2 ** -0.5)
    filtered = lab.exact_topk(vectors, np.array([[1, 0]]), k=3, groups=["a", "b", "b", "a"], query_groups=["b"])[0]
    assert [i for i, _ in filtered] == [2, 1]  # only two rows in the group: fewer than k is the honest answer


def test_compare_counts_recall_rows_and_order():
    reference = [[(1, 0.1), (2, 0.2)], [(3, 0.1), (4, 0.3)]]
    assert lab.compare(reference, reference) == {"recall_at_k": 1.0, "rows_returned_mean": 2.0,
                                                 "same_order_share": 1.0, "max_distance_error": 0.0}
    result = lab.compare(reference, [[(2, 0.2), (1, 0.1)], [(3, 0.1)]])
    assert result["recall_at_k"] == 0.75 and result["rows_returned_mean"] == 1.5 and result["same_order_share"] == 0.0


def test_vector_literal_restores_float32_exactly():
    vector = np.random.default_rng(0).normal(size=64).astype(np.float32)
    literal = lab.vector_literal(vector)
    assert literal.startswith("[") and literal.endswith("]")
    assert np.array_equal(np.array(literal[1:-1].split(","), dtype=np.float32), vector)


def test_sqlite_vec_matches_reference_after_reopen(tmp_path):
    pytest.importorskip("sqlite_vec")
    vectors, groups, queries, query_groups = small_set()
    path = tmp_path / "store.sqlite"
    store = lab.SqliteVecStore(path, "test-encoder@1", 16)
    store.add(vectors, groups)
    store.close()
    store = lab.SqliteVecStore(path, "test-encoder@1", 16)  # vectors come back from the file
    results = [store.search(q, 7) for q in queries]
    assert lab.compare(lab.exact_topk(vectors, queries, 7), results)["recall_at_k"] == 1.0
    filtered = [store.search(q, 7, g) for q, g in zip(queries, query_groups)]
    store.close()
    assert all(groups[i] == g for rows, g in zip(filtered, query_groups) for i, _ in rows)
    report = lab.compare(lab.exact_topk(vectors, queries, 7, groups, query_groups), filtered)
    assert report["recall_at_k"] == 1.0 and report["rows_returned_mean"] == 7.0 and report["max_distance_error"] < 1e-5


def test_store_refuses_another_encoder_and_another_dimension(tmp_path):
    pytest.importorskip("sqlite_vec")
    import sqlite3
    path = tmp_path / "store.sqlite"
    store = lab.SqliteVecStore(path, "model-a@1", 8)
    store.add(np.eye(8, dtype=np.float32), ["x"] * 8)
    with pytest.raises(sqlite3.Error, match="Dimension mismatch"):
        store.search(np.ones(4, dtype=np.float32), 3)
    store.close()
    with pytest.raises(lab.EncoderMismatch, match="model-a@1"):
        lab.SqliteVecStore(path, "model-b@1", 8)  # same dimension: only the stored encoder name catches it


def test_unreachable_postgres_is_a_status_not_a_failure():
    status = lab.PgVector("host=127.0.0.1 port=1 user=course dbname=postgres connect_timeout=2").status()
    assert status["available"] is False and status["reason"]


def test_synthetic_data_is_reproducible_and_unit_length():
    first, second = small_set(), small_set()
    assert np.array_equal(first[0], second[0]) and first[1] == second[1]
    assert np.allclose(np.linalg.norm(first[0], axis=1), 1, atol=1e-6)
    assert len(set(first[1])) == 5


def test_pgvector_matches_reference_when_a_server_is_running():
    pg = lab.PgVector(lab.load("pg_local").CONNINFO)
    if not pg.status()["available"]:
        pytest.skip("no PostgreSQL with pgvector on the pg_local.py port")
    vectors, groups, queries, query_groups = small_set()
    pg.load("pytest_items", vectors, groups)
    pg.load("pytest_q", queries, query_groups)
    pg.create_knn_function()
    try:
        results, timings = pg.knn("pytest_items", "pytest_q", 7)
        assert lab.compare(lab.exact_topk(vectors, queries, 7), results)["recall_at_k"] == 1.0 and len(timings) == 20
        filtered, _ = pg.knn("pytest_items", "pytest_q", 7, filtered=True)
        assert lab.compare(lab.exact_topk(vectors, queries, 7, groups, query_groups), filtered)["recall_at_k"] == 1.0
    finally:
        pg.sql(f"drop table if exists {lab.SCHEMA}.pytest_items; drop table if exists {lab.SCHEMA}.pytest_q;")
