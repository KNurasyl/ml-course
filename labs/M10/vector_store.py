"""Vector stores against the in-memory reference (extension 1.2.0, T34, lesson M10-L08X).

The reference is exact search in memory: cosine distance 1 - cos in float64, ties broken by the smaller id.
Every store must return the same top-k on the same vectors; approximate indexes are measured by the share of
reference rows they keep (recall@k).
- sqlite-vec 0.1.9: one SQLite file inside the process, no server, exact search. The offline path of the lesson.
- pgvector 0.8.7 in PostgreSQL through psql: exact scan, HNSW, IVFFlat and a filtered search. labs/M10/pg_local.py
  starts a local throwaway cluster; an unreachable server or a missing extension is a status, not a failure.
Part 1 stores the course chunks (24 documents, 72 paragraph chunks) with one encoder. The store records that encoder
and refuses queries from another one. Part 2 uses synthetic clustered vectors to show where approximate indexes lose
rows, including a metadata filter. Nothing is trained or tuned: sizes and index parameters are fixed before running.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sqlite3
import subprocess
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/M10/vector_store"
SCHEMA = "m10_l08x"
K = 5            # course chunks per query
K_SYNTHETIC = 10
SPREAD, HARD_SPREAD = 0.6, 3.0  # noise around 64 cluster centres: clustered data and almost structureless data
ENCODERS = ["USER-bge-m3", "bge-m3", "Qwen3-Embedding-0.6B", "all-MiniLM-L6-v2"]


def load(name):
    spec = importlib.util.spec_from_file_location(f"m10_{name}", Path(__file__).with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def exact_topk(vectors, queries, k, groups=None, query_groups=None):
    """Reference search: cosine distance in float64; ties go to the smaller id (= row number)."""
    v = np.asarray(vectors, dtype=np.float64)
    q = np.asarray(queries, dtype=np.float64)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    distance = 1 - q @ v.T
    if groups is not None:
        distance[np.asarray(query_groups)[:, None] != np.asarray(groups)[None, :]] = np.inf
    order = np.argsort(distance, axis=1, kind="stable")[:, :k]
    return [[(int(i), float(distance[row, i])) for i in order[row] if np.isfinite(distance[row, i])]
            for row in range(len(q))]


def compare(reference, results):
    """recall@k against the reference, mean rows returned, share of identical id order, max distance error."""
    recalls, returned, errors, same = [], [], [0.0], 0
    for ref, got in zip(reference, results, strict=True):
        ref_ids, got_ids = [i for i, _ in ref], [i for i, _ in got]
        recalls.append(len(set(ref_ids) & set(got_ids)) / max(1, len(ref_ids)))
        returned.append(len(got))
        ref_distance = dict(ref)
        errors += [abs(d - ref_distance[i]) for i, d in got if i in ref_distance]
        same += got_ids == ref_ids
    return {"recall_at_k": round(float(np.mean(recalls)), 4), "rows_returned_mean": round(float(np.mean(returned)), 2),
            "same_order_share": round(same / len(reference), 4), "max_distance_error": float(max(errors))}


def vector_literal(vector):
    """pgvector text form; 9 significant digits restore every float32 exactly."""
    return "[" + ",".join(f"{x:.9g}" for x in np.asarray(vector, dtype=np.float32)) + "]"


class EncoderMismatch(ValueError):
    pass


class SqliteVecStore:
    """One SQLite file: store_meta remembers the encoder, the vec0 table keeps vectors with a group column."""

    def __init__(self, path, encoder, dimension):
        import sqlite_vec
        self.path, self.dimension = Path(path), int(dimension)
        self.db = sqlite3.connect(self.path)
        self.db.enable_load_extension(True)
        sqlite_vec.load(self.db)
        self.db.enable_load_extension(False)
        self.db.execute("create table if not exists store_meta(key text primary key, value text not null)")
        stored = dict(self.db.execute("select key, value from store_meta"))
        wanted = {"encoder": encoder, "dimension": str(self.dimension)}
        if stored and stored != wanted:
            self.db.close()
            raise EncoderMismatch(f"{self.path.name} built with {stored['encoder']} ({stored['dimension']}), "
                                  f"queries from {encoder} ({self.dimension})")
        if not stored:
            self.db.executemany("insert into store_meta(key, value) values (?, ?)", wanted.items())
            self.db.execute(f"create virtual table items using vec0(id integer primary key, grp text, "
                            f"embedding float[{self.dimension}] distance_metric=cosine)")
            self.db.commit()

    def add(self, vectors, groups):
        rows = [(i, str(g), np.asarray(v, dtype=np.float32).tobytes()) for i, (v, g) in enumerate(zip(vectors, groups))]
        self.db.executemany("insert into items(id, grp, embedding) values (?, ?, ?)", rows)
        self.db.commit()

    def search(self, query, k, group=None):
        sql, args = "select id, distance from items where embedding match ? and k = ?", [np.asarray(query, dtype=np.float32).tobytes(), k]
        if group is not None:  # a metadata column: filtered during the scan, so k rows come back when the group has them
            sql, args = sql + " and grp = ?", args + [str(group)]
        return sorted(((int(i), float(d)) for i, d in self.db.execute(sql, args)), key=lambda row: (row[1], row[0]))

    def close(self):
        self.db.close()


class PgVector:
    """pgvector through the psql client: the SQL stays readable and no Python driver is needed."""

    def __init__(self, conninfo, psql=None):
        self.conninfo, self.psql = conninfo, psql or load("pg_local").tool("psql")

    def sql(self, script, timeout=1800):
        result = subprocess.run([self.psql, self.conninfo, "-X", "-q", "-A", "-t", "-F", "|", "-v", "ON_ERROR_STOP=1", "-f", "-"],
                                input=script, capture_output=True, text=True, encoding="utf-8", timeout=timeout, check=False)
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or f"psql exit {result.returncode}")
        return result.stdout

    def status(self):
        try:
            version = self.sql("select coalesce((select default_version from pg_available_extensions where name = 'vector'), '');",
                               timeout=30).strip()
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
            return {"available": False, "reason": (str(error).splitlines() or [type(error).__name__])[0]}
        if not version:
            return {"available": False, "reason": "extension vector is not available on this server"}
        server = self.sql("show server_version;", timeout=30).strip()
        return {"available": True, "vector": version, "server": server}

    def load(self, table, vectors, groups):
        OUT.mkdir(parents=True, exist_ok=True)
        path = OUT / f"pg-{table}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerows((i, g, vector_literal(v)) for i, (v, g) in enumerate(zip(vectors, groups)))
        self.sql(f"""create extension if not exists vector;
create schema if not exists {SCHEMA};
drop table if exists {SCHEMA}.{table};
create table {SCHEMA}.{table} (id int primary key, grp text not null, embedding vector({len(vectors[0])}) not null);
\\copy {SCHEMA}.{table} from '{path.as_posix()}' with (format csv)
analyze {SCHEMA}.{table};
""")
        path.unlink()

    def create_knn_function(self):
        # Dynamic SQL: each query is planned with its own vector and literal LIMIT, as an application query would be.
        self.sql(f"""create or replace function {SCHEMA}.knn(items regclass, queries regclass, k int, filtered bool)
returns table(qid int, ids int[], distances float8[], ms float8) language plpgsql as $fn$
declare q record; t0 timestamptz; statement text;
begin
  statement := format('select array_agg(id order by d, id), array_agg(d order by d, id) from '
                      '(select id, embedding <=> $1 as d from %s %s order by embedding <=> $1 limit %s) s',
                      items, case when filtered then 'where grp = $2' else '' end, k);
  for q in execute format('select id, grp, embedding from %s order by id', queries) loop
    t0 := clock_timestamp();
    execute statement into ids, distances using q.embedding, q.grp;
    qid := q.id;
    ms := extract(epoch from clock_timestamp() - t0) * 1000;
    return next;
  end loop;
end $fn$;""")

    def knn(self, items, queries, k, filtered=False, settings=""):
        output = self.sql(f"{settings}\nselect qid, ids, distances, ms from {SCHEMA}.knn('{SCHEMA}.{items}', "
                          f"'{SCHEMA}.{queries}', {k}, {str(filtered).lower()});")
        results, timings = [], []
        for line in output.splitlines():
            if not line.strip():
                continue
            _, ids, distances, ms = line.split("|")
            ids = [int(x) for x in ids.strip("{}").split(",") if x]
            distances = [float(x) for x in distances.strip("{}").split(",") if x]
            results.append(list(zip(ids, distances)))
            timings.append(float(ms))
        return results, timings

    def plan(self, items, vector, k, group=None, settings=""):
        where = f"where grp = '{group}' " if group is not None else ""
        output = self.sql(f"{settings}\nexplain (costs off) select id from {SCHEMA}.{items} {where}"
                          f"order by embedding <=> '{vector_literal(vector)}' limit {k};")
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        nodes = [lines[0]] + [line[2:].strip() for line in lines[1:] if line.startswith("->")]  # skip Sort Key, Filter
        return " → ".join(node.split(" on ")[0] for node in nodes)

    def timed(self, script):
        started = time.perf_counter()
        self.sql(script)
        return round(time.perf_counter() - started, 2)

    def size(self, relation):
        return int(self.sql(f"select pg_relation_size('{SCHEMA}.{relation}');").strip())


def median_ms(timings):
    return round(float(np.median(timings)), 3)


def course_vectors(encoder, queries_only=False):
    diag, emb = load("diagnostic_ru"), load("ru_embedders")
    rows = diag.chunks("paragraph")
    documents = [r["title"] + " " + r["text"] for r in rows][:1 if queries_only else None]
    queries = [{**q, "set": "ru-embed-1"} for q in emb.ru_embed_queries()]
    queries += [{"id": q["id"], "set": "diagnostic-1.1.0", "style": "diagnostic", "text": q["text"], "relevant": q["relevant"]}
                for q in diag.QUERIES if q["relevant"]]
    texts = [q["text"] for q in queries]
    if encoder == "all-MiniLM-L6-v2":
        demo = load("demo")
        started = time.perf_counter()
        vectors = demo.pretrained_vectors(documents + texts)
        doc_vectors, query_vectors = vectors[:len(documents)], vectors[len(documents):]
        info = {"encoder": f"{demo.ENCODER_ID}@{demo.ENCODER_REVISION}", "dimension": 384,
                "encode_seconds": round(time.perf_counter() - started, 1)}
    else:
        registry = {e["name"]: e for e in json.loads((ROOT / "metadata/reference-models.json").read_text(encoding="utf-8"))["embedders"]}
        entry = registry[encoder]
        doc_vectors, query_vectors, details = emb.embed(ROOT / entry["path"], documents, texts)
        info = {"encoder": f"{entry['repo_id']}@{entry['revision']}", "dimension": details["dimension"],
                "encode_seconds": round(details["load_seconds"] + details["encode_seconds"], 1)}
    return rows, queries, doc_vectors.astype(np.float32), query_vectors.astype(np.float32), info


def downloaded(encoder):
    if encoder == "all-MiniLM-L6-v2":
        return (ROOT / "models/all-MiniLM-L6-v2/model.safetensors").exists()
    registry = {e["name"]: e for e in json.loads((ROOT / "metadata/reference-models.json").read_text(encoding="utf-8"))["embedders"]}
    path = ROOT / registry[encoder]["path"]
    return any(path.glob("*.safetensors")) or (path / "pytorch_model.bin").exists()


def top1_hits(results, rows, queries, style):
    pairs = [(res, q) for res, q in zip(results, queries) if q["style"] == style]
    return {"hits": sum(bool(res) and rows[res[0][0]]["document_id"] in q["relevant"] for res, q in pairs), "n": len(pairs)}


def course_part(encoder, pg, mismatch=True):
    rows, queries, doc_vectors, query_vectors, info = course_vectors(encoder)
    groups = [r["document_id"] for r in rows]
    reference = exact_topk(doc_vectors, query_vectors, K)
    part = {"encoder": info, "chunks": len(rows), "queries": len(queries), "k": K,
            "float32_bytes": int(doc_vectors.size * 4), "stores": {}}
    part["reference_top1"] = {s: top1_hits(reference, rows, queries, s) for s in ["paraphrase", "term", "diagnostic"]}
    try:
        import sqlite_vec  # noqa: F401
    except ImportError:
        part["stores"]["sqlite-vec"] = {"status": "not_installed"}
    else:
        OUT.mkdir(parents=True, exist_ok=True)
        path = OUT / "course.sqlite"
        path.unlink(missing_ok=True)
        store = SqliteVecStore(path, info["encoder"], info["dimension"])
        started = time.perf_counter()
        store.add(doc_vectors, groups)
        insert_seconds = time.perf_counter() - started
        store.close()
        store = SqliteVecStore(path, info["encoder"], info["dimension"])  # reopened: vectors come from the file
        timings, results = [], []
        for q in query_vectors:
            started = time.perf_counter()
            results.append(store.search(q, K))
            timings.append((time.perf_counter() - started) * 1000)
        try:
            store.search(np.ones(info["dimension"] // 2, dtype=np.float32), K)
            wrong_dimension = "accepted"
        except sqlite3.Error as error:
            wrong_dimension = str(error)
        store.close()
        part["stores"]["sqlite-vec"] = {"status": "measured", "after_reopen": compare(reference, results),
                                        "median_ms": median_ms(timings), "insert_seconds": round(insert_seconds, 3),
                                        "file_bytes": path.stat().st_size, "wrong_dimension_query": wrong_dimension}
        part["mismatch"] = []
        for name in [e for e in ENCODERS if e != encoder and mismatch and downloaded(e)]:
            other = course_vectors(name, queries_only=True)
            if other[4]["dimension"] != info["dimension"]:
                continue  # a different dimension already fails inside the store: see wrong_dimension_query
            try:
                SqliteVecStore(path, other[4]["encoder"], other[4]["dimension"])
                guard = "accepted"
            except EncoderMismatch as error:
                guard = str(error)
            # Same dimension, so nothing fails by itself: score what a store without the guard would return.
            crossed = exact_topk(doc_vectors, other[3], K)
            part["mismatch"].append({"query_encoder": other[4]["encoder"], "dimension": other[4]["dimension"], "guard": guard,
                                     "top1": {s: top1_hits(crossed, rows, queries, s) for s in ["paraphrase", "term", "diagnostic"]}})
    if pg is not None:
        pg.load("course", doc_vectors, groups)
        pg.load("course_q", query_vectors, ["-"] * len(query_vectors))
        pg.create_knn_function()
        exact, exact_ms = pg.knn("course", "course_q", K)
        pg.sql(f"create index course_hnsw on {SCHEMA}.course using hnsw (embedding vector_cosine_ops);")
        with_index, index_ms = pg.knn("course", "course_q", K)  # the planner decides whether the index is worth it
        part["stores"]["pgvector"] = {"status": "measured",
                                      "exact": {**compare(reference, exact), "median_ms": median_ms(exact_ms),
                                                "plan": "no index"},
                                      "with_hnsw_index": {**compare(reference, with_index), "median_ms": median_ms(index_ms),
                                                          "plan": pg.plan("course", query_vectors[0], K)}}
    return part


def synthetic(n=20000, dimension=128, clusters=64, queries=200, groups=100, spread=SPREAD, seed=7):
    """Clustered unit vectors with a group column of `groups` equal-probability values (about 1% of rows each)."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(clusters, dimension))
    data = centers[rng.integers(clusters, size=n)] + spread * rng.normal(size=(n, dimension))
    probes = centers[rng.integers(clusters, size=queries)] + spread * rng.normal(size=(queries, dimension))
    data /= np.linalg.norm(data, axis=1, keepdims=True)
    probes /= np.linalg.norm(probes, axis=1, keepdims=True)
    return (data.astype(np.float32), [f"g{g}" for g in rng.integers(groups, size=n)],
            probes.astype(np.float32), [f"g{g}" for g in rng.integers(groups, size=queries)])


def synthetic_part(pg, n=20000):
    vectors, groups, queries, query_groups = synthetic(n)
    k = K_SYNTHETIC
    started = time.perf_counter()
    reference = exact_topk(vectors, queries, k)
    numpy_ms = (time.perf_counter() - started) * 1000 / len(queries)
    filtered_reference = exact_topk(vectors, queries, k, groups, query_groups)
    part = {"n": n, "dimension": int(vectors.shape[1]), "queries": len(queries), "k": k, "groups": 100, "spread": SPREAD,
            "float32_bytes": int(vectors.size * 4), "numpy_exact_ms_per_query": round(numpy_ms, 3), "stores": {}}
    try:
        import sqlite_vec  # noqa: F401
    except ImportError:
        part["stores"]["sqlite-vec"] = {"status": "not_installed"}
    else:
        path = OUT / "synthetic.sqlite"
        OUT.mkdir(parents=True, exist_ok=True)
        path.unlink(missing_ok=True)
        store = SqliteVecStore(path, f"synthetic-seed7-{n}", vectors.shape[1])
        store.add(vectors, groups)
        measured = {}
        for name, group_of in [("exact", lambda i: None), ("exact_filtered", lambda i: query_groups[i])]:
            timings, results = [], []
            for i, q in enumerate(queries):
                started = time.perf_counter()
                results.append(store.search(q, k, group_of(i)))
                timings.append((time.perf_counter() - started) * 1000)
            measured[name] = {**compare(filtered_reference if group_of(0) else reference, results), "median_ms": median_ms(timings)}
        store.close()
        part["stores"]["sqlite-vec"] = {"status": "measured", **measured, "file_bytes": path.stat().st_size}
    if pg is not None:
        pg.load("items", vectors, groups)
        pg.load("items_q", queries, query_groups)
        pg.create_knn_function()
        q0, g0 = queries[0], query_groups[0]
        out = {"table_bytes": int(pg.sql(f"select pg_table_size('{SCHEMA}.items');").strip())}
        results, timings = pg.knn("items", "items_q", k)
        out["exact"] = {**compare(reference, results), "median_ms": median_ms(timings), "plan": pg.plan("items", q0, k)}
        out["hnsw_build_seconds"] = pg.timed(f"create index items_hnsw on {SCHEMA}.items using hnsw "
                                             f"(embedding vector_cosine_ops) with (m = 16, ef_construction = 64);")
        out["hnsw_bytes"] = pg.size("items_hnsw")
        out["hnsw"] = {}
        for ef in [10, 40, 100, 200]:
            settings = f"set hnsw.ef_search = {ef};"
            results, timings = pg.knn("items", "items_q", k, settings=settings)
            out["hnsw"][f"ef_search={ef}"] = {**compare(reference, results), "median_ms": median_ms(timings),
                                              "plan": pg.plan("items", q0, k, settings=settings)}
        # Filter on a group of about 1% of rows, with the HNSW index present.
        variants = {"hnsw_default_settings": "", "hnsw_iterative_scan": "set hnsw.iterative_scan = relaxed_order;"}
        out["filtered"] = {}
        for name, settings in variants.items():
            results, timings = pg.knn("items", "items_q", k, filtered=True, settings=settings)
            out["filtered"][name] = {**compare(filtered_reference, results), "median_ms": median_ms(timings),
                                     "plan": pg.plan("items", q0, k, g0, settings), "settings": settings}
        pg.sql(f"create index items_grp on {SCHEMA}.items (grp); analyze {SCHEMA}.items;")
        results, timings = pg.knn("items", "items_q", k, filtered=True)
        out["filtered"]["btree_on_group"] = {**compare(filtered_reference, results), "median_ms": median_ms(timings),
                                             "plan": pg.plan("items", q0, k, g0), "settings": "index on grp"}
        pg.sql(f"drop index {SCHEMA}.items_grp; drop index {SCHEMA}.items_hnsw;")
        lists = max(1, n // 1000)  # pgvector README starting point: rows / 1000 lists, sqrt(lists) probes
        out["ivfflat_build_seconds"] = pg.timed(f"create index items_ivf on {SCHEMA}.items using ivfflat "
                                                f"(embedding vector_cosine_ops) with (lists = {lists});")
        out["ivfflat_bytes"] = pg.size("items_ivf")
        out["ivfflat"] = {"lists": lists}
        for probes in sorted({1, round(lists ** 0.5), lists}):
            results, timings = pg.knn("items", "items_q", k, settings=f"set ivfflat.probes = {probes};")
            out["ivfflat"][f"probes={probes}"] = {**compare(reference, results), "median_ms": median_ms(timings)}
        pg.sql(f"drop index {SCHEMA}.items_ivf;")
        # The same index on almost structureless data: neighbours are barely closer than the rest.
        hard, hard_groups, hard_queries, hard_query_groups = synthetic(n, spread=HARD_SPREAD)
        hard_reference = exact_topk(hard, hard_queries, k)
        pg.load("hard", hard, hard_groups)
        pg.load("hard_q", hard_queries, hard_query_groups)
        pg.sql(f"create index hard_hnsw on {SCHEMA}.hard using hnsw (embedding vector_cosine_ops) with (m = 16, ef_construction = 64);")
        out["hnsw_hard_data"] = {"spread": HARD_SPREAD}
        for ef in [10, 40, 100]:
            results, timings = pg.knn("hard", "hard_q", k, settings=f"set hnsw.ef_search = {ef};")
            out["hnsw_hard_data"][f"ef_search={ef}"] = {**compare(hard_reference, results), "median_ms": median_ms(timings)}
        pg.sql(f"drop table {SCHEMA}.hard; drop table {SCHEMA}.hard_q;")
        part["stores"]["pgvector"] = {"status": "measured", **out}
    return part


def run(encoder="auto", use_pg=True, conninfo=None, n=20000, mismatch=True, out_dir=None):
    if encoder == "auto":
        encoder = next(e for e in ENCODERS if downloaded(e))
    pg_status, pg = {"available": False, "reason": "disabled with --no-pg"}, None
    if use_pg:
        pg = PgVector(conninfo or load("pg_local").CONNINFO)
        pg_status = pg.status()
        if not pg_status["available"]:
            pg = None
    try:
        import sqlite_vec
        sqlite_status = {"available": True, "version": sqlite_vec.__version__ if hasattr(sqlite_vec, "__version__") else None}
    except ImportError:
        sqlite_status = {"available": False, "reason": "pip install -r requirements-optional.in"}
    report = {"task": "T34", "reference": "numpy exact cosine, float64, ties by id",
              "fixed_before_running": "corpus and queries of T32, k=5; synthetic n, dimension 128, 64 clusters, seed 7, k=10; "
                                      "HNSW m=16 ef_construction=64; IVFFlat lists=n/1000. Spread 0.6 was the first value; "
                                      "spread 3.0 was added after a probe showed recall falls on structureless data",
              "status": {"sqlite-vec": sqlite_status, "pgvector": pg_status},
              "course": course_part(encoder, pg, mismatch), "synthetic": synthetic_part(pg, n),
              "limits": "Single machine, CPU, small corpus; latency includes Python or PL/pgSQL call overhead. "
                        "Shows behaviour of stores and indexes, not a benchmark of the products."}
    out = Path(out_dir) if out_dir else OUT
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def table_text(report):
    lines = ["| Часть | Вариант | recall@k | строк в ответе | мс на запрос |", "| --- | --- | --- | --- | --- |"]

    def add(part, name, row):
        lines.append(f"| {part} | {name} | {row['recall_at_k']:.3f} | {row['rows_returned_mean']:.1f} | {row.get('median_ms', '-')} |")

    course, synth = report["course"]["stores"], report["synthetic"]["stores"]
    if course.get("sqlite-vec", {}).get("status") == "measured":
        add("курс", "sqlite-vec после переоткрытия", {**course["sqlite-vec"]["after_reopen"], "median_ms": course["sqlite-vec"]["median_ms"]})
    if course.get("pgvector"):
        add("курс", "pgvector без индекса", course["pgvector"]["exact"])
        add("курс", "pgvector с индексом HNSW", course["pgvector"]["with_hnsw_index"])
    if synth.get("sqlite-vec", {}).get("status") == "measured":
        add("синтетика", "sqlite-vec точный", synth["sqlite-vec"]["exact"])
        add("синтетика", "sqlite-vec с фильтром", synth["sqlite-vec"]["exact_filtered"])
    if synth.get("pgvector"):
        p = synth["pgvector"]
        add("синтетика", "pgvector точный", p["exact"])
        for name, row in {**p["hnsw"], **{k: v for k, v in p["ivfflat"].items() if k != "lists"}}.items():
            add("синтетика", name, row)
        for name, row in p["filtered"].items():
            add("синтетика, фильтр 1%", name, row)
        for name, row in p["hnsw_hard_data"].items():
            if name != "spread":
                add("почти без кластеров", f"HNSW {name}", row)
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", default="auto", choices=["auto", *ENCODERS])
    parser.add_argument("--no-pg", action="store_true", help="skip PostgreSQL even if it is reachable")
    parser.add_argument("--pg-conninfo", help="libpq connection string without a password; default: the pg_local.py cluster")
    parser.add_argument("--synthetic-n", type=int, default=20000)
    parser.add_argument("--no-mismatch", action="store_true", help="skip queries from other downloaded encoders of the same dimension")
    args = parser.parse_args()
    report = run(args.encoder, not args.no_pg, args.pg_conninfo, args.synthetic_n, not args.no_mismatch)
    print(table_text(report))
    print(json.dumps(report["status"], ensure_ascii=False))
