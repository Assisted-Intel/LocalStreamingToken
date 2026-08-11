#!/usr/bin/env python3
"""Tests for the RAG vector stores (app/vectorstore/) and the storage-independent
write/retrieve logic in app/rag.py.

Every guarantee is asserted against BOTH backends via parametrisation, because they are
independent implementations that must behave identically from rag.py's point of view.
The Lance cases skip cleanly when lancedb isn't installed, so an install that stays on
the encrypted DuckDB store can still run the suite.

Embedders here are fake and deterministic (sha256-seeded, so identical text always maps
to an identical vector), which means these tests need neither Ollama nor a network —
matching the "no external services" posture of test_crypto.py.
"""

import hashlib
import random
import threading

import pytest

from app import core, crypto, rag

DIM = 32
MODEL = "fake-embed"
BACKENDS = ["duckdb", "lance"]


# ------------------------------ fixtures / helpers ------------------------------
@pytest.fixture(autouse=True)
def _isolated_store(tmp_path):
    """Point both stores at a throwaway directory and unlock, so the DuckDB store is
    genuinely encrypted (as in production) and no state leaks between tests."""
    prev = (core.RAG_DB_FILE, core.RAG_LANCE_DIR)
    kf = tmp_path / "app_key.enc"
    crypto.create_keyfile(kf, password="admin")
    crypto.set_active_key(crypto.unlock(kf, "admin"))
    core.RAG_DB_FILE = tmp_path / "rag.duckdb"
    core.RAG_LANCE_DIR = tmp_path / "rag.lance"
    rag.reset_connection()
    yield
    rag.reset_connection()
    core.RAG_DB_FILE, core.RAG_LANCE_DIR = prev
    crypto.clear_key()


def use(backend):
    """Select a backend, skipping the test when its dependency is absent."""
    if backend == "lance":
        pytest.importorskip("lancedb", reason="lancedb not installed")
    rag.set_backend(backend)
    assert rag.backend_name() == backend


def embed(texts):
    """Deterministic pseudo-embedding: same text -> same vector, always."""
    out = []
    for t in texts:
        rnd = random.Random(hashlib.sha256(t.encode()).digest())
        out.append([rnd.uniform(-1, 1) for _ in range(DIM)])
    return out


_WORDS = "ship harbour signal engine rope captain storm chart anchor lantern tide".split()


def book(seed, paras=20, words=60):
    rnd = random.Random(seed)
    return "\n\n".join(" ".join(rnd.choice(_WORDS) for _ in range(words))
                       for _ in range(paras))


def upsert(source_id, items, embed_fn=embed, **kw):
    """rag.upsert_items with the test defaults, returning (counts, meta)."""
    kw.setdefault("wave_size", 40)
    counts = rag.upsert_items("library", source_id, items, embed_fn, MODEL, **kw)
    return counts, counts.pop("__meta__")


# ------------------------------ round trip ------------------------------
@pytest.mark.parametrize("backend", BACKENDS)
def test_upsert_then_count_and_list(backend):
    use(backend)
    counts, meta = upsert("lib", [("a", book(1), {"label": "A"}),
                                  ("b", book(2), {"label": "B"})])
    total = counts["a"] + counts["b"]
    assert meta["complete"] == {"a", "b"}          # both fully written
    assert meta["failed"] == 0
    assert rag.count_chunks("library", "lib") == total
    listed = {i["item_id"]: i["chunks"] for i in rag.list_items("library", "lib")}
    assert listed == {"a": counts["a"], "b": counts["b"]}


@pytest.mark.parametrize("backend", BACKENDS)
def test_stored_hashes_match_chunk_count(backend):
    use(backend)
    counts, _ = upsert("lib", [("a", book(3), {})])
    # stored_hashes drives compile's staleness check, so it must be per-item and ordered.
    assert len(rag.stored_hashes("library", "lib", "a", MODEL)) == counts["a"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_reupsert_reuses_vectors_and_does_not_duplicate(backend):
    use(backend)
    items = [("a", book(4), {}), ("b", book(5), {})]
    counts, _ = upsert("lib", items)
    total = counts["a"] + counts["b"]

    calls = {"n": 0}

    def counting(texts):
        calls["n"] += len(texts)
        return embed(texts)

    _, meta = upsert("lib", items, embed_fn=counting)
    assert calls["n"] == 0                          # every vector came from the cache
    assert meta["cached"] == total
    assert rag.count_chunks("library", "lib") == total   # replaced, not appended


@pytest.mark.parametrize("backend", BACKENDS)
def test_editing_one_paragraph_reembeds_almost_nothing(backend):
    """The reuse cache is keyed on chunk text, so an edit costs a couple of chunks
    rather than re-embedding the whole document."""
    use(backend)
    original = book(6, paras=30)
    counts, _ = upsert("lib", [("a", original, {})])

    calls = {"n": 0}

    def counting(texts):
        calls["n"] += len(texts)
        return embed(texts)

    edited = original.replace(original.split("\n\n")[7],
                              "the lighthouse keeper wrote nothing that night")
    upsert("lib", [("a", edited, {})], embed_fn=counting)
    assert calls["n"] <= 5, f"re-embedded {calls['n']} chunks for a one-paragraph edit"


# ------------------------------ retrieval ------------------------------
@pytest.mark.parametrize("backend", BACKENDS)
def test_retrieval_modes(backend):
    use(backend)
    marker = "the lighthouse keeper's log survived the wreck"
    upsert("lib", [("a", book(7), {}), ("b", book(8) + "\n\n" + marker, {})])
    qv = embed(["harbour lantern signal"])[0]

    assert len(rag.retrieve("library", ["lib"], [qv], MODEL, 5, mode="vector")) == 5
    kw = rag.retrieve("library", ["lib"], None, MODEL, 5, mode="keyword",
                      queries=["lighthouse keeper"])
    assert any("lighthouse" in r["content"] for r in kw)   # found by words, not vectors
    assert rag.retrieve("library", ["lib"], [qv], MODEL, 5, mode="hybrid",
                        queries=["lighthouse keeper"])


@pytest.mark.parametrize("backend", BACKENDS)
def test_retrieval_is_scoped_to_the_requested_library(backend):
    use(backend)
    upsert("lib1", [("a", book(9), {})])
    upsert("lib2", [("z", book(10), {})])
    qv = embed(["storm chart compass"])[0]
    hits = rag.retrieve("library", ["lib2"], [qv], MODEL, 5, mode="vector")
    assert hits and {h["source_id"] for h in hits} == {"lib2"}


@pytest.mark.parametrize("backend", BACKENDS)
def test_vector_retrieval_is_scoped_to_the_embedding_model(backend):
    """Vectors from a different embedding model must never be searched.

    Lance keys its tables by vector WIDTH, so two models of the same width (768 covers
    nomic-embed-text, bge-base and gte-base alike) land in one table. Without a model
    filter, a scope still holding the previous model's vectors gets ranked against the
    new model's query vector — confident nonsense rather than an honest empty result.
    DuckDB has always filtered on model; this pins both backends to that behaviour.
    """
    use(backend)
    rag.upsert_items("library", "lib", [("a", book(30), {})], embed, "model-a",
                     wave_size=40)
    qv = embed(["storm chart compass"])[0]
    assert rag.retrieve("library", ["lib"], [qv], "model-a", 5, mode="vector")
    assert rag.retrieve("library", ["lib"], [qv], "model-b", 5, mode="vector") == []


@pytest.mark.parametrize("backend", BACKENDS)
def test_keyword_retrieval_ignores_the_embedding_model(backend):
    """BM25 scores chunk TEXT, which doesn't depend on who embedded it — so the model
    filter that guards vector search must not leak into the keyword path."""
    use(backend)
    rag.upsert_items("library", "lib", [("a", "the lantern swung on the anchor rope", {})],
                     embed, "model-a", wave_size=40)
    assert rag.retrieve("library", ["lib"], None, "model-b", 3,
                        mode="keyword", queries=["lantern anchor"])


@pytest.mark.parametrize("backend", BACKENDS)
def test_apostrophe_in_keyword_query_is_safe(backend):
    """Scope filters are built as SQL-ish predicates on the Lance side, so quoting has
    to survive an apostrophe rather than breaking the query."""
    use(backend)
    upsert("lib", [("a", "o'brien signed the captain's log", {})])
    assert isinstance(rag.retrieve("library", ["lib"], None, MODEL, 3,
                                   mode="keyword", queries=["o'brien"]), list)


# ------------------------------ deletion ------------------------------
@pytest.mark.parametrize("backend", BACKENDS)
def test_prune_delete_item_and_delete_source(backend):
    use(backend)
    counts, _ = upsert("lib", [("a", book(11), {}), ("b", book(12), {})])
    assert rag.prune_items("library", "lib", ["a"]) == counts["b"]
    assert rag.count_chunks("library", "lib") == counts["a"]
    rag.delete_item("library", "lib", "a")
    assert rag.count_chunks("library", "lib") == 0

    upsert("lib2", [("z", book(13), {})])
    rag.delete_source("lib2")
    assert rag.count_chunks("library", "lib2") == 0


# ------------------------------ failure / cancellation ------------------------------
@pytest.mark.parametrize("backend", BACKENDS)
def test_dead_embedder_marks_item_incomplete_but_keeps_text(backend):
    """A failed embed must never let compile record the item as cleanly done — that
    was how a dead server silently produced a broken index."""
    use(backend)

    def broken(texts):
        raise RuntimeError("connection refused")

    _, meta = upsert("lib", [("x", book(14), {})], embed_fn=broken)
    assert meta["failed"] > 0
    assert "x" not in meta["complete"]
    assert rag.count_chunks("library", "lib") > 0   # still searchable by keyword


@pytest.mark.parametrize("backend", BACKENDS)
def test_keyword_only_indexing_is_not_a_failure(backend):
    """embed_fn=None is a deliberate keyword-only index, not a broken embedder."""
    use(backend)
    _, meta = upsert("lib", [("k", book(15), {})], embed_fn=None)
    assert meta["failed"] == 0
    assert "k" in meta["complete"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_cancellation_leaves_the_item_incomplete(backend):
    use(backend)
    stop = threading.Event()
    seen = {"n": 0}

    def slow(texts):
        seen["n"] += len(texts)
        if seen["n"] > 20:
            stop.set()
        return embed(texts)

    _, meta = upsert("lib", [("big", book(16, paras=60), {})], embed_fn=slow,
                     wave_size=10, batch_size=8, stop_event=stop)
    assert meta["stopped"]
    assert "big" not in meta["complete"]        # so it recompiles rather than reading done


# ------------------------------ compile orchestration ------------------------------
def test_cancelling_a_forced_rebuild_keeps_the_existing_vectors(tmp_path, monkeypatch):
    """Stopping a compile must not delete what was already indexed.

    The prune used to run against the MANIFEST, which deliberately omits anything that
    didn't embed cleanly. So every item a stopped run hadn't reached yet lost its stored
    chunks — and under "Force full rebuild", where every item is queued, that emptied the
    whole library. Pruning against the library's current items removes deleted items (the
    actual point) without touching items that still exist.
    """
    from app import compile as compile_mod

    monkeypatch.setattr(core, "COMPILED_FILE", tmp_path / "compiled.json")
    rag.set_backend("duckdb")

    items = [{"id": f"i{n}", "type": "write", "label": f"Doc {n}", "content": book(40 + n)}
             for n in range(4)]
    lib = {"id": "lib", "name": "Shelf", "items": items}

    first = compile_mod.compile_library(lib, embed, MODEL)
    assert first["state"] == "compiled"
    indexed = rag.count_chunks("library", "lib")
    assert indexed > 0

    # Force-rebuild everything, then cancel immediately.
    stop = threading.Event()
    stop.set()
    compile_mod.compile_library(lib, embed, MODEL, force=True, stop_event=stop)
    assert rag.count_chunks("library", "lib") == indexed

    # An item genuinely removed from the library IS still pruned.
    lib["items"] = items[:2]
    compile_mod.compile_library(lib, embed, MODEL)
    remaining = {i["item_id"] for i in rag.list_items("library", "lib")}
    assert remaining == {"i0", "i1"}


# ------------------------------ parity ------------------------------
def test_backends_agree_on_retrieval():
    """The regression guard that matters most.

    Lance defaults to L2 distance while the rest of the app uses cosine; omitting
    .metric("cosine") silently returns a *different* ranking rather than raising. Only a
    cross-backend comparison catches that class of bug.
    """
    pytest.importorskip("lancedb", reason="lancedb not installed")
    items = [("a", book(17), {}), ("b", book(18), {})]
    qv = embed(["storm chart compass harbour"])[0]

    rag.set_backend("duckdb")
    upsert("lib", items)
    duck = rag.retrieve("library", ["lib"], [qv], MODEL, 5, mode="vector")

    rag.set_backend("lance")
    upsert("lib", items)
    lance = rag.retrieve("library", ["lib"], [qv], MODEL, 5, mode="vector")

    assert duck and lance
    assert duck[0]["id"] == lance[0]["id"], "backends disagree on the nearest chunk"
    overlap = len({d["id"] for d in duck} & {d["id"] for d in lance})
    assert overlap >= 4, f"only {overlap}/5 shared hits — check the distance metric"


# ------------------------------ migration ------------------------------
def test_migrate_duckdb_to_lance_copies_and_verifies():
    pytest.importorskip("lancedb", reason="lancedb not installed")
    from app.vectorstore import migrate as vs_migrate

    rag.set_backend("duckdb")
    upsert("lib", [("a", book(19), {}), ("b", book(20), {})])
    before = rag.count_chunks("library", "lib")

    summary = vs_migrate.migrate_duckdb_to_lance()
    assert summary["moved"] == summary["total"] == before
    assert summary["verified"] and not summary["mismatches"]

    rag.set_backend("lance")
    assert rag.count_chunks("library", "lib") == before
    # The DuckDB store is a copy source, never a move source, so switching back is lossless.
    assert core.RAG_DB_FILE.exists()
    rag.set_backend("duckdb")
    assert rag.count_chunks("library", "lib") == before


# ------------------------------ encryption sweep interaction ------------------------------
def _fake_lance_store(root):
    """A directory shaped like a real Lance store: nested *.lance fragments, manifests,
    transaction logs and index segments — all plaintext, as the engine writes them."""
    store = root / "rag.lance" / "chunks_768.lance"
    for rel in ("data/0123abc.lance", "_versions/18446744073709551615.manifest",
                "_versions/latest_version_hint.json", "_transactions/0-abc.txn",
                "_indices/deadbeef/index.idx", "_indices/deadbeef/part_0_tokens.lance"):
        p = store / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"LANCE-INTERNAL-BINARY-PAYLOAD" * 4)
    return root / "rag.lance"


def test_encryption_sweep_leaves_the_lance_store_alone(tmp_path, monkeypatch):
    """Regression: migrate.run() walks the whole data profile and encrypts plaintext
    files. The Lance store lives inside that profile and is deliberately plaintext, so
    without a carve-out the sweep grows every internal file by the 36-byte
    header+nonce+tag and the store can no longer be opened
    (LanceError(IO): file size is too small). This destroyed a real 6,785-row store once.
    """
    from app import migrate

    data = tmp_path / "data"
    (data / "profiles" / "default").mkdir(parents=True)
    store = _fake_lance_store(data / "profiles" / "default")
    ordinary = data / "profiles" / "default" / "chats.json"
    ordinary.write_text('[{"id": "x"}]', encoding="utf-8")
    before = {p: p.read_bytes() for p in store.rglob("*") if p.is_file()}

    monkeypatch.setattr(core, "DATA_DIR", data)
    monkeypatch.setattr(core, "SETTINGS_DIR", tmp_path / "settings")
    crypto.set_active_key(b"\x42" * 32)
    migrate.run()

    for p, original in before.items():
        assert p.read_bytes() == original, f"sweep modified {p.name} inside the Lance store"
        assert not crypto.is_encrypted(p.read_bytes())
    # ...while an ordinary app file in the same profile IS still encrypted.
    assert crypto.is_encrypted(ordinary.read_bytes())


def test_forgot_password_wipe_removes_the_plaintext_lance_store(tmp_path, monkeypatch):
    """The Lance store is plaintext, so unlike the encrypted files it would stay fully
    readable after a "wipe my data" reset. It must be deleted outright."""
    from app import migrate

    data = tmp_path / "data"
    (data / "profiles" / "default").mkdir(parents=True)
    store = _fake_lance_store(data / "profiles" / "default")
    monkeypatch.setattr(core, "DATA_DIR", data)
    monkeypatch.setattr(core, "SETTINGS_DIR", tmp_path / "settings")

    crypto.set_active_key(b"\x42" * 32)
    migrate.wipe_encrypted()
    assert not store.exists(), "plaintext chunk text survived a forgot-password wipe"


def test_an_encrypted_lance_store_reports_what_to_do_about_it(tmp_path, monkeypatch):
    """The sweep above is fixed, but a store damaged BEFORE that fix is still on disk.
    Opening one used to surface only Lance's own words —

        LanceError(IO): file size is too small,
        C:\\Users\\runneradmin\\.cargo\\...\\lance-io-9.0.0\\src\\utils.rs:92:20

    — a Rust panic site from whatever machine built the wheel, naming neither the store
    nor the remedy. Detect the app's own encryption header directly and say so."""
    from app.vectorstore import VectorStoreError, lance_backend

    store = _fake_lance_store(tmp_path)
    victim = store / "chunks_768.lance" / "_versions" / "latest_version_hint.json"
    victim.write_bytes(crypto.MAGIC + b"\x00" * 40)
    monkeypatch.setattr(core, "RAG_LANCE_DIR", store)

    with pytest.raises(VectorStoreError) as excinfo:
        lance_backend.LanceBackend()._conn()

    msg = str(excinfo.value)
    assert str(store) in msg                  # names the folder to delete
    assert "Compile Data" in msg              # names the way back
    assert "utils.rs" not in msg              # not the Rust panic site


def test_a_healthy_lance_store_is_not_flagged(tmp_path, monkeypatch):
    """The detector must not fire on a normal store, or it would brick every install."""
    pytest.importorskip("lancedb", reason="lancedb not installed")
    from app.vectorstore import lance_backend

    store = _fake_lance_store(tmp_path)
    monkeypatch.setattr(core, "RAG_LANCE_DIR", store)
    assert lance_backend.LanceBackend()._conn() is not None


def test_a_damaged_table_is_never_created_over(tmp_path, monkeypatch):
    """_table() falls back to create_table when open_table raises, because a missing
    table is ordinary. A CORRUPT one must not take that path: creating a fresh table
    over a store the user might still recover destroys it and buries the cause."""
    from app.vectorstore import VectorStoreError, lance_backend

    monkeypatch.setattr(core, "RAG_LANCE_DIR", tmp_path / "rag.lance")
    be = lance_backend.LanceBackend()

    class Boom:
        def open_table(self, name):
            raise RuntimeError("lance error: LanceError(IO): file size is too small, "
                               "/home/runner/.cargo/lance-io/src/utils.rs:92:20")

        def create_table(self, name, schema=None):
            raise AssertionError("created a table over a damaged store")

    monkeypatch.setattr(be, "_conn", lambda: Boom())
    with pytest.raises(VectorStoreError):
        be._table(768, create=True)


def test_a_missing_table_still_creates_normally(tmp_path, monkeypatch):
    """The other half of the case above: an ordinary "no such table" must still fall
    through to create_table, or a fresh install could never write its first chunk."""
    from app.vectorstore import lance_backend

    monkeypatch.setattr(core, "RAG_LANCE_DIR", tmp_path / "rag.lance")
    be = lance_backend.LanceBackend()
    made = []

    class Empty:
        def open_table(self, name):
            raise RuntimeError(f"Table '{name}' was not found")

        def create_table(self, name, schema=None):
            made.append(name)
            return "table-handle"

    monkeypatch.setattr(be, "_conn", lambda: Empty())
    assert be._table(768, create=True) == "table-handle"
    assert made == ["chunks_768"]


def test_a_damaged_store_does_not_read_as_an_empty_one(tmp_path, monkeypatch):
    """_existing_dims swallows errors and returns [], which is right for "nothing
    indexed yet" and very wrong for "unreadable": every read would quietly return
    nothing and every compile would look like it simply found no data."""
    from app.vectorstore import VectorStoreError, lance_backend

    store = _fake_lance_store(tmp_path)
    (store / "chunks_768.lance" / "data" / "0123abc.lance").write_bytes(
        crypto.MAGIC + b"\x00" * 40)
    monkeypatch.setattr(core, "RAG_LANCE_DIR", store)

    with pytest.raises(VectorStoreError):
        lance_backend.LanceBackend()._existing_dims()


def test_switching_backend_keeps_each_store_independent():
    pytest.importorskip("lancedb", reason="lancedb not installed")
    rag.set_backend("duckdb")
    upsert("lib", [("a", book(21), {})])
    duck_n = rag.count_chunks("library", "lib")

    rag.set_backend("lance")
    assert rag.count_chunks("library", "lib") == 0    # separate store, not shared state

    rag.set_backend("duckdb")
    assert rag.count_chunks("library", "lib") == duck_n
