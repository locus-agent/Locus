"""MarketIndex must heal a corrupted Chroma store by rebuilding."""
from locus.core.market_index import MarketIndex
from locus.markets.gamma import Market

MKT = Market("cond1", "Will X happen?", "ai", 0.5, 0.5, 5000, "", True, [],
             description="desc")


class FakeEmbeddings(list):
    def tolist(self):
        return list(self)


class FakeModel:
    def encode(self, docs, **kwargs):
        return FakeEmbeddings([[0.0, 1.0]] * len(docs))


class FakeCollection:
    def __init__(self, broken=False):
        self.broken = broken
        self.data = {}

    def get(self, include=None):
        return {"ids": list(self.data), "metadatas": list(self.data.values())}

    def upsert(self, ids, embeddings, documents, metadatas):
        if self.broken:
            raise RuntimeError("Failed to apply logs to the hnsw segment writer")
        for i, md in zip(ids, metadatas):
            self.data[i] = md

    def delete(self, ids):
        for i in ids:
            self.data.pop(i, None)

    def count(self):
        return len(self.data)


class FakeClient:
    def __init__(self, first_collection):
        self.collections = [first_collection]
        self.deleted = []

    def delete_collection(self, name):
        self.deleted.append(name)

    def get_or_create_collection(self, name, metadata=None):
        fresh = FakeCollection(broken=False)
        self.collections.append(fresh)
        return fresh


def _index_with(collection):
    idx = MarketIndex(path="/nonexistent")
    idx._client = FakeClient(collection)
    idx._collection = collection
    idx._model = FakeModel()
    idx.ready = True
    return idx


def test_healthy_sync_needs_no_rebuild():
    coll = FakeCollection(broken=False)
    idx = _index_with(coll)
    idx.sync([MKT])
    assert coll.count() == 1
    assert idx._client.deleted == []
    assert idx.ready is True


def test_corrupted_store_is_rebuilt_and_resynced():
    broken = FakeCollection(broken=True)
    idx = _index_with(broken)
    idx.sync([MKT])  # must not raise
    # old collection dropped, fresh one created and populated
    assert idx._client.deleted == ["markets"]
    assert idx._collection is not broken
    assert idx._collection.count() == 1
    assert idx.ready is True


def test_double_failure_raises():
    import pytest

    broken = FakeCollection(broken=True)
    idx = _index_with(broken)
    # make the rebuilt collection broken too
    idx._client.get_or_create_collection = lambda name, metadata=None: FakeCollection(broken=True)
    with pytest.raises(RuntimeError):
        idx.sync([MKT])
