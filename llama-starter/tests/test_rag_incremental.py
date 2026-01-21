"""Unit tests for incremental indexing behavior in SimpleRAG."""

from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from llama_rag import rag as rag_module


class FakeRetriever:
    """Minimal retriever that stores search kwargs for assertions."""

    def __init__(self, store, search_kwargs=None):
        self.store = store
        self.search_kwargs = search_kwargs or {}

    def invoke(self, query):
        k = self.search_kwargs.get("k", len(self.store.docs))
        return self.store.docs[:k]


class FakeVectorStore:
    """In-memory vector store stub used for unit tests."""

    def __init__(self, docs=None):
        self.docs = list(docs or [])
        self.docstore = SimpleNamespace(_dict={str(i): doc for i, doc in enumerate(self.docs)})

    @classmethod
    def from_documents(cls, docs, embeddings):
        return cls(list(docs))

    def add_documents(self, docs):
        start = len(self.docs)
        self.docs.extend(docs)
        for offset, doc in enumerate(docs):
            self.docstore._dict[str(start + offset)] = doc

    def as_retriever(self, search_kwargs=None):
        return FakeRetriever(self, search_kwargs or {})

    def save_local(self, path):
        return None


class FakeChain:
    """Simple chain stub that captures invocations."""

    def __init__(self):
        self.calls = []

    def invoke(self, question):
        self.calls.append(question)
        return "ok"


def test_ingest_documents_appends_chunks(monkeypatch):
    """Adding documents should append to existing vector store instead of overwriting."""

    monkeypatch.setattr(
        rag_module,
        "get_embeddings",
        lambda type="ollama", model="embeddinggemma": object(),
    )
    monkeypatch.setattr(
        rag_module,
        "get_llm",
        lambda type="ollama", model="llama3.2", temperature=0.7: object(),
    )

    def fake_get_vectorstore(embeddings, type="faiss"):
        return FakeVectorStore

    monkeypatch.setattr(rag_module, "get_vectorstore", fake_get_vectorstore)
    monkeypatch.setattr(
        rag_module,
        "create_rag_chain",
        lambda retriever, llm, prompt_template=None: FakeChain(),
    )

    rag = rag_module.SimpleRAG()

    docs = [
        Document(page_content="alpha", metadata={"id": 1}),
        Document(page_content="beta", metadata={"id": 2}),
        Document(page_content="gamma", metadata={"id": 3}),
    ]

    rag.ingest_documents([docs[0]], chunk_size=1000, chunk_overlap=0)
    assert len(rag.vectorstore.docs) == 1
    assert rag.vectorstore.docs[0].metadata["id"] == 1

    rag.ingest_documents([docs[1]], chunk_size=1000, chunk_overlap=0)
    assert len(rag.vectorstore.docs) == 2
    assert [doc.metadata["id"] for doc in rag.vectorstore.docs] == [1, 2]

    rag.retriever.search_kwargs["k"] = 2
    rag.ingest_documents([docs[2]], chunk_size=1000, chunk_overlap=0)
    assert len(rag.vectorstore.docs) == 3
    assert rag.retriever.search_kwargs["k"] == 2
```}