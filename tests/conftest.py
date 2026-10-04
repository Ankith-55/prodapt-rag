import pytest
from helpers import FakeEmbedder, StubLLM, make_raw

from ticketrag.ingest import init_state
from ticketrag.patterns import clean_tickets
from ticketrag.pipeline import Assistant
from ticketrag.retrieve import PatternRetriever


@pytest.fixture
def state(tmp_path):
    clean, _ = clean_tickets(make_raw())
    init_state(clean, tmp_path / "state", FakeEmbedder())
    return tmp_path / "state"


@pytest.fixture
def retriever(state):
    return PatternRetriever(state / "index", state, embedder=FakeEmbedder())


@pytest.fixture
def stub():
    return StubLLM()


@pytest.fixture
def assistant(retriever, stub):
    return Assistant(retriever, stub, gate=0.6)
