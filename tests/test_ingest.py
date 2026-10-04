import pandas as pd
import pytest
from helpers import FakeEmbedder, StubLLM, make_raw

from ticketrag.ingest import apply_batch
from ticketrag.retrieve import PatternRetriever, pid_to_int

NEW_CLASS = [(("Broadband Service", "Outage", ""), [(0.4, "Provider restored the line remotely."),
                                                   (0.3, "Technician visit scheduled."),
                                                   (0.3, "Customer advised to replace the router.")], 35)]  # top-1 share 0.4 -> rag_core


def test_new_class_becomes_searchable_without_rebuild(state):
    emb, llm = FakeEmbedder(), StubLLM()
    report = apply_batch(state, make_raw(NEW_CLASS, prefix="N"), emb, llm)
    assert len(report["new_patterns"]) == 1 and report["examples_generated"] == 4
    assert report["ingested"] == 35

    retriever = PatternRetriever(state / "index", state, embedder=emb)
    top = retriever.search("broadband service outage", k=1)[0]
    assert (top.complaint_type, top.descriptor) == ("Broadband Service", "Outage")
    assert top.total_cases == 35 and top.tier == "rag_core"


def test_reingesting_same_batch_is_a_noop(state):
    emb, llm = FakeEmbedder(), StubLLM()
    batch = make_raw(NEW_CLASS, prefix="N")
    apply_batch(state, batch, emb, llm)
    before = pd.read_parquet(state / "tickets.parquet")
    again = apply_batch(state, batch, emb, llm)
    assert again["ingested"] == 0 and again["duplicates_skipped"] == 35
    assert len(pd.read_parquet(state / "tickets.parquet")) == len(before)


def test_existing_pattern_statistics_refresh(state):
    more_noise = make_raw([SPEC_NOISE], prefix="X")
    report = apply_batch(state, more_noise, FakeEmbedder(), StubLLM())
    assert report["new_patterns"] == [] and report["updated_patterns"] == 1
    pat = pd.read_parquet(state / "patterns.parquet")
    assert int(pat.loc[pat["descriptor"] == "Loud Music/Party", "total_cases"].iloc[0]) == 60 + 20


def test_hot_reload_picks_up_new_state(state):
    emb = FakeEmbedder()
    retriever = PatternRetriever(state / "index", state, embedder=emb)
    n_before = len(retriever.patterns)
    apply_batch(state, make_raw(NEW_CLASS, prefix="N"), emb, StubLLM())
    assert len(retriever.patterns) == n_before  # not visible until reload
    retriever.reload()
    assert len(retriever.patterns) == n_before + 1


def test_embedder_mismatch_is_rejected(state):
    class Other(FakeEmbedder):
        model_name = "different-model"

    with pytest.raises(ValueError):
        apply_batch(state, make_raw(NEW_CLASS, prefix="N"), Other(), StubLLM())


def test_new_pattern_ids_fit_vector_id_space(state):
    apply_batch(state, make_raw(NEW_CLASS, prefix="N"), FakeEmbedder(), StubLLM())
    pat = pd.read_parquet(state / "patterns.parquet")
    assert all(pid_to_int(p) << 4 < 2**63 for p in pat["pattern_id"])


SPEC_NOISE = (("Noise - Residential", "Loud Music/Party", ""), [(1.0, "Police responded and observed no criminal violation upon arrival. Please contact 311 if it persists.")], 20)
