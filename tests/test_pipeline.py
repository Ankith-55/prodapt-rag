def test_answers_through_llm_for_rag_core_pattern(assistant, stub):
    r = assistant.ask("loud music party noise next door")
    assert not r.abstained and r.category == "Noise - Residential" and r.tier == "rag_core"
    assert r.method == "llm" and stub.calls == ["parse", "answer"]
    assert r.answer["steps"] and r.answer["outcomes"] and r.source_block
    assert set(r.timing_ms) >= {"retrieve", "parse", "generate", "total"}


def test_fast_lookup_pattern_skips_generation_call(assistant, stub):
    r = assistant.ask("pothole street condition")
    assert r.tier == "fast_lookup" and r.method == "template"
    assert stub.calls == ["parse"]  # no generation LLM call for deterministic patterns


def test_abstains_below_similarity_gate_without_any_llm_call(assistant, stub):
    r = assistant.ask("quantum espresso invoice")
    assert r.abstained and r.abstain_reason.startswith("low_similarity")
    assert stub.calls == [] and r.candidates  # nearest candidates still returned for a human


def test_abstains_when_llm_finds_no_matching_candidate(assistant, stub):
    stub.parse_pick = 0
    r = assistant.ask("loud music party noise next door")
    assert r.abstained and r.abstain_reason == "llm_no_matching_candidate"


def test_severity_policy_notes(assistant, stub):
    stub.severity = 5
    r = assistant.ask("loud music party noise next door")
    assert any("911" in n for n in r.policy_notes) and any("prioritise" in n for n in r.policy_notes)


def test_low_evidence_note_for_small_patterns(state, stub):
    from helpers import FakeEmbedder, make_raw
    from ticketrag.ingest import apply_batch
    from ticketrag.pipeline import Assistant
    from ticketrag.retrieve import PatternRetriever

    tiny = [(("Tattooing", "Dirty Equipment", ""), [(1.0, "The health department inspected.")], 5)]
    apply_batch(state, make_raw(tiny, prefix="S"), FakeEmbedder(), stub)
    a = Assistant(PatternRetriever(state / "index", state, embedder=FakeEmbedder()), stub, gate=0.6)
    r = a.ask("tattooing dirty equipment")
    assert any("Low evidence" in n for n in r.policy_notes)


def test_llm_outage_degrades_to_retrieval_only_answer(assistant, stub):
    stub.fail = True
    r = assistant.ask_safe("loud music party noise next door")
    assert not r.abstained and r.method == "template_degraded"
    assert r.answer["outcomes"] and any("Degraded" in n for n in r.policy_notes)


def test_llm_outage_still_abstains_when_nothing_similar(assistant, stub):
    stub.fail = True
    assert assistant.ask_safe("quantum espresso invoice").abstained


def test_borderline_match_gets_a_note(retriever, stub):
    from ticketrag.pipeline import Assistant

    top = retriever.search("loud music party noise next door", k=1)[0].score
    a = Assistant(retriever, stub, gate=top - 0.01)  # the match is only 0.01 above the gate
    r = a.ask("loud music party noise next door")
    assert any("Borderline match" in n for n in r.policy_notes)


def test_siblings_are_off_by_default_and_opt_in(retriever, stub):
    from ticketrag.pipeline import Assistant

    off = Assistant(retriever, stub, gate=0.6).ask("loud music party noise next door")
    assert [k for k, v in off.sources.items() if v["kind"] == "pattern"] == ["P1"]
    on = Assistant(retriever, stub, gate=0.6, siblings=1).ask("loud music party noise next door")
    assert len([k for k, v in on.sources.items() if v["kind"] == "pattern"]) == 2
