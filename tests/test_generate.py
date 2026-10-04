from helpers import good_answer
from ticketrag.generate import build_sources, generate_answer, render_template, validate
from ticketrag.parse import Parsed


def _setup(retriever):
    hit = retriever.search("loud music party noise", k=1, top_resolutions=5)[0]
    return hit, build_sources(hit, [], n_res=3)


def _parsed(hit):
    return Parsed(hit, 1, "high", 3, "frustrated", "stub")


def test_validator_drops_uncited_unknown_and_invented_numbers(retriever):
    _, src = _setup(retriever)
    out = {"opening": "I am sorry about this.",
           "summary": {"text": "ok", "citations": ["P1"]},
           "steps": [{"text": "no citation", "citations": []},
                     {"text": "unknown source", "citations": ["R99"]},
                     {"text": "closed in 9999 hours", "citations": ["P1"]},
                     {"text": "this one is fine", "citations": ["R1"]}]}
    clean, violations = validate(out, src)
    assert [s["text"] for s in clean["steps"]] == ["this one is fine"]
    assert len(violations) == 3


def test_validator_rejects_opening_with_numbers(retriever):
    _, src = _setup(retriever)
    out = good_answer() | {"opening": "I know this took 3 days."}
    clean, violations = validate(out, src)
    assert clean["opening"] == "" and any("opening" in v for v in violations)


def test_numbers_copied_from_sources_are_allowed(retriever):
    hit, src = _setup(retriever)
    share = f"{hit.resolutions[0]['share']:.0%}"
    out = good_answer() | {"steps": [{"text": f"Tell the customer {share} of tickets ended this way.", "citations": ["P1", "R1"]}]}
    _, violations = validate(out, src)
    assert violations == []


def test_retry_with_feedback_recovers(retriever, stub):
    hit, src = _setup(retriever)
    bad = good_answer() | {"steps": [{"text": "uncited step", "citations": []}]}
    stub.answers = [bad, good_answer()]
    answer, report = generate_answer(stub, "loud music", _parsed(hit), src)
    assert report.attempts == 2 and report.first_attempt_violations == 1 and report.violations == []
    assert len(answer["steps"]) == 2


def test_outcomes_are_rendered_by_code_with_citations(retriever, stub):
    hit, src = _setup(retriever)
    answer, _ = generate_answer(stub, "loud music", _parsed(hit), src)
    assert len(answer["outcomes"]) == 3
    assert all(o["citations"][0] == "P1" and o["citations"][1] in src.items for o in answer["outcomes"])
    assert answer["outcomes"][0]["text"].startswith(f"{hit.resolutions[0]['share']:.0%} of tickets")


def test_template_answer_passes_its_own_validator(retriever):
    hit, src = _setup(retriever)
    answer, report = render_template(hit, src)
    _, violations = validate({"opening": "No opening needed.", "summary": answer["summary"], "steps": answer["steps"]}, src)
    assert report.method == "template" and violations == []


def test_long_durations_are_given_in_days_once(retriever):
    hit, _ = _setup(retriever)
    hit.median_close_hours, hit.p90_close_hours = 334.5, 400.0
    src = build_sources(hit, [])
    assert "334.5 hours (about 14 days)" in src.block


def test_truncated_source_text_is_detected_and_flagged():
    from ticketrag.generate import is_truncated

    assert is_truncated("x" * 499 + " see if a")  # cut off mid-sentence at the 500-char source limit
    assert not is_truncated("Short complete sentence.")
    assert not is_truncated("y" * 600 + " ends properly.")


def test_future_tense_prediction_is_rejected(retriever):
    _, src = _setup(retriever)
    out = good_answer() | {"steps": [{"text": "Tell the customer that the agency will respond and may investigate.", "citations": ["P1"]},
                                     {"text": "Tell the customer what past tickets recorded.", "citations": ["R1"]}]}
    clean, violations = validate(out, src)
    assert [s["text"] for s in clean["steps"]] == ["Tell the customer what past tickets recorded."]
    assert any("predicts the future" in v for v in violations)


def test_outcomes_include_the_unlisted_remainder(retriever):
    hit, src = _setup(retriever)
    hit.resolutions = hit.resolutions[:2]  # pretend only two of three outcomes are shown
    src = build_sources(hit, [])
    from ticketrag.generate import outcomes_from_sources

    out = outcomes_from_sources(hit, src, n=2)
    shown = sum(float(o["text"].split("%")[0]) for o in out)
    assert out[-1]["text"].endswith("(not listed individually).") and 99 <= shown <= 101
