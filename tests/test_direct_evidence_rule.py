"""The 'Direct Evidence Only' rule must be present in the classification prompt
so the classifier keeps direction neutral for indirect/inferred connections."""
from locus.core import classifier


def test_direct_evidence_rule_in_prompt():
    prompt = classifier.CLASSIFICATION_PROMPT
    assert "CRITICAL RULE — Direct Evidence Only:" in prompt
    assert "One logical step maximum." in prompt
    assert "No implied consequences, second-order effects, or chains of inference." in prompt
    # The neutral fallback is the whole point of the rule.
    assert "direction MUST be neutral" in prompt
    assert "Does this headline alone almost prove the market outcome? If not — neutral." in prompt


def test_direct_evidence_rule_survives_prompt_formatting():
    """The rule text has no stray braces, so .format() still renders it."""
    rendered = classifier.CLASSIFICATION_PROMPT.format(
        question="Will X happen?",
        threshold_line="",
        yes_price=0.5,
        time_remaining="3 days",
        headline="Something happened",
        source="rss",
        track_record="(none)",
    )
    assert "Direct Evidence Only" in rendered
    assert "{" not in rendered.split("Respond with ONLY valid JSON")[0]
