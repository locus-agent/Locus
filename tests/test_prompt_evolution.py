"""Prompt-evolution dashboard payload: the version-over-version 'what changed'
diff and the full evolution block exported to status.json."""
from locus.core import export_status


# --- _diff_key_changes -------------------------------------------------------

def test_new_section_header_is_added():
    prev = "## Task\nclassify this\nSPORTS MARKETS — Extra Strict Rules:\nbe strict"
    latest = prev + "\nPOLITICS MARKETS — Calibration Warning:\nwatch politics"
    assert export_status._diff_key_changes(prev, latest) == [
        "Added Politics Markets — Calibration Warning"
    ]


def test_reworded_section_is_refined_not_added():
    # Same topic ("SPORTS MARKETS"), different wording -> a refinement.
    prev = "SPORTS MARKETS — Extra Strict Rules:"
    latest = "SPORTS MARKETS — Strict but Calibrated Rules:"
    assert export_status._diff_key_changes(prev, latest) == [
        "Refined Sports Markets — Strict but Calibrated Rules"
    ]


def test_no_structural_change_falls_back_to_char_delta():
    prev = "plain text with no headers"
    latest = "plain text with no headers and a bit more"
    changes = export_status._diff_key_changes(prev, latest)
    assert len(changes) == 1
    assert "chars" in changes[0]


def test_identical_text_yields_no_changes():
    txt = "## Task\nNEW SECTION:\nbody"
    assert export_status._diff_key_changes(txt, txt) == []


def test_clean_header_titlecases_shouting_words():
    assert export_status._clean_header("MATERIALITY CALIBRATION REMINDER:") == \
        "Materiality Calibration Reminder"


# --- _prompt_evolution_block -------------------------------------------------

def test_block_empty_when_no_versions(tmp_db):
    assert export_status._prompt_evolution_block() == {
        "version": 0, "last_evolved": None, "lessons_used": 0,
        "key_changes": [], "accuracy_at_evolution": None, "evolution_history": [],
    }


def test_block_v1_diffs_against_base_prompt(tmp_db):
    from locus.core.classifier import CLASSIFICATION_PROMPT
    evolved = CLASSIFICATION_PROMPT + "\nPOLITICS MARKETS — Calibration Warning:\nwatch it"
    tmp_db.save_prompt_version(1, evolved, lessons_count=24, accuracy_at_creation=17.6)

    block = export_status._prompt_evolution_block()
    assert block["version"] == 1
    assert block["last_evolved"] is not None
    assert block["lessons_used"] == 24
    assert block["accuracy_at_evolution"] == 17.6
    assert "Added Politics Markets — Calibration Warning" in block["key_changes"]
    assert len(block["evolution_history"]) == 1
    assert block["evolution_history"][0] == {
        "version": 1,
        "created_at": block["last_evolved"],
        "lessons_count": 24,
        "accuracy": 17.6,
    }


def test_block_v2_diffs_against_v1_and_lists_history(tmp_db):
    tmp_db.save_prompt_version(1, "BASE SECTION:\nx", lessons_count=10, accuracy_at_creation=15.0)
    tmp_db.save_prompt_version(2, "BASE SECTION:\nx\nNEW GUIDANCE:\ny",
                               lessons_count=20, accuracy_at_creation=25.0)

    block = export_status._prompt_evolution_block()
    assert block["version"] == 2
    assert block["accuracy_at_evolution"] == 25.0
    assert "Added New Guidance" in block["key_changes"]
    # History ascending by version (the dashboard sorts for display).
    assert [h["version"] for h in block["evolution_history"]] == [1, 2]
    assert block["evolution_history"][0]["accuracy"] == 15.0
