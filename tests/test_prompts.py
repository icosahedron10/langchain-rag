from ragchat.prompts import orchestrator_prompt, retrieval_prompt


def test_virtual_workspace_path_mapping_is_sandbox_only() -> None:
    mapping = (
        "File tools address the workspace from the virtual root `/`; "
        "shell commands see those same files under `/workspace`."
    )

    sandbox_prompt = orchestrator_prompt(True)
    rag_only_prompt = orchestrator_prompt(False)

    assert mapping in sandbox_prompt
    assert mapping not in rag_only_prompt
    assert "/workspace" not in rag_only_prompt
    assert "`execute` tool" not in rag_only_prompt


def test_orchestrator_prompt_pins_grounding_citations_and_gap_reporting() -> None:
    prompt = orchestrator_prompt(False)

    assert "call search_corpus first" in prompt
    assert "verbatim `evidence` passages" in prompt
    assert "document and page" in prompt
    assert "Never invent or extrapolate citations" in prompt
    assert "remains unresolved" in prompt
    assert "Never guess" in prompt


def test_orchestrator_prompt_forbids_pages_outside_this_turn_and_backfilled_entries() -> None:
    prompt = orchestrator_prompt(False)

    assert "that exact page appears in a `sources` entry returned during this turn" in prompt
    assert "does not contain the entry the user asked about, say so and stop" in prompt
    assert "features, traits, or mechanics from general knowledge" in prompt
    assert '"assuming the usual X"' in prompt
    assert '"the commonly known X features are"' in prompt


def test_retrieval_prompt_delegates_query_and_sufficiency_decisions_to_the_model() -> None:
    prompt = retrieval_prompt()

    assert "exactly one tool" in prompt
    assert "qdrant_hybrid_search" in prompt
    assert "use 3-8 words" in prompt
    assert "rather than a full question" in prompt
    assert "judge for yourself whether the evidence is sufficient" in prompt
    assert "hard budget of three searches" in prompt
    assert "selected_point_ids" in prompt
