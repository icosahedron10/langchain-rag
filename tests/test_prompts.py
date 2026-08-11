from ragchat.prompts import orchestrator_prompt, retrieval_prompt

CORPUS_DESCRIPTION = "the tabletop rulebook corpus"


def test_virtual_workspace_path_mapping_is_sandbox_only() -> None:
    mapping = (
        "File tools address the workspace from the virtual root `/`; "
        "shell commands see those same files under `/workspace`."
    )

    sandbox_prompt = orchestrator_prompt(True, CORPUS_DESCRIPTION)
    rag_only_prompt = orchestrator_prompt(False, CORPUS_DESCRIPTION)

    assert mapping in sandbox_prompt
    assert mapping not in rag_only_prompt
    assert "/workspace" not in rag_only_prompt
    assert "`execute` tool" not in rag_only_prompt


def test_orchestrator_prompt_pins_grounding_citations_and_gap_reporting() -> None:
    prompt = orchestrator_prompt(False, CORPUS_DESCRIPTION)

    assert "call search_corpus first" in prompt
    assert "verbatim `evidence` passages" in prompt
    assert "document and page" in prompt
    assert "Never invent or extrapolate citations" in prompt
    assert "remains unresolved" in prompt
    assert "Never guess" in prompt


def test_orchestrator_prompt_names_the_corpus_domain_and_requires_retrieval_for_it() -> None:
    prompt = orchestrator_prompt(False, CORPUS_DESCRIPTION)

    assert f"The corpus is {CORPUS_DESCRIPTION}." in prompt
    assert "corpus-scoped by default" in prompt
    assert "REQUIRE a search_corpus call" in prompt
    assert "outside the corpus's subject matter entirely" in prompt


def test_retrieval_prompt_delegates_query_and_sufficiency_decisions_to_the_model() -> None:
    prompt = retrieval_prompt(CORPUS_DESCRIPTION)

    assert "exactly one tool" in prompt
    assert "qdrant_hybrid_search" in prompt
    assert "use 3-8 words" in prompt
    assert "rather than a full question" in prompt
    assert "judge for yourself whether the evidence is sufficient" in prompt
    assert "hard budget of three searches" in prompt
    assert "selected_point_ids" in prompt


def test_retrieval_prompt_names_the_corpus_domain_and_pins_in_domain_interpretation() -> None:
    prompt = retrieval_prompt(CORPUS_DESCRIPTION)

    assert f"The corpus is {CORPUS_DESCRIPTION}." in prompt
    assert "interpret ambiguous or informal terms" in prompt
    assert "the domain's own vocabulary" in prompt
