"""All prompt text for the application lives in this package.

Prompts are composed from small named sections. No substantial prompt text
may appear in controllers, managers, providers, retrieval code, or routes.
"""

from ragchat.prompts.orchestrator import orchestrator_prompt
from ragchat.prompts.retrieval import retrieval_prompt

__all__ = ["orchestrator_prompt", "retrieval_prompt"]
