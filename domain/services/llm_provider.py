from typing import Protocol


class LLMProvider(Protocol):
    def generate(
        self, question: str, context_snippets: list[tuple[str, dict]] | None = None
    ) -> str:  # pragma: no cover
        ...
