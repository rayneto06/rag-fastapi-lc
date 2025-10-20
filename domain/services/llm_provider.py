from typing import Protocol, List, Tuple

class LLMProvider(Protocol):
    def generate(self, question: str, context_snippets: List[Tuple[str, dict]] | None = None) -> str:  # pragma: no cover
        ...
