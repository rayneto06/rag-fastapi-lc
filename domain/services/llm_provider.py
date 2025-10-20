from typing import Protocol

class LLMProvider(Protocol):
    def generate(self, prompt: str, max_tokens: int | None = None) -> str:  # pragma: no cover
        ...
