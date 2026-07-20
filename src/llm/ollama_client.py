from pathlib import Path
from typing import Sequence

from ollama import chat


class OllamaClient:
    """
    Client for interacting with an Ollama-hosted language model.

    This class centralizes all communication with the model so that
    higher-level modules (extractor, parser, analyzer, etc.) remain
    independent of the underlying inference backend.
    """

    def __init__(
        self,
        model: str = "qwen3.5:4b",
        temperature: float = 0.0,
        max_tokens: int = 250,
    ) -> None:
        """
        Initialize the Ollama client.

        Args:
            model:
                Name of the Ollama model.

            temperature:
                Sampling temperature.

            max_tokens:
                Maximum number of output tokens.
        """

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(
        self,
        *,
        prompt: str,
        images: Sequence[str | Path] | None = None,
        think: bool = False,
    ) -> str:
        """
        Generate a response from the language model.

        Args:
            prompt:
                Prompt to send to the model.

            images:
                Optional image paths for vision models.

            think:
                Whether to enable the model's reasoning mode.

        Returns:
            Model response as plain text.
        """

        message = {
            "role": "user",
            "content": prompt,
        }

        if images:
            message["images"] = [
                str(Path(image).resolve())
                for image in images
            ]

        response = chat(
            model=self.model,
            think=think,
            messages=[message],
            options={
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        )

        return response.message.content.strip()