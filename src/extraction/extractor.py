from pathlib import Path

from ollama import chat

from llm.prompt_loader import PromptLoader


class IngredientExtractor:
    """
    Extract ingredients from a food package image using Qwen.
    """

    def __init__(
        self,
        model: str = "qwen3.5:4b",
        temperature: float = 0,
        max_tokens: int = 250,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.prompt = PromptLoader.load(
            "extraction_prompt.txt"
        )

    def extract(self, image_path: str | Path) -> str:
        """
        Extract ingredient text.

        Args:
            image_path:
                Path to the food package image.

        Returns:
            Ingredient list as plain text.
        """

        image_path = Path(image_path).resolve()

        if not image_path.exists():
            raise FileNotFoundError(image_path)

        response = chat(
            model=self.model,
            think=False,
            messages=[
                {
                    "role": "user",
                    "content": self.prompt,
                    "images": [str(image_path)],
                }
            ],
            options={
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        )

        return response.message.content.strip()