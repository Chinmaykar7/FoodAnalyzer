from pathlib import Path


class PromptLoader:
    """
    Utility class for loading prompt templates from disk.
    """

    PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts"

    @classmethod
    def load(cls, filename: str) -> str:
        """
        Load a prompt file.

        Args:
            filename: Name of the prompt file.

        Returns:
            Prompt text.

        Raises:
            FileNotFoundError: If the prompt file does not exist.
        """

        path = cls.PROMPT_DIR / filename

        if not path.exists():
            raise FileNotFoundError(f"Prompt not found: {path}")

        return path.read_text(encoding="utf-8")