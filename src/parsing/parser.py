import json

from llm.ollama_client import OllamaClient
from llm.prompt_loader import PromptLoader

NUTRIENT_FIELDS = {
    "energy",
    "protein",
    "carbohydrates",
    "total_sugars",
    "added_sugars",
    "total_fat",
    "saturated_fat",
    "trans_fat",
    "dietary_fiber",
    "sodium",
}


class FoodLabelParser:
    """
    Convert extracted food-label text into structured JSON using the LLM.
    """

    def __init__(self) -> None:
        """
        Initialize the food label parser.
        """

        self.client = OllamaClient(max_tokens=4096)

        self.prompt = PromptLoader.load(
            "json_prompt.txt"
        )

    def parse(
        self,
        label_text: str,
    ) -> dict[str, object]:
        """
        Convert food-label text into structured JSON.

        Args:
            label_text:
                Raw food-label text extracted from the image.

        Returns:
            Parsed food data dictionary.

        Raises:
            ValueError: If the label text is empty or the model response
                does not contain a valid JSON object.
        """

        if not label_text.strip():
            raise ValueError(
                "Cannot parse an empty food label."
            )

        prompt = (
            f"{self.prompt}\n\n"
            f"Food Label Text:\n"
            f"{label_text}"
        )

        response = self.client.generate(
            prompt=prompt,
            think=False,
        )

        parsed = self._extract_json_object(response)
        self._validate(parsed)
        return parsed

    @staticmethod
    def _clean_response(response: str) -> str:
        """
        Remove common formatting added around a JSON model response.

        Args:
            response:
                Raw response returned by the language model.

        Returns:
            Cleaned JSON text.
        """

        cleaned_lines = [
            line
            for line in response.splitlines()
            if not line.strip().startswith("```")
        ]

        return "\n".join(cleaned_lines).strip()

    @classmethod
    def _extract_json_object(
        cls,
        response: str,
    ) -> dict[str, object]:
        """
        Extract the first complete JSON object from a model response.

        Args:
            response:
                Raw response returned by the language model.

        Returns:
            Parsed JSON object.

        Raises:
            ValueError: If no valid JSON object can be extracted.
        """

        cleaned_response = cls._clean_response(response)
        decoding_errors: list[json.JSONDecodeError] = []
        found_incomplete_object = False

        for json_start, character in enumerate(cleaned_response):
            if character != "{":
                continue

            json_end = cls._find_json_object_end(
                cleaned_response,
                json_start,
            )

            if json_end is None:
                found_incomplete_object = True
                continue

            json_candidate = cleaned_response[
                json_start:json_end + 1
            ]

            try:
                parsed_response = json.loads(json_candidate)
            except json.JSONDecodeError as error:
                decoding_errors.append(error)
                continue

            if isinstance(parsed_response, dict):
                return parsed_response

        raise ValueError(
            cls._build_parse_error(
                response=response,
                decoding_errors=decoding_errors,
                found_incomplete_object=found_incomplete_object,
            )
        )

    @staticmethod
    def _find_json_object_end(
        response: str,
        json_start: int,
    ) -> int | None:
        """
        Find the closing brace for a JSON object while respecting strings.

        Args:
            response:
                Cleaned model response.

            json_start:
                Index of the opening JSON object brace.

        Returns:
            Index of the matching closing brace, or ``None`` if incomplete.
        """

        brace_depth = 0
        in_string = False
        escaped_character = False

        for index in range(json_start, len(response)):
            character = response[index]

            if in_string:
                if escaped_character:
                    escaped_character = False
                elif character == "\\":
                    escaped_character = True
                elif character == '"':
                    in_string = False
                continue

            if character == '"':
                in_string = True
            elif character == "{":
                brace_depth += 1
            elif character == "}":
                brace_depth -= 1

                if brace_depth == 0:
                    return index

        return None

    @staticmethod
    def _validate(data: dict[str, object]) -> None:
        """
        Validate the top-level structure of parsed food data.

        Args:
            data:
                Parsed JSON object to validate.

        Raises:
            ValueError: If the structure is invalid.
        """

        if "ingredients" not in data:
            raise ValueError(
                "Missing 'ingredients' key in parsed response."
            )

        if not isinstance(data["ingredients"], list):
            raise ValueError(
                "'ingredients' must be a list."
            )

        for i, ingredient in enumerate(data["ingredients"]):
            if not isinstance(ingredient, dict):
                raise ValueError(
                    f"Ingredient at index {i} must be an object."
                )

            if "name" not in ingredient:
                raise ValueError(
                    f"Ingredient at index {i} is missing 'name'."
                )

            if "children" in ingredient:
                raise ValueError(
                    f"Ingredient at index {i} contains 'children'. "
                    "Ingredients must be a flat list."
                )

            if "parent" in ingredient:
                raise ValueError(
                    f"Ingredient at index {i} contains 'parent'. "
                    "Ingredients must be a flat list."
                )

            if "sub_ingredients" in ingredient:
                raise ValueError(
                    f"Ingredient at index {i} contains 'sub_ingredients'. "
                    "Ingredients must be a flat list."
                )

        if "nutrition" in data:
            nutrition = data["nutrition"]

            if not isinstance(nutrition, dict):
                raise ValueError(
                    "'nutrition' must be an object."
                )

            if "serving_size" in nutrition:
                serving_size = nutrition["serving_size"]

                if not isinstance(serving_size, dict):
                    raise ValueError(
                        "'serving_size' must be an object."
                    )

                if "value" not in serving_size:
                    raise ValueError(
                        "'serving_size' is missing 'value'."
                    )

                if "unit" not in serving_size:
                    raise ValueError(
                        "'serving_size' is missing 'unit'."
                    )

                value = serving_size["value"]
                if value is not None and not isinstance(value, (int, float)):
                    raise ValueError(
                        "'serving_size' 'value' must be a number or null."
                    )

                unit = serving_size["unit"]
                if unit is not None and not isinstance(unit, str):
                    raise ValueError(
                        "'serving_size' 'unit' must be a string or null."
                    )

            for key, section in nutrition.items():
                if key == "serving_size":
                    continue

                if not isinstance(section, dict):
                    raise ValueError(
                        f"Nutrition section '{key}' must be an object."
                    )

                for nutrient_name, nutrient_value in section.items():
                    if not isinstance(nutrient_value, dict):
                        raise ValueError(
                            f"Nutrient '{nutrient_name}' in '{key}' "
                            "must be an object."
                        )

                    if "value" not in nutrient_value:
                        raise ValueError(
                            f"Nutrient '{nutrient_name}' in '{key}' "
                            "is missing 'value'."
                        )

                    if "unit" not in nutrient_value:
                        raise ValueError(
                            f"Nutrient '{nutrient_name}' in '{key}' "
                            "is missing 'unit'."
                        )

                    value = nutrient_value["value"]
                    if value is not None and not isinstance(value, (int, float)):
                        raise ValueError(
                            f"Nutrient '{nutrient_name}' in '{key}' "
                            "'value' must be a number or null."
                        )

                    unit = nutrient_value["unit"]
                    if unit is not None and not isinstance(unit, str):
                        raise ValueError(
                            f"Nutrient '{nutrient_name}' in '{key}' "
                            "'unit' must be a string or null."
                        )

    @classmethod
    def _build_parse_error(
        cls,
        *,
        response: str,
        decoding_errors: list[json.JSONDecodeError],
        found_incomplete_object: bool,
    ) -> str:
        """
        Build a descriptive error message for an invalid model response.

        Args:
            response:
                Raw response returned by the language model.

            decoding_errors:
                JSON decoding errors from complete object candidates.

            found_incomplete_object:
                Whether an object was missing its closing brace.

        Returns:
            Error message suitable for raising as a ``ValueError``.
        """

        response_preview = cls._response_preview(response)

        if found_incomplete_object:
            return (
                "Unable to parse the food label model response as JSON: "
                "the response contains an incomplete JSON object, which may "
                "indicate truncated model output. "
                f"Response preview: {response_preview}"
            )

        if decoding_errors:
            last_error = decoding_errors[-1]

            return (
                "Unable to parse the food label model response as JSON: "
                f"malformed JSON ({last_error.msg} at character "
                f"{last_error.pos}). Response preview: {response_preview}"
            )

        return (
            "Unable to parse the food label model response as JSON: "
            "no JSON object was found. "
            f"Response preview: {response_preview}"
        )

    @staticmethod
    def _response_preview(response: str) -> str:
        """
        Create a bounded single-line response preview for error messages.

        Args:
            response:
                Raw response returned by the language model.

        Returns:
            Readable response preview.
        """

        normalized_response = " ".join(response.split())

        if len(normalized_response) <= 400:
            return repr(normalized_response)

        return repr(
            f"{normalized_response[:300]} ... "
            f"{normalized_response[-100:]}"
        )