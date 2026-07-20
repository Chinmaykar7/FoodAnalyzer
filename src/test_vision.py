from pathlib import Path
from ollama import chat

# Path to the input image
image_path = Path("images/raw/test.jpg").resolve()

if not image_path.exists():
    raise FileNotFoundError(f"Image not found: {image_path}")

prompt = """
You are an OCR engine specialized in reading packaged food labels.

Your task is to extract ONLY the ingredient list from the image.

Rules:
- Return ONLY the ingredient list.
- Preserve the original spelling.
- Preserve percentages exactly.
- Preserve INS/E-number codes exactly.
- Do NOT describe the image.
- Do NOT summarize.
- Do NOT explain.
- Do NOT identify the product.
- Do NOT include nutrition facts.
- Do NOT include allergen information.
- Do NOT include manufacturing details.
- Do NOT include expiry date.
"""

print(f"Using image: {image_path}")
print("Sending image to Qwen...\n")

response = chat(
    model="qwen3.5:4b",
    think=False,
    messages=[
        {
            "role": "user",
            "content": prompt,
            "images": [str(image_path)],
        }
    ],
    options={
        "temperature": 0,
        "num_predict": 250,
    },
)

print("\n========== Extracted Ingredients ==========\n")
print(response.message.content)