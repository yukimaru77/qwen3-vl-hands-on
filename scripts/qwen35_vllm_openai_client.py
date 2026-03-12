import base64
from pathlib import Path
from openai import OpenAI

IMAGE_PATH = Path("data/qwen35_demo.png")


def to_data_url(path: Path) -> str:
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def main() -> None:
    client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="dummy")
    image_url = to_data_url(IMAGE_PATH)
    response = client.chat.completions.create(
        model="Qwen/Qwen3.5-2B",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Where is this? Describe the scene briefly in Japanese."},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        max_tokens=128,
        temperature=0.0,
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
