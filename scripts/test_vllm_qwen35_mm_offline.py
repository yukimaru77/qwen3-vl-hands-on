from pathlib import Path
from PIL import Image
import requests

from transformers import AutoProcessor
from vllm import LLM, SamplingParams

MODEL_ID = "Qwen/Qwen3.5-2B"
IMAGE_URL = "https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3.5/demo/RealWorld/RealWorld-04.png"
IMAGE_PATH = Path("data/qwen35_demo.png")


def main() -> None:
    IMAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not IMAGE_PATH.exists():
        response = requests.get(IMAGE_URL, timeout=60)
        response.raise_for_status()
        IMAGE_PATH.write_bytes(response.content)

    image = Image.open(IMAGE_PATH).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Where is this? Describe the scene briefly."},
            ],
        }
    ]

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    print("=== Prompt Preview ===")
    print(prompt[:500])

    llm = LLM(
        model=MODEL_ID,
        gpu_memory_utilization=0.85,
        max_model_len=8192,
        limit_mm_per_prompt={"image": 1},
    )

    sampling_params = SamplingParams(max_tokens=128, temperature=0.0)
    outputs = llm.generate(
        {
            "prompt": prompt,
            "multi_modal_data": {"image": image},
        },
        sampling_params=sampling_params,
    )

    print("\n=== vLLM Output ===\n")
    print(outputs[0].outputs[0].text)


if __name__ == "__main__":
    main()
