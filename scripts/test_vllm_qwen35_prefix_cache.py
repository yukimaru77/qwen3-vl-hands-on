from pathlib import Path
from time import perf_counter
from PIL import Image
import requests
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

MODEL_ID = "Qwen/Qwen3.5-2B"
IMAGE_URL = "https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3.5/demo/RealWorld/RealWorld-04.png"
IMAGE_PATH = Path("data/qwen35_demo.png")


def build_prompt(processor, image, question: str):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ],
    }]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return {"prompt": prompt, "multi_modal_data": {"image": image}}


def main() -> None:
    IMAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not IMAGE_PATH.exists():
        response = requests.get(IMAGE_URL, timeout=60)
        response.raise_for_status()
        IMAGE_PATH.write_bytes(response.content)

    image = Image.open(IMAGE_PATH).convert("RGB")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    llm = LLM(
        model=MODEL_ID,
        gpu_memory_utilization=0.85,
        max_model_len=8192,
        limit_mm_per_prompt={"image": 1},
        enable_prefix_caching=True,
    )
    sp = SamplingParams(max_tokens=64, temperature=0.0)

    req1 = build_prompt(processor, image, "Where is this? Describe the scene briefly in Japanese.")
    t0 = perf_counter()
    out1 = llm.generate(req1, sampling_params=sp)
    t1 = perf_counter() - t0

    req2 = build_prompt(processor, image, "Where is this place? Summarize the visual clues in Japanese.")
    t2s = perf_counter()
    out2 = llm.generate(req2, sampling_params=sp)
    t2 = perf_counter() - t2s

    print({"first_request_seconds": round(t1, 2), "second_request_seconds": round(t2, 2)})
    print("\n=== first ===\n")
    print(out1[0].outputs[0].text)
    print("\n=== second ===\n")
    print(out2[0].outputs[0].text)


if __name__ == "__main__":
    main()
