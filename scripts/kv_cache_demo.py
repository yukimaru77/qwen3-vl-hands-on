from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from time import perf_counter
import requests
from PIL import Image
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
IMAGE_URL = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
IMAGE_PATH = Path("data/demo.jpeg")

PREFIX_MESSAGES = [
    {"role": "system", "content": [{"type": "text", "text": "You are a precise visual assistant."}]},
    {
        "role": "user",
        "content": [
            {"type": "image", "image": None},
            {
                "type": "text",
                "text": "Understand the image carefully. I will ask multiple follow-up questions about the same image.",
            },
        ],
    },
]

QUESTIONS = [
    "What is the dog doing?",
    "What clues suggest the setting is calm and pleasant?",
    "List 3 visible objects or attributes in the scene.",
]


def ensure_demo_image() -> Path:
    IMAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not IMAGE_PATH.exists():
        response = requests.get(IMAGE_URL, timeout=60)
        response.raise_for_status()
        IMAGE_PATH.write_bytes(response.content)
    return IMAGE_PATH


def decode_new_tokens(processor: AutoProcessor, outputs: torch.Tensor, input_len: int) -> str:
    trimmed = outputs[:, input_len:]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def main() -> None:
    image_path = ensure_demo_image()
    image = Image.open(image_path).convert("RGB")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print({"model": MODEL_ID, "device": device, "dtype": str(dtype), "image": str(image_path)})

    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=dtype, device_map="auto")
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    prefix_messages = deepcopy(PREFIX_MESSAGES)
    prefix_messages[1]["content"][0]["image"] = image

    prefix_inputs = processor.apply_chat_template(
        prefix_messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt",
    )
    prefix_inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in prefix_inputs.items()}

    t0 = perf_counter()
    with torch.inference_mode():
        prefix_outputs = model(**prefix_inputs, use_cache=True, return_dict=True)
    prefill_seconds = perf_counter() - t0
    base_cache = prefix_outputs.past_key_values
    prefix_len = prefix_inputs["input_ids"].shape[1]

    print(f"\nShared multimodal prefix prefetched: {prefix_len} tokens in {prefill_seconds:.2f}s")

    for idx, question in enumerate(QUESTIONS, start=1):
        question_messages = prefix_messages + [
            {"role": "user", "content": [{"type": "text", "text": question}]}
        ]
        full_inputs = processor.apply_chat_template(
            question_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        full_inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in full_inputs.items()}
        delta_ids = full_inputs["input_ids"][:, prefix_len:]

        t1 = perf_counter()
        with torch.inference_mode():
            outputs = model.generate(
                input_ids=delta_ids,
                attention_mask=full_inputs["attention_mask"],
                past_key_values=deepcopy(base_cache),
                max_new_tokens=96,
                do_sample=False,
            )
        elapsed = perf_counter() - t1
        answer = processor.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        print(f"\n--- Question {idx} ---")
        print(question)
        print(f"(decode with reused KV cache: {elapsed:.2f}s)")
        print(answer)


if __name__ == "__main__":
    main()
