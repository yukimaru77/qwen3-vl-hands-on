from pathlib import Path
import requests
from PIL import Image
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
IMAGE_URL = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
IMAGE_PATH = Path("data/demo.jpeg")
PROMPT = "Describe this image in 5 concise bullet points."


def ensure_demo_image() -> Path:
    IMAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not IMAGE_PATH.exists():
        response = requests.get(IMAGE_URL, timeout=60)
        response.raise_for_status()
        IMAGE_PATH.write_bytes(response.content)
    return IMAGE_PATH


def main() -> None:
    image_path = ensure_demo_image()
    image = Image.open(image_path).convert("RGB")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print({"model": MODEL_ID, "device": device, "dtype": str(dtype), "image": str(image_path)})

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        dtype=dtype,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=128)

    prompt_len = inputs["input_ids"].shape[1]
    trimmed = generated_ids[:, prompt_len:]
    output_text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    print("\n=== Prompt ===\n")
    print(PROMPT)
    print("\n=== Model Output ===\n")
    print(output_text)


if __name__ == "__main__":
    main()
