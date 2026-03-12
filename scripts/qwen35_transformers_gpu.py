from pathlib import Path
import requests
from PIL import Image
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID = "Qwen/Qwen3.5-2B"
IMAGE_URL = "https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3.5/demo/RealWorld/RealWorld-04.png"
IMAGE_PATH = Path("data/qwen35_demo.png")
PROMPT = "Where is this? Describe the scene briefly in Japanese."


def main() -> None:
    IMAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not IMAGE_PATH.exists():
        response = requests.get(IMAGE_URL, timeout=60)
        response.raise_for_status()
        IMAGE_PATH.write_bytes(response.content)

    image = Image.open(IMAGE_PATH).convert("RGB")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print({"model": MODEL_ID, "device": device, "dtype": str(dtype), "image": str(IMAGE_PATH)})

    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=dtype, device_map="auto")
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": PROMPT},
        ],
    }]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)

    prompt_len = inputs["input_ids"].shape[1]
    trimmed = generated_ids[:, prompt_len:]
    output_text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    print("\n=== Output ===\n")
    print(output_text)


if __name__ == "__main__":
    main()
