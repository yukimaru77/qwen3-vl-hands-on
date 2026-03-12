from pathlib import Path
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image
import requests

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
IMAGE_URL = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
IMAGE_PATH = Path("data/demo.jpeg")
IMAGE_PATH.parent.mkdir(parents=True, exist_ok=True)

if not IMAGE_PATH.exists():
    resp = requests.get(IMAGE_URL, timeout=60)
    resp.raise_for_status()
    IMAGE_PATH.write_bytes(resp.content)

image = Image.open(IMAGE_PATH).convert("RGB")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

print({"device": device, "dtype": str(dtype), "cuda": torch.cuda.is_available()})

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(MODEL_ID)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Describe this image in 5 bullet points."},
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
out = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print("\n=== MODEL OUTPUT ===\n")
print(out[0])
