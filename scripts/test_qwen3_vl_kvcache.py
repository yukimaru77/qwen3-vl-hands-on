from copy import deepcopy
from pathlib import Path
import torch
from PIL import Image
import requests
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
IMAGE_URL = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
IMAGE_PATH = Path("data/demo.jpeg")
if not IMAGE_PATH.exists():
    resp = requests.get(IMAGE_URL, timeout=60)
    resp.raise_for_status()
    IMAGE_PATH.write_bytes(resp.content)

image = Image.open(IMAGE_PATH).convert("RGB")
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32
print({"device": device, "dtype": str(dtype)})

model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=dtype, device_map="auto")
processor = AutoProcessor.from_pretrained(MODEL_ID)

prefix_messages = [
    {"role": "system", "content": [{"type": "text", "text": "You are a precise visual assistant."}]},
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Understand the image carefully. I will ask follow-up questions about the same image."},
        ],
    },
]

prefix_inputs = processor.apply_chat_template(
    prefix_messages,
    tokenize=True,
    add_generation_prompt=False,
    return_dict=True,
    return_tensors="pt",
)
prefix_inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in prefix_inputs.items()}
with torch.inference_mode():
    prefix_outputs = model(**prefix_inputs, use_cache=True, return_dict=True)
base_cache = prefix_outputs.past_key_values
prefix_len = prefix_inputs["input_ids"].shape[1]
print("prefill tokens", prefix_len)

question_messages = prefix_messages + [
    {"role": "user", "content": [{"type": "text", "text": "What is the dog doing?"}]}
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
print("delta tokens", delta_ids.shape[1])

with torch.inference_mode():
    out = model.generate(
        input_ids=delta_ids,
        attention_mask=full_inputs["attention_mask"],
        past_key_values=deepcopy(base_cache),
        max_new_tokens=64,
        do_sample=False,
    )
print(processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=False))
