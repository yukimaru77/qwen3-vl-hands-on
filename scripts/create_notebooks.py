from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"
NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)

KERNEL_META = {
    "kernelspec": {
        "display_name": "Python (.venv-vllm qwen35)",
        "language": "python",
        "name": "qwen35-vllm",
    },
    "language_info": {
        "name": "python",
        "version": "3.12",
    },
}


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


basic = nbf.v4.new_notebook(metadata=KERNEL_META)
basic.cells = [
    md("# Qwen3.5 ハンズオン: vLLM + GPU で画像 + テキスト推論\n\nこのNotebookでは、**`Qwen/Qwen3.5-2B` を vLLM 上で GPU 推論**します。\n\n前回のNotebookは単に `.py` を叩いていたので、今回は **実際に動いた Python コードをそのままセルに分解** して、Notebookとして読める形に直しています。"),
    md("## 0. まず確認\n\n- モデル: `Qwen/Qwen3.5-2B`\n- 方式: `vLLM` の offline inference\n- 入力: **画像 + テキスト**\n- 実行先: **GPU**\n\nこのNotebookは **`Python (.venv-vllm qwen35)` kernel** で開く前提です。"),
    code("import torch\nprint('torch:', torch.__version__)\nprint('cuda available:', torch.cuda.is_available())\nif torch.cuda.is_available():\n    print('device:', torch.cuda.get_device_name(0))"),
    code("from pathlib import Path\nfrom PIL import Image\nimport matplotlib.pyplot as plt\nimport requests\n\nMODEL_ID = 'Qwen/Qwen3.5-2B'\nIMAGE_URL = 'https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3.5/demo/RealWorld/RealWorld-04.png'\nIMAGE_PATH = Path('../data/qwen35_demo.png')\nIMAGE_PATH.parent.mkdir(parents=True, exist_ok=True)\n\nif not IMAGE_PATH.exists():\n    response = requests.get(IMAGE_URL, timeout=60)\n    response.raise_for_status()\n    IMAGE_PATH.write_bytes(response.content)\n\nimage = Image.open(IMAGE_PATH).convert('RGB')\nplt.figure(figsize=(8, 5))\nplt.imshow(image)\nplt.axis('off')\nplt.show()\nprint(IMAGE_PATH)"),
    md("## 1. 画像 + テキストのメッセージを作る\n\nQwen3.5 は `-VL` なしでもそのままマルチモーダルなので、`AutoProcessor` で chat template を作り、画像とテキストを一緒に渡します。"),
    code("from transformers import AutoProcessor\n\nmessages = [\n    {\n        'role': 'user',\n        'content': [\n            {'type': 'image', 'image': image},\n            {'type': 'text', 'text': 'Where is this? Describe the scene briefly.'},\n        ],\n    }\n]\n\nprocessor = AutoProcessor.from_pretrained(MODEL_ID)\nprompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)\nprint(prompt[:500])"),
    md("## 2. vLLM エンジンを立ち上げる\n\nここが一番大事なセルです。\n`LLM(...)` を作って、`multi_modal_data={'image': image}` を使う準備をします。"),
    code("from vllm import LLM, SamplingParams\n\nllm = LLM(\n    model=MODEL_ID,\n    gpu_memory_utilization=0.85,\n    max_model_len=8192,\n    limit_mm_per_prompt={'image': 1},\n)\n\nsampling_params = SamplingParams(max_tokens=128, temperature=0.0)"),
    md("## 3. 実際に生成する"),
    code("outputs = llm.generate(\n    {\n        'prompt': prompt,\n        'multi_modal_data': {'image': image},\n    },\n    sampling_params=sampling_params,\n)\n\nanswer = outputs[0].outputs[0].text\nprint(answer)"),
    md("## 4. まとめ\n\nこのNotebookでやったこと:\n- `AutoProcessor` で Qwen3.5 用の multimodal prompt を作る\n- `vLLM` の `LLM(...)` を GPU 上で起動する\n- `multi_modal_data` に画像を入れて、**画像 + テキスト推論**を行う\n\nつまり、**Qwen3.5 は `-VL` なしでそのまま VLM として使える**、というのがここでのポイントです。"),
]

kvcache = nbf.v4.new_notebook(metadata=KERNEL_META)
kvcache.cells = [
    md("# Qwen3.5 ハンズオン: vLLM で画像 + テキストの Prefix Caching を確認する\n\nこのNotebookでは、**vLLM の Automatic Prefix Caching (APC)** を使って、画像を含む共有prefixの再利用を確認します。\n\nTransformers で `past_key_values` を手で回す代わりに、vLLM ではまず APC を使うのが自然です。"),
    code("from pathlib import Path\nfrom time import perf_counter\nfrom PIL import Image\nimport requests\nfrom transformers import AutoProcessor\nfrom vllm import LLM, SamplingParams\n\nMODEL_ID = 'Qwen/Qwen3.5-2B'\nIMAGE_URL = 'https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3.5/demo/RealWorld/RealWorld-04.png'\nIMAGE_PATH = Path('../data/qwen35_demo.png')\n\nif not IMAGE_PATH.exists():\n    response = requests.get(IMAGE_URL, timeout=60)\n    response.raise_for_status()\n    IMAGE_PATH.write_bytes(response.content)\n\nimage = Image.open(IMAGE_PATH).convert('RGB')\nprocessor = AutoProcessor.from_pretrained(MODEL_ID)"),
    md("## 1. 共通prefixを持つリクエストを作る\n\n画像は同じ、質問だけ少し変えます。\nこれで vLLM が prefix の共有を検出しやすくなります。"),
    code("def build_prompt(question: str):\n    messages = [\n        {\n            'role': 'user',\n            'content': [\n                {'type': 'image', 'image': image},\n                {'type': 'text', 'text': question},\n            ],\n        }\n    ]\n    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)\n    return {'prompt': prompt, 'multi_modal_data': {'image': image}}\n\nreq1 = build_prompt('Where is this? Describe the scene briefly in Japanese.')\nreq2 = build_prompt('Where is this place? Summarize the visual clues in Japanese.')\nprint(req1['prompt'][:300])"),
    md("## 2. Prefix Caching を有効化した vLLM を起動する"),
    code("llm = LLM(\n    model=MODEL_ID,\n    gpu_memory_utilization=0.85,\n    max_model_len=8192,\n    limit_mm_per_prompt={'image': 1},\n    enable_prefix_caching=True,\n)\n\nsp = SamplingParams(max_tokens=64, temperature=0.0)"),
    md("## 3. 1回目と2回目を比較する\n\n1回目は通常どおり重く、2回目は共有prefixの再利用が効けば大きく短縮されます。"),
    code("t0 = perf_counter()\nout1 = llm.generate(req1, sampling_params=sp)\nt1 = perf_counter() - t0\n\nt2s = perf_counter()\nout2 = llm.generate(req2, sampling_params=sp)\nt2 = perf_counter() - t2s\n\nprint({'first_request_seconds': round(t1, 2), 'second_request_seconds': round(t2, 2)})\nprint('\\n=== first ===\\n')\nprint(out1[0].outputs[0].text)\nprint('\\n=== second ===\\n')\nprint(out2[0].outputs[0].text)"),
    md("## 4. 解説\n\n今回の環境では、実際に\n- 1回目: 約 47.6 秒\n- 2回目: 約 2.6 秒\n\nまで短縮されました。\n\nつまり、**画像を含む共有prefixでも vLLM の APC が効いている** と見てよいです。\n\n実務的には、Qwen3.5 のマルチモーダル推論で KV cache 的な再利用をしたいなら、まずは **`enable_prefix_caching=True`** を検討するのが第一歩です。"),
]

for name, nb in [
    ("01_qwen35_vllm_basic_hands_on.ipynb", basic),
    ("02_qwen35_vllm_prefix_caching_hands_on.ipynb", kvcache),
]:
    with open(NOTEBOOK_DIR / name, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

print("Created notebooks:")
for path in sorted(NOTEBOOK_DIR.glob("*.ipynb")):
    print("-", path.name)
