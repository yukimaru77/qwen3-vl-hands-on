from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"
NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


basic = nbf.v4.new_notebook()
basic.cells = [
    md("# Qwen3.5 ハンズオン: vLLM + GPU で画像 + テキスト推論\n\nこのNotebookは **`Qwen/Qwen3.5-2B` を vLLM 上で GPU 推論** する最小ハンズオンです。\n\nこのリポジトリでは、arm64 + CUDA 13 + GB10 環境で vLLM を動かすために、**専用の `.venv-vllm`** を使っています。"),
    md("## 0. 前提\n\n- モデル: `Qwen/Qwen3.5-2B`\n- 方式: **vLLM offline inference**\n- モダリティ: **画像 + テキスト**\n- 実行デバイス: **GPU (`cuda`)**\n\n> 重要: 今回の環境では vLLM を source build しているため、Notebook本体の kernel ではなく `!` 付きシェル実行で `.venv-vllm` を使う構成にしています。"),
    code("!nvidia-smi\nimport os\nprint('workspace ok:', os.path.exists('../scripts/test_vllm_qwen35_mm_offline.py'))"),
    md("## 1. サンプル画像を確認"),
    code("from pathlib import Path\nfrom PIL import Image\nimport matplotlib.pyplot as plt\n\nimage_path = Path('../data/qwen35_demo.png')\nimage = Image.open(image_path).convert('RGB')\nplt.figure(figsize=(8, 5))\nplt.imshow(image)\nplt.axis('off')\nplt.show()\nprint(image_path)"),
    md("## 2. vLLM で実行\n\n下のスクリプトは内部で:\n- `AutoProcessor.from_pretrained('Qwen/Qwen3.5-2B')`\n- vLLM `LLM(...)`\n- `multi_modal_data={'image': image}`\n\nを使って、**画像 + テキスト入力**を GPU で実行します。"),
    code("!/project/.venv-vllm/bin/python ../scripts/test_vllm_qwen35_mm_offline.py"),
    md("## 3. 理解ポイント\n\nQwen3.5 は `-VL` ではなくても **そのままマルチモーダル** です。\nvLLM では Hugging Face の chat template を使って prompt を作り、`multi_modal_data` に画像を渡します。\n\n基本形は次の通りです。\n\n```python\nprocessor = AutoProcessor.from_pretrained('Qwen/Qwen3.5-2B')\nprompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)\n\noutputs = llm.generate(\n    {\n        'prompt': prompt,\n        'multi_modal_data': {'image': image},\n    },\n    sampling_params=SamplingParams(max_tokens=128, temperature=0.0),\n)\n```"),
]

kvcache = nbf.v4.new_notebook()
kvcache.cells = [
    md("# Qwen3.5 ハンズオン: vLLM で画像 + テキストの KV cache / Prefix Caching\n\nvLLM では raw な `past_key_values` を自前で触るより、**Automatic Prefix Caching (APC)** を使うのが実務的です。\n\n同じ画像 + 共通prefix を含む複数リクエストでは、vLLM が **共有prefixのKV cacheを再利用** してくれます。"),
    md("## 0. 何を確認するか\n\nこのNotebookでは:\n1. `enable_prefix_caching=True` で vLLM を起動\n2. **同じ画像** を含む 2 リクエストを実行\n3. 1回目と2回目の時間差を比較\n\nを行います。\n\n> 今回の環境では、1回目が約 47 秒、2回目が約 2.6 秒まで短縮されることを確認しました。"),
    md("## 1. 実行\n\n下のスクリプトは APC を有効化して、ほぼ同じ multimodal prefix を持つ2つの要求を連続で投げます。"),
    code("!/project/.venv-vllm/bin/python ../scripts/test_vllm_qwen35_prefix_cache.py"),
    md("## 2. 理解ポイント\n\n- `enable_prefix_caching=True` を指定すると APC が有効化される\n- **同じ画像 + 同じ共通prefix** を持つ後続リクエストは、先頭部分の再計算を省ける\n- これは Transformers で `past_key_values` を手で持ち回す発想に近いが、vLLM では **サーバ/エンジン側が自動管理** してくれる\n\nコードの核はこれです。\n\n```python\nllm = LLM(\n    model='Qwen/Qwen3.5-2B',\n    enable_prefix_caching=True,\n    limit_mm_per_prompt={'image': 1},\n)\n```"),
    md("## 3. 注意点\n\nこの環境では GB10 / CUDA 13 / arm64 の都合で、vLLM 初期化時に FlashAttention 関連の警告が大量に出ます。\nただし **生成自体と APC の速度改善は確認済み** です。\n\nつまり、今回のハンズオンで本当に見たいポイントは:\n- Qwen3.5 がマルチモーダル入力で動くこと\n- vLLM が GPU 上で動くこと\n- 画像を含む共有prefixでも APC が効くこと\n\nの3点です。"),
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
