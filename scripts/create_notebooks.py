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
    md("# Qwen3-VL ハンズオン: 画像 + テキスト入力をまず動かす\n\nこのNotebookは、**Hugging Faceで公開されているQwen系VLMをローカルで動かす最小例**を、手を動かしながら理解できる形に整理したものです。\n\n> メモ: ユーザー依頼では `Qwen3.5` と書かれていましたが、**画像+テキスト入力のVLM系列は Hugging Face 上では `Qwen3-VL-*`** として公開されています。通常の `Qwen3.5-*` は主にテキストLLMです。そこで本Notebookでは、VLMとして `Qwen/Qwen3-VL-2B-Instruct` を使います。"),
    md("## 0. 何をやるか\n\nこのNotebookで行うこと:\n1. 実行環境を確認する\n2. サンプル画像を用意する\n3. `AutoModelForImageTextToText` と `AutoProcessor` を読み込む\n4. 画像 + テキストのメッセージを作る\n5. 推論を実行して、出力を読む\n\nまずは**確実に一度動かす**ことを優先しています。"),
    code("from pathlib import Path\nimport requests\nfrom PIL import Image\nimport matplotlib.pyplot as plt\nimport torch\nfrom transformers import AutoModelForImageTextToText, AutoProcessor\n\nMODEL_ID = \"Qwen/Qwen3-VL-2B-Instruct\"\nIMAGE_URL = \"https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg\"\nIMAGE_PATH = Path(\"../data/demo.jpeg\")\nIMAGE_PATH.parent.mkdir(parents=True, exist_ok=True)\n\nprint({\n    \"torch\": torch.__version__,\n    \"cuda_available\": torch.cuda.is_available(),\n    \"model\": MODEL_ID,\n})"),
    md("## 1. サンプル画像を取得して表示\n\n今回はQwen公式サンプル画像を使います。まずは画像を見て、後でモデルの回答が妥当か確認できるようにします。"),
    code("if not IMAGE_PATH.exists():\n    response = requests.get(IMAGE_URL, timeout=60)\n    response.raise_for_status()\n    IMAGE_PATH.write_bytes(response.content)\n\nimage = Image.open(IMAGE_PATH).convert(\"RGB\")\nplt.figure(figsize=(8, 5))\nplt.imshow(image)\nplt.axis(\"off\")\nplt.show()\nprint(IMAGE_PATH)"),
    md("## 2. モデルとProcessorをロード\n\n- `AutoModelForImageTextToText`: 画像+テキストを受けるVLM本体\n- `AutoProcessor`: テキスト整形、画像前処理、chat template適用をまとめて担当\n\nGPUが使える場合は `bfloat16`、そうでなければCPU向けに `float32` を使います。"),
    code("device = \"cuda\" if torch.cuda.is_available() else \"cpu\"\ndtype = torch.bfloat16 if device == \"cuda\" else torch.float32\n\nmodel = AutoModelForImageTextToText.from_pretrained(\n    MODEL_ID,\n    dtype=dtype,\n    device_map=\"auto\",\n)\nprocessor = AutoProcessor.from_pretrained(MODEL_ID)\n\nprint({\"device\": device, \"dtype\": str(dtype)})"),
    md("## 3. 画像 + テキストの入力を作る\n\nQwen3-VLは chat template で入力を作るのが分かりやすいです。\n`content` の中に `image` と `text` を並べると、**マルチモーダル入力**になります。"),
    code("messages = [\n    {\n        \"role\": \"user\",\n        \"content\": [\n            {\"type\": \"image\", \"image\": image},\n            {\"type\": \"text\", \"text\": \"Describe this image in 5 concise bullet points.\"},\n        ],\n    }\n]\n\ninputs = processor.apply_chat_template(\n    messages,\n    tokenize=True,\n    add_generation_prompt=True,\n    return_dict=True,\n    return_tensors=\"pt\",\n)\ninputs = {k: v.to(model.device) if hasattr(v, \"to\") else v for k, v in inputs.items()}\n\n{k: tuple(v.shape) if hasattr(v, \"shape\") else type(v) for k, v in inputs.items()}"),
    md("## 4. 推論\n\n`generate()` で応答を作ります。\n出力には**入力トークンも含まれる**ので、後ろの新規生成部分だけを切り出して decode します。"),
    code("with torch.inference_mode():\n    generated_ids = model.generate(**inputs, max_new_tokens=128)\n\nprompt_len = inputs[\"input_ids\"].shape[1]\ntrimmed = generated_ids[:, prompt_len:]\noutput_text = processor.batch_decode(\n    trimmed,\n    skip_special_tokens=True,\n    clean_up_tokenization_spaces=False,\n)[0]\n\nprint(output_text)"),
    md("## 5. ここまでの理解\n\n最低限押さえるポイント:\n\n- **モデル本体**: `AutoModelForImageTextToText`\n- **前処理**: `AutoProcessor`\n- **マルチモーダル入力**: `content` に `image` と `text` を混在させる\n- **推論**: `processor.apply_chat_template(...)` → `model.generate(...)`\n\nこの形がまずの基本です。ここから、複数画像、動画、解像度制御、量子化、KV cache などに発展できます。"),
]

kvcache = nbf.v4.new_notebook()
kvcache.cells = [
    md("# Qwen3-VL ハンズオン: 画像 + テキスト入力で KV cache を使う\n\nこのNotebookでは、**同じ画像を何度も見せずに、共有のマルチモーダルprefixを一度だけ前計算し、そのKV cacheを再利用する**流れを確認します。\n\n典型ユースケース:\n- 同じ画像に対して複数の質問をしたい\n- 最初の重いprefillを再利用して後続質問を軽くしたい\n- 画像+テキストの共通コンテキストを何度も使いたい"),
    md("## 0. KV cache の考え方\n\n自己回帰生成では、過去トークンの attention 用 Key/Value を毎回計算すると無駄が出ます。\nそこで一度計算したものを `past_key_values` として保持し、次のデコードで再利用します。\n\nQwen3-VLでも基本は同じです。違うのは、**共有prefixの中に画像トークンが含まれる**ことです。\n今回は次の手順で進めます:\n\n1. 共有prefix = `system + image + 説明文` を作る\n2. そのprefixを `use_cache=True` で一度forwardする\n3. 得られた `past_key_values` を保存する\n4. 各質問ごとに、**prefix以降の差分トークンだけ**を流して generate する"),
    code("from copy import deepcopy\nfrom pathlib import Path\nfrom time import perf_counter\n\nimport requests\nfrom PIL import Image\nimport matplotlib.pyplot as plt\nimport torch\nfrom transformers import AutoModelForImageTextToText, AutoProcessor\n\nMODEL_ID = \"Qwen/Qwen3-VL-2B-Instruct\"\nIMAGE_URL = \"https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg\"\nIMAGE_PATH = Path(\"../data/demo.jpeg\")\nIMAGE_PATH.parent.mkdir(parents=True, exist_ok=True)\n\nif not IMAGE_PATH.exists():\n    response = requests.get(IMAGE_URL, timeout=60)\n    response.raise_for_status()\n    IMAGE_PATH.write_bytes(response.content)\n\nimage = Image.open(IMAGE_PATH).convert(\"RGB\")\nplt.figure(figsize=(8, 5))\nplt.imshow(image)\nplt.axis(\"off\")\nplt.show()\n\ndevice = \"cuda\" if torch.cuda.is_available() else \"cpu\"\ndtype = torch.bfloat16 if device == \"cuda\" else torch.float32\n\nmodel = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=dtype, device_map=\"auto\")\nprocessor = AutoProcessor.from_pretrained(MODEL_ID)\nprint({\"device\": device, \"dtype\": str(dtype)})"),
    md("## 1. 共有prefixを作る\n\nここで画像を含む共通コンテキストを作ります。\n後続の質問はこのprefixを共有します。"),
    code("prefix_messages = [\n    {\n        \"role\": \"system\",\n        \"content\": [{\"type\": \"text\", \"text\": \"You are a precise visual assistant.\"}],\n    },\n    {\n        \"role\": \"user\",\n        \"content\": [\n            {\"type\": \"image\", \"image\": image},\n            {\"type\": \"text\", \"text\": \"Understand the image carefully. I will ask multiple follow-up questions about the same image.\"},\n        ],\n    },\n]\n\nprefix_inputs = processor.apply_chat_template(\n    prefix_messages,\n    tokenize=True,\n    add_generation_prompt=False,\n    return_dict=True,\n    return_tensors=\"pt\",\n)\nprefix_inputs = {k: v.to(model.device) if hasattr(v, \"to\") else v for k, v in prefix_inputs.items()}\n\nprefix_len = prefix_inputs[\"input_ids\"].shape[1]\nprefix_len"),
    md("## 2. 共有prefixを一度だけprefillして KV cache を得る"),
    code("t0 = perf_counter()\nwith torch.inference_mode():\n    prefix_outputs = model(**prefix_inputs, use_cache=True, return_dict=True)\nprefill_seconds = perf_counter() - t0\n\nbase_cache = prefix_outputs.past_key_values\nprint(f\"prefix tokens: {prefix_len}\")\nprint(f\"prefill seconds: {prefill_seconds:.2f}\")\nprint(type(base_cache))"),
    md("## 3. 各質問で差分だけ流す\n\nポイントは、各質問の**完全入力**を一度作ってから、\n`prefix_len` 以降だけを `delta_ids` として切り出すことです。\n\nこれにより、共有画像prefixの再計算を避けられます。"),
    code("questions = [\n    \"What is the dog doing?\",\n    \"What clues suggest the setting is calm and pleasant?\",\n    \"List 3 visible objects or attributes in the scene.\",\n]\n\nresults = []\nfor question in questions:\n    question_messages = prefix_messages + [\n        {\"role\": \"user\", \"content\": [{\"type\": \"text\", \"text\": question}]}\n    ]\n\n    full_inputs = processor.apply_chat_template(\n        question_messages,\n        tokenize=True,\n        add_generation_prompt=True,\n        return_dict=True,\n        return_tensors=\"pt\",\n    )\n    full_inputs = {k: v.to(model.device) if hasattr(v, \"to\") else v for k, v in full_inputs.items()}\n\n    delta_ids = full_inputs[\"input_ids\"][:, prefix_len:]\n\n    t1 = perf_counter()\n    with torch.inference_mode():\n        outputs = model.generate(\n            input_ids=delta_ids,\n            attention_mask=full_inputs[\"attention_mask\"],\n            past_key_values=deepcopy(base_cache),\n            max_new_tokens=96,\n            do_sample=False,\n        )\n    elapsed = perf_counter() - t1\n\n    text = processor.batch_decode(\n        outputs,\n        skip_special_tokens=True,\n        clean_up_tokenization_spaces=False,\n    )[0]\n\n    results.append({\n        \"question\": question,\n        \"delta_tokens\": int(delta_ids.shape[1]),\n        \"seconds\": round(elapsed, 2),\n        \"answer\": text,\n    })\n\nresults"),
    md("## 4. 結果を読みやすく表示"),
    code("for item in results:\n    print('=' * 80)\n    print('Question:', item['question'])\n    print('Delta tokens:', item['delta_tokens'])\n    print('Seconds:', item['seconds'])\n    print(item['answer'])\n    print()"),
    md("## 5. 実務上のポイント\n\n- 同じ画像・同じ共通文脈に対して**複数質問**する時に有効\n- 最初の `prefill` は重いが、後続は **差分トークンだけ**で済む\n- `deepcopy(base_cache)` を使うと、質問ごとに独立した分岐がしやすい\n- 長い文脈では KV cache 自体のメモリ消費も増えるため、必要に応じて\n  - offloaded cache\n  - static cache\n  - quantized cache\n  を検討する価値があります\n\nつまり、**画像を含む共有prefixを再利用したいなら、Qwen3-VLでもKV cacheは普通に有効**です。"),
]

with open(NOTEBOOK_DIR / "01_qwen3_vl_basic_hands_on.ipynb", "w", encoding="utf-8") as f:
    nbf.write(basic, f)

with open(NOTEBOOK_DIR / "02_qwen3_vl_kv_cache_hands_on.ipynb", "w", encoding="utf-8") as f:
    nbf.write(kvcache, f)

print("Created notebooks:")
for path in sorted(NOTEBOOK_DIR.glob("*.ipynb")):
    print("-", path)
