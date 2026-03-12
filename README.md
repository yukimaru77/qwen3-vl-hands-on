# qwen3.5-vllm-hands-on

Qwen3.5 の **マルチモーダル推論（画像 + テキスト）** を、**vLLM + GPU** で実行し、さらに **Automatic Prefix Caching (APC)** による共有prefix再利用まで確認できるハンズオンリポジトリです。

このリポジトリでは、以下をまとめています。

1. **Qwen3.5 を vLLM 上で GPU 推論する方法**
2. **画像 + テキスト入力の最小実行例**
3. **画像を含む共通prefixに対する APC / KV cache 再利用の確認**
4. **Jupyter Notebook での分かりやすいハンズオン化**

## 重要な前提

今回対象としているのは **`Qwen/Qwen3.5-2B`** です。

> ポイント: `Qwen3.5` は `-VL` が付いていなくても、**そのままマルチモーダルモデル** です。

そのため、画像 + テキスト入力では `Qwen3-VL` ではなく、**`Qwen3.5-*` 自体** を使っています。

---

## 完成物

### Notebooks

- `notebooks/01_qwen35_vllm_basic_hands_on.ipynb`
  - vLLM + GPU で `Qwen/Qwen3.5-2B` を使い、画像 + テキスト推論を行う基本編
  - **実際に動いた Python コードをセルに分解した Notebook**
- `notebooks/02_qwen35_vllm_prefix_caching_hands_on.ipynb`
  - vLLM の `enable_prefix_caching=True` を使い、画像を含む共有prefixの再利用を確認する編
  - **`.py` を呼ぶだけではなく、APC 検証コード本体をセル化した Notebook**

### 実行スクリプト

- `scripts/test_vllm_qwen35_mm_offline.py`
  - vLLM の offline inference で Qwen3.5 マルチモーダル推論を行う最小例
- `scripts/test_vllm_qwen35_prefix_cache.py`
  - APC を有効化して、同じ画像つきprefixの再利用を確認するスクリプト
- `scripts/run_vllm_qwen35_server.sh`
  - OpenAI-compatible server として vLLM を起動するためのスクリプト
- `scripts/qwen35_vllm_openai_client.py`
  - vLLM OpenAI-compatible endpoint に画像 + テキストを投げるクライアント例
- `scripts/qwen35_transformers_gpu.py`
  - 比較用の Transformers + GPU 実行例

---

## 動作確認済み内容

この作業では、以下を **実際に確認済み** です。

### 1. Qwen3.5 はマルチモーダル

- `Qwen/Qwen3.5-2B` を使用
- `AutoProcessor` に画像を渡せる
- 画像 + テキスト入力の推論が可能

### 2. GPU で実行可能

- `torch.cuda.is_available() == True`
- GPU: `NVIDIA GB10`
- CUDA: `13.0`
- `Qwen/Qwen3.5-2B` を **GPU 上で実行** 済み

### 3. vLLM 上でも Qwen3.5 マルチモーダル推論を実行

- arm64 + CUDA 13 + GB10 環境で、vLLM を **source build** を含めて調整
- `Qwen/Qwen3.5-2B` の **画像 + テキスト入力**を vLLM + GPU で実行

### 4. Prefix Caching / KV cache 再利用の効果を確認

`enable_prefix_caching=True` で、同じ画像を含む共通prefixを持つ2リクエストを連続実行した結果:

- 1回目: **47.6 秒**
- 2回目: **2.63 秒**

つまり、**画像を含む共有prefixでも vLLM の APC による再利用が効く** ことを確認しています。

---

## 環境

- OS: Linux (arm64)
- Python: 3.12
- GPU: NVIDIA GB10
- CUDA: 13.0
- パッケージ管理: `uv`
- コンテナのベースイメージ系統: **NVIDIA CUDA 13.0 系 / Ubuntu 24.04 系**

今回こちらで確認できた情報:
- `PRETTY_NAME="Ubuntu 24.04.3 LTS"`
- `CUDA_VERSION=13.0.1`
- `NVIDIA_PRODUCT_NAME=CUDA`

> 正確な元イメージ名タグまではコンテナ内から断定できませんでしたが、少なくとも **NVIDIA 検証済みの CUDA 13.0 + Ubuntu 24.04 系ベース** で動いています。

### Python 環境の使い分け

このリポジトリでは2つのPython環境を使っています。

- `.venv`
  - Notebook生成や補助用途
- `.venv-vllm`
  - **vLLM + CUDA + Qwen3.5 実行専用**

vLLM 関連は **`.venv-vllm` を使う前提** です。

---

## セットアップ

### 通常環境

```bash
uv sync
```

### vLLM 実行環境

この作業では arm64 + CUDA 13 + GB10 の都合で、vLLM はかなり特殊な調整を行いました。
そのため、このリポジトリの再現では **既存の `.venv-vllm` をそのまま使う前提** が一番簡単です。

もし同じ作業を一から再現したい場合は、以下の要素が必要になります。

- CUDA 13 対応の PyTorch nightly
- vLLM nightly / source build
- Python headers
- FlashAttention 系の追加ビルド

この環境依存部分はかなり強いため、READMEでは **利用方法中心** に記載します。

---

## 実行方法

### 1. vLLM offline inference で基本推論

```bash
/project/.venv-vllm/bin/python scripts/test_vllm_qwen35_mm_offline.py
```

### 2. Prefix caching の確認

```bash
/project/.venv-vllm/bin/python scripts/test_vllm_qwen35_prefix_cache.py
```

### 3. OpenAI-compatible server を起動

```bash
bash scripts/run_vllm_qwen35_server.sh
```

別ターミナルからクライアントを実行:

```bash
/project/.venv-vllm/bin/python scripts/qwen35_vllm_openai_client.py
```

### 4. Notebook を開く

```bash
uv run jupyter lab
```

開いたら kernel を **`Python (.venv-vllm qwen35)`** に切り替えて、以下を順に実行してください。

- `notebooks/01_qwen35_vllm_basic_hands_on.ipynb`
- `notebooks/02_qwen35_vllm_prefix_caching_hands_on.ipynb`

---

## Notebook 実行確認

この作業中に以下を実行し、Notebook の自動実行確認も行っています。

```bash
uv run jupyter nbconvert --to notebook --execute notebooks/01_qwen35_vllm_basic_hands_on.ipynb --output 01_qwen35_vllm_basic_hands_on.executed.ipynb
uv run jupyter nbconvert --to notebook --execute notebooks/02_qwen35_vllm_prefix_caching_hands_on.ipynb --output 02_qwen35_vllm_prefix_caching_hands_on.executed.ipynb
```

---

## vLLM における KV cache の考え方

Transformers では `past_key_values` を手で持ち回すことが多いですが、vLLM では通常、**Automatic Prefix Caching (APC)** を使うのが自然です。

今回の構成では:

- 同じ画像
- 同じ chat template の先頭部分
- 少しだけ異なる後続の質問

という構成にしており、vLLM が共有prefixを自動検出して再利用します。

つまり実務上は、**vLLM で Qwen3.5 マルチモーダルの KV cache を使いたいなら、まず APC を使う** のが第一選択です。

---

## 追加した主なパッケージ

通常環境側:

- `torch`
- `torchvision`
- `transformers` (GitHub最新版)
- `accelerate`
- `qwen-vl-utils`
- `pillow`
- `requests`
- `jupyter`
- `ipykernel`
- `matplotlib`
- `sentencepiece`

vLLM 環境側:

- `vllm`
- CUDA 13 対応 PyTorch nightly
- `triton`
- `openai`

---

## GPU / CUDA 前提

今回の環境では:

- `nvidia-smi` は正常
- `torch.cuda.is_available()` は最終的に `True`
- Qwen3.5 を GPU 上で動作確認済み

ただし、**GB10 + CUDA 13 + arm64** は現時点でかなり新しい組み合わせであり、既製wheelだけでは整わず、vLLM は source build ベースでの調整が必要でした。

---

## 既知の制約

- vLLM 初期化や実行中に **FlashAttention 系の警告 / FATAL 文字列** が大量に出ることがあります
- それでも今回の環境では、**実際の生成と APC の速度改善は確認済み** です
- vLLM 環境はかなり環境依存です
- Notebook の executed 版はサイズが大きくなります

---

## 参考にした主な情報源

- Hugging Face: `Qwen/Qwen3.5-2B`
- Hugging Face Transformers docs / `qwen3_5`
- vLLM multimodal input docs
- vLLM automatic prefix caching docs
- vLLM Qwen3.5 recipes

---

## まとめ

このリポジトリで分かることはシンプルです。

- **Qwen3.5 は `-VL` なしでマルチモーダル**
- **vLLM + GPU で Qwen3.5 の画像 + テキスト推論ができる**
- **画像を含む共有prefixでも APC が効き、後続リクエストが大きく高速化する**

学習用には Notebook、実運用の出発点としては `scripts/` を使うのがおすすめです。
