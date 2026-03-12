# qwen3-vl-hands-on

Qwen系VLMを **Hugging Face + Transformers** で実行し、

1. **画像 + テキスト入力の基本推論**
2. **同じ画像を共有prefixとして再利用する KV cache 実装**

を、**ハンズオン形式のJupyter Notebook** と **実行用スクリプト** にまとめたリポジトリです。

> 注意: ユーザー依頼では「Qwen3.5 のVLM」とありましたが、Hugging Faceでの公開名として画像+テキスト入力のVLM系列は `Qwen3-VL-*` です。本リポジトリでは `Qwen/Qwen3-VL-2B-Instruct` を使用しています。

## できること

- `Qwen/Qwen3-VL-2B-Instruct` で画像+テキスト推論を動かす
- `AutoModelForImageTextToText` / `AutoProcessor` の最小構成を理解する
- 共有のマルチモーダルprefixを一度だけprefillし、`past_key_values` を再利用する
- 学習用に読みやすいNotebookとして手順を追える

## 主要ファイル

- `notebooks/01_qwen3_vl_basic_hands_on.ipynb`
  - 画像+テキスト推論の基本ハンズオン
- `notebooks/02_qwen3_vl_kv_cache_hands_on.ipynb`
  - 画像+テキスト入力での KV cache ハンズオン
- `scripts/basic_inference.py`
  - 最小の基本推論スクリプト
- `scripts/kv_cache_demo.py`
  - 共有画像prefixをKV cacheで再利用する実演スクリプト
- `data/demo.jpeg`
  - Qwen公式サンプル画像

## 実行環境

- Python 3.12
- `uv` による依存管理
- `transformers` は GitHub最新版を利用
  - `Qwen3-VL` サポートが新しいため

## セットアップ

```bash
uv sync
```

## 実行方法

### 1) 基本の画像+テキスト推論

```bash
uv run python scripts/basic_inference.py
```

### 2) KV cache デモ

```bash
uv run python scripts/kv_cache_demo.py
```

### 3) Notebook を開く

```bash
uv run jupyter lab
```

開いたら以下を順番に実行してください。

- `notebooks/01_qwen3_vl_basic_hands_on.ipynb`
- `notebooks/02_qwen3_vl_kv_cache_hands_on.ipynb`

## 検証済みコマンド

この作業中に以下を実行して動作確認しました。

```bash
uv run python scripts/basic_inference.py
uv run python scripts/kv_cache_demo.py
uv run jupyter nbconvert --to notebook --execute notebooks/01_qwen3_vl_basic_hands_on.ipynb --output 01_qwen3_vl_basic_hands_on.executed.ipynb
uv run jupyter nbconvert --to notebook --execute notebooks/02_qwen3_vl_kv_cache_hands_on.ipynb --output 02_qwen3_vl_kv_cache_hands_on.executed.ipynb
```

## 追加した主なパッケージ

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

## GPU / CUDA 前提

このコンテナでは `nvidia-smi` は見えていましたが、今回 `torch.cuda.is_available()` は `False` でした。
そのため、**実行確認はCPUで行っています**。

想定としては以下です。

- GPUが使える環境なら `device="cuda"` でより実用的な速度になります
- CPUでも動作確認は可能ですが、特にKV cacheデモのprefillは遅くなります
- 大きいモデルを使う場合はGPU推奨です

## 既知の制約

- 現在の検証は `Qwen/Qwen3-VL-2B-Instruct` ベースです
- より大きい `4B` / `8B` モデルは、GPUメモリやCUDA対応状況に強く依存します
- CPU実行ではNotebook完走に時間がかかります
- Hugging Face未認証アクセスのため、環境によってはダウンロード制限がかかる場合があります
- `KV cache` 実装は「**同じ画像を含む共有prefixを複数質問で再利用する**」ケースに焦点を当てています

## 参考にした主な情報源

- Hugging Face: `Qwen/Qwen3-VL-2B-Instruct`
- Hugging Face: `Qwen/Qwen3-VL-4B-Instruct`
- QwenLM / Qwen3-VL GitHub repository
- Hugging Face Transformers KV cache guide

## 補足

Notebookは説明重視、`scripts/` は再利用重視で作っています。
学習目的ならNotebook、組み込みや自動化の出発点としては `scripts/` を使うのがおすすめです。
