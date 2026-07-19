# CellSAM Local

[English](README.md) | [日本語](README.ja.md)

CellSAM Local は、Meta の SAM 2.1 モデルを ONNX Runtime Web で動かす、
ブラウザベースの細胞画像セグメンテーションツールです。画像解析を
ローカルで完結させることを目的としており、アップロードした画像は
ブラウザ内に留まります。モデルファイルは初回利用時に Hugging Face から
ダウンロードされます。

低スペックなクライアント PC を LAN 内で使う場合は、同じ UI をサーバモードでも
実行できます。この場合、ブラウザは GPU サーバから UI を開き、セグメンテーションは
同一オリジンの `/api/segment` エンドポイント経由でサーバ側の GPU で実行します。

UI は日本語です。

## 特徴

- 顕微鏡画像や細胞に近い画像をブラウザ内でセグメンテーション
- ONNX Runtime Web による SAM 2.1 自動マスク生成
- WebGPU による高速化と WASM へのフォールバック
- LAN 内 GPU サーバで推論するサーバモード
- ドラッグアンドドロップによる画像読み込み
- `assets/sample.png` のサンプル画像を同梱
- プロンプトグリッド密度、IoU しきい値、マスク面積フィルタ、輪郭線幅を調整可能
- 検出後に明度、面積、信頼度で再推論なしにフィルタリング
- 出力キャンバス上のマスクをクリックして個別に除外、復元
- 信頼度、面積、円形度、明度、メモを含むオブジェクト一覧
- ビルド不要の静的フロントエンド

## 推論モード

| モード | コマンド | 推論の実行場所 | 画像の扱い |
| --- | --- | --- | --- |
| ローカルブラウザモード | `npm run serve` | ユーザーのブラウザ。WebGPU または WASM | 画像はブラウザ内に留まります |
| LAN GPU サーバモード | `npm run serve:gpu` | GPU サーバの `/api/segment` | 画像は LAN 内サーバへ送信されます |

フロントエンドは `/api/health` を自動確認します。同一オリジンに利用可能な
サーババックエンドがあればサーバ推論を使い、なければ従来通りブラウザ内推論に
フォールバックします。

## 仕組み

このアプリは、SAM 2 の自動マスク生成の主要な処理を JavaScript に移植しています。

1. 入力画像を SAM 2 の入力サイズである `1024x1024` にリサイズ、パディングします。
2. 選択した SAM 2 エンコーダを 1 回実行します。
3. `pointsPerSide x pointsPerSide` のプロンプト点グリッドを生成します。
4. 各プロンプト点に対してマスク候補を 1 つデコードします。
5. デコーダの IoU スコアと面積でマスクをフィルタリングします。
6. mask-IoU ベースの non-maximum suppression で重複マスクを取り除きます。
7. 面積、平均明度、周囲長、円形度などの派生指標を計算します。
8. マスク輪郭とオブジェクト一覧をブラウザ上に描画します。

サーバモードでは、手順 1-4 を Python サーバ上で実行します。CUDA が利用できる場合は
ONNX Runtime GPU を使います。サーバは RLE 圧縮した生マスクを返し、後段フィルタ、
描画、クリック除外、メモ、一覧表示は引き続きブラウザ側で処理します。

## モデル

このアプリは、[SharpAI が Hugging Face で公開している](https://huggingface.co/SharpAI)
ONNX 版 SAM 2.1 モデルを使用します。

| UI 表示 | Hugging Face モデル | 目安サイズ | 備考 |
| --- | --- | ---: | --- |
| SAM2.1-Tiny | `SharpAI/sam2-hiera-tiny-onnx` | 155 MB | デフォルト。最も高速 |
| SAM2.1-Small | `SharpAI/sam2-hiera-small-onnx` | 184 MB | バランス型 |
| SAM2.1-Base+ | `SharpAI/sam2-hiera-base-plus-onnx` | 667 MB | 高精度。GPU 推奨 |
| SAM2.1-Large | `SharpAI/sam2-hiera-large-onnx` | 910 MB | 最高精度。強力な GPU 推奨 |

実験的なサーバ側モデル:

| モデルキー | 出典 | 目安サイズ | ライセンス注記 | 状態 |
| --- | --- | ---: | --- | --- |
| `mobile-sam` | [`Heliosoph/sam-onnx`](https://huggingface.co/Heliosoph/sam-onnx) | 43 MB | モデルカードでは Apache-2.0 と記載されています。MobileSAM の ViT-T エンコーダと SAM マスクデコーダの ONNX バンドルです。 | サーバ側ベンチマーク/API 候補。ブラウザ UI にはまだ表示していません |

採用済みモデルのライセンス注記:

- このプロジェクトで使う SharpAI の ONNX モデルカードでは、変換モデルの
  ライセンスは Apache-2.0 と記載されています。
- 実験的に追加した MobileSAM ONNX バンドルも、Hugging Face のモデルカードでは
  Apache-2.0 と記載されています。
- これらは Meta SAM 2.1 モデルからの変換版なので、利用目的に応じて上流の
  [SAM 2 リポジトリ](https://github.com/facebookresearch/sam2) とモデル利用条件も
  確認してください。
- 研究用途で利用可能な候補モデルはローカル検証の対象に含めます。ただし、候補を
  UI またはサーバ API から使える状態にする前に、出典、ライセンス、推論モード、
  制約をこの README と `README.md` に明記します。

候補モデルとライセンスの整理は
[docs/model_candidates.md](docs/model_candidates.md) にまとめています。

WebGPU が利用できる場合、エンコーダには事前最適化済みの `.ort` モデルを使用します。
それ以外の場合は WASM バックエンドで `.onnx` モデルを使用します。

## 動作要件

ローカルブラウザモード:

- ES modules に対応したモダンブラウザ
- WebGPU を使う場合は Chrome または Edge を推奨
- 初回モデルダウンロードのためのネットワーク接続
- ローカル HTTP サーバー
- テスト実行用の Node.js

LAN GPU サーバモード:

- クライアント PC から LAN 内で到達できる GPU サーバ
- Python 3.10+
- ローカル Python 環境の作成と同期に使う `uv`
- GPU 高速化には NVIDIA CUDA 対応の ONNX Runtime 環境
- `server/requirements.txt` の依存パッケージ
- 初回モデルダウンロードのための、サーバ側のネットワーク接続

このアプリは ES modules を使うため、`file://` で `index.html` を直接開く方法では
安定して動作しません。HTTP サーバー経由で配信してください。

## ローカル実行

プロジェクトルートで次を実行します。

```bash
npm run serve
```

ブラウザで次を開きます。

```text
http://localhost:8080
```

アプリ本体の利用にパッケージインストールは不要です。実行時ライブラリは CDN から
読み込まれ、モデル重みはモデルを初めて選択したタイミングでブラウザが
ダウンロードします。

## LAN GPU サーバで実行

GPU サーバ上で次を実行します。

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r server/requirements.txt
npm run serve:gpu
```

クライアント PC のブラウザで次を開きます。

```text
http://<gpu-server-hostname-or-ip>:8080
```

サーバは UI と API を同じオリジンから提供するため、クライアント PC 側で API
エンドポイントを入力する必要はありません。モデルを初めて使うリクエストでは、
ONNX ファイルがサーバ上の `~/.cache/cellsam-local/models` にダウンロードされます。
必要に応じて `CELLSAM_MODEL_CACHE` で保存先を変更できます。

サーバモードでは次の API を提供します。

| エンドポイント | 用途 |
| --- | --- |
| `GET /api/health` | サーバモード、依存関係の準備状態、推論 provider、利用可能モデルを返します |
| `POST /api/segment` | 画像、モデル名、`points_per_side` を受け取り、RLE 形式の生マスクを返します |

## 基本的な使い方

1. ブラウザでアプリを開きます。
2. デフォルトのサンプル画像を使うか、入力エリアをクリックまたはドラッグして画像を読み込みます。
3. SAM 2.1 モデルを選択します。
4. 必要に応じて検出設定を調整します。
5. `検出実行` をクリックしてセグメンテーションを実行します。
6. フィルタスライダーで表示するオブジェクトを絞り込みます。
7. マスクまたは一覧行をクリックして、個別の検出結果を除外、復元します。

## 低スペック PC 向けの推奨設定

強力な GPU がない PC では、最小モデルと粗いプロンプトグリッドから試してください。

| 設定 | 推奨 |
| --- | --- |
| モデル | `SAM2.1-Tiny` |
| `points/side` | `8` |
| IoU しきい値 | まずはデフォルトのまま使い、検出数が少なすぎる場合だけ下げます |
| 最小マスク面積 | 小さな誤検出が多い場合は値を上げます |
| 大きいモデル | WebGPU が利用でき、メモリに余裕がある場合以外は `Base+` と `Large` を避けます |

実行時間には `points/side` が大きく影響します。これはグリッド点ごとにデコーダを
1 回実行するためです。たとえば `8` は 64 回、`16` は 256 回のデコードになります。
WASM/CPU で動かす場合、この差はかなり大きくなります。

## 検出設定

| 設定 | 説明 |
| --- | --- |
| `points/side` | プロンプト点のグリッド密度。値を大きくすると検出数が増える可能性がありますが、実行時間も長くなります。 |
| IoU しきい値 | SAM デコーダ信頼度スコアの下限です。 |
| 最小マスク面積 | 指定ピクセル面積より小さいマスクを除外します。 |
| 最大マスク比 | 画像全体に対して大きすぎるマスクを除外します。 |
| 輪郭線幅 | 出力キャンバスに描画する輪郭線の太さです。 |

推論後、アプリは生マスクをメモリ上に保持します。そのため IoU、面積、明度、信頼度の
フィルタは、各プロンプト点のデコードをやり直さずに調整できます。

## テスト

単体テストを実行します。

```bash
npm test
```

ウォッチモード:

```bash
npm run test:watch
```

テストは Node.js 組み込みの `node:test` ランナーを使い、検出データモデル、
派生指標、フィルタリング、クリックによる除外と復元、サーバ応答の RLE マスク復元を
対象にしています。

ローカルの検証画像でサーバ側推論を計測します。

```bash
npm run benchmark:server -- --limit 2
npm run benchmark:server -- --models tiny mobile-sam --limit 2
```

ベンチマークは `assets/validation/` の画像を読み、サーバ用セグメンターを直接実行し、
JSON レポートを `reports/` に書き出します。どちらのディレクトリも git では無視されるため、
検証用入力と出力をコミット対象から分けて扱えます。

## プロジェクト構成

```text
.
├── index.html              # 日本語 UI レイアウトと ONNX Runtime Web 読み込み
├── css/
│   └── style.css           # アプリケーションスタイル
├── js/
│   ├── app.js              # UI 状態、イベント、モデル読み込み、描画フロー
│   ├── automask.js         # グリッドプロンプト、マスクデコード、後処理、NMS
│   ├── detection.js        # DetectionResult/DetectedObject と指標計算
│   ├── sam2.js             # SAM 2.1 ONNX Runtime Web ラッパー
│   ├── server_api.js       # 同一オリジンのサーバ推論クライアント
│   └── visualize.js        # Canvas 描画ヘルパー
├── docs/
│   └── model_candidates.md # 候補モデルとライセンス注記
├── server/
│   ├── app.py              # LAN 配信用 FastAPI UI/API サーバ
│   ├── run_gpu_python.sh   # GPU 用 Python 環境の共通起動ヘルパー
│   ├── run_gpu_server.sh   # LAN サーバ起動スクリプト
│   ├── segmenter.py        # サーバ側 SAM 2.1 ONNX Runtime 推論
│   └── requirements.txt    # GPU サーバ用 Python 依存パッケージ
├── scripts/
│   └── benchmark_models.py # サーバモデルのベンチマーク
├── assets/
│   └── sample.png          # デフォルトサンプル画像
├── test/
│   ├── detection.test.js   # 検出データモデルのテスト
│   └── server_api.test.js  # サーバ応答デコードのテスト
└── package.json            # スクリプト定義
```

## プライバシー

ローカルブラウザモードでは、画像ファイルはブラウザ内で読み込まれ、このプロジェクトに
よってアップロードされることはありません。ただし、ブラウザは ONNX Runtime Web の
アセットを jsDelivr から、SAM 2.1 モデルファイルを Hugging Face からダウンロードします。

LAN GPU サーバモードでは、画像はこのプロジェクトを実行しているサーバへ送信されます。
信頼できる LAN 内で使う想定です。LAN 外へ公開する場合は、認証やアップロード制限を
追加してください。

## クレジット

- Meta AI による [SAM 2.1](https://github.com/facebookresearch/sam2)
- [SharpAI](https://huggingface.co/SharpAI) による ONNX モデル変換
- [ONNX Runtime Web](https://onnxruntime.ai/docs/get-started/with-javascript/web.html)
