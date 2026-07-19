# CellSAM Local

[English](README.md) | [日本語](README.ja.md)

CellSAM Local は、Meta の SAM 2.1 モデルを ONNX Runtime Web で動かす、
ブラウザベースの細胞画像セグメンテーションツールです。画像解析を
ローカルで完結させることを目的としており、アップロードした画像は
ブラウザ内に留まります。モデルファイルは初回利用時に Hugging Face から
ダウンロードされます。

UI は日本語です。

## 特徴

- 顕微鏡画像や細胞に近い画像をブラウザ内でセグメンテーション
- ONNX Runtime Web による SAM 2.1 自動マスク生成
- WebGPU による高速化と WASM へのフォールバック
- ドラッグアンドドロップによる画像読み込み
- `assets/sample.png` のサンプル画像を同梱
- プロンプトグリッド密度、IoU しきい値、マスク面積フィルタ、輪郭線幅を調整可能
- 検出後に明度、面積、信頼度で再推論なしにフィルタリング
- 出力キャンバス上のマスクをクリックして個別に除外、復元
- 信頼度、面積、円形度、明度、メモを含むオブジェクト一覧
- ビルド不要の静的フロントエンド

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

## モデル

このアプリは、[SharpAI が Hugging Face で公開している](https://huggingface.co/SharpAI)
ONNX 版 SAM 2.1 モデルを使用します。

| UI 表示 | Hugging Face モデル | 目安サイズ | 備考 |
| --- | --- | ---: | --- |
| SAM2.1-Tiny | `SharpAI/sam2-hiera-tiny-onnx` | 155 MB | デフォルト。最も高速 |
| SAM2.1-Small | `SharpAI/sam2-hiera-small-onnx` | 184 MB | バランス型 |
| SAM2.1-Base+ | `SharpAI/sam2-hiera-base-plus-onnx` | 667 MB | 高精度。GPU 推奨 |
| SAM2.1-Large | `SharpAI/sam2-hiera-large-onnx` | 910 MB | 最高精度。強力な GPU 推奨 |

WebGPU が利用できる場合、エンコーダには事前最適化済みの `.ort` モデルを使用します。
それ以外の場合は WASM バックエンドで `.onnx` モデルを使用します。

## 動作要件

- ES modules に対応したモダンブラウザ
- WebGPU を使う場合は Chrome または Edge を推奨
- 初回モデルダウンロードのためのネットワーク接続
- ローカル HTTP サーバー
- テスト実行用の Node.js

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

## 基本的な使い方

1. ブラウザでアプリを開きます。
2. デフォルトのサンプル画像を使うか、入力エリアをクリックまたはドラッグして画像を読み込みます。
3. SAM 2.1 モデルを選択します。
4. 必要に応じて検出設定を調整します。
5. `検出実行` をクリックしてセグメンテーションを実行します。
6. フィルタスライダーで表示するオブジェクトを絞り込みます。
7. マスクまたは一覧行をクリックして、個別の検出結果を除外、復元します。

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
派生指標、フィルタリング、クリックによる除外と復元の挙動を対象にしています。

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
│   └── visualize.js        # Canvas 描画ヘルパー
├── assets/
│   └── sample.png          # デフォルトサンプル画像
├── test/
│   └── detection.test.js   # Node 単体テスト
└── package.json            # スクリプト定義
```

## プライバシー

画像ファイルはブラウザ内で読み込まれ、このプロジェクトによってアップロードされることは
ありません。ただし、ブラウザは ONNX Runtime Web のアセットを jsDelivr から、
SAM 2.1 モデルファイルを Hugging Face からダウンロードします。

## クレジット

- Meta AI による [SAM 2.1](https://github.com/facebookresearch/sam2)
- [SharpAI](https://huggingface.co/SharpAI) による ONNX モデル変換
- [ONNX Runtime Web](https://onnxruntime.ai/docs/get-started/with-javascript/web.html)
