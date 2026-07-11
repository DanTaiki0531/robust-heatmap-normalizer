# robust-heatmap-normalizer

ヒートマップのセル値をJSONで指定し、セル内の数値は元の値のまま、色だけをロバスト標準化後の値で付け直すツールです。

```text
z = (x - median(X)) / IQR(X)
IQR(X) = Q3 - Q1
```

ロバスト標準化は、各ヒートマップごとに全セルを対象として計算します。対角成分も含みます。標準化後の値は色付けだけに使い、セル内アノテーションにはJSONの元値を表示します。

## ファイル構成

```text
.
├── create_json_from_image.py      # 画像から入力JSONを作る
├── robust_heatmap.py              # JSONからヒートマップを生成する
├── input_images/                  # 元画像置き場
├── input_json/                    # 入力JSON置き場
├── output_images/                 # 生成画像の出力先
├── templates/
│   ├── heatmap_values_template.json
│   └── heatmap_values_example.json
└── README.md
```

基本方針は **1枚のヒートマップにつき1つのJSONファイル** です。複数のヒートマップを処理する場合は、画像の枚数ぶんJSONを分けて `input_json/` に置きます。

## 依存関係

JSON入力だけを使う場合:

```bash
pip install numpy pandas matplotlib
```

OCRを使う場合だけ、追加で Pillow と Tesseract が必要です。

```bash
pip install pillow pytesseract
```

## 基本ワークフロー

1. 画像からJSONを作る。デフォルトではOCRで `values` も入ります。
2. 生成されたJSONの `values` を確認し、必要に応じて修正する。
3. JSONからヒートマップ画像を生成する。

## 1. 画像からJSONを作る

1枚の画像からJSONを作る:

```bash
python create_json_from_image.py \
  --input_images input_images/meta_mean_steps_000000_000999.png
```

出力例:

```text
input_json/meta_mean_steps_000000_000999.json
```

デフォルトではOCRでセル値も読み取り、`values` に入れます。OCRは誤認識しやすいため、生成されたJSONの `values` は必ず確認してください。

複数画像をまとめてJSONにする:

```bash
python create_json_from_image.py \
  --input_dir input_images
```

キャプションと軸名を指定して作る:

```bash
python create_json_from_image.py \
  --input_images input_images/meta_mean_steps_000000_000999.png \
  --caption "heatmap_v1_meta | meta | mean | steps 0-999" \
  --x_axis_label "Support task" \
  --y_axis_label "Query task"
```

保存画像名も指定して作る:

```bash
python create_json_from_image.py \
  --input_images input_images/meta_mean_steps_000000_000999.png \
  --output_filename meta_v1.png
```

既存JSONを上書きする:

```bash
python create_json_from_image.py \
  --input_images input_images/meta_mean_steps_000000_000999.png \
  --overwrite
```

値を入れず、空のJSON雛形だけを作る:

```bash
python create_json_from_image.py \
  --input_images input_images/meta_mean_steps_000000_000999.png \
  --empty_values
```

## 2. JSONを編集する

テンプレート:

```text
templates/heatmap_values_template.json
```

値を入れた例:

```text
templates/heatmap_values_example.json
```

JSON形式:

```json
{
  "output_filename": "meta_v1.png",
  "caption": "heatmap_v1_meta | meta | mean | steps 0-999",
  "x_axis_label": "Support task",
  "y_axis_label": "Query task",
  "row_labels": ["comp", "rec", "sci", "talk"],
  "col_labels": ["comp", "rec", "sci", "talk"],
  "values": [
    [0.03, 7.87e-03, 0.01, 4.41e-03],
    [4.11e-03, 0.04, 4.52e-03, 6.30e-04],
    [-0.02, -1.69e-03, 0.03, -5.05e-04],
    [1.31e-03, -1.58e-03, 1.51e-03, 0.03]
  ]
}
```

各項目:

- `output_filename`: 出力PNGファイル名
- `caption`: 図の上部に表示するキャプション
- `x_axis_label`: 横軸名
- `y_axis_label`: 縦軸名
- `row_labels`: 縦軸ラベル
- `col_labels`: 横軸ラベル
- `values`: セル内に表示する元の数値

`values` は `row_labels` と `col_labels` の順番に対応します。

```text
values[0] = row_labels[0] の行
values[0][0] = row_labels[0], col_labels[0]
values[0][1] = row_labels[0], col_labels[1]
values[1][0] = row_labels[1], col_labels[0]
```

4x4で `comp`, `rec`, `sci`, `talk` の場合:

```text
values[0] = comp 行: [comp列, rec列, sci列, talk列]
values[1] = rec  行: [comp列, rec列, sci列, talk列]
values[2] = sci  行: [comp列, rec列, sci列, talk列]
values[3] = talk 行: [comp列, rec列, sci列, talk列]
```

数値は通常の小数表記と指数表記のどちらでも書けます。

```json
7.87e-03
6.30e-04
-5.05e-04
```

## 3. JSONからヒートマップを生成する

`input_json/` 内のすべてのJSONを処理する:

```bash
python robust_heatmap.py
```

入力JSONフォルダと出力先を指定する:

```bash
python robust_heatmap.py \
  --input_json_dir input_json \
  --output_dir output_images
```

特定のJSONだけ処理する:

```bash
python robust_heatmap.py \
  --input_jsons input_json/meta_v1.json \
  --output_dir output_images
```

複数JSONを指定して処理する:

```bash
python robust_heatmap.py \
  --input_jsons input_json/meta_v1.json input_json/samw_v1.json \
  --output_dir output_images
```

外れ値で色が潰れる場合に、描画用の標準化値をクリップする:

```bash
python robust_heatmap.py \
  --input_json_dir input_json \
  --output_dir output_images \
  --clip_value 3
```

カラーマップを変更する:

```bash
python robust_heatmap.py \
  --input_json_dir input_json \
  --output_dir output_images \
  --cmap RdBu_r
```

## 出力

各JSONに対してPNGだけを保存します。

```text
output_images/<output_filenameで指定した名前>
```

例:

```json
"output_filename": "meta_v1.png"
```

の場合:

```text
output_images/meta_v1.png
```

CSVファイルやJSONファイルは出力しません。

色温度の `vmin` / `vmax` は、各JSONごとに個別に計算します。複数JSONをまとめて処理しても、全画像で共通のカラースケールにはしません。

各画像内でロバスト標準化した `z` の絶対値最大を使い、中心が0になるように対称なカラースケールを設定します。`--clip_value` を指定した場合は、クリップ後の範囲で各画像ごとに `vmin` / `vmax` を決めます。

## CSV互換モード

以前のCSV形式も互換用に残しています。新しく作る場合はJSONを推奨します。

```bash
python robust_heatmap.py \
  --input_csvs input_csv/meta_v1.csv \
  --output_dir output_images
```

CSVフォルダを指定して処理する:

```bash
python robust_heatmap.py \
  --input_csv_dir input_csv \
  --output_dir output_images
```

## 画像OCRモード

OCRは誤認識しやすいため、通常はJSON入力を推奨します。必要な場合だけ画像から直接読み取れます。

```bash
python robust_heatmap.py \
  --input_dir input_images \
  --output_dir output_images_from_ocr
```

画像を個別指定する:

```bash
python robust_heatmap.py \
  --input_images input_images/meta_mean_steps_000000_000999.png \
  --output_dir output_images_from_ocr
```

OCR対象をヒートマップ本体に絞る:

```bash
python robust_heatmap.py \
  --input_images input_images/meta_mean_steps_000000_000999.png \
  --output_dir output_images_from_ocr \
  --heatmap_bbox 120 80 500 500
```

画像モードでも出力はPNGのみです。
