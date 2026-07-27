# MADに基づくヒートマップの色付け

`mad_heatmap.py` は、セルの元の値の `0` を色の境界に固定します。

- 負の値: 青系
- `0` 付近: 白または中立色
- 正の値: 赤系

色スケールの範囲は次式で決定します。

```text
median = median(x)
MAD = median(|x - median|)
vmin = -negative_k * MAD
vcenter = 0
vmax = positive_k * MAD
```

`TwoSlopeNorm(vmin, vcenter=0, vmax)` により、負側は `vmin -> 0`、正側は
`0 -> vmax` の各区間で独立に色が補間されます。セル値は描画前に
`[vmin, vmax]` へクリップするため、範囲外の値はそれぞれ最大濃度の
青または赤で表示されます。カラーバーの数値は正規化後の値ではなく、
元データの値に対応します。

## k = 1, 2, 3 を比較する

`--mad_k` を省略すると、デフォルトで `k=1,2,3` の3枚を生成します。

```bash
python3 mad_heatmap.py \
  --input_json_dir input_json \
  --output_dir output_images
```

出力ファイル名に使用した `k` が付きます。

```text
sample_mad_k1.png
sample_mad_k2.png
sample_mad_k3.png
```

特定の `k` だけを使う場合:

```bash
python3 mad_heatmap.py --mad_k 2
```

正側と負側で異なる幅を使う場合:

```bash
python3 mad_heatmap.py --positive_k 3 --negative_k 2
```

この場合、出力名は `sample_mad_kp3_kn2.png` のようになります。
