#!/usr/bin/env python3
"""
Create one input JSON file per heatmap image.

Default behavior tries to fill values from the image using OCR:
  python create_json_from_image.py --input_images input_images/example.png

If you want an empty manually editable JSON skeleton:
  python create_json_from_image.py --input_images input_images/example.png --empty_values

The generated JSON files are written to ./input_json by default and can be used
by robust_heatmap.py after the values are filled or checked.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from robust_heatmap import (
    DEFAULT_LABELS,
    IMAGE_EXTENSIONS,
    collect_image_paths,
    extract_values_from_heatmap_image,
)


DEFAULT_OUTPUT_DIR = Path("input_json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create robust_heatmap.py input JSON files from heatmap images."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input_images",
        nargs="+",
        type=Path,
        help="Image paths to convert into JSON input files.",
    )
    input_group.add_argument(
        "--input_dir",
        type=Path,
        help="Directory containing images to convert into JSON input files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where generated JSON files are saved. Default: ./input_json.",
    )
    parser.add_argument(
        "--row_labels",
        nargs="+",
        default=DEFAULT_LABELS,
        help=f"Row labels for the heatmap. Default: {DEFAULT_LABELS}",
    )
    parser.add_argument(
        "--col_labels",
        nargs="+",
        default=DEFAULT_LABELS,
        help=f"Column labels for the heatmap. Default: {DEFAULT_LABELS}",
    )
    parser.add_argument(
        "--caption",
        default=None,
        help=(
            "Caption to write into every generated JSON. "
            "Defaults to each image filename stem."
        ),
    )
    parser.add_argument(
        "--x_axis_label",
        default="Support task",
        help="X-axis label written to JSON. Default: Support task.",
    )
    parser.add_argument(
        "--y_axis_label",
        default="Query task",
        help="Y-axis label written to JSON. Default: Query task.",
    )
    parser.add_argument(
        "--output_filename",
        default=None,
        help=(
            "PNG output filename to write in JSON. Only valid with one input image. "
            "Defaults to image stem + .png."
        ),
    )
    parser.add_argument(
        "--empty_values",
        action="store_true",
        help="Write null values instead of trying OCR.",
    )
    parser.add_argument(
        "--ocr_lang",
        default="eng",
        help="Tesseract OCR language used unless --empty_values is set. Default: eng.",
    )
    parser.add_argument(
        "--heatmap_bbox",
        nargs=4,
        type=int,
        metavar=("X", "Y", "W", "H"),
        help="Optional crop rectangle in pixels for OCR: X Y W H.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing JSON files.",
    )
    return parser.parse_args()


def ensure_paths_exist(paths: Sequence[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing image path(s): {', '.join(missing)}")


def resolve_output_dir(input_dir: Optional[Path], output_dir: Path) -> Path:
    """Mirror a selected category directory below the JSON output root."""
    if input_dir is None or input_dir.resolve() == Path("input_images").resolve():
        return output_dir
    return output_dir / input_dir.name


def output_png_name(image_path: Path, explicit_name: Optional[str]) -> str:
    if explicit_name:
        name = Path(explicit_name).name
    else:
        name = f"{image_path.stem}.png"
    if Path(name).suffix.lower() != ".png":
        name = f"{name}.png"
    return name


def values_from_ocr(
    image_path: Path,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    ocr_lang: str,
    heatmap_bbox: Optional[tuple[int, int, int, int]],
) -> list[list[float]]:
    values = extract_values_from_heatmap_image(
        image_path=image_path,
        row_labels=row_labels,
        col_labels=col_labels,
        expected_shape=(len(row_labels), len(col_labels)),
        output_csv_path=None,
        ocr_lang=ocr_lang,
        heatmap_bbox=heatmap_bbox,
    )
    return dataframe_to_nested_list(values)


def empty_values(row_count: int, col_count: int) -> list[list[None]]:
    return [[None for _ in range(col_count)] for _ in range(row_count)]


def dataframe_to_nested_list(values: pd.DataFrame) -> list[list[float]]:
    return [
        [float(values.iat[row_idx, col_idx]) for col_idx in range(values.shape[1])]
        for row_idx in range(values.shape[0])
    ]


def build_payload(
    image_path: Path,
    args: argparse.Namespace,
    image_count: int,
) -> dict:
    if args.output_filename and image_count != 1:
        raise ValueError("--output_filename can be used only with one input image.")

    row_labels = list(args.row_labels)
    col_labels = list(args.col_labels)
    bbox = tuple(args.heatmap_bbox) if args.heatmap_bbox else None
    if args.empty_values:
        values = empty_values(len(row_labels), len(col_labels))
    else:
        values = values_from_ocr(
            image_path=image_path,
            row_labels=row_labels,
            col_labels=col_labels,
            ocr_lang=args.ocr_lang,
            heatmap_bbox=bbox,
        )

    return {
        "output_filename": output_png_name(image_path, args.output_filename),
        "caption": args.caption if args.caption is not None else image_path.stem,
        "x_axis_label": args.x_axis_label,
        "y_axis_label": args.y_axis_label,
        "row_labels": row_labels,
        "col_labels": col_labels,
        "values": values,
    }


def write_json(payload: dict, output_path: Path, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists. Use --overwrite to replace it.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(format_heatmap_json(payload), encoding="utf-8")


def compact_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "))


def format_heatmap_json(payload: dict) -> str:
    lines = ["{"]
    lines.append(f'  "output_filename": {compact_json(payload["output_filename"])},')
    lines.append(f'  "caption": {compact_json(payload["caption"])},')
    lines.append(f'  "x_axis_label": {compact_json(payload["x_axis_label"])},')
    lines.append(f'  "y_axis_label": {compact_json(payload["y_axis_label"])},')
    lines.append(f'  "row_labels": {compact_json(payload["row_labels"])},')
    lines.append(f'  "col_labels": {compact_json(payload["col_labels"])},')
    lines.append('  "values": [')
    for row_index, row in enumerate(payload["values"]):
        comma = "," if row_index < len(payload["values"]) - 1 else ""
        lines.append(f"    {compact_json(row)}{comma}")
    lines.append("  ]")
    lines.append("}")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()

    try:
        if args.input_images:
            image_paths = args.input_images
            ensure_paths_exist(image_paths)
        else:
            image_paths = collect_image_paths(args.input_dir)
        unsupported = [
            str(path)
            for path in image_paths
            if path.suffix.lower() not in IMAGE_EXTENSIONS
        ]
        if unsupported:
            raise ValueError(f"Unsupported image extension(s): {', '.join(unsupported)}")

        destination_dir = resolve_output_dir(args.input_dir, args.output_dir)
        written = []
        for image_path in image_paths:
            payload = build_payload(image_path, args, len(image_paths))
            output_path = destination_dir / f"{image_path.stem}.json"
            write_json(payload, output_path, args.overwrite)
            written.append(output_path)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for path in written:
        print(f"Saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
