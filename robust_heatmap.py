#!/usr/bin/env python3
"""
Regenerate heatmaps whose cell colors are based on per-heatmap robust
standardization while cell annotations keep the original values.

Examples:
  Default JSON-folder mode:
    python robust_heatmap.py

    Reads all JSON files in ./input_json and writes heatmap images to ./output_images.

  OCR mode:
    python robust_heatmap.py \
      --input_images path/to/heatmap_meta.png path/to/heatmap_samw.png \
      --output_dir path/to/output

  JSON mode:
    python robust_heatmap.py \
      --input_jsons path/to/meta_values.json path/to/samw_values.json \
      --output_dir path/to/output

Recommended JSON input uses one JSON file per heatmap image. See templates/.

OCR mode requires Pillow plus either pytesseract or a system tesseract command.
OCR is inherently noisy, so JSON input is the recommended workflow.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

_cache_base = Path(tempfile.gettempdir())
os.environ.setdefault("MPLCONFIGDIR", str(_cache_base / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_base / "xdg-cache"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm


DEFAULT_LABELS = ["comp", "rec", "sci", "talk"]
DEFAULT_INPUT_DIR = Path("input_images")
DEFAULT_INPUT_JSON_DIR = Path("input_json")
DEFAULT_INPUT_CSV_DIR = Path("input_csv")
DEFAULT_OUTPUT_DIR = Path("output_images")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
JSON_EXTENSIONS = {".json"}
CSV_EXTENSIONS = {".csv"}
EPS = 1e-12
NUMBER_RE = re.compile(
    r"[-+]?(?:(?:\d+\.\d*)|(?:\.\d+)|(?:\d+))(?:[eE][-+]?\d+)?"
)


@dataclass
class HeatmapData:
    source_path: Path
    name: str
    values: pd.DataFrame
    z_values: pd.DataFrame
    stats: dict
    caption: str = ""
    x_axis_label: str = ""
    y_axis_label: str = ""
    output_filename: str = ""


def build_parser(description: Optional[str] = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description or (
            "Recolor heatmaps using robust standardized cell values while "
            "preserving original numeric annotations."
        )
    )
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument(
        "--input_json_dir",
        type=Path,
        default=None,
        help=(
            "Directory containing JSON files to process. "
            "Defaults to ./input_json when no explicit input is given."
        ),
    )
    input_group.add_argument(
        "--input_jsons",
        nargs="+",
        type=Path,
        help="JSON paths containing heatmap settings and cell values.",
    )
    input_group.add_argument(
        "--input_csv_dir",
        type=Path,
        default=None,
        help=(
            "Directory containing CSV files to process. Kept for compatibility."
        ),
    )
    input_group.add_argument(
        "--input_dir",
        type=Path,
        default=None,
        help=(
            "Directory containing heatmap images to process. "
            "Use this only for OCR mode."
        ),
    )
    input_group.add_argument(
        "--input_images",
        nargs="+",
        type=Path,
        help="Heatmap image paths. Cell values are extracted with OCR.",
    )
    input_group.add_argument(
        "--input_csvs",
        nargs="+",
        type=Path,
        help="CSV paths containing heatmap cell values.",
    )
    parser.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
        type=Path,
        help="Directory where regenerated heatmap images are saved. Default: ./output_images.",
    )
    parser.add_argument(
        "--row_labels",
        nargs="+",
        default=DEFAULT_LABELS,
        help=f"Default row labels used by OCR mode. Default: {DEFAULT_LABELS}",
    )
    parser.add_argument(
        "--col_labels",
        nargs="+",
        default=DEFAULT_LABELS,
        help=f"Default column labels used by OCR mode. Default: {DEFAULT_LABELS}",
    )
    parser.add_argument(
        "--expected_rows",
        type=int,
        default=None,
        help="Expected number of heatmap rows. Defaults to len(row_labels).",
    )
    parser.add_argument(
        "--expected_cols",
        type=int,
        default=None,
        help="Expected number of heatmap columns. Defaults to len(col_labels).",
    )
    parser.add_argument(
        "--clip_value",
        type=float,
        default=None,
        help="Clip z values to [-clip_value, clip_value] before coloring.",
    )
    parser.add_argument(
        "--cmap",
        default="coolwarm",
        help="Diverging matplotlib colormap. Default: coolwarm.",
    )
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=(6.0, 5.2),
        metavar=("WIDTH", "HEIGHT"),
        help="Output figure size in inches. Default: 6.0 5.2.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output image DPI. Default: 300.",
    )
    parser.add_argument(
        "--ocr_lang",
        default="eng",
        help="Tesseract OCR language used in image mode. Default: eng.",
    )
    parser.add_argument(
        "--heatmap_bbox",
        nargs=4,
        type=int,
        metavar=("X", "Y", "W", "H"),
        help=(
            "Optional crop rectangle for OCR in pixels. Use this when titles, "
            "axis labels, or colorbars confuse OCR."
        ),
    )
    return parser


def parse_args(description: Optional[str] = None) -> argparse.Namespace:
    return build_parser(description).parse_args()


def ensure_paths_exist(paths: Sequence[Path], kind: str) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {kind}: {', '.join(missing)}")


def collect_image_paths(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    image_paths = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(
            f"No image files found in {input_dir}. "
            f"Supported extensions: {', '.join(sorted(IMAGE_EXTENSIONS))}"
        )
    return image_paths


def collect_csv_paths(input_csv_dir: Path) -> list[Path]:
    if not input_csv_dir.exists():
        raise FileNotFoundError(f"Input CSV directory does not exist: {input_csv_dir}")
    if not input_csv_dir.is_dir():
        raise NotADirectoryError(f"Input CSV path is not a directory: {input_csv_dir}")

    csv_paths = sorted(
        path
        for path in input_csv_dir.iterdir()
        if path.is_file() and path.suffix.lower() in CSV_EXTENSIONS
    )
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {input_csv_dir}.")
    return csv_paths


def collect_json_paths(input_json_dir: Path) -> list[Path]:
    if not input_json_dir.exists():
        raise FileNotFoundError(f"Input JSON directory does not exist: {input_json_dir}")
    if not input_json_dir.is_dir():
        raise NotADirectoryError(f"Input JSON path is not a directory: {input_json_dir}")

    json_paths = sorted(
        path
        for path in input_json_dir.iterdir()
        if path.is_file() and path.suffix.lower() in JSON_EXTENSIONS
    )
    if not json_paths:
        raise FileNotFoundError(f"No JSON files found in {input_json_dir}.")
    return json_paths


def resolve_json_output_dir(
    input_json_dir: Optional[Path], output_dir: Path
) -> Path:
    """Mirror a selected JSON category directory below the image output root."""
    if (
        input_json_dir is None
        or input_json_dir.resolve() == DEFAULT_INPUT_JSON_DIR.resolve()
    ):
        return output_dir
    return output_dir / input_json_dir.name


def extract_values_from_heatmap_image(
    image_path: Path,
    row_labels: Optional[Sequence[str]] = None,
    col_labels: Optional[Sequence[str]] = None,
    expected_shape: Optional[tuple[int, int]] = None,
    output_csv_path: Optional[Path] = None,
    ocr_lang: str = "eng",
    heatmap_bbox: Optional[tuple[int, int, int, int]] = None,
) -> pd.DataFrame:
    """
    Extract visible numeric cell annotations from a heatmap image with OCR.

    The implementation sorts recognized numeric tokens top-to-bottom and
    left-to-right, then reshapes them into expected_shape. If OCR returns more
    numbers than expected, the first expected count after sorting is used and a
    warning is emitted. OCR is intentionally kept as a fallback; JSON input is
    recommended for reliable results.
    """
    row_labels = list(row_labels or DEFAULT_LABELS)
    col_labels = list(col_labels or DEFAULT_LABELS)
    expected_shape = expected_shape or (len(row_labels), len(col_labels))
    expected_count = expected_shape[0] * expected_shape[1]

    ocr_data = ocr_image_to_data(
        image_path=image_path,
        ocr_lang=ocr_lang,
        heatmap_bbox=heatmap_bbox,
    )

    tokens: list[tuple[int, int, float, str]] = []
    for text, left, top in zip(
        ocr_data.get("text", []),
        ocr_data.get("left", []),
        ocr_data.get("top", []),
    ):
        cleaned = normalize_ocr_number_text(text)
        for match in NUMBER_RE.findall(cleaned):
            try:
                tokens.append((int(top), int(left), float(match), match))
            except ValueError:
                continue

    if len(tokens) != expected_count:
        warnings.warn(
            f"{image_path.name}: OCR found {len(tokens)} numeric values, "
            f"but expected {expected_count}. JSON input is recommended.",
            stacklevel=2,
        )
    if len(tokens) < expected_count:
        raise ValueError(
            f"{image_path.name}: OCR found too few values "
            f"({len(tokens)}/{expected_count}). Try --heatmap_bbox or JSON input."
        )

    tokens.sort(key=lambda item: (item[0], item[1]))
    values = np.array([item[2] for item in tokens[:expected_count]], dtype=float)
    matrix = values.reshape(expected_shape)
    frame = pd.DataFrame(
        matrix,
        index=label_list(row_labels, expected_shape[0], "row"),
        columns=label_list(col_labels, expected_shape[1], "col"),
    )

    return frame


def ocr_image_to_data(
    image_path: Path,
    ocr_lang: str,
    heatmap_bbox: Optional[tuple[int, int, int, int]],
) -> dict:
    """Return Tesseract OCR token data using pytesseract or the tesseract CLI."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "OCR mode requires Pillow for image loading/cropping. "
            "Install pillow, or use JSON input instead."
        ) from exc

    image = Image.open(image_path)
    if heatmap_bbox is not None:
        x, y, width, height = heatmap_bbox
        image = image.crop((x, y, x + width, y + height))

    config = "--psm 6 -c tessedit_char_whitelist=0123456789.-+eE"
    try:
        import pytesseract

        return pytesseract.image_to_data(
            image,
            lang=ocr_lang,
            config=config,
            output_type=pytesseract.Output.DICT,
        )
    except ImportError:
        return ocr_image_to_data_with_cli(image, ocr_lang)


def ocr_image_to_data_with_cli(image: "Image.Image", ocr_lang: str) -> dict:
    tesseract = shutil.which("tesseract")
    if not tesseract:
        raise RuntimeError(
            "OCR mode requires pytesseract or a system tesseract command. "
            "Install one of them, or use JSON input instead."
        )

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        image.save(tmp_path)
        result = subprocess.run(
            [
                tesseract,
                str(tmp_path),
                "stdout",
                "-l",
                ocr_lang,
                "--psm",
                "6",
                "-c",
                "tessedit_char_whitelist=0123456789.-+eE",
                "tsv",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    if result.returncode != 0:
        raise RuntimeError(f"tesseract failed: {result.stderr.strip()}")

    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return {"text": [], "left": [], "top": []}

    header = lines[0].split("\t")
    columns = {name: idx for idx, name in enumerate(header)}
    required = {"text", "left", "top"}
    if not required.issubset(columns):
        raise RuntimeError("tesseract TSV output did not include text/left/top columns.")

    data = {"text": [], "left": [], "top": []}
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) <= max(columns[name] for name in required):
            continue
        data["text"].append(parts[columns["text"]])
        data["left"].append(parts[columns["left"]])
        data["top"].append(parts[columns["top"]])
    return data


def normalize_ocr_number_text(text: str) -> str:
    """Clean common OCR substitutions without trying to be too clever."""
    return (
        str(text)
        .strip()
        .replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace("O", "0")
        .replace("o", "0")
        .replace(",", ".")
    )


def label_list(labels: Sequence[str], size: int, prefix: str) -> list[str]:
    labels = list(labels)
    if len(labels) >= size:
        return labels[:size]
    return labels + [f"{prefix}{idx + 1}" for idx in range(len(labels), size)]


def read_values_csv(csv_path: Path) -> pd.DataFrame:
    values, _, _, _ = read_heatmap_csv(csv_path)
    return values


def read_heatmap_json(json_path: Path) -> tuple[pd.DataFrame, str, str, str, str]:
    """Read one heatmap definition from a JSON file."""
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{json_path}: JSON root must be an object.")

    required = ["row_labels", "col_labels", "values"]
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"{json_path}: missing required key(s): {', '.join(missing)}")

    row_labels = payload["row_labels"]
    col_labels = payload["col_labels"]
    values = payload["values"]
    if not isinstance(row_labels, list) or not row_labels:
        raise ValueError(f"{json_path}: row_labels must be a non-empty list.")
    if not isinstance(col_labels, list) or not col_labels:
        raise ValueError(f"{json_path}: col_labels must be a non-empty list.")
    if not isinstance(values, list) or not values:
        raise ValueError(f"{json_path}: values must be a non-empty 2D list.")

    row_labels = [str(label) for label in row_labels]
    col_labels = [str(label) for label in col_labels]
    matrix = np.array(values, dtype=float)
    expected_shape = (len(row_labels), len(col_labels))
    if matrix.shape != expected_shape:
        raise ValueError(
            f"{json_path}: values shape must be {expected_shape}, "
            f"but got {matrix.shape}."
        )
    if np.isnan(matrix).any():
        raise ValueError(f"{json_path}: values contain NaN.")

    output_filename = str(payload.get("output_filename") or "").strip()
    if output_filename:
        output_filename = ensure_png_suffix(Path(output_filename).name)

    return (
        pd.DataFrame(matrix, index=row_labels, columns=col_labels),
        str(payload.get("caption") or ""),
        str(payload.get("x_axis_label") or ""),
        str(payload.get("y_axis_label") or ""),
        output_filename,
    )


def read_heatmap_csv(csv_path: Path) -> tuple[pd.DataFrame, str, str, str]:
    """Read values and optional plot metadata from a heatmap CSV."""
    raw = pd.read_csv(csv_path)
    lower_cols = {str(col).strip().lower(): col for col in raw.columns}

    if {"row_label", "col_label", "value"}.issubset(lower_cols):
        row_col = lower_cols["row_label"]
        col_col = lower_cols["col_label"]
        value_col = lower_cols["value"]
        validate_single_heatmap_id(raw, lower_cols, csv_path)
        raw[row_col] = raw[row_col].astype(str)
        raw[col_col] = raw[col_col].astype(str)
        raw[value_col] = pd.to_numeric(raw[value_col], errors="raise")
        row_order = list(dict.fromkeys(raw[row_col].tolist()))
        col_order = list(dict.fromkeys(raw[col_col].tolist()))
        frame = raw.pivot(index=row_col, columns=col_col, values=value_col)
        frame = frame.reindex(index=row_order, columns=col_order)
        if frame.isna().any().any():
            raise ValueError(
                f"{csv_path}: CSV must contain exactly one value for every "
                "row_label/col_label combination."
            )
        caption = first_non_empty(raw, lower_cols, "caption")
        x_axis_label = first_non_empty(raw, lower_cols, "x_axis_label")
        y_axis_label = first_non_empty(raw, lower_cols, "y_axis_label")
        return frame.astype(float), caption, x_axis_label, y_axis_label

    frame = pd.read_csv(csv_path, index_col=0)
    try:
        return frame.astype(float), "", "", ""
    except ValueError as exc:
        raise ValueError(
            f"{csv_path}: CSV must be matrix numeric data or long format with "
            "row_label, col_label, value columns."
        ) from exc


def validate_single_heatmap_id(raw: pd.DataFrame, lower_cols: dict, csv_path: Path) -> None:
    heatmap_id_col = lower_cols.get("heatmap_id")
    if heatmap_id_col is None:
        return
    heatmap_ids = raw[heatmap_id_col].dropna().astype(str).map(str.strip)
    heatmap_ids = sorted(set(heatmap_id for heatmap_id in heatmap_ids if heatmap_id))
    if len(heatmap_ids) > 1:
        raise ValueError(
            f"{csv_path}: one CSV must describe exactly one heatmap/image. "
            f"Found multiple heatmap_id values: {', '.join(heatmap_ids)}"
        )


def first_non_empty(raw: pd.DataFrame, lower_cols: dict, column_name: str) -> str:
    source_col = lower_cols.get(column_name)
    if source_col is None:
        return ""
    values = raw[source_col].dropna().astype(str).map(str.strip)
    values = values[values != ""]
    return values.iloc[0] if not values.empty else ""


def robust_standardize(values: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    flat = values.to_numpy(dtype=float).ravel()
    if flat.size == 0:
        raise ValueError("Cannot standardize an empty value matrix.")
    if np.isnan(flat).any():
        raise ValueError("Input values contain NaN. Please fix the input values.")

    q1 = float(np.percentile(flat, 25))
    median = float(np.median(flat))
    q3 = float(np.percentile(flat, 75))
    iqr = q3 - q1
    scale = iqr if not math.isclose(iqr, 0.0, abs_tol=EPS) else EPS
    z = (values.astype(float) - median) / scale
    stats = {
        "median": median,
        "q1": q1,
        "q3": q3,
        "iqr": float(iqr),
        "scale_used": float(scale),
    }
    return z, stats


def plot_robust_colored_heatmap(
    original_values: pd.DataFrame,
    standardized_values: pd.DataFrame,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    output_path: Path,
    *,
    vmin: float,
    vmax: float,
    clip_value: Optional[float] = None,
    cmap: str = "coolwarm",
    figsize: tuple[float, float] = (6.0, 5.2),
    dpi: int = 300,
    caption: str = "",
    x_axis_label: str = "",
    y_axis_label: str = "",
    colorbar_label: str = "Robust standardized value",
    colorbar_ticks: Optional[Sequence[float]] = None,
    colorbar_extend: str = "neither",
) -> None:
    plot_values = standardized_values.astype(float)
    if clip_value is not None:
        if clip_value <= 0:
            raise ValueError("--clip_value must be positive.")
        plot_values = plot_values.clip(lower=-clip_value, upper=clip_value)

    annotations = original_values.apply(lambda column: column.map(format_annotation))
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=figsize)
    image = ax.imshow(plot_values.to_numpy(dtype=float), cmap=cmap, norm=norm)
    cbar = fig.colorbar(image, ax=ax, extend=colorbar_extend)
    if colorbar_ticks is not None:
        cbar.set_ticks(list(colorbar_ticks))
    cbar.set_label(colorbar_label)

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(list(col_labels))
    ax.set_yticklabels(list(row_labels))

    ax.set_xticks(np.arange(plot_values.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(plot_values.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    for row_idx in range(plot_values.shape[0]):
        for col_idx in range(plot_values.shape[1]):
            ax.text(
                col_idx,
                row_idx,
                annotations.iat[row_idx, col_idx],
                ha="center",
                va="center",
                color="black",
            )

    ax.set_title(caption)
    ax.set_xlabel(x_axis_label)
    ax.set_ylabel(y_axis_label)
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)
    ax.set_aspect("equal")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def format_annotation(value: float) -> str:
    value = float(value)
    if value == 0:
        return "0"
    if abs(value) < 0.001 or abs(value) >= 1000:
        return f"{value:.2e}"
    return f"{value:.3g}"


def output_stem(path: Path) -> str:
    return path.stem


def ensure_png_suffix(filename: str) -> str:
    path = Path(filename)
    if path.suffix.lower() == ".png":
        return path.name
    return f"{path.name}.png"


def build_heatmap_data_from_images(args: argparse.Namespace) -> list[HeatmapData]:
    input_images = args.input_images
    if input_images is None:
        input_images = collect_image_paths(args.input_dir or DEFAULT_INPUT_DIR)
    ensure_paths_exist(input_images, "input image(s)")
    expected_shape = (
        args.expected_rows or len(args.row_labels),
        args.expected_cols or len(args.col_labels),
    )
    heatmaps = []
    bbox = tuple(args.heatmap_bbox) if args.heatmap_bbox else None

    for image_path in input_images:
        stem = output_stem(image_path)
        values = extract_values_from_heatmap_image(
            image_path,
            row_labels=args.row_labels,
            col_labels=args.col_labels,
            expected_shape=expected_shape,
            output_csv_path=None,
            ocr_lang=args.ocr_lang,
            heatmap_bbox=bbox,
        )
        z_values, stats = robust_standardize(values)
        heatmaps.append(
            HeatmapData(
                source_path=image_path,
                name=stem,
                values=values,
                z_values=z_values,
                stats=stats,
                output_filename=f"{stem}.png",
            )
        )
    return heatmaps


def build_heatmap_data_from_jsons(args: argparse.Namespace) -> list[HeatmapData]:
    input_jsons = args.input_jsons
    if input_jsons is None:
        input_jsons = collect_json_paths(args.input_json_dir or DEFAULT_INPUT_JSON_DIR)
    ensure_paths_exist(input_jsons, "input JSON(s)")
    heatmaps = []
    for json_path in input_jsons:
        values, caption, x_axis_label, y_axis_label, output_filename = read_heatmap_json(
            json_path
        )
        z_values, stats = robust_standardize(values)
        heatmaps.append(
            HeatmapData(
                source_path=json_path,
                name=output_stem(json_path),
                values=values,
                z_values=z_values,
                stats=stats,
                caption=caption,
                x_axis_label=x_axis_label,
                y_axis_label=y_axis_label,
                output_filename=output_filename or f"{output_stem(json_path)}.png",
            )
        )
    return heatmaps


def build_heatmap_data_from_csvs(args: argparse.Namespace) -> list[HeatmapData]:
    input_csvs = args.input_csvs
    if input_csvs is None:
        input_csvs = collect_csv_paths(args.input_csv_dir or DEFAULT_INPUT_CSV_DIR)
    ensure_paths_exist(input_csvs, "input CSV(s)")
    heatmaps = []
    for csv_path in input_csvs:
        values, caption, x_axis_label, y_axis_label = read_heatmap_csv(csv_path)
        z_values, stats = robust_standardize(values)
        heatmaps.append(
            HeatmapData(
                source_path=csv_path,
                name=output_stem(csv_path),
                values=values,
                z_values=z_values,
                stats=stats,
                caption=caption,
                x_axis_label=x_axis_label,
                y_axis_label=y_axis_label,
                output_filename=f"{output_stem(csv_path)}.png",
            )
        )
    return heatmaps


def compute_color_limits(
    z_values: pd.DataFrame,
    clip_value: Optional[float],
) -> tuple[float, float]:
    all_z = z_values.to_numpy(dtype=float).ravel()
    if clip_value is not None:
        if clip_value <= 0:
            raise ValueError("--clip_value must be positive.")
        all_z = np.clip(all_z, -clip_value, clip_value)

    max_abs = float(np.max(np.abs(all_z))) if all_z.size else 0.0
    if math.isclose(max_abs, 0.0, abs_tol=EPS):
        max_abs = 1.0
    return -max_abs, max_abs


def main() -> int:
    args = parse_args()

    try:
        if args.input_dir or args.input_images:
            heatmaps = build_heatmap_data_from_images(args)
        elif args.input_csv_dir or args.input_csvs:
            heatmaps = build_heatmap_data_from_csvs(args)
        else:
            heatmaps = build_heatmap_data_from_jsons(args)

        output_dir = (
            resolve_json_output_dir(args.input_json_dir, args.output_dir)
            if args.input_jsons is None
            and not (args.input_dir or args.input_images)
            and not (args.input_csv_dir or args.input_csvs)
            else args.output_dir
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        for item in heatmaps:
            vmin, vmax = compute_color_limits(item.z_values, args.clip_value)
            image_output = output_dir / ensure_png_suffix(
                item.output_filename or f"{item.name}.png"
            )
            plot_robust_colored_heatmap(
                item.values,
                item.z_values,
                row_labels=list(item.values.index),
                col_labels=list(item.values.columns),
                output_path=image_output,
                vmin=vmin,
                vmax=vmax,
                clip_value=args.clip_value,
                cmap=args.cmap,
                figsize=tuple(args.figsize),
                dpi=args.dpi,
                caption=item.caption,
                x_axis_label=item.x_axis_label,
                y_axis_label=item.y_axis_label,
            )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Saved {len(heatmaps)} heatmap image(s) to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
