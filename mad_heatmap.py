#!/usr/bin/env python3
"""Generate zero-centered heatmaps whose color limits are based on MAD.

The input formats and command-line options are the same as robust_heatmap.py.
Original values remain visible as cell annotations. The color scale uses

    MAD = median(abs(value - median))
    blue limit = -negative_k * MAD
    red limit = positive_k * MAD

Examples:
    python mad_heatmap.py
    python mad_heatmap.py --input_json_dir input_json --output_dir output_images
    python mad_heatmap.py --input_jsons input_json/sample.json --mad_k 1 2 3
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import robust_heatmap as base


def compute_mad(values: pd.DataFrame) -> tuple[float, float]:
    """Return the median and unscaled median absolute deviation."""
    flat = values.to_numpy(dtype=float).ravel()
    if flat.size == 0:
        raise ValueError("Cannot calculate MAD for an empty value matrix.")
    if np.isnan(flat).any():
        raise ValueError("Input values contain NaN. Please fix the input values.")

    median = float(np.median(flat))
    mad = float(np.median(np.abs(flat - median)))
    return median, mad


def mad_color_limits(
    values: pd.DataFrame,
    *,
    positive_k: float,
    negative_k: float,
) -> tuple[float, float, dict]:
    """Calculate independent zero-centered color limits from MAD."""
    if positive_k <= 0 or negative_k <= 0:
        raise ValueError("MAD scale multipliers must be positive.")

    median, mad = compute_mad(values)
    scale = mad if not math.isclose(mad, 0.0, abs_tol=base.EPS) else base.EPS
    vmin = -negative_k * scale
    vmax = positive_k * scale
    return vmin, vmax, {
        "median": median,
        "mad": mad,
        "scale_used": float(scale),
        "negative_k": float(negative_k),
        "positive_k": float(positive_k),
        "vmin": float(vmin),
        "vcenter": 0.0,
        "vmax": float(vmax),
    }


def scale_directory_name(positive_k: float, negative_k: float) -> str:
    """Return a filesystem-friendly directory name recording both scales."""
    positive_text = format(positive_k, "g").replace(".", "p")
    negative_text = format(negative_k, "g").replace(".", "p")
    return f"kp_{positive_text}_kn_{negative_text}"


def mad_colorbar_ticks(vmin: float, vmax: float) -> list[float]:
    """Place three intervals on each side so negative ticks always appear."""
    negative_ticks = np.linspace(vmin, 0.0, 4)
    positive_ticks = np.linspace(0.0, vmax, 4)[1:]
    return [float(value) for value in np.concatenate((negative_ticks, positive_ticks))]


def colorbar_extend_for_values(
    values: pd.DataFrame, vmin: float, vmax: float
) -> str:
    """Indicate which colorbar ends contain values clipped to maximum color."""
    has_below = bool((values.to_numpy(dtype=float) < vmin).any())
    has_above = bool((values.to_numpy(dtype=float) > vmax).any())
    if has_below and has_above:
        return "both"
    if has_below:
        return "min"
    if has_above:
        return "max"
    return "neither"


def resolve_heatmaps(args) -> tuple[list[base.HeatmapData], Path]:
    """Load inputs using the same JSON, CSV, and OCR behavior as the base tool."""
    if args.input_dir or args.input_images:
        heatmaps = base.build_heatmap_data_from_images(args)
    elif args.input_csv_dir or args.input_csvs:
        heatmaps = base.build_heatmap_data_from_csvs(args)
    else:
        heatmaps = base.build_heatmap_data_from_jsons(args)

    output_dir = (
        base.resolve_json_output_dir(args.input_json_dir, args.output_dir)
        if args.input_jsons is None
        and not (args.input_dir or args.input_images)
        and not (args.input_csv_dir or args.input_csvs)
        else args.output_dir
    )
    return heatmaps, output_dir


def parse_args():
    parser = base.build_parser(
        "Color heatmaps around the original value 0, using multiples of the "
        "median absolute deviation (MAD) as the blue and red limits."
    )
    parser.add_argument(
        "--mad_k",
        nargs="+",
        type=float,
        default=[1.0, 2.0, 3.0],
        metavar="K",
        help=(
            "Common MAD multipliers to compare. Each value creates a separate "
            "image. Default: 1 2 3."
        ),
    )
    parser.add_argument(
        "--positive_k",
        type=float,
        default=None,
        help="Optional red-side MAD multiplier; overrides each --mad_k value.",
    )
    parser.add_argument(
        "--negative_k",
        type=float,
        default=None,
        help="Optional blue-side MAD multiplier; overrides each --mad_k value.",
    )
    for action in parser._actions:
        if action.dest == "clip_value":
            action.help = (
                "Compatibility alias for a single --mad_k value. Prefer --mad_k."
            )
            break
    args = parser.parse_args()
    if args.clip_value is not None:
        args.mad_k = [args.clip_value]
    return args


def main() -> int:
    args = parse_args()

    try:
        heatmaps, output_dir = resolve_heatmaps(args)
        output_dir.mkdir(parents=True, exist_ok=True)

        scales = []
        for common_k in args.mad_k:
            positive_k = args.positive_k if args.positive_k is not None else common_k
            negative_k = args.negative_k if args.negative_k is not None else common_k
            scale = (float(positive_k), float(negative_k))
            if scale not in scales:
                scales.append(scale)

        image_count = 0
        for item in heatmaps:
            for positive_k, negative_k in scales:
                vmin, vmax, stats = mad_color_limits(
                    item.values,
                    positive_k=positive_k,
                    negative_k=negative_k,
                )
                item.stats = stats
                # Explicit clipping makes every value outside the statistical
                # limits use the endpoint blue/red color.
                color_values = item.values.astype(float).clip(vmin, vmax)
                colorbar_ticks = mad_colorbar_ticks(vmin, vmax)
                colorbar_extend = colorbar_extend_for_values(item.values, vmin, vmax)
                source_filename = item.output_filename or f"{item.name}.png"
                scale_output_dir = output_dir / scale_directory_name(
                    positive_k, negative_k
                )
                image_output = scale_output_dir / base.ensure_png_suffix(source_filename)
                base.plot_robust_colored_heatmap(
                    item.values,
                    color_values,
                    row_labels=list(item.values.index),
                    col_labels=list(item.values.columns),
                    output_path=image_output,
                    vmin=vmin,
                    vmax=vmax,
                    clip_value=None,
                    cmap=args.cmap,
                    figsize=tuple(args.figsize),
                    dpi=args.dpi,
                    caption="",
                    x_axis_label=item.x_axis_label,
                    y_axis_label=item.y_axis_label,
                    colorbar_label="Original value (MAD-based limits)",
                    colorbar_ticks=colorbar_ticks,
                    colorbar_extend=colorbar_extend,
                )
                print(
                    f"Saved {image_output} "
                    f"(median={stats['median']:.6g}, MAD={stats['mad']:.6g}, "
                    f"limits=[{vmin:.6g}, {vmax:.6g}])"
                )
                image_count += 1
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Saved {image_count} MAD-scaled heatmap image(s) to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
