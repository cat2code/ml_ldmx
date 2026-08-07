"""Replot saved 20k physical geometry in Moliere-radius units."""

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = (
    PROJECT_ROOT / "outputs/plots_development/006_physical_geometry_20k"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs/plots_development/008_moliere_scaled_physical_geometry_20k"
)
DEFAULT_MOLIERE_RADIUS_MM = 25.0
COLORS = ("#2563eb", "#d97706", "#0f766e", "#7c3aed")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--moliere-radius-mm",
        type=float,
        default=DEFAULT_MOLIERE_RADIUS_MM,
    )
    return parser.parse_args()


def _read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sample_labels(data_dir):
    suffix = "_event_geometry.csv"
    labels = [path.name[: -len(suffix)] for path in Path(data_dir).glob(f"*{suffix}")]
    labels.sort(key=lambda value: (int(value[:-1]) if value[:-1].isdigit() else math.inf, value))
    return labels


def _load_analyses(input_dir):
    data_dir = Path(input_dir) / "data"
    labels = _sample_labels(data_dir)
    if not labels:
        raise FileNotFoundError(f"No *_event_geometry.csv files found in {data_dir}")
    analyses = {}
    for label in labels:
        event_path = data_dir / f"{label}_event_geometry.csv"
        shower_path = data_dir / f"{label}_shower_geometry.csv"
        if not shower_path.exists():
            raise FileNotFoundError(shower_path)
        analyses[label] = {
            "event_path": str(event_path.resolve()),
            "shower_path": str(shower_path.resolve()),
            "event_rows": _read_csv(event_path),
            "shower_rows": _read_csv(shower_path),
        }
    return analyses


def _finite_values(rows, key):
    values = []
    for row in rows:
        value = row.get(key)
        if value in (None, "", "None"):
            continue
        number = float(value)
        if np.isfinite(number):
            values.append(number)
    return np.asarray(values, dtype=float)


def _nice_upper(values, quantile=0.995):
    upper = float(np.quantile(np.asarray(values, dtype=float), quantile))
    if upper <= 0.0:
        return 1.0
    magnitude = 10.0 ** math.floor(math.log10(upper))
    step = magnitude / 2.0
    return max(step, math.ceil(upper / step) * step)


def _scaled_edges(analyses, row_key, metric_key, radius_mm, bins=42):
    pooled_mm = np.concatenate(
        [_finite_values(analysis[row_key], metric_key) for analysis in analyses.values()]
    )
    upper_mm = _nice_upper(pooled_mm)
    return np.linspace(0.0, upper_mm / radius_mm, int(bins) + 1)


def _plot_fraction_histogram(ax, values_mm, edges, radius_mm, color, label):
    values = np.asarray(values_mm, dtype=float) / float(radius_mm)
    clipped = np.clip(values, edges[0], np.nextafter(edges[-1], edges[0]))
    weights = np.full(clipped.size, 1.0 / max(1, clipped.size), dtype=float)
    ax.hist(
        clipped,
        bins=edges,
        weights=weights,
        histtype="step",
        linewidth=2.4,
        color=color,
        label=rf"{label} (median {np.median(values):.2f} $R_{{\mathrm{{M}}}}$)",
    )


def _save_figure(fig, output_stem):
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path]


def plot_centroid_separations(analyses, output_stem, radius_mm):
    metrics = (
        (
            "min_abs_delta_mu_x_mm",
            r"minimum pairwise $|\Delta\mu_x|/R_{\mathrm{M}}$",
            "x separation",
        ),
        (
            "min_abs_delta_mu_y_mm",
            r"minimum pairwise $|\Delta\mu_y|/R_{\mathrm{M}}$",
            "y separation",
        ),
    )
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.7), layout="constrained")
    for axis_index, (metric, xlabel, title) in enumerate(metrics):
        ax = axes[axis_index]
        edges = _scaled_edges(analyses, "event_rows", metric, radius_mm)
        for sample_index, (label, analysis) in enumerate(analyses.items()):
            values_mm = _finite_values(analysis["event_rows"], metric)
            _plot_fraction_histogram(
                ax,
                values_mm,
                edges,
                radius_mm,
                COLORS[sample_index % len(COLORS)],
                label,
            )
        ax.set_xlabel(xlabel, fontsize=15)
        ax.set_ylabel("fraction of events per bin", fontsize=15)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_xlim(edges[0], edges[-1])
        ax.tick_params(labelsize=13)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(title="sample", fontsize=12, title_fontsize=12)
        ax.set_title(title, fontsize=17)
    fig.suptitle(
        "Closest truth-shower centroid separation in Molière-radius units",
        fontsize=20,
    )
    fig.text(
        0.5,
        -0.015,
        rf"$R_{{\mathrm{{M}}}}={radius_mm:g}$ mm; the final bin collects the upper 0.5% tail.",
        ha="center",
        fontsize=10,
    )
    return _save_figure(fig, output_stem)


def plot_shower_widths(analyses, output_stem, radius_mm):
    metrics = (
        ("sigma_x_mm", r"$\sigma_x/R_{\mathrm{M}}$", "x width"),
        ("sigma_y_mm", r"$\sigma_y/R_{\mathrm{M}}$", "y width"),
        (
            "sigma_r_mm",
            r"$\sigma_r/R_{\mathrm{M}}$,  $\sigma_r=\sqrt{\sigma_x^2+\sigma_y^2}$",
            "radial width",
        ),
    )
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.6), layout="constrained")
    for axis_index, (metric, xlabel, title) in enumerate(metrics):
        ax = axes[axis_index]
        edges = _scaled_edges(analyses, "shower_rows", metric, radius_mm)
        for sample_index, (label, analysis) in enumerate(analyses.items()):
            values_mm = _finite_values(analysis["shower_rows"], metric)
            _plot_fraction_histogram(
                ax,
                values_mm,
                edges,
                radius_mm,
                COLORS[sample_index % len(COLORS)],
                label,
            )
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel("fraction of truth showers per bin", fontsize=14)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_xlim(edges[0], edges[-1])
        ax.tick_params(labelsize=12)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(title="sample", fontsize=11, title_fontsize=11)
        ax.set_title(title, fontsize=16)
    fig.suptitle(
        "Energy-fraction-weighted shower widths in Molière-radius units",
        fontsize=20,
    )
    fig.text(
        0.5,
        -0.015,
        rf"$R_{{\mathrm{{M}}}}={radius_mm:g}$ mm; the final bin collects the upper 0.5% tail.",
        ha="center",
        fontsize=10,
    )
    return _save_figure(fig, output_stem)


def _statistics(values):
    values = np.asarray(values, dtype=float)
    quantiles = np.quantile(values, [0.25, 0.5, 0.75])
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "p25": float(quantiles[0]),
        "median": float(quantiles[1]),
        "p75": float(quantiles[2]),
    }


def _build_summary(analyses, radius_mm):
    event_metrics = ("min_abs_delta_mu_x_mm", "min_abs_delta_mu_y_mm")
    shower_metrics = ("sigma_x_mm", "sigma_y_mm", "sigma_r_mm")
    return {
        "moliere_radius_mm": float(radius_mm),
        "scaling": "dimensionless_value = millimetre_value / moliere_radius_mm",
        "source_bundle": str(DEFAULT_INPUT_DIR.resolve()),
        "samples": {
            label: {
                "num_events": len(analysis["event_rows"]),
                "num_truth_showers": len(analysis["shower_rows"]),
                "centroid_separation_in_moliere_radii": {
                    metric.removesuffix("_mm"): _statistics(
                        _finite_values(analysis["event_rows"], metric) / radius_mm
                    )
                    for metric in event_metrics
                },
                "shower_width_in_moliere_radii": {
                    metric.removesuffix("_mm"): _statistics(
                        _finite_values(analysis["shower_rows"], metric) / radius_mm
                    )
                    for metric in shower_metrics
                },
            }
            for label, analysis in analyses.items()
        },
    }


def _write_readme(output_dir, input_dir, summary):
    radius_mm = summary["moliere_radius_mm"]
    lines = [
        "# Molière-radius-scaled physical geometry plots",
        "",
        f"These are alternative versions of plots 01 and 02 from `{Path(input_dir).name}`.",
        "The original millimetre-scaled PNG and PDF files are unchanged.",
        "",
        f"Every horizontal quantity is divided by the approximate LDMX ECal Molière radius",
        f"used for this development comparison, `R_M = {radius_mm:g} mm`:",
        "",
        "- Centroid axes show `minimum |delta mu_x| / R_M` and",
        "  `minimum |delta mu_y| / R_M` per event.",
        "- Width axes show `sigma_x / R_M`, `sigma_y / R_M`, and",
        "  `sigma_r / R_M` per truth shower.",
        "",
        "The transformation changes only the units of the x axes. Event selection,",
        "truth weighting, histogram bin membership, and upper-tail handling are identical",
        "to the source plots. A value of 1 corresponds to 25 mm, or one assumed Molière",
        "radius.",
        "",
        "## Plot index",
        "",
        "1. `01_truth_centroid_separation_distributions_moliere_scaled`",
        "2. `02_truth_shower_width_distributions_moliere_scaled`",
        "",
        "## Median values",
        "",
    ]
    for label, sample in summary["samples"].items():
        separation = sample["centroid_separation_in_moliere_radii"]
        widths = sample["shower_width_in_moliere_radii"]
        lines.extend(
            [
                f"### {label}",
                "",
                "- Closest centroid separation: "
                f"x={separation['min_abs_delta_mu_x']['median']:.3f} R_M, "
                f"y={separation['min_abs_delta_mu_y']['median']:.3f} R_M.",
                "- Shower widths: "
                f"sigma_x={widths['sigma_x']['median']:.3f} R_M, "
                f"sigma_y={widths['sigma_y']['median']:.3f} R_M, "
                f"sigma_r={widths['sigma_r']['median']:.3f} R_M.",
                "",
            ]
        )
    path = Path(output_dir) / "README.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main():
    args = parse_args()
    if args.moliere_radius_mm <= 0.0:
        raise ValueError("--moliere-radius-mm must be positive.")
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    analyses = _load_analyses(input_dir)
    generated = []
    generated.extend(
        plot_centroid_separations(
            analyses,
            output_dir / "01_truth_centroid_separation_distributions_moliere_scaled",
            args.moliere_radius_mm,
        )
    )
    generated.extend(
        plot_shower_widths(
            analyses,
            output_dir / "02_truth_shower_width_distributions_moliere_scaled",
            args.moliere_radius_mm,
        )
    )
    summary = _build_summary(analyses, args.moliere_radius_mm)
    summary["source_bundle"] = str(input_dir)
    summary_path = _write_json(output_dir / "summary.json", summary)
    generated.append(summary_path)
    generated.append(_write_readme(output_dir, input_dir, summary))
    manifest_path = output_dir / "manifest.json"
    _write_json(
        manifest_path,
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "script": str(Path(__file__).resolve()),
            "source_bundle": str(input_dir),
            "moliere_radius_mm": args.moliere_radius_mm,
            "source_data": {
                label: {
                    "event_geometry_csv": analysis["event_path"],
                    "shower_geometry_csv": analysis["shower_path"],
                }
                for label, analysis in analyses.items()
            },
            "generated_files": sorted(
                [str(Path(item).resolve().relative_to(output_dir)) for item in generated]
                + [manifest_path.name]
            ),
        },
    )
    print(f"Saved {len(generated) + 1} artifacts to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
