"""Plot the evaluation separation metric from saved 20k shower geometry."""

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
    / "outputs/plots_development/009_width_normalized_separation_20k"
)
COLORS = ("#2563eb", "#d97706", "#0f766e", "#7c3aed")
CSV_FIELDS = (
    "sample",
    "event_idx",
    "num_truth_showers",
    "num_origin_pairs",
    "min_width_normalized_separation",
    "mean_width_normalized_separation",
    "closest_origin_a",
    "closest_origin_b",
    "closest_centroid_distance_mm",
    "closest_combined_radial_width_mm",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bins", type=int, default=42)
    parser.add_argument("--upper-quantile", type=float, default=0.995)
    return parser.parse_args()


def _sample_labels(data_dir):
    suffix = "_shower_geometry.csv"
    labels = [path.name[: -len(suffix)] for path in Path(data_dir).glob(f"*{suffix}")]
    labels.sort(
        key=lambda value: (
            int(value[:-1]) if value.endswith("e") and value[:-1].isdigit() else math.inf,
            value,
        )
    )
    return labels


def _read_grouped_showers(path):
    grouped = {}
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            event_idx = int(row["event_idx"])
            grouped.setdefault(event_idx, []).append(
                {
                    "origin_label": int(row["origin_label"]),
                    "centroid_x_mm": float(row["centroid_x_mm"]),
                    "centroid_y_mm": float(row["centroid_y_mm"]),
                    "sigma_r_mm": float(row["sigma_r_mm"]),
                }
            )
    return grouped


def _event_metric(sample, event_idx, showers):
    pair_rows = []
    for first in range(len(showers) - 1):
        for second in range(first + 1, len(showers)):
            shower_a = showers[first]
            shower_b = showers[second]
            distance_mm = math.hypot(
                shower_a["centroid_x_mm"] - shower_b["centroid_x_mm"],
                shower_a["centroid_y_mm"] - shower_b["centroid_y_mm"],
            )
            combined_width_mm = math.sqrt(
                shower_a["sigma_r_mm"] ** 2 + shower_b["sigma_r_mm"] ** 2
            )
            if not math.isfinite(combined_width_mm) or combined_width_mm <= 1e-12:
                continue
            pair_rows.append(
                {
                    "origin_a": shower_a["origin_label"],
                    "origin_b": shower_b["origin_label"],
                    "distance_mm": distance_mm,
                    "combined_width_mm": combined_width_mm,
                    "normalized_separation": distance_mm / combined_width_mm,
                }
            )
    if not pair_rows:
        return None
    closest = min(pair_rows, key=lambda row: row["normalized_separation"])
    return {
        "sample": sample,
        "event_idx": event_idx,
        "num_truth_showers": len(showers),
        "num_origin_pairs": len(pair_rows),
        "min_width_normalized_separation": closest["normalized_separation"],
        "mean_width_normalized_separation": float(
            np.mean([row["normalized_separation"] for row in pair_rows])
        ),
        "closest_origin_a": closest["origin_a"],
        "closest_origin_b": closest["origin_b"],
        "closest_centroid_distance_mm": closest["distance_mm"],
        "closest_combined_radial_width_mm": closest["combined_width_mm"],
    }


def _load_analyses(input_dir):
    data_dir = Path(input_dir) / "data"
    labels = _sample_labels(data_dir)
    if not labels:
        raise FileNotFoundError(f"No *_shower_geometry.csv files found in {data_dir}")
    analyses = {}
    for label in labels:
        source_path = data_dir / f"{label}_shower_geometry.csv"
        grouped = _read_grouped_showers(source_path)
        rows = []
        skipped_events = 0
        for event_idx in sorted(grouped):
            row = _event_metric(label, event_idx, grouped[event_idx])
            if row is None:
                skipped_events += 1
            else:
                rows.append(row)
        analyses[label] = {
            "source_path": str(source_path.resolve()),
            "rows": rows,
            "skipped_events": skipped_events,
        }
    return analyses


def _nice_upper(values, quantile):
    upper = float(np.quantile(np.asarray(values, dtype=float), quantile))
    upper = max(1.0, upper)
    magnitude = 10.0 ** math.floor(math.log10(upper))
    step = magnitude / 2.0
    return max(step, math.ceil(upper / step) * step)


def _values(analysis):
    return np.asarray(
        [row["min_width_normalized_separation"] for row in analysis["rows"]],
        dtype=float,
    )


def _plot_distribution(analyses, output_stem, bins, upper_quantile):
    pooled = np.concatenate([_values(analysis) for analysis in analyses.values()])
    upper = _nice_upper(pooled, upper_quantile)
    edges = np.linspace(0.0, upper, int(bins) + 1)
    fig, ax = plt.subplots(figsize=(10.5, 6.4), layout="constrained")
    for sample_index, (label, analysis) in enumerate(analyses.items()):
        values = _values(analysis)
        clipped = np.clip(values, edges[0], np.nextafter(edges[-1], edges[0]))
        weights = np.full(clipped.size, 1.0 / max(1, clipped.size), dtype=float)
        ax.hist(
            clipped,
            bins=edges,
            weights=weights,
            histtype="step",
            linewidth=2.5,
            color=COLORS[sample_index % len(COLORS)],
            label=rf"{label} (median {np.median(values):.2f})",
        )
    ax.set_xlabel(
        r"minimum pairwise $S_{ij}=d_{ij}/\sqrt{\sigma_{r,i}^2+\sigma_{r,j}^2}$",
        fontsize=14,
    )
    ax.set_ylabel("fraction of events per bin", fontsize=14)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlim(edges[0], edges[-1])
    ax.tick_params(labelsize=12)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(title="sample", fontsize=12, title_fontsize=12)
    ax.set_title(
        "Minimum width-normalized truth-shower separation per event",
        fontsize=17,
        pad=12,
    )
    fig.text(
        0.5,
        -0.015,
        "The final bin includes any values beyond the displayed range "
        "(at most the upper 0.5% tail).",
        ha="center",
        fontsize=10,
    )
    output_stem = Path(output_stem)
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path], {"bin_edges": edges.tolist(), "upper": upper}


def _statistics(values):
    values = np.asarray(values, dtype=float)
    p25, median, p75 = np.quantile(values, [0.25, 0.5, 0.75])
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "p25": float(p25),
        "median": float(median),
        "p75": float(p75),
    }


def _write_csv(path, analyses):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for analysis in analyses.values():
            writer.writerows(analysis["rows"])
    return path


def _write_json(path, payload):
    path = Path(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_readme(output_dir, input_dir, summary):
    lines = [
        "# Width-normalized truth-shower separation",
        "",
        "This bundle plots the same event-level separation metric used for",
        "accuracy-versus-separation diagnostics:",
        "",
        "`S_min = min_(i<j) d_ij / sqrt(sigma_r,i^2 + sigma_r,j^2)`",
        "",
        "Here `d_ij` is the full 2D Euclidean distance between the x-y centroids",
        "of truth showers `i` and `j`. Each `sigma_r` is the radial RMS width",
        "computed with reconstructed hit energy times truth-origin fraction as weight.",
        "The metric is dimensionless. A 2e event has one pair; a 3e event has three",
        "pairs, and `S_min` retains the smallest ratio.",
        "",
        f"Source geometry: `{Path(input_dir).name}` (20,000 events per sample).",
        "The original plots and data in that directory are unchanged.",
        "",
        "## Summary",
        "",
    ]
    for label, sample in summary["samples"].items():
        stats = sample["min_width_normalized_separation"]
        lines.extend(
            [
                f"- {label}: median={stats['median']:.3f}, "
                f"IQR=[{stats['p25']:.3f}, {stats['p75']:.3f}], "
                f"({stats['count']:,} events).",
            ]
        )
    lines.extend(
        [
            "",
            "The displayed upper edge is rounded upward from the pooled 99.5th",
            "percentile. Any values beyond that displayed range are collected in",
            "the final bin so that the long upper tail does not compress the main",
            "distribution.",
            "",
        ]
    )
    path = Path(output_dir) / "README.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main():
    args = parse_args()
    if args.bins <= 0:
        raise ValueError("--bins must be positive.")
    if not 0.0 < args.upper_quantile <= 1.0:
        raise ValueError("--upper-quantile must be in (0, 1].")
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    analyses = _load_analyses(input_dir)
    generated = []
    generated.append(
        _write_csv(output_dir / "data/event_width_normalized_separation.csv", analyses)
    )
    plot_paths, histogram = _plot_distribution(
        analyses,
        output_dir / "01_minimum_width_normalized_separation_distribution",
        args.bins,
        args.upper_quantile,
    )
    generated.extend(plot_paths)
    summary = {
        "metric": "min_(i<j) d_ij / sqrt(sigma_r_i^2 + sigma_r_j^2)",
        "distance": "2D Euclidean distance between x-y truth-shower centroids",
        "width": "energy-fraction-weighted radial RMS shower width",
        "histogram": {
            "bins": args.bins,
            "upper_quantile": args.upper_quantile,
            "upper_edge": histogram["upper"],
            "bin_edges": histogram["bin_edges"],
            "upper_tail_collected_in_final_bin": True,
        },
        "samples": {
            label: {
                "source_shower_geometry_csv": analysis["source_path"],
                "skipped_events": analysis["skipped_events"],
                "min_width_normalized_separation": _statistics(_values(analysis)),
            }
            for label, analysis in analyses.items()
        },
    }
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
            "generated_files": sorted(
                [str(Path(item).resolve().relative_to(output_dir)) for item in generated]
                + [manifest_path.name]
            ),
        },
    )
    print(f"Saved {len(generated) + 1} artifacts to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
