from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ml_ldmx.eval.run_comparison import (
    build_binned_profiles,
    build_multiplicity_profiles,
    comparison_summary,
    flatten_matches,
    interesting_event_groups,
    match_record_sets,
    resolve_diagnostic_path,
)
from ml_ldmx.viz.run_comparison import (
    plot_accuracy_by_multiplicity,
    plot_binned_profiles,
    plot_paired_accuracy,
)


def _record(event_idx, accuracy, model_offset=0.0):
    separation = 1.0 + event_idx
    return {
        "event_idx": event_idx,
        "split_position": event_idx,
        "source_file": f"event_{event_idx:06d}.pt",
        "source_entry": None,
        "source_label": "synthetic",
        "electron_count": 2 if event_idx < 6 else 3,
        "num_truth_classes": 2 if event_idx < 6 else 3,
        "num_hits": 100 + event_idx,
        "correct_hits": round((100 + event_idx) * accuracy),
        "incorrect_hits": (100 + event_idx) - round((100 + event_idx) * accuracy),
        "accuracy": accuracy,
        "energy_weighted_accuracy": min(1.0, accuracy + 0.02),
        "loss": 1.0 - accuracy + model_offset,
        "min_origin_centroid_distance_xy": 10.0 * separation,
        "first_layer_min_origin_centroid_distance_xy": 8.0 * separation,
        "early_min_origin_centroid_distance_xy": 9.0 * separation,
        "min_normalized_shower_separation_xy": separation,
        "early_min_normalized_shower_separation_xy": 0.8 * separation,
        "contributor_min_normalized_shower_separation_xy": separation,
        "early_contributor_min_normalized_shower_separation_xy": 0.8 * separation,
        "dominant_min_normalized_shower_separation_xy": 1.1 * separation,
        "early_dominant_min_normalized_shower_separation_xy": 0.9 * separation,
        "mean_shower_width_xy": 4.0,
        "early_mean_shower_width_xy": 3.0,
        "ambiguous_hit_fraction_xy": max(0.0, 0.5 - 0.03 * event_idx),
        "energy_weighted_ambiguous_hit_fraction_xy": max(0.0, 0.4 - 0.02 * event_idx),
        "mean_hit_centroid_margin_xy": separation - 4.0,
    }


class RunComparisonTest(unittest.TestCase):
    def setUp(self):
        self.first_records = [
            _record(idx, min(0.98, 0.45 + 0.045 * idx))
            for idx in range(12)
        ]
        self.second_records = [
            _record(idx, min(0.95, 0.40 + 0.04 * idx), model_offset=0.02)
            for idx in range(12)
        ]

    def _matched_rows(self):
        matches, match_info = match_record_sets(
            "Transformer",
            self.first_records,
            "GravNet",
            self.second_records,
        )
        rows, slugs = flatten_matches(matches, ["Transformer", "GravNet"])
        return rows, slugs, match_info

    def test_matching_and_summary_use_paired_events(self):
        rows, slugs, match_info = self._matched_rows()

        self.assertEqual(match_info["matched_events"], 12)
        self.assertEqual(len(rows), 12)
        self.assertAlmostEqual(rows[0]["accuracy_delta_transformer_minus_gravnet"], 0.05)

        summary = comparison_summary(
            rows,
            ["Transformer", "GravNet"],
            slugs,
            bootstrap_samples=20,
        )
        self.assertEqual(summary["num_matched_events"], 12)
        self.assertGreater(summary["paired_accuracy_delta"]["mean"], 0.0)

    def test_strict_matching_rejects_different_event_sets(self):
        with self.assertRaisesRegex(ValueError, "event sets differ"):
            match_record_sets(
                "Transformer",
                self.first_records,
                "GravNet",
                self.second_records[:-1],
            )

        matches, info = match_record_sets(
            "Transformer",
            self.first_records,
            "GravNet",
            self.second_records[:-1],
            allow_partial=True,
        )
        self.assertEqual(len(matches), 11)
        self.assertEqual(info["first_only_events"], 1)

    def test_profiles_and_interesting_groups_are_created(self):
        labels = ["Transformer", "GravNet"]
        rows, slugs, _match_info = self._matched_rows()
        profiles = build_binned_profiles(
            rows,
            labels,
            slugs,
            num_bins=4,
            bootstrap_samples=20,
        )
        multiplicity = build_multiplicity_profiles(
            rows,
            labels,
            slugs,
            bootstrap_samples=20,
        )
        groups = interesting_event_groups(
            rows,
            labels,
            slugs,
            low_accuracy=0.55,
            high_accuracy=0.8,
            difference_threshold=0.04,
        )

        self.assertTrue(
            any(
                row["metric"] == "contributor_min_normalized_shower_separation_xy"
                for row in profiles
            )
        )
        self.assertEqual({row["electron_count"] for row in multiplicity}, {2, 3})
        self.assertTrue(groups["both_fail"])
        self.assertTrue(groups["transformer_better"])
        self.assertEqual(groups["category_counts"]["transformer_better"], 12)

        with TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            self.assertTrue(
                plot_binned_profiles(
                    profiles,
                    labels,
                    "accuracy",
                    output_dir / "profiles.png",
                    "profiles",
                )
            )
            self.assertTrue(
                plot_paired_accuracy(
                    rows,
                    labels,
                    slugs,
                    output_dir / "paired.png",
                    "paired",
                )
            )
            self.assertTrue(
                plot_accuracy_by_multiplicity(
                    multiplicity,
                    labels,
                    output_dir / "multiplicity.png",
                    "multiplicity",
                )
            )
            for name in ("profiles.png", "paired.png", "multiplicity.png"):
                self.assertGreater((output_dir / name).stat().st_size, 0)

    def test_resolve_diagnostic_path_accepts_run_directory(self):
        with TemporaryDirectory() as temporary_dir:
            run_dir = Path(temporary_dir)
            diagnostics_dir = run_dir / "inspection/best/val"
            diagnostics_dir.mkdir(parents=True)
            diagnostics_path = diagnostics_dir / "val_event_accuracy.json"
            diagnostics_path.write_text("[]", encoding="utf-8")

            self.assertEqual(
                resolve_diagnostic_path(run_dir, split="val", checkpoint="best"),
                diagnostics_path.resolve(),
            )


if __name__ == "__main__":
    unittest.main()
