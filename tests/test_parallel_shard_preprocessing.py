import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from ml_ldmx.datasets.ecal_tpad_shards import (
    SHARD_PAYLOAD_SCHEMA_VERSION,
    MultiShardedECalTpadDataset,
    ShardedECalTpadDataset,
    create_parallel_shard_plan,
    finalize_parallel_shard_plan,
)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _event(local_event_idx=0):
    ecal_raw_energy = torch.tensor([0.0, 9.0])
    ecal_input_energy = torch.log1p(ecal_raw_energy)
    tpad_raw_pe = torch.tensor([8.0])
    tpad = torch.tensor([[1.5, torch.log1p(tpad_raw_pe[0])]])
    return {
        "x": torch.tensor(
            [
                [1.0, 0.0, 1.0, 2.0, 3.0, ecal_input_energy[0], 0.0, 0.0],
                [1.0, 0.0, 4.0, 5.0, 6.0, ecal_input_energy[1], 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, tpad[0, 0], tpad[0, 1]],
            ],
            dtype=torch.float32,
        ),
        "ecal_mask": torch.tensor([True, True, False]),
        "tpad_mask": torch.tensor([False, False, True]),
        "ecal_pos": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        "pos": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        "ecal_input_energy": ecal_input_energy,
        "ecal_raw_energy": ecal_raw_energy,
        "tpad": tpad,
        "tpad_raw_pe": tpad_raw_pe,
        "y": torch.tensor([0, 1]),
        "physical_y": torch.tensor([1, 2]),
        "event_idx": torch.tensor(local_event_idx),
    }


class ParallelShardPreprocessingTest(unittest.TestCase):
    def _plan_with_worker_outputs(self, temporary_dir, omit_task=None):
        temporary_dir = Path(temporary_dir)
        root_2e = temporary_dir / "root/2e"
        root_3e = temporary_dir / "root/3e"
        root_2e.mkdir(parents=True)
        root_3e.mkdir(parents=True)
        for path in (root_2e / "events_1.root", root_2e / "events_2.root", root_3e / "events_1.root"):
            path.touch()

        plan_path = temporary_dir / "output/_slurm/plan.json"
        plan = create_parallel_shard_plan(
            plan_path,
            output_root=temporary_dir / "output",
            root_specs=[(2, "2e", root_2e), (3, "3e", root_3e)],
            ecal_energy_transform="log1p",
            tpad_pe_transform="log1p",
        )
        for task in plan["tasks"]:
            if task["task_index"] == omit_task:
                continue
            shard_path = Path(task["shard_path"])
            shard_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "schema_version": SHARD_PAYLOAD_SCHEMA_VERSION,
                    "source": task["source"],
                    "preprocessing_spec": plan["preprocessing_spec"],
                    "events": [_event()],
                },
                shard_path,
            )
            _write_json(
                Path(task["status_path"]),
                {
                    "status": "complete",
                    "task_index": task["task_index"],
                    "source": task["source"],
                    "shard_path": task["shard_path"],
                    "num_events": 1,
                    "reused": False,
                },
            )
        return plan_path

    def test_finalizer_builds_two_ml_ready_lazy_caches(self):
        with TemporaryDirectory() as temporary_dir:
            plan_path = self._plan_with_worker_outputs(temporary_dir)
            summary = finalize_parallel_shard_plan(
                plan_path,
                expected_events_by_label={"2e": 2, "3e": 1},
                load_all_shards=True,
            )

            self.assertEqual([source["num_events"] for source in summary["sources"]], [2, 1])
            dataset_2e = ShardedECalTpadDataset(Path(temporary_dir) / "output/2e/events")
            dataset_3e = ShardedECalTpadDataset(Path(temporary_dir) / "output/3e/events")
            self.assertEqual(len(dataset_2e), 2)
            self.assertEqual(int(dataset_2e[1]["event_idx"]), 1)
            combined = MultiShardedECalTpadDataset(
                [
                    {
                        "electron_count": 2,
                        "source_label": "2e",
                        "cache_dir": dataset_2e.cache_dir,
                        "dataset": dataset_2e,
                    },
                    {
                        "electron_count": 3,
                        "source_label": "3e",
                        "cache_dir": dataset_3e.cache_dir,
                        "dataset": dataset_3e,
                    },
                ]
            )
            self.assertEqual(int(combined[2]["event_idx"]), 2)
            self.assertTrue(torch.equal(dataset_2e[0]["ecal_raw_energy"], torch.tensor([0.0, 9.0])))
            self.assertTrue(torch.equal(dataset_2e[0]["tpad_raw_pe"], torch.tensor([8.0])))
            self.assertTrue((Path(temporary_dir) / "output/preprocessing_summary.json").exists())

    def test_finalizer_rejects_missing_worker_status(self):
        with TemporaryDirectory() as temporary_dir:
            plan_path = self._plan_with_worker_outputs(temporary_dir, omit_task=1)
            with self.assertRaisesRegex(RuntimeError, "missing status"):
                finalize_parallel_shard_plan(plan_path)


if __name__ == "__main__":
    unittest.main()
