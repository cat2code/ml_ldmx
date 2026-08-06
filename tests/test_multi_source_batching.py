import unittest

import torch

from ml_ldmx.datasets.ecal_tpad_shards import MultiShardedECalTpadDataset


class _SourceDataset:
    def __init__(self, num_events):
        self.num_events = int(num_events)

    def __len__(self):
        return self.num_events

    def __getitem__(self, index):
        return {"local_index": int(index)}

    def order_indices_for_access(self, indices, seed=None):
        indices = list(indices)
        if seed is None:
            return sorted(indices)
        generator = torch.Generator().manual_seed(int(seed))
        order = torch.randperm(len(indices), generator=generator).tolist()
        return [indices[idx] for idx in order]


def _combined(source_counts):
    return MultiShardedECalTpadDataset(
        [
            {
                "electron_count": source_idx + 2,
                "source_label": f"source-{source_idx}",
                "cache_dir": f"/unused/source-{source_idx}",
                "dataset": _SourceDataset(count),
            }
            for source_idx, count in enumerate(source_counts)
        ]
    )


class MultiSourceBatchingTest(unittest.TestCase):
    def test_scan_order_is_source_blocked_but_training_batches_are_mixed(self):
        dataset = _combined([101, 99])
        indices = list(range(len(dataset)))

        blocked = dataset.order_indices_for_access(indices, seed=7)
        blocked_sources = [dataset._source_idx_for_event(index) for index in blocked]
        self.assertEqual(
            sum(left != right for left, right in zip(blocked_sources, blocked_sources[1:])),
            1,
        )

        batches = dataset.balanced_batches_for_access(indices, batch_size=8, seed=7)
        flattened = [index for batch in batches for index in batch]
        self.assertEqual(sorted(flattened), indices)
        self.assertEqual(len(flattened), len(set(flattened)))

        for batch in batches:
            source_counts = [0, 0]
            for index in batch:
                source_counts[dataset._source_idx_for_event(index)] += 1
            self.assertGreater(source_counts[0], 0)
            self.assertGreater(source_counts[1], 0)
            self.assertLessEqual(abs(source_counts[0] - source_counts[1]), 2)

    def test_balanced_batches_are_deterministic_and_keep_source_local_order(self):
        dataset = _combined([17, 15])
        indices = list(range(len(dataset)))

        batches = dataset.balanced_batches_for_access(indices, batch_size=4, seed=19)
        repeated = dataset.balanced_batches_for_access(indices, batch_size=4, seed=19)
        different_seed = dataset.balanced_batches_for_access(indices, batch_size=4, seed=20)
        self.assertEqual(batches, repeated)
        self.assertNotEqual(batches, different_seed)

        flattened = [index for batch in batches for index in batch]
        for source_idx, source in enumerate(dataset.sources):
            actual_local = [
                index - dataset.offsets[source_idx]
                for index in flattened
                if dataset._source_idx_for_event(index) == source_idx
            ]
            source_global = [
                index
                for index in indices
                if dataset._source_idx_for_event(index) == source_idx
            ]
            expected_local = source["dataset"].order_indices_for_access(
                [index - dataset.offsets[source_idx] for index in source_global],
                seed=19 + source_idx + 1,
            )
            self.assertEqual(actual_local, expected_local)

    def test_balanced_batches_validate_batch_size_and_handle_one_source(self):
        dataset = _combined([5])
        with self.assertRaisesRegex(ValueError, "batch_size must be positive"):
            dataset.balanced_batches_for_access(range(5), batch_size=0, seed=1)

        batches = dataset.balanced_batches_for_access(range(5), batch_size=2, seed=1)
        self.assertEqual([len(batch) for batch in batches], [2, 2, 1])
        self.assertEqual(sorted(index for batch in batches for index in batch), list(range(5)))


if __name__ == "__main__":
    unittest.main()
