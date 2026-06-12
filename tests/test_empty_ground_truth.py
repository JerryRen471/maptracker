import pathlib
import sys
import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

from plugin.models.assigner.assigner import HungarianLinesAssigner
from plugin.models.mapers.MapTracker import MapTracker


class EmptyGroundTruthTest(unittest.TestCase):

    def test_batch_data_builds_empty_line_tensor(self):
        tracker = MapTracker.__new__(MapTracker)
        nn.Module.__init__(tracker)
        tracker.head = SimpleNamespace(num_points=20, coord_dim=2)
        tracker.num_decoder_layers = 2

        gts, _, _, valid_idx, _ = tracker.batch_data(
            vectors=[{0: [], 1: [], 2: []}],
            imgs=torch.zeros(1, 1),
            img_metas=[{}],
            device=torch.device("cpu"),
        )

        self.assertEqual(valid_idx, [])
        self.assertEqual(len(gts), 2)
        self.assertEqual(gts[0]["labels"][0].shape, (0,))
        self.assertEqual(gts[0]["lines"][0].shape, (0, 40))

    def test_assigner_returns_matching_cost_for_empty_ground_truth(self):
        assigner = HungarianLinesAssigner.__new__(HungarianLinesAssigner)
        predictions = {
            "lines": torch.zeros(4, 40),
            "scores": torch.zeros(4, 3),
        }
        ground_truth = {
            "lines": torch.zeros(0, 40),
            "labels": torch.zeros(0, dtype=torch.long),
        }

        result, permutation, matching_cost = assigner.assign(
            predictions, ground_truth)

        self.assertEqual(result.num_gts, 0)
        self.assertTrue(torch.equal(
            result.gt_inds, torch.zeros(4, dtype=torch.long)))
        self.assertIsNone(permutation)
        self.assertEqual(matching_cost.shape, (0,))


if __name__ == "__main__":
    unittest.main()
