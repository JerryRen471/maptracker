import importlib.util
import pathlib
import unittest

import torch


MODULE_PATH = (
    pathlib.Path(__file__).parents[1]
    / "plugin"
    / "temporal_visibility.py"
)
SPEC = importlib.util.spec_from_file_location(
    "maptracker_temporal_visibility", MODULE_PATH)
TEMPORAL_VISIBILITY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TEMPORAL_VISIBILITY)


def identity_grid(height, width):
    ys = torch.arange(height, dtype=torch.float32)
    xs = torch.arange(width, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs)
    xx = (2.0 * xx + 1.0) / width - 1.0
    yy = (2.0 * yy + 1.0) / height - 1.0
    return torch.stack([xx, yy], dim=-1)


class TemporalVisibilityTest(unittest.TestCase):

    def test_warped_history_masks_extend_current_supervision(self):
        current_visible = torch.tensor([[
            [1.0, 0.0],
            [0.0, 0.0],
        ]])
        history_visible = [torch.tensor([[
            [0.0, 1.0],
            [0.0, 0.0],
        ]])]
        history_coords = [identity_grid(2, 2).unsqueeze(0)]

        warped_masks, warped_union = (
            TEMPORAL_VISIBILITY.warp_history_visibility_masks(
                history_visible, history_coords))
        supervision_mask = (
            TEMPORAL_VISIBILITY.build_temporal_supervision_mask(
                current_visible, warped_union))

        self.assertEqual(tuple(warped_masks.shape), (1, 1, 2, 2))
        torch.testing.assert_close(
            warped_union,
            torch.tensor([[
                [0.0, 1.0],
                [0.0, 0.0],
            ]]))
        torch.testing.assert_close(
            supervision_mask,
            torch.tensor([[
                [1.0, 1.0],
                [0.0, 0.0],
            ]]))

    def test_history_visibility_gates_warped_features(self):
        history_feat = torch.tensor([[[[
            [1.0, 2.0],
            [3.0, 4.0],
        ]]]])
        history_mask = torch.tensor([[[
            [1.0, 0.0],
            [0.0, 1.0],
        ]]])

        gated = TEMPORAL_VISIBILITY.gate_history_bev_features(
            history_feat, history_mask)

        torch.testing.assert_close(
            gated,
            torch.tensor([[[[
                [1.0, 0.0],
                [0.0, 4.0],
            ]]]]))


if __name__ == "__main__":
    unittest.main()
