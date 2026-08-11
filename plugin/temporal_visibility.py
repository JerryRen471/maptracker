import torch
import torch.nn.functional as F


def _as_bhw(mask):
    if mask.dim() == 3:
        return mask
    if mask.dim() == 4 and mask.shape[1] == 1:
        return mask[:, 0]
    raise ValueError(
        "visibility mask must have shape [B, H, W] or [B, 1, H, W], "
        f"got {tuple(mask.shape)}")


def warp_history_visibility_masks(history_visible_masks, all_history_coord):
    """Warp cached visibility masks with the same grids used for history BEV.

    Args:
        history_visible_masks: list of [B, H, W] or [B, 1, H, W] masks.
        all_history_coord: list of length B. Each entry is [T, H, W, 2].

    Returns:
        warped_masks: [B, T, H, W] binary masks in the current frame.
        warped_union: [B, H, W] binary union across history.
    """
    if len(history_visible_masks) == 0:
        return None, None
    if len(all_history_coord) == 0:
        raise ValueError("all_history_coord is required when history masks exist")

    history_visible_masks = [_as_bhw(mask) for mask in history_visible_masks]
    num_history = len(history_visible_masks)
    batch_size = history_visible_masks[0].shape[0]
    if len(all_history_coord) != batch_size:
        raise ValueError(
            "all_history_coord length must match batch size, "
            f"got {len(all_history_coord)} and {batch_size}")

    warped_masks = []
    for b_i in range(batch_size):
        history_coord = all_history_coord[b_i]
        if history_coord.shape[0] != num_history:
            raise ValueError(
                "history coord length must match history mask length, "
                f"got {history_coord.shape[0]} and {num_history}")

        masks_i = torch.stack(
            [mask[b_i] for mask in history_visible_masks], dim=0)
        masks_i = masks_i[:, None].to(
            device=history_coord.device, dtype=torch.float32)
        history_coord = history_coord.to(
            device=masks_i.device, dtype=torch.float32)
        warped_i = F.grid_sample(
            masks_i,
            history_coord,
            mode="nearest",
            padding_mode="zeros",
            align_corners=False)
        warped_masks.append(warped_i[:, 0])

    warped_masks = torch.stack(warped_masks, dim=0)
    warped_masks = (warped_masks > 0).to(dtype=history_visible_masks[0].dtype)
    warped_union = (warped_masks.sum(dim=1) > 0).to(
        dtype=history_visible_masks[0].dtype)
    return warped_masks, warped_union


def build_temporal_supervision_mask(current_visible_mask,
                                    warped_history_visible_union):
    """Return current-or-history visibility for segmentation supervision."""
    current_visible_mask = _as_bhw(current_visible_mask)
    if warped_history_visible_union is None:
        return current_visible_mask

    warped_history_visible_union = _as_bhw(warped_history_visible_union)
    warped_history_visible_union = warped_history_visible_union.to(
        device=current_visible_mask.device, dtype=current_visible_mask.dtype)
    return torch.maximum(current_visible_mask, warped_history_visible_union)


def gate_history_bev_features(warped_history_bev, warped_history_visible_masks):
    """Zero out warped history BEV features outside geometric visibility."""
    if warped_history_visible_masks is None:
        return warped_history_bev
    if warped_history_bev.dim() != 5:
        raise ValueError(
            "warped_history_bev must have shape [B, T, C, H, W], "
            f"got {tuple(warped_history_bev.shape)}")
    if warped_history_visible_masks.dim() != 4:
        raise ValueError(
            "warped_history_visible_masks must have shape [B, T, H, W], "
            f"got {tuple(warped_history_visible_masks.shape)}")

    visible = warped_history_visible_masks[:, :, None].to(
        device=warped_history_bev.device, dtype=warped_history_bev.dtype)
    return warped_history_bev * visible
