"""
plugin/datasets/waymo_dataset.py

Waymo Open Dataset for MapTracker.

Adapted from `plugin/datasets/argo_dataset.py` with two structural changes:
1. No id2map / AV2MapExtractor — polylines are baked per-frame in samples.
2. cams reduced to 5 (FRONT, FRONT_LEFT, FRONT_RIGHT, SIDE_LEFT, SIDE_RIGHT).
3. get_sample reads `gt_polylines` + `gt_polyline_labels` directly from each
   sample dict and builds map_geoms (no global map JSON file involved).

cat2id (must match config):
    {'ped_crossing': 0, 'divider': 1, 'boundary': 2}
"""
import os
import pickle
from time import time

import mmcv
import numpy as np
from mmdet.datasets import DATASETS
from shapely.geometry import LineString

from .base_dataset import BaseMapDataset
from .visualize.renderer import Renderer


@DATASETS.register_module()
class WaymoMapDataset(BaseMapDataset):
    """Waymo dataset for MapTracker.

    The pkl is expected to be the output of pack_waymo_for_maptracker.py,
    with structure: dict(samples=[...], id2map={}).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # AV2 renderer profile works for us (same 3 categories,
        # only colormap differs cosmetically). Swap to a 'waymo' profile
        # later if you add one to renderer.py.
        self.renderer = Renderer(self.cat2id, self.roi_size, 'av2')

    # ------------------------------------------------------------------
    # Annotation loading
    # ------------------------------------------------------------------
    def load_annotations(self, ann_file):
        start = time()
        ann = mmcv.load(ann_file)

        # No id2map for Waymo — polylines live inside each sample.
        # We still set self.id2map = {} so downstream code that touches it
        # (defensively) does not blow up.
        self.id2map = ann.get('id2map', {})
        samples = ann['samples']

        # Stable ordering: sort by (scene, frame_idx).
        # pack_waymo_for_maptracker.py already does this, but be defensive
        # in case the user post-processed the pkl.
        samples = sorted(samples,
                         key=lambda s: (s['scene_name'],
                                        int(s.get('frame_idx', 0))))
        samples = samples[::self.interval]

        # Make sure samples of the same scene are contiguous (required by
        # base_dataset._set_sequence_group_flag for sequence splitting).
        scene_name2idx = {}
        for idx, s in enumerate(samples):
            scene_name2idx.setdefault(s['scene_name'], []).append(idx)

        rearranged = []
        for scene_name in scene_name2idx:
            for i in scene_name2idx[scene_name]:
                rearranged.append(samples[i])
        samples = rearranged

        print(f'[WaymoMapDataset] loaded {len(samples)} samples from '
              f'{len(scene_name2idx)} scenes in {(time() - start):.2f}s')
        self.samples = samples

    def load_matching(self, matching_file):
        with open(matching_file, 'rb') as f:
            data = pickle.load(f)
        total = sum(len(info['sample_ids']) for info in data.values())
        assert total == len(self.samples), (
            f'matching meta has {total} entries, '
            f'but dataset has {len(self.samples)} samples')
        self.matching_meta = data
        print(f'[WaymoMapDataset] loaded matching meta for {len(data)} scenes')

    # ------------------------------------------------------------------
    # Sample reading — the part that replaces AV2's map_extractor lookup
    # ------------------------------------------------------------------
    def get_sample(self, idx):
        sample = self.samples[idx]

        # Build map_geoms by bucketing per-frame polylines by label.
        # MapTracker expects: {cat_id: list of geometries}.
        # All three categories use LineString here (matching AV2's
        # convention as commented in argo_dataset.py).
        map_label2geom = {cid: [] for cid in self.cat2id.values()}
        for poly, lbl in zip(sample['gt_polylines'],
                             sample['gt_polyline_labels']):
            arr = np.asarray(poly, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[0] < 2:
                continue
            map_label2geom[int(lbl)].append(LineString(arr))

        # ego -> image projection per camera.
        # extrinsics in our packed pkl is ego2cam (4x4).
        ego2img_rts = []
        for c in sample['cams'].values():
            ext = np.asarray(c['extrinsics'], dtype=np.float64)
            intr = np.asarray(c['intrinsics'], dtype=np.float64)
            viewpad = np.eye(4, dtype=np.float64)
            viewpad[:intr.shape[0], :intr.shape[1]] = intr
            ego2img_rts.append(viewpad @ ext)

        input_dict = {
            'token': sample['token'],
            'img_filenames': [c['img_fpath'] for c in sample['cams'].values()],
            'cam_intrinsics': [np.asarray(c['intrinsics'])
                               for c in sample['cams'].values()],
            'cam_extrinsics': [np.asarray(c['extrinsics'])
                               for c in sample['cams'].values()],
            'ego2img': ego2img_rts,
            'map_geoms': map_label2geom,
            'ego2global_translation': np.asarray(sample['e2g_translation']),
            'ego2global_rotation': np.asarray(sample['e2g_rotation']).tolist(),
            'sample_idx': sample['modified_sample_idx'],
            'scene_name': sample['scene_name'],
            'lidar_path': sample.get('lidar_fpath', ''),
        }
        return input_dict
