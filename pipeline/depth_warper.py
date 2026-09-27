"""Depth-based forward splatting shared by all scene types."""

import torch

class DepthWarper:
    """Batched depth-image forward splatting used by the fast render backend."""

    def __init__(self):
        self.dtype = torch.float32
        self._grid_key = None
        self._grid = None

    def forward_warp(self, frame1, mask1, depth1, transformation1, transformation2,
                     intrinsic1, intrinsic2=None):
        b, c, h, w = frame1.shape
        if mask1 is None:
            mask1 = torch.ones((b, 1, h, w), device=frame1.device, dtype=frame1.dtype)
        if intrinsic2 is None:
            intrinsic2 = intrinsic1

        frame1 = frame1.to(self.dtype)
        mask1 = mask1.to(self.dtype)
        depth1 = depth1.to(self.dtype)
        transformation1 = transformation1.to(self.dtype)
        transformation2 = transformation2.to(self.dtype)
        intrinsic1 = intrinsic1.to(self.dtype)
        intrinsic2 = intrinsic2.to(self.dtype)

        trans_points = self.compute_transformed_points(
            depth1, transformation1, transformation2, intrinsic1, intrinsic2
        )
        trans_coordinates = trans_points[:, :, :, :2, 0] / trans_points[:, :, :, 2:3, 0]
        trans_depth = trans_points[:, :, :, 2, 0]
        grid = self.create_grid(b, h, w, trans_coordinates.device, trans_coordinates.dtype)
        flow12 = trans_coordinates.permute(0, 3, 1, 2) - grid
        return self.bilinear_splatting(frame1, mask1, trans_depth, flow12, None, is_image=True)

    def compute_transformed_points(self, depth1, tc1w, tc2w, intrinsic1, intrinsic2):
        b, _, h, w = depth1.shape
        tc2c1 = torch.bmm(tc2w, torch.linalg.inv(tc1w))

        x1d = torch.arange(0, w, device=depth1.device)[None]
        y1d = torch.arange(0, h, device=depth1.device)[:, None]
        x2d = x1d.repeat(h, 1)
        y2d = y1d.repeat(1, w)
        ones_2d = torch.ones((h, w), device=depth1.device)
        ones_4d = ones_2d[None, :, :, None, None].repeat(b, 1, 1, 1, 1)
        pos_vectors = torch.stack([x2d, y2d, ones_2d], dim=2)[None, :, :, :, None]

        intrinsic1_inv = torch.linalg.inv(intrinsic1)[:, None, None]
        intrinsic2_4d = intrinsic2[:, None, None]
        depth_4d = depth1[:, 0][:, :, :, None, None]
        trans_4d = tc2c1[:, None, None]

        unnormalized_pos = torch.matmul(intrinsic1_inv, pos_vectors)
        world_points = depth_4d * unnormalized_pos
        world_points_homo = torch.cat([world_points, ones_4d], dim=3)
        trans_world_homo = torch.matmul(trans_4d, world_points_homo)
        trans_world = trans_world_homo[:, :, :, :3]
        return torch.matmul(intrinsic2_4d, trans_world)

    def bilinear_splatting(self, frame1, mask1, depth1, flow12, flow12_mask, is_image=False):
        b, c, h, w = frame1.shape
        if flow12_mask is None:
            flow12_mask = torch.ones((b, 1, h, w), device=flow12.device, dtype=flow12.dtype)

        grid = self.create_grid(b, h, w, frame1.device, frame1.dtype)
        trans_pos = flow12 + grid
        trans_pos_offset = trans_pos + 1
        trans_pos_floor = torch.floor(trans_pos_offset).long()
        trans_pos_ceil = torch.ceil(trans_pos_offset).long()

        trans_pos_offset = torch.stack([
            torch.clamp(trans_pos_offset[:, 0], min=0, max=w + 1),
            torch.clamp(trans_pos_offset[:, 1], min=0, max=h + 1),
        ], dim=1)
        trans_pos_floor = torch.stack([
            torch.clamp(trans_pos_floor[:, 0], min=0, max=w + 1),
            torch.clamp(trans_pos_floor[:, 1], min=0, max=h + 1),
        ], dim=1)
        trans_pos_ceil = torch.stack([
            torch.clamp(trans_pos_ceil[:, 0], min=0, max=w + 1),
            torch.clamp(trans_pos_ceil[:, 1], min=0, max=h + 1),
        ], dim=1)

        prox_weight_nw = (1 - (trans_pos_offset[:, 1:2] - trans_pos_floor[:, 1:2])) * (
            1 - (trans_pos_offset[:, 0:1] - trans_pos_floor[:, 0:1])
        )
        prox_weight_sw = (1 - (trans_pos_ceil[:, 1:2] - trans_pos_offset[:, 1:2])) * (
            1 - (trans_pos_offset[:, 0:1] - trans_pos_floor[:, 0:1])
        )
        prox_weight_ne = (1 - (trans_pos_offset[:, 1:2] - trans_pos_floor[:, 1:2])) * (
            1 - (trans_pos_ceil[:, 0:1] - trans_pos_offset[:, 0:1])
        )
        prox_weight_se = (1 - (trans_pos_ceil[:, 1:2] - trans_pos_offset[:, 1:2])) * (
            1 - (trans_pos_ceil[:, 0:1] - trans_pos_offset[:, 0:1])
        )

        sat_depth = torch.clamp(depth1, min=0, max=1000)
        log_depth = torch.log(1 + sat_depth)
        depth_weights = torch.exp(log_depth / log_depth.max().clamp_min(1e-6) * 50)
        if depth1.dim() == 3:
            valid_mask = (depth1 >= 0).to(depth1).unsqueeze(1)
            depth_weights = depth_weights.unsqueeze(1)
        else:
            valid_mask = (depth1 >= 0).to(depth1)

        def make_weight(prox_weight):
            return torch.moveaxis(
                prox_weight * mask1 * flow12_mask * valid_mask / depth_weights,
                [0, 1, 2, 3],
                [0, 3, 1, 2],
            )

        weight_nw = make_weight(prox_weight_nw)
        weight_sw = make_weight(prox_weight_sw)
        weight_ne = make_weight(prox_weight_ne)
        weight_se = make_weight(prox_weight_se)

        warped_frame = torch.zeros((b, h + 2, w + 2, c), dtype=torch.float32, device=frame1.device)
        warped_weights = torch.zeros((b, h + 2, w + 2, 1), dtype=torch.float32, device=frame1.device)

        frame1_cl = torch.moveaxis(frame1, [0, 1, 2, 3], [0, 3, 1, 2])
        batch_indices = torch.arange(b, device=frame1.device)[:, None, None]
        warped_frame.index_put_(
            (batch_indices, trans_pos_floor[:, 1], trans_pos_floor[:, 0]),
            frame1_cl * weight_nw,
            accumulate=True,
        )
        warped_frame.index_put_(
            (batch_indices, trans_pos_ceil[:, 1], trans_pos_floor[:, 0]),
            frame1_cl * weight_sw,
            accumulate=True,
        )
        warped_frame.index_put_(
            (batch_indices, trans_pos_floor[:, 1], trans_pos_ceil[:, 0]),
            frame1_cl * weight_ne,
            accumulate=True,
        )
        warped_frame.index_put_(
            (batch_indices, trans_pos_ceil[:, 1], trans_pos_ceil[:, 0]),
            frame1_cl * weight_se,
            accumulate=True,
        )
        warped_weights.index_put_(
            (batch_indices, trans_pos_floor[:, 1], trans_pos_floor[:, 0]),
            weight_nw,
            accumulate=True,
        )
        warped_weights.index_put_(
            (batch_indices, trans_pos_ceil[:, 1], trans_pos_floor[:, 0]),
            weight_sw,
            accumulate=True,
        )
        warped_weights.index_put_(
            (batch_indices, trans_pos_floor[:, 1], trans_pos_ceil[:, 0]),
            weight_ne,
            accumulate=True,
        )
        warped_weights.index_put_(
            (batch_indices, trans_pos_ceil[:, 1], trans_pos_ceil[:, 0]),
            weight_se,
            accumulate=True,
        )

        warped_frame_cf = torch.moveaxis(warped_frame, [0, 1, 2, 3], [0, 2, 3, 1])
        warped_weights_cf = torch.moveaxis(warped_weights, [0, 1, 2, 3], [0, 2, 3, 1])
        cropped_frame = warped_frame_cf[:, :, 1:-1, 1:-1]
        cropped_weights = warped_weights_cf[:, :, 1:-1, 1:-1]

        known_mask = cropped_weights > 0
        zero_value = -1 if is_image else 0
        warped_frame2 = torch.where(
            known_mask,
            cropped_frame / cropped_weights,
            torch.tensor(zero_value, dtype=frame1.dtype, device=frame1.device),
        )
        if is_image:
            warped_frame2 = torch.clamp(warped_frame2, min=-1, max=1)
        return warped_frame2, known_mask.to(frame1)

    def create_grid(self, b, h, w, device=None, dtype=torch.float32):
        key = (h, w, str(device), dtype)
        if self._grid_key != key:
            y, x = torch.meshgrid(torch.arange(h, device=device, dtype=dtype),
                                  torch.arange(w, device=device, dtype=dtype), indexing="ij")
            self._grid = torch.stack([x, y])[None]
            self._grid_key = key
        return self._grid.expand(b, -1, -1, -1)
