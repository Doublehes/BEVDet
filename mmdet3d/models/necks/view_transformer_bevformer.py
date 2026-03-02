# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer
from mmcv.runner import BaseModule, force_fp32
from mmcv.utils import TORCH_VERSION, digit_version
from torch.cuda.amp.autocast_mode import autocast
from torch.utils.checkpoint import checkpoint
import numpy as np

from ..builder import NECKS

from .spatial_cross_attention import SpatialCrossAttention


class FFN(BaseModule):
    def __init__(self, d_model: int, d_ffn: int, drop_prob: float = 0.1):
        super().__init__()
        # FFN核心层
        self.linear1 = nn.Linear(d_model, d_ffn)  # 升维：d_model -> d_ffn
        self.activation = F.relu  # ReLU激活函数
        self.dropout2 = nn.Dropout(drop_prob)     # 激活后的Dropout
        self.linear2 = nn.Linear(d_ffn, d_model)  # 降维：d_ffn -> d_model
        self.dropout3 = nn.Dropout(drop_prob)     # 残差相加前的Dropout
        self.norm2 = nn.LayerNorm(d_model)        # 层归一化

    def forward(self, src):
        # 1. FFN核心计算：线性1 -> ReLU -> Dropout2 -> 线性2
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        # 2. 残差连接：原始输入 + FFN输出（先Dropout3）
        src = src + self.dropout3(src2)
        # 3. 层归一化（Post-LN模式，Transformer原论文方案）
        src = self.norm2(src)
        return src


class AttentionLayer(BaseModule):
    def __init__(self, embed_dims, num_cams, deformable_attention):
        super(AttentionLayer, self).__init__()
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.deformable_attention = SpatialCrossAttention(
            embed_dims=embed_dims,
            num_cams=num_cams,
            deformable_attention=deformable_attention)
        self.norm = nn.LayerNorm(embed_dims)
        self.ffn = FFN(embed_dims, embed_dims * 4, drop_prob=0.1)

    def forward(self, query, key, value, spatial_shapes, level_start_index,
                reference_points_cam, bev_mask):
        output = self.deformable_attention(
            query=query,
            key=key,
            value=value,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reference_points_cam=reference_points_cam,
            bev_mask=bev_mask)
        output = self.norm(output)
        output = self.ffn(output)
        return output


@NECKS.register_module()
class BEVFormerViewTransformer(BaseModule):
    r"""Lift-Splat-Shoot view transformer with BEVPoolv2 implementation.

    Please refer to the `paper <https://arxiv.org/abs/2008.05711>`_ and
        `paper <https://arxiv.org/abs/2211.17111>`

    Args:
        grid_config (dict): Config of grid alone each axis in format of
            (lower_bound, upper_bound, interval). axis in {x,y,z,depth}.
        input_size (tuple(int)): Size of input images in format of (height,
            width).
        downsample (int): Down sample factor from the input size to the feature
            size.
        in_channels (int): Channels of input feature.
        out_channels (int): Channels of transformed feature.
        accelerate (bool): Whether the view transformation is conducted with
            acceleration. Note: the intrinsic and extrinsic of cameras should
            be constant when 'accelerate' is set true.
        sid (bool): Whether to use Spacing Increasing Discretization (SID)
            depth distribution as `STS: Surround-view Temporal Stereo for
            Multi-view 3D Detection`.
        collapse_z (bool): Whether to collapse in z direction.
    """

    def __init__(
        self,
        grid_config,
        input_size,
        downsample=16,
        in_channels=512,
        out_channels=64,
        accelerate=False,
        sid=False,
        collapse_z=True,
        with_cp=False,
        with_depth_from_lidar=False,
    ):
        super(BEVFormerViewTransformer, self).__init__()
        self.with_cp = with_cp
        self.grid_config = grid_config
        self.downsample = downsample
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.initial_flag = True

        self.bev_h = int((grid_config['y'][1] - grid_config['y'][0]) / grid_config['y'][2])
        self.bev_w = int((grid_config['x'][1] - grid_config['x'][0]) / grid_config['x'][2])
        self.bev_embedding = nn.Embedding(self.bev_h * self.bev_w, out_channels)

        self.channel_mapper = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.n_layers = 3
        self.bev_layers = nn.ModuleList([
            AttentionLayer(embed_dims=out_channels, num_cams=6,
                           deformable_attention=dict(type='MSDeformableAttention3D',
                                                     embed_dims=out_channels,
                                                     num_points=8,
                                                     num_levels=1))
            for _ in range(self.n_layers)
        ])

    def get_lidar_coor(self, sensor2ego, ego2global, cam2imgs, post_rots, post_trans,
                       bda):
        """Calculate the locations of the frustum points in the lidar
        coordinate system.

        Args:
            rots (torch.Tensor): Rotation from camera coordinate system to
                lidar coordinate system in shape (B, N_cams, 3, 3).
            trans (torch.Tensor): Translation from camera coordinate system to
                lidar coordinate system in shape (B, N_cams, 3).
            cam2imgs (torch.Tensor): Camera intrinsic matrixes in shape
                (B, N_cams, 3, 3).
            post_rots (torch.Tensor): Rotation in camera coordinate system in
                shape (B, N_cams, 3, 3). It is derived from the image view
                augmentation.
            post_trans (torch.Tensor): Translation in camera coordinate system
                derived from image view augmentation in shape (B, N_cams, 3).

        Returns:
            torch.tensor: Point coordinates in shape
                (B, N_cams, D, ownsample, 3)
        """
        B, N, _, _ = sensor2ego.shape

        # post-transformation
        # B x N x D x H x W x 3
        points = self.frustum.to(sensor2ego) - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3)\
            .matmul(points.unsqueeze(-1))

        # cam_to_ego
        points = torch.cat(
            (points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]), 5)
        combine = sensor2ego[:,:,:3,:3].matmul(torch.inverse(cam2imgs))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += sensor2ego[:,:,:3, 3].view(B, N, 1, 1, 1, 3)
        points = bda[:, :3, :3].view(B, 1, 1, 1, 1, 3, 3).matmul(
            points.unsqueeze(-1)).squeeze(-1)
        points += bda[:, :3, 3].view(B, 1, 1, 1, 1, 3)
        return points

    @staticmethod
    def get_reference_points(H, W, Z=8, num_points_in_pillar=4, dim='3d', bs=1, device='cuda', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == '3d':
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
            ref_3d = torch.stack((xs, ys, zs), -1)
            # import pudb;pudb.set_trace()
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d

    # This function must use fp32!!!
    @force_fp32(apply_to=('reference_points', 'cood_trans'))
    def point_sampling(self, reference_points, cood_trans, img_h, img_w):
        # NOTE: close tf32 here.
        allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        sensor2ego, ego2global, cam2imgs, post_rots, post_trans, bda = cood_trans
        # import pudb;pudb.set_trace()

        # lidar2img = []
        # for img_meta in img_metas:
        #     lidar2img.append(img_meta['lidar2img'])
        # lidar2img = np.asarray(lidar2img)
        # lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4)

        ego2cam = sensor2ego.inverse()
        cam2imgs_new = torch.eye(4, device=cam2imgs.device).view(1, 1, 4, 4).repeat(cam2imgs.shape[0], cam2imgs.shape[1], 1, 1)
        cam2imgs_new[:, :, :3, :3] = cam2imgs
        bda_batch = bda.view(bda.shape[0], 1, 4, 4).repeat(1, cam2imgs.shape[1], 1, 1)
        ego2img = cam2imgs_new @ ego2cam @ bda_batch.inverse()
        post_T = torch.eye(4, device=post_rots.device).view(1, 1, 4, 4).repeat(post_rots.shape[0], post_rots.shape[1], 1, 1)
        post_T[:, :, :3, :3] = post_rots
        post_T[:, :, :3, 3] = post_trans

        lidar2img = post_T @ ego2img

        reference_points = reference_points.clone()

        reference_points[..., 0:1] = reference_points[..., 0:1] * (self.grid_config['x'][1] - self.grid_config['x'][0]) + self.grid_config['x'][0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * (self.grid_config['y'][1] - self.grid_config['y'][0]) + self.grid_config['y'][0]
        reference_points[..., 2:3] = reference_points[..., 2:3] * (self.grid_config['z'][1] - self.grid_config['z'][0]) + self.grid_config['z'][0]

        reference_points = torch.cat(
            (reference_points, torch.ones_like(reference_points[..., :1])), -1)

        reference_points = reference_points.permute(1, 0, 2, 3)
        D, B, num_query = reference_points.size()[:3]
        num_cam = lidar2img.size(1)

        reference_points = reference_points.view(
            D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)

        lidar2img = lidar2img.view(
            1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)

        reference_points_cam = torch.matmul(lidar2img.to(torch.float32),
                                            reference_points.to(torch.float32)).squeeze(-1)
        eps = 1e-5

        bev_mask = (reference_points_cam[..., 2:3] > eps)
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)

        reference_points_cam[..., 0] /= img_w
        reference_points_cam[..., 1] /= img_h

        bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0)
                    & (reference_points_cam[..., 1:2] < 1.0)
                    & (reference_points_cam[..., 0:1] < 1.0)
                    & (reference_points_cam[..., 0:1] > 0.0))
        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            bev_mask = torch.nan_to_num(bev_mask)
        else:
            bev_mask = bev_mask.new_tensor(
                np.nan_to_num(bev_mask.cpu().numpy()))

        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)

        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32

        return reference_points_cam, bev_mask


    def forward(self, input, depth_from_lidar=None):
        """Transform image-view feature into bird-eye-view feature.

        Args:
            input (list(torch.tensor)): of (image-view feature, rots, trans,
                intrins, post_rots, post_trans)

        Returns:
            torch.tensor: Bird-eye-view feature in shape (B, C, H_BEV, W_BEV)
        """
        x = input[0]
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        x = self.channel_mapper(x)
        x = x.view(B, N, self.out_channels, H, W)


        downsample_factor = 16
        img_h, img_w = downsample_factor * H, downsample_factor * W

        # import pudb;pudb.set_trace()

        bev_queries = self.bev_embedding.weight.to(x.dtype)
        bev_queries = bev_queries.unsqueeze(0).repeat(B, 1, 1)  # (B, bev_h*bev_w, embed_dims)
        ref_3d = self.get_reference_points(self.bev_h, self.bev_w, 8, 4, 
                                           dim='3d', bs=B,  device=x.device, dtype=x.dtype)
        reference_points_cam, bev_mask = self.point_sampling(ref_3d, input[1:7], img_h, img_w)
    
        feat_flatten = []
        spatial_shapes = []
        mlvl_feats = [x]
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=x.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_flatten = feat_flatten.permute(0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)

        for layer in self.bev_layers:
            bev_queries = layer(bev_queries, feat_flatten, feat_flatten, 
                                spatial_shapes=spatial_shapes, 
                                level_start_index=level_start_index, 
                                reference_points_cam=reference_points_cam, 
                                bev_mask=bev_mask)

        bev_feat = bev_queries.permute(0, 2, 1).view(B, self.out_channels, self.bev_h, self.bev_w)
        # import pudb;pudb.set_trace()
        return bev_feat, None


