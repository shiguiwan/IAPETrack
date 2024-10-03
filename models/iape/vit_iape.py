import math
import logging
from functools import partial
from collections import OrderedDict
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import Mlp, DropPath
from timm.models.layers import to_2tuple

from lib.models.layers.patch_embed import PatchEmbed
from lib.models.layers.fusion import Cattention, _get_activation_fn
from .utils import combine_tokens, recover_tokens
from .vit import VisionTransformer, Block, Attention
#from lib.models.layers.attn import Attention
#from ..layers.attn_blocks import FSBlock
import torchvision
from lib.utils.box_ops import box_xywh_to_xyxy

_logger = logging.getLogger(__name__)


class AttentionOU(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias, attn_drop=0., proj_drop=0., divide=False, gauss=False, early=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # self.q = nn.Linear(dim, dim, bias=qkv_bias)
        # self.k = nn.Linear(dim, dim, bias=qkv_bias)
        # self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.divide = divide
        self.gauss = gauss
        self.early = early
        if self.divide:
            if not self.early:
                self.divide_global_transform = nn.Sequential(
                    nn.Linear(dim, 384)
                )
                self.divide_local_transform = nn.Sequential(
                    nn.Linear(dim, 384)
                )
            self.divide_predict = nn.Sequential(
                nn.Linear(dim * 2, 384) if self.early else nn.Identity(),
                nn.GELU(),
                nn.Linear(384, 192),
                nn.GELU(),
                nn.Linear(192, 2),
                nn.Identity() if self.gauss else nn.LogSoftmax(dim=-1)
            )
            if self.gauss:
                self.divide_gaussian_filter = nn.Conv2d(2, 2, kernel_size=5, stride=1, padding=2)
                self.init_gaussian_filter()
                self.divide_gaussian_filter.requires_grad = False

    def init_gaussian_filter(self, k_size=5, sigma=1.0):
        center = k_size // 2
        x, y = np.mgrid[0 - center: k_size - center, 0 - center: k_size - center]
        gaussian_kernel = 1 / (2 * np.pi * sigma) * np.exp(-(np.square(x) + np.square(y)) / (2 * np.square(sigma)))
        self.divide_gaussian_filter.weight.data = torch.Tensor(gaussian_kernel).to(
            self.divide_gaussian_filter.weight.device).unsqueeze(0).unsqueeze(0).repeat(2, 2, 1, 1)
        self.divide_gaussian_filter.bias.data.zero_()

    def softmax_with_policy(self, attn, policy, eps=1e-6):
        B, H, N, _ = attn.size()
        group1 = policy[:, :, 0].reshape(B, 1, N, 1) @ policy[:, :, 0].reshape(B, 1, 1, N)
        group2 = policy[:, :, 1].reshape(B, 1, N, 1) @ policy[:, :, 1].reshape(B, 1, 1, N)
        group3 = policy[:, :, 0].reshape(B, 1, N, 1) @ policy[:, :, 2].reshape(B, 1, 1, N) + \
                 policy[:, :, 1].reshape(B, 1, N, 1) @ policy[:, :, 2].reshape(B, 1, 1, N) + \
                 policy[:, :, 2].reshape(B, 1, N, 1) @ policy[:, :, 2].reshape(B, 1, 1, N) + \
                 policy[:, :, 2].reshape(B, 1, N, 1) @ policy[:, :, 0].reshape(B, 1, 1, N) + \
                 policy[:, :, 2].reshape(B, 1, N, 1) @ policy[:, :, 1].reshape(B, 1, 1, N)
        attn_policy = group1 + group2 + group3
        eye = torch.eye(N, dtype=attn_policy.dtype, device=attn_policy.device).view(1, 1, N, N)
        attn_policy = attn_policy + (1.0 - attn_policy) * eye #1,1,321,321
        attn_policy = torch.cat(
            [attn_policy[:, :, :, 0].unsqueeze(3), attn_policy[:, :, :, 1:65], attn_policy[:, :, :, 1:]], dim=3)  ###新增
        # if torch.isnan(attn_policy).any():
        #     print('x:', attn_policy)
        #     raise ValueError('ERROR: attn_policy. softmax_with_policy. fusion. network outputs is NAN! stop training')
        # For stable training
        max_att, _ = torch.max(attn, dim=-1, keepdim=True)
        # if torch.isnan(max_att).any():
        #     print('max_att:', max_att)
        #     raise ValueError('ERROR: max_att. softmaxwithpolicy. network outputs is NAN! stop training')
        attn = attn - max_att ###1,12,321,385
        # if torch.isnan(attn).any():
        #     print('attn:', attn)
        #     raise ValueError('ERROR: attn1. softmaxwithpolicy. network outputs is NAN! stop training')
        attn = attn.to(torch.float32).exp_() * attn_policy.to(torch.float32)
        # if torch.isnan(attn).any():
        #     print('attn:', attn)
        #     raise ValueError('ERROR: attn2. softmaxwithpolicy. network outputs is NAN! stop training')
        attn = (attn + eps / N) / (attn.sum(dim=-1, keepdim=True) + eps)
        # if torch.isnan(attn).any():
        #     print('attn:', attn)
        #     raise ValueError('ERROR: attn3. softmaxwithpolicy. network outputs is NAN! stop training')
        return attn.type_as(max_att)

    def attn_in_group(self, q, k, v):
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(1, -1, self.dim)
        return x

    #def forward(self, x_q, x_k, x_v, template_mask, search_feat_len, token_len, tgt_type=None, attn_masking=True, threshold=0., ratio=0.):
    def forward(self, x, template_mask, search_feat_len, token_len, tgt_type=None, attn_masking=True, threshold=0., ratio=0.):
        # B, N_q, C = x_q.shape #B,385,768
        # _, N_v, _ = x_v.shape
        B, N_v, C = x.shape # B,385,768
        template_feat_len = (N_v-1-search_feat_len)//2
        # if ind == 0:
        #     N_q = N_v
        # else:
        #     N_q = N_v - template_feat_len
        N_q = N_v - template_feat_len
        # B,321,768
        decision = None
        assert not (tgt_type is None and self.early), 'conflict in implementation'
        if self.divide:
            if tgt_type == 'allmax':
                #tgt_rep = x_q[:, token_len:-search_feat_len] #B,64,768
                tgt_rep = x[:, template_feat_len+token_len:template_feat_len*2+token_len]  # B,64,768
                tgt_rep = F.adaptive_max_pool1d(tgt_rep.transpose(1, 2), output_size=1).transpose(1, 2) #B,1,768
            elif tgt_type == 'allavg':
                # tgt_rep = x_q[:, token_len:-search_feat_len] #B,64,768
                tgt_rep = x[:, template_feat_len + token_len:template_feat_len * 2 + token_len]
                tgt_rep = F.adaptive_avg_pool1d(tgt_rep.transpose(1, 2), output_size=1).transpose(1, 2)
            elif tgt_type == 'roimax':
                #tgt_rep = x_q[:, token_len:-search_feat_len] * template_mask.unsqueeze(-1)
                tgt_rep = x[:, template_feat_len + token_len:template_feat_len * 2 + token_len] * template_mask.unsqueeze(-1)
                tgt_rep = F.adaptive_max_pool1d(tgt_rep.transpose(1, 2), output_size=1).transpose(1, 2)
            elif tgt_type == 'roiavg':
                # tgt_rep = x_q[:, token_len:-search_feat_len] * template_mask.unsqueeze(-1)
                tgt_rep = x[:, template_feat_len + token_len:template_feat_len * 2 + token_len] * template_mask.unsqueeze(-1)
                tgt_rep = F.adaptive_avg_pool1d(tgt_rep.transpose(1, 2), output_size=1).transpose(1, 2)
            else:
                raise NotImplementedError
            if self.early:
                tgt_rep = tgt_rep.expand(-1, search_feat_len, -1) #B,256,768
                divide_prediction = self.divide_predict(torch.cat((x[:, -search_feat_len:], tgt_rep), dim=-1)) #B,256,2
            else:
                local_transforms = self.divide_local_transform(x[:, -search_feat_len:])
                if tgt_type is None:
                    divide_prediction = self.divide_predict(local_transforms)
                else:
                    global_transforms = self.divide_global_transform(tgt_rep)
                    divide_prediction = self.divide_predict(global_transforms + local_transforms)

            if self.gauss:
                # Smooth the selection in local neighborhood
                size = int(search_feat_len ** 0.5)
                divide_prediction = self.divide_gaussian_filter(
                    divide_prediction.transpose(1, 2).reshape(B, 2, size, size))
                divide_prediction = F.log_softmax(divide_prediction.reshape(B, 2, -1).transpose(1, 2), dim=-1)

            if self.training:
                # During training
                decision = F.gumbel_softmax(divide_prediction, hard=True) #B,256,2
            else:
                # During inference
                if threshold:
                    # Manual rank based selection
                    decision_rank = (F.softmax(divide_prediction, dim=-1)[:, :, 0] < threshold).long()
                else:
                    # Auto rank based selection
                    decision_rank = torch.argsort(divide_prediction, dim=-1, descending=True)[:, :, 0]

                decision = F.one_hot(decision_rank, num_classes=2)

                if ratio:
                    # Ratio based selection
                    K = int(search_feat_len * ratio)
                    _, indices = torch.topk(divide_prediction[:, :, 1], k=K, dim=-1)
                    force_back = torch.zeros(B, K, dtype=decision.dtype, device=decision.device)
                    force_over = torch.ones(B, K, dtype=decision.dtype, device=decision.device)
                    decision[:, :, 0] = torch.scatter(decision[:, :, 0], -1, indices, force_back)
                    decision[:, :, 1] = torch.scatter(decision[:, :, 1], -1, indices, force_over)

            blank_policy = torch.zeros(B, search_feat_len, 1, dtype=divide_prediction.dtype,
                                       device=divide_prediction.device) #B,256,1
            template_policy = torch.zeros(B, N_q - search_feat_len-token_len, 3, dtype=divide_prediction.dtype,
                                          device=divide_prediction.device) #b,64,3
            template_policy[:, :, 0] = 1
            token_policy = torch.zeros(B, token_len, 3,  dtype=divide_prediction.dtype, device=divide_prediction.device) #####
            token_policy[:, :, -1] = 1
            #policy = torch.cat([token_policy, template_policy, torch.cat([blank_policy, decision], dim=-1)], dim=1)
            policy = torch.cat([token_policy, template_policy, torch.cat([blank_policy, decision], dim=-1)], dim=1)

            qkv = self.qkv(x).reshape(B, N_v, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            # q = self.q(x_q).reshape(B, N_q, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            # k = self.k(x_k).reshape(B, N_v, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            # v = self.v(x_v).reshape(B, N_v, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            if torch.isnan(qkv).any():
                print('qkv:', qkv)
                raise ValueError('ERROR: qkv. fusion. network outputs is NAN! stop training')
            q_o, k, v = qkv[0], qkv[1], qkv[2]  # B,12,320,64 Make torchscript happy (cannot use tensor as tuple)
            # if ind == 0:#
            #     q =q_o
            # else:
            #     q = torch.cat([q_o[:, :, 0].unsqueeze(2), q_o[:, :, template_feat_len + token_len:]], dim=2)
            q = torch.cat([q_o[:, :, 0].unsqueeze(2), q_o[:, :, template_feat_len + token_len:]], dim=2)
            if not attn_masking and not self.training:
                # Conduct three categories separately
                num_group1 = policy[:, :, 0].sum()
                num_group2 = policy[:, :, 1].sum()
                num_group3 = policy[:, :, 2].sum()
                _, E_T_ind = torch.topk(policy[:, :, 0], k=int(num_group1.item()), sorted=False)
                _, E_S_ind = torch.topk(policy[:, :, 1], k=int(num_group2.item()), sorted=False)
                _, E_A_ind = torch.topk(policy[:, :, 2], k=int(num_group3.item()), sorted=False)
                E_T_indices = E_T_ind.unsqueeze(1).unsqueeze(-1).repeat(1, self.num_heads, 1, self.head_dim)
                E_S_indices = E_S_ind.unsqueeze(1).unsqueeze(-1).repeat(1, self.num_heads, 1, self.head_dim)
                E_A_indices = E_A_ind.unsqueeze(1).unsqueeze(-1).repeat(1, self.num_heads, 1, self.head_dim)
                E_T_q = torch.gather(q, 2, E_T_indices)
                E_S_q = torch.gather(q, 2, E_S_indices)
                E_A_q = torch.gather(q, 2, E_A_indices)
                E_T_k = torch.gather(k, 2, torch.cat((E_T_indices, E_A_indices), dim=2))
                E_S_k = torch.gather(k, 2, torch.cat((E_S_indices, E_A_indices), dim=2))
                E_A_k = k
                E_T_v = torch.gather(v, 2, torch.cat((E_T_indices, E_A_indices), dim=2))
                E_S_v = torch.gather(v, 2, torch.cat((E_S_indices, E_A_indices), dim=2))
                E_A_v = v
                E_T_output = self.attn_in_group(E_T_q, E_T_k, E_T_v)
                E_S_output = self.attn_in_group(E_S_q, E_S_k, E_S_v)
                E_A_output = self.attn_in_group(E_A_q, E_A_k, E_A_v)

                x = torch.zeros_like(x, dtype=x.dtype, device=x.device)
                x = torch.scatter(x, 1, E_T_ind.unsqueeze(-1).repeat(1, 1, self.dim), E_T_output)
                x = torch.scatter(x, 1, E_S_ind.unsqueeze(-1).repeat(1, 1, self.dim), E_S_output)
                x = torch.scatter(x, 1, E_A_ind.unsqueeze(-1).repeat(1, 1, self.dim), E_A_output)
                x = self.proj(x)
                if torch.isnan(x).any():
                    print('x:', qkv)
                    raise ValueError('ERROR: x. fusion. network outputs is NAN! stop training')
            else:
                # Conduct three categories together
                attn = (q @ k.transpose(-2, -1)) * self.scale
                # if torch.isnan(attn).any():
                #     print('attn:', attn)
                #     raise ValueError('ERROR: attn1. fusion. network outputs is NAN! stop training')
                attn = self.softmax_with_policy(attn, policy)
                # if torch.isnan(attn).any():
                #     print('attn:', attn)
                #     print('policy:', policy)
                #     raise ValueError('ERROR: attn2. fusion. network outputs is NAN! stop training')
                attn = self.attn_drop(attn)
                # if torch.isnan(attn).any():
                #     print('attn:', attn)
                #     raise ValueError('ERROR: attn3. fusion. network outputs is NAN! stop training')
                x = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
                # if torch.isnan(x).any():
                #     print('x:', x)
                #     raise ValueError('ERROR: x1. attention. fusion. network outputs is NAN! stop training')
                x = self.proj(x)
                # if torch.isnan(x).any():
                #     print('x:', x)
                #     raise ValueError('ERROR: x2. attention. fusion. network outputs is NAN! stop training')
                x = self.proj_drop(x)
                # if torch.isnan(x).any():
                #     print('x:', x)
                #     raise ValueError('ERROR: x3. attention. fusion. network outputs is NAN! stop training')
        else:
            qkv = self.qkv(x).reshape(B, N_v, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            # q = self.q(x_q).reshape(B, N_q, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            # k = self.k(x_k).reshape(B, N_v, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            # v = self.v(x_v).reshape(B, N_v, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            if torch.isnan(qkv).any():
            # if torch.isnan(qkv).any():
                print('qkv:', qkv)
                raise ValueError('ERROR: qkv. unfusion. network outputs is NAN! stop training')
            q_o, k, v = qkv[0], qkv[1], qkv[2]  # B,12,320,64 Make torchscript happy (cannot use tensor as tuple)
            # if ind == 0:#
            #     q =q_o
            # else:
            #     q = torch.cat([q_o[:, :, 0].unsqueeze(2), q_o[:, :, template_feat_len + token_len:]], dim=2)
            q = torch.cat([q_o[:,:,0].unsqueeze(2), q_o[:,:,template_feat_len + token_len:]], dim=2)

            attn = (q @ k.transpose(-2, -1)) * self.scale #B,12,320,320
            # if torch.isnan(attn).any():
            #     print('attn:', attn)
            #     raise ValueError('ERROR: attn1. unfusion. network outputs is NAN! stop training')
            attn = attn.softmax(dim=-1) #B,12,320,320
            # if torch.isnan(attn).any():
            #     print('attn:', attn)
            #     raise ValueError('ERROR: attn2. unfusion. network outputs is NAN! stop training')
            attn = self.attn_drop(attn) #B,12,320,320
            # if torch.isnan(attn).any():
            #     print('attn:', attn)
            #     raise ValueError('ERROR: attn3. unfusion. network outputs is NAN! stop training')
            x = (attn @ v).transpose(1, 2).reshape(B, N_q, C) #B,320,768
            # if torch.isnan(x).any():
            #     print('x:', x)
            #     raise ValueError('ERROR: x1. attention. unfusion. network outputs is NAN! stop training')
            x = self.proj(x) #B,320,768
            # if torch.isnan(x).any():
            #     print('x:', x)
            #     raise ValueError('ERROR: x2. attention. unfusion. network outputs is NAN! stop training')
            x = self.proj_drop(x) #B,320,768
            # if torch.isnan(x).any():
            #     print('x:', x)
            #     raise ValueError('ERROR: x3. attention. unfusion. network outputs is NAN! stop training')
        return x, decision

class FSOU2Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 #drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, divide=False, fs_keep=False,):
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, divide=False, fs_keep=-1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = AttentionOU(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
                              divide=divide)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.fs_keep = fs_keep
        #if fs_keep:
        #self.frefeature = {}
        #self.frefeature2 = {}
        if self.fs_keep != -1:
            self.fusion_attn = nn.MultiheadAttention(dim, num_heads, dropout=0.1)
            self.fusion_dropout = nn.Dropout(0.1)
            self.fusion_norm = nn.LayerNorm(dim)
            #self.fusion_norm2 = nn.LayerNorm(dim)
            self.fusion_crossattn = Cattention(dim)
            #创建RoIAlign层
        self.pooler = torchvision.ops.RoIAlign(output_size=(8, 8), sampling_ratio=-1, spatial_scale=1) ####bs需要改
            #self.pooler = torchvision.ops.RoIAlign(sampling_ratio=-1, spatial_scale=1)  ####bs需要改
            #self.frefeature[self.fs_keep] = None
            #self.frefeature2[self.fs_keep] = None
            # self.frefeature = None
            # self.frefeature2 = None

    #def forward(self, x_m1, x_m2, template_mask, search_feat_len, token_len, threshold, tgt_type):
    def forward(self, x_m1, x_m2, template_mask, box, feature_1, feature_2,  search_feat_len, token_len, ind, threshold, tgt_type):
        if self.fs_keep != -1:
            #if ind == 0:
            if box is not None:
                bs, H, C = feature_1.size()
                # inputTensor_1 = feature_1.transpose(1, 2).view(bs, C, 16, 16)
                # feature_1 = self.pooler(inputTensor_1, [box_xywh_to_xyxy(box)]) ###box的值 #B,768,16,16
                # inputTensor_2 = feature_2.transpose(1, 2).view(bs, C, 16, 16)
                # feature_2 = self.pooler(inputTensor_2, [box_xywh_to_xyxy(box)])  ###box的值 #B,768,16,16
                # x_fm1 = feature_1.view(bs, H//4, C)
                # x_fm2 = feature_2.view(bs, H//4, C)
                # x_2 = self.fusion_attn(x_1.transpose(0, 1), x_fm1.transpose(0, 1), x_fm1.transpose(0, 1))[0].transpose(0, 1)  # 用上一帧特征，增强时间信息
                inputTensor_1 = feature_1.view(bs, 16, 16, C)
                box_1 = box_xywh_to_xyxy(box * 16)
                feature_1, _, _ = self.sample_target_batch(inputTensor_1, box_1, 2, 8)
                feature_1 = feature_1.permute(0, 2, 3, 1)
                inputTensor_2 = feature_2.view(bs, 16, 16, C)
                # box_2 = [box_xywh_to_xyxy(box)] * 16
                feature_2, _, _ = self.sample_target_batch(inputTensor_2, box_1, 2, 8)
                feature_2 = feature_2.permute(0, 2, 3, 1)
                x_fm1 = feature_1.view(bs, H // 4, C)
                x_fm2 = feature_2.view(bs, H // 4, C)
            else:
                x_fm1 = x_m1[:, 1:-search_feat_len]
                x_fm2 = x_m2[:, 1:-search_feat_len]

                #x_fm1 = self.frefeature[self.fs_keep] if self.frefeature[self.fs_keep] is not None else x_1
                #x_2 = self.fusion_attn(x_1, x_fm1, x_fm1)[0] #用上一帧特征，增强时间信息
            x_m1 = torch.cat([x_m1[:, 0].unsqueeze(1), x_fm1, x_m1[:, 1:]], dim=1)
            x_m2 = torch.cat([x_m2[:, 0].unsqueeze(1), x_fm2, x_m2[:, 1:]], dim=1)
            # x_m1 = torch.cat([x_fm1, x_m1], dim=1)
            # x_m2 = torch.cat([x_fm2, x_m2], dim=1)
            x_1 = self.fusion_attn(x_m2, x_m1, x_m1)[0]  # 3,321,768(3,16*16,768)
            x_1 = x_m1 + self.fusion_dropout(x_1)
            #x_1 = self.fusion_norm(x_1)
            #x_2 = self.fusion_norm2(x_2)
            #x_1 = x_1 + self.fusion_dropout(x_2)
            x_m1 = torch.cat([x_1[:, 0].unsqueeze(1), x_1[:, 65:]], dim=1)
            # feat, decision = self.attn(self.norm1(x_m1), self.norm1(x_1), self.norm1(x_1), template_mask, search_feat_len, token_len,
            #                            threshold=threshold, tgt_type=tgt_type)
            feat, decision = self.attn(self.norm1(x_1), template_mask, search_feat_len, token_len,
                                       threshold=threshold, tgt_type=tgt_type)
            # if torch.isnan(feat).any():
            #     print('feat:', feat)
            #     raise ValueError('ERROR: feat. fusion. network outputs is NAN! stop training')
            x = x_m1 + self.drop_path(feat)  # B,320,768
            x_1 = self.fusion_crossattn(x.permute(0, 2, 1), x_m1.permute(0, 2, 1)).permute(0, 2, 1)
            #x = x + self.drop_path(self.mlp(self.norm2(x_1)))  # B,320,768
            x = x_1 + self.drop_path(self.mlp(self.norm2(x_1)))
            # if torch.isnan(x).any():
            #     print('x:', x)
            #     raise ValueError('ERROR: x. fusion. network outputs is NAN! stop training')

        else:
            #if ind == 0:
            if box is not None:
                bs, H, C = feature_1.size()
                # inputTensor_1 = feature_1.transpose(1, 2).view(bs, C, 16, 16)
                # feature_1 = self.pooler(inputTensor_1, [box_xywh_to_xyxy(box)]) ###box的值 #B,768,16,16
                inputTensor_1 = feature_1.view(bs, 16, 16, C)
                box_1 = box_xywh_to_xyxy(box * 16)
                feature_1, _, _ = self.sample_target_batch(inputTensor_1, box_1, 2, 8)
                feature_1 = feature_1.permute(0, 2, 3, 1)
                x_fm1 = feature_1.view(bs, H // 4, C)
            else:
                x_fm1 = x_m1[:, 1:65]
            x_1 = torch.cat([x_m1[:, 0].unsqueeze(1), x_fm1, x_m1[:, 1:]], dim=1)
            # feat, decision = self.attn(self.norm1(x_m1), self.norm1(x_1), self.norm1(x_1), template_mask, search_feat_len, token_len,
            #                            threshold=threshold, tgt_type=tgt_type) #预训练的参数无法使用，影响结果，qkv需要一致，后续再拆开
            feat, decision = self.attn(self.norm1(x_1), template_mask, search_feat_len, token_len,
                                       threshold=threshold, tgt_type=tgt_type)
            # if torch.isnan(feat).any():
            #     print('feat:', feat)
            #     raise ValueError('ERROR: feat. unfusion. network outputs is NAN! stop training')
            x = x_m1 + self.drop_path(feat)  # B,320,768
            x = x + self.drop_path(self.mlp(self.norm2(x)))  # B,320,768
            # if torch.isnan(x).any():
            #     print('x:', x)
            #     raise ValueError('ERROR: x. unfusion. network outputs is NAN! stop training')
        return x, decision
    def sample_target_batch(self, im_feature, target_bb, search_area_factor, output_sz):
        """
        Extracts a square crop centered at target_bb box, of area search_area_factor^2 times target_bb area.

        Args:
            im: cv image.
            target_bb: Target box [x, y, w, h].
            search_area_factor: Ratio of crop size to target size.
            output_sz (float): Size to which the extracted crop is resized (always square). If None, no resizing is done.

        Returns:
            cv image: Extracted crop.
            float: The factor by which the crop has been resized to make the crop size equal output_size.
        """
        # im_crop = list()
        im_crop_padded_list = list()
        att_mask_list = list()
        resize_factor_list = list()
        for i in range(len(im_feature)):
            if not isinstance(target_bb[i], list):
                x, y, w, h = target_bb[i].tolist()
            else:
                x, y, w, h = target_bb[i]

            # Crop image
            crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)

            # if crop_sz < 1:
            #     raise Exception('ERROR: too small bounding box')

            x1 = round(x + 0.5 * w - crop_sz * 0.5)
            x2 = x1 + crop_sz

            y1 = round(y + 0.5 * h - crop_sz * 0.5)
            y2 = y1 + crop_sz

            x1_pad = max(0, -x1)
            x2_pad = max(x2 - im_feature.shape[2] + 1, 0)

            y1_pad = max(0, -y1)
            y2_pad = max(y2 - im_feature.shape[3] + 1, 0)

            # 方法1：不够补0
            # Crop target
            im_crop = im_feature[i, x1 + x1_pad:x2 - x2_pad, y1 + y1_pad:y2 - y2_pad, :]
            # Pad
            # im_crop_padded = cv.copyMakeBorder(im_crop.cpu().numpy(), x1_pad, x2_pad, y1_pad, y2_pad, cv.BORDER_CONSTANT)  # 补0
            w_crop, h_crop, C = im_crop.shape
            wh = max(w_crop, h_crop)
            if w_crop < wh:
                padded = torch.zeros(wh - w_crop, h_crop, C).cuda()
                im_crop_padded = torch.cat([im_crop, padded], dim=0).cuda()
            elif h_crop < wh:
                padded = torch.zeros(w_crop, wh - h_crop, C).cuda()
                im_crop_padded = torch.cat([im_crop, padded], dim=1).cuda()
            else:
                im_crop_padded = im_crop
            # 方法2:不够的话多采样一些
            w_crop = x2 - x2_pad - x1 - x1_pad
            h_crop = y2 - y2_pad - y1 - y1_pad
            wh = max(w_crop, h_crop)
            im_crop_padded = im_feature[i, x1 + x1_pad:x1 + x1_pad + wh, y1 + y1_pad:y1 + y1_pad + wh, :]
            # Deal with attention mask
            # print(im_crop_padded.shape)
            W, H, _ = im_crop_padded.shape
            att_mask = torch.ones((W, H)).cuda()
            end_x, end_y = -x2_pad, -y2_pad
            if y2_pad == 0:
                end_y = None
            if x2_pad == 0:
                end_x = None
            att_mask[x1_pad:end_x, y1_pad:end_y] = 0  # 数据部分为0，pad部分为1
            resize_factor = output_sz / crop_sz
            box = torch.FloatTensor([[0, 0, 1, 1]]).cuda()
            im_crop_padded = self.pooler(im_crop_padded.unsqueeze(0).permute(0, 3, 1, 2), [box])
            att_mask = self.pooler(att_mask.unsqueeze(0).unsqueeze(0), [box])
            im_crop_padded_list.append(im_crop_padded)
            resize_factor_list.append(resize_factor)
            att_mask_list.append(att_mask)
        im_crop_padded_list = torch.cat(im_crop_padded_list, dim=0)
        # resize_factor_list = torch.stack(resize_factor_list)
        att_mask_list = torch.cat(att_mask_list, dim=0).squeeze(1)
        # box = torch.FloatTensor([[0, 0, 1, 1]]).cuda().repeat(len(im_feature),1)
        # im_crop_padded_list = self.pooler(im_crop_padded_list.permute(0, 3, 1, 2), [box])
        # att_mask_list = self.pooler(att_mask_list.unsqueeze(0), [box])
        return im_crop_padded_list, resize_factor_list, att_mask_list

class VisionTransformerFSOU2(VisionTransformer):
    """ Vision Transformer with candidate elimination (CE) module

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', fs_loc=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        # super().__init__()
        super().__init__()
        if isinstance(img_size, tuple):
            self.img_size = img_size
        else:
            self.img_size = to_2tuple(img_size)
        self.patch_size = patch_size
        self.in_chans = in_chans

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.fs_loc = fs_loc
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        blocks = []
        fs_index = 0
        for i in range(depth):
            #fs_keep = False
            fs_keep = -1
            if fs_loc is not None and i in fs_loc:
                #fs_keep = True
                fs_keep += 1
                fs_index += 1
            blocks.append(
                FSOU2Block( ####FSBlock_hi
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                    attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                    divide=bool(i), fs_keep=fs_keep)
            )
            self.blocks = nn.Sequential(*blocks)
        self.norm = norm_layer(embed_dim)
        #self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.track_query_rgb = None
        self.track_query_t = None
        self.track_query_save_rgb = nn.Parameter(torch.zeros(3, 1, embed_dim))  ###batch_size
        self.track_query_save_t = nn.Parameter(torch.zeros(3, 1, embed_dim))
        self.init_weights(weight_init)
        self.prefeature_rgb = {}
        self.prefeature_t = {}
        self.prefeature_rgb_new = {}
        self.prefeature_t_new = {}
        # for i in range(fs_index):
        for i in range(depth):
            self.prefeature_rgb[i] = None
            self.prefeature_t[i] = None
            self.prefeature_rgb_new[i] = None
            self.prefeature_t_new[i] = None
        self.search_box_prv = {}

    def forward_features(self, z_rgb, z_t, x_rgb, x_t, template_mask_rgb, template_mask_t, search_box,
                         search_feat_len, threshold, tgt_type, ind,
                         #track_query_rgb, track_query_t,
                         token_len=1
                         #masks_fusion_rgb, masks_fusion_t
                         ):

        x_rgb = self.patch_embed(x_rgb)
        z_rgb = self.patch_embed(z_rgb)

        x_t = self.patch_embed(x_t)
        z_t = self.patch_embed(z_t)

        x_rgb += self.pos_embed_x
        z_rgb += self.pos_embed_z

        x_t += self.pos_embed_x
        z_t += self.pos_embed_z

        x_rgb = combine_tokens(z_rgb, x_rgb, mode=self.cat_mode)
        x_t = combine_tokens(z_t, x_t, mode=self.cat_mode)

        #训练中track_query清零
        # if i == 0:
        #     track_query_rgb = None
        #     track_query_t = None
        # else:
        #     track_query_rgb = self.track_query_rgb
        #     track_query_t = self.track_query_t
        #训练中track_query不清零
        # if self.track_query_rgb is None:
        #     track_query_rgb = self.track_query_save_rgb
        #     track_query_t = self.track_query_save_t
        # else:
        #     track_query_rgb = self.track_query_rgb
        #     track_query_t = self.track_query_t
        track_query_rgb = self.track_query_rgb
        track_query_t = self.track_query_t
        # print('track_query_rgb:', track_query_rgb)
        # print('track_query_t:', track_query_t)
        B = x_rgb.shape[0]

        new_query = self.cls_token.expand(B, token_len, -1)  # copy B times B,1,768
        query_rgb = new_query if track_query_rgb is None else track_query_rgb + new_query
        query_t = new_query if track_query_t is None else track_query_t + new_query

        # query_rgb = query_rgb + self.cls_pos_embed  # B,1,768
        # query_t = query_t + self.cls_pos_embed  # B,1,768

        x_rgb = torch.cat([query_rgb, x_rgb], dim=1)
        x_t = torch.cat([query_t, x_t], dim=1)

        x_rgb = self.pos_drop(x_rgb)
        x_t = self.pos_drop(x_t)

        decisions_rgb = list()
        decisions_t = list()
        #l = len(search_box)//2
        index = -1
        search_box_feature = {}
        search_box_feature[0] = search_box[0]
        search_box_feature[1] = search_box[1]
        self.search_box_prv[0] = search_box[0]
        self.search_box_prv[1] = search_box[1]
        # fs_index = 0
        for i, blk in enumerate(self.blocks):
            #print(i)
            index += 1
            if self.prefeature_rgb_new[index] is not None:
                feature_rgb = self.prefeature_rgb_new[index]
                feature_t = self.prefeature_t_new[index]
            else:
                feature_rgb = None
                feature_t = None
            self.prefeature_rgb[index] = feature_rgb
            self.prefeature_t[index] = feature_t

            self.prefeature_rgb_new[index] = x_rgb[:, -search_feat_len:].detach()
            self.prefeature_t_new[index] = x_t[:, -search_feat_len:].detach()
            x_rgb_new, decision_rgb = blk(x_rgb, x_t, template_mask_rgb, search_box_feature[0], feature_rgb, feature_t,
                                          search_feat_len, token_len, ind,
                                          threshold=threshold, tgt_type=tgt_type)

            x_t, decision_t = blk(x_t, x_rgb, template_mask_t, search_box_feature[1], feature_t, feature_rgb, search_feat_len,
                                  token_len, ind,
                                  threshold=threshold, tgt_type=tgt_type)
            # if self.fs_loc is not None and i in self.fs_loc:
            #     index += 1
            #     feature_rgb = self.prefeature_rgb[index]
            #     feature_t = self.prefeature_t[index]
            #     x_rgb_new, decision_rgb = blk(x_rgb, x_t, template_mask_rgb, search_box[0], feature_rgb, feature_t, search_feat_len, token_len,
            #                                   threshold=threshold, tgt_type=tgt_type)
            #     self.prefeature_rgb[index] = x_rgb_new[:, -search_feat_len:].detach()
            #
            #     x_t, decision_t = blk(x_t, x_rgb, template_mask_t, search_box[1], feature_t, feature_rgb, search_feat_len, token_len,
            #                           threshold=threshold, tgt_type=tgt_type)
            #     self.prefeature_t[index] = x_t[:, -search_feat_len:].detach()
            # else:
            #     feature = None
            #     x_rgb_new, decision_rgb = blk(x_rgb, x_t, template_mask_rgb, search_box[:l], feature_t, feature_rgb, search_feat_len,
            #                                   token_len, threshold=threshold, tgt_type=tgt_type)
            #     x_t, decision_t = blk(x_t, x_rgb, template_mask_t, search_box[l:], feature_t, feature_rgb, search_feat_len, token_len,
            #                           threshold=threshold, tgt_type=tgt_type)
            x_rgb = x_rgb_new

            if decision_rgb is not None and self.training:  # 统计选入E_A的比例
                map_size = decision_rgb.shape[1]
                decision_rgb = decision_rgb[:, :, -1].sum(dim=-1, keepdim=True) / map_size
                decisions_rgb.append(decision_rgb)
            if decision_t is not None and self.training:  # 统计选入E_A的比例
                map_size = decision_t.shape[1]
                decision_t = decision_t[:, :, -1].sum(dim=-1, keepdim=True) / map_size
                decisions_t.append(decision_t)

        x_rgb = recover_tokens(x_rgb, mode=self.cat_mode)
        x_t = recover_tokens(x_t, mode=self.cat_mode)
        x_rgb = self.norm(x_rgb) ####新加
        x_t = self.norm(x_t) ####新加
        # self.track_query_rgb = (x_rgb[:, :token_len, :].clone()).detach()
        # self.track_query_t = (x_t[:, :token_len, :].clone()).detach()
        self.track_query_rgb = (x_rgb[:, :token_len, :].clone()).detach().cuda()  ######track_query改
        self.track_query_t = (x_t[:, :token_len, :].clone()).detach().cuda()
        self.track_query_save_rgb = nn.Parameter(self.track_query_rgb)
        self.track_query_save_t = nn.Parameter(self.track_query_t)


        if self.training:
            decisions_rgb = torch.cat(decisions_rgb, dim=-1)  # .mean(dim=-1, keepdim=True)
            decisions_t = torch.cat(decisions_t, dim=-1)
            #decisions = torch.cat([decisions_rgb, decisions_t], dim=-1)
            return x_rgb, x_t, decisions_rgb, decisions_t
            #return self.norm(x_rgb), decisions_rgb
        else:
            return x_rgb, x_t, None, None
            #return self.norm(x_rgb), None

    def forward(self, z_rgb, z_t, x_rgb, x_t, template_mask_rgb, template_mask_t, search_box, search_feat_len, threshold, tgt_type, i,
                #track_query,
                #masks_fusion_rgb, masks_fusion_t,
                **kwargs):

        x_rgb, x_t, decisions_rgb, decisions_t = self.forward_features(z_rgb, z_t, x_rgb, x_t, template_mask_rgb, template_mask_t, search_box,
                                             search_feat_len, threshold, tgt_type, i, #track_query
                                             #masks_fusion_rgb, masks_fusion_t
                                             )
        return x_rgb, x_t, decisions_rgb, decisions_t


def _create_vision_transformer(pretrained=False, **kwargs):
    model = VisionTransformerFSOU2(**kwargs)

    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model"], strict=False)
            print('Load pretrained model from: ' + pretrained)

    return model


def vit_base_patch16_224_base_fs(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model


def vit_base_patch16_224_large_fs(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model
