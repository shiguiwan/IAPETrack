import math
import copy
from typing import Optional, List, Any

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import Module
from torch.nn import MultiheadAttention
from torch.nn import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn import Dropout
import time
import numpy as np

# class TransformerEncoder_rgbt(nn.Module):
#
#     # def __init__(self, encoder_layer, num_layers, norm=None):
#     def __init__(self, encoder_layer, num_layers, norm=None):
#         super().__init__()
#         self.layers_rgb = _get_clones(encoder_layer, num_layers)
#         self.layers_t = _get_clones(encoder_layer, num_layers)
#         self.num_layers = num_layers
#         #self.query_scale = MLP(d_model, d_model, d_model, 2)  #####DN-DETR中有，是否需要，待验证
#         self.norm = norm
#
#     def forward(self, src: Tensor, srcc: Tensor,
#                 mask: Optional[Tensor] = None,
#                 src_key_padding_mask: Optional[Tensor] = None,
#                 pos: Optional[Tensor] = None,):
#         output_rgb = src
#         output_t = srcc
#
#         for layer in self.layers_rgb:
#             output_rgb = layer(output_rgb, srcc, src_mask=mask,
#                                src_key_padding_mask=src_key_padding_mask, pos=pos)
#             if torch.isnan(output_rgb).any():
#                 print('memory_rgb:', output_rgb)
#         if self.norm is not None:
#             output_rgb = self.norm(output_rgb)
#         for layer in self.layers_t:
#             output_t = layer(output_t, src, src_mask=mask,
#                              src_key_padding_mask=src_key_padding_mask, pos=pos)
#             if torch.isnan(output_t).any():
#                 print('memory_t:', output_t)
#         if self.norm is not None:
#             output_t = self.norm(output_t)
#         return output_rgb, output_t
#
# def _get_clones(module, N):
#     return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Cattention(nn.Module):

    def __init__(self, in_dim):
        super(Cattention, self).__init__()
        self.channel_in = in_dim
        self.conv1 = nn.Sequential(
            nn.ConvTranspose1d(in_dim * 2, in_dim, kernel_size=1, stride=1),
            # nn.ConvTranspose2d(in_dim * 2, in_dim, kernel_size=1, stride=1),
        )
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.linear1 = nn.Conv2d(in_dim, in_dim // 6, 1, bias=False)
        self.linear2 = nn.Conv2d(in_dim // 6, in_dim, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout()

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, x, y):
        ww = self.linear2(self.dropout(self.activation(self.linear1(self.avg_pool(y).unsqueeze(dim=-1))))).squeeze(
            dim=-1)
        # a = self.avg_pool(y).unsqueeze(dim=-1) #b,768,320->b,768,1,1
        # b = self.linear1(a) ##b,768,1,1->b,128,1,1
        # c = self.activation(b)
        # d = self.dropout(c)
        # ww = self.linear2(d).squeeze(dim=-1) ##b,128,1,1->b,768,1,1->b,768,1
        # e = self.conv1(torch.cat((x, y), 1)) #2,768,320
        # weight = e*ww
        weight = self.conv1(torch.cat((x, y), 1)) * ww

        #2W
        # B,C,W = y.size()
        # y_1 = y[:,:,:64].view(B,C,8,8)
        # ## a_1 = self.avg_pool(y_1)  # b,768，320->b,768,1,1
        # ## b_1 = self.linear1(a_1)  ##b,768,1,1->b,128,1,1
        # ## c_1 = self.activation(b_1)
        # ## d_1 = self.dropout(c_1)
        # ## ww_1 = self.linear2(d_1)  ##b,128,1,1->b,768,1,1->b,768,1
        # ww_1 = self.linear2(self.dropout(self.activation(self.linear1(self.avg_pool(y_1)))))
        # e_1 = self.conv1(torch.cat((x[:,:,:64].view(B,C,8,8), y_1), 1))
        #
        # y_2 = y[:, :, 64:].view(B, C, 16, 16)
        # ## a_2 = self.avg_pool(y_2)  # b,768，320->b,768,1,1
        # ## b_2 = self.linear1(a_2)  ##b,768,1,1->b,128,1,1
        # ## c_2 = self.activation(b_2)
        # ## d_2 = self.dropout(c_2)
        # ## ww_2 = self.linear2(d_2)  ##b,128,1,1->b,768,1,1->b,768,1
        # ww_2 = self.linear2(self.dropout(self.activation(self.linear1(self.avg_pool(y_2)))))
        # e_2 = self.conv1(torch.cat((x[:, :, 64:].view(B,C,16,16), y_2), 1))
        #
        #
        # weight = torch.cat([(e_1 * ww_1).view(B,C,-1), (e_2 * ww_2).view(B,C,-1)], dim=2)


        return x + self.gamma * weight * x

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # self.self_attn = MultiheadAttention_multiflow(d_model, nhead, dropout=dropout)
        channel = 768  ##maybe need to change
        self.cross_attn = Cattention(channel)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm0 = nn.LayerNorm(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        #self.normalize_before = normalize_before  # first normalization, then add

        #self.divide_norm = divide_norm
        self.scale_factor = float(d_model // nhead) ** 0.5
        self._reset_parameters()

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src, srcc,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        # b, c, s = src.permute(1, 2, 0).size()

        # src2 = self.self_attn(self.norm0(src + srcc), self.norm0(src + srcc), src, attn_mask=src_mask,
        #                       key_padding_mask=src_key_padding_mask)[0]
        q = k = self.with_pos_embed(src + srcc, pos)  # add pos to src
        # q = self.with_pos_embed(src + srcc, pos)
        # k = self.with_pos_embed(src, pos)
        #if self.divide_norm:
            # print("encoder divide by norm")
        q = q / torch.norm(q, dim=-1, keepdim=True) * self.scale_factor
        k = k / torch.norm(k, dim=-1, keepdim=True)
        src2 = self.self_attn(q, k, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # src = self.cross_attn(src.permute(0, 2, 3, 1), srcc.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2)
        src = self.cross_attn(src.permute(1, 2, 0), srcc.permute(1, 2, 0).contiguous()).permute(2, 0, 1)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


    def forward(self, src, srcc,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        # if self.normalize_before:
        #    return self.forward_pre(src, srcc, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, srcc, src_mask, src_key_padding_mask, pos)

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "prelu":
        return nn.PReLU()
    if activation == "selu":
        return F.selu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

# def build_transformer_rgbt_encoderlayer(cfg):
#     return TransformerEncoderLayer(
#         d_model=cfg.MODEL.FUSION.HIDDEN_DIM,
#         dropout=cfg.MODEL.FUSION.DROPOUT,
#         nhead=cfg.MODEL.FUSION.NHEADS,
#         dim_feedforward=cfg.MODEL.FUSION.DIM_FEEDFORWARD,
#         activation=cfg.MODEL.FUSION.ACTIVATION,
#         #num_patterns=cfg.MODEL.FUSION.NUM_PATTERNS,
#         #divide_norm=cfg.MODEL.TRANSFORMER.DIVIDE_NORM,
#
#     )
