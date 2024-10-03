import torch
import torch.nn as nn
from timm.models.layers import to_2tuple

from lib.models.layers.patch_embed import PatchEmbed
from lib.models.grm.utils import combine_tokens, recover_tokens


class BaseBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        # For original ViT
        self.pos_embed = None
        self.img_size = [224, 224]
        self.patch_size = 16
        self.embed_dim = 384

        self.cat_mode = 'direct'

        self.pos_embed_x = None
        self.pos_embed_z = None

    def finetune_track(self, cfg, patch_start_index=1):
        search_size = to_2tuple(cfg.DATA.SEARCH.SIZE)
        template_size = to_2tuple(cfg.DATA.TEMPLATE.SIZE)
        new_patch_size = cfg.MODEL.BACKBONE.STRIDE

        self.cat_mode = cfg.MODEL.BACKBONE.CAT_MODE

        # Resize patch embedding
        if new_patch_size != self.patch_size:
            print('inconsistent patch size with the pretrained weights, interpolate the weight')
            old_patch_embed = dict()
            for name, param in self.patch_embed.named_parameters():
                if 'weight' in name:
                    param = nn.functional.interpolate(param, size=(new_patch_size, new_patch_size),
                                                      mode='bicubic', align_corners=False)
                    param = nn.Parameter(param)
                old_patch_embed[name] = param
            self.patch_embed = PatchEmbed(img_size=self.img_size, patch_size=new_patch_size, in_chans=3,
                                          embed_dim=self.embed_dim)
            self.patch_embed.proj.bias = old_patch_embed['proj.bias']
            self.patch_embed.proj.weight = old_patch_embed['proj.weight']

        # For patch embedding
        patch_pos_embed = self.pos_embed[:, patch_start_index:, :]
        patch_pos_embed = patch_pos_embed.transpose(1, 2)
        B, E, Q = patch_pos_embed.shape
        P_H, P_W = self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size #224/16=14
        patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)

        # For search region
        H, W = search_size #256，256
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size #16，16
        search_patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic',
                                                           align_corners=False) ##上采样函数 1,768,16,16
        search_patch_pos_embed = search_patch_pos_embed.flatten(2).transpose(1, 2) #1,256,768

        # For template region
        H, W = template_size #128,128
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size #8,8
        template_patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic',
                                                             align_corners=False) #1,768,8,8
        template_patch_pos_embed = template_patch_pos_embed.flatten(2).transpose(1, 2)

        self.pos_embed_x = nn.Parameter(search_patch_pos_embed)
        self.pos_embed_z = nn.Parameter(template_patch_pos_embed)

    def finetune_track_od(self, cfg, patch_start_index=1):
        search_size = to_2tuple(cfg.DATA.SEARCH.SIZE)
        template_size = to_2tuple(cfg.DATA.TEMPLATE.SIZE)
        new_patch_size = cfg.MODEL.BACKBONE.STRIDE

        self.cat_mode = cfg.MODEL.BACKBONE.CAT_MODE

        # Resize patch embedding
        if new_patch_size != self.patch_size:
            print('inconsistent patch size with the pretrained weights, interpolate the weight')
            old_patch_embed = dict()
            for name, param in self.patch_embed.named_parameters():
                if 'weight' in name:
                    param = nn.functional.interpolate(param, size=(new_patch_size, new_patch_size),
                                                      mode='bicubic', align_corners=False)
                    param = nn.Parameter(param)
                old_patch_embed[name] = param
            self.patch_embed = PatchEmbed(img_size=self.img_size, patch_size=new_patch_size, in_chans=3,
                                          embed_dim=self.embed_dim)
            self.patch_embed.proj.bias = old_patch_embed['proj.bias']
            self.patch_embed.proj.weight = old_patch_embed['proj.weight']

        # For patch embedding
        patch_pos_embed = self.pos_embed[:, patch_start_index:, :]
        patch_pos_embed = patch_pos_embed.transpose(1, 2)
        B, E, Q = patch_pos_embed.shape
        P_H, P_W = self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size #224/16=14
        patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)

        # For search region
        H, W = search_size #256，256
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size #16，16
        search_patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic',
                                                           align_corners=False) ##上采样函数 1,768,16,16
        search_patch_pos_embed = search_patch_pos_embed.flatten(2).transpose(1, 2) #1,256,768

        # For template region
        H, W = template_size #128,128
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size #8,8
        template_patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic',
                                                             align_corners=False) #1,768,8,8
        template_patch_pos_embed = template_patch_pos_embed.flatten(2).transpose(1, 2)

        # For decoder region
        patch_pos_embed_decoder = self.decoder_pos_embed[:, patch_start_index:, :]
        patch_pos_embed_decoder = patch_pos_embed_decoder.transpose(1, 2)
        B, E, Q = patch_pos_embed_decoder.shape
        patch_pos_embed_decoder = patch_pos_embed_decoder.view(B, E, 14, 14)
        template_patch_pos_embed_decoder = nn.functional.interpolate(patch_pos_embed_decoder, size=(8, 8),
                                                                     mode='bicubic',
                                                                     align_corners=False)  # 1,768,8,8
        template_patch_pos_embed_decoder = template_patch_pos_embed_decoder.flatten(2).transpose(1, 2)
        template_patch_pos_embed_decoder = torch.cat(
            [self.decoder_pos_embed[:, 0, :].unsqueeze(1), template_patch_pos_embed_decoder], dim=1)
        self.decoder_pos_embed_z = nn.Parameter(template_patch_pos_embed_decoder)

        self.pos_embed_x = nn.Parameter(search_patch_pos_embed)
        self.pos_embed_z = nn.Parameter(template_patch_pos_embed)


    def finetune_track_o(self, cfg, patch_start_index=1):
        search_size = to_2tuple(cfg.DATA.SEARCH.SIZE)
        template_size = to_2tuple(cfg.DATA.TEMPLATE.SIZE)
        new_patch_size = cfg.MODEL.BACKBONE.STRIDE

        self.cat_mode = cfg.MODEL.BACKBONE.CAT_MODE

        # Resize patch embedding
        if new_patch_size != self.patch_size:
            print('inconsistent patch size with the pretrained weights, interpolate the weight')
            old_patch_embed = dict()
            for name, param in self.patch_embed.named_parameters():
                if 'weight' in name:
                    param = nn.functional.interpolate(param, size=(new_patch_size, new_patch_size),
                                                      mode='bicubic', align_corners=False)
                    param = nn.Parameter(param)
                old_patch_embed[name] = param
            self.patch_embed = PatchEmbed(img_size=self.img_size, patch_size=new_patch_size, in_chans=3,
                                          embed_dim=self.embed_dim)
            self.patch_embed.proj.bias = old_patch_embed['proj.bias']
            self.patch_embed.proj.weight = old_patch_embed['proj.weight']

        # For patch embedding
        patch_pos_embed = self.pos_embed[:, patch_start_index:, :]
        patch_pos_embed = patch_pos_embed.transpose(1, 2)
        B, E, Q = patch_pos_embed.shape
        P_H, P_W = self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size #224/16=14
        patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)

        # For search region
        H, W = search_size #256，256
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size #16，16
        search_patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic',
                                                           align_corners=False) ##上采样函数
        search_patch_pos_embed = search_patch_pos_embed.flatten(2).transpose(1, 2) #1,768,16,16

        # For template region
        H, W = template_size #128,128
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size #8,8
        template_patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic',
                                                             align_corners=False) #1,768,8,8
        template_patch_pos_embed = template_patch_pos_embed.flatten(2).transpose(1, 2)

        self.pos_embed_x = nn.Parameter(search_patch_pos_embed)
        self.pos_embed_z = nn.Parameter(template_patch_pos_embed)

        if patch_start_index > 0:
            cls_pos_embed = self.pos_embed[:, 0:1, :]
            self.cls_pos_embed = nn.Parameter(cls_pos_embed)


    def forward_features(self, z_rgb, z_t, x_rgb, x_t, template_mask_rgb, template_mask_t,
                         search_feat_len, threshold, tgt_type):
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

        x_rgb = self.pos_drop(x_rgb)
        x_t = self.pos_drop(x_t)

        decisions_rgb = list()
        decisions_t = list()
        for i, blk in enumerate(self.blocks):
            #x, decision = blk(x, template_mask, search_feat_len, threshold=threshold, tgt_type=tgt_type)

            x_rgb, decision_rgb = blk(x_rgb, template_mask_rgb, search_feat_len, threshold=threshold, tgt_type=tgt_type)
            x_t, decision_t = blk(x_t, template_mask_t, search_feat_len, threshold=threshold, tgt_type=tgt_type)
            if decision_rgb is not None and self.training: #统计选入E_A的比例
                map_size = decision_rgb.shape[1]
                decision_rgb = decision_rgb[:, :, -1].sum(dim=-1, keepdim=True) / map_size
                decisions_rgb.append(decision_rgb)
            if decision_t is not None and self.training: #统计选入E_A的比例
                map_size = decision_t.shape[1]
                decision_t = decision_t[:, :, -1].sum(dim=-1, keepdim=True) / map_size
                decisions_t.append(decision_t)

        x_rgb = recover_tokens(x_rgb, mode=self.cat_mode)
        x_t = recover_tokens(x_t, mode=self.cat_mode)
        x = torch.cat([self.norm(x_rgb), self.norm(x_t)], dim = -1)

        if self.training:
            decisions_rgb = torch.cat(decisions_rgb, dim=-1)  # .mean(dim=-1, keepdim=True)
            decisions_t = torch.cat(decisions_t, dim=-1)
            decisions = torch.cat([decisions_rgb, decisions_t], dim=-1)
        return x, decisions

    def forward(self, z_rgb, z_t, x_rgb, x_t, template_mask_rgb, template_mask_t, search_feat_len, threshold, tgt_type, **kwargs):
        """
        Joint feature extraction and relation modeling for the basic ViT backbone.

        Args:
            z (torch.Tensor): Template feature, [B, C, H_z, W_z].
            x (torch.Tensor): Search region feature, [B, C, H_x, W_x].

        Returns:
            x (torch.Tensor): Merged template and search region feature, [B, L_z+L_x, C].
            attn : None.
        """

        x, decisions = self.forward_features(z_rgb, z_t, x_rgb, x_t, template_mask_rgb, template_mask_t, search_feat_len, threshold, tgt_type)
        return x, decisions
