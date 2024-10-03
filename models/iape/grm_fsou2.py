"""
Basic GRM model.
"""

import os
import copy
import torch
from torch import nn

from lib.models.layers.head import build_box_head
from lib.models.grm.vit import vit_base_patch16_224_base, vit_base_patch16_224_large
#from lib.models.grm.vit_fsou import vit_base_patch16_224_base_fs, vit_base_patch16_224_large_fs ##########
#from lib.models.grm.vit_fsou2 import vit_base_patch16_224_base_fs, vit_base_patch16_224_large_fs ##########
from lib.models.grm.vit_fsou3 import vit_base_patch16_224_base_fs, vit_base_patch16_224_large_fs
from lib.utils.box_ops import box_xyxy_to_cxcywh, box_cxcywh_to_xywh
#from lib.models.layers.fusion import build_transformer_rgbt_encoderlayer


class GRM_fsou(nn.Module):
    """
    This is the base class for GRM.
    """

    def __init__(self, transformer, box_head,  head_type='CORNER', tgt_type='allmax'):
        """
        Initializes the model.

        Parameters:
            transformer: Torch module of the transformer architecture.
        """

        super().__init__()
        self.backbone = transformer
        self.box_head = box_head
        self.head_type = head_type
        if head_type == 'CORNER' or head_type == 'CENTER':
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)
        self.tgt_type = tgt_type
        self.search_box_i = {}
        self.search_box_i[0] = None
        self.search_box_i[1] = None

    def forward(self, template_rgb: torch.Tensor, template_t: torch.Tensor, search_rgb: torch.Tensor,
                search_t: torch.Tensor, template_mask_rgb=None, template_mask_t=None, search_box=None, threshold=0., score=0,
                #masks_fusion_rgb=None, masks_fusion_t=None
                ):
        # x, decisions = self.backbone(z=template, x=search, template_mask=template_mask, search_feat_len=self.feat_len_s,
        #                              threshold=threshold, tgt_type=self.tgt_type)
        # assert isinstance(search_rgb, list)
        # assert isinstance(search_t, list)
        out_dict = []
        # search_box_i = {}
        # search_box_i[0] = None
        # search_box_i[1] = None
        #score = 0
        for i in range(len(search_rgb)):
            #search_box_i = torch.cat([search_box[i], search_box[i+len(search_rgb)//2]], dim=0)
            #20240412改，search_box_i应为前一帧的gt
            # if len(search_rgb) == 1 and self.training == False:
            #     search_box_i = {}
            #     search_box_i[0] = search_box[0]
            #     search_box_i[1] = search_box[1]
            # else:
            #     if i == 0:
            #         search_box_i = {}
            #         search_box_i[0] = None
            #         search_box_i[1] = None
            #     else:
            #         search_box_i = torch.index_select(search_box, dim=0,
            #                                           index=torch.tensor([i - 1, i - 1 + len(search_rgb) // 2]).cuda())
            search_box_i = self.search_box_i
            x, decisions = self.backbone(z_rgb=template_rgb, z_t=template_t, x_rgb=search_rgb[i], x_t=search_t[i],
                                         template_mask_rgb=template_mask_rgb, template_mask_t=template_mask_t,
                                         search_box = search_box_i,
                                         search_feat_len=self.feat_len_s,
                                         threshold=threshold, tgt_type=self.tgt_type, i=i, score = score
                                         #track_query_rgb=self.track_query_rgb, track_query_t=self.track_query_t,
                                         #masks_fusion_rgb = masks_fusion_rgb, masks_fusion_t = masks_fusion_t
                                         )

            # Forward head
            feat_last = x #4,320,768
            if isinstance(x, list):
                feat_last = x[-1]

            enc_opt = feat_last[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)#B,576,768
            # if self.backbone.add_cls_token:
            #     self.track_query = (x[:, :self.token_len].clone()).detach()
            att = torch.matmul(enc_opt, x[:, :1].transpose(1, 2))
            _, B, _ = att.size()# (B, HW, N) B,576,1
            att = att/(att[:, -1, :].unsqueeze(1).repeat(1,B,1)+1e-6)
            #att = att / (torch.sum(att, 1, keepdim=True) + 1e-6) * B #torch.sum(input, dim, keepdim=False, *, dtype=None)
            opt = (enc_opt.unsqueeze(-1) * att.unsqueeze(-2)).permute(
                (0, 3, 2, 1)).contiguous()  # (B, HW, C, N) --> (B, N, C, HW)B,1,768,576
            #print('decisions:', des)
            # Forward head
            #if i is not 0:
            out = self.forward_head(opt, None)
            self.search_box_i[0] = box_cxcywh_to_xywh(out['pred_boxes'].squeeze(1))
            self.search_box_i[1] = box_cxcywh_to_xywh(out['pred_boxes'].squeeze(1))
            out['decisions'] = decisions
            out['feature'] = x
            out_dict.append(out)
        return out_dict

    #def forward_head(self, cat_feature, gt_score_map=None):
    def forward_head(self, opt, gt_score_map=None):
        """
        cat_feature: Output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C).
        """

        #enc_opt = cat_feature[:, -self.feat_len_s:]  # Encoder output for the search region (B, HW, C)
        #opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == 'CORNER':
            # Run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map}
            return out
        elif self.head_type == 'CENTER':
            # Run the center head
            #score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            score_map_ctr, bbox, size_map, offset_map, max_score = self.box_head(opt_feat, gt_score_map)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map,
                   'max_score': max_score,
                   }
            return out
        else:
            raise NotImplementedError


def build_grm_fsou(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    #pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    pretrained_path = os.path.join('/home/shw/code/GRM_rgbt/pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and ('GRM' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_fs':
        if cfg.MODEL.PRETRAIN_FILE == 'mae_pretrain_vit_base.pth':
            backbone = vit_base_patch16_224_base_fs(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                           fs_loc=cfg.MODEL.BACKBONE.FS_LOC)
            hidden_dim = backbone.embed_dim
            patch_start_index = 1
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_fs':
        if cfg.MODEL.PRETRAIN_FILE == 'mae_pretrain_vit_large.pth':
            backbone = vit_base_patch16_224_large_fs(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                           fs_loc=cfg.MODEL.BACKBONE.FS_LOC)
            hidden_dim = backbone.embed_dim
            patch_start_index = 1
        else:
            raise NotImplementedError
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base':
        if cfg.MODEL.PRETRAIN_FILE == 'mae_pretrain_vit_base.pth':
            backbone = vit_base_patch16_224_base(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
            hidden_dim = backbone.embed_dim
            patch_start_index = 1
        elif cfg.MODEL.PRETRAIN_FILE == 'deit_base_patch16_224-b5f2ef4d.pth':
            backbone = vit_base_patch16_224_base(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
            hidden_dim = backbone.embed_dim
            patch_start_index = 1
        elif cfg.MODEL.PRETRAIN_FILE == 'deit_base_distilled_patch16_224-df68dfff.pth':
            backbone = vit_base_patch16_224_base(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE, distilled=True)
            hidden_dim = backbone.embed_dim
            patch_start_index = 2
        else:
            raise NotImplementedError
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large':
        if cfg.MODEL.PRETRAIN_FILE == 'mae_pretrain_vit_large.pth':
            backbone = vit_base_patch16_224_large(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
            hidden_dim = backbone.embed_dim
            patch_start_index = 1
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)
    # backbone_rgb.finetune_track(cfg=cfg, patch_start_index=patch_start_index)
    # backbone_t.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)
    # if cfg.MODEL.BACKBONE.TYPE == 'vit_base_fs' or 'vit_large_fs':
    #     fusion_i = build_transformer_rgbt_encoderlayer(cfg)
    #
    #     backbone.fusion_rgb = _get_clones(fusion_i, 3)
    #     backbone.fusion_t = _get_clones(fusion_i, 3)

    model = GRM_fsou(
        backbone,
        box_head,
        #fusion,
        head_type=cfg.MODEL.HEAD.TYPE,
        tgt_type=cfg.MODEL.TGT_TYPE
    )

    if 'GRM' in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location='cpu')
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['net'], strict=False)
        print('load pretrained model from ' + cfg.MODEL.PRETRAIN_FILE)
    return model

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

