import torch.nn as nn

from timm.models.layers import to_2tuple
#from lib.models.grm.vit_fsou2_lora import LoRALayer
import math

class PatchEmbed(nn.Module):
    """
    2D image to patch embedding.
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # Allow different input size
        # B, C, H, W = x.shape
        # _assert(H == self.img_size[0], f"input image height ({H}) doesn't match model ({self.img_size[0]})")
        # _assert(W == self.img_size[1], f"input image width ({W}) doesn't match model ({self.img_size[1]})")
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

class LoRALayer():
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights

class PatchEmbedlora(PatchEmbed):
    """
    2D image to patch embedding.
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True,
                 r=8, lora_alpha=32, lora_dropout=0.05, merge_weights=True):
        super(PatchEmbedlora, self).__init__()
        #super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        if r > 0:
            self.lora_A = nn.Parameter(
                self.proj.weight.new_zeros((r * patch_size[0], in_chans * patch_size[0]))
            )
            self.lora_B = nn.Parameter(
              self.proj.weight.new_zeros((embed_dim//self.proj.groups*patch_size[0], r*patch_size[0]))
            )
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.proj.weight.requires_grad = False
        self.reset_parameters()
        self.merged = False
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def reset_parameters(self):
        self.proj.reset_parameters()
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode=True):
        #super(ConvLoRA, self).train(mode)
        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0:
                    # Make sure that the weights are not merged
                    self.proj.weight.data -= (self.lora_B @ self.lora_A).view(self.proj.weight.shape) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                if self.r > 0:
                    # Merge the weights and mark it
                    self.proj.weight.data += (self.lora_B @ self.lora_A).view(self.proj.weight.shape) * self.scaling
                self.merged = True

    def forward(self, x):
        # Allow different input size
        # B, C, H, W = x.shape
        # _assert(H == self.img_size[0], f"input image height ({H}) doesn't match model ({self.img_size[0]})")
        # _assert(W == self.img_size[1], f"input image width ({W}) doesn't match model ({self.img_size[1]})")
        if self.r > 0 and not self.merged:

            delta_weight = (self.lora_B @ self.lora_A).view(self.proj.weight.shape) * self.scaling
            self.proj.weight.data += delta_weight
            x = self.proj(x)
            self.proj.weight.data -= delta_weight
        else:
            x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x