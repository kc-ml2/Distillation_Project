# feature add: vit model DIVNO3 - has special tokens: register tokens
# usually register tokens are trash bins for massive energy for unimportant features
# so I thought that when building model, maintain register tokens
import torch
from torch import nn
import torch.nn.functional as F


class DINOv3_Wrapper(nn.Module):
    def __init__(self, base_model, model_register_tokens=4):
        # model_register_tokesn: existing register token number
        super().__init__()
        self.model = base_model
        self.num_reg = model_register_tokens

    def forward(self, x):
        x = self.model(x) # Output shape: [Batch, 201, 384]
        # Index 0: CLS token
        # Index 1 ~ 4: Register token
        # Index 5 ~ 끝: Patch tokens
        cls_token = x[:, 0:1, :]
        patch_tokens = x[:, (1 + self.num_reg):, :]
        return torch.cat([cls_token, patch_tokens], dim=1) # [Batch, 197, 384]
# end
