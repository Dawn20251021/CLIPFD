import torch
import torch.nn as nn


class FeatureFusion(nn.Module):
    """
    全局特征 + 局部特征 融合模块

    输入:
        global_feat: [B, 768]
        local_feat:  [B, 768]

    输出:
        {
            "fused_feat": [B, 768]
        }
    """
    def __init__(self, feat_dim: int = 768, dropout: float = 0.1):
        super().__init__()

        if not isinstance(feat_dim, int) or feat_dim <= 0:
            raise ValueError(f"feat_dim must be a positive integer, but got {feat_dim}")
        if dropout < 0:
            raise ValueError(f"dropout must be >= 0, but got {dropout}")

        self.feat_dim = feat_dim

        self.fusion = nn.Sequential(
            nn.LayerNorm(feat_dim * 2),
            nn.Linear(feat_dim * 2, feat_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, global_feat: torch.Tensor, local_feat: torch.Tensor) -> dict:
        if global_feat.dim() != 2 or local_feat.dim() != 2:
            raise ValueError(f"global_feat or local_feat must have shape [B, {self.feat_dim}], but got {tuple(global_feat.shape)}")

        if global_feat.size(1) != self.feat_dim or local_feat.size(1) != self.feat_dim :
            raise ValueError(f"Expected global_feat or local_feat dim={self.feat_dim}, but got {global_feat.size(1)}")


        if global_feat.size(0) != local_feat.size(0):
            raise ValueError(
                f"Batch size mismatch: global batch={global_feat.size(0)}, local batch={local_feat.size(0)}"
            )

        fused_input = torch.cat([global_feat, local_feat], dim=1)   # [B, 1536]
        fused_feat = self.fusion(fused_input)                       # [B, 768]

        return {
            "fused_feat": fused_feat
        }