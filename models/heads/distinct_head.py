import torch
import torch.nn as nn


class ClassifierHead(nn.Module):
    """
    通用分类头：
    输入一个表征向量，输出分类 logits
    输入:
        x: [B, in_dim]

    输出:
        {"logits": [B, num_classes]}

    说明:
    - 当 num_classes=1 时，可用于二分类，通常配合 BCEWithLogitsLoss
    - 当 num_classes>=2 时，可用于多分类，通常配合 CrossEntropyLoss
    """

    def __init__(self, in_dim: int, num_classes: int):
        """
        :param in_dim: 输入特征维度，例如全局分支的表征向量维度 768
        :param num_classes: 类别数
                            - 1: 二分类单 logit 输出
                            - 2/3/...: 多分类输出
        """
        super().__init__()

        if not isinstance(in_dim, int) or in_dim <= 0:
            raise ValueError(f"in_dim must be a positive integer, but got {in_dim}")

        if not isinstance(num_classes, int) or num_classes <= 0:
            raise ValueError(f"num_classes must be a positive integer, but got {num_classes}")

        self.in_dim = in_dim
        self.num_classes = num_classes
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> dict:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"x must be a torch.Tensor, but got {type(x)}")

        if x.dim() != 2:
            raise ValueError(f"Input x must have shape [B, in_dim], but got {tuple(x.shape)}")

        if x.size(1) != self.in_dim:
            raise ValueError(
                f"Input feature dim mismatch: expected {self.in_dim}, got {x.size(1)}"
            )

        logits = self.fc(x)   # [B, num_classes]

        return {
            "logits": logits
        }











