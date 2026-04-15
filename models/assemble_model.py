# from .orign_CLIP_model import clip
# import torch.nn as nn
#
# # 该项目文件完成对主体、分支、头部架构的组合，向外提供统一的完整的分类模型接口
#
# # 由于是直接使用的clip源码，但是只用到了图像编码器部分，因此重构一下该部分即可；
# class CLIPModel(nn.Module):
#     def __init__(self, name="ViT-L/14", freeze=False,device="cpu"):
#         super().__init__()
#
#         self.backbone, self.preprocess = clip.load(name, device=device)
#         self.backbone_name = name
#         self.global_dim = 768
#         self.local_dim = 1024
#
#         if freeze:
#             self.freeze()
#
#     def extract_global_feat(self, x):
#         return self.backbone.encode_image(x)
#
#     def freeze(self):
#         for p in self.backbone.parameters():
#             p.requires_grad = False
#
#     def unfreeze(self):
#         for p in self.backbone.parameters():
#             p.requires_grad = True
#
#     def forward(self, x):
#         global_feat = self.extract_global_feat(x)
#         return {
#             "global_feat": global_feat
#         }


