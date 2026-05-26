# CLIPFD

基于 CLIP 全局—局部特征融合的 AI 图像伪造检测框架。

---

## 目录

- [项目简介](#项目简介)
- [任务定义](#任务定义)
- [模型结构](#模型结构)
- [项目特点](#项目特点)
- [项目目录](#项目目录)
- [环境依赖](#环境依赖)
- [数据组织方式](#数据组织方式)
- [训练说明](#训练说明)
- [测试说明](#测试说明)
- [消融实验说明](#消融实验说明)
- [评估指标与输出文件](#评估指标与输出文件)
- [参数说明](#参数说明)
- [使用建议](#使用建议)

---

## 项目简介

CLIPFD 是一个面向 AI 图像伪造检测任务的三分类框架，主要用于识别以下三类图像：

- 真实图像（Real）
- AI 生成图像（AI Generate）
- AI 修改图像（AI Edit）

本项目以冻结的 CLIP 图像编码器为主干，在此基础上构建全局分支、局部分支、特征融合模块和分类头，用于完成图像真实性判别任务。

当前框架支持以下三种三分类模式：

- `full`：全局特征与局部特征融合后进行三分类
- `global_only`：仅使用全局特征进行三分类
- `local_only`：仅使用局部特征进行三分类

该设计既可用于主模型训练，也便于开展消融实验。

---

## 任务定义

本项目将图像真实性识别建模为一个三分类问题：

| 类别编号 | 类别名称 | 含义 |
|---|---|---|
| 0 | 真实图 | 未经过 AI 生成或 AI 修改的图像 |
| 1 | AI生成图 | 由生成模型直接生成的图像 |
| 2 | AI修改图 | 在真实图基础上经 AI 工具局部修改的图像 |

同时，项目还支持一个辅助二分类任务：

- 真实图
- AI 介入图（包含 AI 生成图和 AI 修改图）

该辅助任务通过全局辅助头参与训练，用于增强模型的判别能力。

---

## 模型结构

当前模型整体流程如下：

1. 使用 CLIP 图像编码器提取：
   - 全局特征 `global_feat_raw`
   - patch token 特征 `patch_tokens`
2. 全局特征经过 `GlobalAdapter` 得到任务适配后的 `global_feat`
3. patch token 经 `LocalPatchBranch` 建模后得到局部特征 `local_feat`
4. 根据 `fusion_mode` 选择最终三分类输入特征：
   - `full`：使用融合特征
   - `global_only`：使用全局特征
   - `local_only`：使用局部特征
5. 通过最终分类头 `final_head` 输出三分类结果
6. 若启用 `use_global_aux_head`，则额外输出全局辅助二分类结果 `global_logits`

### 局部分支说明

局部分支的主要作用是：

- 将 patch token 还原为二维 patch 网格特征图
- 通过投影层将 token 特征映射到局部检测特征空间
- 通过残差局部卷积块建模 patch 间局部关系
- 通过全局平均池化得到局部表征
- 将局部表征映射到与全局特征相同的 768 维空间

当前默认实现中，局部分支返回的是 `local_feat`；早期基于 mask 的局部热图逻辑保留在代码注释中，当前主流程不直接输出 `local_heatmap`。

---

## 项目特点

- 基于 CLIP 图像编码器，充分利用预训练视觉表征能力
- 同时建模全局语义信息与局部异常特征
- 支持 `GlobalAdapter`，增强全局特征与任务的匹配程度
- 支持全局辅助二分类头，用于辅助监督
- 支持 `full / global_only / local_only` 三种三分类模式
- 支持完整训练、测试、评估报告与混淆矩阵保存流程
- 支持用于消融实验的灵活参数配置

---

## 项目目录

```text
CLIPFD/
├── models/
│   ├── assemble_model.py              # 模型组装
│   ├── branches/
│   │   ├── global_branch.py           # 全局特征适配层
│   │   └── local_branch.py            # 局部分支
│   ├── fusion/
│   │   └── fusion.py                  # 特征融合模块
│   ├── heads/
│   │   └── distinct_head.py           # 分类头
│   └── orign_CLIP_model/
│       └── feature_extract.py         # CLIP特征提取
│
├── options/
│   ├── base_options.py                # 基础参数配置
│   ├── train_options.py               # 训练参数配置
│   └── test_options.py                # 测试参数配置
│
├── trainer/
│   └── trainer.py                     # 训练与评估逻辑
│
├── utils/
│   ├── eval_report.py                 # 评估报告与混淆矩阵
│   └── training_monitor.py            # 训练过程可视化
│
├── train.py                           # 训练入口
├── test.py                            # 测试入口
├── pretrained_weights/                # 测试或推理使用的模型权重
├── datasets/                          # 数据集目录
└── checkpoints/                       # 训练输出目录






