"""P1 sanity-check baseline: 3D CNN + concat fusion + 单头 Srg

设计原则:
- 物理一致性: Srg ∈ [0,1] → sigmoid 输出 (硬规则 #1)
- 容量小: 360 样本养不起大模型 (硬规则 #2 反过拟合)
- 不上 Cross-Attn (留给 P1.2 完整版); 这一步只验证 pipeline 通 + 看裸模型表现 vs phi-only baseline

输入: voxel (B, 1, D, D, D) + features (B, F)
输出: Srg_hat (B,) ∈ (0, 1)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Conv3dBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(c_in, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(c_out),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleSrgNet(nn.Module):
    """voxel CNN + features MLP → concat → head → sigmoid Srg.

    Args:
        n_features: csv 数值特征维度 (本数据集 = 20)
        cnn_channels: CNN 三层通道宽度
        feat_hidden: 特征 MLP 隐层
        head_hidden: 融合后 head 隐层
        dropout: P 后续 P3 MC-dropout 用; 这里默认 0.1 也起小正则
    """

    def __init__(
        self,
        n_features: int = 20,
        cnn_channels: tuple[int, ...] = (16, 32, 64),
        feat_hidden: int = 32,
        head_hidden: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        chs = (1,) + tuple(cnn_channels)
        self.cnn = nn.Sequential(*[Conv3dBlock(chs[i], chs[i + 1]) for i in range(len(cnn_channels))])
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.cnn_out_dim = cnn_channels[-1]

        self.feat_mlp = nn.Sequential(
            nn.Linear(n_features, feat_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feat_hidden, feat_hidden),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.Linear(self.cnn_out_dim + feat_hidden, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, voxel: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        h_v = self.cnn(voxel)
        h_v = self.global_pool(h_v).flatten(1)        # (B, C_last)
        h_f = self.feat_mlp(features)                  # (B, F)
        h = torch.cat([h_v, h_f], dim=1)
        out = self.head(h).squeeze(-1)                 # (B,) linear, 不加 sigmoid
        return out                                     # 物理约束在 train.py 用 logit_y / eval clip 实现


class SimpleSrgNetSigmoid(SimpleSrgNet):
    """与 SimpleSrgNet 同结构, 但加 sigmoid head (复现 P1.2b 0.493 baseline)."""
    def forward(self, voxel, features):
        return torch.sigmoid(super().forward(voxel, features))


class SimpleTauGateNet(SimpleSrgNet):
    """SimpleSrgNet + TauGate · 物理引导版本 (PoreFlowNet v2)

    设计差异 vs PoreFlowNet v1:
    - 保留 sigmoid head (复现 P1.2b 已知最佳协议)
    - 用 concat fusion (P2 数据显示 Cross-Attn 在单 token 上反而退步)
    - TauGate 调制 voxel pathway

    目标: P1.2b SimpleSrgNet+sigmoid (R²=0.493) + TauGate 物理引导 → R² ≥ 0.50
    """
    def __init__(self, n_features: int = 20, tau_idx: int = 1, **kwargs) -> None:
        super().__init__(n_features=n_features, **kwargs)
        from models_3d import TauGate
        self.tau_gate = TauGate(self.cnn_out_dim, tau_idx=tau_idx)

    def forward(self, voxel, features):
        h_v = self.cnn(voxel)
        h_v = self.global_pool(h_v).flatten(1)
        h_v = self.tau_gate(h_v, features)             # 🔑 物理引导
        h_f = self.feat_mlp(features)
        h = torch.cat([h_v, h_f], dim=1)
        return torch.sigmoid(self.head(h).squeeze(-1))   # sigmoid head


class PhiOnlyBaseline(nn.Module):
    """sanity check 的 sanity check: 仅靠 phi 一列预测 Srg, 看 CNN+特征模型有没有真"超过它"."""

    def __init__(self, n_features: int = 20) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_features, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
        )

    def forward(self, voxel: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        return self.mlp(features).squeeze(-1)   # linear, 不加 sigmoid
