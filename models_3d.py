"""P1.3a 3D backbone + 多模态融合模块库

设计原则:
- 小样本 (360) 友好: ResNet-18-tiny (base 32, ~2-3M params)
- 物理一致: sigmoid 头, Srg ∈ (0, 1)
- 融合方式可换: Concat / Cross-Attn / FiLM 共享 backbone API

参考:
- He 2016 (ResNet) - BasicBlock 结构
- Perez 2018 (FiLM) - feature-wise linear modulation
- Hu 2018 (SE) - 通道注意力
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================
# 3D ResNet (轻量版, 360 样本友好)
# =========================================================================

def conv3x3x3(c_in: int, c_out: int, stride: int = 1) -> nn.Conv3d:
    return nn.Conv3d(c_in, c_out, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, c_in: int, c_out: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = conv3x3x3(c_in, c_out, stride)
        self.bn1 = nn.BatchNorm3d(c_out)
        self.conv2 = conv3x3x3(c_out, c_out)
        self.bn2 = nn.BatchNorm3d(c_out)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or c_in != c_out:
            self.downsample = nn.Sequential(
                nn.Conv3d(c_in, c_out, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(c_out),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        return out


class ResNet3D(nn.Module):
    """轻量 3D ResNet · base_width=32, layers=(2,2,2,2) → ~2.5M params · 输出 (B, C_last)."""

    def __init__(
        self,
        in_channels: int = 1,
        base_width: int = 32,
        layers: tuple[int, ...] = (2, 2, 2, 2),
        global_pool: bool = True,
    ) -> None:
        super().__init__()
        self.base_width = base_width
        self.global_pool = global_pool

        widths = [base_width * (2 ** i) for i in range(len(layers))]   # 32, 64, 128, 256

        # stem: 128^3 → 64^3
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, base_width, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm3d(base_width),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),    # 64^3 → 32^3
        )

        # stages: 32^3 → 32^3 → 16^3 → 8^3 → 4^3
        self.stages = nn.ModuleList()
        c_prev = base_width
        for i, (n_blocks, w) in enumerate(zip(layers, widths)):
            stride_first = 1 if i == 0 else 2
            blocks = []
            blocks.append(BasicBlock3D(c_prev, w, stride=stride_first))
            for _ in range(n_blocks - 1):
                blocks.append(BasicBlock3D(w, w, stride=1))
            self.stages.append(nn.Sequential(*blocks))
            c_prev = w

        self.out_channels = c_prev
        self.gap = nn.AdaptiveAvgPool3d(1) if global_pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        if self.global_pool:
            x = self.gap(x).flatten(1)
        return x


# =========================================================================
# 融合模块: Concat / Cross-Attn / FiLM
# =========================================================================

class ConcatFusion(nn.Module):
    """B1: 直接 concat (basline 实现)."""

    def __init__(self, voxel_dim: int, feat_dim: int, out_dim: int) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(voxel_dim + feat_dim, out_dim),
            nn.ReLU(inplace=True),
        )
        self.out_dim = out_dim

    def forward(self, h_v: torch.Tensor, h_f: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.cat([h_v, h_f], dim=1))


class CrossAttnFusion(nn.Module):
    """B1 王鹏暂定: feature query, voxel as key/value (单头版).

    h_f → Q  (B, 1, D)   特征查询
    h_v → K, V  (B, 1, D) 体素响应  (单 token 版本; 真正的 Cross-Attn 应保留体素 spatial)

    实现策略: 由于体素已 GAP 成单向量, 这里实现 "scalar 注意力": 用 features 决定 voxel 的通道权重.
    更彻底的版本(spatial Cross-Attn) 留给 P1.3b · 因为要 backbone 输出 spatial map (不 GAP).
    """

    def __init__(self, voxel_dim: int, feat_dim: int, out_dim: int, heads: int = 4) -> None:
        super().__init__()
        # 投影到共享 embed 维
        self.embed_dim = max(voxel_dim, feat_dim)
        self.q_proj = nn.Linear(feat_dim, self.embed_dim)
        self.k_proj = nn.Linear(voxel_dim, self.embed_dim)
        self.v_proj = nn.Linear(voxel_dim, self.embed_dim)
        self.heads = heads
        self.out_proj = nn.Linear(self.embed_dim, out_dim)
        self.feat_residual = nn.Linear(feat_dim, out_dim)
        self.out_dim = out_dim

    def forward(self, h_v: torch.Tensor, h_f: torch.Tensor) -> torch.Tensor:
        # (B, embed_dim)
        q = self.q_proj(h_f)
        k = self.k_proj(h_v)
        v = self.v_proj(h_v)
        # 单 token 单 head: scaled dot-product
        scale = self.embed_dim ** -0.5
        attn = torch.sigmoid((q * k).sum(dim=-1, keepdim=True) * scale)   # (B, 1)
        # 用 attention 权重门控 V, 加 feature residual
        attended = attn * v
        out = self.out_proj(F.relu(attended)) + self.feat_residual(h_f)
        return out


class FiLMFusion(nn.Module):
    """B2 FiLM (Perez 2018): features 生成 (γ, β), 调制 voxel 特征.

    h_v_modulated = γ ⊙ h_v + β
    """

    def __init__(self, voxel_dim: int, feat_dim: int, out_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.gamma_net = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, voxel_dim),
        )
        self.beta_net = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, voxel_dim),
        )
        self.fc = nn.Sequential(
            nn.Linear(voxel_dim + feat_dim, out_dim),
            nn.ReLU(inplace=True),
        )
        self.out_dim = out_dim

    def forward(self, h_v: torch.Tensor, h_f: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma_net(h_f)
        beta = self.beta_net(h_f)
        h_v_mod = gamma * h_v + beta
        return self.fc(torch.cat([h_v_mod, h_f], dim=1))


# =========================================================================
# 整体网络
# =========================================================================

FUSION_REGISTRY = {
    "concat": ConcatFusion,
    "cross_attn": CrossAttnFusion,
    "film": FiLMFusion,
}


# =========================================================================
# Backbone 库 (轻量, 360 样本友好, < 2M params)
# =========================================================================

class ResNet10Tiny(nn.Module):
    """ResNet10-tiny: base=16, layers=(1,1,1,1), ~500K params.

    对照 R18 (8.3M, 过拟合) 看 ResNet 风格本身在小样本是否可行.
    """

    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        cnn = ResNet3D(in_channels=in_channels, base_width=16, layers=(1, 1, 1, 1))
        self.backbone = cnn
        self.out_channels = cnn.out_channels   # 128

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class InceptionBlock3D(nn.Module):
    """MS-PoreNet 用的 3D Inception 块: 1x1x1, 3x3x3, 5x5x5 三个分支并行."""

    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        c_branch = c_out // 4
        self.b1 = nn.Sequential(
            nn.Conv3d(c_in, c_branch, kernel_size=1, bias=False),
            nn.BatchNorm3d(c_branch),
            nn.ReLU(inplace=True),
        )
        self.b3 = nn.Sequential(
            nn.Conv3d(c_in, c_branch, kernel_size=1, bias=False),
            nn.BatchNorm3d(c_branch),
            nn.ReLU(inplace=True),
            nn.Conv3d(c_branch, c_branch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(c_branch),
            nn.ReLU(inplace=True),
        )
        self.b5 = nn.Sequential(
            nn.Conv3d(c_in, c_branch, kernel_size=1, bias=False),
            nn.BatchNorm3d(c_branch),
            nn.ReLU(inplace=True),
            nn.Conv3d(c_branch, c_branch, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm3d(c_branch),
            nn.ReLU(inplace=True),
        )
        self.bp = nn.Sequential(
            nn.MaxPool3d(3, stride=1, padding=1),
            nn.Conv3d(c_in, c_branch, kernel_size=1, bias=False),
            nn.BatchNorm3d(c_branch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.b1(x), self.b3(x), self.b5(x), self.bp(x)], dim=1)


class MSPoreNet(nn.Module):
    """MS-PoreNet: 3D Inception 多尺度卷积 backbone, ~800K params.

    Stem: 5×5×5 stride 2 → 64³
    Inception block × 3 (含 maxpool 下采样): 64³ → 32³ → 16³ → 8³
    GAP → C 维.
    """

    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 24, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),   # → 32³
        )
        self.inc1 = InceptionBlock3D(24, 64)    # 64 ch
        self.pool1 = nn.MaxPool3d(2)            # → 16³
        self.inc2 = InceptionBlock3D(64, 96)    # 96 ch
        self.pool2 = nn.MaxPool3d(2)            # → 8³
        self.inc3 = InceptionBlock3D(96, 128)
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.out_channels = 128

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.inc1(x); x = self.pool1(x)
        x = self.inc2(x); x = self.pool2(x)
        x = self.inc3(x)
        return self.gap(x).flatten(1)


class MBConvBlock3D(nn.Module):
    """MBConv (depthwise separable + SE) for CoAtNet 风格的 conv stage."""

    def __init__(self, c_in: int, c_out: int, expansion: int = 4, stride: int = 1) -> None:
        super().__init__()
        c_hidden = c_in * expansion
        self.use_residual = (stride == 1 and c_in == c_out)
        layers = []
        if expansion != 1:
            layers += [nn.Conv3d(c_in, c_hidden, 1, bias=False),
                       nn.BatchNorm3d(c_hidden), nn.GELU()]
        layers += [
            nn.Conv3d(c_hidden, c_hidden, 3, stride=stride, padding=1, groups=c_hidden, bias=False),
            nn.BatchNorm3d(c_hidden), nn.GELU(),
            # SE
            nn.AdaptiveAvgPool3d(1),     # → squeeze; 用一个 conv-1x1 模拟
        ]
        self.proj1 = nn.Sequential(*layers[:-1])    # 不含 GAP
        self.se_fc = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(c_hidden, c_hidden // 4, 1),
            nn.GELU(),
            nn.Conv3d(c_hidden // 4, c_hidden, 1),
            nn.Sigmoid(),
        )
        self.proj2 = nn.Sequential(
            nn.Conv3d(c_hidden, c_out, 1, bias=False),
            nn.BatchNorm3d(c_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj1(x)
        h = h * self.se_fc(h)
        h = self.proj2(h)
        if self.use_residual:
            h = h + x
        return h


class TransformerEncoderLayer3D(nn.Module):
    """标准 Transformer encoder block, 用于 PoreFormer / PoreCoAt 后期 stage."""

    def __init__(self, dim: int, heads: int = 4, mlp_ratio: float = 4.0, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C) — N tokens
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class PoreCoAt(nn.Module):
    """PoreCoAt: CoAtNet 简版 (2 conv stages + 2 attn stages), ~1.5M params.

    Stem: stride-2 conv → 64³
    Stage 1: MBConv × 2, 64³ → 32³
    Stage 2: MBConv × 2, 32³ → 16³
    Stage 3: TransEncoder × 2 (16³ = 4096 tokens 太多, GAP 到 8³ = 512 tokens)
    Stage 4: TransEncoder × 2
    GAP → C
    """

    def __init__(self, in_channels: int = 1, dim: int = 96) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 24, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(24), nn.GELU(),                # 64³
            nn.MaxPool3d(2),                              # 32³
        )
        self.s1 = nn.Sequential(MBConvBlock3D(24, 48, stride=2),
                                MBConvBlock3D(48, 48))     # 16³
        self.s2 = nn.Sequential(MBConvBlock3D(48, dim, stride=2),
                                MBConvBlock3D(dim, dim))   # 8³ = 512 tokens
        # tokens for transformer
        self.attn_layers = nn.ModuleList([
            TransformerEncoderLayer3D(dim, heads=4, mlp_ratio=2.0, dropout=0.1) for _ in range(2)
        ])
        self.norm = nn.LayerNorm(dim)
        self.out_channels = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.s1(x)
        x = self.s2(x)                  # (B, dim, 8, 8, 8)
        B, C, D1, D2, D3 = x.shape
        tokens = x.flatten(2).transpose(1, 2)   # (B, 512, dim)
        for layer in self.attn_layers:
            tokens = layer(tokens)
        tokens = self.norm(tokens)
        return tokens.mean(dim=1)       # GAP over tokens → (B, dim)


class PoreFormer(nn.Module):
    """PoreFormer: Conv stem + Transformer encoder × 4, ~1.5M params.

    跟 PoreCoAt 不同点: stem 直接降到 8³ 进 transformer (无 conv stage), 类似 ViT.
    论文叙事: '为数字岩心代理设计的轻量 vision transformer'.
    """

    def __init__(self, in_channels: int = 1, dim: int = 128, depth: int = 4) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=4, stride=4, padding=0, bias=False),  # 32³
            nn.BatchNorm3d(32), nn.GELU(),
            nn.Conv3d(32, 64, kernel_size=2, stride=2, padding=0, bias=False),           # 16³
            nn.BatchNorm3d(64), nn.GELU(),
            nn.Conv3d(64, dim, kernel_size=2, stride=2, padding=0, bias=False),          # 8³
            nn.BatchNorm3d(dim),
        )
        # learnable positional embed for 8³ = 512 tokens
        self.pos_embed = nn.Parameter(torch.zeros(1, 512, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer3D(dim, heads=4, mlp_ratio=2.0, dropout=0.1) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.out_channels = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        B, C, D, _, _ = x.shape
        tokens = x.flatten(2).transpose(1, 2) + self.pos_embed
        for layer in self.layers:
            tokens = layer(tokens)
        tokens = self.norm(tokens)
        return tokens.mean(dim=1)       # (B, dim)


# =========================================================================
# 通用包装: backbone + features MLP + fusion + sigmoid head
# =========================================================================

class GenericSrgNet(nn.Module):
    """适配任意 backbone (输出 (B, C)) + features MLP + 任意 fusion + sigmoid 头."""

    def __init__(
        self,
        backbone: nn.Module,
        n_features: int = 20,
        feat_hidden: int = 64,
        fusion: str = "cross_attn",
        head_hidden: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.cnn = backbone
        v_dim = backbone.out_channels
        self.feat_mlp = nn.Sequential(
            nn.Linear(n_features, feat_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feat_hidden, feat_hidden),
            nn.ReLU(inplace=True),
        )
        fusion_cls = FUSION_REGISTRY[fusion]
        self.fusion = fusion_cls(voxel_dim=v_dim, feat_dim=feat_hidden, out_dim=head_hidden)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(head_hidden, 1))

    def forward(self, voxel: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        h_v = self.cnn(voxel)
        h_f = self.feat_mlp(features)
        h = self.fusion(h_v, h_f)
        return self.head(h).squeeze(-1)   # linear, 由 train.py logit_y 或 clip 处理 [0,1] 约束


# =========================================================================
# PoreFlowNet · 我们提出的方法
# =========================================================================

class TauGate(nn.Module):
    """物理引导 attention: 用 tau (迂曲度) 调制 voxel pathway.

    动机: P1.5 单 feature 分析发现 tau 是 Srg 最强单 feature (LORO R²=0.449,
    Pearson +0.723). 物理意义: 流体路径越曲折 → 残余气越难被驱替.
    实现: 从 features 拆出 tau → small MLP → sigmoid gate (0,1) → 调制 voxel pathway.

    Why useful: 给模型注入显式物理先验, voxel pathway 在 tau 高时被放大,
    在 tau 低时被抑制 (此时 features 已足够主导).
    """

    def __init__(self, voxel_dim: int, tau_idx: int = 1, hidden: int = 16) -> None:
        super().__init__()
        self.tau_idx = tau_idx   # tau 在 features 中的列索引 (经标准化后)
        self.gate_net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Linear(hidden, voxel_dim),
            nn.Sigmoid(),
        )

    def forward(self, h_v: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        tau = features[:, self.tau_idx:self.tau_idx + 1]   # (B, 1)
        gate = self.gate_net(tau)                            # (B, voxel_dim)
        return h_v * gate                                    # 物理引导 channel-wise gating


class PoreFlowNet(nn.Module):
    """PoreFlowNet · 数字岩心两相流端到端代理网络.

    设计:
      - Backbone: 轻量 3D CNN (16/32/64), P1.5 验证 75K params 是 360 样本上的甜点
      - TauGate (物理引导): tau → channel gate, 调制 voxel features
      - Cross-Attn Fusion: features 作 query, voxel 作 K/V, 防 head 走 features 捷径
      - 训练协议: augment (XY rot/flip) + cosine LR + long training (80ep) + early-stop val_mse
      - Head: linear out + clip [0,1] (P1.4b 验证 sigmoid 头不是 bug, linear 等价)

    创新点 (论文消融):
      1. TauGate 物理引导 (vs 无 gate)
      2. Cross-Attn fusion (vs concat)
      3. 训练协议 (vs 短训 + 无增强)
    """

    def __init__(self, n_features: int = 20, tau_idx: int = 1) -> None:
        super().__init__()
        # CNN backbone (Simple 76K)
        from model import Conv3dBlock
        self.cnn = nn.Sequential(
            Conv3dBlock(1, 16),
            Conv3dBlock(16, 32),
            Conv3dBlock(32, 64),
        )
        self.gap = nn.AdaptiveAvgPool3d(1)
        v_dim = 64

        # 物理引导 TauGate
        self.tau_gate = TauGate(v_dim, tau_idx=tau_idx)

        # features MLP
        self.feat_mlp = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 64),
            nn.GELU(),
        )
        f_dim = 64

        # Cross-Attn fusion (q=features, k=v=voxel)
        embed_dim = max(v_dim, f_dim)
        self.q_proj = nn.Linear(f_dim, embed_dim)
        self.k_proj = nn.Linear(v_dim, embed_dim)
        self.v_proj = nn.Linear(v_dim, embed_dim)
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.f_residual = nn.Linear(f_dim, embed_dim)

        # head
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def forward(self, voxel: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        # voxel pathway
        h_v = self.gap(self.cnn(voxel)).flatten(1)         # (B, 64)
        h_v = self.tau_gate(h_v, features)                  # 物理引导 gate

        # feature pathway
        h_f = self.feat_mlp(features)                       # (B, 64)

        # Cross-Attn (单 token 单头版本)
        q = self.q_proj(h_f); k = self.k_proj(h_v); v = self.v_proj(h_v)
        scale = q.shape[-1] ** -0.5
        attn = torch.sigmoid((q * k).sum(dim=-1, keepdim=True) * scale)
        attended = self.attn_norm(attn * v + self.f_residual(h_f))

        return self.head(attended).squeeze(-1)              # linear out, eval 时 clip


# === ablation 变体 (用同一基础, 切换组件) ===

class PoreFlowNet_NoTauGate(PoreFlowNet):
    def forward(self, voxel, features):
        h_v = self.gap(self.cnn(voxel)).flatten(1)         # 不调 tau gate
        h_f = self.feat_mlp(features)
        q = self.q_proj(h_f); k = self.k_proj(h_v); v = self.v_proj(h_v)
        scale = q.shape[-1] ** -0.5
        attn = torch.sigmoid((q * k).sum(dim=-1, keepdim=True) * scale)
        attended = self.attn_norm(attn * v + self.f_residual(h_f))
        return self.head(attended).squeeze(-1)


class PoreFlowNet_NoCrossAttn(nn.Module):
    """Concat fusion 版本 (替 Cross-Attn)."""

    def __init__(self, n_features: int = 20, tau_idx: int = 1) -> None:
        super().__init__()
        from model import Conv3dBlock
        self.cnn = nn.Sequential(Conv3dBlock(1, 16), Conv3dBlock(16, 32), Conv3dBlock(32, 64))
        self.gap = nn.AdaptiveAvgPool3d(1)
        v_dim = 64
        self.tau_gate = TauGate(v_dim, tau_idx=tau_idx)
        self.feat_mlp = nn.Sequential(
            nn.Linear(n_features, 64), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(64, 64), nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(v_dim + 64, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1)
        )

    def forward(self, voxel, features):
        h_v = self.gap(self.cnn(voxel)).flatten(1)
        h_v = self.tau_gate(h_v, features)
        h_f = self.feat_mlp(features)
        return self.head(torch.cat([h_v, h_f], dim=1)).squeeze(-1)


class VoxelOnlyCNN(nn.Module):
    """voxel-only baseline (无 features)."""

    def __init__(self, n_features: int = 20) -> None:
        super().__init__()
        from model import Conv3dBlock
        self.cnn = nn.Sequential(Conv3dBlock(1, 16), Conv3dBlock(16, 32), Conv3dBlock(32, 64))
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Dropout(0.1), nn.Linear(32, 1))

    def forward(self, voxel, features):
        h = self.gap(self.cnn(voxel)).flatten(1)
        return self.head(h).squeeze(-1)


def make_model(name: str, n_features: int = 20) -> nn.Module:
    """Factory · 一行换 backbone."""
    # 来自 model.py 的轻量模型 (P2 v2)
    if name == "simple_taugate":
        from model import SimpleTauGateNet
        return SimpleTauGateNet(n_features=n_features, tau_idx=1)
    if name == "simple_sigmoid":
        from model import SimpleSrgNetSigmoid
        return SimpleSrgNetSigmoid(n_features=n_features)
    if name == "simple":
        from model import SimpleSrgNet
        return SimpleSrgNet(n_features=n_features)
    if name == "poreflownet":
        return PoreFlowNet(n_features=n_features, tau_idx=1)   # tau 在 csv 第 2 列 (index 1)
    if name == "poreflownet_no_taugate":
        return PoreFlowNet_NoTauGate(n_features=n_features, tau_idx=1)
    if name == "poreflownet_no_crossattn":
        return PoreFlowNet_NoCrossAttn(n_features=n_features, tau_idx=1)
    if name == "voxel_only_cnn":
        return VoxelOnlyCNN(n_features=n_features)
    if name == "resnet10_tiny_crossattn":
        return GenericSrgNet(ResNet10Tiny(), n_features=n_features, fusion="cross_attn")
    if name == "ms_porenet_crossattn":
        return GenericSrgNet(MSPoreNet(), n_features=n_features, fusion="cross_attn")
    if name == "porecoat_crossattn":
        return GenericSrgNet(PoreCoAt(), n_features=n_features, fusion="cross_attn")
    if name == "poreformer_crossattn":
        return GenericSrgNet(PoreFormer(), n_features=n_features, fusion="cross_attn")
    raise ValueError(name)


class ResNetSrgNet(nn.Module):
    """3D ResNet + 18 维 features MLP + 可换 fusion + sigmoid 头.

    Args:
        n_features: csv 数值特征维度 (本数据集 = 20)
        base_width: ResNet 起始通道
        layers: 4 个 stage 的 block 数
        feat_hidden: features MLP 隐层
        fusion: 'concat' / 'cross_attn' / 'film'
        head_hidden: 融合后 head 隐层
        dropout: 全局 dropout
    """

    def __init__(
        self,
        n_features: int = 20,
        base_width: int = 32,
        layers: tuple[int, ...] = (2, 2, 2, 2),
        feat_hidden: int = 64,
        fusion: str = "concat",
        head_hidden: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.cnn = ResNet3D(in_channels=1, base_width=base_width, layers=layers)
        v_dim = self.cnn.out_channels

        self.feat_mlp = nn.Sequential(
            nn.Linear(n_features, feat_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feat_hidden, feat_hidden),
            nn.ReLU(inplace=True),
        )

        fusion_cls = FUSION_REGISTRY[fusion]
        self.fusion = fusion_cls(voxel_dim=v_dim, feat_dim=feat_hidden, out_dim=head_hidden)
        self.fusion_name = fusion

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, voxel: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        h_v = self.cnn(voxel)
        h_f = self.feat_mlp(features)
        h = self.fusion(h_v, h_f)
        return self.head(h).squeeze(-1)   # linear out; 物理约束由 train.py logit_y / clip 处理
