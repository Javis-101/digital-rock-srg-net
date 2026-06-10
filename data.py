"""数字岩心数据集 · 读 voxel_{128,256}.npz · 按母岩心 prefix 分组 split

硬规则:
- csv 已经在 p0_cache.py 按列名读完合进 npz, 此处直接用 npz
- split 必须按 prefix (母岩心 id) 分组, 同岩心子块禁止跨集 (防 leakage)
- voxel 保持 uint8 在 RAM, __getitem__ 时 cast 到 float32, 否则 256^3 一份就是 5.6 GB
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class CachedNPZ:
    voxel: np.ndarray       # (N, D, D, D) uint8
    features: np.ndarray    # (N, F) float32
    Srg: np.ndarray         # (N,) float32  ∈ [0, 1]
    K: np.ndarray           # (N,) float32  > 0
    logK: np.ndarray        # (N,) float32  log10(K)
    sample_id: np.ndarray   # (N,) <U8
    prefix: np.ndarray      # (N,) <U2  母岩心 id
    feature_names: np.ndarray
    D: int
    n: int

    @classmethod
    def load(cls, path: str | Path) -> "CachedNPZ":
        z = np.load(path, allow_pickle=True)   # sample_id 是 object 数组, 需 allow_pickle
        voxel = z["voxel"]
        return cls(
            voxel=voxel,
            features=z["features"].astype(np.float32),
            Srg=z["Srg"].astype(np.float32),
            K=z["K"].astype(np.float32),
            logK=z["logK"].astype(np.float32),
            sample_id=z["sample_id"],
            prefix=z["prefix"],
            feature_names=z["feature_names"],
            D=int(voxel.shape[1]),
            n=int(voxel.shape[0]),
        )


def group_kfold_indices(prefix: np.ndarray, val_prefix: str) -> tuple[np.ndarray, np.ndarray]:
    """leave-one-rock-out: val = 指定母岩心的所有子块, train = 其余."""
    val = np.where(prefix == val_prefix)[0]
    train = np.where(prefix != val_prefix)[0]
    return train, val


class RockDataset(Dataset):
    """单分辨率岩心数据集.

    Args:
        cache: CachedNPZ
        idx: 用本子集的样本索引 (从 group_kfold_indices 来)
        feat_mean / feat_std: 训练集统计的特征归一化参数 (由外部传入, 防 leakage)
        augment: 训练集开数据增强 (XY 平面 90° 旋转 + 镜像; Z 是贯通方向不旋转)
    """

    def __init__(
        self,
        cache: CachedNPZ,
        idx: np.ndarray,
        feat_mean: np.ndarray,
        feat_std: np.ndarray,
        augment: bool = False,
    ) -> None:
        self.cache = cache
        self.idx = idx
        self.feat_mean = feat_mean.astype(np.float32)
        self.feat_std = np.where(feat_std > 1e-8, feat_std, 1.0).astype(np.float32)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.idx)

    def _augment_xy(self, vox: np.ndarray) -> np.ndarray:
        # vox: (D, D, D), Z 是第 0 轴; 在 (Y, X) = (axis 1, 2) 上做 4 种旋转 + 2 种镜像
        k = np.random.randint(4)
        if k:
            vox = np.rot90(vox, k=k, axes=(1, 2))
        if np.random.rand() < 0.5:
            vox = vox[:, ::-1, :]
        if np.random.rand() < 0.5:
            vox = vox[:, :, ::-1]
        return np.ascontiguousarray(vox)

    def __getitem__(self, i: int) -> dict:
        j = int(self.idx[i])
        vox = self.cache.voxel[j]                 # uint8 (D, D, D)
        if self.augment:
            vox = self._augment_xy(vox)
        vox_t = torch.from_numpy(vox.astype(np.float32))[None]   # (1, D, D, D)
        feat = self.cache.features[j]
        feat_norm = (feat - self.feat_mean) / self.feat_std
        return {
            "voxel": vox_t,
            "features": torch.from_numpy(feat_norm),
            "Srg": torch.tensor(self.cache.Srg[j], dtype=torch.float32),
            "logK": torch.tensor(self.cache.logK[j], dtype=torch.float32),
            "sample_id": str(self.cache.sample_id[j]),
            "prefix": str(self.cache.prefix[j]),
        }


def compute_feat_stats(features: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """从 train_idx 子集算 mean/std, 防 leakage."""
    sub = features[train_idx]
    return sub.mean(axis=0), sub.std(axis=0)
