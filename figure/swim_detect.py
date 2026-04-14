from __future__ import annotations

from dataclasses import dataclass
from zfish._io import import16chFlt
from typing import List, Sequence, Tuple, Optional

import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset


# ============================================================
# Utilities
# ============================================================

def get_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Dataset
# ============================================================

@dataclass
class WindowIndex:
    recording_idx: int
    start: int
    end: int


class SwimBoutDataset(Dataset):
    """
    Dataset for framewise swim-bout detection from 2-channel ephys.

    Each item is:
        x_window: shape (2, window_size)
        y_window: shape (1, window_size)
    """

    def __init__(
        self,
        signals: Sequence[np.ndarray],
        labels: Sequence[np.ndarray],
        window_size: int = 6000,
        stride: int = 1000,
        normalize: bool = True,
        training: bool = True,
    ) -> None:
        super().__init__()

        if len(signals) != len(labels):
            raise ValueError("signals and labels must have the same length.")

        self.signals: List[np.ndarray] = []
        self.labels: List[np.ndarray] = []
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.training = training

        self.indices: List[WindowIndex] = []

        for i, (x, y) in enumerate(zip(signals, labels)):
            if x.ndim != 2 or x.shape[0] != 2:
                raise ValueError(f"signals[{i}] must have shape (2, n_frames), got {x.shape}")
            if y.ndim != 1:
                raise ValueError(f"labels[{i}] must have shape (n_frames,), got {y.shape}")
            if x.shape[1] != y.shape[0]:
                raise ValueError(
                    f"signals[{i}] and labels[{i}] length mismatch: "
                    f"{x.shape[1]} vs {y.shape[0]}"
                )
            if x.shape[1] < self.window_size:
                raise ValueError(
                    f"signals[{i}] is shorter than window_size: {x.shape[1]} < {self.window_size}"
                )

            x = x.astype(np.float32, copy=True)
            y = y.astype(np.float32, copy=True)

            if normalize:
                mean = x.mean(axis=1, keepdims=True)
                std = x.std(axis=1, keepdims=True)
                std = np.maximum(std, 1e-6)
                x = (x - mean) / std

            self.signals.append(x)
            self.labels.append(y)

            n_frames = x.shape[1]
            starts = list(range(0, n_frames - self.window_size + 1, self.stride))
            if len(starts) == 0 or starts[-1] != n_frames - self.window_size:
                starts.append(n_frames - self.window_size)

            for start in starts:
                end = start + self.window_size
                self.indices.append(WindowIndex(i, start, end))

    def __len__(self) -> int:
        return len(self.indices)

    def _augment(self, x: np.ndarray) -> np.ndarray:
        """
        Lightweight augmentation for robustness.
        x shape: (2, T)
        """
        scale = np.random.uniform(0.9, 1.1, size=(2, 1)).astype(np.float32)
        x = x * scale

        noise_std = np.random.uniform(0.0, 0.05)
        x = x + np.random.randn(*x.shape).astype(np.float32) * noise_std

        if np.random.rand() < 0.1:
            ch = np.random.randint(0, 2)
            x[ch] *= np.random.uniform(0.0, 0.3)

        return x

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        win = self.indices[idx]
        x = self.signals[win.recording_idx][:, win.start:win.end].copy()
        y = self.labels[win.recording_idx][win.start:win.end].copy()

        if self.training:
            x = self._augment(x)

        x_tensor = torch.from_numpy(x)
        y_tensor = torch.from_numpy(y[None, :])

        return x_tensor, y_tensor


# ============================================================
# Model blocks
# ============================================================

class ConvBlock1D(nn.Module):
    """Two-layer 1D conv block with BatchNorm and ReLU."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 7) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock1D(nn.Module):
    """Downsampling block: MaxPool -> ConvBlock"""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 7) -> None:
        super().__init__()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv = ConvBlock1D(in_ch, out_ch, kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.conv(x)
        return x


class UpBlock1D(nn.Module):
    """
    Upsampling block:
    - interpolate to match skip length
    - concatenate skip
    - ConvBlock
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, kernel_size: int = 7) -> None:
        super().__init__()
        self.conv = ConvBlock1D(in_ch + skip_ch, out_ch, kernel_size=kernel_size)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


# ============================================================
# 1D U-Net
# ============================================================

class UNet1D(nn.Module):
    """
    1D U-Net for framewise segmentation.

    Input:
        (B, 2, T)
    Output:
        (B, 1, T)
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        base_channels: int = 32,
        kernel_size: int = 7,
    ) -> None:
        super().__init__()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 16

        self.inc = ConvBlock1D(in_channels, c1, kernel_size=kernel_size)
        self.down1 = DownBlock1D(c1, c2, kernel_size=kernel_size)
        self.down2 = DownBlock1D(c2, c3, kernel_size=kernel_size)
        self.down3 = DownBlock1D(c3, c4, kernel_size=kernel_size)
        self.down4 = DownBlock1D(c4, c5, kernel_size=kernel_size)

        self.up1 = UpBlock1D(c5, c4, c4, kernel_size=kernel_size)
        self.up2 = UpBlock1D(c4, c3, c3, kernel_size=kernel_size)
        self.up3 = UpBlock1D(c3, c2, c2, kernel_size=kernel_size)
        self.up4 = UpBlock1D(c2, c1, c1, kernel_size=kernel_size)

        self.outc = nn.Conv1d(c1, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits


# ============================================================
# Losses
# ============================================================

class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation."""

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = probs.contiguous().view(probs.shape[0], -1)
        targets = targets.contiguous().view(targets.shape[0], -1)

        intersection = (probs * targets).sum(dim=1)
        denom = probs.sum(dim=1) + targets.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """BCE with logits + Dice loss."""

    def __init__(self, pos_weight: Optional[float] = None, dice_weight: float = 0.5) -> None:
        super().__init__()

        if pos_weight is not None:
            self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))
        else:
            self.bce = nn.BCEWithLogitsLoss()

        self.dice = DiceLoss()
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if hasattr(self.bce, "pos_weight") and self.bce.pos_weight is not None:
            self.bce.pos_weight = self.bce.pos_weight.to(logits.device)

        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        dice = self.dice(probs, targets)
        return bce + self.dice_weight * dice


# ============================================================
# Training / evaluation
# ============================================================

def compute_binary_metrics(
    probs: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> dict:
    """Compute simple framewise metrics."""
    preds = (probs >= threshold).float()

    tp = (preds * targets).sum().item()
    fp = (preds * (1.0 - targets)).sum().item()
    fn = ((1.0 - preds) * targets).sum().item()
    tn = ((1.0 - preds) * (1.0 - targets)).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: Optional[float] = 1.0,
) -> Tuple[float, dict]:
    model.train()

    total_loss = 0.0
    metric_sums = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}
    n_batches = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()

        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            metrics = compute_binary_metrics(probs, y)
            for k in metric_sums:
                metric_sums[k] += metrics[k]

        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    avg_metrics = {k: v / max(n_batches, 1) for k, v in metric_sums.items()}
    return avg_loss, avg_metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, dict]:
    model.eval()

    total_loss = 0.0
    metric_sums = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}
    n_batches = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item()

        probs = torch.sigmoid(logits)
        metrics = compute_binary_metrics(probs, y)
        for k in metric_sums:
            metric_sums[k] += metrics[k]

        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    avg_metrics = {k: v / max(n_batches, 1) for k, v in metric_sums.items()}
    return avg_loss, avg_metrics


# ============================================================
# Full-recording inference
# ============================================================

@torch.no_grad()
def predict_full_recording(
    model: nn.Module,
    signal: np.ndarray,
    device: torch.device,
    window_size: int = 6000,
    stride: int = 1000,
    normalize: bool = True,
    batch_size: int = 16,
) -> np.ndarray:
    """
    Reconstruct framewise probabilities for a full recording.
    """
    if signal.ndim != 2 or signal.shape[0] != 2:
        raise ValueError(f"signal must have shape (2, n_frames), got {signal.shape}")

    x = signal.astype(np.float32, copy=True)

    if normalize:
        mean = x.mean(axis=1, keepdims=True)
        std = x.std(axis=1, keepdims=True)
        std = np.maximum(std, 1e-6)
        x = (x - mean) / std

    n_frames = x.shape[1]
    if n_frames < window_size:
        raise ValueError(f"Recording length {n_frames} is smaller than window_size {window_size}")

    starts = list(range(0, n_frames - window_size + 1, stride))
    if len(starts) == 0 or starts[-1] != n_frames - window_size:
        starts.append(n_frames - window_size)

    prob_sum = np.zeros(n_frames, dtype=np.float32)
    prob_count = np.zeros(n_frames, dtype=np.float32)

    model.eval()

    batch_windows = []
    batch_ranges = []

    for start in starts:
        end = start + window_size
        x_win = x[:, start:end]
        batch_windows.append(x_win)
        batch_ranges.append((start, end))

        if len(batch_windows) == batch_size:
            x_batch = torch.from_numpy(np.stack(batch_windows, axis=0)).to(device)
            logits = model(x_batch)
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()

            for p, (s, e) in zip(probs, batch_ranges):
                prob_sum[s:e] += p
                prob_count[s:e] += 1.0

            batch_windows = []
            batch_ranges = []

    if batch_windows:
        x_batch = torch.from_numpy(np.stack(batch_windows, axis=0)).to(device)
        logits = model(x_batch)
        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()

        for p, (s, e) in zip(probs, batch_ranges):
            prob_sum[s:e] += p
            prob_count[s:e] += 1.0

    probs_full = prob_sum / np.maximum(prob_count, 1e-6)
    return probs_full


# ============================================================
# Postprocessing
# ============================================================

def probs_to_bouts(
    probs: np.ndarray,
    threshold: float = 0.5,
    min_duration_frames: int = 300,
    min_gap_frames: int = 120,
) -> List[Tuple[int, int]]:
    """
    Convert probability trace into [(start, end), ...] intervals.
    end is exclusive.
    """
    if probs.ndim != 1:
        raise ValueError("probs must be a 1D array.")

    mask = probs >= threshold
    n = len(mask)

    segments: List[Tuple[int, int]] = []
    in_seg = False
    start = 0

    for i in range(n):
        if mask[i] and not in_seg:
            in_seg = True
            start = i
        elif not mask[i] and in_seg:
            in_seg = False
            segments.append((start, i))

    if in_seg:
        segments.append((start, n))

    segments = [(s, e) for s, e in segments if (e - s) >= min_duration_frames]

    if not segments:
        return []

    merged = [segments[0]]
    for s, e in segments[1:]:
        prev_s, prev_e = merged[-1]
        if s - prev_e < min_gap_frames:
            merged[-1] = (prev_s, e)
        else:
            merged.append((s, e))

    return merged


# ============================================================
# Helpers for single-recording block split
# ============================================================

def estimate_pos_weight(labels: Sequence[np.ndarray]) -> float:
    """
    Estimate positive class weight for BCEWithLogitsLoss:
        pos_weight = #negative / #positive
    """
    total_pos = 0.0
    total_count = 0.0
    for y in labels:
        y = y.astype(np.float32)
        total_pos += y.sum()
        total_count += y.size
    total_neg = total_count - total_pos
    pos_weight = total_neg / max(total_pos, 1.0)
    return float(pos_weight)


def build_label_mask_from_intervals(
    n_frames: int,
    start_frame: np.ndarray,
    end_frame: np.ndarray,
) -> np.ndarray:
    """Build a framewise binary mask from labeled intervals."""
    y = np.zeros(n_frames, dtype=np.float32)

    for s, e in zip(start_frame, end_frame):
        s = int(max(0, s))
        e = int(min(n_frames, e))
        if e > s:
            y[s:e] = 1.0

    return y


def make_block_split_indices(
    dataset: SwimBoutDataset,
    n_blocks: int = 10,
    val_blocks: Optional[Sequence[int]] = None,
) -> tuple[list[int], list[int]]:
    """
    Split a single recording into contiguous time blocks and assign windows
    to train/val based on full containment.

    Parameters
    ----------
    dataset :
        SwimBoutDataset built from a single recording.
    n_blocks :
        Number of contiguous blocks across the recording.
    val_blocks :
        Which block ids to use for validation.
        Default: last 20% of blocks.
    """
    if len(dataset.signals) != 1:
        raise ValueError("make_block_split_indices expects exactly one recording in the dataset.")

    n_frames = dataset.signals[0].shape[1]
    block_edges = np.linspace(0, n_frames, n_blocks + 1, dtype=int)

    if val_blocks is None:
        n_val_blocks = max(1, int(round(0.2 * n_blocks)))
        val_blocks = list(range(n_blocks - n_val_blocks, n_blocks))

    val_blocks = set(int(b) for b in val_blocks)

    train_idx: list[int] = []
    val_idx: list[int] = []

    for i, win in enumerate(dataset.indices):
        assigned = False
        for b in range(n_blocks):
            block_start = block_edges[b]
            block_end = block_edges[b + 1]

            # Window must lie fully within a block
            if win.start >= block_start and win.end <= block_end:
                if b in val_blocks:
                    val_idx.append(i)
                else:
                    train_idx.append(i)
                assigned = True
                break

        # Windows crossing block boundaries are discarded

    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError(
            "Block split produced an empty train or validation set. "
            "Try reducing n_blocks, reducing window_size, or using a smaller stride."
        )

    return train_idx, val_idx


def make_single_recording_block_dataloaders(
    x: np.ndarray,
    y: np.ndarray,
    window_size: int = 6000,
    stride_train: int = 1000,
    stride_val: int = 1000,
    batch_size: int = 16,
    n_blocks: int = 10,
    val_blocks: Optional[Sequence[int]] = None,
    device_type: str = "cpu",
) -> tuple[DataLoader, DataLoader, SwimBoutDataset, SwimBoutDataset]:
    """
    Create train/val dataloaders for a single recording using contiguous block split.
    """
    if stride_train != stride_val:
        raise ValueError(
            "For block-based splitting, it is best to use the same stride for train and val "
            "so the split is clean and directly comparable."
        )

    base_train_ds = SwimBoutDataset(
        [x],
        [y],
        window_size=window_size,
        stride=stride_train,
        normalize=True,
        training=True,
    )

    base_val_ds = SwimBoutDataset(
        [x],
        [y],
        window_size=window_size,
        stride=stride_val,
        normalize=True,
        training=False,
    )

    if len(base_train_ds.indices) != len(base_val_ds.indices):
        raise RuntimeError("Train and val base datasets should have identical indices here.")

    train_idx, val_idx = make_block_split_indices(
        dataset=base_train_ds,
        n_blocks=n_blocks,
        val_blocks=val_blocks,
    )

    train_ds = Subset(base_train_ds, train_idx)
    val_ds = Subset(base_val_ds, val_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device_type == "cuda"),
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device_type == "cuda"),
        drop_last=False,
    )

    return train_loader, val_loader, base_train_ds, base_val_ds


# ============================================================
# Main training function
# ============================================================

def train_single_recording_block_split(
    data_path: str,
    label_path: str,
    window_size: int = 6000,
    stride: int = 1000,
    batch_size: int = 16,
    num_epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    n_blocks: int = 10,
    val_blocks: Optional[Sequence[int]] = None,
    save_path: str = "unet1d_swim_bout.pt",
) -> None:
    """
    Train on one recording using a contiguous block split.
    """
    set_seed(42)
    device = get_device()
    print("Using device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # ---------------- Load recording ----------------
    res = import16chFlt(data_path)
    x = np.vstack((res["fltCh0"], res["fltCh1"])).astype(np.float32)
    n_frames = x.shape[1]

    # ---------------- Load labels ----------------
    try:
        label_data = pd.read_excel(label_path, sheet_name="swim_bouts")
    except Exception:
        label_data = pd.read_excel(label_path)

    required_cols = {"start_frame", "end_frame"}
    if not required_cols.issubset(label_data.columns):
        raise ValueError(f"Label file must contain columns: {sorted(required_cols)}")

    start_frame = np.asarray(label_data["start_frame"], dtype=np.int64)
    end_frame = np.asarray(label_data["end_frame"], dtype=np.int64)

    if len(start_frame) == 0:
        raise ValueError("No labeled swim bouts were found in the label file.")

    if end_frame.max() > n_frames:
        raise ValueError(
            f"Label end_frame exceeds recording length: max end_frame={end_frame.max()}, n_frames={n_frames}"
        )

    y = build_label_mask_from_intervals(n_frames, start_frame, end_frame)

    print(f"Recording length: {n_frames} frames")
    print(f"Number of labeled bouts: {len(start_frame)}")
    print(f"Positive fraction: {y.mean():.6f}")

    # ---------------- Create loaders ----------------
    train_loader, val_loader, _, _ = make_single_recording_block_dataloaders(
        x=x,
        y=y,
        window_size=window_size,
        stride_train=stride,
        stride_val=stride,
        batch_size=batch_size,
        n_blocks=n_blocks,
        val_blocks=val_blocks,
        device_type=device.type,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # ---------------- Model ----------------
    model = UNet1D(
        in_channels=2,
        out_channels=1,
        base_channels=32,
        kernel_size=7,
    ).to(device)

    pos_weight = estimate_pos_weight([y])
    print("Estimated pos_weight:", pos_weight)

    criterion = BCEDiceLoss(pos_weight=pos_weight, dice_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # ---------------- Train ----------------
    best_val_f1 = -math.inf
    best_state = None

    for epoch in range(1, num_epochs + 1):
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        val_loss, val_metrics = evaluate(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"train_f1={train_metrics['f1']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Loaded best model with val_f1={best_val_f1:.4f}")

    # ---------------- Full-recording inference ----------------
    probs_full = predict_full_recording(
        model,
        x,
        device=device,
        window_size=window_size,
        stride=stride,
        normalize=True,
        batch_size=batch_size,
    )

    bouts = probs_to_bouts(
        probs_full,
        threshold=0.5,
        min_duration_frames=int(0.03 * 6000),
        min_gap_frames=int(0.02 * 6000),
    )

    print("Predicted bouts:", bouts[:20], "..." if len(bouts) > 20 else "")
    print(f"Total predicted bouts: {len(bouts)}")

    # ---------------- Save model ----------------
    torch.save(model.state_dict(), save_path)
    print(f"Saved model to {save_path}")


import numpy as np
import torch
import pandas as pd
from zfish._io import import16chFlt

# Reuse your UNet1D, predict_full_recording, probs_to_bouts, get_device
# definitions from training code.

def load_trained_model(
    checkpoint_path: str,
    device: torch.device,
    base_channels: int = 32,
    kernel_size: int = 7,
) -> torch.nn.Module:
    model = UNet1D(
        in_channels=2,
        out_channels=1,
        base_channels=base_channels,
        kernel_size=kernel_size,
    ).to(device)

    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

if __name__ == "__main__":
    train_single_recording_block_split(
        data_path=r"D:\EnData\10161\S1\res.16chFlt",
        label_path=r"D:\EnData\10161\S1\labeled_bouts.xlsx",
        batch_size=16,
        num_epochs=13
    )