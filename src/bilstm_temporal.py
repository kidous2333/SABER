"""
bilstm_temporal.py
BiLSTM temporal model, intended to replace the current sliding-window statistical features + LGBM approach.

Input: per-frame probability distributions [T, n_classes] from the first-layer model
Output: refined per-frame probability distributions [T, n_classes]

Advantages:
  1. Learns long-term dependencies (LSTM memory cells)
  2. Bidirectional modeling (BiLSTM leverages both past and future information)
  3. End-to-end training (no need for hand-crafted sliding window features)
"""

import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Tuple

logger = logging.getLogger(__name__)


class BiLSTMTemporalModel(nn.Module):
    """
    BiLSTM temporal refinement model.

    Base architecture:
      Input [B, L, D] -> BiLSTM -> LayerNorm -> Dropout -> Linear -> logits

    Optional enhancements (controlled via cfg toggles):
      - use_layer_norm:        Add LayerNorm after LSTM output (Pre-LN style)
      - use_deep_projection:   2-layer MLP + GELU instead of single Linear
      - use_attention_pooling: Multi-head self-attention after BiLSTM (hybrid architecture)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        output_dim: int = None,
        use_layer_norm: bool = True,
        use_deep_projection: bool = False,
        use_attention_pooling: bool = False,
        nhead: int = 4,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim or input_dim
        self.use_layer_norm = use_layer_norm
        self.use_deep_projection = use_deep_projection
        self.use_attention_pooling = use_attention_pooling

        lstm_hidden = hidden_dim  # hidden size per layer in BiLSTM
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        bilstm_out_dim = hidden_dim * 2  # bidirectional concat

        # ── LayerNorm (Pre-LN style: after transform, before activation) ──
        self.ln = nn.LayerNorm(bilstm_out_dim) if use_layer_norm else nn.Identity()

        # ── Attention Pooling (hybrid architecture: BiLSTM + Self-Attention) ──
        if use_attention_pooling:
            while bilstm_out_dim % nhead != 0 and nhead > 1:
                nhead -= 1
            self.attn = nn.MultiheadAttention(
                embed_dim=bilstm_out_dim,
                num_heads=nhead,
                dropout=dropout,
                batch_first=True,
            )
            self.attn_ln = nn.LayerNorm(bilstm_out_dim)
            self.attn_nhead = nhead
        else:
            self.attn = None

        # ── Projection head ──
        self.dropout = nn.Dropout(dropout)
        if use_deep_projection:
            # 2-layer MLP + GELU
            proj_hidden = bilstm_out_dim
            self.proj = nn.Sequential(
                nn.Linear(bilstm_out_dim, proj_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(proj_hidden, self.output_dim),
            )
        else:
            self.proj = nn.Linear(bilstm_out_dim, self.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq_len, input_dim]
        Returns: [batch, seq_len, output_dim]
        """
        # BiLSTM
        lstm_out, _ = self.bilstm(x)  # [B, L, hidden_dim*2]

        # LayerNorm
        out = self.ln(lstm_out)

        # Attention pooling (residual connection)
        if self.attn is not None:
            attn_out, _ = self.attn(out, out, out)
            out = self.attn_ln(out + attn_out)

        # Dropout + projection head
        out = self.dropout(out)
        logits = self.proj(out)  # [B, L, output_dim]

        return logits


class ProbabilityDataset(Dataset):
    """
    Temporal probability dataset.

    Splits a long sequence into fixed-length segments (chunks), supporting sliding window sampling.
    """

    def __init__(
        self,
        proba: np.ndarray,
        labels: np.ndarray,
        chunk_size: int = 512,
        stride: int = 256,
    ):
        """
        proba: [T, n_classes] probabilities from the first-layer model
        labels: [T] ground truth labels
        chunk_size: sequence length per sample
        stride: sliding window step (overlap when stride < chunk_size)
        """
        self.proba = torch.from_numpy(proba).float()
        self.labels = torch.from_numpy(labels).long()
        self.chunk_size = chunk_size
        self.stride = stride

        # Compute chunk indices
        T = len(proba)
        self.indices = []
        for start in range(0, T - chunk_size + 1, stride):
            self.indices.append((start, start + chunk_size))

        # Pad the last segment if shorter than chunk_size
        if len(self.indices) == 0 or self.indices[-1][1] < T:
            self.indices.append((max(0, T - chunk_size), T))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start, end = self.indices[idx]
        return (
            self.proba[start:end],    # [chunk_size, n_classes]
            self.labels[start:end],   # [chunk_size]
        )


def train_bilstm_temporal(
    proba_train: np.ndarray,
    y_train: np.ndarray,
    proba_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
    hidden_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.3,
    chunk_size: int = 512,
    stride_train: int = 256,
    stride_val: int = 512,
    batch_size: int = 32,
    epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cuda",
    early_stopping_patience: int = 5,
) -> Tuple[nn.Module, dict]:
    """
    Train a BiLSTM temporal model.

    Returns:
        model: Trained model
        history: Training history {"train_loss": [...], "val_loss": [...], "val_acc": [...]}
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logger.info(f"BiLSTM training device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        logger.info(f"BiLSTM training device: {device} (GPU not available, using CPU)")
        if device == "cuda":
            logger.warning("Config requires GPU, but CUDA is not available; falling back to CPU")

    # Build dataset
    train_dataset = ProbabilityDataset(proba_train, y_train, chunk_size, stride_train)
    val_dataset = ProbabilityDataset(proba_val, y_val, chunk_size, stride_val)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
    )

    logger.info(
        f"Training set: {len(train_dataset)} chunks, validation set: {len(val_dataset)} chunks"
    )

    # Build model
    model = BiLSTMTemporalModel(
        input_dim=n_classes,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        output_dim=n_classes,
    ).to(device)

    # Loss function: cross-entropy (handles softmax internally)
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # Training loop
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        # ---- Training phase ----
        model.train()
        train_loss = 0.0
        for proba_chunk, label_chunk in train_loader:
            proba_chunk = proba_chunk.to(device)  # [B, chunk_size, n_classes]
            label_chunk = label_chunk.to(device)  # [B, chunk_size]

            optimizer.zero_grad()
            logits = model(proba_chunk)  # [B, chunk_size, n_classes]

            # reshape for CrossEntropyLoss: [B*chunk_size, n_classes] vs [B*chunk_size]
            loss = criterion(
                logits.reshape(-1, n_classes),
                label_chunk.reshape(-1),
            )
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ---- Validation phase ----
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for proba_chunk, label_chunk in val_loader:
                proba_chunk = proba_chunk.to(device)
                label_chunk = label_chunk.to(device)

                logits = model(proba_chunk)
                loss = criterion(
                    logits.reshape(-1, n_classes),
                    label_chunk.reshape(-1),
                )
                val_loss += loss.item()

                # Compute accuracy
                pred = logits.argmax(dim=-1)  # [B, chunk_size]
                correct += (pred == label_chunk).sum().item()
                total += label_chunk.numel()

        val_loss /= len(val_loader)
        val_acc = correct / total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        logger.info(
            f"Epoch {epoch+1}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.6f}"
        )

        # Learning rate scheduling
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < old_lr:
            logger.info(f"Learning rate decreased: {old_lr:.6f} -> {new_lr:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    return model, history


def predict_bilstm_temporal(
    model: nn.Module,
    proba: np.ndarray,
    chunk_size: int = 512,
    stride: int = 256,
    batch_size: int = 32,
    device: str = "cuda",
) -> np.ndarray:
    """
    Predict the entire sequence using a trained BiLSTM model.

    For overlapping regions, the predictions are averaged.

    Returns: [T, n_classes] refined probability distribution
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    T, n_classes = proba.shape
    proba_tensor = torch.from_numpy(proba).float()

    # Accumulate predictions (for averaging overlapping regions)
    proba_sum = np.zeros((T, n_classes), dtype=np.float32)
    count = np.zeros(T, dtype=np.int32)

    # Sliding window prediction
    with torch.no_grad():
        for start in range(0, T, stride):
            end = min(start + chunk_size, T)
            actual_size = end - start

            # Pad if shorter than chunk_size
            if actual_size < chunk_size:
                chunk = torch.zeros(chunk_size, n_classes)
                chunk[:actual_size] = proba_tensor[start:end]
            else:
                chunk = proba_tensor[start:end]

            chunk = chunk.unsqueeze(0).to(device)  # [1, chunk_size, n_classes]
            logits = model(chunk)  # [1, chunk_size, n_classes]
            proba_chunk = torch.softmax(logits, dim=-1).cpu().numpy()[0]  # [chunk_size, n_classes]

            # Accumulate into result
            proba_sum[start:end] += proba_chunk[:actual_size]
            count[start:end] += 1

    # Average overlapping regions
    proba_refined = proba_sum / count[:, np.newaxis]

    return proba_refined
