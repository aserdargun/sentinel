"""Temporal Convolutional Network (TCN) anomaly detector.

Dilated causal convolutions with residual blocks reconstruct input
sequences.  Anomaly scores are per-window mean squared reconstruction
error, padded to match the original input length.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import structlog

from sentinel.core.base_model import BaseAnomalyDetector
from sentinel.core.device import resolve_device
from sentinel.core.exceptions import SentinelError
from sentinel.core.registry import register_model
from sentinel.data.preprocessors import create_windows

logger = structlog.get_logger(__name__)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ---------------------------------------------------------------------------
# PyTorch modules (only defined when torch is available)
# ---------------------------------------------------------------------------
if HAS_TORCH:

    class _CausalConv1d(nn.Module):
        """1-D causal convolution with left-padding.

        Ensures that the output at time *t* depends only on inputs at
        times <= *t*.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of the convolving kernel.
            dilation: Spacing between kernel elements.
        """

        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            dilation: int,
        ) -> None:
            super().__init__()
            self._padding = (kernel_size - 1) * dilation
            self.conv = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=0,
                dilation=dilation,
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Apply causal convolution.

            Args:
                x: Tensor of shape ``(batch, channels, seq_len)``.

            Returns:
                Tensor of shape ``(batch, out_channels, seq_len)``.
            """
            # Left-pad so output length == input length.
            x = nn.functional.pad(x, (self._padding, 0))
            return self.conv(x)

    class _TemporalBlock(nn.Module):
        """Single residual block of the TCN.

        Two causal convolutions with ReLU and dropout, plus a skip
        connection.  A 1x1 convolution is used when the number of
        channels changes between input and output.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Kernel size for both convolutions.
            dilation: Dilation factor (typically ``2**i``).
            dropout: Dropout probability.
        """

        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            dilation: int,
            dropout: float,
        ) -> None:
            super().__init__()
            self.conv1 = _CausalConv1d(in_channels, out_channels, kernel_size, dilation)
            self.relu1 = nn.ReLU()
            self.dropout1 = nn.Dropout(dropout)

            self.conv2 = _CausalConv1d(
                out_channels, out_channels, kernel_size, dilation
            )
            self.relu2 = nn.ReLU()
            self.dropout2 = nn.Dropout(dropout)

            # 1x1 conv for residual alignment when channel dims differ.
            self.downsample: nn.Module | None = None
            if in_channels != out_channels:
                self.downsample = nn.Conv1d(in_channels, out_channels, 1)

            self.relu_out = nn.ReLU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass through the temporal block.

            Args:
                x: Tensor of shape ``(batch, in_channels, seq_len)``.

            Returns:
                Tensor of shape ``(batch, out_channels, seq_len)``.
            """
            out = self.dropout1(self.relu1(self.conv1(x)))
            out = self.dropout2(self.relu2(self.conv2(out)))

            residual = x if self.downsample is None else self.downsample(x)
            return self.relu_out(out + residual)

    class _TCNEncoder(nn.Module):
        """Stack of temporal blocks forming the TCN encoder.

        Args:
            n_features: Number of input features (channels in first layer).
            num_channels: List of output channel sizes for each block.
            kernel_size: Kernel size shared across all blocks.
            dropout: Dropout probability.
        """

        def __init__(
            self,
            n_features: int,
            num_channels: list[int],
            kernel_size: int,
            dropout: float,
        ) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            in_ch = n_features
            for i, out_ch in enumerate(num_channels):
                dilation = 2**i
                layers.append(
                    _TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout)
                )
                in_ch = out_ch
            self.network = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Encode the input sequence.

            Args:
                x: Tensor of shape ``(batch, n_features, seq_len)``.

            Returns:
                Tensor of shape ``(batch, num_channels[-1], seq_len)``.
            """
            return self.network(x)

    class _TCNModel(nn.Module):
        """Full TCN reconstruction model: encoder + linear decoder.

        Args:
            n_features: Number of input features.
            num_channels: Channel sizes per temporal block.
            kernel_size: Kernel size.
            dropout: Dropout probability.
        """

        def __init__(
            self,
            n_features: int,
            num_channels: list[int],
            kernel_size: int,
            dropout: float,
        ) -> None:
            super().__init__()
            self.encoder = _TCNEncoder(n_features, num_channels, kernel_size, dropout)
            # Map final encoder channels back to n_features for reconstruction.
            self.decoder = nn.Conv1d(num_channels[-1], n_features, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Reconstruct the input.

            Args:
                x: Tensor of shape ``(batch, n_features, seq_len)``.

            Returns:
                Reconstructed tensor with the same shape as *x*.
            """
            encoded = self.encoder(x)
            return self.decoder(encoded)

    # -----------------------------------------------------------------------
    # Detector class (registered only when torch is importable)
    # -----------------------------------------------------------------------

    @register_model("tcn")
    class TCNDetector(BaseAnomalyDetector):
        """TCN-based anomaly detector using dilated causal convolutions.

        Input sequences are reconstructed through a stack of temporal
        blocks with exponentially increasing dilation.  Each block
        contains two causal convolutions with ReLU activation, dropout,
        and a residual shortcut.  A final 1x1 convolution maps the
        encoder output back to the original feature dimension.

        Anomaly scores equal the per-window mean squared reconstruction
        error.

        Args:
            num_channels: List of channel sizes per temporal block.
            kernel_size: Kernel size for causal convolutions.
            dropout: Dropout probability.
            seq_len: Sliding window length.
            learning_rate: Optimiser learning rate.
            epochs: Number of training epochs.
            batch_size: Mini-batch size.
            device: Device string (``"auto"``, ``"cpu"``, ``"cuda"``,
                ``"mps"``).

        Example::

            detector = TCNDetector(num_channels=[32, 32], seq_len=30)
            detector.fit(X_train)
            scores = detector.score(X_test)
        """

        def __init__(
            self,
            num_channels: list[int] | None = None,
            kernel_size: int = 3,
            dropout: float = 0.1,
            seq_len: int = 50,
            learning_rate: float = 1e-3,
            epochs: int = 100,
            batch_size: int = 32,
            device: str = "auto",
        ) -> None:
            self.num_channels: list[int] = (
                num_channels if num_channels is not None else [32, 32, 32]
            )
            self.kernel_size = kernel_size
            self.dropout = dropout
            self.seq_len = seq_len
            self.learning_rate = learning_rate
            self.epochs = epochs
            self.batch_size = batch_size
            self.device_str = device

            self._device: torch.device | None = None
            self._model: _TCNModel | None = None
            self._n_features: int | None = None
            self._is_fitted: bool = False

        # ------------------------------------------------------------------ #
        # Public interface                                                     #
        # ------------------------------------------------------------------ #

        def fit(self, X: np.ndarray, **kwargs: Any) -> None:
            """Train the TCN on sliding windows extracted from *X*.

            Args:
                X: 2-D array of shape ``(n_samples, n_features)``.
                **kwargs: Ignored; present for interface compatibility.

            Raises:
                SentinelError: If *X* has fewer rows than ``seq_len``.
            """
            if X.ndim == 1:
                X = X.reshape(-1, 1)

            n_samples, n_features = X.shape
            if n_samples < self.seq_len:
                raise SentinelError(
                    f"TCN requires at least {self.seq_len} samples, got {n_samples}"
                )

            self._n_features = n_features
            self._device = torch.device(resolve_device(self.device_str))

            windows = create_windows(X, self.seq_len)
            # windows: (n_windows, seq_len, n_features)

            self._model = _TCNModel(
                n_features=n_features,
                num_channels=self.num_channels,
                kernel_size=self.kernel_size,
                dropout=self.dropout,
            ).to(self._device)

            self._train_loop(windows)
            self._is_fitted = True

        def score(self, X: np.ndarray) -> np.ndarray:
            """Compute per-sample anomaly scores via reconstruction error.

            Sliding windows are reconstructed, and the MSE for each
            window is assigned to its last time-step.  The first
            ``seq_len - 1`` samples receive the score of the earliest
            available window.

            Args:
                X: 2-D array of shape ``(n_samples, n_features)``.

            Returns:
                1-D array of length ``n_samples`` with anomaly scores.

            Raises:
                SentinelError: If the model has not been fitted or if
                    *X* has fewer rows than ``seq_len``.
            """
            self._check_fitted()

            if X.ndim == 1:
                X = X.reshape(-1, 1)

            n_samples = X.shape[0]
            if n_samples < self.seq_len:
                raise SentinelError(
                    f"TCN scoring requires at least {self.seq_len} samples, "
                    f"got {n_samples}"
                )

            windows = create_windows(X, self.seq_len)
            window_scores = self._score_windows(windows)

            # Map window scores back to individual time-steps.
            scores = np.empty(n_samples, dtype=np.float64)
            # Each window_scores[i] corresponds to the window ending at
            # position (i + seq_len - 1).  Assign to last position.
            for i, ws in enumerate(window_scores):
                scores[i + self.seq_len - 1] = ws
            # Pad the leading positions with the first window's score.
            scores[: self.seq_len - 1] = window_scores[0]

            return scores

        def save(self, path: str) -> None:
            """Save the model state_dict and config to *path*.

            Creates two files:

            * ``config.json`` -- Hyperparameters and metadata.
            * ``model.pt`` -- PyTorch ``state_dict``.

            All writes use atomic temp-file + rename.

            Args:
                path: Directory in which to store artifacts.

            Raises:
                SentinelError: If the model has not been fitted.
            """
            self._check_fitted()
            os.makedirs(path, exist_ok=True)

            # -- config.json ---------------------------------------------------
            config = {
                "model_name": "tcn",
                "num_channels": self.num_channels,
                "kernel_size": self.kernel_size,
                "dropout": self.dropout,
                "seq_len": self.seq_len,
                "learning_rate": self.learning_rate,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "device": self.device_str,
                "n_features": self._n_features,
            }
            config_path = os.path.join(path, "config.json")
            tmp_config = config_path + ".tmp"
            with open(tmp_config, "w") as f:
                json.dump(config, f, indent=2)
            os.rename(tmp_config, config_path)

            # -- model.pt (state_dict) -----------------------------------------
            assert self._model is not None
            model_path = os.path.join(path, "model.pt")
            tmp_model = model_path + ".tmp"
            torch.save(self._model.state_dict(), tmp_model)
            os.rename(tmp_model, model_path)

        def load(self, path: str) -> None:
            """Load a previously saved model from *path*.

            Args:
                path: Directory containing ``config.json`` and ``model.pt``.

            Raises:
                SentinelError: If required files are missing.
            """
            config_path = os.path.join(path, "config.json")
            model_path = os.path.join(path, "model.pt")

            if not os.path.isfile(config_path):
                raise SentinelError(f"Config not found: {config_path}")
            if not os.path.isfile(model_path):
                raise SentinelError(f"Model file not found: {model_path}")

            with open(config_path) as f:
                config = json.load(f)

            self.num_channels = list(config["num_channels"])
            self.kernel_size = int(config["kernel_size"])
            self.dropout = float(config["dropout"])
            self.seq_len = int(config["seq_len"])
            self.learning_rate = float(config["learning_rate"])
            self.epochs = int(config["epochs"])
            self.batch_size = int(config["batch_size"])
            self.device_str = config.get("device", "auto")
            self._n_features = int(config["n_features"])

            self._device = torch.device(resolve_device(self.device_str))
            self._model = _TCNModel(
                n_features=self._n_features,
                num_channels=self.num_channels,
                kernel_size=self.kernel_size,
                dropout=self.dropout,
            ).to(self._device)

            state_dict = torch.load(
                model_path, map_location=self._device, weights_only=True
            )
            self._model.load_state_dict(state_dict)
            self._is_fitted = True

        def get_params(self) -> dict[str, Any]:
            """Return model hyperparameters and metadata.

            Returns:
                Dict with all constructor arguments plus ``n_features``
                (``None`` if the model has not been fitted).
            """
            return {
                "model_name": "tcn",
                "num_channels": self.num_channels,
                "kernel_size": self.kernel_size,
                "dropout": self.dropout,
                "seq_len": self.seq_len,
                "learning_rate": self.learning_rate,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "device": self.device_str,
                "n_features": self._n_features,
            }

        # ------------------------------------------------------------------ #
        # Private helpers                                                      #
        # ------------------------------------------------------------------ #

        def _check_fitted(self) -> None:
            """Raise if the model has not been fitted."""
            if not self._is_fitted:
                raise SentinelError(
                    "TCNDetector has not been fitted. Call fit() first."
                )

        def _train_loop(self, windows: np.ndarray) -> None:
            """Run the training loop over windowed data.

            Args:
                windows: 3-D array ``(n_windows, seq_len, n_features)``.
            """
            assert self._model is not None

            # Conv1d expects (batch, channels, seq_len), so transpose the
            # last two dimensions of each window.
            tensor = torch.tensor(windows, dtype=torch.float32, device=self._device)
            tensor = tensor.permute(0, 2, 1)  # -> (n, n_features, seq_len)

            dataset = TensorDataset(tensor)
            loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=0,
            )

            optimizer = torch.optim.Adam(
                self._model.parameters(), lr=self.learning_rate
            )
            criterion = nn.MSELoss()

            self._model.train()
            for epoch in range(self.epochs):
                epoch_loss = 0.0
                n_batches = 0
                for (batch,) in loader:
                    optimizer.zero_grad()
                    reconstructed = self._model(batch)
                    loss = criterion(reconstructed, batch)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    n_batches += 1

                avg_loss = epoch_loss / max(n_batches, 1)
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    logger.info(
                        "TCN epoch %d/%d  loss=%.6f", epoch + 1, self.epochs, avg_loss
                    )

        def _score_windows(self, windows: np.ndarray) -> np.ndarray:
            """Compute reconstruction MSE per window.

            Args:
                windows: 3-D array ``(n_windows, seq_len, n_features)``.

            Returns:
                1-D array of per-window MSE scores.
            """
            assert self._model is not None

            self._model.eval()
            tensor = torch.tensor(windows, dtype=torch.float32, device=self._device)
            tensor = tensor.permute(0, 2, 1)  # -> (n, n_features, seq_len)

            all_scores: list[np.ndarray] = []
            with torch.no_grad():
                for start in range(0, tensor.shape[0], self.batch_size):
                    batch = tensor[start : start + self.batch_size]
                    reconstructed = self._model(batch)
                    # MSE per window: mean over (channels, seq_len).
                    mse = (batch - reconstructed).pow(2).mean(dim=(1, 2)).cpu().numpy()
                    all_scores.append(mse)

            return np.concatenate(all_scores).astype(np.float64)
