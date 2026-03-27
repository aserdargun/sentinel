"""Elman RNN sequence-to-sequence reconstruction anomaly detector.

Trains an Elman RNN to reconstruct its input window. A sliding window of
``seq_len`` timesteps is fed through the RNN and a linear output layer
reconstructs the same window. The anomaly score for each window is the MSE
between the input and the reconstruction. Higher reconstruction error
indicates a more anomalous observation.
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import json
import os
from typing import Any

import numpy as np

from sentinel.core.base_model import BaseAnomalyDetector
from sentinel.core.device import resolve_device
from sentinel.core.exceptions import SentinelError

if HAS_TORCH:
    from sentinel.core.registry import register_model
    from sentinel.data.preprocessors import create_windows

    class _RNNNetwork(nn.Module):
        """Internal Elman RNN network for sequence reconstruction.

        Args:
            n_features: Number of input features.
            hidden_dim: RNN hidden state dimension.
            num_layers: Number of stacked RNN layers.
            dropout: Dropout probability between RNN layers.
        """

        def __init__(
            self,
            n_features: int,
            hidden_dim: int,
            num_layers: int,
            dropout: float,
        ) -> None:
            super().__init__()
            self.rnn = nn.RNN(
                input_size=n_features,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                nonlinearity="tanh",
            )
            self.fc = nn.Linear(hidden_dim, n_features)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass: reconstruct the input sequence.

            Args:
                x: Input tensor of shape ``(batch, seq_len, n_features)``.

            Returns:
                Reconstructed sequence of shape ``(batch, seq_len, n_features)``.
            """
            # rnn_out: (batch, seq_len, hidden_dim)
            rnn_out, _ = self.rnn(x)
            # Apply linear layer at every timestep to reconstruct all features.
            return self.fc(rnn_out)

    @register_model("rnn")
    class RNNDetector(BaseAnomalyDetector):
        """Elman RNN sequence-to-sequence reconstruction anomaly detector.

        Trains an Elman RNN to reconstruct sliding windows of ``seq_len``
        timesteps. At scoring time, the anomaly score for each window is
        the mean squared error between the input and reconstructed sequence.
        Points at the start of the series that lack a full window are
        assigned the mean score.

        Args:
            hidden_dim: RNN hidden state dimension.
            num_layers: Number of stacked RNN layers.
            seq_len: Input window length for reconstruction.
            learning_rate: Optimizer learning rate.
            epochs: Number of training epochs.
            batch_size: Mini-batch size for training.
            dropout: Dropout probability between RNN layers.
            device: Device string (``"auto"``, ``"cpu"``, ``"cuda"``, ``"mps"``).

        Example::

            detector = RNNDetector(hidden_dim=64, seq_len=50)
            detector.fit(X_train)
            scores = detector.score(X_test)
        """

        def __init__(
            self,
            hidden_dim: int = 64,
            num_layers: int = 2,
            seq_len: int = 50,
            learning_rate: float = 0.001,
            epochs: int = 100,
            batch_size: int = 32,
            dropout: float = 0.1,
            device: str = "auto",
        ) -> None:
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.seq_len = seq_len
            self.learning_rate = learning_rate
            self.epochs = epochs
            self.batch_size = batch_size
            self.dropout = dropout
            self.device = resolve_device(device)

            self._model: _RNNNetwork | None = None
            self._n_features: int | None = None
            self._is_fitted: bool = False

        # ------------------------------------------------------------------
        # Public interface
        # ------------------------------------------------------------------

        def fit(self, X: np.ndarray, **kwargs: Any) -> None:
            """Train the RNN on sliding windows of the input data.

            Creates sliding windows of length ``seq_len``. Each window serves
            as both input and reconstruction target. The model is trained to
            minimise MSE between the input window and its reconstruction.

            Args:
                X: Training data of shape ``(n_samples, n_features)``.
                **kwargs: Ignored; present for interface compatibility.

            Raises:
                SentinelError: If ``X`` has fewer rows than ``seq_len``.
            """
            if X.ndim == 1:
                X = X.reshape(-1, 1)

            n_samples, n_features = X.shape
            if n_samples < self.seq_len:
                raise SentinelError(
                    f"RNN requires at least {self.seq_len} samples "
                    f"(seq_len), got {n_samples}"
                )

            self._n_features = n_features

            # Build sliding windows: shape (n_windows, seq_len, n_features).
            windows = create_windows(X, seq_len=self.seq_len, stride=1)

            windows_t = torch.tensor(windows, dtype=torch.float32)

            dataset = TensorDataset(windows_t)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            # Build the network and move to device.
            self._model = _RNNNetwork(
                n_features=n_features,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                dropout=self.dropout,
            )
            dev = torch.device(self.device)
            self._model.to(dev)

            optimizer = torch.optim.Adam(
                self._model.parameters(), lr=self.learning_rate
            )
            criterion = nn.MSELoss()

            self._model.train()
            for _epoch in range(self.epochs):
                for (batch_x,) in loader:
                    batch_x = batch_x.to(dev)

                    reconstructed = self._model(batch_x)
                    loss = criterion(reconstructed, batch_x)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            self._is_fitted = True

        def score(self, X: np.ndarray) -> np.ndarray:
            """Compute anomaly scores via reconstruction error.

            For each sliding window the model reconstructs the input and the
            score is the per-window MSE between the input and reconstruction.
            The first ``seq_len - 1`` samples that lack a complete window are
            assigned the mean score of the remaining windows.

            Args:
                X: Data of shape ``(n_samples, n_features)``.

            Returns:
                1-D array of length ``n_samples`` with anomaly scores
                (higher is more anomalous).

            Raises:
                SentinelError: If the model has not been fitted or if
                    ``X`` has fewer rows than ``seq_len``.
            """
            self._check_fitted()

            if X.ndim == 1:
                X = X.reshape(-1, 1)

            n_samples = X.shape[0]
            if n_samples < self.seq_len:
                raise SentinelError(
                    f"RNN scoring requires at least {self.seq_len} samples "
                    f"(seq_len), got {n_samples}"
                )

            # Build windows of length seq_len.
            windows = create_windows(X, seq_len=self.seq_len, stride=1)

            windows_t = torch.tensor(windows, dtype=torch.float32)
            dev = torch.device(self.device)
            windows_t = windows_t.to(dev)

            assert self._model is not None
            self._model.eval()
            with torch.no_grad():
                reconstructed = self._model(windows_t).cpu().numpy()

            # Per-window MSE across timesteps and features.
            window_scores = np.mean((windows - reconstructed) ** 2, axis=(1, 2))

            # Pad: each window's score corresponds to the *last* sample in
            # that window, i.e. index (seq_len - 1) onwards.
            # The first (seq_len - 1) samples get the mean score.
            mean_score = float(np.mean(window_scores))
            scores = np.full(n_samples, mean_score, dtype=np.float64)
            scores[self.seq_len - 1 :] = window_scores

            return scores

        def save(self, path: str) -> None:
            """Save model state_dict and config to disk.

            Writes ``config.json`` (hyperparameters) and ``model.pt``
            (PyTorch state_dict). All writes are atomic (write to temp
            file, then rename).

            Args:
                path: Directory in which to store artifacts.

            Raises:
                SentinelError: If the model has not been fitted.
            """
            self._check_fitted()
            os.makedirs(path, exist_ok=True)

            # -- config.json ---------------------------------------------------
            config = {
                "model_name": "rnn",
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "seq_len": self.seq_len,
                "learning_rate": self.learning_rate,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "dropout": self.dropout,
                "device": self.device,
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
            """Load a previously saved RNN model from disk.

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

            self.hidden_dim = int(config["hidden_dim"])
            self.num_layers = int(config["num_layers"])
            self.seq_len = int(config["seq_len"])
            self.learning_rate = float(config["learning_rate"])
            self.epochs = int(config["epochs"])
            self.batch_size = int(config["batch_size"])
            self.dropout = float(config["dropout"])
            self.device = str(config["device"])
            self._n_features = int(config["n_features"])

            # Rebuild the network and load weights.
            self._model = _RNNNetwork(
                n_features=self._n_features,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                dropout=self.dropout,
            )
            dev = torch.device(self.device)
            self._model.load_state_dict(
                torch.load(model_path, map_location=dev, weights_only=True)
            )
            self._model.to(dev)
            self._model.eval()
            self._is_fitted = True

        def get_params(self) -> dict[str, Any]:
            """Return model parameters.

            Returns:
                Dict containing all hyperparameters and ``n_features``
                (``None`` if not yet fitted).
            """
            return {
                "model_name": "rnn",
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "seq_len": self.seq_len,
                "learning_rate": self.learning_rate,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "dropout": self.dropout,
                "device": self.device,
                "n_features": self._n_features,
            }

        # ------------------------------------------------------------------
        # Private helpers
        # ------------------------------------------------------------------

        def _check_fitted(self) -> None:
            """Raise if the model has not been fitted."""
            if not self._is_fitted:
                raise SentinelError(
                    "RNNDetector has not been fitted. Call fit() first."
                )
