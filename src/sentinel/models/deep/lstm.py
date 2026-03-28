"""LSTM predictor anomaly detector.

Trains an LSTM network to predict the next timestep from a sliding window
of ``seq_len`` steps. The anomaly score for each window is the MSE between
the predicted and actual next timestep. Higher reconstruction error indicates
a more anomalous observation.
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

    class _LSTMNetwork(nn.Module):
        """Internal LSTM network for next-step prediction.

        Args:
            n_features: Number of input features.
            hidden_dim: LSTM hidden state dimension.
            num_layers: Number of stacked LSTM layers.
            dropout: Dropout probability between LSTM layers.
        """

        def __init__(
            self,
            n_features: int,
            hidden_dim: int,
            num_layers: int,
            dropout: float,
        ) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=n_features,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.fc = nn.Linear(hidden_dim, n_features)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass: predict the next timestep.

            Args:
                x: Input tensor of shape ``(batch, seq_len, n_features)``.

            Returns:
                Predicted next step of shape ``(batch, n_features)``.
            """
            # lstm_out: (batch, seq_len, hidden_dim)
            lstm_out, _ = self.lstm(x)
            # Take the output from the last timestep.
            last_hidden = lstm_out[:, -1, :]
            return self.fc(last_hidden)

    @register_model("lstm")
    class LSTMDetector(BaseAnomalyDetector):
        """LSTM next-step predictor for anomaly detection.

        Trains an LSTM to forecast the next timestep given a window of
        ``seq_len`` preceding steps. At scoring time, the anomaly score
        for each window is the mean squared error between the predicted
        and actual next timestep. Points at the start of the series that
        lack a full window are assigned the mean score.

        Args:
            hidden_dim: LSTM hidden state dimension.
            num_layers: Number of stacked LSTM layers.
            seq_len: Input window length for prediction.
            learning_rate: Optimizer learning rate.
            epochs: Number of training epochs.
            batch_size: Mini-batch size for training.
            dropout: Dropout probability between LSTM layers.
            device: Device string (``"auto"``, ``"cpu"``, ``"cuda"``, ``"mps"``).

        Example::

            detector = LSTMDetector(hidden_dim=64, seq_len=50)
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
            self.device_str = device

            # Resolved at fit-time so that resolve_device() sees current
            # hardware state.
            self._device: torch.device | None = None
            self._model: _LSTMNetwork | None = None
            self._n_features: int | None = None
            self._is_fitted: bool = False

        # ------------------------------------------------------------------
        # Public interface
        # ------------------------------------------------------------------

        def fit(self, X: np.ndarray, **kwargs: Any) -> None:
            """Train the LSTM predictor on data.

            Creates sliding windows of length ``seq_len + 1``. For each
            window the first ``seq_len`` steps are the input and the last
            step is the prediction target. The model is trained to minimise
            MSE between predicted and actual next step.

            Args:
                X: Training data of shape ``(n_samples, n_features)``.
                **kwargs: Ignored; present for interface compatibility.

            Raises:
                SentinelError: If ``X`` has fewer rows than ``seq_len + 1``.
            """
            if X.ndim == 1:
                X = X.reshape(-1, 1)

            n_samples, n_features = X.shape
            min_rows = self.seq_len + 1
            if n_samples < min_rows:
                raise SentinelError(
                    f"LSTM requires at least {min_rows} samples "
                    f"(seq_len + 1), got {n_samples}"
                )

            self._n_features = n_features
            self._device = torch.device(resolve_device(self.device_str))

            # Build sliding windows: shape (n_windows, seq_len+1, n_features).
            windows = create_windows(X, seq_len=self.seq_len + 1, stride=1)
            # Input: all but the last step; Target: the last step.
            inputs = windows[:, :-1, :]  # (n_windows, seq_len, n_features)
            targets = windows[:, -1, :]  # (n_windows, n_features)

            inputs_t = torch.tensor(inputs, dtype=torch.float32)
            targets_t = torch.tensor(targets, dtype=torch.float32)

            dataset = TensorDataset(inputs_t, targets_t)
            loader = DataLoader(
                dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
            )

            # Build the network and move to device.
            self._model = _LSTMNetwork(
                n_features=n_features,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                dropout=self.dropout,
            )
            self._model.to(self._device)

            optimizer = torch.optim.Adam(
                self._model.parameters(), lr=self.learning_rate
            )
            criterion = nn.MSELoss()

            self._model.train()
            for _epoch in range(self.epochs):
                for batch_x, batch_y in loader:
                    batch_x = batch_x.to(self._device)
                    batch_y = batch_y.to(self._device)

                    preds = self._model(batch_x)
                    loss = criterion(preds, batch_y)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            self._is_fitted = True

        def score(self, X: np.ndarray) -> np.ndarray:
            """Compute anomaly scores via next-step prediction error.

            For each sliding window the model predicts the next timestep
            and the score is the per-sample MSE between predicted and
            actual. The first ``seq_len`` samples that lack a complete
            window are assigned the mean score of the remaining windows.

            Args:
                X: Data of shape ``(n_samples, n_features)``.

            Returns:
                1-D array of length ``n_samples`` with anomaly scores
                (higher is more anomalous).

            Raises:
                SentinelError: If the model has not been fitted or if
                    ``X`` has fewer rows than ``seq_len + 1``.
            """
            self._check_fitted()

            if X.ndim == 1:
                X = X.reshape(-1, 1)

            n_samples = X.shape[0]
            min_rows = self.seq_len + 1
            if n_samples < min_rows:
                raise SentinelError(
                    f"LSTM scoring requires at least {min_rows} samples "
                    f"(seq_len + 1), got {n_samples}"
                )

            # Build windows of length seq_len+1.
            windows = create_windows(X, seq_len=self.seq_len + 1, stride=1)
            inputs = windows[:, :-1, :]  # (n_windows, seq_len, n_features)
            actuals = windows[:, -1, :]  # (n_windows, n_features)

            inputs_t = torch.tensor(inputs, dtype=torch.float32)
            inputs_t = inputs_t.to(self._device)

            assert self._model is not None
            self._model.eval()
            with torch.no_grad():
                preds = self._model(inputs_t).cpu().numpy()

            # Per-window MSE across features.
            window_scores = np.mean((preds - actuals) ** 2, axis=1)

            # Pad: each window's score corresponds to the *last* sample in
            # that (seq_len+1)-length window, i.e. index seq_len onwards.
            # The first seq_len samples get the mean score.
            mean_score = float(np.mean(window_scores))
            scores = np.full(n_samples, mean_score, dtype=np.float64)
            scores[self.seq_len :] = window_scores

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
                "model_name": "lstm",
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "seq_len": self.seq_len,
                "learning_rate": self.learning_rate,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "dropout": self.dropout,
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
            """Load a previously saved LSTM model from disk.

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
            self.device_str = str(config["device"])
            self._n_features = int(config["n_features"])

            self._device = torch.device(resolve_device(self.device_str))

            # Rebuild the network and load weights.
            self._model = _LSTMNetwork(
                n_features=self._n_features,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                dropout=self.dropout,
            )
            self._model.load_state_dict(
                torch.load(model_path, map_location=self._device, weights_only=True)
            )
            self._model.to(self._device)
            self._model.eval()
            self._is_fitted = True

        def get_params(self) -> dict[str, Any]:
            """Return model parameters.

            Returns:
                Dict containing all hyperparameters and ``n_features``
                (``None`` if not yet fitted).
            """
            return {
                "model_name": "lstm",
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "seq_len": self.seq_len,
                "learning_rate": self.learning_rate,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "dropout": self.dropout,
                "device": self.device_str,
                "n_features": self._n_features,
            }

        # ------------------------------------------------------------------
        # Private helpers
        # ------------------------------------------------------------------

        def _check_fitted(self) -> None:
            """Raise if the model has not been fitted."""
            if not self._is_fitted:
                raise SentinelError(
                    "LSTMDetector has not been fitted. Call fit() first."
                )
