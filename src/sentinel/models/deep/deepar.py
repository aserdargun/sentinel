"""DeepAR autoregressive anomaly detector.

Trains an autoregressive LSTM that outputs Gaussian parameters (mu, sigma)
per timestep per feature. The negative log-likelihood (NLL) of the observed
values under those Gaussians serves as the anomaly score -- higher NLL
indicates a more anomalous observation.

Reference: Salinas et al., "DeepAR: Probabilistic Forecasting with
Autoregressive Recurrent Networks", 2020.

Loss:
    L = -sum_t sum_f log N(x_{t,f} | mu_{t,f}, sigma_{t,f}^2)
    where sigma = softplus(linear(h_t)) to ensure positivity.

Inference uses single-step rolling forecast (no sampling).
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
import math
import os
from typing import Any

import numpy as np

from sentinel.core.base_model import BaseAnomalyDetector
from sentinel.core.device import resolve_device
from sentinel.core.exceptions import SentinelError

if HAS_TORCH:
    from sentinel.core.registry import register_model
    from sentinel.data.preprocessors import create_windows

    # Minimum sigma to avoid log(0) and division-by-zero in NLL computation.
    _MIN_SIGMA = 1e-6

    class _DeepARNetwork(nn.Module):
        """Internal autoregressive LSTM with Gaussian output heads.

        The network processes a sequence of observations and produces
        per-timestep Gaussian parameters (mu, sigma) for each feature.

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
            # Two heads: one for mu, one for pre-softplus sigma.
            self.mu_head = nn.Linear(hidden_dim, n_features)
            self.sigma_head = nn.Linear(hidden_dim, n_features)
            self.softplus = nn.Softplus()

        def forward(
            self,
            x: torch.Tensor,
            hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
        ) -> tuple[
            torch.Tensor,
            torch.Tensor,
            tuple[torch.Tensor, torch.Tensor],
        ]:
            """Forward pass: produce Gaussian parameters for each timestep.

            Args:
                x: Input tensor of shape ``(batch, seq_len, n_features)``.
                hidden: Optional initial hidden state for the LSTM.

            Returns:
                Tuple of (mu, sigma, hidden_state) where:
                - mu: ``(batch, seq_len, n_features)``
                - sigma: ``(batch, seq_len, n_features)`` (positive)
                - hidden_state: LSTM hidden and cell state tuple.
            """
            # lstm_out: (batch, seq_len, hidden_dim)
            lstm_out, hidden_state = self.lstm(x, hidden)

            mu = self.mu_head(lstm_out)
            sigma = self.softplus(self.sigma_head(lstm_out)) + _MIN_SIGMA

            return mu, sigma, hidden_state

    def _gaussian_nll(
        x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        """Compute per-element Gaussian negative log-likelihood.

        Args:
            x: Ground truth values ``(batch, seq_len, n_features)``.
            mu: Predicted means ``(batch, seq_len, n_features)``.
            sigma: Predicted stds ``(batch, seq_len, n_features)``.

        Returns:
            Per-element NLL tensor of the same shape as ``x``.
        """
        variance = sigma**2
        return 0.5 * (torch.log(2 * math.pi * variance) + (x - mu) ** 2 / variance)

    @register_model("deepar")
    class DeepARDetector(BaseAnomalyDetector):
        """DeepAR autoregressive probabilistic anomaly detector.

        Trains an autoregressive LSTM to model the conditional distribution
        of each feature at each timestep as a Gaussian N(mu, sigma^2). The
        model is trained with teacher forcing: at training time, the true
        previous observation is fed as input regardless of the model's
        prediction.

        At scoring time, single-step rolling forecasts are produced. The
        anomaly score for each timestep is the sum of per-feature negative
        log-likelihoods: points that are unlikely under the learned
        distribution receive high scores.

        Args:
            hidden_dim: LSTM hidden state dimension.
            num_layers: Number of stacked LSTM layers.
            seq_len: Input window length for autoregressive conditioning.
            learning_rate: Optimizer learning rate.
            epochs: Number of training epochs.
            batch_size: Mini-batch size for training.
            dropout: Dropout probability between LSTM layers.
            device: Device string (``"auto"``, ``"cpu"``, ``"cuda"``,
                ``"mps"``).

        Example::

            detector = DeepARDetector(hidden_dim=64, seq_len=50)
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
            # hardware state (matches VAE/other deep models).
            self._device: torch.device | None = None
            self._model: _DeepARNetwork | None = None
            self._n_features: int | None = None
            self._is_fitted: bool = False

        # ------------------------------------------------------------------
        # Public interface
        # ------------------------------------------------------------------

        def fit(self, X: np.ndarray, **kwargs: Any) -> None:
            """Train the DeepAR model on data.

            Creates sliding windows of length ``seq_len``. Within each
            window, the model receives timesteps ``[0..seq_len-2]`` as input
            and predicts the Gaussian distribution for timesteps
            ``[1..seq_len-1]`` (teacher forcing). The loss is the mean
            Gaussian NLL over all timesteps and features.

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
                    f"DeepAR requires at least {self.seq_len} samples "
                    f"(seq_len), got {n_samples}"
                )

            self._n_features = n_features

            # Build sliding windows: (n_windows, seq_len, n_features).
            windows = create_windows(X, seq_len=self.seq_len, stride=1)

            # Teacher forcing: input = windows[:, :-1, :], target = windows[:, 1:, :]
            inputs = windows[:, :-1, :]  # (n_windows, seq_len-1, n_features)
            targets = windows[:, 1:, :]  # (n_windows, seq_len-1, n_features)

            inputs_t = torch.tensor(inputs, dtype=torch.float32)
            targets_t = torch.tensor(targets, dtype=torch.float32)

            dataset = TensorDataset(inputs_t, targets_t)
            loader = DataLoader(
                dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
            )

            # Resolve device at fit-time.
            self._device = torch.device(resolve_device(self.device_str))

            # Build network and move to device.
            self._model = _DeepARNetwork(
                n_features=n_features,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                dropout=self.dropout,
            )
            self._model.to(self._device)

            optimizer = torch.optim.Adam(
                self._model.parameters(), lr=self.learning_rate
            )

            self._model.train()
            for _epoch in range(self.epochs):
                for batch_x, batch_y in loader:
                    batch_x = batch_x.to(self._device)
                    batch_y = batch_y.to(self._device)

                    mu, sigma, _ = self._model(batch_x)
                    nll = _gaussian_nll(batch_y, mu, sigma)
                    loss = nll.mean()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            self._is_fitted = True

        def score(self, X: np.ndarray) -> np.ndarray:
            """Compute anomaly scores via single-step rolling NLL.

            For each sliding window of ``seq_len`` steps, the model
            conditions on the first ``seq_len - 1`` steps and predicts the
            Gaussian distribution for each subsequent step. The anomaly
            score at each timestep is the sum of per-feature NLL values.

            The first ``seq_len - 1`` samples that lack a full conditioning
            window are assigned the mean score of the remaining timesteps.

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
                    f"DeepAR scoring requires at least {self.seq_len} "
                    f"samples (seq_len), got {n_samples}"
                )

            # Build windows of length seq_len.
            windows = create_windows(X, seq_len=self.seq_len, stride=1)

            # Input: steps [0..seq_len-2], target: last step [seq_len-1].
            inputs = windows[:, :-1, :]  # (n_windows, seq_len-1, n_features)
            actuals = windows[:, -1, :]  # (n_windows, n_features)

            inputs_t = torch.tensor(inputs, dtype=torch.float32)
            inputs_t = inputs_t.to(self._device)

            assert self._model is not None
            self._model.eval()
            with torch.no_grad():
                mu, sigma, _ = self._model(inputs_t)

            # Take prediction for the last timestep only.
            # mu/sigma: (n_windows, seq_len-1, n_features) -> last step
            mu_last = mu[:, -1, :].cpu().numpy()  # (n_windows, n_features)
            sigma_last = sigma[:, -1, :].cpu().numpy()

            # Per-timestep NLL summed over features:
            # NLL_f = 0.5 * (log(2*pi*sigma^2) + (x - mu)^2 / sigma^2)
            # score = sum_f NLL_f
            variance = sigma_last**2
            nll_per_feature = 0.5 * (
                np.log(2 * np.pi * variance) + (actuals - mu_last) ** 2 / variance
            )
            window_scores = np.sum(nll_per_feature, axis=1)  # (n_windows,)

            # Pad: each window score corresponds to the last sample in the
            # window (index seq_len - 1 onwards). The first seq_len - 1
            # samples get the mean score.
            pad_len = self.seq_len - 1
            mean_score = float(np.mean(window_scores))
            scores = np.full(n_samples, mean_score, dtype=np.float64)
            scores[pad_len:] = window_scores

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
                "model_name": "deepar",
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
            """Load a previously saved DeepAR model from disk.

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

            # Re-resolve device at load time (handles cross-device loading).
            self._device = torch.device(resolve_device(self.device_str))

            # Rebuild the network and load weights.
            self._model = _DeepARNetwork(
                n_features=self._n_features,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                dropout=self.dropout,
            )
            self._model.load_state_dict(
                torch.load(
                    model_path,
                    map_location=self._device,
                    weights_only=True,
                )
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
                "model_name": "deepar",
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
                    "DeepARDetector has not been fitted. Call fit() first."
                )
