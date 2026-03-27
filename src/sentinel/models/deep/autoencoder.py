"""Vanilla feedforward Autoencoder anomaly detector.

Learns a compressed representation of normal data via a symmetric
encoder-decoder architecture.  Anomaly scores are per-sample MSE
reconstruction errors: normal data reconstructs well, anomalies do not.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from sentinel.core.base_model import BaseAnomalyDetector
from sentinel.core.device import resolve_device
from sentinel.core.exceptions import SentinelError
from sentinel.core.registry import register_model

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ---------------------------------------------------------------------------
# Internal PyTorch module (only used when torch is available)
# ---------------------------------------------------------------------------

if HAS_TORCH:

    class _AutoencoderModule(nn.Module):
        """Symmetric encoder-decoder with ReLU activations.

        Args:
            input_dim: Number of input features.
            hidden_dims: Sizes of encoder hidden layers.  The decoder
                mirrors them in reverse order.
        """

        def __init__(self, input_dim: int, hidden_dims: list[int]) -> None:
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dims = hidden_dims

            # --- Encoder ---
            encoder_layers: list[nn.Module] = []
            prev_dim = input_dim
            for h_dim in hidden_dims:
                encoder_layers.append(nn.Linear(prev_dim, h_dim))
                encoder_layers.append(nn.ReLU())
                prev_dim = h_dim
            self.encoder = nn.Sequential(*encoder_layers)

            # --- Decoder (reverse of encoder) ---
            decoder_layers: list[nn.Module] = []
            reversed_dims = list(reversed(hidden_dims))
            for i in range(len(reversed_dims) - 1):
                decoder_layers.append(nn.Linear(reversed_dims[i], reversed_dims[i + 1]))
                decoder_layers.append(nn.ReLU())
            # Final layer maps back to input_dim (no activation).
            decoder_layers.append(nn.Linear(reversed_dims[-1], input_dim))
            self.decoder = nn.Sequential(*decoder_layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Encode and reconstruct *x*.

            Args:
                x: Input tensor of shape ``(batch, input_dim)``.

            Returns:
                Reconstructed tensor of the same shape.
            """
            z = self.encoder(x)
            return self.decoder(z)


# ---------------------------------------------------------------------------
# Public detector class
# ---------------------------------------------------------------------------

if HAS_TORCH:

    @register_model("autoencoder")
    class AutoencoderDetector(BaseAnomalyDetector):
        """Vanilla feedforward Autoencoder for anomaly detection.

        A symmetric encoder-decoder is trained on normal data to minimise
        MSE reconstruction error.  At inference, the per-sample MSE serves
        as an anomaly score: poorly reconstructed samples are more likely
        to be anomalous.

        Args:
            hidden_dims: Encoder layer sizes.  The decoder mirrors them.
            learning_rate: Adam optimiser learning rate.
            epochs: Number of training epochs.
            batch_size: Mini-batch size for the ``DataLoader``.
            device: Hardware target (``"auto"``, ``"cpu"``, ``"cuda"``,
                ``"mps"``).  ``"auto"`` resolves to the best available.

        Example::

            detector = AutoencoderDetector(hidden_dims=[64, 32, 16])
            detector.fit(X_train)
            scores = detector.score(X_test)
        """

        def __init__(
            self,
            hidden_dims: list[int] | None = None,
            learning_rate: float = 0.001,
            epochs: int = 100,
            batch_size: int = 32,
            device: str = "auto",
        ) -> None:
            self.hidden_dims = hidden_dims if hidden_dims is not None else [64, 32, 16]
            self.learning_rate = learning_rate
            self.epochs = epochs
            self.batch_size = batch_size
            self.device_str = device

            # Resolved at fit-time so that resolve_device() sees current
            # hardware state.
            self._device: torch.device | None = None
            self._model: _AutoencoderModule | None = None
            self._n_features: int | None = None
            self._is_fitted: bool = False

        # ------------------------------------------------------------------
        # Public interface
        # ------------------------------------------------------------------

        def fit(self, X: np.ndarray, **kwargs: Any) -> None:
            """Train the autoencoder on 2-D data.

            Seeds are set at the start so training is reproducible when
            the global seed has been configured.

            Args:
                X: Training data of shape ``(n_samples, n_features)``.
                **kwargs: Accepts ``seed`` (``int | None``) to set
                    torch manual seeds before training.

            Raises:
                SentinelError: If ``X`` has fewer than 2 rows.
            """
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            if X.shape[0] < 2:
                raise SentinelError(
                    f"Autoencoder requires at least 2 samples, got {X.shape[0]}"
                )

            # Seed management.
            seed: int | None = kwargs.get("seed")
            if seed is not None:
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

            self._n_features = X.shape[1]
            self._device = torch.device(resolve_device(self.device_str))

            # Build network.
            self._model = _AutoencoderModule(
                input_dim=self._n_features,
                hidden_dims=self.hidden_dims,
            ).to(self._device)

            # Data loader.
            tensor_x = torch.tensor(X, dtype=torch.float32)
            dataset = TensorDataset(tensor_x)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            # Training loop.
            criterion = nn.MSELoss()
            optimiser = torch.optim.Adam(
                self._model.parameters(), lr=self.learning_rate
            )

            self._model.train()
            for _epoch in range(self.epochs):
                for (batch,) in loader:
                    batch = batch.to(self._device)
                    reconstructed = self._model(batch)
                    loss = criterion(reconstructed, batch)

                    optimiser.zero_grad()
                    loss.backward()
                    optimiser.step()

            self._is_fitted = True

        def score(self, X: np.ndarray) -> np.ndarray:
            """Compute per-sample MSE reconstruction error.

            Args:
                X: Data of shape ``(n_samples, n_features)``.

            Returns:
                1-D array of length ``n_samples`` with anomaly scores.

            Raises:
                SentinelError: If the model has not been fitted.
            """
            self._check_fitted()

            if X.ndim == 1:
                X = X.reshape(-1, 1)

            self._model.eval()  # type: ignore[union-attr]
            tensor_x = torch.tensor(X, dtype=torch.float32).to(self._device)

            with torch.no_grad():
                reconstructed = self._model(tensor_x)  # type: ignore[misc]
                mse = torch.mean((tensor_x - reconstructed) ** 2, dim=1)

            return mse.cpu().numpy().astype(np.float64)

        def save(self, path: str) -> None:
            """Save model state_dict and config to disk.

            Two files are written into *path*:

            * ``config.json`` -- Architecture parameters and metadata.
            * ``model.pt`` -- PyTorch ``state_dict``.

            All writes are atomic (write to ``.tmp``, then ``os.rename``).

            Args:
                path: Directory in which to store artifacts.

            Raises:
                SentinelError: If the model has not been fitted.
            """
            self._check_fitted()
            os.makedirs(path, exist_ok=True)

            # -- config.json ---------------------------------------------------
            config = {
                "model_name": "autoencoder",
                "hidden_dims": self.hidden_dims,
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
            model_path = os.path.join(path, "model.pt")
            tmp_model = model_path + ".tmp"
            torch.save(
                self._model.state_dict(),  # type: ignore[union-attr]
                tmp_model,
            )
            os.rename(tmp_model, model_path)

        def load(self, path: str) -> None:
            """Load a previously saved autoencoder from disk.

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

            self.hidden_dims = config["hidden_dims"]
            self.learning_rate = float(config["learning_rate"])
            self.epochs = int(config["epochs"])
            self.batch_size = int(config["batch_size"])
            self.device_str = config["device"]
            self._n_features = int(config["n_features"])

            self._device = torch.device(resolve_device(self.device_str))

            self._model = _AutoencoderModule(
                input_dim=self._n_features,
                hidden_dims=self.hidden_dims,
            ).to(self._device)
            self._model.load_state_dict(
                torch.load(model_path, map_location=self._device, weights_only=True)
            )
            self._is_fitted = True

        def get_params(self) -> dict[str, Any]:
            """Return model parameters.

            Returns:
                Dict containing architecture parameters and metadata.
            """
            return {
                "model_name": "autoencoder",
                "hidden_dims": self.hidden_dims,
                "learning_rate": self.learning_rate,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
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
                    "AutoencoderDetector has not been fitted. Call fit() first."
                )
