"""GANomaly-style anomaly detector.

Implements the encoder-decoder-encoder architecture from Akcay et al., 2018.
There is no separate discriminator. Instead, two encoders and one decoder
form a pipeline: Encoder1 maps input to latent space, the Decoder
reconstructs from that latent code, and Encoder2 re-encodes the
reconstruction. The anomaly score combines reconstruction error with
latent-space distance.

Loss:
    L = lambda_recon * MSE(x, G(x)) + lambda_latent * MSE(z1, E2(G(x)))
    where z1 = E1(x), G(x) = Decoder(z1), z2 = E2(G(x))

Anomaly score:
    MSE(x, G(x)) + MSE(z1, z2)   (per sample)
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
# Internal PyTorch modules (only used when torch is available)
# ---------------------------------------------------------------------------

if HAS_TORCH:

    class _Encoder(nn.Module):
        """MLP encoder mapping input features to a latent vector.

        Args:
            input_dim: Number of input features.
            hidden_dim: Width of the hidden layer.
            latent_dim: Dimensionality of the latent representation.
        """

        def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Encode *x* into latent space.

            Args:
                x: Input tensor of shape ``(batch, input_dim)``.

            Returns:
                Latent tensor of shape ``(batch, latent_dim)``.
            """
            return self.net(x)

    class _Decoder(nn.Module):
        """MLP decoder reconstructing input features from a latent vector.

        Args:
            latent_dim: Dimensionality of the latent representation.
            hidden_dim: Width of the hidden layer.
            output_dim: Number of output features (must equal input_dim).
        """

        def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            """Decode latent *z* back to input space.

            Args:
                z: Latent tensor of shape ``(batch, latent_dim)``.

            Returns:
                Reconstructed tensor of shape ``(batch, output_dim)``.
            """
            return self.net(z)

    class _GANomalyModule(nn.Module):
        """Combined Encoder1-Decoder-Encoder2 architecture.

        Args:
            input_dim: Number of input features.
            encoder_dim: Hidden-layer width for both encoders.
            decoder_dim: Hidden-layer width for the decoder.
            latent_dim: Size of the latent representation.
        """

        def __init__(
            self,
            input_dim: int,
            encoder_dim: int,
            decoder_dim: int,
            latent_dim: int,
        ) -> None:
            super().__init__()
            self.encoder1 = _Encoder(input_dim, encoder_dim, latent_dim)
            self.decoder = _Decoder(latent_dim, decoder_dim, input_dim)
            self.encoder2 = _Encoder(input_dim, encoder_dim, latent_dim)

        def forward(
            self, x: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Forward pass through the full pipeline.

            Args:
                x: Input tensor of shape ``(batch, input_dim)``.

            Returns:
                Tuple of (reconstructed x, z1 from Encoder1, z2 from Encoder2).
            """
            z1 = self.encoder1(x)
            x_hat = self.decoder(z1)
            z2 = self.encoder2(x_hat)
            return x_hat, z1, z2


# ---------------------------------------------------------------------------
# Public detector class
# ---------------------------------------------------------------------------

if HAS_TORCH:

    @register_model("gan")
    class GANDetector(BaseAnomalyDetector):
        """GANomaly-style anomaly detector (Akcay et al., 2018).

        Uses an encoder-decoder-encoder architecture without a separate
        discriminator.  Encoder1 maps the input to latent space, the
        Decoder reconstructs from that code, and Encoder2 re-encodes the
        reconstruction.  Training minimises a weighted sum of reconstruction
        loss and latent-consistency loss.  At inference the anomaly score
        is ``MSE(x, G(x)) + MSE(z1, z2)`` per sample.

        Args:
            encoder_dim: Hidden-layer width for both encoders.
            decoder_dim: Hidden-layer width for the decoder.
            latent_dim: Dimensionality of the shared latent space.
            lambda_recon: Weight for the reconstruction loss term.
            lambda_latent: Weight for the latent-consistency loss term.
            learning_rate: Adam optimiser learning rate.
            epochs: Number of training epochs.
            batch_size: Mini-batch size for the DataLoader.
            device: Hardware target (``"auto"``, ``"cpu"``, ``"cuda"``,
                ``"mps"``).  ``"auto"`` resolves to the best available.

        Example::

            detector = GANDetector(encoder_dim=64, latent_dim=16)
            detector.fit(X_train)
            scores = detector.score(X_test)
        """

        def __init__(
            self,
            encoder_dim: int = 64,
            decoder_dim: int = 64,
            latent_dim: int = 16,
            lambda_recon: float = 1.0,
            lambda_latent: float = 1.0,
            learning_rate: float = 0.001,
            epochs: int = 100,
            batch_size: int = 32,
            device: str = "auto",
        ) -> None:
            self.encoder_dim = encoder_dim
            self.decoder_dim = decoder_dim
            self.latent_dim = latent_dim
            self.lambda_recon = lambda_recon
            self.lambda_latent = lambda_latent
            self.learning_rate = learning_rate
            self.epochs = epochs
            self.batch_size = batch_size
            self.device_str = device

            # Resolved at fit-time so that resolve_device() sees current
            # hardware state.
            self._device: torch.device | None = None
            self._model: _GANomalyModule | None = None
            self._n_features: int | None = None
            self._is_fitted: bool = False

        # ------------------------------------------------------------------
        # Public interface
        # ------------------------------------------------------------------

        def fit(self, X: np.ndarray, **kwargs: Any) -> None:
            """Train the GANomaly model on 2-D data.

            The combined loss is:
                ``lambda_recon * MSE(x, x_hat) + lambda_latent * MSE(z1, z2)``

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
                    f"GANDetector requires at least 2 samples, got {X.shape[0]}"
                )

            # Seed management.
            seed: int | None = kwargs.get("seed")
            if seed is not None:
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

            self._n_features = X.shape[1]
            self._device = torch.device(resolve_device(self.device_str))

            # Build network.
            self._model = _GANomalyModule(
                input_dim=self._n_features,
                encoder_dim=self.encoder_dim,
                decoder_dim=self.decoder_dim,
                latent_dim=self.latent_dim,
            ).to(self._device)

            # Data loader.
            tensor_x = torch.tensor(X, dtype=torch.float32)
            dataset = TensorDataset(tensor_x)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            # Training loop.
            mse_loss = nn.MSELoss()
            optimiser = torch.optim.Adam(
                self._model.parameters(), lr=self.learning_rate
            )

            self._model.train()
            for _epoch in range(self.epochs):
                for (batch,) in loader:
                    batch = batch.to(self._device)
                    x_hat, z1, z2 = self._model(batch)

                    recon_loss = mse_loss(x_hat, batch)
                    latent_loss = mse_loss(z1, z2)
                    loss = (
                        self.lambda_recon * recon_loss
                        + self.lambda_latent * latent_loss
                    )

                    optimiser.zero_grad()
                    loss.backward()
                    optimiser.step()

            self._is_fitted = True

        def score(self, X: np.ndarray) -> np.ndarray:
            """Compute per-sample anomaly scores.

            The score for each sample is:
                ``mean((x - x_hat)^2, dim=features) + mean((z1 - z2)^2, dim=latent)``

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
                x_hat, z1, z2 = self._model(tensor_x)  # type: ignore[misc]
                recon_error = torch.mean((tensor_x - x_hat) ** 2, dim=1)
                latent_error = torch.mean((z1 - z2) ** 2, dim=1)
                scores = recon_error + latent_error

            return scores.cpu().numpy().astype(np.float64)

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
                "model_name": "gan",
                "encoder_dim": self.encoder_dim,
                "decoder_dim": self.decoder_dim,
                "latent_dim": self.latent_dim,
                "lambda_recon": self.lambda_recon,
                "lambda_latent": self.lambda_latent,
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
            """Load a previously saved GANomaly model from disk.

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

            self.encoder_dim = int(config["encoder_dim"])
            self.decoder_dim = int(config["decoder_dim"])
            self.latent_dim = int(config["latent_dim"])
            self.lambda_recon = float(config["lambda_recon"])
            self.lambda_latent = float(config["lambda_latent"])
            self.learning_rate = float(config["learning_rate"])
            self.epochs = int(config["epochs"])
            self.batch_size = int(config["batch_size"])
            self.device_str = config["device"]
            self._n_features = int(config["n_features"])

            self._device = torch.device(resolve_device(self.device_str))

            self._model = _GANomalyModule(
                input_dim=self._n_features,
                encoder_dim=self.encoder_dim,
                decoder_dim=self.decoder_dim,
                latent_dim=self.latent_dim,
            ).to(self._device)
            self._model.load_state_dict(
                torch.load(
                    model_path,
                    map_location=self._device,
                    weights_only=True,
                )
            )
            self._is_fitted = True

        def get_params(self) -> dict[str, Any]:
            """Return model parameters.

            Returns:
                Dict containing architecture parameters and metadata.
            """
            return {
                "model_name": "gan",
                "encoder_dim": self.encoder_dim,
                "decoder_dim": self.decoder_dim,
                "latent_dim": self.latent_dim,
                "lambda_recon": self.lambda_recon,
                "lambda_latent": self.lambda_latent,
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
                    "GANDetector has not been fitted. Call fit() first."
                )
