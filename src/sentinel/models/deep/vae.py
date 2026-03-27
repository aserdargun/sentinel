"""Variational Autoencoder (VAE) anomaly detector.

Learns a latent-variable generative model of normal data via the ELBO
objective: reconstruction loss (MSE) plus a KL divergence regularizer
that keeps the approximate posterior close to a standard normal prior.

At inference the anomaly score is the per-sample MSE reconstruction error
only -- the KL term is used during training but excluded from scoring
because it measures latent-space regularity, not data fidelity.

Reference: Kingma & Welling, "Auto-Encoding Variational Bayes", 2014.
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

    class _VAEModule(nn.Module):
        """Variational Autoencoder with symmetric encoder/decoder.

        The encoder produces ``mu`` and ``log_var`` for the approximate
        posterior ``q(z|x) = N(mu, diag(exp(log_var)))``.  A sample is
        drawn via the reparameterization trick and passed through the
        decoder to reconstruct the input.

        Args:
            input_dim: Number of input features.
            hidden_dims: Sizes of encoder hidden layers (decoder mirrors
                them in reverse).
            latent_dim: Dimensionality of the latent space *z*.
        """

        def __init__(
            self,
            input_dim: int,
            hidden_dims: list[int],
            latent_dim: int,
        ) -> None:
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dims = hidden_dims
            self.latent_dim = latent_dim

            # --- Encoder (shared body) ---
            encoder_layers: list[nn.Module] = []
            prev_dim = input_dim
            for h_dim in hidden_dims:
                encoder_layers.append(nn.Linear(prev_dim, h_dim))
                encoder_layers.append(nn.ReLU())
                prev_dim = h_dim
            self.encoder = nn.Sequential(*encoder_layers)

            # Separate heads for mu and log_var.
            self.fc_mu = nn.Linear(prev_dim, latent_dim)
            self.fc_log_var = nn.Linear(prev_dim, latent_dim)

            # --- Decoder ---
            decoder_layers: list[nn.Module] = []
            reversed_dims = list(reversed(hidden_dims))
            # First layer maps from latent_dim to the widest hidden dim.
            decoder_layers.append(nn.Linear(latent_dim, reversed_dims[0]))
            decoder_layers.append(nn.ReLU())
            for i in range(len(reversed_dims) - 1):
                decoder_layers.append(nn.Linear(reversed_dims[i], reversed_dims[i + 1]))
                decoder_layers.append(nn.ReLU())
            # Final layer maps back to input_dim (no activation).
            decoder_layers.append(nn.Linear(reversed_dims[-1], input_dim))
            self.decoder = nn.Sequential(*decoder_layers)

        def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """Run the encoder and return distributional parameters.

            Args:
                x: Input tensor of shape ``(batch, input_dim)``.

            Returns:
                Tuple ``(mu, log_var)`` each of shape ``(batch, latent_dim)``.
            """
            h = self.encoder(x)
            return self.fc_mu(h), self.fc_log_var(h)

        @staticmethod
        def reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
            """Sample z ~ q(z|x) using the reparameterization trick.

            ``z = mu + eps * exp(0.5 * log_var)`` where ``eps ~ N(0, I)``.

            Args:
                mu: Mean of the approximate posterior.
                log_var: Log-variance of the approximate posterior.

            Returns:
                Sampled latent tensor of the same shape as *mu*.
            """
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std

        def decode(self, z: torch.Tensor) -> torch.Tensor:
            """Reconstruct input from latent code.

            Args:
                z: Latent tensor of shape ``(batch, latent_dim)``.

            Returns:
                Reconstructed tensor of shape ``(batch, input_dim)``.
            """
            return self.decoder(z)

        def forward(
            self, x: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Full forward pass: encode, sample, decode.

            Args:
                x: Input tensor of shape ``(batch, input_dim)``.

            Returns:
                Tuple ``(x_hat, mu, log_var)`` where *x_hat* is the
                reconstruction and *mu*/*log_var* parameterise the
                approximate posterior.
            """
            mu, log_var = self.encode(x)
            z = self.reparameterize(mu, log_var)
            x_hat = self.decode(z)
            return x_hat, mu, log_var


# ---------------------------------------------------------------------------
# Public detector class
# ---------------------------------------------------------------------------

if HAS_TORCH:

    @register_model("vae")
    class VAEDetector(BaseAnomalyDetector):
        """Variational Autoencoder for anomaly detection.

        The model is trained on normal data by maximising the ELBO:

        .. math::

            \\mathcal{L} = \\mathrm{MSE}(x, \\hat{x})
                         + \\beta \\cdot \\mathrm{KL}(q(z|x) \\| p(z))

        where ``p(z) = N(0, I)`` and ``beta`` (``kl_weight``) controls the
        strength of the KL regularizer.

        At inference the anomaly score is the per-sample reconstruction MSE
        only -- the KL divergence is **not** included because it measures
        latent regularity rather than input fidelity.

        Args:
            latent_dim: Dimensionality of the latent space.
            kl_weight: Multiplier for the KL divergence term (beta-VAE).
            hidden_dims: Encoder layer sizes. The decoder mirrors them.
            learning_rate: Adam optimiser learning rate.
            epochs: Number of training epochs.
            batch_size: Mini-batch size for the ``DataLoader``.
            device: Hardware target (``"auto"``, ``"cpu"``, ``"cuda"``,
                ``"mps"``).  ``"auto"`` resolves to the best available.

        Example::

            detector = VAEDetector(latent_dim=16, kl_weight=1.0)
            detector.fit(X_train)
            scores = detector.score(X_test)
        """

        def __init__(
            self,
            latent_dim: int = 16,
            kl_weight: float = 1.0,
            hidden_dims: list[int] | None = None,
            learning_rate: float = 0.001,
            epochs: int = 100,
            batch_size: int = 32,
            device: str = "auto",
        ) -> None:
            self.latent_dim = latent_dim
            self.kl_weight = kl_weight
            self.hidden_dims = hidden_dims if hidden_dims is not None else [64, 32]
            self.learning_rate = learning_rate
            self.epochs = epochs
            self.batch_size = batch_size
            self.device_str = device

            # Resolved at fit-time so that resolve_device() sees current
            # hardware state.
            self._device: torch.device | None = None
            self._model: _VAEModule | None = None
            self._n_features: int | None = None
            self._is_fitted: bool = False

        # ------------------------------------------------------------------
        # Public interface
        # ------------------------------------------------------------------

        def fit(self, X: np.ndarray, **kwargs: Any) -> None:
            """Train the VAE on 2-D data.

            The ELBO loss is ``MSE(x, x_hat) + kl_weight * KL(q||p)``
            where KL is computed analytically for the diagonal-Gaussian
            posterior against a standard-normal prior.

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
                    f"VAE requires at least 2 samples, got {X.shape[0]}"
                )

            # Seed management.
            seed: int | None = kwargs.get("seed")
            if seed is not None:
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

            self._n_features = X.shape[1]
            self._device = torch.device(resolve_device(self.device_str))

            # Build network.
            self._model = _VAEModule(
                input_dim=self._n_features,
                hidden_dims=self.hidden_dims,
                latent_dim=self.latent_dim,
            ).to(self._device)

            # Data loader.
            tensor_x = torch.tensor(X, dtype=torch.float32)
            dataset = TensorDataset(tensor_x)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            # Optimiser.
            optimiser = torch.optim.Adam(
                self._model.parameters(), lr=self.learning_rate
            )

            # Training loop.
            self._model.train()
            for _epoch in range(self.epochs):
                for (batch,) in loader:
                    batch = batch.to(self._device)
                    x_hat, mu, log_var = self._model(batch)

                    loss = self._elbo_loss(batch, x_hat, mu, log_var)

                    optimiser.zero_grad()
                    loss.backward()
                    optimiser.step()

            self._is_fitted = True

        def score(self, X: np.ndarray) -> np.ndarray:
            """Compute per-sample MSE reconstruction error.

            Only reconstruction error is used for scoring -- the KL term
            is excluded because it measures latent regularity, not how
            well an individual sample is reconstructed.

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
                # Use the posterior mean directly (no sampling) so that
                # scores are deterministic across calls.
                mu, _log_var = self._model.encode(tensor_x)  # type: ignore[union-attr]
                x_hat = self._model.decode(mu)  # type: ignore[union-attr]
                mse = torch.mean((tensor_x - x_hat) ** 2, dim=1)

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
                "model_name": "vae",
                "latent_dim": self.latent_dim,
                "kl_weight": self.kl_weight,
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
            """Load a previously saved VAE from disk.

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

            self.latent_dim = int(config["latent_dim"])
            self.kl_weight = float(config["kl_weight"])
            self.hidden_dims = config["hidden_dims"]
            self.learning_rate = float(config["learning_rate"])
            self.epochs = int(config["epochs"])
            self.batch_size = int(config["batch_size"])
            self.device_str = config["device"]
            self._n_features = int(config["n_features"])

            self._device = torch.device(resolve_device(self.device_str))

            self._model = _VAEModule(
                input_dim=self._n_features,
                hidden_dims=self.hidden_dims,
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
                "model_name": "vae",
                "latent_dim": self.latent_dim,
                "kl_weight": self.kl_weight,
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

        def _elbo_loss(
            self,
            x: torch.Tensor,
            x_hat: torch.Tensor,
            mu: torch.Tensor,
            log_var: torch.Tensor,
        ) -> torch.Tensor:
            """Compute the negative ELBO loss.

            ``L = MSE(x, x_hat) + kl_weight * KL(q(z|x) || p(z))``

            where KL is computed analytically:
            ``KL = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))``

            Args:
                x: Original input of shape ``(batch, n_features)``.
                x_hat: Reconstruction of the same shape.
                mu: Mean of the approximate posterior, shape
                    ``(batch, latent_dim)``.
                log_var: Log-variance, same shape as *mu*.

            Returns:
                Scalar loss tensor.
            """
            recon_loss = nn.functional.mse_loss(x_hat, x, reduction="mean")
            kl_loss = -0.5 * torch.mean(
                torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
            )
            return recon_loss + self.kl_weight * kl_loss

        def _check_fitted(self) -> None:
            """Raise if the model has not been fitted."""
            if not self._is_fitted:
                raise SentinelError(
                    "VAEDetector has not been fitted. Call fit() first."
                )
