"""DDPM-based Diffusion anomaly detector.

Implements the Denoising Diffusion Probabilistic Model (Ho et al., 2020)
for anomaly detection.  A noise-prediction network epsilon_theta is trained
on normal data using the simplified denoising objective.  At inference the
anomaly score is the MSE between the original input and a denoised
reconstruction obtained at a fixed noise level t = T // 2.

Loss formulation:
    Forward:  q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
    Reverse:  epsilon_theta(x_t, t) predicts the noise added at step t
    L = MSE(epsilon, epsilon_theta(x_t, t))
    Anomaly score = MSE(x_0, x_hat_0)  where x_hat_0 = denoise(x_t, t) at t = T//2
"""

from __future__ import annotations

import json
import math
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
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_TIMESTEPS = 100
_DEFAULT_HIDDEN_DIM = 64
_DEFAULT_NUM_LAYERS = 2
_DEFAULT_LEARNING_RATE = 0.001
_DEFAULT_EPOCHS = 100
_DEFAULT_BATCH_SIZE = 32
_DEFAULT_BETA_START = 1e-4
_DEFAULT_BETA_END = 0.02
_TIME_EMBED_DIM = 64

# ---------------------------------------------------------------------------
# Internal PyTorch modules (only used when torch is available)
# ---------------------------------------------------------------------------

if HAS_TORCH:

    class _SinusoidalTimeEmbedding(nn.Module):
        """Sinusoidal positional encoding of diffusion timestep t.

        Produces a fixed-dimensional embedding vector from a scalar timestep,
        following the positional encoding scheme from "Attention Is All You
        Need" (Vaswani et al., 2017).

        Args:
            embed_dim: Dimensionality of the output embedding.
        """

        def __init__(self, embed_dim: int) -> None:
            super().__init__()
            self.embed_dim = embed_dim

        def forward(self, t: torch.Tensor) -> torch.Tensor:
            """Embed scalar timesteps into a vector representation.

            Args:
                t: Integer timestep tensor of shape ``(batch,)``.

            Returns:
                Embedding tensor of shape ``(batch, embed_dim)``.
            """
            half_dim = self.embed_dim // 2
            log_scale = -math.log(10000.0) / (half_dim - 1)
            freqs = torch.exp(
                torch.arange(half_dim, dtype=torch.float32, device=t.device) * log_scale
            )
            # t may be int, cast to float for multiplication
            args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
            return torch.cat([torch.sin(args), torch.cos(args)], dim=1)

    class _NoisePredictor(nn.Module):
        """MLP noise predictor epsilon_theta(x_t, t).

        Takes the noisy input x_t concatenated with a sinusoidal time
        embedding and passes through hidden layers to predict the noise
        that was added.

        Args:
            input_dim: Number of data features (dimensionality of x).
            hidden_dim: Width of each hidden layer.
            num_layers: Number of hidden layers (minimum 1).
            time_embed_dim: Dimensionality of the sinusoidal time embedding.
        """

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            num_layers: int,
            time_embed_dim: int = _TIME_EMBED_DIM,
        ) -> None:
            super().__init__()
            self.time_embed = _SinusoidalTimeEmbedding(time_embed_dim)

            layers: list[nn.Module] = []
            prev_dim = input_dim + time_embed_dim
            for _ in range(num_layers):
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
                prev_dim = hidden_dim
            layers.append(nn.Linear(prev_dim, input_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            """Predict the noise added at timestep t.

            Args:
                x_t: Noisy input tensor of shape ``(batch, input_dim)``.
                t: Integer timestep tensor of shape ``(batch,)``.

            Returns:
                Predicted noise tensor of the same shape as ``x_t``.
            """
            t_emb = self.time_embed(t)
            h = torch.cat([x_t, t_emb], dim=1)
            return self.net(h)

# ---------------------------------------------------------------------------
# Public detector class
# ---------------------------------------------------------------------------

if HAS_TORCH:

    @register_model("diffusion")
    class DiffusionDetector(BaseAnomalyDetector):
        """DDPM-based Diffusion anomaly detector.

        Trains a denoising diffusion probabilistic model on normal data.
        At inference the anomaly score is the MSE between the original
        sample and a denoised reconstruction at a fixed noise level
        ``t = timesteps // 2``.

        Args:
            timesteps: Number of diffusion steps T in the noise schedule.
            hidden_dim: Width of the noise predictor hidden layers.
            num_layers: Number of hidden layers in the noise predictor.
            learning_rate: Adam optimiser learning rate.
            epochs: Number of training epochs.
            batch_size: Mini-batch size for the DataLoader.
            beta_start: Starting value of the linear beta schedule.
            beta_end: Ending value of the linear beta schedule.
            device: Hardware target (``"auto"``, ``"cpu"``, ``"cuda"``,
                ``"mps"``).  ``"auto"`` resolves to the best available.

        Example::

            detector = DiffusionDetector(timesteps=100, hidden_dim=64)
            detector.fit(X_train)
            scores = detector.score(X_test)
        """

        def __init__(
            self,
            timesteps: int = _DEFAULT_TIMESTEPS,
            hidden_dim: int = _DEFAULT_HIDDEN_DIM,
            num_layers: int = _DEFAULT_NUM_LAYERS,
            learning_rate: float = _DEFAULT_LEARNING_RATE,
            epochs: int = _DEFAULT_EPOCHS,
            batch_size: int = _DEFAULT_BATCH_SIZE,
            beta_start: float = _DEFAULT_BETA_START,
            beta_end: float = _DEFAULT_BETA_END,
            device: str = "auto",
        ) -> None:
            self.timesteps = timesteps
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.learning_rate = learning_rate
            self.epochs = epochs
            self.batch_size = batch_size
            self.beta_start = beta_start
            self.beta_end = beta_end
            self.device_str = device

            # Populated during fit / load.
            self._device: torch.device | None = None
            self._model: _NoisePredictor | None = None
            self._n_features: int | None = None
            self._is_fitted: bool = False

            # Noise schedule tensors (populated in _build_schedule).
            self._betas: torch.Tensor | None = None
            self._alphas: torch.Tensor | None = None
            self._alpha_bar: torch.Tensor | None = None

        # ------------------------------------------------------------------
        # Public interface
        # ------------------------------------------------------------------

        def fit(self, X: np.ndarray, **kwargs: Any) -> None:
            """Train the noise predictor on 2-D data.

            The model is trained on normal data only (anomalies should be
            filtered upstream).  Uses the simplified DDPM denoising objective:
            ``L = MSE(epsilon, epsilon_theta(x_t, t))``.

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
                    f"DiffusionDetector requires at least 2 samples, got {X.shape[0]}"
                )

            # Seed management.
            seed: int | None = kwargs.get("seed")
            if seed is not None:
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

            self._n_features = X.shape[1]
            self._device = torch.device(resolve_device(self.device_str))

            # Build noise schedule.
            self._build_schedule()

            # Build noise predictor network.
            self._model = _NoisePredictor(
                input_dim=self._n_features,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
            ).to(self._device)

            # Data loader.
            tensor_x = torch.tensor(X, dtype=torch.float32)
            dataset = TensorDataset(tensor_x)
            loader = DataLoader(
                dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
            )

            # Training loop.
            criterion = nn.MSELoss()
            optimiser = torch.optim.Adam(
                self._model.parameters(), lr=self.learning_rate
            )

            self._model.train()
            for _epoch in range(self.epochs):
                for (batch,) in loader:
                    batch = batch.to(self._device)
                    batch_len = batch.shape[0]

                    # Sample random timesteps for each item in the batch.
                    t = torch.randint(
                        0, self.timesteps, (batch_len,), device=self._device
                    )

                    # Sample noise and create noisy input via forward process.
                    epsilon = torch.randn_like(batch)
                    x_t = self._forward_diffusion(batch, t, epsilon)

                    # Predict the noise and compute loss.
                    epsilon_pred = self._model(x_t, t)
                    loss = criterion(epsilon_pred, epsilon)

                    optimiser.zero_grad()
                    loss.backward()
                    optimiser.step()

            self._is_fitted = True

        def score(self, X: np.ndarray) -> np.ndarray:
            """Compute per-sample anomaly scores via denoised reconstruction.

            For each sample x_0, noise is added at the fixed level
            ``t = T // 2`` and then removed using the learned noise predictor.
            The anomaly score is the MSE between x_0 and the denoised
            reconstruction x_hat_0.

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

            # Fixed noise level: mid-schedule.
            t_fixed = self.timesteps // 2
            t_tensor = torch.full(
                (tensor_x.shape[0],),
                t_fixed,
                dtype=torch.long,
                device=self._device,
            )

            with torch.no_grad():
                # Add noise at t = T//2.
                epsilon = torch.randn_like(tensor_x)
                x_t = self._forward_diffusion(tensor_x, t_tensor, epsilon)

                # Denoise to recover x_hat_0.
                x_hat_0 = self._denoise(x_t, t_tensor)

                # Anomaly score = per-sample MSE(x_0, x_hat_0).
                mse = torch.mean((tensor_x - x_hat_0) ** 2, dim=1)

            return mse.cpu().numpy().astype(np.float64)

        def save(self, path: str) -> None:
            """Save model state_dict and config to disk.

            Two files are written into *path*:

            * ``config.json`` -- Architecture parameters and schedule config.
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
                "model_name": "diffusion",
                "timesteps": self.timesteps,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "learning_rate": self.learning_rate,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "beta_start": self.beta_start,
                "beta_end": self.beta_end,
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
            """Load a previously saved diffusion model from disk.

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

            self.timesteps = int(config["timesteps"])
            self.hidden_dim = int(config["hidden_dim"])
            self.num_layers = int(config["num_layers"])
            self.learning_rate = float(config["learning_rate"])
            self.epochs = int(config["epochs"])
            self.batch_size = int(config["batch_size"])
            self.beta_start = float(config["beta_start"])
            self.beta_end = float(config["beta_end"])
            self.device_str = config["device"]
            self._n_features = int(config["n_features"])

            self._device = torch.device(resolve_device(self.device_str))

            # Rebuild schedule and network.
            self._build_schedule()

            self._model = _NoisePredictor(
                input_dim=self._n_features,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
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
                "model_name": "diffusion",
                "timesteps": self.timesteps,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "learning_rate": self.learning_rate,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "beta_start": self.beta_start,
                "beta_end": self.beta_end,
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
                    "DiffusionDetector has not been fitted. Call fit() first."
                )

        def _build_schedule(self) -> None:
            """Compute the linear beta noise schedule and derived quantities.

            Populates ``_betas``, ``_alphas``, and ``_alpha_bar`` tensors
            on the target device.
            """
            self._betas = torch.linspace(
                self.beta_start, self.beta_end, self.timesteps
            ).to(self._device)
            self._alphas = (1.0 - self._betas).to(self._device)
            self._alpha_bar = torch.cumprod(self._alphas, dim=0).to(self._device)

        def _forward_diffusion(
            self,
            x_0: torch.Tensor,
            t: torch.Tensor,
            epsilon: torch.Tensor,
        ) -> torch.Tensor:
            """Apply the forward diffusion process.

            Computes x_t = sqrt(alpha_bar_t) * x_0
            + sqrt(1 - alpha_bar_t) * epsilon.

            Args:
                x_0: Clean input tensor of shape ``(batch, n_features)``.
                t: Timestep indices of shape ``(batch,)``.
                epsilon: Noise tensor of the same shape as ``x_0``.

            Returns:
                Noisy tensor x_t of the same shape as ``x_0``.
            """
            alpha_bar_t = self._alpha_bar[t].unsqueeze(1)  # type: ignore[index]
            sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
            sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar_t)
            return sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * epsilon

        def _denoise(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            """Recover x_hat_0 from x_t using the predicted noise.

            Uses the closed-form inversion of the forward process:
            x_hat_0 = (x_t - sqrt(1 - alpha_bar_t) * eps_pred)
            / sqrt(alpha_bar_t)

            Args:
                x_t: Noisy tensor of shape ``(batch, n_features)``.
                t: Timestep indices of shape ``(batch,)``.

            Returns:
                Denoised reconstruction x_hat_0 of the same shape.
            """
            alpha_bar_t = self._alpha_bar[t].unsqueeze(1)  # type: ignore[index]
            sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
            sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar_t)

            epsilon_pred = self._model(x_t, t)  # type: ignore[misc]
            x_hat_0 = (x_t - sqrt_one_minus_alpha_bar * epsilon_pred) / sqrt_alpha_bar
            return x_hat_0
