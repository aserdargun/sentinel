"""TadGAN anomaly detector (Geiger et al., 2020).

Encoder-generator-critic architecture with cycle-consistency loss for
time series anomaly detection.  An LSTM-based encoder maps input sequences
to a latent space, and an LSTM-based generator reconstructs the sequence
from the latent code.  A Wasserstein critic (MLP) distinguishes real from
generated sequences and provides a complementary anomaly signal.

Training alternates between critic and generator updates.  The critic
is trained with gradient penalty (WGAN-GP) for Lipschitz constraint
enforcement.  The generator is trained with a reconstruction loss plus
a cycle-consistency loss that ensures the encoder produces similar latent
codes for original and reconstructed sequences.

Anomaly score = alpha * (1 - critic_score_normalized)
    + (1 - alpha) * reconstruction_error

Trained on normal data only (``training_mode: normal_only``).
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

    class _Encoder(nn.Module):
        """LSTM encoder that maps a sequence to a latent vector.

        Args:
            n_features: Number of input features per timestep.
            hidden_dim: LSTM hidden state size.
            latent_dim: Dimensionality of the output latent vector.
            num_layers: Number of stacked LSTM layers.
        """

        def __init__(
            self,
            n_features: int,
            hidden_dim: int,
            latent_dim: int,
            num_layers: int = 1,
        ) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=n_features,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
            )
            self.fc_mu = nn.Linear(hidden_dim, latent_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Encode a sequence into a latent vector.

            Args:
                x: Input tensor of shape ``(batch, seq_len, n_features)``.

            Returns:
                Latent tensor of shape ``(batch, latent_dim)``.
            """
            # Take the last hidden state from the LSTM.
            _, (h_n, _) = self.lstm(x)
            # h_n shape: (num_layers, batch, hidden_dim) -- use last layer.
            h_last = h_n[-1]
            return self.fc_mu(h_last)

    class _Generator(nn.Module):
        """LSTM generator that reconstructs a sequence from a latent vector.

        The latent vector is repeated across the sequence length and fed
        through an LSTM decoder followed by a linear projection back to
        feature space.

        Args:
            n_features: Number of output features per timestep.
            hidden_dim: LSTM hidden state size.
            latent_dim: Dimensionality of the input latent vector.
            seq_len: Output sequence length.
            num_layers: Number of stacked LSTM layers.
        """

        def __init__(
            self,
            n_features: int,
            hidden_dim: int,
            latent_dim: int,
            seq_len: int,
            num_layers: int = 1,
        ) -> None:
            super().__init__()
            self.seq_len = seq_len
            self.fc = nn.Linear(latent_dim, hidden_dim)
            self.lstm = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
            )
            self.output_proj = nn.Linear(hidden_dim, n_features)

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            """Generate a sequence from a latent vector.

            Args:
                z: Latent tensor of shape ``(batch, latent_dim)``.

            Returns:
                Reconstructed tensor of shape ``(batch, seq_len, n_features)``.
            """
            h = torch.relu(self.fc(z))
            # Repeat latent across sequence length.
            h = h.unsqueeze(1).repeat(1, self.seq_len, 1)
            out, _ = self.lstm(h)
            return self.output_proj(out)

    class _Critic(nn.Module):
        """Wasserstein critic (MLP) that scores real vs generated sequences.

        Takes a flattened sequence as input and outputs a scalar
        (no sigmoid -- Wasserstein distance).

        Args:
            n_features: Number of features per timestep.
            seq_len: Sequence length.
            hidden_dim: MLP hidden layer size.
        """

        def __init__(
            self,
            n_features: int,
            seq_len: int,
            hidden_dim: int,
        ) -> None:
            super().__init__()
            input_dim = n_features * seq_len
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Score a sequence (higher = more real).

            Args:
                x: Sequence tensor of shape ``(batch, seq_len, n_features)``.

            Returns:
                Scalar scores of shape ``(batch, 1)``.
            """
            flat = x.reshape(x.size(0), -1)
            return self.net(flat)

    # -------------------------------------------------------------------
    # Public detector class
    # -------------------------------------------------------------------

    @register_model("tadgan")
    class TadGANDetector(BaseAnomalyDetector):
        """TadGAN anomaly detector with cycle-consistency and Wasserstein critic.

        An LSTM encoder maps sequences to a latent space, an LSTM generator
        reconstructs from the latent code, and an MLP critic provides a
        Wasserstein distance signal.  The generator is trained with
        reconstruction loss plus cycle-consistency on the latent space.
        The critic is trained with gradient penalty (WGAN-GP).

        The anomaly score combines normalised critic output and reconstruction
        error:

        .. math::

            score = \\alpha (1 - C_{norm}) + (1 - \\alpha) MSE(x, \\hat{x})

        Trained on normal data only (``training_mode: normal_only``).

        Args:
            hidden_dim: LSTM hidden dimension for encoder and generator.
            latent_dim: Dimensionality of the latent space.
            seq_len: Sliding-window length for sequence input.
            cycle_weight: Weight for the cycle-consistency loss term.
            critic_iterations: Number of critic updates per generator update.
            gp_weight: Gradient penalty coefficient (lambda_gp).
            alpha: Blending weight between critic score and reconstruction
                error.  ``alpha=1`` uses only critic, ``alpha=0`` uses only
                reconstruction.
            learning_rate: Adam optimiser learning rate.
            epochs: Number of training epochs.
            batch_size: Mini-batch size.
            device: Hardware target (``"auto"``, ``"cpu"``, ``"cuda"``,
                ``"mps"``).

        Example::

            detector = TadGANDetector(hidden_dim=64, latent_dim=16, seq_len=50)
            detector.fit(X_train)
            scores = detector.score(X_test)
        """

        def __init__(
            self,
            hidden_dim: int = 64,
            latent_dim: int = 16,
            seq_len: int = 50,
            cycle_weight: float = 10.0,
            critic_iterations: int = 5,
            gp_weight: float = 10.0,
            alpha: float = 0.5,
            learning_rate: float = 0.001,
            epochs: int = 100,
            batch_size: int = 32,
            device: str = "auto",
        ) -> None:
            self.hidden_dim = hidden_dim
            self.latent_dim = latent_dim
            self.seq_len = seq_len
            self.cycle_weight = cycle_weight
            self.critic_iterations = critic_iterations
            self.gp_weight = gp_weight
            self.alpha = alpha
            self.learning_rate = learning_rate
            self.epochs = epochs
            self.batch_size = batch_size
            self.device_str = device

            self._device: torch.device | None = None
            self._encoder: _Encoder | None = None
            self._generator: _Generator | None = None
            self._critic: _Critic | None = None
            self._n_features: int | None = None
            self._is_fitted: bool = False

        # ------------------------------------------------------------------
        # Public interface
        # ------------------------------------------------------------------

        def fit(self, X: np.ndarray, **kwargs: Any) -> None:
            """Train TadGAN on sliding windows extracted from 2-D data.

            Creates windows of length ``seq_len``, then trains the encoder,
            generator, and critic with alternating updates.  The critic is
            updated ``critic_iterations`` times per generator update, with
            gradient penalty enforced for the Lipschitz constraint.

            Args:
                X: Training data of shape ``(n_samples, n_features)``.
                **kwargs: Accepts ``seed`` (``int | None``) to set torch
                    manual seeds before training.

            Raises:
                SentinelError: If *X* has fewer rows than ``seq_len``.
            """
            if X.ndim == 1:
                X = X.reshape(-1, 1)

            n_samples, n_features = X.shape
            if n_samples < self.seq_len:
                raise SentinelError(
                    f"TadGAN requires at least {self.seq_len} samples, got {n_samples}"
                )

            # Seed management.
            seed: int | None = kwargs.get("seed")
            if seed is not None:
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

            self._n_features = n_features
            self._device = torch.device(resolve_device(self.device_str))

            # Create 3-D sliding windows.
            windows = create_windows(X, self.seq_len, stride=1)
            tensor = torch.tensor(windows, dtype=torch.float32)
            dataset = TensorDataset(tensor)
            loader = DataLoader(
                dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
            )

            # Build networks.
            self._encoder = _Encoder(
                n_features=n_features,
                hidden_dim=self.hidden_dim,
                latent_dim=self.latent_dim,
            ).to(self._device)

            self._generator = _Generator(
                n_features=n_features,
                hidden_dim=self.hidden_dim,
                latent_dim=self.latent_dim,
                seq_len=self.seq_len,
            ).to(self._device)

            self._critic = _Critic(
                n_features=n_features,
                seq_len=self.seq_len,
                hidden_dim=self.hidden_dim,
            ).to(self._device)

            # Separate optimisers for generator path (E+G) and critic.
            opt_eg = torch.optim.Adam(
                list(self._encoder.parameters()) + list(self._generator.parameters()),
                lr=self.learning_rate,
                betas=(0.5, 0.999),
            )
            opt_c = torch.optim.Adam(
                self._critic.parameters(),
                lr=self.learning_rate,
                betas=(0.5, 0.999),
            )

            # Training loop with alternating updates.
            self._encoder.train()
            self._generator.train()
            self._critic.train()

            for epoch in range(1, self.epochs + 1):
                epoch_loss_c = 0.0
                epoch_loss_g = 0.0
                n_batches = 0

                for (batch,) in loader:
                    batch = batch.to(self._device)

                    # ----- Critic updates -----
                    for _ in range(self.critic_iterations):
                        opt_c.zero_grad()

                        # Encode and reconstruct (detach from generator).
                        with torch.no_grad():
                            z = self._encoder(batch)
                            x_hat = self._generator(z)

                        # Wasserstein loss: E[C(real)] - E[C(fake)].
                        c_real = self._critic(batch)
                        c_fake = self._critic(x_hat)
                        loss_c = c_fake.mean() - c_real.mean()

                        # Gradient penalty.
                        gp = self._gradient_penalty(batch, x_hat)
                        loss_c = loss_c + self.gp_weight * gp

                        loss_c.backward()
                        opt_c.step()

                    # ----- Generator + Encoder update -----
                    opt_eg.zero_grad()

                    z = self._encoder(batch)
                    x_hat = self._generator(z)

                    # Reconstruction loss.
                    loss_recon = nn.functional.mse_loss(x_hat, batch)

                    # Cycle-consistency loss: E(x) ~= E(G(E(x))).
                    z_cycle = self._encoder(x_hat)
                    loss_cycle = nn.functional.mse_loss(z, z_cycle)

                    # Generator adversarial loss: maximise critic on fakes
                    # (i.e., minimise -E[C(G(E(x)))]).
                    c_fake_for_g = self._critic(x_hat)
                    loss_g_adv = -c_fake_for_g.mean()

                    loss_g = loss_recon + self.cycle_weight * loss_cycle + loss_g_adv
                    loss_g.backward()
                    opt_eg.step()

                    epoch_loss_c += loss_c.item()
                    epoch_loss_g += loss_g.item()
                    n_batches += 1

                avg_c = epoch_loss_c / max(n_batches, 1)
                avg_g = epoch_loss_g / max(n_batches, 1)
                if epoch % max(1, self.epochs // 10) == 0 or epoch == 1:
                    logger.info(
                        "TadGAN epoch %d/%d  loss_C=%.6f  loss_G=%.6f",
                        epoch,
                        self.epochs,
                        avg_c,
                        avg_g,
                    )

            self._is_fitted = True

        def score(self, X: np.ndarray) -> np.ndarray:
            """Compute per-sample anomaly scores via critic + reconstruction.

            Each sliding window is scored by both the critic and the
            reconstruction error.  The two signals are combined as:

            ``score = alpha * (1 - C_norm) + (1 - alpha) * recon_error``

            where ``C_norm`` is the critic output min-max normalised to [0, 1].
            The first ``seq_len - 1`` samples are padded with the score of the
            first complete window.

            Args:
                X: Data of shape ``(n_samples, n_features)``.

            Returns:
                1-D array of length ``n_samples`` with anomaly scores.

            Raises:
                SentinelError: If the model has not been fitted or data
                    is too short.
            """
            self._check_fitted()

            if X.ndim == 1:
                X = X.reshape(-1, 1)

            n_samples = X.shape[0]
            if n_samples < self.seq_len:
                raise SentinelError(
                    f"TadGAN scoring requires at least {self.seq_len} "
                    f"samples, got {n_samples}"
                )

            windows = create_windows(X, self.seq_len, stride=1)
            window_scores = self._score_windows(windows)

            # Pad to original length: first (seq_len - 1) samples get the
            # score of the first complete window.
            pad_len = self.seq_len - 1
            scores = np.concatenate([np.full(pad_len, window_scores[0]), window_scores])
            return scores

        def save(self, path: str) -> None:
            """Save encoder, generator, critic state_dicts and config to disk.

            Three model files are written into *path*:

            * ``config.json`` -- Hyperparameters and metadata.
            * ``encoder.pt`` -- Encoder state dict.
            * ``generator.pt`` -- Generator state dict.
            * ``critic.pt`` -- Critic state dict.

            All writes are atomic (write to temp file, then rename).

            Args:
                path: Directory in which to store artifacts.

            Raises:
                SentinelError: If the model has not been fitted.
            """
            self._check_fitted()
            os.makedirs(path, exist_ok=True)

            # -- config.json ---------------------------------------------------
            config = {
                "model_name": "tadgan",
                "hidden_dim": self.hidden_dim,
                "latent_dim": self.latent_dim,
                "seq_len": self.seq_len,
                "cycle_weight": self.cycle_weight,
                "critic_iterations": self.critic_iterations,
                "gp_weight": self.gp_weight,
                "alpha": self.alpha,
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

            # -- Model state dicts (one per network) ---------------------------
            assert self._encoder is not None
            assert self._generator is not None
            assert self._critic is not None

            for name, module in [
                ("encoder", self._encoder),
                ("generator", self._generator),
                ("critic", self._critic),
            ]:
                model_path = os.path.join(path, f"{name}.pt")
                tmp_model = model_path + ".tmp"
                torch.save(module.state_dict(), tmp_model)
                os.rename(tmp_model, model_path)

        def load(self, path: str) -> None:
            """Load a previously saved TadGAN model from disk.

            Reconstructs the network architectures from ``config.json``
            and loads weights from ``encoder.pt``, ``generator.pt``, and
            ``critic.pt``.

            Args:
                path: Directory containing config and model files.

            Raises:
                SentinelError: If required files are missing.
            """
            config_path = os.path.join(path, "config.json")
            if not os.path.isfile(config_path):
                raise SentinelError(f"Config not found: {config_path}")

            required_files = ["encoder.pt", "generator.pt", "critic.pt"]
            for fname in required_files:
                fpath = os.path.join(path, fname)
                if not os.path.isfile(fpath):
                    raise SentinelError(f"Model file not found: {fpath}")

            with open(config_path) as f:
                config = json.load(f)

            self.hidden_dim = int(config["hidden_dim"])
            self.latent_dim = int(config["latent_dim"])
            self.seq_len = int(config["seq_len"])
            self.cycle_weight = float(config["cycle_weight"])
            self.critic_iterations = int(config["critic_iterations"])
            self.gp_weight = float(config["gp_weight"])
            self.alpha = float(config["alpha"])
            self.learning_rate = float(config["learning_rate"])
            self.epochs = int(config["epochs"])
            self.batch_size = int(config["batch_size"])
            self.device_str = config.get("device", "auto")
            self._n_features = int(config["n_features"])

            self._device = torch.device(resolve_device(self.device_str))

            # Reconstruct networks.
            self._encoder = _Encoder(
                n_features=self._n_features,
                hidden_dim=self.hidden_dim,
                latent_dim=self.latent_dim,
            ).to(self._device)

            self._generator = _Generator(
                n_features=self._n_features,
                hidden_dim=self.hidden_dim,
                latent_dim=self.latent_dim,
                seq_len=self.seq_len,
            ).to(self._device)

            self._critic = _Critic(
                n_features=self._n_features,
                seq_len=self.seq_len,
                hidden_dim=self.hidden_dim,
            ).to(self._device)

            # Load state dicts.
            for name, module in [
                ("encoder", self._encoder),
                ("generator", self._generator),
                ("critic", self._critic),
            ]:
                state_dict = torch.load(
                    os.path.join(path, f"{name}.pt"),
                    map_location=self._device,
                    weights_only=True,
                )
                module.load_state_dict(state_dict)

            self._encoder.eval()
            self._generator.eval()
            self._critic.eval()
            self._is_fitted = True

        def get_params(self) -> dict[str, Any]:
            """Return model hyperparameters and metadata.

            Returns:
                Dict with all constructor arguments plus ``n_features``
                (``None`` if not yet fitted).
            """
            return {
                "model_name": "tadgan",
                "hidden_dim": self.hidden_dim,
                "latent_dim": self.latent_dim,
                "seq_len": self.seq_len,
                "cycle_weight": self.cycle_weight,
                "critic_iterations": self.critic_iterations,
                "gp_weight": self.gp_weight,
                "alpha": self.alpha,
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
                    "TadGANDetector has not been fitted. Call fit() first."
                )

        def _gradient_penalty(
            self,
            real: torch.Tensor,
            fake: torch.Tensor,
        ) -> torch.Tensor:
            """Compute WGAN-GP gradient penalty.

            Interpolates between real and fake sequences and enforces unit
            gradient norm on the critic output with respect to the
            interpolated input.

            Args:
                real: Real sequence batch ``(batch, seq_len, n_features)``.
                fake: Generated sequence batch of the same shape.

            Returns:
                Scalar gradient penalty loss.
            """
            assert self._critic is not None
            batch_size = real.size(0)

            # Random interpolation coefficient per sample.
            eps = torch.rand(batch_size, 1, 1, device=self._device)
            interpolated = (eps * real + (1 - eps) * fake).requires_grad_(True)

            c_interp = self._critic(interpolated)

            gradients = torch.autograd.grad(
                outputs=c_interp,
                inputs=interpolated,
                grad_outputs=torch.ones_like(c_interp),
                create_graph=True,
                retain_graph=True,
            )[0]

            # Flatten gradients per sample and compute L2 norm.
            gradients = gradients.reshape(batch_size, -1)
            grad_norm = gradients.norm(2, dim=1)
            penalty = ((grad_norm - 1.0) ** 2).mean()
            return penalty

        def _score_windows(self, windows: np.ndarray) -> np.ndarray:
            """Compute combined critic + reconstruction score per window.

            Processes windows in batches.  The critic scores are min-max
            normalised across the full set, then combined with
            reconstruction error using the ``alpha`` blending weight.

            Args:
                windows: 3-D array ``(n_windows, seq_len, n_features)``.

            Returns:
                1-D array of per-window anomaly scores.
            """
            assert self._encoder is not None
            assert self._generator is not None
            assert self._critic is not None

            self._encoder.eval()
            self._generator.eval()
            self._critic.eval()

            all_recon_errors: list[np.ndarray] = []
            all_critic_scores: list[np.ndarray] = []

            with torch.no_grad():
                for start in range(0, len(windows), self.batch_size):
                    batch_np = windows[start : start + self.batch_size]
                    batch = torch.tensor(
                        batch_np, dtype=torch.float32, device=self._device
                    )

                    # Reconstruction error.
                    z = self._encoder(batch)
                    x_hat = self._generator(z)
                    recon_err = (batch - x_hat).pow(2).mean(dim=(1, 2)).cpu().numpy()
                    all_recon_errors.append(recon_err)

                    # Critic score (higher = more real).
                    c_score = self._critic(batch).squeeze(-1).cpu().numpy()
                    all_critic_scores.append(c_score)

            recon_errors = np.concatenate(all_recon_errors).astype(np.float64)
            critic_scores = np.concatenate(all_critic_scores).astype(np.float64)

            # Min-max normalise critic scores to [0, 1].
            c_min = critic_scores.min()
            c_max = critic_scores.max()
            if c_max - c_min > 1e-12:
                critic_norm = (critic_scores - c_min) / (c_max - c_min)
            else:
                # Constant critic output -- treat all as equally normal.
                critic_norm = np.ones_like(critic_scores) * 0.5

            # Combined score: alpha * (1 - critic_norm) + (1-alpha) * recon.
            # Higher critic_norm = more real = less anomalous, so invert.
            combined = (
                self.alpha * (1.0 - critic_norm) + (1.0 - self.alpha) * recon_errors
            )
            return combined
