"""LSTM Autoencoder anomaly detector.

Encodes multivariate time-series windows with an LSTM encoder into a fixed-
length latent vector, then reconstructs the original sequence with an LSTM
decoder. Anomaly scores are per-window mean squared reconstruction error,
padded to match the original input length.
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
        """LSTM encoder that compresses a sequence to a latent vector.

        Args:
            n_features: Number of input features per timestep.
            hidden_dim: LSTM hidden state size.
            latent_dim: Dimensionality of the output latent vector.
        """

        def __init__(
            self,
            n_features: int,
            hidden_dim: int,
            latent_dim: int,
        ) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=n_features,
                hidden_size=hidden_dim,
                batch_first=True,
            )
            self.fc = nn.Linear(hidden_dim, latent_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Encode a batch of sequences.

            Args:
                x: Tensor of shape ``(batch, seq_len, n_features)``.

            Returns:
                Latent tensor of shape ``(batch, latent_dim)``.
            """
            _, (h_n, _) = self.lstm(x)
            # h_n shape: (1, batch, hidden_dim) -> squeeze to (batch, hidden_dim)
            latent = self.fc(h_n.squeeze(0))
            return latent

    class _Decoder(nn.Module):
        """LSTM decoder that reconstructs a sequence from a latent vector.

        Args:
            latent_dim: Dimensionality of the input latent vector.
            hidden_dim: LSTM hidden state size.
            n_features: Number of output features per timestep.
            seq_len: Length of the sequence to reconstruct.
        """

        def __init__(
            self,
            latent_dim: int,
            hidden_dim: int,
            n_features: int,
            seq_len: int,
        ) -> None:
            super().__init__()
            self.seq_len = seq_len
            self.fc = nn.Linear(latent_dim, hidden_dim)
            self.lstm = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=n_features,
                batch_first=True,
            )

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            """Decode a batch of latent vectors into sequences.

            Args:
                z: Latent tensor of shape ``(batch, latent_dim)``.

            Returns:
                Reconstructed tensor of shape ``(batch, seq_len, n_features)``.
            """
            # Project latent to hidden dim and repeat across timesteps.
            h = self.fc(z)  # (batch, hidden_dim)
            h_repeated = h.unsqueeze(1).repeat(1, self.seq_len, 1)
            # Decode with LSTM.
            output, _ = self.lstm(h_repeated)
            return output

    class _LSTMAutoencoder(nn.Module):
        """Full LSTM Autoencoder (encoder + decoder).

        Args:
            n_features: Number of input/output features per timestep.
            encoder_dim: LSTM hidden size for the encoder.
            decoder_dim: LSTM hidden size for the decoder.
            latent_dim: Size of the bottleneck latent vector.
            seq_len: Sequence length to reconstruct.
        """

        def __init__(
            self,
            n_features: int,
            encoder_dim: int,
            decoder_dim: int,
            latent_dim: int,
            seq_len: int,
        ) -> None:
            super().__init__()
            self.encoder = _Encoder(n_features, encoder_dim, latent_dim)
            self.decoder = _Decoder(latent_dim, decoder_dim, n_features, seq_len)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Encode then decode the input sequence.

            Args:
                x: Input tensor of shape ``(batch, seq_len, n_features)``.

            Returns:
                Reconstructed tensor of same shape as *x*.
            """
            z = self.encoder(x)
            return self.decoder(z)

    # -------------------------------------------------------------------
    # Detector class (only registered when torch is available)
    # -------------------------------------------------------------------

    @register_model("lstm_ae")
    class LSTMAEDetector(BaseAnomalyDetector):
        """LSTM Autoencoder anomaly detector.

        An LSTM encoder compresses each sliding window of multivariate time
        series data into a low-dimensional latent vector.  An LSTM decoder then
        reconstructs the original window.  The anomaly score for each window is
        the mean squared reconstruction error averaged over timesteps and
        features.  Scores are padded to match the original input length by
        assigning the first ``seq_len - 1`` samples the score of the first
        complete window.

        Trained on normal data only (``training_mode: normal_only``).

        Args:
            encoder_dim: Hidden dimension of the encoder LSTM.
            decoder_dim: Hidden dimension of the decoder LSTM.
            latent_dim: Size of the bottleneck latent vector.
            seq_len: Length of sliding-window sequences.
            learning_rate: Adam optimiser learning rate.
            epochs: Number of training epochs.
            batch_size: Mini-batch size for training.
            device: Hardware device (``"auto"``, ``"cpu"``, ``"cuda"``,
                ``"mps"``).

        Example::

            detector = LSTMAEDetector(encoder_dim=64, latent_dim=16)
            detector.fit(X_train)
            scores = detector.score(X_test)
        """

        def __init__(
            self,
            encoder_dim: int = 64,
            decoder_dim: int = 64,
            latent_dim: int = 16,
            seq_len: int = 50,
            learning_rate: float = 1e-3,
            epochs: int = 100,
            batch_size: int = 32,
            device: str = "auto",
        ) -> None:
            self.encoder_dim = encoder_dim
            self.decoder_dim = decoder_dim
            self.latent_dim = latent_dim
            self.seq_len = seq_len
            self.learning_rate = learning_rate
            self.epochs = epochs
            self.batch_size = batch_size
            self.device_str = device

            # Resolved at fit-time so that resolve_device() sees current
            # hardware state.
            self._device: torch.device | None = None
            self._model: _LSTMAutoencoder | None = None
            self._n_features: int | None = None
            self._is_fitted: bool = False

        # ----------------------------------------------------------------
        # Public interface
        # ----------------------------------------------------------------

        def fit(self, X: np.ndarray, **kwargs: Any) -> None:
            """Train the LSTM Autoencoder on 2-D data.

            Creates sliding windows of length ``seq_len`` from *X*, then
            trains the encoder-decoder with MSE reconstruction loss using
            Adam.

            Args:
                X: Training data of shape ``(n_samples, n_features)``.
                **kwargs: Ignored; present for interface compatibility.

            Raises:
                SentinelError: If data is too short to form at least one
                    window.
            """
            if X.ndim == 1:
                X = X.reshape(-1, 1)

            n_samples, n_features = X.shape
            if n_samples < self.seq_len:
                raise SentinelError(
                    f"LSTM-AE requires at least {self.seq_len} samples, got {n_samples}"
                )

            self._n_features = n_features

            self._device = torch.device(resolve_device(self.device_str))

            # Create 3-D sliding windows.
            windows = create_windows(X, self.seq_len, stride=1)
            tensor = torch.tensor(windows, dtype=torch.float32)
            dataset = TensorDataset(tensor)
            loader = DataLoader(
                dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
            )

            # Build model and move to device.
            self._model = _LSTMAutoencoder(
                n_features=n_features,
                encoder_dim=self.encoder_dim,
                decoder_dim=self.decoder_dim,
                latent_dim=self.latent_dim,
                seq_len=self.seq_len,
            )
            self._model.to(self._device)

            optimiser = torch.optim.Adam(
                self._model.parameters(), lr=self.learning_rate
            )
            criterion = nn.MSELoss()

            # Training loop.
            self._model.train()
            for epoch in range(1, self.epochs + 1):
                epoch_loss = 0.0
                n_batches = 0
                for (batch,) in loader:
                    batch = batch.to(self._device)
                    reconstructed = self._model(batch)
                    loss = criterion(reconstructed, batch)

                    optimiser.zero_grad()
                    loss.backward()
                    optimiser.step()

                    epoch_loss += loss.item()
                    n_batches += 1

                avg_loss = epoch_loss / max(n_batches, 1)
                if epoch % max(1, self.epochs // 10) == 0 or epoch == 1:
                    logger.info(
                        "LSTM-AE epoch %d/%d  loss=%.6f",
                        epoch,
                        self.epochs,
                        avg_loss,
                    )

            self._is_fitted = True

        def score(self, X: np.ndarray) -> np.ndarray:
            """Compute per-sample anomaly scores via reconstruction error.

            Each sliding window is reconstructed and scored as the mean
            squared error averaged over timesteps and features.  The first
            ``seq_len - 1`` samples (which have no complete window starting
            at their position) are assigned the score of the first complete
            window.

            Args:
                X: Data of shape ``(n_samples, n_features)``.

            Returns:
                1-D array of length ``n_samples`` with anomaly scores.

            Raises:
                SentinelError: If the model has not been fitted or data is
                    too short.
            """
            self._check_fitted()

            if X.ndim == 1:
                X = X.reshape(-1, 1)

            n_samples = X.shape[0]
            if n_samples < self.seq_len:
                raise SentinelError(
                    f"LSTM-AE scoring requires at least {self.seq_len} "
                    f"samples, got {n_samples}"
                )

            windows = create_windows(X, self.seq_len, stride=1)
            tensor = torch.tensor(windows, dtype=torch.float32)
            tensor = tensor.to(self._device)

            assert self._model is not None
            self._model.eval()
            with torch.no_grad():
                reconstructed = self._model(tensor)

            # Per-window MSE: mean over timesteps and features.
            mse = (tensor - reconstructed).pow(2).mean(dim=(1, 2)).cpu().numpy()

            # Pad to original length: prepend first window score.
            pad_len = self.seq_len - 1
            scores = np.concatenate([np.full(pad_len, mse[0]), mse])
            return scores

        def save(self, path: str) -> None:
            """Save model state_dict and config to disk.

            Writes ``config.json`` (hyperparameters) and ``model.pt``
            (PyTorch state dict) into *path*.  All writes are atomic
            (write to temp file, then rename).

            Args:
                path: Directory to store model artifacts.

            Raises:
                SentinelError: If the model has not been fitted.
            """
            self._check_fitted()
            os.makedirs(path, exist_ok=True)

            # -- config.json ---------------------------------------------------
            config = {
                "model_name": "lstm_ae",
                "encoder_dim": self.encoder_dim,
                "decoder_dim": self.decoder_dim,
                "latent_dim": self.latent_dim,
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

            # -- model.pt (state dict) -----------------------------------------
            assert self._model is not None
            model_path = os.path.join(path, "model.pt")
            tmp_model = model_path + ".tmp"
            torch.save(self._model.state_dict(), tmp_model)
            os.rename(tmp_model, model_path)

        def load(self, path: str) -> None:
            """Load a previously saved LSTM-AE model from disk.

            Reconstructs the network architecture from ``config.json`` and
            loads weights from ``model.pt``.

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
            self.seq_len = int(config["seq_len"])
            self.learning_rate = float(config["learning_rate"])
            self.epochs = int(config["epochs"])
            self.batch_size = int(config["batch_size"])
            self.device_str = str(config["device"])
            self._n_features = int(config["n_features"])

            self._device = torch.device(resolve_device(self.device_str))

            # Rebuild architecture and load weights.
            self._model = _LSTMAutoencoder(
                n_features=self._n_features,
                encoder_dim=self.encoder_dim,
                decoder_dim=self.decoder_dim,
                latent_dim=self.latent_dim,
                seq_len=self.seq_len,
            )
            self._model.load_state_dict(
                torch.load(model_path, map_location=self._device, weights_only=True)
            )
            self._model.to(self._device)
            self._model.eval()
            self._is_fitted = True

        def get_params(self) -> dict[str, Any]:
            """Return model parameters as a dictionary.

            Returns:
                Dict containing all hyperparameters and the inferred
                ``n_features`` (``None`` if not yet fitted).
            """
            return {
                "model_name": "lstm_ae",
                "encoder_dim": self.encoder_dim,
                "decoder_dim": self.decoder_dim,
                "latent_dim": self.latent_dim,
                "seq_len": self.seq_len,
                "learning_rate": self.learning_rate,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "device": self.device_str,
                "n_features": self._n_features,
            }

        # ----------------------------------------------------------------
        # Private helpers
        # ----------------------------------------------------------------

        def _check_fitted(self) -> None:
            """Raise if the model has not been fitted."""
            if not self._is_fitted:
                raise SentinelError(
                    "LSTMAEDetector has not been fitted. Call fit() first."
                )
