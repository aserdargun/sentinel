"""TranAD anomaly detector (Tuli et al., 2022).

Transformer-based anomaly detection with self-conditioned adversarial
training.  A shared Transformer encoder feeds two decoder heads: D1 (focus
decoder) produces the primary reconstruction, and D2 (adversarial decoder,
conditioned on D1's output) competes against it.  The combined loss
``L = MSE(x, O1) - lambda_adv * MSE(x, O2)`` encourages D1 to minimise
reconstruction error while D2 maximises it, sharpening the boundary
between normal and anomalous patterns.

Anomaly score = per-window MSE(x, O1).  Trained on normal data only.
"""

from __future__ import annotations

import json
import logging
import math
import os
from typing import Any

import numpy as np

from sentinel.core.base_model import BaseAnomalyDetector
from sentinel.core.device import resolve_device
from sentinel.core.exceptions import SentinelError
from sentinel.core.registry import register_model
from sentinel.data.preprocessors import create_windows

logger = logging.getLogger(__name__)

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

    class _PositionalEncoding(nn.Module):
        """Sinusoidal positional encoding added to transformer input.

        Args:
            d_model: Embedding dimension.
            max_len: Maximum sequence length supported.
            dropout: Dropout probability applied after adding encoding.
        """

        def __init__(
            self, d_model: int, max_len: int = 5000, dropout: float = 0.1
        ) -> None:
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)

            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2, dtype=torch.float32)
                * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            # Shape: (1, max_len, d_model) for batch-first usage.
            pe = pe.unsqueeze(0)
            self.register_buffer("pe", pe)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Add positional encoding to input embeddings.

            Args:
                x: Tensor of shape ``(batch, seq_len, d_model)``.

            Returns:
                Tensor of the same shape with positional encoding added.
            """
            x = x + self.pe[:, : x.size(1)]
            return self.dropout(x)

    class _TranADModule(nn.Module):
        """TranAD network: shared encoder, two decoder heads.

        Architecture::

            input -> input_proj -> pos_enc -> Encoder -> encoded
            encoded          -> Decoder1 -> O1 -> output_proj -> recon1
            encoded + O1     -> Decoder2 -> O2 -> output_proj -> recon2

        Args:
            n_features: Number of input/output features.
            d_model: Transformer embedding dimension.
            nhead: Number of attention heads.
            num_layers: Number of transformer encoder/decoder layers.
            seq_len: Sequence length.
            dropout: Dropout probability.
        """

        def __init__(
            self,
            n_features: int,
            d_model: int,
            nhead: int,
            num_layers: int,
            seq_len: int,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.n_features = n_features
            self.d_model = d_model
            self.seq_len = seq_len

            # Project features into d_model space.
            self.input_proj = nn.Linear(n_features, d_model)
            self.pos_encoder = _PositionalEncoding(
                d_model, max_len=seq_len, dropout=dropout
            )

            # Shared transformer encoder.
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

            # Decoder 1 (focus) — standard cross-attention on encoder output.
            decoder_layer_1 = nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.decoder1 = nn.TransformerDecoder(
                decoder_layer_1, num_layers=num_layers
            )

            # Decoder 2 (adversarial) — conditioned on O1 via memory input.
            decoder_layer_2 = nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.decoder2 = nn.TransformerDecoder(
                decoder_layer_2, num_layers=num_layers
            )

            # Output projection back to feature space.
            self.output_proj = nn.Linear(d_model, n_features)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """Forward pass producing two reconstructions.

            Args:
                x: Input tensor of shape ``(batch, seq_len, n_features)``.

            Returns:
                Tuple ``(O1, O2)`` where each has shape
                ``(batch, seq_len, n_features)``.  O1 is the focus
                reconstruction, O2 is the adversarial reconstruction.
            """
            # Encode.
            src = self.input_proj(x)
            src = self.pos_encoder(src)
            encoded = self.encoder(src)

            # Decoder 1: standard reconstruction.
            tgt1 = self.pos_encoder(self.input_proj(x))
            o1_hidden = self.decoder1(tgt1, encoded)
            o1 = self.output_proj(o1_hidden)

            # Decoder 2: conditioned on O1 (concatenation of encoder
            # memory and O1 hidden states via the memory argument).
            # We use o1_hidden as the target and encoded as memory,
            # so D2 sees D1's output.
            o2_hidden = self.decoder2(o1_hidden, encoded)
            o2 = self.output_proj(o2_hidden)

            return o1, o2

    # -------------------------------------------------------------------
    # Public detector class
    # -------------------------------------------------------------------

    @register_model("tranad")
    class TranADDetector(BaseAnomalyDetector):
        """TranAD anomaly detector with self-conditioned adversarial training.

        A shared Transformer encoder feeds two competing decoder heads.
        The focus decoder (D1) minimises reconstruction error while the
        adversarial decoder (D2), conditioned on D1's output, maximises
        it.  The combined loss

        .. math::

            L = MSE(x, O_1) - \\lambda_{adv} \\cdot MSE(x, O_2)

        trains both decoders in a single phase (no alternating min-max).
        At inference, ``score = MSE(x, O_1)`` per sliding window.

        Trained on normal data only (``training_mode: normal_only``).

        Args:
            d_model: Transformer embedding dimension.
            nhead: Number of attention heads in each transformer layer.
            num_layers: Number of encoder/decoder layers.
            seq_len: Sliding-window length for sequence input.
            adversarial_weight: Weight ``lambda_adv`` on the adversarial
                decoder loss term.
            learning_rate: Adam optimiser learning rate.
            epochs: Number of training epochs.
            batch_size: Mini-batch size.
            device: Hardware target (``"auto"``, ``"cpu"``, ``"cuda"``,
                ``"mps"``).

        Example::

            detector = TranADDetector(d_model=64, nhead=4, seq_len=30)
            detector.fit(X_train)
            scores = detector.score(X_test)
        """

        def __init__(
            self,
            d_model: int = 64,
            nhead: int = 4,
            num_layers: int = 2,
            seq_len: int = 50,
            adversarial_weight: float = 1.0,
            learning_rate: float = 1e-3,
            epochs: int = 100,
            batch_size: int = 32,
            device: str = "auto",
        ) -> None:
            self.d_model = d_model
            self.nhead = nhead
            self.num_layers = num_layers
            self.seq_len = seq_len
            self.adversarial_weight = adversarial_weight
            self.learning_rate = learning_rate
            self.epochs = epochs
            self.batch_size = batch_size
            self.device_str = device

            self._device: torch.device = torch.device(resolve_device(device))
            self._model: _TranADModule | None = None
            self._n_features: int | None = None
            self._is_fitted: bool = False

        # ------------------------------------------------------------------
        # Public interface
        # ------------------------------------------------------------------

        def fit(self, X: np.ndarray, **kwargs: Any) -> None:
            """Train TranAD on sliding windows extracted from 2-D data.

            Creates windows of length ``seq_len``, then trains the shared
            encoder and both decoder heads with the self-conditioned
            adversarial loss.

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
                    f"TranAD requires at least {self.seq_len} samples, got {n_samples}"
                )

            # Seed management.
            seed: int | None = kwargs.get("seed")
            if seed is not None:
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

            self._n_features = n_features

            # Create 3-D sliding windows.
            windows = create_windows(X, self.seq_len, stride=1)
            tensor = torch.tensor(windows, dtype=torch.float32)
            dataset = TensorDataset(tensor)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            # Build model.
            self._model = _TranADModule(
                n_features=n_features,
                d_model=self.d_model,
                nhead=self.nhead,
                num_layers=self.num_layers,
                seq_len=self.seq_len,
            ).to(self._device)

            optimiser = torch.optim.Adam(
                self._model.parameters(), lr=self.learning_rate
            )

            # Training loop.
            self._model.train()
            for epoch in range(1, self.epochs + 1):
                epoch_loss = 0.0
                n_batches = 0
                for (batch,) in loader:
                    batch = batch.to(self._device)

                    o1, o2 = self._model(batch)

                    # L = MSE(x, O1) - lambda_adv * MSE(x, O2)
                    loss_focus = nn.functional.mse_loss(o1, batch)
                    loss_adv = nn.functional.mse_loss(o2, batch)
                    loss = loss_focus - self.adversarial_weight * loss_adv

                    optimiser.zero_grad()
                    loss.backward()
                    optimiser.step()

                    epoch_loss += loss.item()
                    n_batches += 1

                avg_loss = epoch_loss / max(n_batches, 1)
                if epoch % max(1, self.epochs // 10) == 0 or epoch == 1:
                    logger.info(
                        "TranAD epoch %d/%d  loss=%.6f",
                        epoch,
                        self.epochs,
                        avg_loss,
                    )

            self._is_fitted = True

        def score(self, X: np.ndarray) -> np.ndarray:
            """Compute per-sample anomaly scores via focus-decoder MSE.

            Each sliding window is passed through the model.  The anomaly
            score for each window is ``MSE(x, O1)`` averaged over
            timesteps and features.  The first ``seq_len - 1`` samples
            are padded with the score of the first complete window.

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
                    f"TranAD scoring requires at least {self.seq_len} "
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
            """Save model state_dict and config to disk.

            Writes ``config.json`` (hyperparameters and metadata) and
            ``model.pt`` (PyTorch state dict) into *path*.  All writes
            are atomic (write to temp file, then rename).

            Args:
                path: Directory in which to store artifacts.

            Raises:
                SentinelError: If the model has not been fitted.
            """
            self._check_fitted()
            os.makedirs(path, exist_ok=True)

            # -- config.json ---------------------------------------------------
            config = {
                "model_name": "tranad",
                "d_model": self.d_model,
                "nhead": self.nhead,
                "num_layers": self.num_layers,
                "seq_len": self.seq_len,
                "adversarial_weight": self.adversarial_weight,
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
            """Load a previously saved TranAD model from disk.

            Reconstructs the network architecture from ``config.json``
            and loads weights from ``model.pt``.

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

            self.d_model = int(config["d_model"])
            self.nhead = int(config["nhead"])
            self.num_layers = int(config["num_layers"])
            self.seq_len = int(config["seq_len"])
            self.adversarial_weight = float(config["adversarial_weight"])
            self.learning_rate = float(config["learning_rate"])
            self.epochs = int(config["epochs"])
            self.batch_size = int(config["batch_size"])
            self.device_str = config.get("device", "auto")
            self._n_features = int(config["n_features"])

            self._device = torch.device(resolve_device(self.device_str))

            self._model = _TranADModule(
                n_features=self._n_features,
                d_model=self.d_model,
                nhead=self.nhead,
                num_layers=self.num_layers,
                seq_len=self.seq_len,
            ).to(self._device)

            state_dict = torch.load(
                model_path, map_location=self._device, weights_only=True
            )
            self._model.load_state_dict(state_dict)
            self._model.eval()
            self._is_fitted = True

        def get_params(self) -> dict[str, Any]:
            """Return model hyperparameters and metadata.

            Returns:
                Dict with all constructor arguments plus ``n_features``
                (``None`` if not yet fitted).
            """
            return {
                "model_name": "tranad",
                "d_model": self.d_model,
                "nhead": self.nhead,
                "num_layers": self.num_layers,
                "seq_len": self.seq_len,
                "adversarial_weight": self.adversarial_weight,
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
                    "TranADDetector has not been fitted. Call fit() first."
                )

        def _score_windows(self, windows: np.ndarray) -> np.ndarray:
            """Compute focus-decoder MSE per window.

            Processes windows in batches to avoid OOM on large datasets.

            Args:
                windows: 3-D array ``(n_windows, seq_len, n_features)``.

            Returns:
                1-D array of per-window MSE scores.
            """
            assert self._model is not None

            self._model.eval()
            all_scores: list[np.ndarray] = []

            with torch.no_grad():
                for start in range(0, len(windows), self.batch_size):
                    batch_np = windows[start : start + self.batch_size]
                    batch = torch.tensor(
                        batch_np, dtype=torch.float32, device=self._device
                    )
                    o1, _o2 = self._model(batch)
                    # Per-window MSE: mean over (seq_len, n_features).
                    mse = (batch - o1).pow(2).mean(dim=(1, 2)).cpu().numpy()
                    all_scores.append(mse)

            return np.concatenate(all_scores).astype(np.float64)
