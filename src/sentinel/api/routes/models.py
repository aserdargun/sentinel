"""Model registry routes: list all registered models."""

from __future__ import annotations

import structlog
from fastapi import APIRouter

from sentinel.api.deps import get_registry
from sentinel.api.schemas import ModelInfo, ModelListResponse
from sentinel.core.types import ModelCategory

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/models", tags=["models"])

# Map model names to categories for display purposes.
_STATISTICAL_MODELS = {"zscore", "isolation_forest", "matrix_profile"}
_ENSEMBLE_MODELS = {"hybrid_ensemble"}

# Brief descriptions for known models.
_DESCRIPTIONS: dict[str, str] = {
    "zscore": "Rolling Z-Score detector with configurable window and sigma threshold.",
    "isolation_forest": (
        "Sklearn IsolationForest wrapper for tree-based anomaly detection."
    ),
    "matrix_profile": (
        "STOMP-based matrix profile via stumpy for subsequence distance scoring."
    ),
    "autoencoder": "Vanilla feedforward autoencoder with MSE reconstruction loss.",
    "rnn": "Elman RNN sequence-to-sequence reconstruction model.",
    "lstm": "LSTM-based next-step forecasting with prediction error scoring.",
    "gru": "GRU-based sequence prediction, lighter alternative to LSTM.",
    "lstm_ae": "LSTM Autoencoder with LSTM encoder-decoder architecture.",
    "tcn": "Temporal Convolutional Network with dilated causal convolutions.",
    "vae": "Variational Autoencoder with ELBO loss and reparameterization.",
    "gan": "GANomaly-style encoder-decoder-encoder anomaly detection.",
    "tadgan": "TadGAN with encoder-generator-critic and cycle consistency.",
    "tranad": "TranAD transformer with self-conditioned adversarial training.",
    "deepar": "DeepAR autoregressive RNN with Gaussian likelihood scoring.",
    "diffusion": "DDPM diffusion-based anomaly detection via denoising.",
    "hybrid_ensemble": "Weighted combination of statistical and deep model scores.",
}


def _get_category(name: str) -> str:
    """Determine the model category from its name.

    Args:
        name: Registered model name.

    Returns:
        Category string: 'statistical', 'deep', or 'ensemble'.
    """
    if name in _STATISTICAL_MODELS:
        return ModelCategory.STATISTICAL.value
    if name in _ENSEMBLE_MODELS:
        return ModelCategory.ENSEMBLE.value
    return ModelCategory.DEEP.value


@router.get(
    "",
    response_model=ModelListResponse,
)
async def list_registered_models() -> ModelListResponse:
    """List all registered anomaly detection models.

    Returns model names, categories, and brief descriptions from the
    model registry.

    Returns:
        Response containing a list of registered models.
    """
    registry = get_registry()

    models: list[ModelInfo] = []
    for name in sorted(registry.keys()):
        models.append(
            ModelInfo(
                name=name,
                category=_get_category(name),
                description=_DESCRIPTIONS.get(name, ""),
            )
        )

    logger.info("models.listed", count=len(models))
    return ModelListResponse(models=models)
