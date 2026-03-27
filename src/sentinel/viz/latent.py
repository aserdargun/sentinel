"""Latent space projection visualization.

Projects high-dimensional latent embeddings into 2-D using t-SNE or UMAP,
colored by anomaly score.  Useful for inspecting cluster separation in
VAE, Autoencoder, or any model that produces embeddings.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from sentinel.core.exceptions import SentinelError  # noqa: E402


def plot_latent(
    embeddings: np.ndarray,
    scores: np.ndarray | None = None,
    output_path: str | Path | None = None,
    method: str = "tsne",
    title: str = "Latent Space Projection",
    figsize: tuple[float, float] = (10, 8),
    perplexity: float = 30.0,
    random_state: int = 42,
) -> Figure:
    """2-D projection of latent embeddings colored by anomaly score.

    Reduces the embedding dimensionality to 2 using either t-SNE
    (``sklearn.manifold.TSNE``) or UMAP (``umap.UMAP``).  Each point is
    colored by its anomaly score so that clusters of normal vs anomalous
    samples are visually distinguishable.

    Args:
        embeddings: 2-D array of shape ``(n_samples, n_dims)`` with
            latent representations.
        scores: 1-D array of anomaly scores for coloring.  If ``None``
            a uniform color is used.
        output_path: Where to save the figure.  Format inferred from
            extension.  If ``None`` the figure is returned without saving.
        method: Dimensionality reduction method.  One of ``"tsne"`` or
            ``"umap"``.  UMAP requires the ``umap-learn`` package.
        title: Figure title.
        figsize: Width and height in inches.
        perplexity: Perplexity parameter for t-SNE (ignored for UMAP).
        random_state: Random seed for reproducibility.

    Returns:
        The matplotlib ``Figure`` object.

    Raises:
        SentinelError: If the requested *method* is not supported or the
            required library is unavailable.
        ValueError: If *embeddings* is not 2-D.
    """
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2-D, got shape {embeddings.shape}")

    projection = _reduce(
        embeddings,
        method=method,
        perplexity=perplexity,
        random_state=random_state,
    )

    fig, ax = plt.subplots(figsize=figsize)

    if scores is not None:
        scatter = ax.scatter(
            projection[:, 0],
            projection[:, 1],
            c=scores,
            cmap="coolwarm",
            s=12,
            alpha=0.7,
            edgecolors="none",
        )
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("Anomaly Score", fontsize=9)
    else:
        ax.scatter(
            projection[:, 0],
            projection[:, 1],
            s=12,
            alpha=0.7,
            color="steelblue",
            edgecolors="none",
        )

    method_label = method.upper()
    ax.set_xlabel(f"{method_label} Dimension 1")
    ax.set_ylabel(f"{method_label} Dimension 2")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


def _reduce(
    embeddings: np.ndarray,
    method: str,
    perplexity: float,
    random_state: int,
) -> np.ndarray:
    """Reduce embeddings to 2 dimensions.

    Args:
        embeddings: Input array of shape ``(n_samples, n_dims)``.
        method: ``"tsne"`` or ``"umap"``.
        perplexity: t-SNE perplexity.
        random_state: Random seed.

    Returns:
        2-D array of shape ``(n_samples, 2)``.

    Raises:
        SentinelError: If the method is unknown or its dependency is
            missing.
    """
    if method == "tsne":
        return _tsne(embeddings, perplexity=perplexity, random_state=random_state)
    elif method == "umap":
        return _umap(embeddings, random_state=random_state)
    else:
        raise SentinelError(
            f"Unknown projection method '{method}'. Use 'tsne' or 'umap'."
        )


def _tsne(
    embeddings: np.ndarray,
    perplexity: float,
    random_state: int,
) -> np.ndarray:
    """Run t-SNE via sklearn.

    Args:
        embeddings: Input array.
        perplexity: t-SNE perplexity (clamped to n_samples - 1 if needed).
        random_state: Random seed.

    Returns:
        2-D projection.
    """
    from sklearn.manifold import TSNE

    n_samples = embeddings.shape[0]
    effective_perplexity = min(perplexity, max(1.0, n_samples - 1.0))

    tsne = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
    )
    return tsne.fit_transform(embeddings)  # type: ignore[return-value]


def _umap(
    embeddings: np.ndarray,
    random_state: int,
) -> np.ndarray:
    """Run UMAP dimensionality reduction.

    Args:
        embeddings: Input array.
        random_state: Random seed.

    Returns:
        2-D projection.

    Raises:
        SentinelError: If ``umap-learn`` is not installed.
    """
    try:
        import umap  # type: ignore[import-untyped]
    except ImportError as exc:
        raise SentinelError(
            "UMAP requires the 'umap-learn' package. "
            "Install it with: uv add --group explain umap-learn"
        ) from exc

    reducer = umap.UMAP(
        n_components=2,
        random_state=random_state,
    )
    return reducer.fit_transform(embeddings)  # type: ignore[return-value]
