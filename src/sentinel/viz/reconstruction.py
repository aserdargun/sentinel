"""Reconstruction visualization: original vs reconstructed overlay + error heatmap.

Provides two sub-plots: (1) an overlay of the original and reconstructed
time series per feature and (2) a per-feature reconstruction error heatmap
using ``imshow``.  Designed for evaluating autoencoder-family models.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402


def plot_reconstruction(
    original: np.ndarray,
    reconstructed: np.ndarray,
    feature_names: Sequence[str] | None = None,
    output_path: str | Path | None = None,
    title: str = "Reconstruction Analysis",
    figsize: tuple[float, float] = (14, 10),
    max_features_overlay: int = 6,
) -> Figure:
    """Plot original vs reconstructed time series with error heatmap.

    Creates a two-row figure:

    * **Top panel** -- overlay of original and reconstructed values for
      up to *max_features_overlay* features.
    * **Bottom panel** -- per-feature reconstruction error heatmap
      (``imshow``), where brighter cells indicate larger errors.

    Args:
        original: Array of shape ``(n_samples, n_features)`` with
            original input values.
        reconstructed: Array of same shape as *original* with
            model-reconstructed values.
        feature_names: Optional list of feature names for axis labels.
            If ``None``, features are numbered ``Feature 0``, etc.
        output_path: Where to save the figure.  Format inferred from
            extension.  If ``None`` the figure is returned without saving.
        title: Figure super-title.
        figsize: Width and height of the figure in inches.
        max_features_overlay: Maximum number of features to show in the
            overlay panel.  If there are more features, only the first N
            are plotted.

    Returns:
        The matplotlib ``Figure`` object.

    Raises:
        ValueError: If *original* and *reconstructed* shapes do not match.
    """
    if original.shape != reconstructed.shape:
        raise ValueError(
            f"Shape mismatch: original {original.shape} vs "
            f"reconstructed {reconstructed.shape}"
        )

    if original.ndim == 1:
        original = original.reshape(-1, 1)
        reconstructed = reconstructed.reshape(-1, 1)

    n_samples, n_features = original.shape
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(n_features)]

    error = np.abs(original - reconstructed)

    fig, (ax_overlay, ax_heatmap) = plt.subplots(
        2,
        1,
        figsize=figsize,
        gridspec_kw={"height_ratios": [1.2, 1]},
    )
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # -- Top: overlay of original vs reconstructed --
    n_show = min(n_features, max_features_overlay)
    x_idx = np.arange(n_samples)
    colors = plt.cm.tab10(np.linspace(0, 1, n_show))  # type: ignore[attr-defined]

    for i in range(n_show):
        color = colors[i]
        ax_overlay.plot(
            x_idx,
            original[:, i],
            color=color,
            linewidth=0.9,
            label=f"{feature_names[i]} (orig)",
        )
        ax_overlay.plot(
            x_idx,
            reconstructed[:, i],
            color=color,
            linewidth=0.9,
            linestyle="--",
            alpha=0.7,
            label=f"{feature_names[i]} (recon)",
        )

    ax_overlay.set_ylabel("Value")
    ax_overlay.set_title("Original vs Reconstructed")
    ax_overlay.legend(
        loc="upper left",
        fontsize=6,
        ncol=min(n_show, 3),
        framealpha=0.8,
    )
    ax_overlay.grid(True, alpha=0.3)

    # -- Bottom: per-feature error heatmap --
    # Transpose so features are on the y-axis and time on the x-axis.
    im = ax_heatmap.imshow(
        error.T,
        aspect="auto",
        cmap="hot",
        interpolation="nearest",
    )
    ax_heatmap.set_xlabel("Time Step")
    ax_heatmap.set_ylabel("Feature")
    ax_heatmap.set_title("Per-Feature Reconstruction Error")

    # Label y-axis with feature names.
    ax_heatmap.set_yticks(range(n_features))
    ax_heatmap.set_yticklabels([str(fn) for fn in feature_names], fontsize=7)

    cbar = fig.colorbar(im, ax=ax_heatmap, fraction=0.02, pad=0.02)
    cbar.set_label("Absolute Error", fontsize=8)

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig
