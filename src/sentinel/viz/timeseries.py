"""Time series visualization with anomaly region highlighting.

Plots time series data with anomaly scores overlaid and anomalous regions
shaded in red.  Uses the non-interactive ``Agg`` backend so plots can be
generated on headless servers.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402


def plot_timeseries(
    timestamps: np.ndarray | Sequence[str],
    values: np.ndarray,
    scores: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    threshold: float | None = None,
    title: str = "Time Series with Anomaly Detection",
    output_path: str | Path | None = None,
    figsize: tuple[float, float] = (14, 8),
) -> Figure:
    """Plot a time series with optional anomaly highlighting.

    Creates a two-panel figure.  The top panel shows the raw feature
    values with anomalous regions shaded red.  The bottom panel shows
    anomaly scores with an optional threshold line.

    Args:
        timestamps: 1-D array or sequence of timestamp strings /
            datetime objects for the x-axis.
        values: 2-D array of shape ``(n_samples, n_features)`` or
            1-D array of shape ``(n_samples,)`` with feature values.
        scores: 1-D array of anomaly scores aligned with *timestamps*.
            If ``None`` the score panel is omitted.
        labels: 1-D binary array (0/1) indicating ground-truth or
            predicted anomaly labels.  Used to shade anomalous regions.
        threshold: Anomaly threshold drawn as a horizontal line on the
            score panel.
        title: Figure title.
        output_path: Where to save the figure.  The format is inferred
            from the extension (e.g. ``.png``, ``.pdf``).  If ``None``
            the figure is returned without saving.
        figsize: Width and height of the figure in inches.

    Returns:
        The matplotlib ``Figure`` object.
    """
    has_scores = scores is not None
    n_axes = 2 if has_scores else 1

    fig, axes = plt.subplots(
        n_axes,
        1,
        figsize=figsize,
        sharex=True,
        squeeze=False,
    )
    ax_ts = axes[0, 0]

    # -- Top panel: feature values --
    if values.ndim == 1:
        values = values.reshape(-1, 1)

    n_features = values.shape[1]
    x_idx = np.arange(len(timestamps))

    for i in range(n_features):
        label = f"Feature {i}" if n_features > 1 else "Value"
        ax_ts.plot(x_idx, values[:, i], linewidth=0.8, label=label, alpha=0.85)

    # Shade anomalous regions.
    if labels is not None:
        _shade_anomalies(ax_ts, x_idx, labels)

    ax_ts.set_ylabel("Value")
    ax_ts.set_title(title)
    if n_features <= 10:
        ax_ts.legend(loc="upper left", fontsize=7, ncol=min(n_features, 5))
    ax_ts.grid(True, alpha=0.3)

    # -- Bottom panel: anomaly scores --
    if has_scores:
        assert scores is not None  # type narrowing
        ax_sc = axes[1, 0]
        ax_sc.plot(x_idx, scores, color="darkorange", linewidth=0.8, label="Score")

        if threshold is not None:
            ax_sc.axhline(
                threshold,
                color="red",
                linestyle="--",
                linewidth=1.0,
                label=f"Threshold ({threshold:.4f})",
            )

        if labels is not None:
            _shade_anomalies(ax_sc, x_idx, labels)

        ax_sc.set_ylabel("Anomaly Score")
        ax_sc.legend(loc="upper left", fontsize=7)
        ax_sc.grid(True, alpha=0.3)

    # X-axis tick labels.
    bottom_ax = axes[-1, 0]
    _set_timestamp_ticks(bottom_ax, x_idx, timestamps)
    bottom_ax.set_xlabel("Time")

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


def _shade_anomalies(
    ax: plt.Axes,
    x_idx: np.ndarray,
    labels: np.ndarray,
) -> None:
    """Shade contiguous anomalous regions on an axis.

    Args:
        ax: Matplotlib axes to shade on.
        x_idx: Integer index array for the x-axis.
        labels: Binary label array (1 = anomalous).
    """
    in_anomaly = False
    start = 0

    for i, lbl in enumerate(labels):
        if lbl == 1 and not in_anomaly:
            start = i
            in_anomaly = True
        elif lbl == 0 and in_anomaly:
            ax.axvspan(x_idx[start], x_idx[i - 1], alpha=0.2, color="red", label=None)
            in_anomaly = False

    # Close trailing anomaly region.
    if in_anomaly:
        ax.axvspan(x_idx[start], x_idx[-1], alpha=0.2, color="red", label=None)


def _set_timestamp_ticks(
    ax: plt.Axes,
    x_idx: np.ndarray,
    timestamps: np.ndarray | Sequence[str],
    max_ticks: int = 10,
) -> None:
    """Set readable timestamp tick labels on the x-axis.

    Args:
        ax: Matplotlib axes.
        x_idx: Integer index array.
        timestamps: Original timestamp values.
        max_ticks: Maximum number of tick labels to display.
    """
    n = len(timestamps)
    if n <= max_ticks:
        tick_positions = x_idx
    else:
        step = max(1, n // max_ticks)
        tick_positions = x_idx[::step]

    tick_labels = [str(timestamps[i])[:19] for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
