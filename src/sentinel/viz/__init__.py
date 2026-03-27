"""Visualization module: matplotlib-based plots for anomaly detection.

Re-exports:
    - :func:`plot_timeseries` -- time series with anomaly region shading
    - :func:`plot_reconstruction` -- original vs reconstructed overlay + heatmap
    - :func:`plot_latent` -- t-SNE / UMAP projection of latent embeddings
"""

from sentinel.viz.latent import plot_latent
from sentinel.viz.reconstruction import plot_reconstruction
from sentinel.viz.timeseries import plot_timeseries

__all__ = [
    "plot_latent",
    "plot_reconstruction",
    "plot_timeseries",
]
