"""Data pipeline for Sentinel."""

from sentinel.data.features import (
    add_fft_features,
    add_lags,
    add_rolling_stats,
    add_temporal_features,
)
from sentinel.data.loaders import load_csv, load_file, load_parquet
from sentinel.data.pi_connector import PIConnectionError, PIConnector, is_pi_available
from sentinel.data.preprocessors import (
    chronological_split,
    create_windows,
    fill_nan,
    scale_minmax,
    scale_zscore,
    to_numpy,
)
from sentinel.data.streaming import stream_from_dataframe, stream_from_parquet
from sentinel.data.synthetic import generate_synthetic
from sentinel.data.validators import (
    get_feature_columns,
    separate_labels,
    validate_dataframe,
)

__all__ = [
    "PIConnectionError",
    "PIConnector",
    "add_fft_features",
    "add_lags",
    "add_rolling_stats",
    "add_temporal_features",
    "chronological_split",
    "create_windows",
    "fill_nan",
    "generate_synthetic",
    "get_feature_columns",
    "is_pi_available",
    "load_csv",
    "load_file",
    "load_parquet",
    "scale_minmax",
    "scale_zscore",
    "separate_labels",
    "stream_from_dataframe",
    "stream_from_parquet",
    "to_numpy",
    "validate_dataframe",
]
