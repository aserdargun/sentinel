"""Interpretability module: explain why anomalies were flagged.

Re-exports:
    - :class:`SHAPExplainer` -- SHAP-based feature attribution
    - :class:`ReconstructionExplainer` -- per-feature reconstruction error
"""

from sentinel.explain.reconstruction import ReconstructionExplainer
from sentinel.explain.shap_explainer import SHAPExplainer

__all__ = [
    "ReconstructionExplainer",
    "SHAPExplainer",
]
