"""Device selection utility for PyTorch models."""

from __future__ import annotations


def resolve_device(device: str = "auto") -> str:
    """Resolve device string to best available hardware.

    Args:
        device: One of "auto", "cpu", "cuda", "mps".

    Returns:
        Resolved device string suitable for torch.device().
    """
    if device != "auto":
        return device

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass

    return "cpu"
