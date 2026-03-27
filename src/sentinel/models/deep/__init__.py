"""Deep learning models — optional dependency group 'deep'.

Models are registered only if torch is available. Each model is imported
independently so that a missing sibling module does not block the rest.
"""

try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

if _HAS_TORCH:
    try:
        from sentinel.models.deep import autoencoder  # noqa: F401
    except ImportError:
        pass

    try:
        from sentinel.models.deep import lstm  # noqa: F401
    except ImportError:
        pass

    try:
        from sentinel.models.deep import lstm_ae  # noqa: F401
    except ImportError:
        pass

    try:
        from sentinel.models.deep import tcn  # noqa: F401
    except ImportError:
        pass
