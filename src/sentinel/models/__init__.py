"""Model zoo — triggers registration of all built-in models."""

from sentinel.models import statistical  # noqa: F401

try:
    from sentinel.models import deep  # noqa: F401
except ImportError:
    pass
