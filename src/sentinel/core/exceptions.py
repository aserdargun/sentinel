"""Sentinel exception hierarchy."""


class SentinelError(Exception):
    """Base exception for all Sentinel errors."""


class ModelNotFoundError(SentinelError):
    """Raised when a model name is not found in the registry."""


class ValidationError(SentinelError):
    """Raised when data validation fails."""


class ConfigError(SentinelError):
    """Raised when configuration is invalid."""
