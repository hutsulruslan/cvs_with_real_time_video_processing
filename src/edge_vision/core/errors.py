class EdgeVisionError(Exception):
    """Base exception for expected application errors."""


class ConfigurationError(EdgeVisionError):
    """Raised when application configuration is missing or invalid."""


class ApplicationError(EdgeVisionError):
    """Raised for high-level application lifecycle errors."""
