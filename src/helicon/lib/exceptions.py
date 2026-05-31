__all__ = [
    "HeliconError",
    "HeliconExit",
    "HeliconValueError",
    "HeliconIOError",
    "HeliconTypeError",
    "HeliconValidationError",
    "HeliconFileExistsError",
    "HeliconConfigError",
    "HeliconDependencyError",
]


class HeliconExit(Exception):
    """Exception for successful early termination.

    Raised when a plugin has completed its work and written all output,
    and wishes to exit without further processing. Unlike HeliconError,
    this is not an error condition.
    """


class HeliconError(Exception):
    """Base exception for all helicon errors."""


class HeliconValueError(HeliconError, ValueError):
    """Invalid value or argument."""


class HeliconIOError(HeliconError, IOError):
    """I/O error accessing files or URLs."""


class HeliconTypeError(HeliconError, TypeError):
    """Type mismatch error."""


class HeliconValidationError(HeliconError):
    """CLI argument validation error."""


class HeliconFileExistsError(HeliconError):
    """Output file already exists."""


class HeliconConfigError(HeliconError):
    """Configuration or environment error."""


class HeliconDependencyError(HeliconError):
    """Missing optional dependency."""
