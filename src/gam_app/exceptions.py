class GamAppError(Exception):
    """Base application exception."""


class ConfigurationError(GamAppError):
    """Invalid experiment configuration."""


class DataValidationError(GamAppError):
    """Input data violates the experiment contract."""


class CheckpointError(GamAppError):
    """Checkpoint is incomplete or incompatible."""
