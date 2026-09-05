from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .comparison import ComparabilityAssessment


class GamAppError(Exception):
    """Base application exception."""


class ConfigurationError(GamAppError):
    """Invalid experiment configuration."""


class DataValidationError(GamAppError):
    """Input data violates the experiment contract."""


class CheckpointError(GamAppError):
    """Checkpoint is incomplete or incompatible."""


class RunComparabilityError(DataValidationError):
    """Raised when results cannot be compared as paired folds."""

    def __init__(
        self,
        message: str,
        *,
        assessment: ComparabilityAssessment | None = None,
    ) -> None:
        super().__init__(message)
        self.assessment = assessment


class SensitivityManifestError(GamAppError):
    """Raised when sensitivity manifest creation or validation fails."""


class DiagnosticReviewError(GamAppError):
    """Raised when diagnostic review cannot be completed or is invalid."""
