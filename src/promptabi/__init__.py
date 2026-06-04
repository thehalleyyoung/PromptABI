"""Public API for PromptABI.

PromptABI verifies the discrete interface layer around LLM systems: prompts,
chat templates, tokenizer metadata, stop policies, schemas, tools, and
framework budget contracts. The initial package exposes stable typed building
blocks that later checkers can reuse without changing the embedding API.
"""

from ._version import __version__
from .config import ConfigError, VerificationConfig, load_config
from .diagnostics import (
    ArtifactRef,
    Diagnostic,
    DiagnosticSeverity,
    SourceSpan,
    WitnessTrace,
)
from .session import VerificationResult, VerificationSession

__all__ = [
    "__version__",
    "ArtifactRef",
    "ConfigError",
    "Diagnostic",
    "DiagnosticSeverity",
    "SourceSpan",
    "VerificationConfig",
    "VerificationResult",
    "VerificationSession",
    "WitnessTrace",
    "load_config",
]

