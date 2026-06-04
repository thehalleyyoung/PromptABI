"""Public API for PromptABI.

PromptABI verifies the discrete interface layer around LLM systems: prompts,
chat templates, tokenizer metadata, stop policies, schemas, tools, and
framework budget contracts. The initial package exposes stable typed building
blocks that later checkers can reuse without changing the embedding API.
"""

from ._version import __version__
from .artifacts import (
    ArtifactBundle,
    ArtifactKind,
    ArtifactLocation,
    ArtifactProvenance,
    ChatTemplateArtifact,
    FrameworkTruncationConfigArtifact,
    GrammarArtifact,
    PromptSegment,
    PromptSegmentArtifact,
    ProviderConfigArtifact,
    SchemaArtifact,
    SpecialToken,
    SpecialTokenMapArtifact,
    StopPolicyArtifact,
    TokenizerArtifact,
    ToolDefinitionArtifact,
    TruncationStrategy,
)
from .config import ConfigError, VerificationConfig, discover_config, load_config
from .diagnostics import (
    ArtifactRef,
    Diagnostic,
    DiagnosticSeverity,
    SourceSpan,
    WitnessStep,
    WitnessTrace,
)
from .loaders import ArtifactLoadError, ArtifactLoader, ArtifactLoadWarning, LoadedArtifact, load_artifact
from .session import VerificationResult, VerificationSession

__all__ = [
    "__version__",
    "ArtifactBundle",
    "ArtifactKind",
    "ArtifactLocation",
    "ArtifactLoadError",
    "ArtifactLoadWarning",
    "ArtifactLoader",
    "ArtifactProvenance",
    "ArtifactRef",
    "ChatTemplateArtifact",
    "ConfigError",
    "Diagnostic",
    "DiagnosticSeverity",
    "FrameworkTruncationConfigArtifact",
    "GrammarArtifact",
    "LoadedArtifact",
    "PromptSegment",
    "PromptSegmentArtifact",
    "ProviderConfigArtifact",
    "SchemaArtifact",
    "SourceSpan",
    "SpecialToken",
    "SpecialTokenMapArtifact",
    "StopPolicyArtifact",
    "TokenizerArtifact",
    "ToolDefinitionArtifact",
    "TruncationStrategy",
    "VerificationConfig",
    "VerificationResult",
    "VerificationSession",
    "WitnessStep",
    "WitnessTrace",
    "discover_config",
    "load_artifact",
    "load_config",
]
