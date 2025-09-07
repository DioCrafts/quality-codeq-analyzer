"""
CodeQ: Advanced Code Quality Analyzer
=====================================

A multi-language, explainable code quality measurement tool that calculates
and reports metrics for Python, TypeScript/JavaScript, Go, and Rust codebases.

Main Features:
- Complexity analysis (cyclomatic, cognitive, Halstead)
- Maintainability index with banding
- Coupling/cohesion metrics (fan-in/out, instability)
- Configurable code smells detection
- Documentation coverage analysis
- Test coverage integration
- Security pattern detection
- Technical debt estimation
- Multiple output formats (JSON, HTML, SARIF)
"""

__version__ = "0.1.0"
__author__ = "CodeQ Team"

# Main classes and functions for public API
from .aggregate import (
    MetricsAggregator,
    ScoringWeights,
    FileScore,
    PackageScore,
    RepositoryScore,
    AggregationLevel,
    generate_quality_gates
)
from .cli import main as cli_main
from .astparse import ASTParser
from .metrics import MetricsCalculator
from .smells import SmellDetector
from .coupling import CouplingAnalyzer
from .coverage import CoverageParser
from .report import ReportGenerator
from .utils import FileUtils, PathUtils

# Default configuration
DEFAULT_RULES_PATH = "rules/defaults.yaml"
DEFAULT_CACHE_DIR = ".codeq_cache"

__all__ = [
    # Core classes
    "MetricsAggregator",
    "ASTParser",
    "MetricsCalculator",
    "SmellDetector",
    "CouplingAnalyzer",
    "CoverageParser",
    "ReportGenerator",

    # Data models
    "FileScore",
    "PackageScore",
    "RepositoryScore",
    "ScoringWeights",
    "AggregationLevel",

    # Utilities
    "FileUtils",
    "PathUtils",
    "generate_quality_gates",
    "cli_main",

    # Constants
    "DEFAULT_RULES_PATH",
    "DEFAULT_CACHE_DIR",
    "__version__",
    "__author__",
]
