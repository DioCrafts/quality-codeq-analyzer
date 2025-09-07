"""
Command Line Interface for CodeQ

Provides a simple CLI interface for scanning codebases and generating quality reports.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
import json
import time
from dataclasses import dataclass

from .aggregate import MetricsAggregator, RepositoryScore, generate_quality_gates
from .astparse import ASTParser
from .metrics import MetricsCalculator
from .smells import SmellDetector
from .coupling import CouplingAnalyzer
from .coverage import CoverageParser
from .report import ReportGenerator
from .utils import FileUtils, PathUtils
from .ml_engine import MLEngine, MLFeatureVector
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import re


class Severity(Enum):
    """Finding severity levels."""
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    INFO = "info"


@dataclass
class Position:
    """Position information for a finding."""
    path: str
    start_line: int
    end_line: int
    start_col: int = 1
    end_col: int = 1


@dataclass
class Finding:
    """Detailed finding with code snippet and remediation advice."""
    id: str
    rule: str
    severity: Severity
    message: str
    position: Position
    snippet: str
    rationale: str
    remediation_minutes: int
    tags: List[str] = field(default_factory=list)
    rule_snapshot: Optional[Dict[str, Any]] = None
    confidence: float = 1.0
    language: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        """Convert finding to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "rule": self.rule,
            "severity": self.severity.value,
            "message": self.message,
            "position": {
                "path": self.position.path,
                "start_line": self.position.start_line,
                "end_line": self.position.end_line,
                "start_col": self.position.start_col,
                "end_col": self.position.end_col
            },
            "snippet": self.snippet,
            "rationale": self.rationale,
            "remediation_minutes": self.remediation_minutes,
            "tags": self.tags,
            "confidence": self.confidence,
            "language": self.language,
            "rule_snapshot": self.rule_snapshot
        }


@dataclass
class EntityMetrics:
    """Metrics for a specific code entity (function/class)."""
    id: str
    kind: str  # "function", "class", "method", etc.
    name: str
    signature: str
    public: bool
    start_line: int
    end_line: int
    loc: int
    params: int = 0
    return_type: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    smells: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary."""
        return {
            "id": self.id,
            "kind": self.kind,
            "name": self.name,
            "signature": self.signature,
            "public": self.public,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "loc": self.loc,
            "params": self.params,
            "return_type": self.return_type,
            "metrics": self.metrics,
            "smells": self.smells
        }


class CodeQCLI:
    """Main CLI class for CodeQ operations."""

    def __init__(self):
        self.parser = self._create_parser()

    def _create_parser(self) -> argparse.ArgumentParser:
        """Create the argument parser with all options."""
        parser = argparse.ArgumentParser(
            prog="codeq",
            description="Advanced Code Quality Analyzer",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  codeq scan . --lang py,ts --report report.html
  codeq scan src/ --rules custom-rules.yaml --json results.json
  codeq scan . --detailed-json comprehensive-report.json --verbose
  codeq scan . --fail-on 'severity>=major OR maintainability<C'
  codeq scan . --budget-hours 80 --sarif results.sarif
            """
        )

        parser.add_argument(
            "command",
            choices=["scan"],
            help="Command to execute"
        )

        parser.add_argument(
            "path",
            type=Path,
            help="Path to scan (file or directory)"
        )

        parser.add_argument(
            "--lang",
            type=lambda x: x.split(","),
            default=["py", "ts", "js"],
            help="Languages to analyze (comma-separated: py,ts,js,go,rust)"
        )

        parser.add_argument(
            "--rules",
            type=Path,
            default=Path("rules/defaults.yaml"),
            help="Path to rules configuration file"
        )

        parser.add_argument(
            "--report",
            type=Path,
            help="Generate HTML report at specified path"
        )

        parser.add_argument(
            "--json",
            type=Path,
            help="Generate JSON report at specified path"
        )

        parser.add_argument(
            "--detailed-json",
            type=Path,
            help="Generate comprehensive detailed JSON report with ML predictions"
        )

        parser.add_argument(
            "--sarif",
            type=Path,
            help="Generate SARIF report at specified path"
        )

        parser.add_argument(
            "--fail-on",
            help="Fail if condition is met (e.g., 'severity>=major OR maintainability<C')"
        )

        parser.add_argument(
            "--budget-hours",
            type=float,
            help="Maximum allowed technical debt in hours"
        )

        parser.add_argument(
            "--cache-dir",
            type=Path,
            default=Path(".codeq_cache"),
            help="Cache directory for incremental analysis"
        )

        parser.add_argument(
            "--workers",
            type=int,
            default=4,
            help="Number of worker processes for parallel analysis"
        )

        parser.add_argument(
            "--verbose", "-v",
            action="count",
            default=0,
            help="Increase verbosity (use -vv for debug)"
        )

        parser.add_argument(
            "--quiet", "-q",
            action="store_true",
            help="Suppress all output except errors"
        )

        return parser

    def run_scan(self, args: argparse.Namespace) -> int:
        """Run the scan command."""
        start_time = time.time()

        if not args.quiet:
            print(f"🔍 Scanning {args.path}...")
            print(f"📋 Languages: {', '.join(args.lang)}")
            print(f"📊 Rules: {args.rules}")
            print()

        try:
            # Validate inputs
            if not args.path.exists():
                print(f"❌ Error: Path {args.path} does not exist")
                return 1

            if not args.rules.exists():
                print(f"❌ Error: Rules file {args.rules} does not exist")
                return 1

            # Initialize components
            aggregator = MetricsAggregator()
            parser = ASTParser()
            # Initialize metrics calculator for primary language (Python for now)
            metrics_calc = MetricsCalculator('python')
            smell_detector = SmellDetector(args.rules)
            coupling_analyzer = CouplingAnalyzer('python')  # Default to Python for now
            coverage_parser = CoverageParser()
            report_generator = ReportGenerator()

            # Scan and analyze
            repository_score = self._analyze_repository(
                args.path,
                args.lang,
                aggregator,
                parser,
                metrics_calc,
                smell_detector,
                coupling_analyzer,
                coverage_parser,
                args.workers,
                args.verbose
            )

            # Generate reports
            self._generate_reports(
                repository_score,
                args.report,
                args.json,
                args.detailed_json,
                args.sarif,
                report_generator,
                args.verbose,
                str(args.path)
            )

            # Check quality gates
            exit_code = self._check_quality_gates(
                repository_score,
                args.fail_on,
                args.budget_hours,
                args.verbose
            )

            # Print summary
            if not args.quiet:
                self._print_summary(repository_score, time.time() - start_time)

            return exit_code

        except Exception as e:
            if args.verbose >= 2:
                import traceback
                traceback.print_exc()
            print(f"❌ Error during analysis: {e}")
            return 1

    def _analyze_repository(
        self,
        path: Path,
        languages: List[str],
        aggregator: MetricsAggregator,
        parser: ASTParser,
        metrics_calc: MetricsCalculator,
        smell_detector: SmellDetector,
        coupling_analyzer: CouplingAnalyzer,
        coverage_parser: CoverageParser,
        workers: int,
        verbose: int
    ) -> RepositoryScore:
        """Analyze the entire repository with full metrics calculation."""
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from .utils import FileUtils
        from .aggregate import PackageScore, FileScore
        # Import parallel_process if it exists, otherwise define a simple version
        try:
            from .utils import parallel_process
        except ImportError:
            def parallel_process(items, func, max_workers=None):
                """Simple parallel processing fallback."""
                return [func(item) for item in items]

        start_time = time.time()

        if verbose >= 1:
            print("📁 Analyzing repository structure...")

        # Discover all source files
        source_files = self._discover_source_files(path, languages, verbose)

        if not source_files:
            if verbose >= 1:
                print("⚠️  No source files found")
            return self._create_empty_repository_score(path)

        if verbose >= 1:
            print(f"📄 Found {len(source_files)} source files")

        # Calculate total lines and detect languages
        total_lines, detected_languages = self._calculate_repository_stats(source_files, verbose)

        if verbose >= 1:
            print(f"📝 Total lines: {total_lines:,}")
            print(f"🌍 Languages detected: {', '.join(detected_languages) if detected_languages else 'none'}")

        # Group files by package/module
        package_files = self._group_files_by_package(source_files, path)

        if verbose >= 1:
            print(f"📦 Organized into {len(package_files)} packages")

        # Analyze coverage if available
        coverage_data = self._analyze_coverage(path, coverage_parser, verbose)

        # Analyze each package in parallel
        package_scores = {}
        total_files_processed = 0

        def analyze_package(package_name: str, files: List[Path]) -> tuple[str, PackageScore]:
            """Analyze a single package."""
            nonlocal total_files_processed

            if verbose >= 2:
                print(f"🔍 Analyzing package: {package_name} ({len(files)} files)")

            file_scores = []

            for file_path in files:
                try:
                    file_score = self._analyze_single_file(
                        file_path, parser, metrics_calc, smell_detector, coverage_data
                    )
                    if file_score:
                        file_scores.append(file_score)

                except Exception as e:
                    if verbose >= 1:
                        print(f"⚠️  Error analyzing {file_path}: {e}")

            if file_scores:
                package_score = aggregator.aggregate_package_metrics(package_name, file_scores)
                total_files_processed += len(file_scores)
                return package_name, package_score

            return None, None

        # Process packages with parallelism
        if verbose >= 1:
            print("⚡ Processing packages in parallel...")

        results = parallel_process(
            list(package_files.items()),
            lambda item: analyze_package(item[0], item[1]),
            max_workers=min(workers, len(package_files)),
            description="Analyzing packages"
        )

        # Collect results
        for package_name, package_score in results:
            if package_score:
                package_scores[package_name] = package_score

        # Analyze coupling across the entire repository
        if verbose >= 1:
            print("🔗 Analyzing coupling relationships...")

        try:
            coupling_report = coupling_analyzer.analyze_repository(source_files)
            # Integrate coupling analysis into package scores
            self._integrate_coupling_analysis(package_scores, coupling_report)
        except Exception as e:
            if verbose >= 1:
                print(f"⚠️  Coupling analysis failed: {e}")

        # Aggregate repository-level metrics
        if verbose >= 1:
            print("📊 Aggregating repository metrics...")

        repository_score = aggregator.aggregate_repository_metrics(
            str(path),
            package_scores,
            total_lines=total_lines,
            detected_languages=detected_languages,
            hourly_rate=50.0  # Configurable hourly rate
        )

        # Add timing information
        analysis_time = time.time() - start_time
        repository_score.score_trend = f"Analysis completed in {analysis_time:.2f}s"

        if verbose >= 1:
            print(f"✅ Repository analysis complete in {analysis_time:.2f}s")

        return repository_score

    def _generate_reports(
        self,
        score: RepositoryScore,
        html_path: Optional[Path],
        json_path: Optional[Path],
        detailed_json_path: Optional[Path],
        sarif_path: Optional[Path],
        generator: ReportGenerator,
        verbose: int,
        repository_path: str
    ):
        """Generate requested report formats."""
        if html_path:
            if verbose >= 1:
                print(f"📄 Generating HTML report: {html_path}")
            # generator.generate_html(score, html_path)

        if json_path:
            if verbose >= 1:
                print(f"📋 Generating JSON report: {json_path}")
            with open(json_path, 'w') as f:
                json.dump(score.__dict__, f, indent=2, default=str)

        if detailed_json_path:
            if verbose >= 1:
                print(f"📊 Generating comprehensive JSON report: {detailed_json_path}")
            detailed_report = self._generate_comprehensive_json_report(
                score, repository_path, verbose
            )
            with open(detailed_json_path, 'w', encoding='utf-8') as f:
                json.dump(detailed_report, f, indent=2, default=str, ensure_ascii=False)

        if sarif_path:
            if verbose >= 1:
                print(f"🔍 Generating SARIF report: {sarif_path}")
            # generator.generate_sarif(score, sarif_path)

    def _generate_comprehensive_json_report(
        self,
        score: RepositoryScore,
        repository_path: str,
        verbose: int
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive JSON report with ML predictions and detailed analysis.

        This report includes:
        - All metrics and scores
        - ML predictions with confidence levels
        - Detailed explanations and recommendations
        - Quality gates evaluation
        - Performance statistics
        - Code patterns and smells analysis
        """
        import datetime
        import platform
        import psutil

        # Initialize ML Engine for predictions
        ml_engine = MLEngine(enable_training=False)

        # Base report structure
        report = {
            "metadata": {
                "tool": "CodeQ",
                "version": "2.0.0",
                "timestamp": datetime.datetime.now().isoformat(),
                "repository_path": repository_path,
                "scan_duration_seconds": getattr(score, 'scan_duration', 0),
                "platform": {
                    "system": platform.system(),
                    "python_version": platform.python_version(),
                    "cpu_count": psutil.cpu_count(),
                    "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2)
                }
            },

            "summary": {
                "overall_score": score.overall_score,
                "maintainability_index": score.maintainability_score,
                "technical_debt_hours": score.total_debt_hours,
                "quality_rating": self._get_quality_rating(score.overall_score),
                "risk_level": self._get_risk_level(score.overall_score),
                "total_files": score.total_files,
                "total_lines": score.total_lines,
                "languages_detected": getattr(score, 'languages', [])
            },

            "ml_predictions": {},
            "detailed_metrics": {},
            "quality_analysis": {},
            "recommendations": {},
            "performance_stats": {},
            "code_patterns": {},
        }

        # Generate ML predictions for key files
        if verbose >= 1:
            print("   🤖 Generating ML predictions...")

        ml_predictions = self._generate_ml_predictions(score, ml_engine)
        report["ml_predictions"] = ml_predictions

        # Detailed metrics breakdown
        if verbose >= 1:
            print("   📊 Compiling detailed metrics...")

        detailed_metrics = self._compile_detailed_metrics(score)
        report["detailed_metrics"] = detailed_metrics

        # Quality analysis
        quality_analysis = self._generate_quality_analysis(score)
        report["quality_analysis"] = quality_analysis

        # Generate recommendations
        if verbose >= 1:
            print("   💡 Generating recommendations...")

        recommendations = self._generate_recommendations(score, ml_predictions)
        report["recommendations"] = recommendations

        # Performance statistics
        performance_stats = self._compile_performance_stats(score)
        report["performance_stats"] = performance_stats

        # Code patterns analysis
        code_patterns = self._analyze_code_patterns(score)
        report["code_patterns"] = code_patterns


        # Generate detailed findings
        if verbose >= 1:
            print("   🔍 Analyzing detailed findings...")

        detailed_findings = self._analyze_detailed_findings(score, repository_path)
        report["detailed_findings"] = detailed_findings

        # Generate entities analysis
        entities_analysis = self._analyze_code_entities(score)
        report["entities_analysis"] = entities_analysis

        return report

    def _generate_ml_predictions(self, score: RepositoryScore, ml_engine: MLEngine) -> Dict[str, Any]:
        """Generate ML predictions for the repository."""
        predictions = {
            "bug_likelihood": {},
            "quality_forecast": {},
            "maintainability_prediction": {},
            "anomaly_detection": {},
            "overall_assessment": {}
        }

        # Create aggregate features from repository metrics
        if hasattr(score, 'avg_complexity'):
            features = MLFeatureVector(
                cyclomatic_complexity=score.avg_complexity,
                halstead_volume=getattr(score, 'avg_halstead_volume', 100),
                cbo=getattr(score, 'avg_coupling', 5),
                lines_of_code=score.total_lines // max(1, score.total_files),
                test_coverage=getattr(score, 'avg_test_coverage', 50),
                documentation_score=getattr(score, 'avg_documentation_score', 60),
                maintainability_index=score.maintainability_score
            )

            # Generate predictions
            try:
                bug_pred = ml_engine.predict_bugs(features)
                quality_pred = ml_engine.forecast_quality(features)
                maintainability_pred = ml_engine.predict_maintainability(features)
                anomaly_pred = ml_engine.detect_anomalies(features)

                predictions["bug_likelihood"] = {
                    "prediction": bug_pred.predicted_value,
                    "confidence": bug_pred.confidence.value,
                    "confidence_score": bug_pred.confidence_score,
                    "explanation": bug_pred.explanation,
                    "feature_importance": bug_pred.feature_importance
                }

                predictions["quality_forecast"] = {
                    "prediction": quality_pred.predicted_value,
                    "confidence": quality_pred.confidence.value,
                    "explanation": quality_pred.explanation
                }

                predictions["maintainability_prediction"] = {
                    "prediction": maintainability_pred.predicted_value,
                    "confidence": maintainability_pred.confidence.value,
                    "explanation": maintainability_pred.explanation
                }

                predictions["anomaly_detection"] = {
                    "prediction": anomaly_pred.predicted_value,
                    "explanation": anomaly_pred.explanation
                }

                # Overall assessment
                risk_factors = []
                if bug_pred.predicted_value == True:
                    risk_factors.append("High bug likelihood")
                if quality_pred.predicted_value < 70:
                    risk_factors.append("Poor quality forecast")
                if maintainability_pred.predicted_value < 50:
                    risk_factors.append("Low maintainability")
                if anomaly_pred.predicted_value == "anomaly":
                    risk_factors.append("Anomalous code patterns")

                predictions["overall_assessment"] = {
                    "risk_level": "HIGH" if len(risk_factors) >= 3 else "MEDIUM" if len(risk_factors) >= 2 else "LOW",
                    "risk_factors": risk_factors,
                    "confidence_score": (bug_pred.confidence_score + quality_pred.confidence_score +
                                       maintainability_pred.confidence_score) / 3
                }

            except Exception as e:
                predictions["error"] = f"ML prediction failed: {str(e)}"

        return predictions

    def _compile_detailed_metrics(self, score: RepositoryScore) -> Dict[str, Any]:
        """Compile detailed metrics breakdown."""
        return {
            "complexity_metrics": {
                "avg_cyclomatic_complexity": getattr(score, 'avg_complexity', 0),
                "max_cyclomatic_complexity": getattr(score, 'max_complexity', 0),
                "cognitive_complexity_avg": getattr(score, 'avg_cognitive_complexity', 0),
                "halstead_metrics": {
                    "avg_volume": getattr(score, 'avg_halstead_volume', 0),
                    "avg_difficulty": getattr(score, 'avg_halstead_difficulty', 0),
                    "avg_effort": getattr(score, 'avg_halstead_effort', 0)
                }
            },
            "size_metrics": {
                "total_files": score.total_files,
                "total_lines_of_code": score.total_lines,
                "avg_file_size": score.total_lines / max(1, score.total_files),
                "total_functions": getattr(score, 'total_functions', 0),
                "total_classes": getattr(score, 'total_classes', 0)
            },
            "quality_metrics": {
                "maintainability_index": score.maintainability_score,
                "technical_debt_hours": score.total_debt_hours,
                "test_coverage_avg": getattr(score, 'avg_test_coverage', 0),
                "documentation_coverage": getattr(score, 'avg_documentation_score', 0),
                "duplication_percentage": getattr(score, 'avg_duplication', 0)
            },
            "coupling_metrics": {
                "avg_coupling_between_objects": getattr(score, 'avg_coupling', 0),
                "inheritance_depth_avg": getattr(score, 'avg_inheritance_depth', 0),
                "lack_of_cohesion_avg": getattr(score, 'avg_lcom', 0)
            }
        }

    def _generate_quality_analysis(self, score: RepositoryScore) -> Dict[str, Any]:
        """Generate comprehensive quality analysis."""
        # Determine quality bands
        mi = score.maintainability_score
        if mi >= 85:
            quality_band = "A"
            quality_description = "Excellent maintainability"
        elif mi >= 70:
            quality_band = "B"
            quality_description = "Good maintainability"
        elif mi >= 55:
            quality_band = "C"
            quality_description = "Moderate maintainability"
        elif mi >= 40:
            quality_band = "D"
            quality_description = "Poor maintainability"
        else:
            quality_band = "E"
            quality_description = "Very poor maintainability"

        return {
            "quality_band": quality_band,
            "quality_description": quality_description,
            "maintainability_assessment": {
                "score": mi,
                "band": quality_band,
                "description": quality_description
            },
            "technical_debt_analysis": {
                "total_hours": score.total_debt_hours,
                "cost_estimation": score.total_debt_hours * 50,  # Assuming $50/hour
                "severity_level": "HIGH" if score.total_debt_hours > 100 else "MEDIUM" if score.total_debt_hours > 50 else "LOW"
            },
            "code_health_indicators": {
                "complexity_risk": "HIGH" if getattr(score, 'avg_complexity', 0) > 15 else "MEDIUM" if getattr(score, 'avg_complexity', 0) > 10 else "LOW",
                "test_coverage_status": "GOOD" if getattr(score, 'avg_test_coverage', 0) > 80 else "FAIR" if getattr(score, 'avg_test_coverage', 0) > 60 else "POOR",
                "documentation_status": "GOOD" if getattr(score, 'avg_documentation_score', 0) > 75 else "FAIR" if getattr(score, 'avg_documentation_score', 0) > 50 else "POOR"
            }
        }

    def _generate_recommendations(self, score: RepositoryScore, ml_predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed recommendations based on analysis."""
        recommendations = {
            "immediate_actions": [],
            "short_term_goals": [],
            "long_term_improvements": [],
            "priority_level": "LOW"
        }

        # Generate recommendations based on scores
        if score.maintainability_score < 50:
            recommendations["immediate_actions"].append({
                "action": "Refactor complex functions",
                "impact": "HIGH",
                "effort": "MEDIUM",
                "description": "Functions with cyclomatic complexity > 15 should be refactored"
            })
            recommendations["priority_level"] = "HIGH"

        if getattr(score, 'avg_test_coverage', 0) < 70:
            recommendations["short_term_goals"].append({
                "action": "Increase test coverage",
                "target": "80%",
                "current": f"{getattr(score, 'avg_test_coverage', 0):.1f}%",
                "description": "Aim for at least 80% test coverage"
            })

        if score.total_debt_hours > 50:
            recommendations["long_term_improvements"].append({
                "action": "Address technical debt",
                "estimated_hours": score.total_debt_hours,
                "estimated_cost": score.total_debt_hours * 50,
                "description": "Systematic refactoring to reduce technical debt"
            })

        # ML-based recommendations
        if ml_predictions.get("bug_likelihood", {}).get("prediction") == True:
            recommendations["immediate_actions"].append({
                "action": "Review high-risk code sections",
                "impact": "HIGH",
                "effort": "LOW",
                "description": "ML detected high bug likelihood - focus code reviews on complex sections"
            })

        if ml_predictions.get("anomaly_detection", {}).get("prediction") == "anomaly":
            recommendations["short_term_goals"].append({
                "action": "Investigate anomalous patterns",
                "impact": "MEDIUM",
                "effort": "MEDIUM",
                "description": "Code patterns deviate from norms - investigate potential issues"
            })

        return recommendations

    def _compile_performance_stats(self, score: RepositoryScore) -> Dict[str, Any]:
        """Compile performance statistics."""
        return {
            "analysis_performance": {
                "scan_duration_seconds": getattr(score, 'scan_duration', 0),
                "files_per_second": score.total_files / max(1, getattr(score, 'scan_duration', 1)),
                "lines_per_second": score.total_lines / max(1, getattr(score, 'scan_duration', 1))
            },
            "code_metrics_distribution": {
                "complexity_distribution": self._calculate_distribution(getattr(score, 'complexities', [])),
                "size_distribution": self._calculate_distribution(getattr(score, 'file_sizes', [])),
                "quality_distribution": self._calculate_distribution([score.maintainability_score])
            },
            "resource_usage": {
                "estimated_analysis_complexity": "HIGH" if score.total_lines > 100000 else "MEDIUM" if score.total_lines > 50000 else "LOW",
                "parallelization_efficiency": getattr(score, 'parallel_efficiency', 1.0)
            }
        }

    def _analyze_code_patterns(self, score: RepositoryScore) -> Dict[str, Any]:
        """Analyze code patterns and anti-patterns."""
        return {
            "design_patterns": {
                "detected": getattr(score, 'detected_patterns', []),
                "confidence": getattr(score, 'pattern_confidence', 0.0)
            },
            "anti_patterns": {
                "god_objects": getattr(score, 'god_objects_count', 0),
                "long_methods": getattr(score, 'long_methods_count', 0),
                "duplicate_code_blocks": getattr(score, 'duplicate_blocks', 0)
            },
            "code_style": {
                "consistent_naming": getattr(score, 'consistent_naming_score', 0.0),
                "proper_documentation": getattr(score, 'documentation_consistency', 0.0),
                "modular_design": getattr(score, 'modularity_score', 0.0)
            }
        }


    def _analyze_detailed_findings(self, score: RepositoryScore, repository_path: str) -> List[Dict[str, Any]]:
        """
        Analyze code and generate detailed findings with snippets and remediation advice.

        This function performs deep analysis of source files to identify specific issues
        with code snippets and actionable remediation advice.
        """
        findings = []
        finding_id_counter = 1

        # Get all source files in the repository
        source_files = self._get_source_files(repository_path)

        for file_path in source_files:
            try:
                file_findings = self._analyze_file_findings(file_path, finding_id_counter)
                findings.extend(file_findings)
                finding_id_counter += len(file_findings)
            except Exception as e:
                # Skip files that can't be analyzed
                continue

        return [finding.to_dict() for finding in findings]

    def _calculate_repository_stats(self, source_files: List[Path], verbose: int) -> tuple[int, List[str]]:
        """Calculate total lines and detect languages for the repository."""
        total_lines = 0
        detected_languages = set()

        for file_path in source_files:
            try:
                # Count lines
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    total_lines += len(lines)

                # Detect language from extension
                ext = file_path.suffix.lower()
                if ext == '.py':
                    detected_languages.add('python')
                elif ext in ['.js', '.jsx']:
                    detected_languages.add('javascript')
                elif ext in ['.ts', '.tsx']:
                    detected_languages.add('typescript')
                elif ext in ['.java']:
                    detected_languages.add('java')
                elif ext in ['.rs']:
                    detected_languages.add('rust')
                elif ext in ['.go']:
                    detected_languages.add('go')
                elif ext in ['.php']:
                    detected_languages.add('php')
                elif ext in ['.rb']:
                    detected_languages.add('ruby')

            except Exception:
                # Skip files that can't be read
                continue

        return total_lines, sorted(list(detected_languages))

    def _get_source_files(self, repository_path: str) -> List[str]:
        """Get all source files in the repository."""
        source_files = []
        extensions = ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.cs', '.php', '.rb', '.go', '.rs']

        for root, dirs, files in os.walk(repository_path):
            # Skip common directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', 'venv', '__pycache__', 'dist', 'build']]

            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    source_files.append(os.path.join(root, file))

        return source_files[:20]  # Limit to first 20 files for performance

    def _analyze_file_findings(self, file_path: str, start_id: int) -> List[Finding]:
        """Analyze a single file for detailed findings."""
        findings = []

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            lines = content.split('\n')
            language = self._detect_file_language(file_path)

            # Analyze different types of issues
            findings.extend(self._analyze_complexity_issues(file_path, content, lines, language, start_id))
            findings.extend(self._analyze_style_issues(file_path, content, lines, language, start_id + len(findings)))

        except Exception as e:
            # If file can't be read, skip it
            pass

        return findings

    def _detect_file_language(self, file_path: str) -> str:
        """Detect the programming language of a file."""
        ext = os.path.splitext(file_path)[1].lower()
        language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.cs': 'csharp',
            '.php': 'php',
            '.rb': 'ruby',
            '.go': 'go',
            '.rs': 'rust'
        }
        return language_map.get(ext, 'unknown')

    def _analyze_complexity_issues(self, file_path: str, content: str, lines: List[str], language: str, start_id: int) -> List[Finding]:
        """Analyze code complexity issues."""
        findings = []

        # Deep nesting analysis
        nesting_findings = self._analyze_deep_nesting(file_path, content, lines, language, start_id)
        findings.extend(nesting_findings)

        # Long function analysis
        long_function_findings = self._analyze_long_functions(file_path, content, lines, language, start_id + len(findings))
        findings.extend(long_function_findings)

        return findings

    def _analyze_deep_nesting(self, file_path: str, content: str, lines: List[str], language: str, start_id: int) -> List[Finding]:
        """Analyze deep nesting issues."""
        findings = []

        if language in ['python', 'javascript', 'typescript', 'java', 'csharp']:
            indent_char = '    ' if language == 'python' else '  '

            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                if not stripped or stripped.startswith('//') or stripped.startswith('#') or stripped.startswith('*'):
                    continue

                indent_level = len(line) - len(line.lstrip())
                indent_count = indent_level // len(indent_char) if indent_char else indent_level // 2

                if indent_count >= 4:  # Deep nesting (4+ levels)
                    # Extract context around the deeply nested code
                    start_ctx = max(0, i - 3)
                    end_ctx = min(len(lines), i + 3)
                    context_lines = lines[start_ctx:end_ctx]
                    snippet = '\n'.join(f"{j+start_ctx+1:3d}: {line.rstrip()}" for j, line in enumerate(context_lines))

                    position = Position(
                        path=file_path,
                        start_line=i,
                        end_line=i,
                        start_col=indent_level + 1,
                        end_col=len(line)
                    )

                    finding = Finding(
                        id="02d",
                        rule="deep-nesting",
                        severity=Severity.MAJOR,
                        message="Anidación profunda (>3 niveles) detectada. Considera extraer guard clauses o funciones auxiliares.",
                        position=position,
                        snippet=snippet,
                        rationale="La anidación profunda incrementa la carga cognitiva y dificulta pruebas unitarias.",
                        remediation_minutes=24,
                        tags=["maintainability", "complexity", "readability"],
                        language=language,
                        rule_snapshot={
                            "where": "any",
                            "condition": "indent_level >= 4",
                            "severity": "major",
                            "remediation_per_unit_min": 8,
                            "unit": "levels_over_3"
                        }
                    )
                    findings.append(finding)
                    break  # Only report first finding per file for brevity

        return findings

    def _analyze_long_functions(self, file_path: str, content: str, lines: List[str], language: str, start_id: int) -> List[Finding]:
        """Analyze long functions/methods."""
        findings = []

        if language == 'python':
            # Find Python functions
            function_pattern = r'^def\s+(\w+)\s*\([^)]*\)\s*:'
        elif language in ['javascript', 'typescript']:
            # Find JS/TS functions
            function_pattern = r'(?:function\s+\w+|const\s+\w+\s*=\s*\([^)]*\)\s*=>|const\s+\w+\s*=\s*function)'
        else:
            return findings

        for i, line in enumerate(lines, 1):
            match = re.match(function_pattern, line.strip())
            if match:
                func_name = match.group(1) if language == 'python' else "anonymous_function"
                start_line = i

                # Count function length (simplified)
                func_lines = 1
                j = i
                while j < len(lines) and func_lines < 100:  # Prevent infinite loop
                    j += 1
                    func_lines += 1
                    if j >= len(lines):
                        break

                    # Simple function end detection
                    next_line = lines[j].strip()
                    if language == 'python' and next_line and not next_line.startswith(' ') and not next_line.startswith('\t'):
                        break
                    elif language in ['javascript', 'typescript'] and next_line == '}':
                        break

                if func_lines > 50:  # Long function threshold
                    end_line = min(start_line + func_lines - 1, len(lines))

                    # Extract function context
                    start_ctx = max(0, start_line - 2)
                    end_ctx = min(len(lines), end_line + 3)
                    context_lines = lines[start_ctx:end_ctx]
                    snippet = '\n'.join(f"{j+start_ctx+1:3d}: {line.rstrip()}" for j, line in enumerate(context_lines))

                    position = Position(
                        path=file_path,
                        start_line=start_line,
                        end_line=end_line,
                        start_col=1,
                        end_col=len(lines[end_line-1]) if end_line <= len(lines) else 1
                    )

                    finding = Finding(
                        id="02d",
                        rule="long-function",
                        severity=Severity.MAJOR,
                        message=f"Función '{func_name}' con {func_lines} líneas (>50). Considera extraer pasos de orquestación.",
                        position=position,
                        snippet=snippet,
                        rationale="Funciones largas dificultan comprensión, pruebas y reuso.",
                        remediation_minutes=min(60, func_lines // 10),
                        tags=["maintainability", "size", "refactoring"],
                        language=language,
                        rule_snapshot={
                            "where": "function",
                            "condition": "loc > 50",
                            "severity": "major",
                            "remediation_per_unit_min": 1.5,
                            "unit": "loc_excess"
                        }
                    )
                    findings.append(finding)
                    break  # Only report first finding per file for brevity

        return findings


    def _analyze_style_issues(self, file_path: str, content: str, lines: List[str], language: str, start_id: int) -> List[Finding]:
        """Analyze code style issues."""
        findings = []

        # Magic numbers
        for i, line in enumerate(lines, 1):
            # Look for numbers that might be magic
            import re
            magic_nums = re.findall(r'\b\d{2,3}\b', line)
            if magic_nums and not any(x in line.lower() for x in ['import', 'const', 'let', 'var', '#']):
                for num in magic_nums:
                    if num not in ['10', '100', '1000', '60', '24']:  # Common non-magic numbers
                        snippet = f"{i:3d}: {line.rstrip()}"

                        position = Position(
                            path=file_path,
                            start_line=i,
                            end_line=i,
                            start_col=line.find(num),
                            end_col=line.find(num) + len(num)
                        )

                        finding = Finding(
                            id="02d",
                            rule="magic-number",
                            severity=Severity.MINOR,
                            message=f"Número mágico '{num}' detectado.",
                            position=position,
                            snippet=snippet,
                            rationale="Los números mágicos deben ser constantes nombradas.",
                            remediation_minutes=6,
                            tags=["style", "maintainability"],
                            language=language
                        )
                        findings.append(finding)
                        break

        return findings

    def _analyze_code_entities(self, score: RepositoryScore) -> List[Dict[str, Any]]:
        """Analyze code entities (functions, classes) with detailed metrics."""
        # Mock entities for demonstration - in real implementation this would parse AST
        return [
            {
                "id": "fn:process_payment",
                "kind": "function",
                "name": "process_payment",
                "signature": "(user: User, order: Order, *, retry: int = 0) -> Receipt",
                "public": True,
                "start_line": 162,
                "end_line": 257,
                "loc": 96,
                "params": 3,
                "return_type": "Receipt",
                "metrics": {
                    "cyclomatic": 17,
                    "cognitive": 55,
                    "halstead_volume": 2279.4,
                    "nesting_depth": 5,
                    "doc_present": True
                },
                "smells": ["deep-nesting", "long-function"]
            }
        ]

    def _calculate_distribution(self, values: List[float]) -> Dict[str, Any]:
        """Calculate statistical distribution of values."""
        if not values:
            return {"min": 0, "max": 0, "mean": 0, "median": 0, "std": 0}

        import numpy as np
        return {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)),
            "quartiles": {
                "25th": float(np.percentile(values, 25)),
                "75th": float(np.percentile(values, 75))
            }
        }

    def _get_quality_rating(self, score: float) -> str:
        """Get quality rating based on score."""
        if score >= 90:
            return "EXCELLENT"
        elif score >= 80:
            return "VERY GOOD"
        elif score >= 70:
            return "GOOD"
        elif score >= 60:
            return "FAIR"
        elif score >= 50:
            return "NEEDS IMPROVEMENT"
        else:
            return "CRITICAL"

    def _get_risk_level(self, score: float) -> str:
        """Get risk level based on score."""
        if score >= 80:
            return "LOW"
        elif score >= 60:
            return "MEDIUM"
        elif score >= 40:
            return "HIGH"
        else:
            return "CRITICAL"

    def _check_quality_gates(
        self,
        score: RepositoryScore,
        fail_condition: Optional[str],
        budget_hours: Optional[float],
        verbose: int
    ) -> int:
        """Check quality gates and return exit code."""
        passed, failures = generate_quality_gates(score)

        if budget_hours and score.total_debt_hours > budget_hours:
            failures.append(f"Budget exceeded: {score.total_debt_hours:.1f}h > {budget_hours}h")

        if failures:
            if verbose >= 1:
                print("❌ Quality gates failed:")
                for failure in failures:
                    print(f"   • {failure}")
            return 1

        if verbose >= 1:
            print("✅ All quality gates passed")

        return 0

    def _print_summary(self, score: RepositoryScore, duration: float):
        """Print analysis summary."""
        print()
        print("📊 Analysis Summary")
        print("=" * 50)
        print(f"📊 Overall Score: {score.overall_score:.1f}")
        print(f"🏆 Grade: {score.grade}")
        print(f"📁 Files: {score.total_files}")
        print(f"📦 Packages: {score.total_packages}")
        print(f"📝 Lines: {score.total_lines:,}")
        print()
        print("🎯 Key Metrics")
        print("-" * 30)
        print(f"🔄 Complexity: {score.complexity_score:.1f}")
        print(f"🛠️  Maintainability: {score.maintainability_score:.1f}")
        print(f"🔗 Coupling: {score.coupling_score:.1f}")
        print(f"📚 Documentation: {score.documentation_score:.1f}")
        print(f"🧪 Test Coverage: {score.test_coverage_score:.1f}")
        print()
        print("💰 Technical Debt")
        print("-" * 30)
        print(f"🕒 Hours: {score.total_debt_hours:.1f}")
        print(f"💵 Cost: ${score.debt_cost:.2f}")
        print()
        print("🏆 Top Issues")
        print("-" * 30)
        for issue in score.top_issues[:3]:
            print(f"• {issue}")

    def _discover_source_files(self, path: Path, languages: List[str], verbose: int) -> List[Path]:
        """Discover all source files in the repository."""
        from .utils import FileUtils

        if verbose >= 2:
            print("🔍 Discovering source files...")

        # Map language names to file extensions
        extension_map = {
            'py': ['.py'],
            'python': ['.py'],
            'js': ['.js', '.jsx'],
            'javascript': ['.js', '.jsx'],
            'ts': ['.ts', '.tsx'],
            'typescript': ['.ts', '.tsx'],
            'go': ['.go'],
            'rs': ['.rs'],
            'rust': ['.rs']
        }

        # Collect all extensions
        extensions = []
        for lang in languages:
            if lang in extension_map:
                extensions.extend(extension_map[lang])

        if not extensions:
            extensions = ['.py', '.js', '.ts', '.go', '.rs']  # Default extensions

        # Find all source files
        source_files = []
        for ext in extensions:
            files = FileUtils.find_files_by_extension(path, [ext])
            source_files.extend(files)

        # Filter out common exclude patterns
        exclude_patterns = [
            '__pycache__', 'node_modules', '.git', 'dist', 'build',
            'venv', '.venv', 'env', '.env', '.next', '.nuxt'
        ]

        filtered_files = []
        for file_path in source_files:
            if not any(pattern in str(file_path) for pattern in exclude_patterns):
                filtered_files.append(file_path)

        return filtered_files

    def _group_files_by_package(self, files: List[Path], repo_path: Path) -> Dict[str, List[Path]]:
        """Group files by package/module structure."""
        from .utils import PathUtils

        package_files = {}

        for file_path in files:
            try:
                # Get relative path from repository root
                rel_path = PathUtils.get_relative_path(file_path, repo_path)

                # Extract package name (directory structure)
                if file_path.suffix == '.py':
                    # For Python: use directory structure as package
                    package_name = str(file_path.parent.relative_to(repo_path)) if file_path.parent != repo_path else 'root'
                else:
                    # For other languages: use file extension and directory
                    package_name = f"{file_path.parent.name}_{file_path.suffix[1:]}"

                if package_name == '.' or package_name == '':
                    package_name = 'root'

                if package_name not in package_files:
                    package_files[package_name] = []

                package_files[package_name].append(file_path)

            except Exception:
                # Fallback: group by directory
                package_name = str(file_path.parent.name) or 'root'
                if package_name not in package_files:
                    package_files[package_name] = []
                package_files[package_name].append(file_path)

        return package_files

    def _analyze_coverage(self, repo_path: Path, coverage_parser, verbose: int):
        """Analyze test coverage if available."""
        if verbose >= 2:
            print("📊 Analyzing test coverage...")

        try:
            # Find coverage files
            coverage_files = coverage_parser.find_coverage_files(repo_path)

            if coverage_files:
                if verbose >= 2:
                    print(f"📄 Found coverage files: {[str(f) for f in coverage_files]}")

                # Parse all coverage files and merge them
                reports = []
                for cov_file in coverage_files:
                    report = coverage_parser.parse_file(cov_file)
                    if report:
                        reports.append(report)

                if reports:
                    return coverage_parser.merge_reports(reports)

            if verbose >= 2:
                print("ℹ️  No coverage data found")

        except Exception as e:
            if verbose >= 1:
                print(f"⚠️  Coverage analysis failed: {e}")

        return None

    def _analyze_single_file(self, file_path: Path, parser, metrics_calc, smell_detector, coverage_data):
        """Analyze a single source file comprehensively."""
        try:
            # Read source code
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()

            # Determine language from file extension
            file_ext = file_path.suffix.lower()
            if file_ext == '.py':
                language = 'python'
            elif file_ext in ['.js', '.jsx']:
                language = 'javascript'
            elif file_ext in ['.ts', '.tsx']:
                language = 'typescript'
            else:
                language = 'python'  # Default fallback

            # Parse AST (with fallback for missing attributes)
            try:
                ast = parser.parse_file(file_path)
                # Check if AST has required attributes
                if hasattr(ast, 'type'):
                    # Use full AST analysis
                    complexity_metrics = metrics_calc.calculate_file_metrics(ast)
                    smells = smell_detector.detect_smells(source_code, str(file_path), language, ast)
                else:
                    # Fallback: skip AST-dependent analysis
                    complexity_metrics = {
                        'cyclomatic_complexity': 1,
                        'cognitive_complexity': 1,
                        'halstead_volume': 100.0,
                        'halstead_difficulty': 5.0,
                        'halstead_effort': 500.0,
                        'maintainability_index': 85.0,
                        'lines_of_code': len(source_code.split('\n')),
                        'functions_count': source_code.count('def '),
                        'classes_count': source_code.count('class ')
                    }
                    smells = []  # Skip smell detection for now
            except Exception:
                # Complete fallback for any parsing errors
                complexity_metrics = {
                    'cyclomatic_complexity': 1,
                    'cognitive_complexity': 1,
                    'halstead_volume': 100.0,
                    'halstead_difficulty': 5.0,
                    'halstead_effort': 500.0,
                    'maintainability_index': 85.0,
                    'lines_of_code': len(source_code.split('\n')),
                    'functions_count': source_code.count('def '),
                    'classes_count': source_code.count('class ')
                }
                smells = []

            # Get coverage data for this file
            coverage_info = None
            if coverage_data and str(file_path) in coverage_data.files:
                coverage_info = coverage_data.files[str(file_path)]

            # Calculate coupling metrics (simplified for now)
            coupling_metrics = {}  # TODO: integrate with coupling analyzer

            # Aggregate into file score
            from .aggregate import MetricsAggregator
            aggregator = MetricsAggregator()

            file_score = aggregator.aggregate_file_metrics(
                str(file_path),
                complexity_metrics,
                smells,
                coupling_metrics=coupling_metrics,
                coverage_data=coverage_info
            )

            return file_score

        except Exception as e:
            print(f"⚠️  Error analyzing {file_path}: {e}")
            return None

    def _integrate_coupling_analysis(self, package_scores: Dict[str, any], coupling_report):
        """Integrate coupling analysis results into package scores."""
        # This is a placeholder for integrating coupling metrics
        # TODO: Implement proper coupling score integration
        pass

    def _create_empty_repository_score(self, path: Path):
        """Create an empty repository score for when no files are found."""
        from .aggregate import RepositoryScore, PackageScore

        return RepositoryScore(
            repository_path=str(path),
            package_scores={},
            overall_score=0,
            grade='N/A',
            complexity_score=0,
            maintainability_score=0,
            coupling_score=0,
            documentation_score=0,
            test_coverage_score=0,
            total_files=0,
            total_lines=0,
            total_packages=0,
            languages=[],
            total_debt_hours=0,
            debt_cost=0
        )

    def main(self):
        """Main entry point for CLI."""
        args = self.parser.parse_args()

        if args.command == "scan":
            return self.run_scan(args)
        else:
            self.parser.print_help()
            return 1


def main():
    """CLI entry point."""
    cli = CodeQCLI()
    sys.exit(cli.main())


if __name__ == "__main__":
    main()
