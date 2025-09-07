"""
Aggregation and scoring module for CodeQ.
Combines metrics from different analyzers into unified scores.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from pathlib import Path
import statistics
import math


class AggregationLevel(Enum):
    FILE = "file"
    PACKAGE = "package"
    MODULE = "module"
    REPOSITORY = "repository"


@dataclass
class FileScore:
    """Aggregated scores for a single file."""
    file_path: str
    
    # Core metrics
    complexity_score: float  # 0-100, lower is better
    maintainability_score: float  # 0-100, higher is better
    coupling_score: float  # 0-100, lower is better
    documentation_score: float  # 0-100, higher is better
    test_coverage_score: float  # 0-100, higher is better
    
    # Aggregated score
    overall_score: float  # 0-100, weighted average
    grade: str  # A, B, C, D, F
    
    # Raw metrics for reference
    raw_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Issues found
    smell_count: int = 0
    
    @property
    def needs_attention(self) -> bool:
        """File needs immediate attention if grade is D or F."""
        return self.grade in ['D', 'F']
    
    @property
    def technical_debt_hours(self) -> float:
        """Estimated hours to fix all issues."""
        return self.raw_metrics.get('tech_debt_hours', 0.0)


@dataclass
class PackageScore:
    """Aggregated scores for a package/directory."""
    package_path: str
    file_scores: List[FileScore]
    
    # Aggregated metrics
    avg_complexity: float = 0.0
    avg_maintainability: float = 0.0
    avg_coupling: float = 0.0
    avg_documentation: float = 0.0
    avg_test_coverage: float = 0.0
    
    overall_score: float = 0.0
    grade: str = "C"
    
    # Statistics
    total_files: int = 0
    total_lines: int = 0
    total_functions: int = 0
    total_classes: int = 0
    
    @property
    def worst_files(self) -> List[FileScore]:
        """Get files with lowest scores."""
        return sorted(self.file_scores, key=lambda f: f.overall_score)[:5]
    
    @property
    def total_technical_debt(self) -> float:
        """Total technical debt for package."""
        return sum(f.technical_debt_hours for f in self.file_scores)


@dataclass
class RepositoryScore:
    """Top-level aggregated scores for entire repository."""
    repository_path: str
    package_scores: Dict[str, PackageScore]
    
    # Overall metrics
    overall_score: float
    grade: str
    
    # Breakdown by category
    complexity_score: float
    maintainability_score: float
    coupling_score: float
    documentation_score: float
    test_coverage_score: float
    
    # Statistics
    total_files: int
    total_lines: int
    total_packages: int

    # Technical debt
    total_debt_hours: float
    debt_cost: float  # In configured currency

    # Languages
    languages: Optional[List[str]] = None
    
    # Trends (if historical data available)
    score_trend: Optional[str] = None  # "improving", "declining", "stable"
    
    @property
    def health_status(self) -> str:
        """Overall health status of repository."""
        if self.grade == 'A':
            return "Excellent"
        elif self.grade == 'B':
            return "Good"
        elif self.grade == 'C':
            return "Fair"
        elif self.grade == 'D':
            return "Poor"
        else:
            return "Critical"
    
    @property
    def top_issues(self) -> List[str]:
        """Identify top issues to address."""
        issues = []
        
        if self.complexity_score > 70:
            issues.append("High complexity")
        if self.maintainability_score < 40:
            issues.append("Poor maintainability")
        if self.coupling_score > 70:
            issues.append("High coupling")
        if self.documentation_score < 40:
            issues.append("Poor documentation")
        if self.test_coverage_score < 60:
            issues.append("Low test coverage")
            
        return issues


@dataclass
class ScoringWeights:
    """Configurable weights for score calculation."""
    complexity: float = 0.20
    maintainability: float = 0.25
    coupling: float = 0.15
    documentation: float = 0.125
    test_coverage: float = 0.125
    
    def validate(self) -> bool:
        """Ensure weights sum to 1.0."""
        total = (
            self.complexity + self.maintainability +
            self.coupling + self.documentation + self.test_coverage +
        )
        return abs(total - 1.0) < 0.001
    
    def normalize(self):
        """Normalize weights to sum to 1.0."""
        total = (
            self.complexity + self.maintainability +
            self.coupling + self.documentation + self.test_coverage +
        )
        if total > 0:
            self.complexity /= total
            self.maintainability /= total
            self.coupling /= total
            self.documentation /= total
            self.test_coverage /= total
 /= total


class MetricsAggregator:
    """Aggregates metrics from different analyzers into unified scores."""
    
    def __init__(
        self,
        weights: Optional[ScoringWeights] = None,
        grading_thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Initialize aggregator with scoring configuration.
        
        Args:
            weights: Custom scoring weights
            grading_thresholds: Grade boundaries (A, B, C, D, F)
        """
        self.weights = weights or ScoringWeights()
        self.weights.normalize()
        
        self.grading_thresholds = grading_thresholds or {
            'A': 85,
            'B': 70,
            'C': 55,
            'D': 40,
            'F': 0
        }
        
    def aggregate_file_metrics(
        self,
        file_path: str,
        complexity_metrics: Dict[str, Any],
        smell_metrics: List[Any],
        coupling_metrics: Optional[Dict[str, Any]] = None,
        coverage_data: Optional[Dict[str, Any]] = None
    ) -> FileScore:
        """
        Aggregate metrics for a single file.

        Args:
            file_path: Path to file
            complexity_metrics: From metrics.py
            smell_metrics: From smells.py
            coupling_metrics: From coupling.py
            coverage_data: Test coverage information

        Returns:
            FileScore with aggregated metrics
        """
        # Calculate individual scores
        complexity_score = self._calculate_complexity_score(complexity_metrics)
        maintainability_score = self._calculate_maintainability_score(complexity_metrics)
        coupling_score = self._calculate_coupling_score(coupling_metrics)
        documentation_score = self._calculate_documentation_score(complexity_metrics)
        test_coverage_score = self._calculate_coverage_score(coverage_data)
        
        # Calculate overall weighted score
        overall_score = (
            self.weights.complexity * (100 - complexity_score) +  # Invert: lower is better
            self.weights.maintainability * maintainability_score +
            self.weights.coupling * (100 - coupling_score) +  # Invert
            self.weights.documentation * documentation_score +
            self.weights.test_coverage * test_coverage_score
        )
        
        # Determine grade
        grade = self._calculate_grade(overall_score)
        
        # Count issues
        smell_count = len(smell_metrics)
        
        return FileScore(
            file_path=file_path,
            complexity_score=complexity_score,
            maintainability_score=maintainability_score,
            coupling_score=coupling_score,
            documentation_score=documentation_score,
            test_coverage_score=test_coverage_score,
            overall_score=overall_score,
            grade=grade,
            raw_metrics={
                'complexity': complexity_metrics,
                'smells': smell_metrics,
                'coupling': coupling_metrics,
                'coverage': coverage_data
            },
            smell_count=smell_count,
        )
    
    def _calculate_complexity_score(self, metrics: Dict[str, Any]) -> float:
        """
        Calculate complexity score (0-100, lower is better).
        
        Args:
            metrics: Complexity metrics dictionary
            
        Returns:
            Score where 0 is best, 100 is worst
        """
        if not metrics:
            return 50.0  # Default middle score
            
        cyclomatic = metrics.get('cyclomatic', 1)
        cognitive = metrics.get('cognitive', 1)
        
        # Map complexity to score
        # CC < 5: excellent (0-20)
        # CC 5-10: good (20-40)
        # CC 10-20: fair (40-60)
        # CC 20-50: poor (60-80)
        # CC > 50: critical (80-100)
        
        if cyclomatic < 5:
            cyc_score = cyclomatic * 4
        elif cyclomatic < 10:
            cyc_score = 20 + (cyclomatic - 5) * 4
        elif cyclomatic < 20:
            cyc_score = 40 + (cyclomatic - 10) * 2
        elif cyclomatic < 50:
            cyc_score = 60 + (cyclomatic - 20) * 0.67
        else:
            cyc_score = min(100, 80 + (cyclomatic - 50) * 0.4)
            
        # Similar for cognitive complexity
        if cognitive < 7:
            cog_score = cognitive * 2.86
        elif cognitive < 15:
            cog_score = 20 + (cognitive - 7) * 2.5
        elif cognitive < 25:
            cog_score = 40 + (cognitive - 15) * 2
        else:
            cog_score = min(100, 60 + (cognitive - 25) * 1.6)
            
        # Average both scores
        return (cyc_score + cog_score) / 2
    
    def _calculate_maintainability_score(self, metrics: Dict[str, Any]) -> float:
        """
        Calculate maintainability score (0-100, higher is better).
        
        Args:
            metrics: Should contain 'maintainability_index' or components
            
        Returns:
            Score where 100 is best, 0 is worst
        """
        if not metrics:
            return 50.0
            
        # Use MI if available
        mi = metrics.get('maintainability_index')
        if mi is not None:
            # MI is already 0-100, just ensure bounds
            return max(0, min(100, mi))
            
        # Otherwise estimate from other metrics
        loc = metrics.get('loc', 0)
        cyclomatic = metrics.get('cyclomatic', 1)
        
        # Simple estimation
        if loc < 50 and cyclomatic < 10:
            return 85
        elif loc < 200 and cyclomatic < 20:
            return 65
        else:
            return 40
    
    
    def _calculate_coupling_score(self, metrics: Optional[Dict[str, Any]]) -> float:
        """
        Calculate coupling score (0-100, lower is better).
        
        Args:
            metrics: Coupling metrics
            
        Returns:
            Score where 0 is no coupling, 100 is excessive coupling
        """
        if not metrics:
            return 30.0  # Default moderate coupling
            
        fan_out = metrics.get('efferent_coupling', 0)
        instability = metrics.get('instability', 0.5)
        
        # Map fan-out to score
        if fan_out < 5:
            fan_score = fan_out * 4
        elif fan_out < 20:
            fan_score = 20 + (fan_out - 5) * 2.67
        else:
            fan_score = min(100, 60 + (fan_out - 20) * 2)
            
        # Factor in instability
        instability_score = instability * 100
        
        # Average both
        return (fan_score + instability_score) / 2
    
    def _calculate_documentation_score(self, metrics: Dict[str, Any]) -> float:
        """
        Calculate documentation score (0-100, higher is better).
        
        Args:
            metrics: Should contain comment/doc ratios
            
        Returns:
            Score where 100 is well-documented, 0 is no documentation
        """
        if not metrics:
            return 50.0
            
        comment_lines = metrics.get('comments', 0)
        total_lines = metrics.get('loc', 1)
        
        comment_ratio = comment_lines / total_lines
        
        # Ideal range is 10-30% comments
        if comment_ratio < 0.1:
            # Too few comments
            score = comment_ratio * 500  # 0-50 points
        elif comment_ratio <= 0.3:
            # Ideal range
            score = 50 + ((comment_ratio - 0.1) / 0.2) * 50  # 50-100 points
        else:
            # Too many comments (might indicate complex code)
            score = max(60, 100 - (comment_ratio - 0.3) * 100)
            
        return score
    
    def _calculate_coverage_score(self, coverage_data: Optional[Dict[str, Any]]) -> float:
        """
        Calculate test coverage score (0-100, higher is better).
        
        Args:
            coverage_data: Test coverage information
            
        Returns:
            Score equal to coverage percentage
        """
        if not coverage_data:
            return 0.0  # No coverage data means no tests
            
        line_coverage = coverage_data.get('line_coverage', 0)
        branch_coverage = coverage_data.get('branch_coverage', 0)
        
        # Weight line coverage more heavily
        return line_coverage * 0.7 + branch_coverage * 0.3
    
    
    def _calculate_grade(self, score: float) -> str:
        """
        Convert numeric score to letter grade.
        
        Args:
            score: Numeric score 0-100
            
        Returns:
            Letter grade (A, B, C, D, F)
        """
        if score >= self.grading_thresholds['A']:
            return 'A'
        elif score >= self.grading_thresholds['B']:
            return 'B'
        elif score >= self.grading_thresholds['C']:
            return 'C'
        elif score >= self.grading_thresholds['D']:
            return 'D'
        else:
            return 'F'
    
    def aggregate_package_metrics(
        self,
        package_path: str,
        file_scores: List[FileScore]
    ) -> PackageScore:
        """
        Aggregate metrics for a package/directory.
        
        Args:
            package_path: Path to package
            file_scores: List of file scores in package
            
        Returns:
            PackageScore with aggregated metrics
        """
        if not file_scores:
            return PackageScore(
                package_path=package_path,
                file_scores=[],
                grade='N/A'
            )
            
        # Calculate averages
        avg_complexity = statistics.mean(f.complexity_score for f in file_scores)
        avg_maintainability = statistics.mean(f.maintainability_score for f in file_scores)
        avg_coupling = statistics.mean(f.coupling_score for f in file_scores)
        avg_documentation = statistics.mean(f.documentation_score for f in file_scores)
        avg_test_coverage = statistics.mean(f.test_coverage_score for f in file_scores)
        
        # Calculate overall score
        overall_score = statistics.mean(f.overall_score for f in file_scores)
        grade = self._calculate_grade(overall_score)
        
        # Get statistics from raw metrics
        total_lines = sum(
            f.raw_metrics.get('complexity', {}).get('loc', 0)
            for f in file_scores
        )
        
        return PackageScore(
            package_path=package_path,
            file_scores=file_scores,
            avg_complexity=avg_complexity,
            avg_maintainability=avg_maintainability,
            avg_coupling=avg_coupling,
            avg_documentation=avg_documentation,
            avg_test_coverage=avg_test_coverage,
            overall_score=overall_score,
            grade=grade,
            total_files=len(file_scores),
            total_lines=total_lines
        )
    
    def aggregate_repository_metrics(
        self,
        repository_path: str,
        package_scores: Dict[str, PackageScore],
        total_lines: int = 0,
        detected_languages: List[str] = None,
        hourly_rate: float = 50.0
    ) -> RepositoryScore:
        """
        Aggregate metrics for entire repository.
        
        Args:
            repository_path: Path to repository
            package_scores: Dict of package scores
            hourly_rate: Cost per hour for debt calculation
            
        Returns:
            RepositoryScore with top-level metrics
        """
        if not package_scores:
            return RepositoryScore(
                repository_path=repository_path,
                package_scores={},
                overall_score=0,
                grade='N/A',
                complexity_score=0,
                maintainability_score=0,
                coupling_score=0,
                documentation_score=0,
                test_coverage_score=0,
                total_files=0,
                total_lines=total_lines,
                total_packages=0,
                languages=detected_languages or [],
                total_debt_hours=0,
                debt_cost=0
            )
            
        # Collect all file scores
        all_file_scores = []
        for package in package_scores.values():
            all_file_scores.extend(package.file_scores)
            
        # Calculate weighted averages (by file count)
        total_files = len(all_file_scores)
        
        if total_files == 0:
            return RepositoryScore(
                repository_path=repository_path,
                package_scores=package_scores,
                overall_score=0,
                grade='N/A',
                complexity_score=0,
                maintainability_score=0,
                coupling_score=0,
                documentation_score=0,
                test_coverage_score=0,
                total_files=0,
                total_lines=total_lines,
                languages=detected_languages or [],
                total_packages=len(package_scores),
                total_debt_hours=0,
                debt_cost=0
            )
            
        # Calculate repository-wide scores
        complexity_score = statistics.mean(f.complexity_score for f in all_file_scores)
        maintainability_score = statistics.mean(f.maintainability_score for f in all_file_scores)
        coupling_score = statistics.mean(f.coupling_score for f in all_file_scores)
        documentation_score = statistics.mean(f.documentation_score for f in all_file_scores)
        test_coverage_score = statistics.mean(f.test_coverage_score for f in all_file_scores)
        
        overall_score = statistics.mean(f.overall_score for f in all_file_scores)
        grade = self._calculate_grade(overall_score)
        
        # Use pre-calculated totals
        # total_lines is already calculated globally
        if total_lines == 0:  # Fallback if not calculated
            total_lines = sum(p.total_lines for p in package_scores.values())
        total_debt_hours = sum(p.total_technical_debt for p in package_scores.values())
        debt_cost = total_debt_hours * hourly_rate
        
        return RepositoryScore(
            repository_path=repository_path,
            package_scores=package_scores,
            overall_score=overall_score,
            grade=grade,
            complexity_score=complexity_score,
            maintainability_score=maintainability_score,
            coupling_score=coupling_score,
            documentation_score=documentation_score,
            test_coverage_score=test_coverage_score,
            total_files=total_files,
            total_lines=total_lines,
            total_packages=len(package_scores),
            languages=detected_languages or [],
            total_debt_hours=total_debt_hours,
            debt_cost=debt_cost
        )


def generate_quality_gates(
    score: RepositoryScore,
    gates: Optional[Dict[str, Any]] = None
) -> Tuple[bool, List[str]]:
    """
    Check if repository passes quality gates.
    
    Args:
        score: Repository score
        gates: Custom gate definitions
        
    Returns:
        Tuple of (passed: bool, failures: List[str])
    """
    if gates is None:
        gates = {
            'min_overall_score': 60,
            'min_test_coverage': 60,
            'max_complexity': 50,
            'max_debt_hours': 100
        }
        
    failures = []
    
    if score.overall_score < gates.get('min_overall_score', 0):
        failures.append(
            f"Overall score {score.overall_score:.1f} < {gates['min_overall_score']}"
        )
        
        
    if score.test_coverage_score < gates.get('min_test_coverage', 0):
        failures.append(
            f"Test coverage {score.test_coverage_score:.1f}% < {gates['min_test_coverage']}%"
        )
        
    if score.complexity_score > gates.get('max_complexity', 100):
        failures.append(
            f"Complexity score {score.complexity_score:.1f} > {gates['max_complexity']}"
        )

    if score.total_debt_hours > gates.get('max_debt_hours', float('inf')):
        failures.append(
            f"Technical debt {score.total_debt_hours:.1f}h > {gates['max_debt_hours']}h"
        )
        
    passed = len(failures) == 0
    return passed, failures