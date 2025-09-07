"""
Intelligent Recommendations Engine for CodeQ

Provides AI-powered recommendations for code quality improvement based on
analysis results, patterns, and best practices.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import re

from .aggregate import FileScore, PackageScore, RepositoryScore


class RecommendationPriority(Enum):
    """Priority levels for recommendations."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class RecommendationCategory(Enum):
    """Categories of recommendations."""
    REFACTORING = "refactoring"
    ARCHITECTURE = "architecture"
    TESTING = "testing"
    PERFORMANCE = "performance"
    MAINTAINABILITY = "maintainability"
    DOCUMENTATION = "documentation"


@dataclass
class Recommendation:
    """A specific recommendation for code improvement."""
    title: str
    description: str
    category: RecommendationCategory
    priority: RecommendationPriority
    file_path: str
    line_number: Optional[int]
    confidence: float  # 0.0 to 1.0
    impact: str  # "high", "medium", "low"
    effort: str  # "high", "medium", "low"
    tags: List[str]
    code_example: Optional[str] = None
    related_issues: List[str] = None

    def __post_init__(self):
        if self.related_issues is None:
            self.related_issues = []


@dataclass
class RecommendationSet:
    """A collection of recommendations with metadata."""
    recommendations: List[Recommendation]
    total_impact: str
    estimated_effort_days: float
    categories: Dict[str, int]
    priorities: Dict[str, int]

    @property
    def critical_count(self) -> int:
        """Number of critical recommendations."""
        return self.priorities.get("critical", 0)

    @property
    def high_impact_count(self) -> int:
        """Number of high impact recommendations."""
        return len([r for r in self.recommendations if r.impact == "high"])


class RecommendationEngine:
    """
    Intelligent recommendation engine that analyzes code quality metrics
    and provides actionable improvement suggestions.
    """

    def __init__(self):
        self.patterns = self._load_recommendation_patterns()

    def _load_recommendation_patterns(self) -> Dict[str, Any]:
        """Load recommendation patterns and templates."""
        return {
            "complexity": {
                "cyclomatic_high": {
                    "title": "High Cyclomatic Complexity",
                    "template": "Function '{function}' has cyclomatic complexity of {value}. Consider breaking it into smaller functions.",
                    "priority": RecommendationPriority.HIGH,
                    "category": RecommendationCategory.REFACTORING,
                    "effort": "medium"
                },
                "cognitive_high": {
                    "title": "High Cognitive Complexity",
                    "template": "Function '{function}' has high cognitive complexity. Consider simplifying the control flow.",
                    "priority": RecommendationPriority.MEDIUM,
                    "category": RecommendationCategory.REFACTORING,
                    "effort": "medium"
                }
            },
            "maintainability": {
                "low_mi": {
                    "title": "Low Maintainability Index",
                    "template": "File has low maintainability index ({value}). Consider refactoring to improve code quality.",
                    "priority": RecommendationPriority.HIGH,
                    "category": RecommendationCategory.MAINTAINABILITY,
                    "effort": "high"
                }
            },
            "coupling": {
                "high_fan_out": {
                    "title": "High Fan-Out Coupling",
                    "template": "Class '{class_name}' has high fan-out ({value}). Consider reducing dependencies.",
                    "priority": RecommendationPriority.MEDIUM,
                    "category": RecommendationCategory.ARCHITECTURE,
                    "effort": "high"
                }
            },
            "testing": {
                "low_coverage": {
                    "title": "Low Test Coverage",
                    "template": "Test coverage is {value}%. Aim for at least 80% coverage.",
                    "priority": RecommendationPriority.HIGH,
                    "category": RecommendationCategory.TESTING,
                    "effort": "high"
                }
            }
        }

    def analyze_repository(self, repository_score: RepositoryScore) -> RecommendationSet:
        """
        Analyze repository and generate comprehensive recommendations.

        Args:
            repository_score: Complete repository analysis results

        Returns:
            Set of prioritized recommendations
        """
        recommendations = []

        # Analyze each package
        for package_name, package_score in repository_score.package_scores.items():
            package_recs = self._analyze_package(package_name, package_score)
            recommendations.extend(package_recs)

        # Add repository-level recommendations
        repo_recs = self._analyze_repository_level(repository_score)
        recommendations.extend(repo_recs)

        # Sort by priority and impact
        recommendations.sort(key=self._recommendation_sort_key, reverse=True)

        # Calculate metadata
        categories = {}
        priorities = {}

        for rec in recommendations:
            categories[rec.category.value] = categories.get(rec.category.value, 0) + 1
            priorities[rec.priority.value] = priorities.get(rec.priority.value, 0) + 1

        total_impact = self._calculate_total_impact(recommendations)
        estimated_effort = self._estimate_effort_days(recommendations)

        return RecommendationSet(
            recommendations=recommendations,
            total_impact=total_impact,
            estimated_effort_days=estimated_effort,
            categories=categories,
            priorities=priorities
        )

    def _analyze_package(self, package_name: str, package_score: PackageScore) -> List[Recommendation]:
        """Analyze a package and generate recommendations."""
        recommendations = []

        # Analyze complexity issues
        if package_score.avg_complexity > 50:
            rec = Recommendation(
                title="Package Complexity Issues",
                description=f"Package '{package_name}' has average complexity of {package_score.avg_complexity:.1f}. "
                           "Consider breaking down complex modules.",
                category=RecommendationCategory.ARCHITECTURE,
                priority=RecommendationPriority.HIGH,
                file_path=package_name,
                line_number=None,
                confidence=0.8,
                impact="high",
                effort="high",
                tags=["complexity", "architecture", "refactoring"]
            )
            recommendations.append(rec)

        # Analyze worst files
        for file_score in package_score.worst_files[:3]:  # Top 3 worst files
            file_recs = self._analyze_file(file_score)
            recommendations.extend(file_recs)

        return recommendations

    def _analyze_file(self, file_score: FileScore) -> List[Recommendation]:
        """Analyze a file and generate recommendations."""
        recommendations = []
        file_path = file_score.file_path

        # Complexity recommendations
        if file_score.complexity_score > 70:
            rec = Recommendation(
                title="High Complexity Function(s)",
                description=f"File '{file_path}' has high complexity ({file_score.complexity_score:.1f}). "
                           "Consider refactoring complex functions.",
                category=RecommendationCategory.REFACTORING,
                priority=RecommendationPriority.HIGH,
                file_path=file_path,
                line_number=None,
                confidence=0.9,
                impact="high",
                effort="medium",
                tags=["complexity", "refactoring"]
            )
            recommendations.append(rec)

        # Maintainability recommendations
        if file_score.maintainability_score < 40:
            rec = Recommendation(
                title="Low Maintainability",
                description=f"File '{file_path}' has low maintainability ({file_score.maintainability_score:.1f}). "
                           "Consider improving code structure and documentation.",
                category=RecommendationCategory.MAINTAINABILITY,
                priority=RecommendationPriority.MEDIUM,
                file_path=file_path,
                line_number=None,
                confidence=0.8,
                impact="medium",
                effort="high",
                tags=["maintainability", "documentation"]
            )
            recommendations.append(rec)

        # Documentation recommendations
        if file_score.documentation_score < 50:
            rec = Recommendation(
                title="Missing Documentation",
                description=f"File '{file_path}' lacks proper documentation ({file_score.documentation_score:.1f}). "
                           "Add docstrings and comments.",
                category=RecommendationCategory.DOCUMENTATION,
                priority=RecommendationPriority.MEDIUM,
                file_path=file_path,
                line_number=None,
                confidence=0.7,
                impact="medium",
                effort="low",
                tags=["documentation"]
            )
            recommendations.append(rec)

        # Test coverage recommendations
        if file_score.test_coverage_score < 60:
            rec = Recommendation(
                title="Low Test Coverage",
                description=f"File '{file_path}' has low test coverage ({file_score.test_coverage_score:.1f}%). "
                           "Add comprehensive unit tests.",
                category=RecommendationCategory.TESTING,
                priority=RecommendationPriority.HIGH,
                file_path=file_path,
                line_number=None,
                confidence=0.8,
                impact="high",
                effort="high",
                tags=["testing", "coverage"]
            )
            recommendations.append(rec)

        return recommendations

    def _analyze_repository_level(self, repository_score: RepositoryScore) -> List[Recommendation]:
        """Generate repository-level recommendations."""
        recommendations = []

        # Overall quality assessment
        if repository_score.overall_score < 60:
            rec = Recommendation(
                title="Overall Code Quality Issues",
                description=f"Repository has low overall quality score ({repository_score.overall_score:.1f}). "
                           "Implement systematic refactoring plan.",
                category=RecommendationCategory.ARCHITECTURE,
                priority=RecommendationPriority.CRITICAL,
                file_path="",
                line_number=None,
                confidence=0.95,
                impact="critical",
                effort="high",
                tags=["quality", "refactoring", "architecture"]
            )
            recommendations.append(rec)

        # Technical debt assessment
        if repository_score.total_debt_hours > 200:
            rec = Recommendation(
                title="High Technical Debt",
                description=f"Repository has {repository_score.total_debt_hours:.0f} hours of technical debt "
                           f"(€{repository_score.debt_cost:,.0f}). Prioritize debt reduction.",
                category=RecommendationCategory.MAINTAINABILITY,
                priority=RecommendationPriority.CRITICAL,
                file_path="",
                line_number=None,
                confidence=0.9,
                impact="critical",
                effort="high",
                tags=["technical-debt", "cost", "maintenance"]
            )
            recommendations.append(rec)


        # Architecture recommendations
        if repository_score.coupling_score > 70:
            rec = Recommendation(
                title="Tight Coupling Issues",
                description=f"Repository has high coupling ({repository_score.coupling_score:.1f}). "
                           "Consider dependency injection and interface segregation.",
                category=RecommendationCategory.ARCHITECTURE,
                priority=RecommendationPriority.HIGH,
                file_path="",
                line_number=None,
                confidence=0.8,
                impact="high",
                effort="high",
                tags=["coupling", "architecture", "design-patterns"]
            )
            recommendations.append(rec)

        return recommendations

    def _recommendation_sort_key(self, rec: Recommendation) -> Tuple[int, int, float]:
        """Sort key for recommendations (priority, impact, confidence)."""
        priority_order = {
            RecommendationPriority.CRITICAL: 5,
            RecommendationPriority.HIGH: 4,
            RecommendationPriority.MEDIUM: 3,
            RecommendationPriority.LOW: 2,
            RecommendationPriority.INFO: 1
        }

        impact_order = {
            "critical": 3,
            "high": 2,
            "medium": 1,
            "low": 0
        }

        return (
            priority_order[rec.priority],
            impact_order.get(rec.impact, 0),
            rec.confidence
        )

    def _calculate_total_impact(self, recommendations: List[Recommendation]) -> str:
        """Calculate overall impact of recommendations."""
        if not recommendations:
            return "none"

        critical_count = sum(1 for r in recommendations if r.priority == RecommendationPriority.CRITICAL)
        high_count = sum(1 for r in recommendations if r.priority == RecommendationPriority.HIGH)

        if critical_count > 0:
            return "critical"
        elif high_count > 5:
            return "high"
        elif high_count > 0:
            return "medium"
        else:
            return "low"

    def _estimate_effort_days(self, recommendations: List[Recommendation]) -> float:
        """Estimate total effort in days for implementing recommendations."""
        effort_days = {
            "low": 0.5,
            "medium": 2.0,
            "high": 5.0
        }

        total_days = 0.0
        for rec in recommendations:
            days = effort_days.get(rec.effort, 1.0)
            # Weight by priority
            if rec.priority == RecommendationPriority.CRITICAL:
                days *= 1.5
            elif rec.priority == RecommendationPriority.HIGH:
                days *= 1.2

            total_days += days

        return round(total_days, 1)

    def generate_action_plan(self, recommendations: List[Recommendation]) -> Dict[str, Any]:
        """Generate a structured action plan from recommendations."""
        # Group by priority
        plan = {
            "immediate_actions": [],  # Critical
            "short_term": [],         # High
            "medium_term": [],        # Medium
            "long_term": [],          # Low
            "monitoring": []          # Info
        }

        for rec in recommendations:
            action_item = {
                "title": rec.title,
                "description": rec.description,
                "file": rec.file_path,
                "effort": rec.effort,
                "impact": rec.impact,
                "tags": rec.tags
            }

            if rec.priority == RecommendationPriority.CRITICAL:
                plan["immediate_actions"].append(action_item)
            elif rec.priority == RecommendationPriority.HIGH:
                plan["short_term"].append(action_item)
            elif rec.priority == RecommendationPriority.MEDIUM:
                plan["medium_term"].append(action_item)
            elif rec.priority == RecommendationPriority.LOW:
                plan["long_term"].append(action_item)
            else:
                plan["monitoring"].append(action_item)

        return plan

    def export_recommendations(self, recommendation_set: RecommendationSet, format: str = "json") -> str:
        """Export recommendations in various formats."""
        data = {
            "metadata": {
                "total_recommendations": len(recommendation_set.recommendations),
                "total_impact": recommendation_set.total_impact,
                "estimated_effort_days": recommendation_set.estimated_effort_days,
                "critical_count": recommendation_set.critical_count,
                "high_impact_count": recommendation_set.high_impact_count
            },
            "categories": recommendation_set.categories,
            "priorities": recommendation_set.priorities,
            "recommendations": [
                {
                    "title": r.title,
                    "description": r.description,
                    "category": r.category.value,
                    "priority": r.priority.value,
                    "file_path": r.file_path,
                    "line_number": r.line_number,
                    "confidence": r.confidence,
                    "impact": r.impact,
                    "effort": r.effort,
                    "tags": r.tags
                }
                for r in recommendation_set.recommendations
            ],
            "action_plan": self.generate_action_plan(recommendation_set.recommendations)
        }

        if format == "json":
            return json.dumps(data, indent=2, ensure_ascii=False)
        else:
            # Simple text format
            output = []
            output.append(f"CodeQ Recommendations Report")
            output.append("=" * 50)
            output.append(f"Total Recommendations: {len(recommendation_set.recommendations)}")
            output.append(f"Total Impact: {recommendation_set.total_impact}")
            output.append(f"Estimated Effort: {recommendation_set.estimated_effort_days} days")
            output.append("")

            for rec in recommendation_set.recommendations:
                output.append(f"[{rec.priority.value.upper()}] {rec.title}")
                output.append(f"File: {rec.file_path}")
                output.append(f"Impact: {rec.impact}, Effort: {rec.effort}")
                output.append(f"Description: {rec.description}")
                output.append("")

            return "\n".join(output)
