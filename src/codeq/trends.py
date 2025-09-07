"""
Trend Analysis Module for CodeQ

Analyzes historical code quality metrics to identify trends, patterns,
and predict future quality issues.
"""

from typing import Dict, List, Any, Optional, Tuple, NamedTuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import json
import statistics
from pathlib import Path

from .aggregate import RepositoryScore


class TrendDirection(Enum):
    """Direction of quality trends."""
    IMPROVING = "improving"
    DECLINING = "declining"
    STABLE = "stable"
    VOLATILE = "volatile"


class QualityMetric(Enum):
    """Available quality metrics for trend analysis."""
    COMPLEXITY = "complexity_score"
    MAINTAINABILITY = "maintainability_score"
    COUPLING = "coupling_score"
    DOCUMENTATION = "documentation_score"
    TEST_COVERAGE = "test_coverage_score"
    OVERALL = "overall_score"
    TECHNICAL_DEBT = "total_debt_hours"


@dataclass
class HistoricalDataPoint:
    """A single historical data point."""
    timestamp: datetime
    metrics: Dict[str, float]
    commit_hash: Optional[str] = None
    branch: str = "main"
    author: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "metrics": self.metrics,
            "commit_hash": self.commit_hash,
            "branch": self.branch,
            "author": self.author
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HistoricalDataPoint':
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metrics=data["metrics"],
            commit_hash=data.get("commit_hash"),
            branch=data.get("branch", "main"),
            author=data.get("author")
        )


@dataclass
class TrendAnalysis:
    """Analysis of quality trends over time."""
    metric: str
    direction: TrendDirection
    confidence: float  # 0.0 to 1.0
    slope: float  # Rate of change per day
    volatility: float  # Standard deviation of changes
    data_points: int
    time_span_days: int
    prediction: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric,
            "direction": self.direction.value,
            "confidence": self.confidence,
            "slope": self.slope,
            "volatility": self.volatility,
            "data_points": self.data_points,
            "time_span_days": self.time_span_days,
            "prediction": self.prediction
        }


@dataclass
class TrendReport:
    """Complete trend analysis report."""
    analyses: Dict[str, TrendAnalysis]
    overall_trend: TrendDirection
    recommendations: List[str]
    alerts: List[str]
    generated_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "analyses": {k: v.to_dict() for k, v in self.analyses.items()},
            "overall_trend": self.overall_trend.value,
            "recommendations": self.recommendations,
            "alerts": self.alerts,
            "generated_at": self.generated_at.isoformat()
        }


class TrendAnalyzer:
    """
    Analyzes historical code quality data to identify trends and patterns.

    Supports forecasting, anomaly detection, and actionable insights.
    """

    def __init__(self, history_file: Optional[Path] = None):
        self.history_file = history_file or Path(".codeq_history.json")
        self.history_data: List[HistoricalDataPoint] = []
        self._load_history()

    def add_data_point(self, repository_score: RepositoryScore,
                      commit_hash: Optional[str] = None,
                      branch: str = "main",
                      author: Optional[str] = None) -> None:
        """
        Add a new data point to the historical record.

        Args:
            repository_score: Current repository analysis results
            commit_hash: Git commit hash
            branch: Git branch name
            author: Commit author
        """
        data_point = HistoricalDataPoint(
            timestamp=datetime.now(),
            metrics={
                "complexity_score": repository_score.complexity_score,
                "maintainability_score": repository_score.maintainability_score,
                "coupling_score": repository_score.coupling_score,
                "documentation_score": repository_score.documentation_score,
                "test_coverage_score": repository_score.test_coverage_score,
                "overall_score": repository_score.overall_score,
                "total_debt_hours": repository_score.total_debt_hours
            },
            commit_hash=commit_hash,
            branch=branch,
            author=author
        )

        self.history_data.append(data_point)
        self._save_history()

    def analyze_trends(self, days: int = 30, min_data_points: int = 5) -> TrendReport:
        """
        Analyze quality trends over the specified time period.

        Args:
            days: Number of days to analyze
            min_data_points: Minimum data points required for analysis

        Returns:
            Complete trend analysis report
        """
        # Filter data for the specified time period
        cutoff_date = datetime.now() - timedelta(days=days)
        relevant_data = [
            dp for dp in self.history_data
            if dp.timestamp >= cutoff_date
        ]

        if len(relevant_data) < min_data_points:
            return self._create_insufficient_data_report()

        analyses = {}
        alerts = []
        recommendations = []

        # Analyze each metric
        for metric in QualityMetric:
            if self._has_metric_data(relevant_data, metric.value):
                analysis = self._analyze_metric_trend(relevant_data, metric.value, days)
                analyses[metric.value] = analysis

                # Generate alerts and recommendations
                metric_alerts, metric_recs = self._generate_metric_insights(analysis, metric.value)
                alerts.extend(metric_alerts)
                recommendations.extend(metric_recs)

        # Determine overall trend
        overall_trend = self._calculate_overall_trend(analyses)

        return TrendReport(
            analyses=analyses,
            overall_trend=overall_trend,
            recommendations=list(set(recommendations)),  # Remove duplicates
            alerts=list(set(alerts)),  # Remove duplicates
            generated_at=datetime.now()
        )

    def _analyze_metric_trend(self, data_points: List[HistoricalDataPoint],
                             metric: str, days: int) -> TrendAnalysis:
        """Analyze trend for a specific metric."""
        # Extract metric values with timestamps
        values = []
        timestamps = []

        for dp in sorted(data_points, key=lambda x: x.timestamp):
            if metric in dp.metrics:
                values.append(dp.metrics[metric])
                timestamps.append(dp.timestamp)

        if len(values) < 2:
            return TrendAnalysis(
                metric=metric,
                direction=TrendDirection.STABLE,
                confidence=0.0,
                slope=0.0,
                volatility=0.0,
                data_points=len(values),
                time_span_days=days
            )

        # Calculate basic statistics
        try:
            # Calculate slope (rate of change)
            slope = self._calculate_slope(timestamps, values)

            # Calculate volatility (standard deviation of changes)
            volatility = self._calculate_volatility(values)

            # Determine trend direction
            direction, confidence = self._determine_trend_direction(slope, volatility, values)

        except Exception:
            # Fallback if calculations fail
            slope = 0.0
            volatility = 0.0
            direction = TrendDirection.STABLE
            confidence = 0.0

        return TrendAnalysis(
            metric=metric,
            direction=direction,
            confidence=confidence,
            slope=slope,
            volatility=volatility,
            data_points=len(values),
            time_span_days=days
        )

    def _calculate_slope(self, timestamps: List[datetime], values: List[float]) -> float:
        """Calculate the slope (rate of change) of the metric over time."""
        if len(timestamps) < 2:
            return 0.0

        # Convert timestamps to days since first measurement
        base_time = timestamps[0]
        days_since_start = [
            (ts - base_time).total_seconds() / (24 * 3600)
            for ts in timestamps
        ]

        # Calculate linear regression slope
        n = len(values)
        sum_x = sum(days_since_start)
        sum_y = sum(values)
        sum_xy = sum(x * y for x, y in zip(days_since_start, values))
        sum_x2 = sum(x * x for x in days_since_start)

        denominator = n * sum_x2 - sum_x * sum_x
        if abs(denominator) < 1e-10:
            return 0.0

        return (n * sum_xy - sum_x * sum_y) / denominator

    def _calculate_volatility(self, values: List[float]) -> float:
        """Calculate volatility as standard deviation of changes."""
        if len(values) < 2:
            return 0.0

        # Calculate differences between consecutive values
        changes = []
        for i in range(1, len(values)):
            changes.append(values[i] - values[i-1])

        if not changes:
            return 0.0

        try:
            return statistics.stdev(changes)
        except statistics.StatisticsError:
            return 0.0

    def _determine_trend_direction(self, slope: float, volatility: float,
                                  values: List[float]) -> Tuple[TrendDirection, float]:
        """Determine trend direction and confidence."""
        # For "good" metrics (higher is better): complexity, coupling, debt are negative
        negative_metrics = ["complexity_score", "coupling_score", "total_debt_hours"]

        # Adjust slope interpretation based on metric type
        if slope > 0.1:  # Significant positive slope
            direction = TrendDirection.IMPROVING
            confidence = min(abs(slope) * 10, 0.9)  # Higher slope = higher confidence
        elif slope < -0.1:  # Significant negative slope
            direction = TrendDirection.DECLINING
            confidence = min(abs(slope) * 10, 0.9)
        else:
            # Check volatility for stable vs volatile
            if volatility > statistics.mean(values) * 0.1:  # High volatility
                direction = TrendDirection.VOLATILE
                confidence = min(volatility / (statistics.mean(values) * 0.2), 0.8)
            else:
                direction = TrendDirection.STABLE
                confidence = 0.7

        return direction, confidence

    def _generate_metric_insights(self, analysis: TrendAnalysis, metric: str) -> Tuple[List[str], List[str]]:
        """Generate alerts and recommendations for a metric."""
        alerts = []
        recommendations = []

        if analysis.direction == TrendDirection.DECLINING and analysis.confidence > 0.7:
            alerts.append(f"Declining trend in {metric.replace('_', ' ')}")
            recommendations.append(f"Address declining {metric.replace('_', ' ')} trend")

        elif analysis.direction == TrendDirection.VOLATILE and analysis.confidence > 0.6:
            alerts.append(f"Volatile {metric.replace('_', ' ')} measurements")
            recommendations.append(f"Investigate causes of {metric.replace('_', ' ')} volatility")

        elif analysis.direction == TrendDirection.IMPROVING and analysis.confidence > 0.8:
            recommendations.append(f"Good improvement trend in {metric.replace('_', ' ')}")

        return alerts, recommendations

    def _calculate_overall_trend(self, analyses: Dict[str, TrendAnalysis]) -> TrendDirection:
        """Calculate overall trend across all metrics."""
        if not analyses:
            return TrendDirection.STABLE

        # Count different trend types
        improving = sum(1 for a in analyses.values() if a.direction == TrendDirection.IMPROVING)
        declining = sum(1 for a in analyses.values() if a.direction == TrendDirection.DECLINING)
        volatile = sum(1 for a in analyses.values() if a.direction == TrendDirection.VOLATILE)

        # Determine overall trend
        if declining > improving and declining > volatile:
            return TrendDirection.DECLINING
        elif improving > declining and improving > volatile:
            return TrendDirection.IMPROVING
        elif volatile > improving and volatile > declining:
            return TrendDirection.VOLATILE
        else:
            return TrendDirection.STABLE

    def _has_metric_data(self, data_points: List[HistoricalDataPoint], metric: str) -> bool:
        """Check if metric has sufficient data points."""
        count = sum(1 for dp in data_points if metric in dp.metrics)
        return count >= 3  # Need at least 3 data points for meaningful analysis

    def _create_insufficient_data_report(self) -> TrendReport:
        """Create report when there's insufficient historical data."""
        return TrendReport(
            analyses={},
            overall_trend=TrendDirection.STABLE,
            recommendations=["Collect more historical data for trend analysis"],
            alerts=["Insufficient historical data for trend analysis"],
            generated_at=datetime.now()
        )

    def _load_history(self) -> None:
        """Load historical data from file."""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.history_data = [
                        HistoricalDataPoint.from_dict(item) for item in data
                    ]
            except Exception:
                # If loading fails, start with empty history
                self.history_data = []

    def _save_history(self) -> None:
        """Save historical data to file."""
        try:
            data = [dp.to_dict() for dp in self.history_data]
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception:
            # Silently fail if saving fails
            pass

    def get_metric_history(self, metric: str, days: int = 30) -> List[Tuple[datetime, float]]:
        """Get historical data for a specific metric."""
        cutoff_date = datetime.now() - timedelta(days=days)
        return [
            (dp.timestamp, dp.metrics[metric])
            for dp in self.history_data
            if dp.timestamp >= cutoff_date and metric in dp.metrics
        ]

    def predict_future_value(self, metric: str, days_ahead: int = 30) -> Optional[float]:
        """
        Predict future value of a metric using simple linear regression.

        Args:
            metric: Metric to predict
            days_ahead: Days into the future to predict

        Returns:
            Predicted value or None if insufficient data
        """
        history = self.get_metric_history(metric, days=90)  # Use last 90 days

        if len(history) < 3:
            return None

        # Simple linear extrapolation
        timestamps, values = zip(*history)

        # Convert timestamps to days
        base_time = timestamps[0]
        days_since_start = [
            (ts - base_time).total_seconds() / (24 * 3600)
            for ts in timestamps
        ]

        # Calculate slope
        slope = self._calculate_slope(list(timestamps), list(values))

        # Extrapolate to future
        last_time = timestamps[-1]
        future_time = last_time + timedelta(days=days_ahead)
        days_to_future = (future_time - base_time).total_seconds() / (24 * 3600)

        return values[-1] + slope * (days_to_future - days_since_start[-1])

    def clear_history(self) -> None:
        """Clear all historical data."""
        self.history_data.clear()
        self._save_history()

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics of historical data."""
        if not self.history_data:
            return {"total_data_points": 0}

        metrics_summary = {}
        for metric in QualityMetric:
            values = [
                dp.metrics[metric.value]
                for dp in self.history_data
                if metric.value in dp.metrics
            ]

            if values:
                metrics_summary[metric.value] = {
                    "count": len(values),
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "min": min(values),
                    "max": max(values),
                    "std_dev": statistics.stdev(values) if len(values) > 1 else 0
                }

        return {
            "total_data_points": len(self.history_data),
            "date_range": {
                "start": min(dp.timestamp for dp in self.history_data).isoformat(),
                "end": max(dp.timestamp for dp in self.history_data).isoformat()
            },
            "metrics": metrics_summary
        }
