"""
Report generation module for CodeQ.
Generates HTML, JSON, SARIF reports and SVG badges.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from datetime import datetime
import json
import html
import base64
from io import BytesIO


@dataclass
class ReportMetadata:
    """Metadata for generated reports."""
    tool_name: str = "CodeQ"
    tool_version: str = "1.0.0"
    analysis_timestamp: str = ""
    repository_path: str = ""
    repository_url: Optional[str] = None
    commit_sha: Optional[str] = None
    branch: Optional[str] = None
    analysis_duration_seconds: float = 0.0
    files_analyzed: int = 0
    rules_applied: int = 0


class ReportGenerator:
    """Base class for report generators."""
    
    def __init__(self, metadata: Optional[ReportMetadata] = None):
        self.metadata = metadata or ReportMetadata()
        if not self.metadata.analysis_timestamp:
            self.metadata.analysis_timestamp = datetime.now().isoformat()
    
    def generate(self, data: Any, output_path: Optional[Path] = None) -> str:
        """Generate report content."""
        raise NotImplementedError


class JSONReportGenerator(ReportGenerator):
    """Generates machine-readable JSON reports."""
    
    def generate(
        self,
        repository_score: Any,
        smells: List[Any],
        output_path: Optional[Path] = None
    ) -> str:
        """
        Generate JSON report.
        
        Args:
            repository_score: RepositoryScore object
            smells: List of all detected code smells
            output_path: Optional path to write JSON file
            
        Returns:
            JSON string
        """
        report = {
            "metadata": asdict(self.metadata),
            "summary": {
                "overall_score": repository_score.overall_score,
                "grade": repository_score.grade,
                "health_status": repository_score.health_status,
                "total_files": repository_score.total_files,
                "total_lines": repository_score.total_lines,
                "total_packages": repository_score.total_packages,
                "technical_debt": {
                    "hours": repository_score.total_debt_hours,
                    "cost": repository_score.debt_cost
                }
            },
            "scores": {
                "complexity": repository_score.complexity_score,
                "maintainability": repository_score.maintainability_score,
                "duplication": repository_score.duplication_score,
                "coupling": repository_score.coupling_score,
                "documentation": repository_score.documentation_score,
                "test_coverage": repository_score.test_coverage_score,
                "security": repository_score.security_score
            },
            "top_issues": repository_score.top_issues,
            "packages": {},
            "issues": []
        }
        
        # Add package details
        for pkg_path, pkg_score in repository_score.package_scores.items():
            report["packages"][pkg_path] = {
                "overall_score": pkg_score.overall_score,
                "grade": pkg_score.grade,
                "total_files": pkg_score.total_files,
                "total_lines": pkg_score.total_lines,
                "worst_files": [
                    {
                        "path": f.file_path,
                        "score": f.overall_score,
                        "grade": f.grade
                    }
                    for f in pkg_score.worst_files
                ]
            }
        
        # Add issues (smells)
        for smell in smells:
            report["issues"].append({
                "id": smell.smell_id,
                "severity": smell.severity.value,
                "file": smell.file_path,
                "location": {
                    "start_line": smell.line_start,
                    "end_line": smell.line_end,
                    "start_column": smell.column_start,
                    "end_column": smell.column_end
                },
                "message": smell.message,
                "rule": smell.rule_violated,
                "remediation_minutes": smell.remediation_minutes,
                "explanation": smell.explanation
            })
        
        json_str = json.dumps(report, indent=2, default=str)
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json_str)
            
        return json_str


class HTMLReportGenerator(ReportGenerator):
    """Generates human-readable HTML reports with visualizations."""
    
    def generate(
        self,
        repository_score: Any,
        smells: List[Any],
        output_path: Optional[Path] = None
    ) -> str:
        """
        Generate HTML report with charts and tables.
        
        Args:
            repository_score: RepositoryScore object
            smells: List of all detected code smells
            output_path: Optional path to write HTML file
            
        Returns:
            HTML string
        """
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CodeQ Report - {self.metadata.repository_path}</title>
    <style>
        {self._get_css()}
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
</head>
<body>
    <div class="container">
        <header>
            <h1>CodeQ Quality Report</h1>
            <div class="metadata">
                <span>Repository: <strong>{self.metadata.repository_path}</strong></span>
                <span>Analysis: <strong>{self.metadata.analysis_timestamp}</strong></span>
                <span>Files: <strong>{repository_score.total_files}</strong></span>
                <span>Lines: <strong>{repository_score.total_lines:,}</strong></span>
            </div>
        </header>
        
        <section class="summary">
            <div class="score-card grade-{repository_score.grade.lower()}">
                <div class="score-value">{repository_score.overall_score:.1f}</div>
                <div class="score-grade">Grade {repository_score.grade}</div>
                <div class="score-status">{repository_score.health_status}</div>
            </div>
            
            <div class="metrics-grid">
                {self._generate_metric_cards(repository_score)}
            </div>
        </section>
        
        <section class="charts">
            <div class="chart-container">
                <canvas id="radarChart"></canvas>
            </div>
            <div class="chart-container">
                <canvas id="trendChart"></canvas>
            </div>
        </section>
        
        <section class="technical-debt">
            <h2>Technical Debt</h2>
            <div class="debt-summary">
                <div class="debt-hours">
                    <span class="value">{repository_score.total_debt_hours:.1f}</span>
                    <span class="label">Hours</span>
                </div>
                <div class="debt-cost">
                    <span class="value">€{repository_score.debt_cost:,.0f}</span>
                    <span class="label">Estimated Cost</span>
                </div>
            </div>
        </section>
        
        <section class="top-issues">
            <h2>Top Issues to Address</h2>
            {self._generate_issues_table(smells[:20])}
        </section>
        
        <section class="packages">
            <h2>Package Breakdown</h2>
            {self._generate_packages_table(repository_score.package_scores)}
        </section>
    </div>
    
    <script>
        {self._get_javascript(repository_score)}
    </script>
</body>
</html>"""
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(html_content)
            
        return html_content
    
    def _get_css(self) -> str:
        """Get CSS styles for the report."""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        
        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
        }
        
        header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .metadata {
            display: flex;
            gap: 30px;
            opacity: 0.9;
            font-size: 0.9em;
        }
        
        section {
            padding: 30px;
        }
        
        .summary {
            display: grid;
            grid-template-columns: 300px 1fr;
            gap: 30px;
            background: #f8f9fa;
        }
        
        .score-card {
            background: white;
            border-radius: 15px;
            padding: 30px;
            text-align: center;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
            position: relative;
            overflow: hidden;
        }
        
        .score-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 5px;
            background: #28a745;
        }
        
        .score-card.grade-a::before { background: #28a745; }
        .score-card.grade-b::before { background: #17a2b8; }
        .score-card.grade-c::before { background: #ffc107; }
        .score-card.grade-d::before { background: #fd7e14; }
        .score-card.grade-f::before { background: #dc3545; }
        
        .score-value {
            font-size: 4em;
            font-weight: bold;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .score-grade {
            font-size: 1.5em;
            margin: 10px 0;
            font-weight: 600;
        }
        
        .score-status {
            color: #6c757d;
            font-size: 1.1em;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        
        .metric-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .metric-card .label {
            color: #6c757d;
            font-size: 0.9em;
            margin-bottom: 5px;
        }
        
        .metric-card .value {
            font-size: 1.8em;
            font-weight: bold;
            color: #333;
        }
        
        .metric-card .bar {
            height: 4px;
            background: #e9ecef;
            border-radius: 2px;
            margin-top: 10px;
            overflow: hidden;
        }
        
        .metric-card .bar-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            border-radius: 2px;
            transition: width 0.3s ease;
        }
        
        .charts {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            padding: 30px;
        }
        
        .chart-container {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .technical-debt {
            background: #f8f9fa;
        }
        
        .debt-summary {
            display: flex;
            gap: 30px;
            justify-content: center;
        }
        
        .debt-hours, .debt-cost {
            background: white;
            padding: 30px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .debt-hours .value, .debt-cost .value {
            display: block;
            font-size: 2.5em;
            font-weight: bold;
            color: #dc3545;
            margin-bottom: 10px;
        }
        
        .debt-hours .label, .debt-cost .label {
            color: #6c757d;
            font-size: 1.1em;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        th {
            background: #f8f9fa;
            padding: 15px;
            text-align: left;
            font-weight: 600;
            color: #495057;
            border-bottom: 2px solid #dee2e6;
        }
        
        td {
            padding: 12px 15px;
            border-bottom: 1px solid #dee2e6;
        }
        
        tr:hover {
            background: #f8f9fa;
        }
        
        .severity-critical { color: #dc3545; font-weight: bold; }
        .severity-major { color: #fd7e14; font-weight: bold; }
        .severity-minor { color: #ffc107; }
        .severity-info { color: #17a2b8; }
        
        h2 {
            margin-bottom: 20px;
            color: #333;
            font-size: 1.8em;
        }
        """
    
    def _get_javascript(self, repository_score: Any) -> str:
        """Generate JavaScript for charts."""
        return f"""
        // Radar Chart
        const radarCtx = document.getElementById('radarChart').getContext('2d');
        new Chart(radarCtx, {{
            type: 'radar',
            data: {{
                labels: ['Complexity', 'Maintainability', 'Duplication', 'Coupling', 'Documentation', 'Testing', 'Security'],
                datasets: [{{
                    label: 'Current Scores',
                    data: [
                        {100 - repository_score.complexity_score:.1f},
                        {repository_score.maintainability_score:.1f},
                        {100 - repository_score.duplication_score:.1f},
                        {100 - repository_score.coupling_score:.1f},
                        {repository_score.documentation_score:.1f},
                        {repository_score.test_coverage_score:.1f},
                        {repository_score.security_score:.1f}
                    ],
                    backgroundColor: 'rgba(102, 126, 234, 0.2)',
                    borderColor: 'rgba(102, 126, 234, 1)',
                    pointBackgroundColor: 'rgba(102, 126, 234, 1)',
                    pointBorderColor: '#fff',
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: 'rgba(102, 126, 234, 1)'
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    r: {{
                        beginAtZero: true,
                        max: 100,
                        ticks: {{
                            stepSize: 20
                        }}
                    }}
                }}
            }}
        }});
        
        // Trend Chart (placeholder data)
        const trendCtx = document.getElementById('trendChart').getContext('2d');
        new Chart(trendCtx, {{
            type: 'line',
            data: {{
                labels: ['Week 1', 'Week 2', 'Week 3', 'Week 4', 'Current'],
                datasets: [{{
                    label: 'Overall Score',
                    data: [65, 68, 70, 72, {repository_score.overall_score:.1f}],
                    borderColor: 'rgba(102, 126, 234, 1)',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    tension: 0.4
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 100
                    }}
                }}
            }}
        }});
        """
    
    def _generate_metric_cards(self, repository_score: Any) -> str:
        """Generate HTML for metric cards."""
        metrics = [
            ("Complexity", 100 - repository_score.complexity_score, "Lower is better"),
            ("Maintainability", repository_score.maintainability_score, "Higher is better"),
            ("Duplication", 100 - repository_score.duplication_score, "Lower is better"),
            ("Coupling", 100 - repository_score.coupling_score, "Lower is better"),
            ("Documentation", repository_score.documentation_score, "Higher is better"),
            ("Test Coverage", repository_score.test_coverage_score, "Higher is better"),
            ("Security", repository_score.security_score, "Higher is better"),
        ]
        
        cards = []
        for label, value, hint in metrics:
            cards.append(f"""
                <div class="metric-card">
                    <div class="label">{label}</div>
                    <div class="value">{value:.0f}</div>
                    <div class="bar">
                        <div class="bar-fill" style="width: {value}%"></div>
                    </div>
                </div>
            """)
        
        return '\n'.join(cards)
    
    def _generate_issues_table(self, smells: List[Any]) -> str:
        """Generate HTML table for issues."""
        if not smells:
            return "<p>No issues detected!</p>"
            
        rows = []
        for smell in smells:
            severity_class = f"severity-{smell.severity.value}"
            rows.append(f"""
                <tr>
                    <td><span class="{severity_class}">{smell.severity.value.upper()}</span></td>
                    <td>{html.escape(smell.smell_id)}</td>
                    <td>{html.escape(smell.file_path)}</td>
                    <td>{smell.line_start}:{smell.column_start}</td>
                    <td>{html.escape(smell.message)}</td>
                    <td>{smell.remediation_minutes} min</td>
                </tr>
            """)
        
        return f"""
            <table>
                <thead>
                    <tr>
                        <th>Severity</th>
                        <th>Issue</th>
                        <th>File</th>
                        <th>Location</th>
                        <th>Message</th>
                        <th>Fix Time</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        """
    
    def _generate_packages_table(self, package_scores: Dict[str, Any]) -> str:
        """Generate HTML table for packages."""
        if not package_scores:
            return "<p>No packages analyzed.</p>"
            
        rows = []
        for pkg_path, pkg_score in package_scores.items():
            rows.append(f"""
                <tr>
                    <td>{html.escape(pkg_path)}</td>
                    <td>{pkg_score.overall_score:.1f}</td>
                    <td>{pkg_score.grade}</td>
                    <td>{pkg_score.total_files}</td>
                    <td>{pkg_score.total_lines:,}</td>
                    <td>{pkg_score.total_technical_debt:.1f}h</td>
                </tr>
            """)
        
        return f"""
            <table>
                <thead>
                    <tr>
                        <th>Package</th>
                        <th>Score</th>
                        <th>Grade</th>
                        <th>Files</th>
                        <th>Lines</th>
                        <th>Tech Debt</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        """


class SARIFReportGenerator(ReportGenerator):
    """Generates SARIF (Static Analysis Results Interchange Format) reports for CI/CD."""
    
    def generate(
        self,
        smells: List[Any],
        rules_config: Dict[str, Any],
        output_path: Optional[Path] = None
    ) -> str:
        """
        Generate SARIF report.
        
        Args:
            smells: List of detected code smells
            rules_config: Rules configuration
            output_path: Optional path to write SARIF file
            
        Returns:
            SARIF JSON string
        """
        sarif = {
            "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
            "version": "2.1.0",
            "runs": [{
                "tool": {
                    "driver": {
                        "name": self.metadata.tool_name,
                        "version": self.metadata.tool_version,
                        "informationUri": "https://github.com/yourusername/codeq",
                        "rules": self._generate_rules(rules_config)
                    }
                },
                "results": self._generate_results(smells),
                "invocations": [{
                    "executionSuccessful": True,
                    "endTimeUtc": self.metadata.analysis_timestamp
                }]
            }]
        }
        
        sarif_json = json.dumps(sarif, indent=2)
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(sarif_json)
            
        return sarif_json
    
    def _generate_rules(self, rules_config: Dict[str, Any]) -> List[Dict]:
        """Generate SARIF rules from configuration."""
        rules = []
        
        for category, category_rules in rules_config.items():
            if category in ['metadata', 'global']:
                continue
                
            if isinstance(category_rules, dict):
                for rule_id, rule_config in category_rules.items():
                    if isinstance(rule_config, dict):
                        rules.append({
                            "id": f"{category}.{rule_id}",
                            "name": rule_id.replace('_', ' ').title(),
                            "shortDescription": {
                                "text": f"Detects {rule_id.replace('_', ' ')}"
                            },
                            "fullDescription": {
                                "text": rule_config.get('description', f"Rule for detecting {rule_id}")
                            },
                            "defaultConfiguration": {
                                "level": self._sarif_level(rule_config.get('severity', 'minor'))
                            },
                            "properties": {
                                "remediation_minutes": rule_config.get('remediation_minutes', 30)
                            }
                        })
        
        return rules
    
    def _generate_results(self, smells: List[Any]) -> List[Dict]:
        """Generate SARIF results from detected smells."""
        results = []
        
        for smell in smells:
            results.append({
                "ruleId": smell.smell_id,
                "level": self._sarif_level(smell.severity.value),
                "message": {
                    "text": smell.message
                },
                "locations": [{
                    "physicalLocation": {
                        "artifactLocation": {
                            "uri": smell.file_path
                        },
                        "region": {
                            "startLine": smell.line_start,
                            "endLine": smell.line_end,
                            "startColumn": smell.column_start,
                            "endColumn": smell.column_end
                        }
                    }
                }],
                "properties": {
                    "remediation_minutes": smell.remediation_minutes,
                    "explanation": smell.explanation
                }
            })
        
        return results
    
    def _sarif_level(self, severity: str) -> str:
        """Map severity to SARIF level."""
        mapping = {
            'critical': 'error',
            'major': 'warning',
            'minor': 'note',
            'info': 'note'
        }
        return mapping.get(severity, 'note')


class BadgeGenerator:
    """Generates SVG badges for repository quality."""
    
    @staticmethod
    def generate_score_badge(
        score: float,
        grade: str,
        label: str = "code quality"
    ) -> str:
        """
        Generate SVG badge for quality score.
        
        Args:
            score: Numeric score (0-100)
            grade: Letter grade
            label: Badge label
            
        Returns:
            SVG string
        """
        # Determine color based on grade
        colors = {
            'A': '#28a745',  # Green
            'B': '#17a2b8',  # Blue
            'C': '#ffc107',  # Yellow
            'D': '#fd7e14',  # Orange
            'F': '#dc3545',  # Red
        }
        color = colors.get(grade, '#6c757d')
        
        # Calculate widths
        label_width = len(label) * 7 + 10
        value_text = f"{score:.0f} ({grade})"
        value_width = len(value_text) * 7 + 10
        total_width = label_width + value_width
        
        svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{total_width}" height="20">
    <linearGradient id="b" x2="0" y2="100%">
        <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
        <stop offset="1" stop-opacity=".1"/>
    </linearGradient>
    <clipPath id="a">
        <rect width="{total_width}" height="20" rx="3" fill="#fff"/>
    </clipPath>
    <g clip-path="url(#a)">
        <path fill="#555" d="M0 0h{label_width}v20H0z"/>
        <path fill="{color}" d="M{label_width} 0h{value_width}v20H{label_width}z"/>
        <path fill="url(#b)" d="M0 0h{total_width}v20H0z"/>
    </g>
    <g fill="#fff" text-anchor="middle" font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="11">
        <text x="{label_width/2}" y="15" fill="#010101" fill-opacity=".3">{label}</text>
        <text x="{label_width/2}" y="14">{label}</text>
        <text x="{label_width + value_width/2}" y="15" fill="#010101" fill-opacity=".3">{value_text}</text>
        <text x="{label_width + value_width/2}" y="14">{value_text}</text>
    </g>
</svg>"""
        
        return svg
    
    @staticmethod
    def generate_coverage_badge(coverage: float) -> str:
        """Generate test coverage badge."""
        # Determine color based on coverage
        if coverage >= 80:
            color = '#28a745'
        elif coverage >= 60:
            color = '#ffc107'
        else:
            color = '#dc3545'
            
        return BadgeGenerator._simple_badge("coverage", f"{coverage:.0f}%", color)
    
    @staticmethod
    def generate_debt_badge(hours: float) -> str:
        """Generate technical debt badge."""
        if hours < 10:
            color = '#28a745'
        elif hours < 50:
            color = '#ffc107'
        else:
            color = '#dc3545'
            
        return BadgeGenerator._simple_badge("tech debt", f"{hours:.0f}h", color)
    
    @staticmethod
    def _simple_badge(label: str, value: str, color: str) -> str:
        """Generate a simple badge."""
        label_width = len(label) * 7 + 10
        value_width = len(value) * 7 + 10
        total_width = label_width + value_width
        
        return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{total_width}" height="20">
    <linearGradient id="b" x2="0" y2="100%">
        <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
        <stop offset="1" stop-opacity=".1"/>
    </linearGradient>
    <clipPath id="a">
        <rect width="{total_width}" height="20" rx="3" fill="#fff"/>
    </clipPath>
    <g clip-path="url(#a)">
        <path fill="#555" d="M0 0h{label_width}v20H0z"/>
        <path fill="{color}" d="M{label_width} 0h{value_width}v20H{label_width}z"/>
        <path fill="url(#b)" d="M0 0h{total_width}v20H0z"/>
    </g>
    <g fill="#fff" text-anchor="middle" font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="11">
        <text x="{label_width/2}" y="15" fill="#010101" fill-opacity=".3">{label}</text>
        <text x="{label_width/2}" y="14">{label}</text>
        <text x="{label_width + value_width/2}" y="15" fill="#010101" fill-opacity=".3">{value}</text>
        <text x="{label_width + value_width/2}" y="14">{value}</text>
    </g>
</svg>"""


def generate_all_reports(
    repository_score: Any,
    smells: List[Any],
    rules_config: Dict[str, Any],
    output_dir: Path,
    formats: List[str] = None
) -> Dict[str, Path]:
    """
    Generate reports in all requested formats.
    
    Args:
        repository_score: Repository analysis results
        smells: Detected code smells
        rules_config: Rules configuration
        output_dir: Directory for output files
        formats: List of formats to generate (json, html, sarif, badges)
        
    Returns:
        Dict mapping format to output path
    """
    if formats is None:
        formats = ['json', 'html', 'sarif', 'badges']
        
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    outputs = {}
    metadata = ReportMetadata(
        repository_path=str(repository_score.repository_path),
        files_analyzed=repository_score.total_files
    )
    
    if 'json' in formats:
        generator = JSONReportGenerator(metadata)
        json_path = output_dir / 'report.json'
        generator.generate(repository_score, smells, json_path)
        outputs['json'] = json_path
        
    if 'html' in formats:
        generator = HTMLReportGenerator(metadata)
        html_path = output_dir / 'report.html'
        generator.generate(repository_score, smells, html_path)
        outputs['html'] = html_path
        
    if 'sarif' in formats:
        generator = SARIFReportGenerator(metadata)
        sarif_path = output_dir / 'report.sarif'
        generator.generate(smells, rules_config, sarif_path)
        outputs['sarif'] = sarif_path
        
    if 'badges' in formats:
        badges_dir = output_dir / 'badges'
        badges_dir.mkdir(exist_ok=True)
        
        # Generate various badges
        score_badge = BadgeGenerator.generate_score_badge(
            repository_score.overall_score,
            repository_score.grade
        )
        (badges_dir / 'quality.svg').write_text(score_badge)
        
        coverage_badge = BadgeGenerator.generate_coverage_badge(
            repository_score.test_coverage_score
        )
        (badges_dir / 'coverage.svg').write_text(coverage_badge)
        
        debt_badge = BadgeGenerator.generate_debt_badge(
            repository_score.total_debt_hours
        )
        (badges_dir / 'debt.svg').write_text(debt_badge)
        
        outputs['badges'] = badges_dir
        
    return outputs