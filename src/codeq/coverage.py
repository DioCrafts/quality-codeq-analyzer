"""
Coverage Parser Module for CodeQ

Parses test coverage reports in various formats (LCOV, coverage.xml, etc.)
and provides coverage data for quality analysis.
"""

from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import xml.etree.ElementTree as ET
import re
from dataclasses import dataclass


@dataclass
class CoverageData:
    """Coverage data for a single file."""
    file_path: str
    line_coverage: float  # Percentage of lines covered
    branch_coverage: float  # Percentage of branches covered
    function_coverage: float  # Percentage of functions covered
    lines_total: int
    lines_covered: int
    branches_total: int
    branches_covered: int
    functions_total: int
    functions_covered: int
    uncovered_lines: List[int]  # List of uncovered line numbers


@dataclass
class CoverageReport:
    """Complete coverage report for a project."""
    files: Dict[str, CoverageData]
    overall_line_coverage: float
    overall_branch_coverage: float
    overall_function_coverage: float
    total_files: int
    total_lines: int
    total_branches: int
    total_functions: int


class CoverageParser:
    """
    Parser for various test coverage report formats.

    Supports:
    - LCOV (.info files)
    - Cobertura XML (coverage.xml)
    - JaCoCo XML
    - Simple JSON format
    """

    def __init__(self):
        self.supported_formats = {
            'lcov': self._parse_lcov,
            'cobertura': self._parse_cobertura,
            'jacoco': self._parse_jacoco,
            'json': self._parse_json,
        }

    def parse_file(self, file_path: Union[str, Path]) -> Optional[CoverageReport]:
        """
        Parse a coverage report file.

        Args:
            file_path: Path to coverage report file

        Returns:
            CoverageReport object or None if parsing failed
        """
        file_path = Path(file_path)

        if not file_path.exists():
            return None

        # Detect format from file extension or content
        format_type = self._detect_format(file_path)
        if not format_type:
            return None

        try:
            parser = self.supported_formats.get(format_type)
            if parser:
                return parser(file_path)
        except Exception as e:
            print(f"Error parsing coverage file {file_path}: {e}")
            return None

        return None

    def find_coverage_files(self, directory: Union[str, Path]) -> List[Path]:
        """
        Find coverage report files in a directory.

        Args:
            directory: Directory to search

        Returns:
            List of found coverage files
        """
        directory = Path(directory)
        coverage_files = []

        # Common coverage file patterns
        patterns = [
            "**/coverage.xml",
            "**/jacoco.xml",
            "**/lcov.info",
            "**/coverage.json",
            "**/cobertura.xml",
            "**/.coverage",
        ]

        for pattern in patterns:
            coverage_files.extend(directory.glob(pattern))

        return coverage_files

    def _detect_format(self, file_path: Path) -> Optional[str]:
        """Detect coverage format from file extension and content."""
        # Check extension first
        suffix = file_path.suffix.lower()
        if suffix == '.xml':
            # Check content to distinguish between cobertura and jacoco
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read(500)  # Read first 500 chars
                    if 'cobertura' in content.lower():
                        return 'cobertura'
                    elif 'jacoco' in content.lower():
                        return 'jacoco'
                    else:
                        return 'cobertura'  # Default XML format
            except:
                return 'cobertura'
        elif suffix == '.info' or file_path.name == 'lcov.info':
            return 'lcov'
        elif suffix == '.json':
            return 'json'
        else:
            # Try to detect from content
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline().strip()
                    if first_line.startswith('TN:'):  # LCOV format
                        return 'lcov'
                    elif first_line.startswith('{'):  # JSON format
                        return 'json'
            except:
                pass

        return None

    def _parse_lcov(self, file_path: Path) -> Optional[CoverageReport]:
        """Parse LCOV format coverage report."""
        files = {}
        current_file = None
        lines_total = 0
        lines_covered = 0

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    if line.startswith('SF:'):
                        # Source file
                        current_file = line[3:]
                        files[current_file] = CoverageData(
                            file_path=current_file,
                            line_coverage=0.0,
                            branch_coverage=0.0,
                            function_coverage=0.0,
                            lines_total=0,
                            lines_covered=0,
                            branches_total=0,
                            branches_covered=0,
                            functions_total=0,
                            functions_covered=0,
                            uncovered_lines=[]
                        )
                    elif line.startswith('DA:') and current_file:
                        # Line coverage: DA:<line>,<hits>
                        parts = line[3:].split(',')
                        if len(parts) == 2:
                            line_num = int(parts[0])
                            hits = int(parts[1])
                            files[current_file].lines_total += 1
                            if hits > 0:
                                files[current_file].lines_covered += 1
                            else:
                                files[current_file].uncovered_lines.append(line_num)
                    elif line.startswith('BRDA:') and current_file:
                        # Branch coverage: BRDA:<line>,<block>,<branch>,<taken>
                        parts = line[5:].split(',')
                        if len(parts) == 4 and parts[3] != '-':
                            taken = int(parts[3])
                            files[current_file].branches_total += 1
                            if taken > 0:
                                files[current_file].branches_covered += 1
                    elif line.startswith('FN:') and current_file:
                        # Function definition: FN:<line>,<name>
                        files[current_file].functions_total += 1
                    elif line.startswith('FNDA:') and current_file:
                        # Function coverage: FNDA:<hits>,<name>
                        parts = line[5:].split(',')
                        if len(parts) == 2:
                            hits = int(parts[0])
                            if hits > 0:
                                files[current_file].functions_covered += 1

            # Calculate percentages
            total_lines = 0
            total_lines_covered = 0
            total_branches = 0
            total_branches_covered = 0
            total_functions = 0
            total_functions_covered = 0

            for file_data in files.values():
                if file_data.lines_total > 0:
                    file_data.line_coverage = (file_data.lines_covered / file_data.lines_total) * 100
                if file_data.branches_total > 0:
                    file_data.branch_coverage = (file_data.branches_covered / file_data.branches_total) * 100
                if file_data.functions_total > 0:
                    file_data.function_coverage = (file_data.functions_covered / file_data.functions_total) * 100

                total_lines += file_data.lines_total
                total_lines_covered += file_data.lines_covered
                total_branches += file_data.branches_total
                total_branches_covered += file_data.branches_covered
                total_functions += file_data.functions_total
                total_functions_covered += file_data.functions_covered

            # Calculate overall coverage
            overall_line_coverage = (total_lines_covered / total_lines * 100) if total_lines > 0 else 0
            overall_branch_coverage = (total_branches_covered / total_branches * 100) if total_branches > 0 else 0
            overall_function_coverage = (total_functions_covered / total_functions * 100) if total_functions > 0 else 0

            return CoverageReport(
                files=files,
                overall_line_coverage=overall_line_coverage,
                overall_branch_coverage=overall_branch_coverage,
                overall_function_coverage=overall_function_coverage,
                total_files=len(files),
                total_lines=total_lines,
                total_branches=total_branches,
                total_functions=total_functions
            )

        except Exception as e:
            print(f"Error parsing LCOV file: {e}")
            return None

    def _parse_cobertura(self, file_path: Path) -> Optional[CoverageReport]:
        """Parse Cobertura XML format coverage report."""
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()

            files = {}
            total_lines = 0
            total_lines_covered = 0
            total_branches = 0
            total_branches_covered = 0
            total_functions = 0
            total_functions_covered = 0

            # Parse each file
            for file_elem in root.findall('.//class'):
                filename = file_elem.get('filename')
                if not filename:
                    continue

                lines_total = 0
                lines_covered = 0
                branches_total = 0
                branches_covered = 0
                functions_total = 0
                functions_covered = 0
                uncovered_lines = []

                # Parse line coverage
                for line_elem in file_elem.findall('.//line'):
                    line_num = int(line_elem.get('number', 0))
                    hits = int(line_elem.get('hits', 0))
                    is_branch = line_elem.get('branch') == 'true'

                    if is_branch:
                        branches_total += 1
                        if hits > 0:
                            branches_covered += 1
                    else:
                        lines_total += 1
                        if hits > 0:
                            lines_covered += 1
                        else:
                            uncovered_lines.append(line_num)

                # Parse method coverage
                for method_elem in file_elem.findall('.//method'):
                    functions_total += 1
                    # Cobertura doesn't always provide method-level coverage
                    # This is a simplification
                    functions_covered += 1

                # Calculate percentages
                line_coverage = (lines_covered / lines_total * 100) if lines_total > 0 else 0
                branch_coverage = (branches_covered / branches_total * 100) if branches_total > 0 else 0
                function_coverage = (functions_covered / functions_total * 100) if functions_total > 0 else 0

                files[filename] = CoverageData(
                    file_path=filename,
                    line_coverage=line_coverage,
                    branch_coverage=branch_coverage,
                    function_coverage=function_coverage,
                    lines_total=lines_total,
                    lines_covered=lines_covered,
                    branches_total=branches_total,
                    branches_covered=branches_covered,
                    functions_total=functions_total,
                    functions_covered=functions_covered,
                    uncovered_lines=uncovered_lines
                )

                total_lines += lines_total
                total_lines_covered += lines_covered
                total_branches += branches_total
                total_branches_covered += branches_covered
                total_functions += functions_total
                total_functions_covered += functions_covered

            # Calculate overall coverage
            overall_line_coverage = (total_lines_covered / total_lines * 100) if total_lines > 0 else 0
            overall_branch_coverage = (total_branches_covered / total_branches * 100) if total_branches > 0 else 0
            overall_function_coverage = (total_functions_covered / total_functions * 100) if total_functions > 0 else 0

            return CoverageReport(
                files=files,
                overall_line_coverage=overall_line_coverage,
                overall_branch_coverage=overall_branch_coverage,
                overall_function_coverage=overall_function_coverage,
                total_files=len(files),
                total_lines=total_lines,
                total_branches=total_branches,
                total_functions=total_functions
            )

        except Exception as e:
            print(f"Error parsing Cobertura XML: {e}")
            return None

    def _parse_jacoco(self, file_path: Path) -> Optional[CoverageReport]:
        """Parse JaCoCo XML format coverage report."""
        # JaCoCo format is similar to Cobertura, but with different structure
        return self._parse_cobertura(file_path)  # Placeholder implementation

    def _parse_json(self, file_path: Path) -> Optional[CoverageReport]:
        """Parse JSON format coverage report."""
        try:
            import json
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            files = {}
            total_lines = 0
            total_lines_covered = 0
            total_branches = 0
            total_branches_covered = 0
            total_functions = 0
            total_functions_covered = 0

            # Parse JSON structure (varies by tool)
            for filename, file_data in data.items():
                lines_total = file_data.get('lines', {}).get('total', 0)
                lines_covered = file_data.get('lines', {}).get('covered', 0)
                branches_total = file_data.get('branches', {}).get('total', 0)
                branches_covered = file_data.get('branches', {}).get('covered', 0)
                functions_total = file_data.get('functions', {}).get('total', 0)
                functions_covered = file_data.get('functions', {}).get('covered', 0)

                # Extract uncovered lines
                uncovered_lines = []
                if 'lines' in file_data and 'details' in file_data['lines']:
                    for line_info in file_data['lines']['details']:
                        if line_info.get('hits', 0) == 0:
                            uncovered_lines.append(line_info.get('line', 0))

                line_coverage = (lines_covered / lines_total * 100) if lines_total > 0 else 0
                branch_coverage = (branches_covered / branches_total * 100) if branches_total > 0 else 0
                function_coverage = (functions_covered / functions_total * 100) if functions_total > 0 else 0

                files[filename] = CoverageData(
                    file_path=filename,
                    line_coverage=line_coverage,
                    branch_coverage=branch_coverage,
                    function_coverage=function_coverage,
                    lines_total=lines_total,
                    lines_covered=lines_covered,
                    branches_total=branches_total,
                    branches_covered=branches_covered,
                    functions_total=functions_total,
                    functions_covered=functions_covered,
                    uncovered_lines=uncovered_lines
                )

                total_lines += lines_total
                total_lines_covered += lines_covered
                total_branches += branches_total
                total_branches_covered += branches_covered
                total_functions += functions_total
                total_functions_covered += functions_covered

            overall_line_coverage = (total_lines_covered / total_lines * 100) if total_lines > 0 else 0
            overall_branch_coverage = (total_branches_covered / total_branches * 100) if total_branches > 0 else 0
            overall_function_coverage = (total_functions_covered / total_functions * 100) if total_functions > 0 else 0

            return CoverageReport(
                files=files,
                overall_line_coverage=overall_line_coverage,
                overall_branch_coverage=overall_branch_coverage,
                overall_function_coverage=overall_function_coverage,
                total_files=len(files),
                total_lines=total_lines,
                total_branches=total_branches,
                total_functions=total_functions
            )

        except Exception as e:
            print(f"Error parsing JSON coverage: {e}")
            return None

    def merge_reports(self, reports: List[CoverageReport]) -> Optional[CoverageReport]:
        """
        Merge multiple coverage reports.

        Args:
            reports: List of coverage reports to merge

        Returns:
            Merged coverage report
        """
        if not reports:
            return None

        all_files = {}
        total_lines = 0
        total_lines_covered = 0
        total_branches = 0
        total_branches_covered = 0
        total_functions = 0
        total_functions_covered = 0

        # Merge file data
        for report in reports:
            for filename, file_data in report.files.items():
                if filename not in all_files:
                    all_files[filename] = CoverageData(
                        file_path=filename,
                        line_coverage=0.0,
                        branch_coverage=0.0,
                        function_coverage=0.0,
                        lines_total=0,
                        lines_covered=0,
                        branches_total=0,
                        branches_covered=0,
                        functions_total=0,
                        functions_covered=0,
                        uncovered_lines=[]
                    )

                # Combine coverage data (simple merge - could be improved)
                all_files[filename].lines_total += file_data.lines_total
                all_files[filename].lines_covered += file_data.lines_covered
                all_files[filename].branches_total += file_data.branches_total
                all_files[filename].branches_covered += file_data.branches_covered
                all_files[filename].functions_total += file_data.functions_total
                all_files[filename].functions_covered += file_data.functions_covered
                all_files[filename].uncovered_lines.extend(file_data.uncovered_lines)

        # Recalculate percentages and totals
        for file_data in all_files.values():
            if file_data.lines_total > 0:
                file_data.line_coverage = (file_data.lines_covered / file_data.lines_total) * 100
            if file_data.branches_total > 0:
                file_data.branch_coverage = (file_data.branches_covered / file_data.branches_total) * 100
            if file_data.functions_total > 0:
                file_data.function_coverage = (file_data.functions_covered / file_data.functions_total) * 100

            total_lines += file_data.lines_total
            total_lines_covered += file_data.lines_covered
            total_branches += file_data.branches_total
            total_branches_covered += file_data.branches_covered
            total_functions += file_data.functions_total
            total_functions_covered += file_data.functions_covered

        overall_line_coverage = (total_lines_covered / total_lines * 100) if total_lines > 0 else 0
        overall_branch_coverage = (total_branches_covered / total_branches * 100) if total_branches > 0 else 0
        overall_function_coverage = (total_functions_covered / total_functions * 100) if total_functions > 0 else 0

        return CoverageReport(
            files=all_files,
            overall_line_coverage=overall_line_coverage,
            overall_branch_coverage=overall_branch_coverage,
            overall_function_coverage=overall_function_coverage,
            total_files=len(all_files),
            total_lines=total_lines,
            total_branches=total_branches,
            total_functions=total_functions
        )
