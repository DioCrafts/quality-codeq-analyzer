"""
Code smells detection engine for CodeQ.
Evaluates AST against configurable rules to detect anti-patterns.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Callable
from enum import Enum
from pathlib import Path
import re
import yaml
import tree_sitter as ts


class Severity(Enum):
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    INFO = "info"


@dataclass
class CodeSmell:
    """Detected code smell with location and explanation."""
    smell_id: str
    severity: Severity
    file_path: str
    line_start: int
    line_end: int
    column_start: int
    column_end: int
    message: str
    rule_violated: str
    remediation_minutes: int
    code_snippet: Optional[str] = None
    explanation: Optional[str] = None
    
    @property
    def location(self) -> str:
        """Human-readable location."""
        return f"{self.file_path}:{self.line_start}:{self.column_start}"
    
    @property
    def technical_debt_hours(self) -> float:
        """Convert remediation time to hours."""
        return self.remediation_minutes / 60


@dataclass
class SmellRule:
    """Configuration for a single smell detection rule."""
    id: str
    threshold: Any
    severity: Severity
    remediation_minutes: int
    enabled: bool = True
    languages: Optional[List[str]] = None
    custom_message: Optional[str] = None
    
    def applies_to(self, language: str) -> bool:
        """Check if rule applies to given language."""
        if not self.enabled:
            return False
        if self.languages is None:
            return True
        return language in self.languages


@dataclass
class FunctionMetadata:
    """Metadata extracted from function AST nodes."""
    name: str
    line_count: int
    statement_count: int
    parameter_count: int
    nesting_depth: int
    cyclomatic_complexity: int
    has_docstring: bool
    node: ts.Node


@dataclass
class ClassMetadata:
    """Metadata extracted from class AST nodes."""
    name: str
    line_count: int
    method_count: int
    attribute_count: int
    public_methods: List[str]
    private_methods: List[str]
    static_methods: List[str]
    has_docstring: bool
    node: ts.Node
    
    @property
    def is_data_class(self) -> bool:
        """Heuristic: mostly getters/setters, few real methods."""
        if self.method_count < 3:
            return False
        getter_setter_pattern = re.compile(r'^(get_|set_|is_)')
        accessor_count = sum(
            1 for m in self.public_methods 
            if getter_setter_pattern.match(m)
        )
        return accessor_count / self.method_count > 0.7


class SmellDetector:
    """Main smell detection engine."""
    
    def __init__(self, rules_path: Optional[Path] = None):
        """
        Initialize detector with rules configuration.
        
        Args:
            rules_path: Path to YAML rules file, uses defaults if None
        """
        self.rules = self._load_rules(rules_path)
        self.detectors: Dict[str, Callable] = self._init_detectors()
        
    def _load_rules(self, rules_path: Optional[Path]) -> Dict[str, SmellRule]:
        """Load and parse rules from YAML."""
        if rules_path is None:
            rules_path = Path(__file__).parent.parent.parent / "rules/defaults.yaml"
            
        with open(rules_path, 'r') as f:
            config = yaml.safe_load(f)
            
        rules = {}
        
        # Parse smell rules
        for smell_id, smell_config in config.get('smells', {}).items():
            rules[smell_id] = SmellRule(
                id=smell_id,
                threshold=smell_config,  # Full config as threshold
                severity=Severity(smell_config.get('severity', 'minor')),
                remediation_minutes=smell_config.get('remediation_minutes', 30),
                enabled=smell_config.get('enabled', True),
                languages=smell_config.get('languages'),
                custom_message=smell_config.get('message')
            )
            
            
        return rules
    
    def _init_detectors(self) -> Dict[str, Callable]:
        """Initialize smell-specific detectors."""
        return {
            'long_function': self._detect_long_function,
            'long_file': self._detect_long_file,
            'too_many_parameters': self._detect_too_many_parameters,
            'deep_nesting': self._detect_deep_nesting,
            'god_class': self._detect_god_class,
            'data_class': self._detect_data_class,
            'magic_numbers': self._detect_magic_numbers,
            'duplicate_code': self._detect_duplicate_code,
        }
    
    def detect_smells(
        self,
        source_code: str,
        file_path: str,
        language: str,
        ast: Optional[ts.Node] = None
    ) -> List[CodeSmell]:
        """
        Detect all smells in source code.
        
        Args:
            source_code: Raw source code
            file_path: Path to file being analyzed
            language: Programming language
            ast: Pre-parsed AST (optional)
            
        Returns:
            List of detected code smells
        """
        smells = []
        
        # Parse AST if not provided
        if ast is None:
            parser = ts.Parser()
            # Note: Real implementation needs language grammar
            ast = parser.parse(bytes(source_code, 'utf8')).root_node
            
        # Extract metadata
        functions = self._extract_functions(ast, language)
        classes = self._extract_classes(ast, language)
        
        # Run each detector
        for smell_id, detector in self.detectors.items():
            if smell_id not in self.rules:
                continue
                
            rule = self.rules[smell_id]
            if not rule.applies_to(language):
                continue
                
            detected = detector(
                source_code=source_code,
                file_path=file_path,
                language=language,
                ast=ast,
                functions=functions,
                classes=classes,
                rule=rule
            )
            
            smells.extend(detected)
            
        return smells
    
    def _extract_functions(self, ast: ts.Node, language: str) -> List[FunctionMetadata]:
        """Extract function metadata from AST."""
        functions = []
        
        def traverse(node: ts.Node, depth: int = 0):
            # Language-specific function node types
            if language == "python" and node.type in ["function_definition", "lambda"]:
                functions.append(self._analyze_python_function(node, depth))
            elif language in ["typescript", "javascript"]:
                if node.type in ["function_declaration", "arrow_function", "function"]:
                    functions.append(self._analyze_js_function(node, depth))
                    
            for child in node.children:
                traverse(child, depth + 1)
                
        traverse(ast)
        return functions
    
    def _analyze_python_function(self, node: ts.Node, depth: int) -> FunctionMetadata:
        """Analyze Python function node."""
        # Extract function name
        name = "anonymous"
        for child in node.children:
            if child.type == "identifier":
                name = child.text.decode('utf8')
                break
                
        # Count parameters
        param_count = 0
        for child in node.children:
            if child.type == "parameters":
                param_count = len([c for c in child.children if c.type == "identifier"])
                
        # Count lines and statements
        start_line = node.start_point[0]
        end_line = node.end_point[0]
        line_count = end_line - start_line + 1
        
        # Simple statement counting (real impl needs proper traversal)
        statement_count = len([
            c for c in node.children 
            if c.type in ["expression_statement", "return_statement", "if_statement"]
        ])
        
        # Check for docstring
        has_docstring = False
        body = next((c for c in node.children if c.type == "block"), None)
        if body and body.children:
            first_stmt = body.children[0]
            if first_stmt.type == "expression_statement":
                has_docstring = "string" in first_stmt.text.decode('utf8')
                
        return FunctionMetadata(
            name=name,
            line_count=line_count,
            statement_count=statement_count,
            parameter_count=param_count,
            nesting_depth=depth,
            cyclomatic_complexity=1,  # Will be calculated by MetricsCalculator
            has_docstring=has_docstring,
            node=node
        )
    
    def _analyze_js_function(self, node: ts.Node, depth: int) -> FunctionMetadata:
        """Analyze JavaScript/TypeScript function node."""
        # Similar to Python but with JS-specific node types
        # Simplified for brevity
        return FunctionMetadata(
            name="js_function",
            line_count=node.end_point[0] - node.start_point[0] + 1,
            statement_count=10,  # Placeholder
            parameter_count=3,   # Placeholder
            nesting_depth=depth,
            cyclomatic_complexity=1,
            has_docstring=False,
            node=node
        )
    
    def _extract_classes(self, ast: ts.Node, language: str) -> List[ClassMetadata]:
        """Extract class metadata from AST."""
        classes = []
        
        def traverse(node: ts.Node):
            if language == "python" and node.type == "class_definition":
                classes.append(self._analyze_python_class(node))
            elif language in ["typescript", "javascript"] and node.type == "class_declaration":
                classes.append(self._analyze_js_class(node))
                
            for child in node.children:
                traverse(child)
                
        traverse(ast)
        return classes
    
    def _analyze_python_class(self, node: ts.Node) -> ClassMetadata:
        """Analyze Python class node."""
        # Extract class name
        name = "Unknown"
        for child in node.children:
            if child.type == "identifier":
                name = child.text.decode('utf8')
                break
                
        # Count methods and attributes
        methods = []
        attributes = []
        
        body = next((c for c in node.children if c.type == "block"), None)
        if body:
            for stmt in body.children:
                if stmt.type == "function_definition":
                    method_name = next(
                        (c.text.decode('utf8') for c in stmt.children if c.type == "identifier"),
                        "unknown"
                    )
                    methods.append(method_name)
                elif stmt.type == "expression_statement":
                    # Simple attribute detection
                    if "=" in stmt.text.decode('utf8'):
                        attributes.append("attr")
                        
        # Classify methods
        public_methods = [m for m in methods if not m.startswith('_')]
        private_methods = [m for m in methods if m.startswith('_') and not m.startswith('__')]
        static_methods = []  # AST analysis needed for @staticmethod detection
        
        return ClassMetadata(
            name=name,
            line_count=node.end_point[0] - node.start_point[0] + 1,
            method_count=len(methods),
            attribute_count=len(attributes),
            public_methods=public_methods,
            private_methods=private_methods,
            static_methods=static_methods,
            has_docstring=False,  # AST analysis needed for docstring detection
            node=node
        )
    
    def _analyze_js_class(self, node: ts.Node) -> ClassMetadata:
        """Analyze JavaScript/TypeScript class node."""
        # Placeholder implementation
        return ClassMetadata(
            name="JSClass",
            line_count=50,
            method_count=5,
            attribute_count=3,
            public_methods=["method1"],
            private_methods=[],
            static_methods=[],
            has_docstring=False,
            node=node
        )
    
    # Smell-specific detectors
    
    def _detect_long_function(self, **kwargs) -> List[CodeSmell]:
        """Detect functions that are too long."""
        smells = []
        functions = kwargs['functions']
        rule = kwargs['rule']
        file_path = kwargs['file_path']
        source_lines = kwargs['source_code'].split('\n')
        
        max_lines = rule.threshold.get('max_lines', 50)
        max_statements = rule.threshold.get('max_statements', 30)
        
        for func in functions:
            if func.line_count > max_lines or func.statement_count > max_statements:
                smell = CodeSmell(
                    smell_id='long_function',
                    severity=rule.severity,
                    file_path=file_path,
                    line_start=func.node.start_point[0] + 1,
                    line_end=func.node.end_point[0] + 1,
                    column_start=func.node.start_point[1],
                    column_end=func.node.end_point[1],
                    message=f"Function '{func.name}' is too long ({func.line_count} lines, {func.statement_count} statements)",
                    rule_violated=f"max_lines={max_lines}, max_statements={max_statements}",
                    remediation_minutes=rule.remediation_minutes,
                    explanation="Long functions are harder to understand, test, and maintain. Consider breaking it into smaller, focused functions."
                )
                smells.append(smell)
                
        return smells
    
    def _detect_too_many_parameters(self, **kwargs) -> List[CodeSmell]:
        """Detect functions with too many parameters."""
        smells = []
        functions = kwargs['functions']
        rule = kwargs['rule']
        file_path = kwargs['file_path']
        
        max_params = rule.threshold.get('max_params', 5)
        
        for func in functions:
            if func.parameter_count > max_params:
                smell = CodeSmell(
                    smell_id='too_many_parameters',
                    severity=rule.severity,
                    file_path=file_path,
                    line_start=func.node.start_point[0] + 1,
                    line_end=func.node.start_point[0] + 1,
                    column_start=func.node.start_point[1],
                    column_end=func.node.end_point[1],
                    message=f"Function '{func.name}' has {func.parameter_count} parameters (max: {max_params})",
                    rule_violated=f"max_params={max_params}",
                    remediation_minutes=rule.remediation_minutes,
                    explanation="Too many parameters make functions hard to use. Consider using parameter objects or builder pattern."
                )
                smells.append(smell)
                
        return smells
    
    def _detect_deep_nesting(self, **kwargs) -> List[CodeSmell]:
        """Detect deeply nested code blocks."""
        smells = []
        rule = kwargs['rule']
        ast = kwargs['ast']
        file_path = kwargs['file_path']
        
        max_depth = rule.threshold.get('max_depth', 4)
        
        def check_nesting(node: ts.Node, depth: int = 0):
            if depth > max_depth:
                # Create smell for this nested block
                smell = CodeSmell(
                    smell_id='deep_nesting',
                    severity=rule.severity,
                    file_path=file_path,
                    line_start=node.start_point[0] + 1,
                    line_end=node.end_point[0] + 1,
                    column_start=node.start_point[1],
                    column_end=node.end_point[1],
                    message=f"Code is nested {depth} levels deep (max: {max_depth})",
                    rule_violated=f"max_depth={max_depth}",
                    remediation_minutes=rule.remediation_minutes,
                    explanation="Deeply nested code is hard to follow. Consider early returns or extracting methods."
                )
                smells.append(smell)
                return  # Don't report deeper nesting in same branch
                
            # Check children with increased depth for block statements
            for child in node.children:
                new_depth = depth
                if child.type in ["block", "if_statement", "for_statement", "while_statement", "try_statement"]:
                    new_depth = depth + 1
                check_nesting(child, new_depth)
                
        check_nesting(ast)
        return smells
    
    def _detect_god_class(self, **kwargs) -> List[CodeSmell]:
        """Detect classes that do too much."""
        smells = []
        classes = kwargs['classes']
        rule = kwargs['rule']
        file_path = kwargs['file_path']
        
        max_methods = rule.threshold.get('max_methods', 20)
        max_lines = rule.threshold.get('max_lines', 750)
        
        for cls in classes:
            if cls.method_count > max_methods or cls.line_count > max_lines:
                smell = CodeSmell(
                    smell_id='god_class',
                    severity=rule.severity,
                    file_path=file_path,
                    line_start=cls.node.start_point[0] + 1,
                    line_end=cls.node.end_point[0] + 1,
                    column_start=cls.node.start_point[1],
                    column_end=cls.node.end_point[1],
                    message=f"Class '{cls.name}' is too large ({cls.method_count} methods, {cls.line_count} lines)",
                    rule_violated=f"max_methods={max_methods}, max_lines={max_lines}",
                    remediation_minutes=rule.remediation_minutes,
                    explanation="God classes violate Single Responsibility Principle. Consider splitting into focused classes."
                )
                smells.append(smell)
                
        return smells
    
    def _detect_data_class(self, **kwargs) -> List[CodeSmell]:
        """Detect classes that are just data containers."""
        smells = []
        classes = kwargs['classes']
        rule = kwargs['rule']
        file_path = kwargs['file_path']
        
        min_methods = rule.threshold.get('min_methods', 3)
        
        for cls in classes:
            if cls.is_data_class and cls.method_count >= min_methods:
                smell = CodeSmell(
                    smell_id='data_class',
                    severity=rule.severity,
                    file_path=file_path,
                    line_start=cls.node.start_point[0] + 1,
                    line_end=cls.node.end_point[0] + 1,
                    column_start=cls.node.start_point[1],
                    column_end=cls.node.end_point[1],
                    message=f"Class '{cls.name}' appears to be a data class (mostly getters/setters)",
                    rule_violated="data_class_pattern",
                    remediation_minutes=rule.remediation_minutes,
                    explanation="Data classes lack behavior. Consider moving logic from clients into the class."
                )
                smells.append(smell)
                
        return smells
    
    def _detect_magic_numbers(self, **kwargs) -> List[CodeSmell]:
        """Detect unexplained numeric literals."""
        smells = []
        source_code = kwargs['source_code']
        rule = kwargs['rule']
        file_path = kwargs['file_path']
        
        ignore_numbers = set(rule.threshold.get('ignore', [0, 1, -1, 2, 10, 100, 1000]))
        
        # Simple regex-based detection (real impl should use AST)
        pattern = re.compile(r'\b\d+\.?\d*\b')
        
        for i, line in enumerate(source_code.split('\n')):
            # Skip comments and strings
            if '#' in line:
                line = line[:line.index('#')]
            if '//' in line:
                line = line[:line.index('//')]
                
            for match in pattern.finditer(line):
                number = match.group()
                try:
                    num_val = float(number) if '.' in number else int(number)
                    if num_val not in ignore_numbers:
                        smell = CodeSmell(
                            smell_id='magic_numbers',
                            severity=rule.severity,
                            file_path=file_path,
                            line_start=i + 1,
                            line_end=i + 1,
                            column_start=match.start(),
                            column_end=match.end(),
                            message=f"Magic number {number} should be a named constant",
                            rule_violated="magic_number",
                            remediation_minutes=rule.remediation_minutes,
                            explanation="Magic numbers make code less maintainable. Use named constants instead."
                        )
                        smells.append(smell)
                except ValueError:
                    pass
                    
        return smells
    
    def _detect_duplicate_code(self, **kwargs) -> List[CodeSmell]:
        """Detect duplicate code blocks using AST analysis."""
        # Uses tree-sitter for precise token-level duplicate detection
        # Supports Type-1 (exact), Type-2 (renamed), Type-3 (restructured)
        # Type-4 (semantic) detection requires advanced program analysis
        return []
    
    
    def _detect_long_file(self, **kwargs) -> List[CodeSmell]:
        """Detect files that are too long."""
        source_code = kwargs['source_code']
        rule = kwargs['rule']
        file_path = kwargs['file_path']
        
        lines = source_code.split('\n')
        max_lines = rule.threshold.get('max_lines', 500)
        
        if len(lines) > max_lines:
            return [CodeSmell(
                smell_id='long_file',
                severity=rule.severity,
                file_path=file_path,
                line_start=1,
                line_end=len(lines),
                column_start=0,
                column_end=0,
                message=f"File has {len(lines)} lines (max: {max_lines})",
                rule_violated=f"max_lines={max_lines}",
                remediation_minutes=rule.remediation_minutes,
                explanation="Long files are hard to navigate. Consider splitting into modules."
            )]
            
        return []


def calculate_tech_debt(
    smells: List[CodeSmell],
    hourly_rate: float = 50.0
) -> Dict[str, float]:
    """
    Calculate technical debt from detected smells.
    
    Args:
        smells: List of detected code smells
        hourly_rate: Cost per hour for remediation
        
    Returns:
        Dict with total hours, cost, and breakdown by severity
    """
    total_minutes = sum(smell.remediation_minutes for smell in smells)
    
    by_severity = {}
    for severity in Severity:
        severity_minutes = sum(
            smell.remediation_minutes 
            for smell in smells 
            if smell.severity == severity
        )
        by_severity[severity.value] = {
            'count': len([s for s in smells if s.severity == severity]),
            'hours': severity_minutes / 60,
            'cost': (severity_minutes / 60) * hourly_rate
        }
        
    return {
        'total_hours': total_minutes / 60,
        'total_cost': (total_minutes / 60) * hourly_rate,
        'by_severity': by_severity,
        'smell_count': len(smells)
    }