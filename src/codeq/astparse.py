"""
AST Parser Module for CodeQ

Uses tree-sitter for parsing source code into ASTs across multiple languages.
Provides utilities for traversing and analyzing AST nodes with advanced error handling and optimizations.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import re
import hashlib
import time
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache

try:
    from tree_sitter import Language, Parser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    print("Warning: tree-sitter not available. AST parsing will be limited.")


class LanguageType(Enum):
    """Supported programming languages."""
    PYTHON = "python"
    TYPESCRIPT = "typescript"
    JAVASCRIPT = "javascript"
    GO = "go"
    RUST = "rust"
    JAVA = "java"
    CSHARP = "csharp"
    CPP = "cpp"
    PHP = "php"
    RUBY = "ruby"
    SWIFT = "swift"
    KOTLIN = "kotlin"
    SCALA = "scala"


@dataclass
class ASTNode:
    """Represents a node in the AST."""
    type: str
    start_line: int
    end_line: int
    start_column: int
    end_column: int
    text: str
    children: List['ASTNode']
    parent: Optional['ASTNode'] = None

    @property
    def line_count(self) -> int:
        """Get the number of lines spanned by this node."""
        return self.end_line - self.start_line + 1

    @property
    def is_leaf(self) -> bool:
        """Check if this is a leaf node."""
        return len(self.children) == 0


@dataclass
class FunctionNode:
    """Represents a function/method definition."""
    name: str
    start_line: int
    end_line: int
    parameters: List[str]
    complexity: int = 0
    node: ASTNode = None


@dataclass
class ClassNode:
    """Represents a class definition."""
    name: str
    start_line: int
    end_line: int
    methods: List[FunctionNode]
    node: ASTNode = None


@dataclass
class ParseError:
    """Represents a parsing error."""
    file_path: str
    line: int
    column: int
    message: str
    error_type: str


@dataclass
class ParseMetrics:
    """Metrics about the parsing process."""
    parse_time: float
    file_size: int
    nodes_count: int
    functions_count: int
    classes_count: int
    errors: List[ParseError]


@dataclass
class FileAST:
    """Represents the parsed AST of a source file."""
    file_path: str
    language: LanguageType
    root_node: Optional[ASTNode]
    functions: List[FunctionNode]
    classes: List[ClassNode]
    imports: List[str]
    comments: List[str]
    metrics: ParseMetrics
    parse_errors: List[ParseError]


class ASTParser:
    """
    AST Parser using tree-sitter for multiple programming languages.

    Provides methods to parse source files and extract structural information
    needed for code quality analysis with advanced error handling and caching.
    """

    def __init__(self, enable_cache: bool = True, cache_size: int = 1000):
        self.parsers: Dict[LanguageType, Parser] = {}
        self._initialize_parsers()
        self.enable_cache = enable_cache
        self.cache_size = cache_size

        # Cache for file hashes and AST results
        self._file_cache: Dict[str, FileAST] = {}
        self._hash_cache: Dict[str, str] = {}

        # Performance metrics
        self.total_parse_time = 0.0
        self.total_files_parsed = 0
        self.cache_hits = 0
        self.cache_misses = 0

    def _initialize_parsers(self):
        """Initialize tree-sitter parsers for supported languages."""
        if not TREE_SITTER_AVAILABLE:
            return

        # Language mappings for tree-sitter
        language_configs = {
            LanguageType.PYTHON: ("python", "tree_sitter_python"),
            LanguageType.TYPESCRIPT: ("typescript", "tree_sitter_typescript"),
            LanguageType.JAVASCRIPT: ("javascript", "tree_sitter_javascript"),
            LanguageType.GO: ("go", "tree_sitter_go"),
            LanguageType.RUST: ("rust", "tree_sitter_rust"),
            LanguageType.JAVA: ("java", "tree_sitter_java"),
            LanguageType.CSHARP: ("c_sharp", "tree_sitter_c_sharp"),
            LanguageType.CPP: ("cpp", "tree_sitter_cpp"),
            LanguageType.PHP: ("php", "tree_sitter_php"),
            LanguageType.RUBY: ("ruby", "tree_sitter_ruby"),
            LanguageType.SWIFT: ("swift", "tree_sitter_swift"),
            LanguageType.KOTLIN: ("kotlin", "tree_sitter_kotlin"),
            LanguageType.SCALA: ("scala", "tree_sitter_scala"),
        }

        for lang_type, (lang_name, package_name) in language_configs.items():
            try:
                # Import language dynamically
                language_module = __import__(package_name)
                language = getattr(language_module, f'language_{lang_name}')
                parser = Parser()
                parser.set_language(language)
                self.parsers[lang_type] = parser
            except ImportError:
                # Silently skip unavailable languages
                pass
            except Exception as e:
                print(f"Warning: Failed to initialize parser for {lang_name}: {e}")

    def parse_file(self, file_path: Union[str, Path]) -> Optional[FileAST]:
        """
        Parse a source file into an AST with caching and error handling.

        Args:
            file_path: Path to the source file

        Returns:
            FileAST object or None if parsing failed
        """
        file_path = Path(file_path)
        start_time = time.time()

        if not file_path.exists():
            return None

        # Check cache first
        if self.enable_cache:
            cached_result = self._check_cache(file_path)
            if cached_result:
                self.cache_hits += 1
                return cached_result

        self.cache_misses += 1

        # Detect language from file extension
        language = self._detect_language(file_path)
        if not language:
            return self._create_error_ast(file_path, "Unsupported file type")

        # Read file content with multiple encoding attempts
        content, encoding_error = self._read_file_content(file_path)
        if content is None:
            return self._create_error_ast(file_path, f"Failed to read file: {encoding_error}")

        parse_errors = []

        # Parse with tree-sitter
        ast, parse_error = self._parse_content_safe(content, language)
        if parse_error:
            parse_errors.append(parse_error)

        # Extract structural information
        functions = self._extract_functions(ast, language) if ast else []
        classes = self._extract_classes(ast, functions, language) if ast else []
        imports = self._extract_imports(ast, language) if ast else []
        comments = self._extract_comments(content)

        # Calculate metrics
        parse_time = time.time() - start_time
        metrics = ParseMetrics(
            parse_time=parse_time,
            file_size=len(content.encode('utf-8')),
            nodes_count=self._count_nodes(ast) if ast else 0,
            functions_count=len(functions),
            classes_count=len(classes),
            errors=parse_errors
        )

        result = FileAST(
            file_path=str(file_path),
            language=language,
            root_node=ast,
            functions=functions,
            classes=classes,
            imports=imports,
            comments=comments,
            metrics=metrics,
            parse_errors=parse_errors
        )

        # Cache the result
        if self.enable_cache:
            self._cache_result(file_path, result)

        self.total_parse_time += parse_time
        self.total_files_parsed += 1

        return result

    def _detect_language(self, file_path: Path) -> Optional[LanguageType]:
        """Detect programming language from file extension."""
        extension_map = {
            '.py': LanguageType.PYTHON,
            '.pyw': LanguageType.PYTHON,
            '.ts': LanguageType.TYPESCRIPT,
            '.tsx': LanguageType.TYPESCRIPT,
            '.js': LanguageType.JAVASCRIPT,
            '.jsx': LanguageType.JAVASCRIPT,
            '.mjs': LanguageType.JAVASCRIPT,
            '.cjs': LanguageType.JAVASCRIPT,
            '.go': LanguageType.GO,
            '.rs': LanguageType.RUST,
            '.java': LanguageType.JAVA,
            '.cs': LanguageType.CSHARP,
            '.cpp': LanguageType.CPP,
            '.cc': LanguageType.CPP,
            '.cxx': LanguageType.CPP,
            '.c++': LanguageType.CPP,
            '.hpp': LanguageType.CPP,
            '.hxx': LanguageType.CPP,
            '.h++': LanguageType.CPP,
            '.php': LanguageType.PHP,
            '.rb': LanguageType.RUBY,
            '.swift': LanguageType.SWIFT,
            '.kt': LanguageType.KOTLIN,
            '.scala': LanguageType.SCALA,
        }

        return extension_map.get(file_path.suffix.lower())

    def _read_file_content(self, file_path: Path) -> Tuple[Optional[str], Optional[str]]:
        """Read file content with multiple encoding attempts."""
        encodings_to_try = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']

        for encoding in encodings_to_try:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                    return content, None
            except UnicodeDecodeError:
                continue
            except Exception as e:
                return None, str(e)

        return None, "All encoding attempts failed"

    def _parse_content_safe(self, content: str, language: LanguageType) -> Tuple[Optional[ASTNode], Optional[ParseError]]:
        """Parse content safely with error handling."""
        try:
            return self._parse_content(content, language), None
        except Exception as e:
            error = ParseError(
                file_path="",  # Will be set by caller
                line=0,
                column=0,
                message=str(e),
                error_type="ParseError"
            )
            return None, error

    def _create_error_ast(self, file_path: Path, error_message: str) -> Optional[FileAST]:
        """Create an error AST when parsing fails."""
        error = ParseError(
            file_path=str(file_path),
            line=0,
            column=0,
            message=error_message,
            error_type="FileReadError"
        )

        metrics = ParseMetrics(
            parse_time=0.0,
            file_size=file_path.stat().st_size if file_path.exists() else 0,
            nodes_count=0,
            functions_count=0,
            classes_count=0,
            errors=[error]
        )

        return FileAST(
            file_path=str(file_path),
            language=LanguageType.PYTHON,  # Default
            root_node=None,
            functions=[],
            classes=[],
            imports=[],
            comments=[],
            metrics=metrics,
            parse_errors=[error]
        )

    def _count_nodes(self, node: Optional[ASTNode]) -> int:
        """Count total nodes in AST."""
        if not node:
            return 0

        count = 1
        for child in node.children:
            count += self._count_nodes(child)
        return count

    def _check_cache(self, file_path: Path) -> Optional[FileAST]:
        """Check if file is in cache and still valid."""
        if not self.enable_cache:
            return None

        file_key = str(file_path.resolve())

        try:
            # Calculate file hash
            file_hash = self._calculate_file_hash(file_path)

            if file_key in self._file_cache:
                cached_ast = self._file_cache[file_key]
                cached_hash = self._hash_cache.get(file_key)

                if cached_hash == file_hash:
                    return cached_ast
                else:
                    # File changed, remove from cache
                    del self._file_cache[file_key]
                    if file_key in self._hash_cache:
                        del self._hash_cache[file_key]

        except Exception:
            # Cache check failed, proceed without cache
            pass

        return None

    def _cache_result(self, file_path: Path, ast: FileAST):
        """Cache parsing result."""
        if not self.enable_cache:
            return

        file_key = str(file_path.resolve())

        try:
            file_hash = self._calculate_file_hash(file_path)

            # Implement LRU-style cache eviction
            if len(self._file_cache) >= self.cache_size:
                # Remove oldest entry (simple implementation)
                oldest_key = next(iter(self._file_cache))
                del self._file_cache[oldest_key]
                if oldest_key in self._hash_cache:
                    del self._hash_cache[oldest_key]

            self._file_cache[file_key] = ast
            self._hash_cache[file_key] = file_hash

        except Exception:
            # Cache storage failed, continue without caching
            pass

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate hash of file content for cache validation."""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                return hashlib.md5(content).hexdigest()
        except Exception:
            return ""

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0

        return {
            'cache_enabled': self.enable_cache,
            'cache_size': len(self._file_cache),
            'max_cache_size': self.cache_size,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'total_parse_time': self.total_parse_time,
            'total_files_parsed': self.total_files_parsed,
            'avg_parse_time': self.total_parse_time / self.total_files_parsed if self.total_files_parsed > 0 else 0
        }

    def clear_cache(self):
        """Clear all cached results."""
        self._file_cache.clear()
        self._hash_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0

    def _parse_content(self, content: str, language: LanguageType) -> Optional[ASTNode]:
        """Parse source code content into AST using tree-sitter."""
        if language not in self.parsers:
            return None

        parser = self.parsers[language]

        try:
            tree = parser.parse(content.encode('utf-8'))
            return self._convert_tree_sitter_node(tree.root_node, content)
        except Exception:
            return None

    def _convert_tree_sitter_node(self, ts_node, content: str, parent: Optional[ASTNode] = None) -> ASTNode:
        """Convert tree-sitter node to our ASTNode format."""
        # Get node text
        start_byte = ts_node.start_byte
        end_byte = ts_node.end_byte
        text = content[start_byte:end_byte]

        # Get position information
        start_line = ts_node.start_point[0] + 1  # Convert to 1-based
        start_column = ts_node.start_point[1]
        end_line = ts_node.end_point[0] + 1
        end_column = ts_node.end_point[1]

        # Convert children
        children = []
        for child in ts_node.children:
            child_node = self._convert_tree_sitter_node(child, content)
            child_node.parent = parent
            children.append(child_node)

        return ASTNode(
            type=ts_node.type,
            start_line=start_line,
            end_line=end_line,
            start_column=start_column,
            end_column=end_column,
            text=text,
            children=children,
            parent=parent
        )

    def _extract_functions(self, root_node: ASTNode, language: LanguageType) -> List[FunctionNode]:
        """Extract function definitions from AST."""
        functions = []

        def traverse(node: ASTNode):
            # Language-specific function detection
            if language == LanguageType.PYTHON:
                if node.type == "function_definition":
                    func = self._extract_python_function(node)
                    if func:
                        functions.append(func)
            elif language in [LanguageType.TYPESCRIPT, LanguageType.JAVASCRIPT]:
                if node.type in ["function_declaration", "function", "method_definition"]:
                    func = self._extract_js_function(node)
                    if func:
                        functions.append(func)
            elif language == LanguageType.GO:
                if node.type == "function_declaration":
                    func = self._extract_go_function(node)
                    if func:
                        functions.append(func)
            elif language == LanguageType.RUST:
                if node.type == "function_item":
                    func = self._extract_rust_function(node)
                    if func:
                        functions.append(func)

            for child in node.children:
                traverse(child)

        traverse(root_node)
        return functions

    def _extract_python_function(self, node: ASTNode) -> Optional[FunctionNode]:
        """Extract Python function information."""
        # Find function name
        name = None
        parameters = []

        for child in node.children:
            if child.type == "identifier":
                name = child.text
            elif child.type == "parameters":
                # Extract parameter names
                for param in child.children:
                    if param.type == "identifier":
                        parameters.append(param.text)

        if name:
            return FunctionNode(
                name=name,
                start_line=node.start_line,
                end_line=node.end_line,
                parameters=parameters,
                node=node
            )

        return None

    def _extract_js_function(self, node: ASTNode) -> Optional[FunctionNode]:
        """Extract JavaScript/TypeScript function information."""
        # Similar implementation for JS/TS
        return self._extract_python_function(node)  # Placeholder

    def _extract_go_function(self, node: ASTNode) -> Optional[FunctionNode]:
        """Extract Go function information."""
        return self._extract_python_function(node)  # Placeholder

    def _extract_rust_function(self, node: ASTNode) -> Optional[FunctionNode]:
        """Extract Rust function information."""
        return self._extract_python_function(node)  # Placeholder

    def _extract_classes(self, root_node: ASTNode, functions: List[FunctionNode],
                        language: LanguageType) -> List[ClassNode]:
        """Extract class definitions from AST."""
        classes = []

        def traverse(node: ASTNode):
            if language == LanguageType.PYTHON:
                if node.type == "class_definition":
                    cls = self._extract_python_class(node, functions)
                    if cls:
                        classes.append(cls)
            # Add similar logic for other languages

            for child in node.children:
                traverse(child)

        traverse(root_node)
        return classes

    def _extract_python_class(self, node: ASTNode, functions: List[FunctionNode]) -> Optional[ClassNode]:
        """Extract Python class information."""
        name = None

        for child in node.children:
            if child.type == "identifier":
                name = child.text
                break

        if name:
            # Find methods within this class
            class_methods = [
                func for func in functions
                if node.start_line <= func.start_line <= node.end_line
            ]

            return ClassNode(
                name=name,
                start_line=node.start_line,
                end_line=node.end_line,
                methods=class_methods,
                node=node
            )

        return None

    def _extract_imports(self, root_node: ASTNode, language: LanguageType) -> List[str]:
        """Extract import statements from AST."""
        imports = []

        def traverse(node: ASTNode):
            if language == LanguageType.PYTHON:
                if node.type in ["import_statement", "import_from_statement"]:
                    imports.append(node.text.strip())
            elif language in [LanguageType.TYPESCRIPT, LanguageType.JAVASCRIPT]:
                if node.type in ["import_statement", "import_declaration"]:
                    imports.append(node.text.strip())
            # Add similar logic for Go and Rust

            for child in node.children:
                traverse(child)

        traverse(root_node)
        return imports

    def _extract_comments(self, content: str) -> List[str]:
        """Extract comments from source code."""
        comments = []

        # Language-specific comment patterns
        comment_patterns = [
            r'#.*$',  # Python-style
            r'//.*$',  # C-style single line
            r'/\*.*?\*/',  # C-style multi-line
        ]

        for pattern in comment_patterns:
            matches = re.findall(pattern, content, re.MULTILINE)
            comments.extend(matches)

        return comments

    def calculate_complexity(self, function_node: FunctionNode) -> int:
        """
        Calculate cyclomatic complexity for a function.

        Args:
            function_node: Function to analyze

        Returns:
            Cyclomatic complexity score
        """
        if not function_node.node:
            return 1

        complexity = 1  # Base complexity

        def traverse(node: ASTNode):
            nonlocal complexity

            # Language-specific complexity patterns
            if node.type in ["if_statement", "elif_clause", "else_clause"]:
                complexity += 1
            elif node.type in ["for_statement", "while_statement"]:
                complexity += 1
            elif node.type in ["case_clause", "when_clause"]:
                complexity += 1
            elif node.type in ["&&", "||", "?"]:
                complexity += 1
            elif node.type in ["catch_clause", "finally_clause"]:
                complexity += 1

            for child in node.children:
                traverse(child)

        traverse(function_node.node)
        return complexity

    def get_node_text(self, node: ASTNode, full_content: str) -> str:
        """Extract the text content of an AST node."""
        lines = full_content.split('\n')
        if node.start_line == node.end_line:
            return lines[node.start_line - 1][node.start_column:node.end_column]
        else:
            result = []
            # First line
            result.append(lines[node.start_line - 1][node.start_column:])
            # Middle lines
            for i in range(node.start_line, node.end_line - 1):
                result.append(lines[i])
            # Last line
            result.append(lines[node.end_line - 1][:node.end_column])
            return '\n'.join(result)
