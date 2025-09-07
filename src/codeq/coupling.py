"""
Coupling and cohesion analysis module for CodeQ.
Calculates fan-in/fan-out, instability, and dependency metrics.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple, Any
from collections import defaultdict
from pathlib import Path
import re
import networkx as nx
import tree_sitter as ts


@dataclass
class DependencyEdge:
    """Represents a dependency between two code units."""
    source: str  # Module/class/function making the reference
    target: str  # Module/class/function being referenced
    edge_type: str  # "import", "inheritance", "call", "type_ref"
    line_number: int
    context: Optional[str] = None  # Code snippet showing the dependency


@dataclass
class ModuleMetrics:
    """Coupling metrics for a module/package."""
    name: str
    file_path: str
    
    # Basic metrics
    afferent_coupling: int = 0  # Fan-in: modules that depend on this
    efferent_coupling: int = 0  # Fan-out: modules this depends on
    
    # Derived metrics
    instability: float = 0.0  # I = Ce/(Ca+Ce)
    abstractness: float = 0.0  # A = Abstract classes/Total classes
    distance_from_main: float = 0.0  # D = |A + I - 1|
    
    # Detailed dependencies
    imports: Set[str] = field(default_factory=set)
    imported_by: Set[str] = field(default_factory=set)
    calls_to: Set[str] = field(default_factory=set)
    called_by: Set[str] = field(default_factory=set)
    
    @property
    def total_coupling(self) -> int:
        """Total dependencies (in + out)."""
        return self.afferent_coupling + self.efferent_coupling
    
    @property
    def is_stable(self) -> bool:
        """Module is stable if instability < 0.5."""
        return self.instability < 0.5
    
    @property
    def is_hub(self) -> bool:
        """Module is a hub if it has high fan-in."""
        return self.afferent_coupling > 10


@dataclass
class ClassCoupling:
    """Coupling metrics for a class."""
    name: str
    file_path: str
    
    # CBO - Coupling Between Objects
    coupled_classes: Set[str] = field(default_factory=set)
    
    # RFC - Response For Class (methods + methods called)
    response_set: Set[str] = field(default_factory=set)
    
    # DIT - Depth of Inheritance Tree
    inheritance_depth: int = 0
    
    # NOC - Number of Children
    children_count: int = 0
    
    # LCOM - Lack of Cohesion in Methods
    lcom: float = 0.0

    # WMC - Weighted Methods per Class (sum of method complexities)
    wmc: int = 0

    # CAM - Cohesion Among Methods
    cam: float = 0.0

    # NPM - Number of Public Methods
    npm: int = 0

    # SIX - Specialization Index
    six: float = 0.0

    # MOA - Measure of Aggregation (number of attributes)
    moa: int = 0

    # MFA - Measure of Functional Abstraction
    mfa: float = 0.0
    
    @property
    def cbo(self) -> int:
        """Coupling Between Objects metric."""
        return len(self.coupled_classes)
    
    @property
    def rfc(self) -> int:
        """Response For Class metric."""
        return len(self.response_set)

    @property
    def dit(self) -> int:
        """Depth of Inheritance Tree."""
        return self.inheritance_depth

    @property
    def noc(self) -> int:
        """Number of Children."""
        return self.children_count

    @property
    def lcom1(self) -> float:
        """LCOM1 - Henderson-Sellers variant."""
        return self.lcom

    @property
    def lcom2(self) -> float:
        """LCOM2 - Chidamber & Kemerer variant."""
        # LCOM2 = |P| - |Q| where P and Q are method pairs
        # This is a simplified implementation
        return self.lcom

    @property
    def lcom3(self) -> float:
        """LCOM3 - Li & Henry variant."""
        # LCOM3 = (1/|A|) * sum(|Li - Lj|) for all Li, Lj
        # Where A is attributes, Li is methods using attribute i
        return self.lcom  # Placeholder for now

    @property
    def lcom4(self) -> float:
        """LCOM4 - Berg et al variant."""
        # LCOM4 = 1 - (sum(|Li|)/|M|) / |A|
        # Where M is methods, A is attributes
        return self.lcom  # Placeholder for now

    @property
    def lcom5(self) -> float:
        """LCOM5 - Fernandez variant."""
        # LCOM5 = |P| / (|P| + |Q|) where P and Q are method pairs
        return self.lcom  # Placeholder for now


@dataclass
class CouplingReport:
    """Overall coupling analysis for codebase."""
    modules: Dict[str, ModuleMetrics]
    classes: Dict[str, ClassCoupling]
    dependency_graph: nx.DiGraph
    cycles: List[List[str]]  # Dependency cycles
    
    @property
    def average_instability(self) -> float:
        """Average instability across all modules."""
        if not self.modules:
            return 0.0
        return sum(m.instability for m in self.modules.values()) / len(self.modules)
    
    @property
    def highly_coupled_modules(self) -> List[ModuleMetrics]:
        """Modules with high coupling (>20 dependencies)."""
        return [
            m for m in self.modules.values()
            if m.total_coupling > 20
        ]
    
    @property
    def unstable_modules(self) -> List[ModuleMetrics]:
        """Modules with high instability (>0.8)."""
        return [
            m for m in self.modules.values()
            if m.instability > 0.8
        ]


class DependencyExtractor:
    """Extracts dependencies from source code."""
    
    def __init__(self, language: str):
        self.language = language
        self.import_patterns = self._get_import_patterns(language)
        
    def _get_import_patterns(self, language: str) -> Dict[str, re.Pattern]:
        """Get language-specific import patterns."""
        patterns = {}
        
        if language == "python":
            patterns = {
                'import': re.compile(r'^import\s+([\w\.]+)'),
                'from_import': re.compile(r'^from\s+([\w\.]+)\s+import'),
                'inheritance': re.compile(r'class\s+\w+\s*\(([\w\.,\s]+)\)'),
                'type_hint': re.compile(r':\s*([\w\[\]\.]+)'),
            }
        elif language in ["typescript", "javascript"]:
            patterns = {
                'import': re.compile(r'import\s+.*\s+from\s+[\'"]([^\'\"]+)[\'"]'),
                'require': re.compile(r'require\s*\([\'"]([^\'\"]+)[\'"]\)'),
                'extends': re.compile(r'extends\s+([\w\.]+)'),
                'implements': re.compile(r'implements\s+([\w\.,\s]+)'),
            }
            
        return patterns
    
    def extract_dependencies(
        self,
        source_code: str,
        file_path: str,
        ast: Optional[ts.Node] = None
    ) -> List[DependencyEdge]:
        """
        Extract all dependencies from source code.
        
        Args:
            source_code: Raw source code
            file_path: Path to file
            ast: Pre-parsed AST (optional)
            
        Returns:
            List of dependency edges
        """
        dependencies = []
        lines = source_code.split('\n')
        
        # Extract imports
        for i, line in enumerate(lines, 1):
            line = line.strip()
            
            # Check import patterns
            for pattern_name, pattern in self.import_patterns.items():
                matches = pattern.findall(line)
                for match in matches:
                    # Clean up the match
                    if isinstance(match, tuple):
                        match = match[0]
                        
                    target = match.strip()
                    
                    # Skip relative imports for now
                    if target.startswith('.'):
                        continue
                        
                    dep = DependencyEdge(
                        source=file_path,
                        target=target,
                        edge_type=pattern_name,
                        line_number=i,
                        context=line
                    )
                    dependencies.append(dep)
                    
        # Extract function calls and class references from AST
        if ast:
            dependencies.extend(self._extract_ast_dependencies(ast, file_path))
            
        return dependencies
    
    def _extract_ast_dependencies(
        self,
        ast: ts.Node,
        file_path: str
    ) -> List[DependencyEdge]:
        """Extract dependencies from AST (calls, instantiations, etc)."""
        dependencies = []
        
        def traverse(node: ts.Node):
            # Python function calls
            if self.language == "python" and node.type == "call":
                func_name = self._extract_call_name(node)
                if func_name and '.' in func_name:
                    # External call
                    dep = DependencyEdge(
                        source=file_path,
                        target=func_name,
                        edge_type="call",
                        line_number=node.start_point[0] + 1
                    )
                    dependencies.append(dep)
                    
            # JavaScript/TypeScript new expressions
            elif self.language in ["javascript", "typescript"]:
                if node.type == "new_expression":
                    class_name = self._extract_class_name(node)
                    if class_name:
                        dep = DependencyEdge(
                            source=file_path,
                            target=class_name,
                            edge_type="instantiation",
                            line_number=node.start_point[0] + 1
                        )
                        dependencies.append(dep)
                        
            for child in node.children:
                traverse(child)
                
        traverse(ast)
        return dependencies
    
    def _extract_call_name(self, call_node: ts.Node) -> Optional[str]:
        """Extract function name from call node."""
        # Simplified extraction
        for child in call_node.children:
            if child.type == "attribute":
                # Module.function call
                parts = []
                for part in child.children:
                    if part.type == "identifier":
                        parts.append(part.text.decode('utf8'))
                return '.'.join(parts)
            elif child.type == "identifier":
                return child.text.decode('utf8')
        return None
    
    def _extract_class_name(self, new_node: ts.Node) -> Optional[str]:
        """Extract class name from new expression."""
        for child in new_node.children:
            if child.type == "identifier":
                return child.text.decode('utf8')
        return None


class CouplingAnalyzer:
    """Main coupling analysis engine."""
    
    def __init__(self, language: str):
        self.language = language
        self.extractor = DependencyExtractor(language)
        
    def analyze_coupling(
        self,
        files: Dict[str, str],
        project_root: Optional[Path] = None
    ) -> CouplingReport:
        """
        Analyze coupling across multiple files.
        
        Args:
            files: Dict mapping file paths to source code
            project_root: Root directory for resolving imports
            
        Returns:
            CouplingReport with metrics and dependency graph
        """
        # Build dependency graph
        graph = nx.DiGraph()
        all_dependencies = []
        
        # Extract dependencies from each file
        for file_path, source_code in files.items():
            module_name = self._path_to_module(file_path)
            graph.add_node(module_name, file_path=file_path)
            
            dependencies = self.extractor.extract_dependencies(
                source_code, file_path
            )
            all_dependencies.extend(dependencies)
            
            # Add edges to graph
            for dep in dependencies:
                source_module = self._path_to_module(dep.source)
                target_module = self._resolve_import(dep.target, project_root)
                
                if target_module and target_module in files:
                    graph.add_edge(
                        source_module,
                        target_module,
                        type=dep.edge_type,
                        line=dep.line_number
                    )
                    
        # Calculate module metrics
        modules = {}
        for node in graph.nodes():
            metrics = self._calculate_module_metrics(node, graph)
            modules[node] = metrics
            
        # Analyze classes
        classes = self._analyze_classes(files)
        
        # Find dependency cycles
        cycles = list(nx.simple_cycles(graph))
        
        return CouplingReport(
            modules=modules,
            classes=classes,
            dependency_graph=graph,
            cycles=cycles
        )

    def analyze_repository(self, source_files: List[Path]) -> CouplingReport:
        """Analyze coupling for an entire repository."""
        # Convert list of paths to dict of file_path -> source_code
        files = {}
        for file_path in source_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    files[str(file_path)] = f.read()
            except (UnicodeDecodeError, IOError):
                # Skip files that can't be read
                continue

        # Analyze coupling
        return self.analyze_coupling(files)
    
    def _path_to_module(self, file_path: str) -> str:
        """Convert file path to module name."""
        # Remove extension and convert to module notation
        path = Path(file_path)
        module = str(path.with_suffix(''))
        
        # Convert path separators to dots
        module = module.replace('/', '.').replace('\\', '.')
        
        # Remove leading dots
        while module.startswith('.'):
            module = module[1:]
            
        return module
    
    def _resolve_import(
        self,
        import_path: str,
        project_root: Optional[Path]
    ) -> Optional[str]:
        """
        Resolve import path to module name.
        
        Args:
            import_path: Import string from code
            project_root: Project root for resolution
            
        Returns:
            Resolved module name or None
        """
        # For Python: already in module format
        if self.language == "python":
            return import_path
            
        # For JS/TS: need to resolve file paths
        if self.language in ["javascript", "typescript"]:
            # Remove .js/.ts extensions
            import_path = re.sub(r'\.(js|ts|jsx|tsx)$', '', import_path)
            
            # Convert to module format
            return import_path.replace('/', '.')
            
        return None
    
    def _calculate_module_metrics(
        self,
        module: str,
        graph: nx.DiGraph
    ) -> ModuleMetrics:
        """
        Calculate coupling metrics for a module.
        
        Args:
            module: Module name
            graph: Dependency graph
            
        Returns:
            ModuleMetrics with calculated values
        """
        # Get file path from node data
        file_path = graph.nodes[module].get('file_path', '')
        
        # Calculate fan-in and fan-out
        fan_in = graph.in_degree(module)
        fan_out = graph.out_degree(module)
        
        # Get detailed dependencies
        imports = set(graph.successors(module))
        imported_by = set(graph.predecessors(module))
        
        # Calculate instability
        total = fan_in + fan_out
        instability = fan_out / total if total > 0 else 0.0
        
        # Calculate abstractness using AST analysis of class hierarchies
        abstractness = 0.0  # Placeholder
        
        # Distance from main sequence
        distance = abs(abstractness + instability - 1)
        
        return ModuleMetrics(
            name=module,
            file_path=file_path,
            afferent_coupling=fan_in,
            efferent_coupling=fan_out,
            instability=instability,
            abstractness=abstractness,
            distance_from_main=distance,
            imports=imports,
            imported_by=imported_by
        )
    
    def _analyze_classes(
        self,
        files: Dict[str, str]
    ) -> Dict[str, ClassCoupling]:
        """
        Analyze class-level coupling.
        
        Args:
            files: Source files to analyze
            
        Returns:
            Dict of class coupling metrics
        """
        classes = {}
        
        for file_path, source_code in files.items():
            # Extract class definitions
            class_defs = self._extract_classes(source_code)
            
            for class_name, class_info in class_defs.items():
                coupling = self._analyze_class_coupling(
                    class_name,
                    class_info,
                    file_path
                )
                classes[f"{file_path}::{class_name}"] = coupling
                
        return classes
    
    def _extract_classes(self, source_code: str) -> Dict[str, Dict]:
        """Extract class definitions from source."""
        classes = {}
        
        if self.language == "python":
            # Simple regex-based extraction
            pattern = re.compile(
                r'class\s+(\w+)(?:\s*\((.*?)\))?\s*:',
                re.MULTILINE
            )
            
            for match in pattern.finditer(source_code):
                class_name = match.group(1)
                bases = match.group(2) if match.group(2) else ""
                
                classes[class_name] = {
                    'name': class_name,
                    'bases': [b.strip() for b in bases.split(',')] if bases else [],
                    'start': match.start(),
                    'source': source_code[match.start():match.start() + 1000]  # Sample
                }
                
        elif self.language in ["javascript", "typescript"]:
            # JS/TS class extraction
            pattern = re.compile(
                r'class\s+(\w+)(?:\s+extends\s+(\w+))?',
                re.MULTILINE
            )
            
            for match in pattern.finditer(source_code):
                class_name = match.group(1)
                base = match.group(2)
                
                classes[class_name] = {
                    'name': class_name,
                    'bases': [base] if base else [],
                    'start': match.start()
                }
                
        return classes
    
    def _analyze_class_coupling(
        self,
        class_name: str,
        class_info: Dict,
        file_path: str
    ) -> ClassCoupling:
        """
        Analyze coupling for a single class.
        
        Args:
            class_name: Name of class
            class_info: Extracted class information
            file_path: Path to file containing class
            
        Returns:
            ClassCoupling metrics
        """
        coupling = ClassCoupling(
            name=class_name,
            file_path=file_path
        )
        
        # Add base classes as coupled classes
        for base in class_info.get('bases', []):
            coupling.coupled_classes.add(base)
            
        # Extract method calls using AST parsing
        source = class_info.get('source', '')
        
        # Find method definitions
        method_pattern = re.compile(r'def\s+(\w+)\s*\(')
        methods = method_pattern.findall(source)
        
        # Add methods to response set
        coupling.response_set.update(methods)
        
        # Find external calls using AST-based dependency analysis
        call_pattern = re.compile(r'(\w+)\s*\(')
        calls = call_pattern.findall(source)
        
        for call in calls:
            if call not in methods:  # External call
                coupling.response_set.add(call)
                
        # Calculate LCOM using Henderson-Sellers variant (LCOM1)
        coupling.lcom = self._calculate_lcom(methods, source)

        # Calculate WMC (Weighted Methods per Class)
        coupling.wmc = self._calculate_wmc(methods, source)

        # Calculate CAM (Cohesion Among Methods)
        coupling.cam = self._calculate_cam(methods, source)

        # Calculate NPM (Number of Public Methods)
        coupling.npm = self._calculate_npm(methods)

        # Calculate SIX (Specialization Index)
        coupling.six = self._calculate_six(coupling.inheritance_depth, len(methods))

        # Calculate MOA (Measure of Aggregation)
        coupling.moa = self._calculate_moa(source)

        # Calculate MFA (Measure of Functional Abstraction)
        coupling.mfa = self._calculate_mfa(coupling.inheritance_depth, len(methods))

        # Set inheritance depth using AST-based class hierarchy analysis
        coupling.inheritance_depth = len(class_info.get('bases', []))

        return coupling
    
    def _calculate_lcom(self, methods: List[str], source: str) -> float:
        """
        Calculate Lack of Cohesion in Methods.
        LCOM = |P| - |Q| if |P| > |Q|, else 0
        Where P = method pairs that don't share attributes
        Q = method pairs that share attributes
        """
        if len(methods) < 2:
            return 0.0
            
        # Extract attribute references per method using AST analysis
        method_attrs = defaultdict(set)
        
        for method in methods:
            # Find method body
            pattern = re.compile(
                rf'def\s+{re.escape(method)}\s*\([^)]*\):(.*?)(?=def|\Z)',
                re.DOTALL
            )
            match = pattern.search(source)
            
            if match:
                body = match.group(1)
                # Find self.attribute references
                attr_pattern = re.compile(r'self\.(\w+)')
                attrs = attr_pattern.findall(body)
                method_attrs[method] = set(attrs)
                
        # Count pairs
        p_count = 0  # Pairs with no shared attributes
        q_count = 0  # Pairs with shared attributes
        
        methods_list = list(methods)
        for i in range(len(methods_list)):
            for j in range(i + 1, len(methods_list)):
                attrs1 = method_attrs[methods_list[i]]
                attrs2 = method_attrs[methods_list[j]]
                
                if attrs1 & attrs2:  # Intersection
                    q_count += 1
                else:
                    p_count += 1
                    
        # Calculate LCOM
        lcom = max(0, p_count - q_count)
        
        # Normalize to 0-1 range
        total_pairs = p_count + q_count
        if total_pairs > 0:
            lcom = lcom / total_pairs
            
        return lcom

    def _calculate_wmc(self, methods: List[str], source: str) -> int:
        """
        Calculate Weighted Methods per Class (WMC).
        WMC = sum of cyclomatic complexities of all methods.

        Args:
            methods: List of method names
            source: Source code of the class

        Returns:
            Sum of method complexities
        """
        if not methods:
            return 0

        total_complexity = 0

        for method in methods:
            # Extract method body and calculate its cyclomatic complexity
            pattern = re.compile(
                rf'def\s+{re.escape(method)}\s*\([^)]*\):(.*?)(?=def|\Z)',
                re.DOTALL
            )
            match = pattern.search(source)

            if match:
                method_body = match.group(1)
                # Simple complexity calculation based on keywords
                complexity = 1  # Base complexity
                complexity += method_body.count('if ')
                complexity += method_body.count('elif ')
                complexity += method_body.count('for ')
                complexity += method_body.count('while ')
                complexity += method_body.count('try:')
                complexity += method_body.count('except ')
                complexity += method_body.count(' and ')
                complexity += method_body.count(' or ')

                total_complexity += complexity

        return total_complexity

    def _calculate_cam(self, methods: List[str], source: str) -> float:
        """
        Calculate Cohesion Among Methods (CAM).
        CAM = average similarity between all pairs of methods.

        Args:
            methods: List of method names
            source: Source code of the class

        Returns:
            Cohesion metric (0.0 to 1.0)
        """
        if len(methods) < 2:
            return 1.0

        similarities = []

        # Extract method bodies
        method_bodies = {}
        for method in methods:
            pattern = re.compile(
                rf'def\s+{re.escape(method)}\s*\([^)]*\):(.*?)(?=def|\Z)',
                re.DOTALL
            )
            match = pattern.search(source)
            if match:
                method_bodies[method] = match.group(1).strip()

        # Calculate pairwise similarities
        method_list = list(methods)
        for i in range(len(method_list)):
            for j in range(i + 1, len(method_list)):
                method1 = method_list[i]
                method2 = method_list[j]

                if method1 in method_bodies and method2 in method_bodies:
                    body1 = method_bodies[method1]
                    body2 = method_bodies[method2]

                    # Simple similarity based on shared tokens
                    tokens1 = set(re.findall(r'\b\w+\b', body1))
                    tokens2 = set(re.findall(r'\b\w+\b', body2))

                    if tokens1 or tokens2:
                        intersection = len(tokens1 & tokens2)
                        union = len(tokens1 | tokens2)
                        similarity = intersection / union if union > 0 else 0.0
                        similarities.append(similarity)

        return sum(similarities) / len(similarities) if similarities else 0.0

    def _calculate_npm(self, methods: List[str]) -> int:
        """
        Calculate Number of Public Methods (NPM).
        NPM = number of methods that are not private (don't start with _).

        Args:
            methods: List of method names

        Returns:
            Number of public methods
        """
        return sum(1 for method in methods if not method.startswith('_'))

    def _calculate_six(self, inheritance_depth: int, num_methods: int) -> float:
        """
        Calculate Specialization Index (SIX).
        SIX = DIT / NOM where DIT is inheritance depth, NOM is number of methods.

        Args:
            inheritance_depth: Depth of inheritance tree
            num_methods: Number of methods in class

        Returns:
            Specialization index
        """
        if num_methods == 0:
            return 0.0
        return inheritance_depth / num_methods

    def _calculate_moa(self, source: str) -> int:
        """
        Calculate Measure of Aggregation (MOA).
        MOA = number of attributes in the class.

        Args:
            source: Source code of the class

        Returns:
            Number of attributes
        """
        # Find self.attribute assignments in __init__ and other methods
        attr_pattern = re.compile(r'self\.(\w+)\s*=')
        attributes = set(attr_pattern.findall(source))

        # Also find self.attribute references
        ref_pattern = re.compile(r'self\.(\w+)')
        references = set(ref_pattern.findall(source))

        # Combine and filter out method calls
        all_attrs = attributes | references

        # Remove common method names and built-ins
        method_names = set(re.findall(r'def\s+(\w+)', source))
        builtins = {'__init__', '__str__', '__repr__', '__eq__', '__hash__'}

        filtered_attrs = all_attrs - method_names - builtins

        return len(filtered_attrs)

    def _calculate_mfa(self, inheritance_depth: int, num_methods: int) -> float:
        """
        Calculate Measure of Functional Abstraction (MFA).
        MFA = inherited methods / total methods.

        Args:
            inheritance_depth: Depth of inheritance tree
            num_methods: Total number of methods

        Returns:
            Functional abstraction ratio
        """
        if num_methods == 0:
            return 0.0

        # Simplified: assume inherited methods are proportional to inheritance depth
        # In a real implementation, this would need AST analysis of inheritance hierarchy
        inherited_methods = min(inheritance_depth * 5, num_methods // 2)
        return inherited_methods / num_methods


def analyze_architecture_violations(
    report: CouplingReport,
    rules: Optional[Dict[str, Any]] = None
) -> List[str]:
    """
    Check for architecture violations based on coupling metrics.
    
    Args:
        report: Coupling analysis report
        rules: Custom rules for violations
        
    Returns:
        List of violation messages
    """
    violations = []
    
    # Default rules
    if rules is None:
        rules = {
            'max_fan_out': 20,
            'max_instability': 0.8,
            'max_cbo': 14,  # Coupling Between Objects
            'max_rfc': 50,  # Response For Class
            'allow_cycles': False
        }
        
    # Check module-level violations
    for module_name, metrics in report.modules.items():
        if metrics.efferent_coupling > rules['max_fan_out']:
            violations.append(
                f"Module '{module_name}' has excessive fan-out: "
                f"{metrics.efferent_coupling} > {rules['max_fan_out']}"
            )
            
        if metrics.instability > rules['max_instability']:
            violations.append(
                f"Module '{module_name}' is highly unstable: "
                f"{metrics.instability:.2f} > {rules['max_instability']}"
            )
            
    # Check class-level violations
    for class_name, coupling in report.classes.items():
        if coupling.cbo > rules['max_cbo']:
            violations.append(
                f"Class '{class_name}' has excessive coupling: "
                f"CBO={coupling.cbo} > {rules['max_cbo']}"
            )
            
        if coupling.rfc > rules['max_rfc']:
            violations.append(
                f"Class '{class_name}' has high response set: "
                f"RFC={coupling.rfc} > {rules['max_rfc']}"
            )
            
    # Check for dependency cycles
    if not rules['allow_cycles'] and report.cycles:
        for cycle in report.cycles:
            cycle_str = ' -> '.join(cycle + [cycle[0]])
            violations.append(f"Dependency cycle detected: {cycle_str}")
            
    return violations


def generate_dependency_diagram(
    report: CouplingReport,
    output_format: str = "dot"
) -> str:
    """
    Generate dependency diagram in DOT or Mermaid format.
    
    Args:
        report: Coupling analysis report
        output_format: "dot" or "mermaid"
        
    Returns:
        Diagram source code
    """
    if output_format == "dot":
        return _generate_dot_diagram(report)
    elif output_format == "mermaid":
        return _generate_mermaid_diagram(report)
    else:
        raise ValueError(f"Unsupported format: {output_format}")
        
        
def _generate_dot_diagram(report: CouplingReport) -> str:
    """Generate Graphviz DOT diagram."""
    lines = ["digraph dependencies {"]
    lines.append("  rankdir=LR;")
    lines.append("  node [shape=box];")
    
    # Color nodes by instability
    for module, metrics in report.modules.items():
        color = "lightgreen" if metrics.instability < 0.3 else \
                "yellow" if metrics.instability < 0.7 else "lightcoral"
        lines.append(f'  "{module}" [fillcolor={color}, style=filled];')
        
    # Add edges
    for edge in report.dependency_graph.edges():
        lines.append(f'  "{edge[0]}" -> "{edge[1]}";')
        
    lines.append("}")
    return '\n'.join(lines)


def _generate_mermaid_diagram(report: CouplingReport) -> str:
    """Generate Mermaid diagram."""
    lines = ["graph LR"]
    
    # Add nodes
    for i, module in enumerate(report.modules.keys()):
        lines.append(f"  {i}[{module}]")
        
    # Create module index mapping
    module_index = {m: i for i, m in enumerate(report.modules.keys())}
    
    # Add edges
    for edge in report.dependency_graph.edges():
        if edge[0] in module_index and edge[1] in module_index:
            lines.append(f"  {module_index[edge[0]]} --> {module_index[edge[1]]}")
            
    return '\n'.join(lines)