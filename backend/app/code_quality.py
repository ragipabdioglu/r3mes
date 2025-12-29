"""
Code Quality Utilities

Provides utilities for detecting and fixing code quality issues:
- Dead code detection
- Code duplication detection
- Type safety validation
"""

import ast
import logging
from typing import List, Dict, Set, Tuple, Optional
from pathlib import Path
import re

logger = logging.getLogger(__name__)


class DeadCodeDetector:
    """Detects dead code (unused functions, classes, variables)."""
    
    def __init__(self, source_dir: str):
        self.source_dir = Path(source_dir)
        self.defined_functions: Set[str] = set()
        self.defined_classes: Set[str] = set()
        self.used_names: Set[str] = set()
        self.imports: Dict[str, Set[str]] = {}
    
    def analyze_file(self, file_path: Path) -> Dict[str, List[str]]:
        """Analyze a Python file for dead code."""
        issues = {
            'unused_imports': [],
            'unused_functions': [],
            'unused_classes': [],
            'unused_variables': []
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                tree = ast.parse(content, filename=str(file_path))
            
            # Collect defined names
            defined = set()
            imported = set()
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    defined.add(node.name)
                elif isinstance(node, ast.ClassDef):
                    defined.add(node.name)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        imported.add(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imported.add(node.module.split('.')[0])
                    for alias in node.names:
                        imported.add(alias.name)
            
            # Simple heuristic: if a function/class is defined but never called
            # (this is a simplified check - full analysis would require call graph)
            for name in defined:
                if name.startswith('_'):
                    continue  # Skip private names
                # Check if name is used elsewhere (simplified)
                if name not in self.used_names:
                    if name in [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
                        issues['unused_functions'].append(name)
                    elif name in [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
                        issues['unused_classes'].append(name)
        
        except Exception as e:
            logger.warning(f"Error analyzing {file_path}: {e}")
        
        return issues


class DuplicationDetector:
    """Detects code duplication."""
    
    def __init__(self, min_lines: int = 5):
        self.min_lines = min_lines
        self.duplications: List[Tuple[str, str, int]] = []
    
    def find_duplications(self, file_path: Path) -> List[Dict]:
        """Find duplicated code blocks in a file."""
        duplications = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Simple approach: find repeated sequences
            for i in range(len(lines) - self.min_lines):
                sequence = ''.join(lines[i:i+self.min_lines])
                for j in range(i + self.min_lines, len(lines) - self.min_lines):
                    other_sequence = ''.join(lines[j:j+self.min_lines])
                    if sequence.strip() and sequence == other_sequence:
                        duplications.append({
                            'file': str(file_path),
                            'line1': i + 1,
                            'line2': j + 1,
                            'lines': self.min_lines
                        })
        
        except Exception as e:
            logger.warning(f"Error checking duplication in {file_path}: {e}")
        
        return duplications


class TypeSafetyChecker:
    """Checks type safety in Python code."""
    
    def __init__(self):
        self.issues: List[Dict] = []
    
    def check_file(self, file_path: Path) -> List[Dict]:
        """Check type safety in a file."""
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                tree = ast.parse(content, filename=str(file_path))
            
            # Check for functions without type hints
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                    # Check if function has type hints
                    has_return_hint = node.returns is not None
                    has_param_hints = all(
                        arg.annotation is not None
                        for arg in node.args.args
                        if arg.arg != 'self' and arg.arg != 'cls'
                    )
                    
                    if not has_param_hints and len(node.args.args) > 0:
                        issues.append({
                            'type': 'missing_type_hints',
                            'file': str(file_path),
                            'line': node.lineno,
                            'function': node.name,
                            'issue': 'Function parameters missing type hints'
                        })
                    
                    if not has_return_hint and node.name not in ['__init__', '__str__', '__repr__']:
                        issues.append({
                            'type': 'missing_return_hint',
                            'file': str(file_path),
                            'line': node.lineno,
                            'function': node.name,
                            'issue': 'Function missing return type hint'
                        })
        
        except Exception as e:
            logger.warning(f"Error checking types in {file_path}: {e}")
        
        return issues


def analyze_codebase(source_dir: str = "backend/app") -> Dict:
    """
    Analyze codebase for quality issues.
    
    Returns:
        Dictionary with analysis results
    """
    source_path = Path(source_dir)
    results = {
        'dead_code': [],
        'duplications': [],
        'type_safety': [],
        'summary': {}
    }
    
    dead_detector = DeadCodeDetector(source_dir)
    dup_detector = DuplicationDetector(min_lines=5)
    type_checker = TypeSafetyChecker()
    
    python_files = list(source_path.rglob("*.py"))
    
    for file_path in python_files:
        # Skip __pycache__ and test files
        if '__pycache__' in str(file_path) or 'test' in str(file_path).lower():
            continue
        
        # Dead code detection
        dead_issues = dead_detector.analyze_file(file_path)
        if any(dead_issues.values()):
            results['dead_code'].append({
                'file': str(file_path),
                'issues': dead_issues
            })
        
        # Duplication detection
        dups = dup_detector.find_duplications(file_path)
        if dups:
            results['duplications'].extend(dups)
        
        # Type safety checking
        type_issues = type_checker.check_file(file_path)
        if type_issues:
            results['type_safety'].extend(type_issues)
    
    # Summary
    results['summary'] = {
        'total_files': len(python_files),
        'dead_code_issues': sum(len(d['issues']['unused_functions']) + len(d['issues']['unused_classes']) for d in results['dead_code']),
        'duplications': len(results['duplications']),
        'type_safety_issues': len(results['type_safety'])
    }
    
    return results

