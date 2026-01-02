#!/usr/bin/env python3
"""
R3MES API Documentation Generator

Automatically generates comprehensive API documentation from code.
"""

import ast
import inspect
import json
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import importlib.util
import sys
from dataclasses import dataclass, field


@dataclass
class APIEndpoint:
    """API endpoint information."""
    name: str
    method: str
    path: str
    description: str
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    request_body: Optional[Dict[str, Any]] = None
    responses: List[Dict[str, Any]] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


@dataclass
class APIClass:
    """API class information."""
    name: str
    description: str
    methods: List[Dict[str, Any]] = field(default_factory=list)
    attributes: List[Dict[str, Any]] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)


@dataclass
class APIModule:
    """API module information."""
    name: str
    description: str
    classes: List[APIClass] = field(default_factory=list)
    functions: List[Dict[str, Any]] = field(default_factory=list)
    constants: List[Dict[str, Any]] = field(default_factory=list)


class APIDocumentationGenerator:
    """Generates comprehensive API documentation."""
    
    def __init__(self, project_root: str, output_dir: str = "docs/api"):
        """
        Initialize API documentation generator.
        
        Args:
            project_root: Root directory of the project
            output_dir: Output directory for documentation
        """
        self.project_root = Path(project_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Documentation data
        self.modules = []
        self.endpoints = []
        
        self.logger.info(f"API documentation generator initialized (root: {project_root})")
    
    def analyze_module(self, module_path: Path) -> APIModule:
        """Analyze a Python module and extract API information."""
        try:
            # Read and parse the module
            with open(module_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            tree = ast.parse(source_code)
            
            # Extract module docstring
            module_doc = ast.get_docstring(tree) or ""
            
            # Create module info
            module_name = module_path.stem
            api_module = APIModule(
                name=module_name,
                description=module_doc
            )
            
            # Analyze classes and functions
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    api_class = self._analyze_class(node, source_code)
                    api_module.classes.append(api_class)
                elif isinstance(node, ast.FunctionDef) and not self._is_method(node, tree):
                    func_info = self._analyze_function(node)
                    api_module.functions.append(func_info)
                elif isinstance(node, ast.Assign):
                    const_info = self._analyze_constant(node)
                    if const_info:
                        api_module.constants.append(const_info)
            
            return api_module
            
        except Exception as e:
            self.logger.error(f"Error analyzing module {module_path}: {e}")
            return APIModule(name=module_path.stem, description="Error analyzing module")
    
    def _analyze_class(self, class_node: ast.ClassDef, source_code: str) -> APIClass:
        """Analyze a class and extract API information."""
        class_doc = ast.get_docstring(class_node) or ""
        
        api_class = APIClass(
            name=class_node.name,
            description=class_doc
        )
        
        # Analyze methods
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef):
                method_info = self._analyze_function(node, is_method=True)
                api_class.methods.append(method_info)
            elif isinstance(node, ast.Assign):
                attr_info = self._analyze_attribute(node)
                if attr_info:
                    api_class.attributes.append(attr_info)
        
        # Extract examples from docstring
        api_class.examples = self._extract_examples_from_docstring(class_doc)
        
        return api_class
    
    def _analyze_function(self, func_node: ast.FunctionDef, is_method: bool = False) -> Dict[str, Any]:
        """Analyze a function and extract API information."""
        func_doc = ast.get_docstring(func_node) or ""
        
        # Extract parameters
        parameters = []
        for arg in func_node.args.args:
            if is_method and arg.arg == 'self':
                continue
            
            param_info = {
                "name": arg.arg,
                "type": self._get_type_annotation(arg.annotation) if arg.annotation else "Any",
                "description": self._extract_param_description(func_doc, arg.arg),
                "required": True,  # Default, can be refined
            }
            parameters.append(param_info)
        
        # Extract return type
        return_type = self._get_type_annotation(func_node.returns) if func_node.returns else "Any"
        
        return {
            "name": func_node.name,
            "description": func_doc,
            "parameters": parameters,
            "return_type": return_type,
            "is_async": isinstance(func_node, ast.AsyncFunctionDef),
            "examples": self._extract_examples_from_docstring(func_doc),
        }
    
    def _analyze_constant(self, assign_node: ast.Assign) -> Optional[Dict[str, Any]]:
        """Analyze a constant assignment."""
        if len(assign_node.targets) == 1 and isinstance(assign_node.targets[0], ast.Name):
            name = assign_node.targets[0].id
            
            # Only document constants (uppercase names)
            if name.isupper():
                value = self._get_literal_value(assign_node.value)
                return {
                    "name": name,
                    "value": value,
                    "type": type(value).__name__ if value is not None else "Unknown",
                }
        
        return None
    
    def _analyze_attribute(self, assign_node: ast.Assign) -> Optional[Dict[str, Any]]:
        """Analyze a class attribute."""
        if len(assign_node.targets) == 1 and isinstance(assign_node.targets[0], ast.Name):
            name = assign_node.targets[0].id
            value = self._get_literal_value(assign_node.value)
            
            return {
                "name": name,
                "value": value,
                "type": type(value).__name__ if value is not None else "Unknown",
            }
        
        return None
    
    def _get_type_annotation(self, annotation: ast.AST) -> str:
        """Extract type annotation as string."""
        try:
            if isinstance(annotation, ast.Name):
                return annotation.id
            elif isinstance(annotation, ast.Constant):
                return str(annotation.value)
            elif isinstance(annotation, ast.Attribute):
                return f"{self._get_type_annotation(annotation.value)}.{annotation.attr}"
            elif isinstance(annotation, ast.Subscript):
                base = self._get_type_annotation(annotation.value)
                slice_val = self._get_type_annotation(annotation.slice)
                return f"{base}[{slice_val}]"
            else:
                return "Any"
        except Exception:
            return "Any"
    
    def _get_literal_value(self, node: ast.AST) -> Any:
        """Extract literal value from AST node."""
        try:
            if isinstance(node, ast.Constant):
                return node.value
            elif isinstance(node, ast.Str):  # Python < 3.8
                return node.s
            elif isinstance(node, ast.Num):  # Python < 3.8
                return node.n
            elif isinstance(node, ast.List):
                return [self._get_literal_value(item) for item in node.elts]
            elif isinstance(node, ast.Dict):
                return {
                    self._get_literal_value(k): self._get_literal_value(v)
                    for k, v in zip(node.keys, node.values)
                }
            else:
                return None
        except Exception:
            return None
    
    def _is_method(self, func_node: ast.FunctionDef, tree: ast.AST) -> bool:
        """Check if a function is a method of a class."""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if func_node in node.body:
                    return True
        return False
    
    def _extract_param_description(self, docstring: str, param_name: str) -> str:
        """Extract parameter description from docstring."""
        lines = docstring.split('\n')
        in_args_section = False
        
        for line in lines:
            line = line.strip()
            if line.lower().startswith('args:') or line.lower().startswith('parameters:'):
                in_args_section = True
                continue
            elif line.lower().startswith('returns:') or line.lower().startswith('yields:'):
                in_args_section = False
                continue
            
            if in_args_section and param_name in line:
                # Extract description after parameter name
                parts = line.split(':', 1)
                if len(parts) > 1:
                    return parts[1].strip()
        
        return ""
    
    def _extract_examples_from_docstring(self, docstring: str) -> List[str]:
        """Extract code examples from docstring."""
        examples = []
        lines = docstring.split('\n')
        in_example = False
        current_example = []
        
        for line in lines:
            stripped = line.strip()
            
            if stripped.lower().startswith('example') or '>>>' in stripped:
                in_example = True
                if current_example:
                    examples.append('\n'.join(current_example))
                    current_example = []
            
            if in_example:
                if stripped.startswith('>>>') or stripped.startswith('...'):
                    current_example.append(line)
                elif stripped and not stripped.lower().startswith('example'):
                    current_example.append(line)
                elif not stripped and current_example:
                    # End of example
                    examples.append('\n'.join(current_example))
                    current_example = []
                    in_example = False
        
        if current_example:
            examples.append('\n'.join(current_example))
        
        return examples
    
    def analyze_fastapi_endpoints(self, app_module_path: Path) -> List[APIEndpoint]:
        """Analyze FastAPI endpoints from application module."""
        endpoints = []
        
        try:
            # Import the module dynamically
            spec = importlib.util.spec_from_file_location("app_module", app_module_path)
            if spec and spec.loader:
                app_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(app_module)
                
                # Look for FastAPI app instance
                app = None
                for attr_name in dir(app_module):
                    attr = getattr(app_module, attr_name)
                    if hasattr(attr, 'routes'):  # FastAPI app has routes
                        app = attr
                        break
                
                if app:
                    # Extract routes
                    for route in app.routes:
                        if hasattr(route, 'methods') and hasattr(route, 'path'):
                            endpoint = APIEndpoint(
                                name=route.name or "unnamed",
                                method=list(route.methods)[0] if route.methods else "GET",
                                path=route.path,
                                description=self._get_route_description(route),
                            )
                            endpoints.append(endpoint)
                
        except Exception as e:
            self.logger.error(f"Error analyzing FastAPI endpoints: {e}")
        
        return endpoints
    
    def _get_route_description(self, route) -> str:
        """Get description from FastAPI route."""
        if hasattr(route, 'endpoint') and route.endpoint:
            return inspect.getdoc(route.endpoint) or ""
        return ""
    
    def scan_project(self, include_patterns: List[str] = None, exclude_patterns: List[str] = None):
        """Scan the entire project for API documentation."""
        if include_patterns is None:
            include_patterns = ["*.py"]
        
        if exclude_patterns is None:
            exclude_patterns = ["test_*.py", "*_test.py", "__pycache__", ".git"]
        
        self.logger.info("Scanning project for API documentation...")
        
        # Find all Python files
        python_files = []
        for pattern in include_patterns:
            python_files.extend(self.project_root.rglob(pattern))
        
        # Filter out excluded files
        filtered_files = []
        for file_path in python_files:
            should_exclude = False
            for exclude_pattern in exclude_patterns:
                if exclude_pattern in str(file_path):
                    should_exclude = True
                    break
            
            if not should_exclude:
                filtered_files.append(file_path)
        
        self.logger.info(f"Found {len(filtered_files)} Python files to analyze")
        
        # Analyze each file
        for file_path in filtered_files:
            try:
                module = self.analyze_module(file_path)
                self.modules.append(module)
                
                # Check for FastAPI endpoints
                if "app" in file_path.name.lower() or "main" in file_path.name.lower():
                    endpoints = self.analyze_fastapi_endpoints(file_path)
                    self.endpoints.extend(endpoints)
                    
            except Exception as e:
                self.logger.error(f"Error analyzing {file_path}: {e}")
        
        self.logger.info(f"Analysis complete: {len(self.modules)} modules, {len(self.endpoints)} endpoints")
    
    def generate_markdown_docs(self):
        """Generate Markdown documentation."""
        # Generate main API documentation
        main_doc_path = self.output_dir / "README.md"
        with open(main_doc_path, 'w') as f:
            f.write(self._generate_main_markdown())
        
        # Generate module documentation
        modules_dir = self.output_dir / "modules"
        modules_dir.mkdir(exist_ok=True)
        
        for module in self.modules:
            module_doc_path = modules_dir / f"{module.name}.md"
            with open(module_doc_path, 'w') as f:
                f.write(self._generate_module_markdown(module))
        
        # Generate endpoint documentation
        if self.endpoints:
            endpoints_doc_path = self.output_dir / "endpoints.md"
            with open(endpoints_doc_path, 'w') as f:
                f.write(self._generate_endpoints_markdown())
        
        self.logger.info(f"Markdown documentation generated in {self.output_dir}")
    
    def _generate_main_markdown(self) -> str:
        """Generate main API documentation markdown."""
        content = [
            "# R3MES Miner Engine API Documentation",
            "",
            "This documentation provides comprehensive information about the R3MES Miner Engine API.",
            "",
            "## Overview",
            "",
            "The R3MES Miner Engine provides a complete solution for decentralized AI training with:",
            "",
            "- **BitNet 1.58-bit Training**: Efficient quantized neural network training",
            "- **LoRA Adapters**: Low-rank adaptation for parameter-efficient training",
            "- **Blockchain Integration**: Decentralized coordination and verification",
            "- **IPFS Storage**: Distributed gradient storage",
            "- **Privacy Protection**: TEE-based secure computation",
            "",
            "## Modules",
            "",
        ]
        
        for module in self.modules:
            content.append(f"- [{module.name}](modules/{module.name}.md) - {module.description.split('.')[0] if module.description else 'No description'}")
        
        if self.endpoints:
            content.extend([
                "",
                "## API Endpoints",
                "",
                f"The system provides {len(self.endpoints)} HTTP endpoints. See [endpoints documentation](endpoints.md) for details.",
            ])
        
        content.extend([
            "",
            "## Quick Start",
            "",
            "```python",
            "from r3mes.miner.engine import MinerEngine",
            "from bridge.blockchain_client import BlockchainClient",
            "",
            "# Initialize blockchain client",
            "blockchain_client = BlockchainClient(",
            "    node_url='localhost:9090',",
            "    chain_id='remes-test',",
            "    private_key='your_private_key'",
            ")",
            "",
            "# Create miner engine",
            "miner = MinerEngine(",
            "    blockchain_client=blockchain_client,",
            "    model_hidden_size=768,",
            "    lora_rank=8",
            ")",
            "",
            "# Start mining",
            "await miner.start_async()",
            "```",
            "",
            "## Architecture",
            "",
            "```",
            "┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐",
            "│   Miner Node    │    │  Serving Node   │    │ Proposer Node   │",
            "│                 │    │                 │    │                 │",
            "│ • BitNet Train  │    │ • Model Serve   │    │ • Aggregation   │",
            "│ • LoRA Adapt    │    │ • Inference     │    │ • Consensus     │",
            "│ • IPFS Upload   │    │ • Load Balance  │    │ • Validation    │",
            "└─────────────────┘    └─────────────────┘    └─────────────────┘",
            "         │                       │                       │",
            "         └───────────────────────┼───────────────────────┘",
            "                                 │",
            "                    ┌─────────────────┐",
            "                    │   Blockchain    │",
            "                    │                 │",
            "                    │ • Coordination  │",
            "                    │ • Verification  │",
            "                    │ • Rewards       │",
            "                    └─────────────────┘",
            "```",
            "",
            "---",
            "",
            f"*Generated automatically from source code*",
        ])
        
        return "\n".join(content)
    
    def _generate_module_markdown(self, module: APIModule) -> str:
        """Generate markdown documentation for a module."""
        content = [
            f"# {module.name}",
            "",
            module.description or "No description available.",
            "",
        ]
        
        if module.constants:
            content.extend([
                "## Constants",
                "",
            ])
            for const in module.constants:
                content.extend([
                    f"### {const['name']}",
                    "",
                    f"**Type:** `{const['type']}`  ",
                    f"**Value:** `{const['value']}`",
                    "",
                ])
        
        if module.functions:
            content.extend([
                "## Functions",
                "",
            ])
            for func in module.functions:
                content.extend(self._format_function_markdown(func))
        
        if module.classes:
            content.extend([
                "## Classes",
                "",
            ])
            for cls in module.classes:
                content.extend(self._format_class_markdown(cls))
        
        return "\n".join(content)
    
    def _format_function_markdown(self, func: Dict[str, Any]) -> List[str]:
        """Format function documentation as markdown."""
        content = [
            f"### {func['name']}",
            "",
        ]
        
        if func.get('is_async'):
            content.append("**Async Function**")
            content.append("")
        
        content.append(func['description'] or "No description available.")
        content.append("")
        
        if func['parameters']:
            content.extend([
                "**Parameters:**",
                "",
            ])
            for param in func['parameters']:
                content.append(f"- `{param['name']}` ({param['type']}): {param['description'] or 'No description'}")
            content.append("")
        
        content.extend([
            f"**Returns:** `{func['return_type']}`",
            "",
        ])
        
        if func['examples']:
            content.extend([
                "**Examples:**",
                "",
            ])
            for example in func['examples']:
                content.extend([
                    "```python",
                    example,
                    "```",
                    "",
                ])
        
        return content
    
    def _format_class_markdown(self, cls: APIClass) -> List[str]:
        """Format class documentation as markdown."""
        content = [
            f"### {cls.name}",
            "",
            cls.description or "No description available.",
            "",
        ]
        
        if cls.attributes:
            content.extend([
                "**Attributes:**",
                "",
            ])
            for attr in cls.attributes:
                content.append(f"- `{attr['name']}` ({attr['type']}): {attr.get('value', 'No default value')}")
            content.append("")
        
        if cls.methods:
            content.extend([
                "**Methods:**",
                "",
            ])
            for method in cls.methods:
                content.extend(self._format_function_markdown(method))
        
        if cls.examples:
            content.extend([
                "**Examples:**",
                "",
            ])
            for example in cls.examples:
                content.extend([
                    "```python",
                    example,
                    "```",
                    "",
                ])
        
        return content
    
    def _generate_endpoints_markdown(self) -> str:
        """Generate markdown documentation for API endpoints."""
        content = [
            "# API Endpoints",
            "",
            "This document describes all HTTP API endpoints provided by the R3MES Miner Engine.",
            "",
        ]
        
        # Group endpoints by tags or path
        grouped_endpoints = {}
        for endpoint in self.endpoints:
            group = endpoint.tags[0] if endpoint.tags else "General"
            if group not in grouped_endpoints:
                grouped_endpoints[group] = []
            grouped_endpoints[group].append(endpoint)
        
        for group, endpoints in grouped_endpoints.items():
            content.extend([
                f"## {group}",
                "",
            ])
            
            for endpoint in endpoints:
                content.extend([
                    f"### {endpoint.method} {endpoint.path}",
                    "",
                    endpoint.description or "No description available.",
                    "",
                ])
                
                if endpoint.parameters:
                    content.extend([
                        "**Parameters:**",
                        "",
                    ])
                    for param in endpoint.parameters:
                        content.append(f"- `{param['name']}` ({param.get('type', 'string')}): {param.get('description', 'No description')}")
                    content.append("")
                
                if endpoint.examples:
                    content.extend([
                        "**Examples:**",
                        "",
                    ])
                    for example in endpoint.examples:
                        content.extend([
                            "```bash",
                            f"curl -X {endpoint.method} {endpoint.path}",
                            "```",
                            "",
                        ])
        
        return "\n".join(content)
    
    def generate_json_schema(self) -> Dict[str, Any]:
        """Generate JSON schema for the API."""
        schema = {
            "openapi": "3.0.0",
            "info": {
                "title": "R3MES Miner Engine API",
                "version": "1.0.0",
                "description": "Decentralized AI Training Engine API"
            },
            "paths": {},
            "components": {
                "schemas": {}
            }
        }
        
        # Add endpoints to schema
        for endpoint in self.endpoints:
            if endpoint.path not in schema["paths"]:
                schema["paths"][endpoint.path] = {}
            
            schema["paths"][endpoint.path][endpoint.method.lower()] = {
                "summary": endpoint.name,
                "description": endpoint.description,
                "parameters": endpoint.parameters,
                "responses": {
                    "200": {
                        "description": "Success"
                    }
                }
            }
        
        return schema
    
    def export_documentation(self, formats: List[str] = None):
        """Export documentation in multiple formats."""
        if formats is None:
            formats = ["markdown", "json"]
        
        if "markdown" in formats:
            self.generate_markdown_docs()
        
        if "json" in formats:
            schema = self.generate_json_schema()
            schema_path = self.output_dir / "openapi.json"
            with open(schema_path, 'w') as f:
                json.dump(schema, f, indent=2)
            self.logger.info(f"JSON schema exported to {schema_path}")


def generate_api_docs(
    project_root: str = ".",
    output_dir: str = "docs/api",
    formats: List[str] = None,
) -> APIDocumentationGenerator:
    """
    Generate API documentation for R3MES Miner Engine.
    
    Args:
        project_root: Root directory of the project
        output_dir: Output directory for documentation
        formats: List of output formats ("markdown", "json")
        
    Returns:
        APIDocumentationGenerator instance
    """
    generator = APIDocumentationGenerator(project_root, output_dir)
    generator.scan_project()
    generator.export_documentation(formats)
    return generator


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate R3MES API Documentation")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument("--output-dir", default="docs/api", help="Output directory")
    parser.add_argument("--formats", nargs="+", default=["markdown", "json"], help="Output formats")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    generator = generate_api_docs(
        project_root=args.project_root,
        output_dir=args.output_dir,
        formats=args.formats,
    )
    
    print(f"✅ API documentation generated in {args.output_dir}")
    print(f"📊 Analyzed {len(generator.modules)} modules and {len(generator.endpoints)} endpoints")