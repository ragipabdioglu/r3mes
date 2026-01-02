#!/usr/bin/env python3
"""
R3MES Backend Test Coverage Analysis and Improvement Script

Analyzes current test coverage and generates comprehensive test cases
to achieve 80%+ coverage target.
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Tuple

def run_coverage_analysis() -> Dict:
    """Run pytest with coverage analysis."""
    print("🔍 Running test coverage analysis...")
    
    # Change to backend directory
    backend_dir = Path(__file__).parent.parent
    os.chdir(backend_dir)
    
    try:
        # Run pytest with coverage
        result = subprocess.run([
            sys.executable, "-m", "pytest",
            "--cov=app",
            "--cov-report=json",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov",
            "-v"
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            print(f"❌ Tests failed:\n{result.stdout}\n{result.stderr}")
            return {}
        
        print(f"✅ Tests completed:\n{result.stdout}")
        
        # Load coverage data
        coverage_file = backend_dir / "coverage.json"
        if coverage_file.exists():
            with open(coverage_file) as f:
                return json.load(f)
        
    except subprocess.TimeoutExpired:
        print("❌ Test execution timed out")
    except Exception as e:
        print(f"❌ Error running tests: {e}")
    
    return {}

def analyze_coverage_gaps(coverage_data: Dict) -> List[Tuple[str, float, List[int]]]:
    """Analyze coverage gaps and identify files needing more tests."""
    gaps = []
    
    if not coverage_data or "files" not in coverage_data:
        return gaps
    
    for file_path, file_data in coverage_data["files"].items():
        if not file_path.startswith("app/"):
            continue
        
        # Skip test files and __init__.py
        if "test_" in file_path or "__init__.py" in file_path:
            continue
        
        coverage_percent = file_data["summary"]["percent_covered"]
        missing_lines = file_data["missing_lines"]
        
        if coverage_percent < 80:
            gaps.append((file_path, coverage_percent, missing_lines))
    
    # Sort by lowest coverage first
    gaps.sort(key=lambda x: x[1])
    return gaps

def generate_test_cases(file_path: str, missing_lines: List[int]) -> str:
    """Generate test cases for missing coverage."""
    module_name = file_path.replace("/", ".").replace(".py", "")
    test_file_name = f"test_{Path(file_path).stem}_coverage.py"
    
    return f'''"""
Generated test cases for {file_path}
Coverage improvement tests - targeting lines: {missing_lines}
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient

from app.{Path(file_path).stem} import *
from app.main import app

class Test{Path(file_path).stem.title()}Coverage:
    """Comprehensive test coverage for {module_name}."""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    @pytest.fixture
    def mock_database(self):
        return Mock()
    
    def test_error_handling_paths(self, mock_database):
        """Test error handling code paths."""
        # TODO: Add specific error condition tests
        # Target lines: {missing_lines[:10]}  # First 10 missing lines
        pass
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # TODO: Add edge case tests
        pass
    
    @pytest.mark.asyncio
    async def test_async_error_paths(self):
        """Test async error handling paths."""
        # TODO: Add async error tests
        pass
    
    def test_configuration_variations(self):
        """Test different configuration scenarios."""
        # TODO: Add configuration tests
        pass
    
    @pytest.mark.parametrize("input_value,expected", [
        # TODO: Add parametrized test cases
    ])
    def test_input_variations(self, input_value, expected):
        """Test various input scenarios."""
        pass
'''

def create_missing_tests(gaps: List[Tuple[str, float, List[int]]]) -> None:
    """Create test files for modules with low coverage."""
    tests_dir = Path("tests")
    
    for file_path, coverage_percent, missing_lines in gaps[:5]:  # Top 5 priority files
        print(f"📝 Generating tests for {file_path} ({coverage_percent:.1f}% coverage)")
        
        test_content = generate_test_cases(file_path, missing_lines)
        test_file_name = f"test_{Path(file_path).stem}_coverage.py"
        test_file_path = tests_dir / test_file_name
        
        if not test_file_path.exists():
            with open(test_file_path, "w") as f:
                f.write(test_content)
            print(f"✅ Created {test_file_path}")
        else:
            print(f"⚠️  Test file {test_file_path} already exists")

def generate_coverage_report(coverage_data: Dict, gaps: List[Tuple[str, float, List[int]]]) -> None:
    """Generate a comprehensive coverage report."""
    if not coverage_data:
        print("❌ No coverage data available")
        return
    
    total_coverage = coverage_data.get("totals", {}).get("percent_covered", 0)
    
    report = f"""
# R3MES Backend Test Coverage Report

## Overall Coverage: {total_coverage:.1f}%

### Target: 80%+ Coverage
{'✅ TARGET ACHIEVED' if total_coverage >= 80 else '❌ NEEDS IMPROVEMENT'}

## Files Needing Attention:

"""
    
    for file_path, coverage_percent, missing_lines in gaps[:10]:
        status = "🔴" if coverage_percent < 50 else "🟡" if coverage_percent < 70 else "🟢"
        report += f"{status} **{file_path}**: {coverage_percent:.1f}% ({len(missing_lines)} missing lines)\n"
    
    report += f"""

## Recommendations:

1. **Priority Files**: Focus on files with <50% coverage first
2. **Error Paths**: Add tests for exception handling and error conditions
3. **Edge Cases**: Test boundary conditions and invalid inputs
4. **Async Code**: Ensure async functions have proper test coverage
5. **Configuration**: Test different environment configurations

## Next Steps:

1. Run `python scripts/test_coverage.py` to generate missing test files
2. Implement the TODO items in generated test files
3. Focus on error handling and edge cases
4. Re-run coverage analysis to track progress

Generated on: {__import__('datetime').datetime.now().isoformat()}
"""
    
    with open("COVERAGE_REPORT.md", "w") as f:
        f.write(report)
    
    print(f"📊 Coverage report saved to COVERAGE_REPORT.md")
    print(f"📈 Current coverage: {total_coverage:.1f}%")
    print(f"🎯 Target coverage: 80%")
    print(f"📋 Files needing improvement: {len(gaps)}")

def main():
    """Main test coverage analysis workflow."""
    print("🚀 R3MES Backend Test Coverage Analysis")
    print("=" * 50)
    
    # Run coverage analysis
    coverage_data = run_coverage_analysis()
    
    if not coverage_data:
        print("❌ Could not generate coverage data")
        return 1
    
    # Analyze gaps
    gaps = analyze_coverage_gaps(coverage_data)
    
    # Generate missing test files
    if gaps:
        print(f"\n📋 Found {len(gaps)} files with <80% coverage")
        create_missing_tests(gaps)
    
    # Generate report
    generate_coverage_report(coverage_data, gaps)
    
    total_coverage = coverage_data.get("totals", {}).get("percent_covered", 0)
    
    if total_coverage >= 80:
        print("\n🎉 Congratulations! Test coverage target achieved!")
        return 0
    else:
        print(f"\n📈 Progress needed: {80 - total_coverage:.1f}% more coverage required")
        return 1

if __name__ == "__main__":
    sys.exit(main())