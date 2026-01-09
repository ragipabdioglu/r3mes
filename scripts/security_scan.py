#!/usr/bin/env python3
"""
R3MES Automated Security Scanning Script

Performs comprehensive security analysis including:
- Dependency vulnerability scanning
- Static code analysis
- Configuration security checks
- Docker image scanning
"""

import os
import sys
import subprocess
import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

class SecurityScanner:
    """Comprehensive security scanner for R3MES project."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "scans": {},
            "summary": {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0}
        }
    
    def run_dependency_scan(self) -> Dict:
        """Scan Python dependencies for vulnerabilities using safety."""
        print("🔍 Scanning Python dependencies...")
        
        try:
            # Install safety if not available
            subprocess.run([sys.executable, "-m", "pip", "install", "safety"], 
                         capture_output=True, check=False)
            
            # Run safety check
            result = subprocess.run([
                sys.executable, "-m", "safety", "check", 
                "--json", "--full-report"
            ], capture_output=True, text=True, cwd=self.project_root / "backend")
            
            if result.returncode == 0:
                return {"status": "clean", "vulnerabilities": []}
            else:
                try:
                    vulns = json.loads(result.stdout)
                    return {"status": "vulnerabilities_found", "vulnerabilities": vulns}
                except json.JSONDecodeError:
                    return {"status": "error", "message": result.stderr}
                    
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def run_bandit_scan(self) -> Dict:
        """Run Bandit static security analysis on Python code."""
        print("🔍 Running Bandit static analysis...")
        
        try:
            # Install bandit if not available
            subprocess.run([sys.executable, "-m", "pip", "install", "bandit"], 
                         capture_output=True, check=False)
            
            # Run bandit
            result = subprocess.run([
                sys.executable, "-m", "bandit", "-r", "backend/app",
                "-f", "json", "-o", "bandit_report.json"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            report_file = self.project_root / "bandit_report.json"
            if report_file.exists():
                with open(report_file) as f:
                    return json.load(f)
            else:
                return {"results": [], "metrics": {}}
                
        except Exception as e:
            return {"error": str(e)}
    
    def scan_docker_images(self) -> Dict:
        """Scan Docker images for vulnerabilities using trivy."""
        print("🔍 Scanning Docker images...")
        
        images_to_scan = [
            "postgres:16-alpine",
            "redis:7-alpine"
        ]
        
        results = {}
        
        for image in images_to_scan:
            try:
                # Check if trivy is available
                result = subprocess.run([
                    "trivy", "image", "--format", "json", image
                ], capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    try:
                        scan_result = json.loads(result.stdout)
                        results[image] = scan_result
                    except json.JSONDecodeError:
                        results[image] = {"error": "Failed to parse trivy output"}
                else:
                    results[image] = {"error": f"Trivy scan failed: {result.stderr}"}
                    
            except subprocess.TimeoutExpired:
                results[image] = {"error": "Scan timeout"}
            except FileNotFoundError:
                results[image] = {"error": "Trivy not installed"}
            except Exception as e:
                results[image] = {"error": str(e)}
        
        return results
    
    def check_configuration_security(self) -> Dict:
        """Check configuration files for security issues."""
        print("🔍 Checking configuration security...")
        
        issues = []
        
        # Check .env files for sensitive data
        env_files = list(self.project_root.glob("**/.env*"))
        for env_file in env_files:
            if env_file.name == ".env":
                issues.append({
                    "severity": "high",
                    "file": str(env_file),
                    "issue": "Production .env file found in repository",
                    "recommendation": "Move to secure secret management"
                })
        
        # Check Docker Compose for hardcoded secrets
        compose_files = list(self.project_root.glob("**/docker-compose*.yml"))
        for compose_file in compose_files:
            try:
                with open(compose_file) as f:
                    content = f.read()
                    if "password:" in content.lower() and "${" not in content:
                        issues.append({
                            "severity": "critical",
                            "file": str(compose_file),
                            "issue": "Hardcoded password detected",
                            "recommendation": "Use environment variables"
                        })
            except Exception as e:
                issues.append({
                    "severity": "low",
                    "file": str(compose_file),
                    "issue": f"Could not analyze file: {e}",
                    "recommendation": "Manual review required"
                })
        
        # Check Kubernetes manifests
        k8s_files = list(self.project_root.glob("k8s/*.yaml"))
        for k8s_file in k8s_files:
            try:
                with open(k8s_file) as f:
                    content = yaml.safe_load(f)
                    if isinstance(content, dict) and content.get("kind") == "Secret":
                        if content.get("type") != "Opaque":
                            issues.append({
                                "severity": "medium",
                                "file": str(k8s_file),
                                "issue": "Non-opaque secret type",
                                "recommendation": "Use Sealed Secrets or external secret management"
                            })
            except Exception as e:
                issues.append({
                    "severity": "low",
                    "file": str(k8s_file),
                    "issue": f"Could not analyze YAML: {e}",
                    "recommendation": "Manual review required"
                })
        
        return {"issues": issues}
    
    def check_file_permissions(self) -> Dict:
        """Check for files with overly permissive permissions."""
        print("🔍 Checking file permissions...")
        
        issues = []
        sensitive_patterns = ["*.key", "*.pem", "*.p12", "*.pfx", ".env*"]
        
        for pattern in sensitive_patterns:
            for file_path in self.project_root.glob(f"**/{pattern}"):
                if file_path.is_file():
                    stat = file_path.stat()
                    mode = oct(stat.st_mode)[-3:]
                    
                    # Check if file is readable by others (world-readable)
                    if int(mode[2]) & 4:  # Others have read permission
                        issues.append({
                            "severity": "high",
                            "file": str(file_path),
                            "permissions": mode,
                            "issue": "Sensitive file is world-readable",
                            "recommendation": "chmod 600 or move to secure location"
                        })
        
        return {"issues": issues}
    
    def generate_report(self) -> str:
        """Generate comprehensive security report."""
        report = f"""# R3MES Security Scan Report

**Scan Date**: {self.results['timestamp']}

## Summary

- 🔴 Critical: {self.results['summary']['critical']}
- 🟠 High: {self.results['summary']['high']}
- 🟡 Medium: {self.results['summary']['medium']}
- 🔵 Low: {self.results['summary']['low']}
- ℹ️ Info: {self.results['summary']['info']}

## Scan Results

"""
        
        for scan_name, scan_result in self.results["scans"].items():
            report += f"### {scan_name.replace('_', ' ').title()}\n\n"
            
            if scan_name == "dependency_scan":
                if scan_result["status"] == "clean":
                    report += "✅ No known vulnerabilities found in dependencies\n\n"
                else:
                    report += f"❌ Found {len(scan_result.get('vulnerabilities', []))} vulnerabilities\n\n"
            
            elif scan_name == "bandit_scan":
                issues = scan_result.get("results", [])
                if not issues:
                    report += "✅ No security issues found in static analysis\n\n"
                else:
                    report += f"❌ Found {len(issues)} potential security issues\n\n"
            
            elif scan_name == "configuration_security":
                issues = scan_result.get("issues", [])
                if not issues:
                    report += "✅ No configuration security issues found\n\n"
                else:
                    report += f"❌ Found {len(issues)} configuration issues\n\n"
                    for issue in issues[:5]:  # Show first 5
                        report += f"- **{issue['severity'].upper()}**: {issue['issue']} in `{issue['file']}`\n"
                    report += "\n"
        
        report += """
## Recommendations

1. **Critical Issues**: Address immediately before production deployment
2. **High Issues**: Fix within 24 hours
3. **Medium Issues**: Address within 1 week
4. **Low Issues**: Address in next maintenance cycle

## Next Steps

1. Review detailed findings in individual scan outputs
2. Implement fixes based on severity
3. Re-run security scan to verify fixes
4. Set up automated security scanning in CI/CD pipeline

---
*Generated by R3MES Security Scanner*
"""
        
        return report
    
    def run_all_scans(self) -> Dict:
        """Run all security scans and generate report."""
        print("🚀 Starting comprehensive security scan...")
        print("=" * 50)
        
        # Run dependency scan
        self.results["scans"]["dependency_scan"] = self.run_dependency_scan()
        
        # Run static analysis
        self.results["scans"]["bandit_scan"] = self.run_bandit_scan()
        
        # Run Docker image scan
        self.results["scans"]["docker_scan"] = self.scan_docker_images()
        
        # Check configuration security
        self.results["scans"]["configuration_security"] = self.check_configuration_security()
        
        # Check file permissions
        self.results["scans"]["file_permissions"] = self.check_file_permissions()
        
        # Calculate summary
        for scan_result in self.results["scans"].values():
            if isinstance(scan_result, dict) and "issues" in scan_result:
                for issue in scan_result["issues"]:
                    severity = issue.get("severity", "info")
                    self.results["summary"][severity] += 1
        
        # Generate and save report
        report_content = self.generate_report()
        report_file = self.project_root / "SECURITY_REPORT.md"
        with open(report_file, "w") as f:
            f.write(report_content)
        
        print(f"📊 Security report saved to {report_file}")
        
        # Save detailed results
        results_file = self.project_root / "security_scan_results.json"
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2)
        
        print(f"📋 Detailed results saved to {results_file}")
        
        return self.results

def main():
    """Main security scanning workflow."""
    project_root = Path(__file__).parent.parent
    scanner = SecurityScanner(project_root)
    
    results = scanner.run_all_scans()
    
    # Exit with error code if critical or high severity issues found
    critical = results["summary"]["critical"]
    high = results["summary"]["high"]
    
    if critical > 0:
        print(f"\n❌ CRITICAL: {critical} critical security issues found!")
        return 2
    elif high > 0:
        print(f"\n⚠️  WARNING: {high} high severity security issues found!")
        return 1
    else:
        print("\n✅ No critical or high severity security issues found!")
        return 0

if __name__ == "__main__":
    sys.exit(main())