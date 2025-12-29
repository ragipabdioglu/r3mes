#!/usr/bin/env python3
"""
Debug Log Analysis Script

Analyzes debug log files to identify patterns, errors, and performance bottlenecks.
"""

import json
import os
import sys
import re
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict, Counter
from datetime import datetime


def parse_json_log_line(line: str) -> Dict[str, Any]:
    """Parse a JSON-formatted log line"""
    try:
        return json.loads(line.strip())
    except json.JSONDecodeError:
        return None


def parse_text_log_line(line: str) -> Dict[str, Any]:
    """Parse a text-formatted log line (basic parsing)"""
    # Basic regex for common log formats
    # [timestamp] LEVEL message [key=value]...
    pattern = r'\[(.*?)\]\s+(\w+)\s+(.*?)(?:\s+\[(.*?)\])?$'
    match = re.match(pattern, line.strip())
    if match:
        timestamp_str, level, message, extra = match.groups()
        result = {
            "timestamp": timestamp_str,
            "level": level,
            "message": message,
        }
        # Parse extra fields if present
        if extra:
            for kv in extra.split(','):
                if '=' in kv:
                    k, v = kv.split('=', 1)
                    result[k.strip()] = v.strip()
        return result
    return None


def analyze_log_file(filepath: Path, log_format: str = "auto") -> Dict[str, Any]:
    """Analyze a log file"""
    stats = {
        "file": str(filepath),
        "total_lines": 0,
        "parsed_lines": 0,
        "level_counts": Counter(),
        "error_messages": [],
        "warning_messages": [],
        "function_call_counts": Counter(),
        "performance_entries": [],
        "unique_errors": set(),
    }
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                stats["total_lines"] += 1
                
                # Auto-detect format
                if log_format == "auto":
                    if line.strip().startswith('{'):
                        log_format = "json"
                    else:
                        log_format = "text"
                
                # Parse line
                if log_format == "json":
                    entry = parse_json_log_line(line)
                else:
                    entry = parse_text_log_line(line)
                
                if not entry:
                    continue
                
                stats["parsed_lines"] += 1
                
                # Count levels
                level = entry.get("level", "").upper()
                stats["level_counts"][level] += 1
                
                # Collect errors
                if level in ("ERROR", "FATAL", "CRITICAL"):
                    message = entry.get("message", "")
                    stats["error_messages"].append({
                        "line": line_num,
                        "message": message,
                        "timestamp": entry.get("timestamp", ""),
                    })
                    stats["unique_errors"].add(message[:100])  # First 100 chars
                
                # Collect warnings
                if level == "WARN":
                    stats["warning_messages"].append({
                        "line": line_num,
                        "message": entry.get("message", ""),
                        "timestamp": entry.get("timestamp", ""),
                    })
                
                # Extract function/operation names
                message = entry.get("message", "")
                if "function" in entry:
                    stats["function_call_counts"][entry["function"]] += 1
                elif "operation" in entry:
                    stats["function_call_counts"][entry["operation"]] += 1
                
                # Collect performance entries
                if "duration" in entry or "duration_ms" in entry:
                    stats["performance_entries"].append(entry)
                
    except Exception as e:
        stats["error"] = str(e)
    
    # Convert set to list for JSON serialization
    stats["unique_errors"] = list(stats["unique_errors"])
    stats["level_counts"] = dict(stats["level_counts"])
    stats["function_call_counts"] = dict(stats["function_call_counts"])
    
    return stats


def generate_report(stats_list: List[Dict[str, Any]]) -> str:
    """Generate a human-readable report"""
    report = []
    report.append("=" * 80)
    report.append("R3MES Debug Log Analysis Report")
    report.append("=" * 80)
    report.append(f"Generated: {datetime.now().isoformat()}")
    report.append("")
    
    for stats in stats_list:
        report.append(f"\nFile: {stats['file']}")
        report.append("-" * 80)
        report.append(f"Total lines: {stats['total_lines']}")
        report.append(f"Parsed lines: {stats['parsed_lines']}")
        report.append("")
        
        # Level distribution
        report.append("Log Level Distribution:")
        for level, count in sorted(stats['level_counts'].items()):
            percentage = (count / stats['parsed_lines'] * 100) if stats['parsed_lines'] > 0 else 0
            report.append(f"  {level:8s}: {count:6d} ({percentage:5.1f}%)")
        report.append("")
        
        # Errors
        error_count = len(stats['error_messages'])
        if error_count > 0:
            report.append(f"Errors found: {error_count}")
            report.append(f"Unique errors: {len(stats['unique_errors'])}")
            report.append("\nFirst 10 errors:")
            for i, err in enumerate(stats['error_messages'][:10], 1):
                report.append(f"  {i}. Line {err['line']}: {err['message'][:100]}")
            report.append("")
        
        # Warnings
        warning_count = len(stats['warning_messages'])
        if warning_count > 0:
            report.append(f"Warnings found: {warning_count}")
            report.append("")
        
        # Function calls
        if stats['function_call_counts']:
            report.append("Most called functions:")
            for func, count in sorted(stats['function_call_counts'].items(), key=lambda x: x[1], reverse=True)[:10]:
                report.append(f"  {func}: {count}")
            report.append("")
        
        # Performance entries
        perf_count = len(stats['performance_entries'])
        if perf_count > 0:
            report.append(f"Performance entries: {perf_count}")
            report.append("")
    
    report.append("=" * 80)
    return "\n".join(report)


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: analyze_debug_logs.py <log_file1> [log_file2] ...")
        print("       analyze_debug_logs.py --dir <log_directory>")
        sys.exit(1)
    
    log_files = []
    
    if sys.argv[1] == "--dir":
        # Analyze all log files in directory
        log_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path.home() / ".r3mes" / "logs"
        log_files = list(log_dir.glob("*.log"))
        if not log_files:
            print(f"No log files found in {log_dir}")
            sys.exit(1)
    else:
        # Analyze specified files
        log_files = [Path(f) for f in sys.argv[1:]]
    
    # Analyze each file
    stats_list = []
    for log_file in log_files:
        if not log_file.exists():
            print(f"Warning: File not found: {log_file}", file=sys.stderr)
            continue
        print(f"Analyzing {log_file}...", file=sys.stderr)
        stats = analyze_log_file(log_file)
        stats_list.append(stats)
    
    # Generate report
    report = generate_report(stats_list)
    print(report)
    
    # Save JSON summary
    output_file = Path.home() / ".r3mes" / "log_analysis.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(stats_list, f, indent=2)
    print(f"\nJSON summary saved to: {output_file}", file=sys.stderr)


if __name__ == "__main__":
    main()
