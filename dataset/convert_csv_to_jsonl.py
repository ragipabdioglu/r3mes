#!/usr/bin/env python3
"""
CSV to JSONL Converter for BitNet Training

Converts haberler.csv to JSONL format suitable for BitNet LLM training.
"""

import csv
import json
import sys
from pathlib import Path
from typing import Dict, Any


def convert_csv_to_jsonl(
    input_file: str,
    output_file: str,
    text_column: str = "haber",
    category_column: str = "kategori",
    include_category: bool = True,
    format_type: str = "text"
):
    """
    Convert CSV file to JSONL format for BitNet training.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output JSONL file
        text_column: Name of the column containing text data
        category_column: Name of the category column (optional)
        include_category: Whether to include category in output
        format_type: Output format type ("text" or "instruction")
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    if not input_path.exists():
        print(f"❌ Error: Input file not found: {input_file}")
        sys.exit(1)
    
    print(f"📖 Reading CSV file: {input_file}")
    print(f"📝 Writing JSONL file: {output_file}")
    print(f"📊 Format: {format_type}")
    print(f"🏷️  Include category: {include_category}")
    print("-" * 60)
    
    total_rows = 0
    skipped_rows = 0
    
    with open(input_path, 'r', encoding='utf-8') as csvfile, \
         open(output_path, 'w', encoding='utf-8') as jsonlfile:
        
        reader = csv.DictReader(csvfile)
        
        for row_num, row in enumerate(reader, start=1):
            # Skip empty rows
            if not row.get(text_column, '').strip():
                skipped_rows += 1
                continue
            
            text = row[text_column].strip()
            category = row.get(category_column, '').strip() if include_category else None
            
            # Create JSON object based on format type
            if format_type == "text":
                # Simple text format (most common for LLM training)
                json_obj = {
                    "text": text
                }
                if include_category and category:
                    json_obj["category"] = category
                    
            elif format_type == "instruction":
                # Instruction format (for fine-tuning with categories)
                json_obj = {
                    "instruction": "Bu haberin kategorisini belirle:",
                    "input": text,
                    "output": category if category else "genel"
                }
            else:
                # Default: text format
                json_obj = {"text": text}
                if include_category and category:
                    json_obj["category"] = category
            
            # Write JSON line
            jsonlfile.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
            total_rows += 1
            
            # Progress indicator
            if total_rows % 1000 == 0:
                print(f"✅ Processed {total_rows} rows...")
    
    print("-" * 60)
    print(f"✅ Conversion complete!")
    print(f"   Total rows: {total_rows}")
    print(f"   Skipped rows: {skipped_rows}")
    print(f"   Output file: {output_file}")
    print(f"   File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert CSV to JSONL format for BitNet training"
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        default="haberler.csv",
        help="Input CSV file (default: haberler.csv)"
    )
    parser.add_argument(
        "-o", "--output",
        default="haberler.jsonl",
        help="Output JSONL file (default: haberler.jsonl)"
    )
    parser.add_argument(
        "--text-column",
        default="haber",
        help="Name of the text column (default: haber)"
    )
    parser.add_argument(
        "--category-column",
        default="kategori",
        help="Name of the category column (default: kategori)"
    )
    parser.add_argument(
        "--no-category",
        action="store_true",
        help="Don't include category in output"
    )
    parser.add_argument(
        "--format",
        choices=["text", "instruction"],
        default="text",
        help="Output format type (default: text)"
    )
    
    args = parser.parse_args()
    
    convert_csv_to_jsonl(
        input_file=args.input_file,
        output_file=args.output,
        text_column=args.text_column,
        category_column=args.category_column,
        include_category=not args.no_category,
        format_type=args.format
    )


if __name__ == "__main__":
    main()

