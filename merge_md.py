#!/usr/bin/env python3
"""
Basit bir Markdown birleştirme aracı.

Özellikler:
- Verilen .md dosyalarını birleştirir
- VEYA verilen bir klasör içindeki tüm .md dosyalarını bulup birleştirir
- Dosyalar ada göre sıralanır
- Her dosya arasına başlık ve ayraç ekler
- Tek bir çıktı dosyası üretir

Kullanım:
  Tek tek dosyalar:
    python merge_md.py cikti.md giris1.md giris2.md

  Klasörden otomatik:
    python merge_md.py cikti.md ./docs
"""

import sys
from pathlib import Path


def collect_md_files(paths: list[Path]) -> list[Path]:
    md_files: list[Path] = []

    for path in paths:
        if path.is_dir():
            md_files.extend(sorted(path.rglob("*.md")))
        elif path.is_file() and path.suffix == ".md":
            md_files.append(path)
        else:
            print(f"Uyarı: {path} geçersiz, atlanıyor.")

    return md_files


def merge_md(output_file: Path, md_files: list[Path]):
    with output_file.open("w", encoding="utf-8") as out:
        for md_file in md_files:
            out.write("\n\n---\n\n")
            out.write(f"# {md_file.name}\n\n")
            content = md_file.read_text(encoding="utf-8")
            out.write(content.rstrip() + "\n")


def main():
    if len(sys.argv) < 3:
        print("Kullanım: python merge_md.py cikti.md <md_dosyaları | klasör>")
        sys.exit(1)

    output = Path(sys.argv[1])
    inputs = [Path(p) for p in sys.argv[2:]]

    md_files = collect_md_files(inputs)

    if not md_files:
        print("Birleştirilecek .md dosyası bulunamadı.")
        sys.exit(1)

    merge_md(output, md_files)
    print(f"Birleştirme tamamlandı → {output} ({len(md_files)} dosya)")


if __name__ == "__main__":
    main()
