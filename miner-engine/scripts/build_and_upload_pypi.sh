#!/usr/bin/env bash
set -euo pipefail

# R3MES Python package build helper
#
# This script:
#   1. Ensures build dependencies (build, twine) are installed
#   2. Cleans previous build artifacts
#   3. Builds source distribution (sdist) + wheel
#   4. Prints next-step instructions for manual upload
#
# IMPORTANT:
#   - This script DOES NOT upload anything by itself.
#   - Actual PyPI publication (pip install r3mes) is intentionally left
#     as a manual step to be done at the very end of the project.
#
# For uploads, prefer:
#   - TestPyPI first (smoke test)
#   - Then real PyPI with manual twine commands

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

echo "==> R3MES Python package build"
echo "Project root: ${ROOT_DIR}"

if ! command -v python >/dev/null 2>&1; then
  echo "ERROR: python not found in PATH."
  exit 1
fi

echo "==> Ensuring build dependencies are installed (pip, build, twine)..."
python -m pip install --upgrade pip >/dev/null
python -m pip install --upgrade build twine >/dev/null

echo "==> Cleaning previous build artifacts (dist/, build/, *.egg-info)..."
rm -rf dist build ./*.egg-info

echo "==> Building r3mes package (sdist + wheel)..."
python -m build

echo
echo "==> Build artifacts:"
ls -1 dist || true

echo
echo "Next steps (manual):"
echo "  1) Test upload to TestPyPI (recommended):"
echo "       python -m twine upload --repository testpypi dist/*"
echo "  2) Verify installation from TestPyPI in a clean venv:"
echo "       pip install -i https://test.pypi.org/simple/ r3mes"
echo "  3) When the project is READY FOR FINAL RELEASE, upload to real PyPI:"
echo "       python -m twine upload dist/*"
echo
echo "Note: Configure TWINE_USERNAME / TWINE_PASSWORD (or API token) or use a .pypirc file before upload."

