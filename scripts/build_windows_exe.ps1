# Build Windows EXE Script for R3MES Desktop Launcher (PowerShell)
#
# Builds the Tauri desktop launcher as a Windows executable (.exe) or installer (.msi).
# Validates the build output and prepares it for distribution.

param(
    [ValidateSet("exe", "msi")]
    [string]$BuildType = "exe",
    [switch]$NoValidate,
    [string]$OutputDir = ""
)

$ErrorActionPreference = "Stop"

# Colors for output
function Write-ColorOutput($ForegroundColor) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) {
        Write-Output $args
    }
    $host.UI.RawUI.ForegroundColor = $fc
}

# Script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$LauncherDir = Join-Path $ProjectRoot "desktop-launcher-tauri"

if ([string]::IsNullOrEmpty($OutputDir)) {
    $OutputDir = Join-Path $LauncherDir "dist"
}

Write-Host "=========================================="
Write-Host "R3MES Desktop Launcher - Windows Build" -ForegroundColor Cyan
Write-Host "=========================================="
Write-Host "Build type: $BuildType"
Write-Host "Output directory: $OutputDir"
Write-Host "Working directory: $LauncherDir"
Write-Host "=========================================="
Write-Host ""

# Check prerequisites
Write-Host "Checking prerequisites..."

# Check Node.js
try {
    $NodeVersion = node --version
    Write-Host "✅ Node.js: $NodeVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Error: Node.js is not installed" -ForegroundColor Red
    exit 1
}

# Check npm
try {
    $NpmVersion = npm --version
    Write-Host "✅ npm: $NpmVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Error: npm is not installed" -ForegroundColor Red
    exit 1
}

# Check Rust
try {
    $RustVersion = rustc --version
    Write-Host "✅ Rust: $RustVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Error: Rust is not installed" -ForegroundColor Red
    Write-Host "Install from: https://rustup.rs/"
    exit 1
}

Write-Host ""

# Navigate to launcher directory
Set-Location $LauncherDir

# Install dependencies
Write-Host "Installing dependencies..."
if (-not (Test-Path "node_modules")) {
    npm install
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Error: Failed to install dependencies" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "✅ Dependencies already installed" -ForegroundColor Green
}
Write-Host ""

# Build frontend
Write-Host "Building frontend..."
npm run build
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Error: Frontend build failed" -ForegroundColor Red
    exit 1
}
Write-Host "✅ Frontend build completed" -ForegroundColor Green
Write-Host ""

# Build Tauri application
Write-Host "Building Tauri application..."
Write-Host "This may take several minutes..." -ForegroundColor Yellow

if ($BuildType -eq "msi") {
    # Build MSI installer
    npm run tauri build -- --bundles msi
} else {
    # Build EXE (default)
    npm run tauri build
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Error: Tauri build failed" -ForegroundColor Red
    exit 1
}
Write-Host "✅ Tauri build completed" -ForegroundColor Green
Write-Host ""

# Find output files
$BuildOutputDir = Join-Path $LauncherDir "src-tauri\target\release\bundle"
$ExeFile = $null
$MsiFile = $null

if (Test-Path $BuildOutputDir) {
    # Find EXE file
    $ExeFiles = Get-ChildItem -Path $BuildOutputDir -Filter "*.exe" -Recurse -File | Select-Object -First 1
    if ($ExeFiles) {
        $ExeFile = $ExeFiles.FullName
    }
    
    # Find MSI file
    if ($BuildType -eq "msi") {
        $MsiFiles = Get-ChildItem -Path (Join-Path $BuildOutputDir "msi") -Filter "*.msi" -File | Select-Object -First 1
        if ($MsiFiles) {
            $MsiFile = $MsiFiles.FullName
        }
    }
}

# Create output directory
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
}

# Copy output files
if ($ExeFile -and (Test-Path $ExeFile)) {
    $ExeName = Split-Path -Leaf $ExeFile
    Copy-Item $ExeFile $OutputDir
    Write-Host "✅ EXE copied to: $OutputDir\$ExeName" -ForegroundColor Green
    
    $ExeSize = (Get-Item $ExeFile).Length
    $ExeSizeMB = [math]::Round($ExeSize / 1MB, 2)
    Write-Host "   Size: $ExeSizeMB MB"
}

if ($MsiFile -and (Test-Path $MsiFile)) {
    $MsiName = Split-Path -Leaf $MsiFile
    Copy-Item $MsiFile $OutputDir
    Write-Host "✅ MSI copied to: $OutputDir\$MsiName" -ForegroundColor Green
    
    $MsiSize = (Get-Item $MsiFile).Length
    $MsiSizeMB = [math]::Round($MsiSize / 1MB, 2)
    Write-Host "   Size: $MsiSizeMB MB"
}

Write-Host ""

# Validate build (if enabled)
if (-not $NoValidate) {
    Write-Host "Validating build..."
    
    if ($ExeFile -and (Test-Path $ExeFile)) {
        # Check if file exists and is not empty
        $FileInfo = Get-Item $ExeFile
        if ($FileInfo.Length -eq 0) {
            Write-Host "❌ Error: EXE file is empty" -ForegroundColor Red
            exit 1
        }
        
        Write-Host "✅ EXE validation passed" -ForegroundColor Green
    }
    
    if ($MsiFile -and (Test-Path $MsiFile)) {
        $FileInfo = Get-Item $MsiFile
        if ($FileInfo.Length -eq 0) {
            Write-Host "❌ Error: MSI file is empty" -ForegroundColor Red
            exit 1
        }
        Write-Host "✅ MSI validation passed" -ForegroundColor Green
    }
} else {
    Write-Host "Skipping build validation" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=========================================="
Write-Host "✅ Build completed successfully!" -ForegroundColor Green
Write-Host "=========================================="
if ($ExeFile) {
    $ExeName = Split-Path -Leaf $ExeFile
    Write-Host "EXE: $OutputDir\$ExeName"
}
if ($MsiFile) {
    $MsiName = Split-Path -Leaf $MsiFile
    Write-Host "MSI: $OutputDir\$MsiName"
}
Write-Host ""
Write-Host "Next steps:"
Write-Host "1. Test the executable on a Windows system"
Write-Host "2. Create a code signature (recommended for distribution)"
Write-Host "3. Upload to release distribution platform"
Write-Host "=========================================="

