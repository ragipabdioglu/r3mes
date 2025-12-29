# R3MES Windows Setup Script
# Automated setup for Windows users

Write-Host "R3MES Miner Setup for Windows" -ForegroundColor Cyan
Write-Host "=" * 50

# Check Python installation
Write-Host "`n[1/5] Checking Python Installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ $pythonVersion" -ForegroundColor Green
        
        # Check Python version (3.10+)
        $versionMatch = $pythonVersion -match "Python (\d+)\.(\d+)"
        if ($versionMatch) {
            $major = [int]$matches[1]
            $minor = [int]$matches[2]
            if ($major -ge 3 -and $minor -ge 10) {
                Write-Host "   Python version is compatible (3.10+)" -ForegroundColor Green
            } else {
                Write-Host "❌ Python 3.10+ required (found $major.$minor)" -ForegroundColor Red
                exit 1
            }
        }
    } else {
        Write-Host "❌ Python not found. Please install Python 3.10+ from python.org" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "❌ Python not found. Please install Python 3.10+ from python.org" -ForegroundColor Red
    exit 1
}

# Check CUDA
Write-Host "`n[2/5] Checking CUDA..." -ForegroundColor Yellow
& "$PSScriptRoot\check_cuda.ps1"
if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠️  CUDA check failed, but continuing..." -ForegroundColor Yellow
}

# Install/Upgrade pip
Write-Host "`n[3/5] Installing/Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to upgrade pip" -ForegroundColor Red
    exit 1
}

# Install R3MES package
Write-Host "`n[4/5] Installing R3MES package..." -ForegroundColor Yellow
$packagePath = Split-Path -Parent $PSScriptRoot
Set-Location $packagePath
pip install -e .
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to install R3MES package" -ForegroundColor Red
    exit 1
}
Write-Host "✅ R3MES package installed" -ForegroundColor Green

# Run setup wizard
Write-Host "`n[5/5] Running Setup Wizard..." -ForegroundColor Yellow
r3mes-miner setup
if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠️  Setup wizard completed with warnings" -ForegroundColor Yellow
}

Write-Host "`n" + ("=" * 50)
Write-Host "✅ Setup Complete!" -ForegroundColor Green
Write-Host "`nNext steps:"
Write-Host "  1. Make sure your blockchain node is running"
Write-Host "  2. Run 'r3mes-miner start' to begin mining"
Write-Host "  3. Run 'r3mes-miner status' to check your configuration"

