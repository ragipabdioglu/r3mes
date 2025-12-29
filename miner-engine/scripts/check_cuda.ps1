# R3MES CUDA Auto-Detection Script for Windows
# This script checks NVIDIA driver, CUDA toolkit, and PyTorch CUDA support

Write-Host "R3MES CUDA Auto-Detection" -ForegroundColor Cyan
Write-Host "=" * 50

# Check nvidia-smi availability
Write-Host "`n[1/4] Checking NVIDIA Driver..." -ForegroundColor Yellow
try {
    $nvidiaSmi = & nvidia-smi --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ NVIDIA Driver detected" -ForegroundColor Green
        $nvidiaSmi | Select-Object -First 1
    } else {
        Write-Host "❌ nvidia-smi not found. Please install NVIDIA drivers." -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "❌ nvidia-smi not found. Please install NVIDIA drivers." -ForegroundColor Red
    exit 1
}

# Check CUDA version from nvidia-smi
Write-Host "`n[2/4] Checking CUDA Version..." -ForegroundColor Yellow
try {
    $nvidiaSmiOutput = & nvidia-smi 2>&1
    $cudaVersionLine = $nvidiaSmiOutput | Select-String "CUDA Version"
    if ($cudaVersionLine) {
        $cudaVersion = ($cudaVersionLine -split "CUDA Version:")[1].Trim().Split()[0]
        Write-Host "✅ CUDA Version: $cudaVersion" -ForegroundColor Green
    } else {
        Write-Host "⚠️  CUDA version not found in nvidia-smi output" -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠️  Could not determine CUDA version" -ForegroundColor Yellow
}

# Check CUDA toolkit installation
Write-Host "`n[3/4] Checking CUDA Toolkit Installation..." -ForegroundColor Yellow
$cudaPaths = @(
    "$env:ProgramFiles\NVIDIA GPU Computing Toolkit\CUDA",
    "${env:ProgramFiles(x86)}\NVIDIA GPU Computing Toolkit\CUDA"
)

$cudaFound = $false
foreach ($basePath in $cudaPaths) {
    if (Test-Path $basePath) {
        $cudaVersions = Get-ChildItem $basePath -Directory | Sort-Object Name -Descending
        if ($cudaVersions) {
            $latestCuda = $cudaVersions[0].Name
            Write-Host "✅ CUDA Toolkit found: $latestCuda at $basePath\$latestCuda" -ForegroundColor Green
            $cudaFound = $true
            
            # Add to PATH if not already there
            $cudaBinPath = "$basePath\$latestCuda\bin"
            if ($env:PATH -notlike "*$cudaBinPath*") {
                Write-Host "   Adding CUDA to PATH..." -ForegroundColor Yellow
                $env:PATH = "$cudaBinPath;$env:PATH"
                [Environment]::SetEnvironmentVariable("PATH", $env:PATH, [EnvironmentVariableTarget]::User)
            }
            break
        }
    }
}

if (-not $cudaFound) {
    Write-Host "⚠️  CUDA Toolkit not found in standard locations" -ForegroundColor Yellow
    Write-Host "   You may need to install CUDA Toolkit from NVIDIA" -ForegroundColor Yellow
}

# Check PyTorch CUDA support
Write-Host "`n[4/4] Checking PyTorch CUDA Support..." -ForegroundColor Yellow
try {
    $pythonCheck = python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); print('CUDA Version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ PyTorch CUDA check:" -ForegroundColor Green
        $pythonCheck
    } else {
        Write-Host "❌ PyTorch not installed or CUDA not available" -ForegroundColor Red
        Write-Host "   Install PyTorch with CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu121" -ForegroundColor Yellow
    }
} catch {
    Write-Host "❌ Could not check PyTorch CUDA support" -ForegroundColor Red
    Write-Host "   Make sure Python and PyTorch are installed" -ForegroundColor Yellow
}

Write-Host "`n" + ("=" * 50)
Write-Host "CUDA Auto-Detection Complete" -ForegroundColor Cyan

