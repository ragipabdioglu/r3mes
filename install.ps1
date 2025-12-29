# R3MES Unified Installation Script (Windows PowerShell)
# Installs all required dependencies and components for R3MES

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

function Write-Success { Write-ColorOutput Green $args }
function Write-Warning { Write-ColorOutput Yellow $args }
function Write-Error { Write-ColorOutput Red $args }
function Write-Info { Write-ColorOutput Cyan $args }

# Check if command exists
function Test-Command {
    param($Command)
    $null -ne (Get-Command $Command -ErrorAction SilentlyContinue)
}

# Check and install Chocolatey
function Install-Chocolatey {
    if (Test-Command choco) {
        Write-Success "Chocolatey is already installed"
        return
    }
    
    Write-Info "Installing Chocolatey..."
    Set-ExecutionPolicy Bypass -Scope Process -Force
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
    iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
    Write-Success "Chocolatey installed successfully"
}

# Check and install Go
function Install-Go {
    Write-Info "Checking Go installation..."
    if (Test-Command go) {
        $goVersion = (go version).Split(' ')[2].Substring(2)
        Write-Success "Go is installed: $goVersion"
        
        $requiredVersion = "1.21"
        if ([version]$goVersion -lt [version]$requiredVersion) {
            Write-Warning "Go version $goVersion is too old. Required: >= $requiredVersion"
            Write-Info "Installing Go via Chocolatey..."
            choco install golang -y
        }
    } else {
        Write-Warning "Go is not installed. Installing..."
        choco install golang -y
        Write-Success "Go installed successfully"
    }
}

# Check and install Python
function Install-Python {
    Write-Info "Checking Python installation..."
    if (Test-Command python) {
        $pythonVersion = (python --version).Split(' ')[1]
        Write-Success "Python is installed: $pythonVersion"
        
        $requiredVersion = "3.10"
        if ([version]$pythonVersion -lt [version]$requiredVersion) {
            Write-Warning "Python version $pythonVersion is too old. Required: >= $requiredVersion"
            Write-Info "Installing Python via Chocolatey..."
            choco install python311 -y
        }
    } else {
        Write-Warning "Python is not installed. Installing..."
        choco install python311 -y
        Write-Success "Python installed successfully"
    }
}

# Check and install Node.js
function Install-NodeJS {
    Write-Info "Checking Node.js installation..."
    if (Test-Command node) {
        $nodeVersion = (node --version).Substring(1)
        Write-Success "Node.js is installed: $nodeVersion"
        
        $requiredVersion = "18.0.0"
        if ([version]$nodeVersion -lt [version]$requiredVersion) {
            Write-Warning "Node.js version $nodeVersion is too old. Required: >= $requiredVersion"
            Write-Info "Installing Node.js via Chocolatey..."
            choco install nodejs-lts -y
        }
    } else {
        Write-Warning "Node.js is not installed. Installing..."
        choco install nodejs-lts -y
        Write-Success "Node.js installed successfully"
    }
}

# Check GPU drivers
function Check-GPU {
    Write-Info "Checking GPU drivers..."
    
    # Check NVIDIA
    if (Test-Command nvidia-smi) {
        $nvidiaVersion = (nvidia-smi --query-gpu=driver_version --format=csv,noheader | Select-Object -First 1)
        Write-Success "NVIDIA GPU detected: Driver $nvidiaVersion"
        
        # Check CUDA
        if (Test-Command nvcc) {
            $cudaVersion = (nvcc --version | Select-String "release" | ForEach-Object { $_.Line.Split(' ')[5].TrimEnd(',') })
            Write-Success "CUDA installed: $cudaVersion"
        } else {
            Write-Warning "CUDA not found. Install from: https://developer.nvidia.com/cuda-downloads"
        }
    } else {
        Write-Warning "No NVIDIA GPU detected"
    }
    
    # Check AMD
    if (Test-Path "C:\Program Files\AMD\ROCm") {
        Write-Success "AMD GPU detected (ROCm)"
    }
    
    # Check Intel
    if (Test-Path "C:\Windows\System32\igfx*.dll") {
        Write-Success "Intel GPU detected"
    }
    
    if (-not (Test-Command nvidia-smi) -and -not (Test-Path "C:\Program Files\AMD\ROCm")) {
        Write-Warning "No GPU detected. R3MES can run in CPU-only mode, but mining will be slower."
    }
}

# Check and install Docker (optional)
function Install-Docker {
    Write-Info "Checking Docker installation..."
    if (Test-Command docker) {
        $dockerVersion = (docker --version).Split(' ')[2].TrimEnd(',')
        Write-Success "Docker is installed: $dockerVersion"
    } else {
        Write-Warning "Docker is not installed. (Optional - only needed for containerized deployments)"
        $installDocker = Read-Host "Install Docker? (y/n)"
        if ($installDocker -eq "y") {
            choco install docker-desktop -y
            Write-Success "Docker installed successfully"
        }
    }
}

# Check and install IPFS
function Install-IPFS {
    Write-Info "Checking IPFS installation..."
    if (Test-Command ipfs) {
        $ipfsVersion = (ipfs version --number)
        Write-Success "IPFS is installed: $ipfsVersion"
    } else {
        Write-Warning "IPFS is not installed. Installing..."
        choco install ipfs -y
        Write-Success "IPFS installed successfully"
    }
}

# Check disk space
function Check-DiskSpace {
    Write-Info "Checking disk space..."
    $drive = (Get-Location).Drive.Name
    $disk = Get-WmiObject -Class Win32_LogicalDisk -Filter "DeviceID='$drive`:'"
    $availableGB = [math]::Round($disk.FreeSpace / 1GB, 2)
    $requiredGB = 50
    
    if ($availableGB -lt $requiredGB) {
        Write-Error "Insufficient disk space. Required: ${requiredGB}GB, Available: ${availableGB}GB"
        exit 1
    }
    
    Write-Success "Disk space OK: ${availableGB}GB available"
}

# Check RAM
function Check-RAM {
    Write-Info "Checking RAM..."
    $totalRAM = [math]::Round((Get-WmiObject -Class Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 2)
    $requiredRAM = 8
    
    if ($totalRAM -lt $requiredRAM) {
        Write-Warning "Low RAM detected: ${totalRAM}GB. Recommended: >= ${requiredRAM}GB"
    } else {
        Write-Success "RAM OK: ${totalRAM}GB"
    }
}

# Main installation function
function Main {
    Write-Success "=========================================="
    Write-Success "R3MES Unified Installation Script"
    Write-Success "=========================================="
    Write-Output ""
    
    Install-Chocolatey
    
    Write-Output ""
    Write-Info "Checking system requirements..."
    Check-DiskSpace
    Check-RAM
    Check-GPU
    
    Write-Output ""
    Write-Info "Installing dependencies..."
    Install-Go
    Install-Python
    Install-NodeJS
    Install-IPFS
    Install-Docker
    
    Write-Output ""
    Write-Success "=========================================="
    Write-Success "Installation Complete!"
    Write-Success "=========================================="
    Write-Output ""
    Write-Output "Next steps:"
    Write-Output "  1. Restart your terminal or PowerShell"
    Write-Output "  2. Navigate to R3MES directory"
    Write-Output "  3. Run component-specific install scripts:"
    Write-Output "     - scripts\install_founder.sh (for validators)"
    Write-Output "     - scripts\install_miner_pypi.sh (for miners)"
    Write-Output "     - scripts\install_web_dashboard.sh (for web dashboard)"
    Write-Output ""
}

# Run main function
Main

