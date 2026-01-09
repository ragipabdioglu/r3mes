# R3MES Windows Service Installation Script
# Run as Administrator

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("install", "uninstall", "start", "stop", "status")]
    [string]$Action = "install"
)

$ServiceName = "R3MES"
$ServiceDisplayName = "R3MES Network Service"
$ServiceDescription = "R3MES Decentralized AI Training Network - Node, Miner, and Backend Services"
$R3MESHome = "$env:LOCALAPPDATA\R3MES"
$ExecutablePath = "$R3MESHome\bin\r3mes.exe"
$LogPath = "$R3MESHome\logs"

function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$timestamp] [$Level] $Message"
}

function Test-Administrator {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Install-R3MESService {
    if (-not (Test-Administrator)) {
        Write-Log "This script requires Administrator privileges. Please run as Administrator." "ERROR"
        exit 1
    }

    # Check if service already exists
    $existingService = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
    if ($existingService) {
        Write-Log "Service '$ServiceName' already exists. Use 'uninstall' first to reinstall." "WARN"
        return
    }

    # Create directories
    if (-not (Test-Path $R3MESHome)) {
        New-Item -ItemType Directory -Path $R3MESHome -Force | Out-Null
        Write-Log "Created R3MES home directory: $R3MESHome"
    }

    if (-not (Test-Path $LogPath)) {
        New-Item -ItemType Directory -Path $LogPath -Force | Out-Null
        Write-Log "Created log directory: $LogPath"
    }

    # Check if executable exists
    if (-not (Test-Path $ExecutablePath)) {
        Write-Log "R3MES executable not found at: $ExecutablePath" "ERROR"
        Write-Log "Please install R3MES first using the installer." "ERROR"
        exit 1
    }

    # Create the Windows service using NSSM (Non-Sucking Service Manager) or sc.exe
    try {
        # Try using NSSM if available (recommended)
        $nssmPath = Get-Command nssm -ErrorAction SilentlyContinue
        if ($nssmPath) {
            & nssm install $ServiceName $ExecutablePath "start --all"
            & nssm set $ServiceName DisplayName $ServiceDisplayName
            & nssm set $ServiceName Description $ServiceDescription
            & nssm set $ServiceName AppDirectory $R3MESHome
            & nssm set $ServiceName AppStdout "$LogPath\service.log"
            & nssm set $ServiceName AppStderr "$LogPath\service-error.log"
            & nssm set $ServiceName AppRotateFiles 1
            & nssm set $ServiceName AppRotateBytes 10485760
            & nssm set $ServiceName Start SERVICE_AUTO_START
            & nssm set $ServiceName AppEnvironmentExtra "R3MES_HOME=$R3MESHome" "R3MES_ENV=production"
            Write-Log "Service installed successfully using NSSM"
        }
        else {
            # Fallback to sc.exe (basic service creation)
            $binPath = "`"$ExecutablePath`" start --all --service"
            sc.exe create $ServiceName binPath= $binPath start= auto DisplayName= $ServiceDisplayName
            sc.exe description $ServiceName $ServiceDescription
            sc.exe failure $ServiceName reset= 86400 actions= restart/10000/restart/30000/restart/60000
            Write-Log "Service installed successfully using sc.exe"
            Write-Log "Note: For better service management, consider installing NSSM (nssm.cc)" "WARN"
        }

        # Set service recovery options
        sc.exe failure $ServiceName reset= 86400 actions= restart/10000/restart/30000/restart/60000

        Write-Log "R3MES service installed successfully!"
        Write-Log "Use 'r3mes-service.ps1 -Action start' to start the service"
    }
    catch {
        Write-Log "Failed to install service: $_" "ERROR"
        exit 1
    }
}

function Uninstall-R3MESService {
    if (-not (Test-Administrator)) {
        Write-Log "This script requires Administrator privileges." "ERROR"
        exit 1
    }

    $existingService = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
    if (-not $existingService) {
        Write-Log "Service '$ServiceName' does not exist." "WARN"
        return
    }

    # Stop service if running
    if ($existingService.Status -eq "Running") {
        Write-Log "Stopping service..."
        Stop-Service -Name $ServiceName -Force
        Start-Sleep -Seconds 2
    }

    # Remove service
    try {
        $nssmPath = Get-Command nssm -ErrorAction SilentlyContinue
        if ($nssmPath) {
            & nssm remove $ServiceName confirm
        }
        else {
            sc.exe delete $ServiceName
        }
        Write-Log "Service uninstalled successfully!"
    }
    catch {
        Write-Log "Failed to uninstall service: $_" "ERROR"
        exit 1
    }
}

function Start-R3MESService {
    $service = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
    if (-not $service) {
        Write-Log "Service '$ServiceName' not found. Please install first." "ERROR"
        exit 1
    }

    if ($service.Status -eq "Running") {
        Write-Log "Service is already running."
        return
    }

    Write-Log "Starting R3MES service..."
    Start-Service -Name $ServiceName
    Start-Sleep -Seconds 3

    $service = Get-Service -Name $ServiceName
    if ($service.Status -eq "Running") {
        Write-Log "Service started successfully!"
    }
    else {
        Write-Log "Failed to start service. Check logs at: $LogPath" "ERROR"
    }
}

function Stop-R3MESService {
    $service = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
    if (-not $service) {
        Write-Log "Service '$ServiceName' not found." "ERROR"
        exit 1
    }

    if ($service.Status -eq "Stopped") {
        Write-Log "Service is already stopped."
        return
    }

    Write-Log "Stopping R3MES service..."
    Stop-Service -Name $ServiceName -Force
    Start-Sleep -Seconds 2

    $service = Get-Service -Name $ServiceName
    if ($service.Status -eq "Stopped") {
        Write-Log "Service stopped successfully!"
    }
    else {
        Write-Log "Service may still be stopping..." "WARN"
    }
}

function Get-R3MESServiceStatus {
    $service = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
    if (-not $service) {
        Write-Log "Service '$ServiceName' not found."
        return
    }

    Write-Host ""
    Write-Host "=== R3MES Service Status ===" -ForegroundColor Cyan
    Write-Host "Name:         $($service.Name)"
    Write-Host "Display Name: $($service.DisplayName)"
    Write-Host "Status:       $($service.Status)" -ForegroundColor $(if ($service.Status -eq "Running") { "Green" } else { "Yellow" })
    Write-Host "Start Type:   $($service.StartType)"
    Write-Host ""

    # Show recent log entries
    $logFile = "$LogPath\service.log"
    if (Test-Path $logFile) {
        Write-Host "=== Recent Log Entries ===" -ForegroundColor Cyan
        Get-Content $logFile -Tail 10
    }
}

# Main execution
switch ($Action) {
    "install"   { Install-R3MESService }
    "uninstall" { Uninstall-R3MESService }
    "start"     { Start-R3MESService }
    "stop"      { Stop-R3MESService }
    "status"    { Get-R3MESServiceStatus }
}
