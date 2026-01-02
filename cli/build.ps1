# R3MES CLI Build Script for Windows
# Builds CLI for multiple platforms

param(
    [string]$Version = "v0.1.0"
)

Write-Host "🔨 Building R3MES CLI..." -ForegroundColor Blue

# Build information
$BuildTime = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
$GitCommit = try { git rev-parse --short HEAD } catch { "unknown" }

# Build flags
$LdFlags = "-X main.Version=$Version -X main.BuildTime=$BuildTime -X main.GitCommit=$GitCommit"

# Create build directory
$BuildDir = "build"
if (!(Test-Path $BuildDir)) {
    New-Item -ItemType Directory -Path $BuildDir | Out-Null
}

Write-Host "Build Information:" -ForegroundColor Blue
Write-Host "  Version: $Version" -ForegroundColor Green
Write-Host "  Build Time: $BuildTime" -ForegroundColor Green
Write-Host "  Git Commit: $GitCommit" -ForegroundColor Green
Write-Host ""

# Change to CLI directory
Set-Location "cli\r3mes-cli"

# Download dependencies
Write-Host "📦 Downloading dependencies..." -ForegroundColor Yellow
go mod download
go mod tidy

# Build for different platforms
$platforms = @(
    @{OS="linux"; ARCH="amd64"},
    @{OS="linux"; ARCH="arm64"},
    @{OS="darwin"; ARCH="amd64"},
    @{OS="darwin"; ARCH="arm64"},
    @{OS="windows"; ARCH="amd64"}
)

foreach ($platform in $platforms) {
    $GOOS = $platform.OS
    $GOARCH = $platform.ARCH
    
    $outputName = "r3mes-cli-$Version-$GOOS-$GOARCH"
    if ($GOOS -eq "windows") {
        $outputName += ".exe"
    }
    
    Write-Host "🔨 Building for $GOOS/$GOARCH..." -ForegroundColor Yellow
    
    $env:GOOS = $GOOS
    $env:GOARCH = $GOARCH
    
    $buildResult = go build -ldflags $LdFlags -o "..\..\$BuildDir\$outputName" .
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Successfully built $outputName" -ForegroundColor Green
    } else {
        Write-Host "❌ Failed to build for $GOOS/$GOARCH" -ForegroundColor Red
        exit 1
    }
}

# Create checksums
Set-Location "..\..\$BuildDir"
Write-Host "📝 Creating checksums..." -ForegroundColor Yellow

$files = Get-ChildItem "r3mes-cli-*"
$checksums = @()
foreach ($file in $files) {
    $hash = Get-FileHash $file.Name -Algorithm SHA256
    $checksums += "$($hash.Hash.ToLower())  $($file.Name)"
}
$checksums | Out-File -FilePath "checksums.txt" -Encoding UTF8

Write-Host ""
Write-Host "🎉 Build completed successfully!" -ForegroundColor Green
Write-Host "Built binaries:" -ForegroundColor Blue
Get-ChildItem "r3mes-cli-*" | Format-Table Name, Length, LastWriteTime

Write-Host ""
Write-Host "Usage:" -ForegroundColor Blue
Write-Host "  Windows: r3mes-cli-$Version-windows-amd64.exe --help" -ForegroundColor Green
Write-Host "  Linux: ./r3mes-cli-$Version-linux-amd64 --help" -ForegroundColor Green