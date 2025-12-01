# Dapr Installation Script for Windows
# This script installs Dapr CLI and initializes it in standalone mode

Write-Host "Installing Dapr CLI for EL-AMANECER..." -ForegroundColor Cyan

# Check if Dapr is already installed
$daprInstalled = Get-Command dapr -ErrorAction SilentlyContinue

if ($daprInstalled) {
    Write-Host "Dapr CLI already installed" -ForegroundColor Green
    dapr --version
}
else {
    Write-Host "Downloading and installing Dapr CLI..." -ForegroundColor Yellow
    
    # Install Dapr CLI using the official installer
    try {
        $installScript = "$env:TEMP\install-dapr.ps1"
        Invoke-WebRequest -Uri "https://raw.githubusercontent.com/dapr/cli/master/install/install.ps1" -OutFile $installScript
        & $installScript
        Remove-Item $installScript
        
        Write-Host "Dapr CLI installed successfully" -ForegroundColor Green
    }
    catch {
        Write-Host "Failed to install Dapr CLI: $_" -ForegroundColor Red
        exit 1
    }
}

# Initialize Dapr in standalone mode
Write-Host ""
Write-Host "Initializing Dapr in standalone mode..." -ForegroundColor Cyan

try {
    dapr init
    Write-Host "Dapr initialized successfully" -ForegroundColor Green
}
catch {
    Write-Host "Failed to initialize Dapr: $_" -ForegroundColor Red
    exit 1
}

# Verify installation
Write-Host ""
Write-Host "Verifying Dapr installation..." -ForegroundColor Cyan

$daprStatus = dapr --version
Write-Host $daprStatus -ForegroundColor White

# Check if Redis is running (required for state store)
Write-Host ""
Write-Host "Checking Redis..." -ForegroundColor Cyan
$redisRunning = docker ps --filter "name=dapr_redis" --format "{{.Names}}"

if ($redisRunning) {
    Write-Host "Redis is running (Dapr state store)" -ForegroundColor Green
}
else {
    Write-Host "Redis not detected. State store may not work." -ForegroundColor Yellow
}

# Check if Zipkin is running (optional, for tracing)
Write-Host ""
Write-Host "Checking Zipkin..." -ForegroundColor Cyan
$zipkinRunning = docker ps --filter "name=dapr_zipkin" --format "{{.Names}}"

if ($zipkinRunning) {
    Write-Host "Zipkin is running (Dapr tracing)" -ForegroundColor Green
}
else {
    Write-Host "Zipkin not running (optional)" -ForegroundColor Gray
}

Write-Host ""
Write-Host "Dapr installation complete!" -ForegroundColor Green
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Run: python scripts/verify_dapr.py" -ForegroundColor White
Write-Host "  2. Test with: dapr run --app-id test --dapr-http-port 3500" -ForegroundColor White
