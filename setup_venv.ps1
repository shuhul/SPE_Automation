# Run this script once after installing Python 3.10 for all users.
# It rebuilds the shared .venv and installs all required packages.

$ErrorActionPreference = "Stop"
$Root = "C:\Users\Public\Shared Confocal Files\SPE_Automation"
$Python = "C:\Program Files\Python310\python.exe"
$Venv = "$Root\.venv"

if (-not (Test-Path $Python)) {
    Write-Error "Python 3.10 not found at '$Python'. Install it for all users first."
    exit 1
}

Write-Host "Python found: $((& $Python --version 2>&1))"

# Recreate venv
if (Test-Path $Venv) {
    Write-Host "Removing old venv..."
    Remove-Item -Recurse -Force $Venv
}
Write-Host "Creating new venv..."
& $Python -m venv $Venv

$Pip = "$Venv\Scripts\pip.exe"

# Install packages from requirements.txt
Write-Host "Installing packages..."
& $Pip install --upgrade pip
& $Pip install -r "$Root\requirements.txt"

# Install MATLAB engine from local MATLAB installation
Write-Host "Installing MATLAB engine..."
& $Pip install "C:\Program Files\MATLAB\R2025b\extern\engines\python"

Write-Host ""
Write-Host "Done! Select the kernel '.venv (Python 3.10)' in VS Code to use the notebook."
