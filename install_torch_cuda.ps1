$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root ".venv\Scripts\python.exe"
$wheelUrl = "https://download.pytorch.org/whl/cu130/torch-2.10.0%2Bcu130-cp312-cp312-win_amd64.whl"

if (-not (Test-Path $python)) {
    throw "Python executable not found at $python. Create the local .venv first."
}

Write-Output "Installing CUDA-enabled PyTorch from:"
Write-Output $wheelUrl

& $python -m pip install --upgrade --force-reinstall --no-deps $wheelUrl

if ($LASTEXITCODE -ne 0) {
    throw "CUDA PyTorch installation failed with exit code $LASTEXITCODE."
}

Write-Output "Installed CUDA-enabled PyTorch into .venv."
