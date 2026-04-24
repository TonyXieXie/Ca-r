$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root ".venv\Scripts\python.exe"
$script = Join-Path $root "play_carracing.py"

if (-not (Test-Path $python)) {
    throw "Python executable not found at $python. Create the local .venv first."
}

if (-not (Test-Path $script)) {
    throw "Launcher script not found at $script."
}

& $python $script @args
