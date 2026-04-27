[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$RunDir = "runs\\carracing_ppo_20260423_153611",
    [string]$Checkpoint = "",
    [int]$Episodes = 3,
    [int]$Seed = 10,
    [int]$Fps = 50,
    [int]$MaxSteps = 0,
    [switch]$Cpu,
    [switch]$NoRender,
    [switch]$Stochastic,
    [switch]$DomainRandomize,
    [switch]$NoDomainRandomize,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root ".venv\Scripts\python.exe"
$script = Join-Path $root "run_trained_carracing.py"

if (-not (Test-Path $python)) {
    throw "Python executable not found at $python. Create the local .venv first."
}

if (-not (Test-Path $script)) {
    throw "Runner script not found at $script."
}

$device = if ($Cpu) { "cpu" } else { "auto" }
$renderMode = if ($NoRender) { "none" } else { "human" }
$resolvedRunDir = if ([System.IO.Path]::IsPathRooted($RunDir)) { $RunDir } else { Join-Path $root $RunDir }

$argsList = @(
    $script,
    "--run-dir", $resolvedRunDir,
    "--episodes", "$Episodes",
    "--render-mode", $renderMode,
    "--device", $device,
    "--fps", "$Fps"
)

if (-not [string]::IsNullOrWhiteSpace($Checkpoint)) {
    $resolvedCheckpoint = if ([System.IO.Path]::IsPathRooted($Checkpoint)) { $Checkpoint } else { Join-Path $root $Checkpoint }
    $argsList += @("--checkpoint", $resolvedCheckpoint)
}

if ($Seed -ge 0) {
    $argsList += @("--seed", "$Seed")
}

if ($MaxSteps -gt 0) {
    $argsList += @("--max-steps", "$MaxSteps")
}

if ($Stochastic) { $argsList += "--stochastic" }
if ($DomainRandomize) { $argsList += "--domain-randomize" }
if ($NoDomainRandomize) { $argsList += "--no-domain-randomize" }
if ($ExtraArgs) { $argsList += $ExtraArgs }

Write-Output "Run directory: $resolvedRunDir"
if (-not [string]::IsNullOrWhiteSpace($Checkpoint)) {
    Write-Output "Checkpoint override: $resolvedCheckpoint"
}
Write-Output "Render mode: $renderMode"
Write-Output "Device: $device"

Push-Location $root
try {
    & $python @argsList
}
finally {
    Pop-Location
}
