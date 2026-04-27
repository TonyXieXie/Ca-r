[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$RunName = "",
    [int]$TotalTimesteps = 1000000,
    [int]$NumEnvs = 4,
    [int]$NumSteps = 128,
    [int]$NumFrames = 1,
    [int]$ImageSize = 96,
    [ValidateSet("async", "sync")]
    [string]$VectorEnv = "async",
    [int]$EvalEvery = 5,
    [int]$EvalEpisodes = 3,
    [switch]$NoAsyncEval,
    [int]$MaxPendingEvals = 4,
    [ValidateSet("async", "sync")]
    [string]$EvalVectorEnv = "async",
    [switch]$Cpu,
    [switch]$CaptureVideo,
    [switch]$DomainRandomize,
    [switch]$FailOnNonFinite,
    [switch]$LrAnneal,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root ".venv\Scripts\python.exe"
$script = Join-Path $root "train_ppo_carracing.py"

if (-not (Test-Path $python)) {
    throw "Python executable not found at $python. Create the local .venv first."
}

if (-not (Test-Path $script)) {
    throw "Training script not found at $script."
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
if ([string]::IsNullOrWhiteSpace($RunName)) {
    $runFolder = "carracing_ppo_$timestamp"
} else {
    $sanitized = ($RunName -replace '[^a-zA-Z0-9_\-]', '_')
    $runFolder = "${sanitized}_$timestamp"
}

$saveDir = Join-Path $root "runs\$runFolder"
$logFile = Join-Path $saveDir "train.log"
$metricsFile = Join-Path $saveDir "metrics.jsonl"
New-Item -ItemType Directory -Path $saveDir -Force | Out-Null

$device = if ($Cpu) { "cpu" } else { "auto" }

$argsList = @(
    $script,
    "--save-dir", $saveDir,
    "--total-timesteps", "$TotalTimesteps",
    "--num-envs", "$NumEnvs",
    "--num-steps", "$NumSteps",
    "--num-frames", "$NumFrames",
    "--image-size", "$ImageSize",
    "--vector-env", "$VectorEnv",
    "--eval-every", "$EvalEvery",
    "--eval-episodes", "$EvalEpisodes",
    "--max-pending-evals", "$MaxPendingEvals",
    "--eval-vector-env", "$EvalVectorEnv",
    "--save-every", "10",
    "--log-every", "1",
    "--device", $device
)

if ($CaptureVideo) { $argsList += "--capture-video" }
if ($DomainRandomize) { $argsList += "--domain-randomize" }
if ($FailOnNonFinite) { $argsList += "--fail-on-nonfinite" }
if ($LrAnneal) { $argsList += "--lr-anneal" }
if ($NoAsyncEval) { $argsList += "--no-async-eval" }
if ($ExtraArgs) { $argsList += $ExtraArgs }

Write-Output "Run directory: $saveDir"
Write-Output "Log file: $logFile"
Write-Output "Metrics file: $metricsFile"

Push-Location $root
try {
    & $python @argsList 2>&1 | Tee-Object -FilePath $logFile
}
finally {
    Pop-Location
}

