param(
    [string]$Config = "configs/full_pipeline_config.yaml",
    [switch]$DryRun,
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $root

if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    $PythonExe = (Get-Command python).Source
}

$args = @("orchestrate_full_pipeline.py", "--config", $Config)
if ($DryRun) {
    $args += "--dry-run"
}

Write-Host "Using Python: $PythonExe"
Write-Host "Working dir: $root"
Write-Host "Config: $Config"

& $PythonExe @args
exit $LASTEXITCODE

