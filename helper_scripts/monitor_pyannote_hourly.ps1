param(
    [string]$Root = "D:\medgemma_data\dementianet_updated_files",
    [string]$PythonExe = "C:\Users\victy\anaconda_new\envs\audio\python.exe",
    [int]$IntervalSeconds = 3600
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $Root

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}

$snapDir = Join-Path $Root "helper_scripts\progress_snapshots"
New-Item -ItemType Directory -Path $snapDir -Force | Out-Null

while ($true) {
    & $PythonExe helper_scripts\report_pyannote_progress.py --runs-root runs --out-json helper_scripts\pyannote_progress_live.json | Out-Host
    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    Copy-Item -LiteralPath "helper_scripts\pyannote_progress_live.json" -Destination (Join-Path $snapDir "pyannote_progress_$ts.json") -Force
    Start-Sleep -Seconds $IntervalSeconds
}

