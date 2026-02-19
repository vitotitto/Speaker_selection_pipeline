param(
    [string]$DataRoot = "D:\medgemma_data\dementianet_updated_files",
    [string]$PipelineRoot = "D:\medgemma_data\speaker_selection_pipeline",
    [string]$RunListFile = "D:\medgemma_data\dementianet_updated_files\helper_scripts\missing_control_stage1_runs_valid.txt",
    [string]$PythonExe = "C:\Users\victy\anaconda_new\envs\audio\python.exe",
    [string]$RunsRoot = "D:\medgemma_data\dementianet_updated_files\runs",
    [string]$CsvDir = "D:\medgemma_data\dementianet_updated_files\csv_sources",
    [string]$OutputRoot = "D:\medgemma_data\dementianet_updated_files\processed_7_continuity_full",
    [string]$ProgressJson = "D:\medgemma_data\dementianet_updated_files\helper_scripts\missing_control_overnight_progress.json",
    [string]$ProgressCsv = "D:\medgemma_data\dementianet_updated_files\helper_scripts\missing_control_overnight_progress.csv",
    [string]$LogFile = "D:\medgemma_data\dementianet_updated_files\helper_scripts\missing_control_overnight.log",
    [int]$PollSeconds = 60,
    [int]$MaxWaitHours = 24,
    [switch]$SkipPyannoteWait
)

$ErrorActionPreference = "Stop"

function Write-Log {
    param([string]$Message)
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "$ts - $Message"
    Write-Host $line
    Add-Content -Path $LogFile -Value $line -Encoding UTF8
}

function Save-Progress {
    param(
        [string]$Phase,
        [hashtable]$Metrics
    )

    $payload = [ordered]@{
        timestamp = (Get-Date).ToString("o")
        phase = $Phase
        metrics = $Metrics
    }
    $payload | ConvertTo-Json -Depth 6 | Set-Content -Path $ProgressJson -Encoding UTF8

    if (-not (Test-Path $ProgressCsv)) {
        "timestamp,phase,key,value" | Set-Content -Path $ProgressCsv -Encoding UTF8
    }
    foreach ($kv in $Metrics.GetEnumerator()) {
        $row = ('"{0}","{1}","{2}","{3}"' -f ((Get-Date).ToString("o")), $Phase, $kv.Key, $kv.Value)
        Add-Content -Path $ProgressCsv -Value $row -Encoding UTF8
    }
}

function Get-PyannoteProcesses {
    $needle = "*run_pyannote_api.py*missing_control_stage1_runs_valid.txt*"
    return Get-CimInstance Win32_Process |
        Where-Object { $_.CommandLine -like $needle } |
        Select-Object -ExpandProperty ProcessId
}

function Get-PyannoteState {
    param([string[]]$RunDirs)

    $counts = [ordered]@{
        total = $RunDirs.Count
        no_state = 0
        created = 0
        running = 0
        succeeded = 0
        failed = 0
        canceled = 0
        unknown = 0
        has_output = 0
    }

    foreach ($runDir in $RunDirs) {
        $meta = Join-Path $runDir "metadata"
        $statePath = Join-Path $meta "pyannote_job.json"
        $outputPath = Join-Path $meta "pyannote_job_output.json"

        if (Test-Path $outputPath) {
            $counts.has_output++
        }

        if (-not (Test-Path $statePath)) {
            $counts.no_state++
            continue
        }

        try {
            $j = Get-Content -Path $statePath -Raw | ConvertFrom-Json
            $status = [string]$j.status
            if ([string]::IsNullOrWhiteSpace($status)) {
                $counts.unknown++
                continue
            }
            $status = $status.ToLowerInvariant()
            if ($counts.ContainsKey($status)) {
                $counts[$status]++
            } else {
                $counts.unknown++
            }
        } catch {
            $counts.unknown++
        }
    }

    return $counts
}

function Invoke-PythonRunStage {
    param(
        [string]$StageName,
        [string]$ScriptPath,
        [string[]]$RunDirs,
        [scriptblock]$BuildArgs
    )

    $ok = 0
    $failed = 0
    $skipped = 0

    for ($i = 0; $i -lt $RunDirs.Count; $i++) {
        $runDir = $RunDirs[$i]
        $label = "[{0}/{1}]" -f ($i + 1), $RunDirs.Count
        $args = & $BuildArgs $runDir

        Write-Log "$StageName $label START $runDir"
        $started = Get-Date
        & $PythonExe $ScriptPath @args
        $exitCode = $LASTEXITCODE
        $elapsed = [math]::Round(((Get-Date) - $started).TotalSeconds, 2)

        if ($exitCode -eq 0) {
            $ok++
            Write-Log "$StageName $label OK (${elapsed}s)"
        } else {
            $failed++
            Write-Log "$StageName $label FAIL exit=$exitCode (${elapsed}s)"
        }

        Save-Progress -Phase "stage_$StageName" -Metrics @{
            stage = $StageName
            total = $RunDirs.Count
            done = ($i + 1)
            ok = $ok
            failed = $failed
            skipped = $skipped
        }
    }

    return @{
        ok = $ok
        failed = $failed
        skipped = $skipped
        total = $RunDirs.Count
    }
}

if (-not (Test-Path $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}
if (-not (Test-Path $RunListFile)) {
    throw "Run list not found: $RunListFile"
}

New-Item -ItemType Directory -Path (Split-Path -Parent $LogFile) -Force | Out-Null
New-Item -ItemType Directory -Path (Split-Path -Parent $ProgressJson) -Force | Out-Null
New-Item -ItemType Directory -Path (Split-Path -Parent $ProgressCsv) -Force | Out-Null

Write-Log "Starting missing_control overnight chain."
Write-Log "DataRoot=$DataRoot"
Write-Log "PipelineRoot=$PipelineRoot"
Write-Log "RunListFile=$RunListFile"

Set-Location -Path $DataRoot
$env:PYTHONPATH = $PipelineRoot

$runDirs = Get-Content -Path $RunListFile -Encoding UTF8 |
    ForEach-Object { $_.Trim() } |
    Where-Object { $_ -ne "" -and -not $_.StartsWith("#") } |
    Where-Object { Test-Path $_ }

if ($runDirs.Count -eq 0) {
    throw "No valid run directories found in $RunListFile"
}

Write-Log "Loaded $($runDirs.Count) run(s) from list."

$eligibleRuns = @()
$missingOutputRuns = @()

if ($SkipPyannoteWait) {
    Write-Log "SkipPyannoteWait enabled: using all listed runs without waiting for cloud outputs."
    $eligibleRuns = @($runDirs)
} else {
    $waitStart = Get-Date
    $stableNoStateTicks = 0

    while ($true) {
        $state = Get-PyannoteState -RunDirs $runDirs
        $activeProcIds = Get-PyannoteProcesses
        $activeProcCount = @($activeProcIds).Count

        Save-Progress -Phase "waiting_pyannote" -Metrics @{
            total = $state.total
            no_state = $state.no_state
            created = $state.created
            running = $state.running
            succeeded = $state.succeeded
            failed = $state.failed
            canceled = $state.canceled
            has_output = $state.has_output
            pyannote_processes = $activeProcCount
        }

        Write-Log ("pyannote wait status: total={0} no_state={1} created={2} running={3} succeeded={4} failed={5} canceled={6} has_output={7} proc={8}" -f `
            $state.total, $state.no_state, $state.created, $state.running, $state.succeeded, $state.failed, $state.canceled, $state.has_output, $activeProcCount)

        $allTerminalOrOutput = ($state.no_state -eq 0 -and $state.created -eq 0 -and $state.running -eq 0)
        if ($allTerminalOrOutput -and $activeProcCount -eq 0) {
            break
        }

        if ($activeProcCount -eq 0 -and $state.no_state -gt 0) {
            $stableNoStateTicks++
        } else {
            $stableNoStateTicks = 0
        }

        # If submission finished and no-state stalls for 20 minutes, continue with what we have.
        if ($stableNoStateTicks -ge [math]::Ceiling(1200 / [math]::Max($PollSeconds, 1))) {
            Write-Log "No pyannote process and no-state stalled for 20+ minutes. Continuing with completed outputs."
            break
        }

        $elapsedHours = ((Get-Date) - $waitStart).TotalHours
        if ($elapsedHours -ge $MaxWaitHours) {
            Write-Log "Reached MaxWaitHours=$MaxWaitHours. Continuing with completed outputs."
            break
        }

        Start-Sleep -Seconds $PollSeconds
    }

    foreach ($runDir in $runDirs) {
        $outputPath = Join-Path $runDir "metadata\\pyannote_job_output.json"
        if (Test-Path $outputPath) {
            $eligibleRuns += $runDir
        } else {
            $missingOutputRuns += $runDir
        }
    }
}

$eligibleListPath = Join-Path (Split-Path -Parent $ProgressJson) "missing_control_pyannote_succeeded.txt"
$missingListPath = Join-Path (Split-Path -Parent $ProgressJson) "missing_control_pyannote_missing_output.txt"
$eligibleRuns | Set-Content -Path $eligibleListPath -Encoding UTF8
$missingOutputRuns | Set-Content -Path $missingListPath -Encoding UTF8

Write-Log "Eligible after pyannote: $($eligibleRuns.Count) run(s). Missing pyannote output: $($missingOutputRuns.Count)."
Write-Log "Eligible list: $eligibleListPath"
Write-Log "Missing list: $missingListPath"

if ($eligibleRuns.Count -eq 0) {
    throw "No eligible runs with pyannote output. Stopping."
}

$screenScript = Join-Path $PipelineRoot "helper_scripts\\run_content_screening.py"
$speakerScript = Join-Path $PipelineRoot "helper_scripts\\run_speaker_analysis.py"
$auditScript = Join-Path $PipelineRoot "helper_scripts\\run_audit.py"
$extractScript = Join-Path $PipelineRoot "helper_scripts\\run_audio_extraction.py"

$screenResult = Invoke-PythonRunStage -StageName "content_screening" -ScriptPath $screenScript -RunDirs $eligibleRuns -BuildArgs {
    param($runDir)
    @(
        "--run-dir", $runDir,
        "--runs-root", $RunsRoot,
        "--csv-dir", $CsvDir,
        "--provider", "gemini",
        "--model", "gemini-3-flash-preview",
        "--force"
    )
}

$speakerResult = Invoke-PythonRunStage -StageName "speaker_analysis" -ScriptPath $speakerScript -RunDirs $eligibleRuns -BuildArgs {
    param($runDir)
    @(
        "--run-dir", $runDir,
        "--runs-root", $RunsRoot,
        "--csv-dir", $CsvDir,
        "--provider", "gemini",
        "--model", "gemini-2.0-flash",
        "--force"
    )
}

$auditResult = Invoke-PythonRunStage -StageName "audit" -ScriptPath $auditScript -RunDirs $eligibleRuns -BuildArgs {
    param($runDir)
    @(
        "--run-dir", $runDir,
        "--runs-root", $RunsRoot,
        "--csv-dir", $CsvDir,
        "--force"
    )
}

$extractResult = Invoke-PythonRunStage -StageName "extraction" -ScriptPath $extractScript -RunDirs $eligibleRuns -BuildArgs {
    param($runDir)
    @(
        "--run-dir", $runDir,
        "--runs-root", $RunsRoot,
        "--output-root", $OutputRoot,
        "--speaker-analysis-file", "speaker_analysis.json",
        "--selection-mode", "continuity_first",
        "--quality-threshold", "0.70",
        "--top-quality-fraction", "0.75",
        "--max-total-minutes", "5.0",
        "--min-segment-duration-s", "4.0",
        "--max-gap-s", "0.75",
        "--acoustic-post-filter",
        "--acoustic-min-speech-band-ratio", "0.38",
        "--acoustic-min-voiced-ratio", "0.28",
        "--acoustic-max-music-score", "0.60",
        "--acoustic-min-speaker-consistency", "0.60",
        "--acoustic-min-subject-similarity", "0.45",
        "--denoise-output",
        "--denoise-strength", "0.65",
        "--force"
    )
}

$finalSummary = [ordered]@{
    timestamp = (Get-Date).ToString("o")
    data_root = $DataRoot
    pipeline_root = $PipelineRoot
    run_list_file = $RunListFile
    eligible_runs = $eligibleRuns.Count
    missing_pyannote_output = $missingOutputRuns.Count
    results = [ordered]@{
        content_screening = $screenResult
        speaker_analysis = $speakerResult
        audit = $auditResult
        extraction = $extractResult
    }
}

$finalSummaryPath = Join-Path (Split-Path -Parent $ProgressJson) "missing_control_overnight_summary.json"
$finalSummary | ConvertTo-Json -Depth 8 | Set-Content -Path $finalSummaryPath -Encoding UTF8
Save-Progress -Phase "completed" -Metrics @{
    eligible_runs = $eligibleRuns.Count
    extraction_ok = $extractResult.ok
    extraction_failed = $extractResult.failed
}

Write-Log "Overnight chain complete. Summary: $finalSummaryPath"
