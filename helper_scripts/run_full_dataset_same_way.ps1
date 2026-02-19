param(
    [string]$Root = "D:\medgemma_data\dementianet_updated_files",
    [string]$PythonExe = "C:\Users\victy\anaconda_new\envs\audio\python.exe",
    [string]$CondaEnv = "audio",
    [string]$PyannoteAudioFile = "audio_16k.wav",
    [switch]$SkipPyannote,
    [switch]$SkipSpeakerAnalysis,
    [switch]$SkipAudit,
    [switch]$SkipExtraction
)

$ErrorActionPreference = "Stop"

function Invoke-Stage {
    param(
        [string]$Name,
        [scriptblock]$Action
    )

    $stageStart = Get-Date
    Write-Host "[$($stageStart.ToString("s"))] START $Name"
    & $Action | Out-Host
    $stageEnd = Get-Date
    $elapsed = [math]::Round(($stageEnd - $stageStart).TotalSeconds, 2)
    Write-Host "[$($stageEnd.ToString("s"))] DONE  $Name (${elapsed}s)"
    return [pscustomobject]@{
        stage = $Name
        started = $stageStart.ToString("o")
        ended = $stageEnd.ToString("o")
        elapsed_seconds = $elapsed
    }
}

Set-Location -LiteralPath $Root
$env:PYTHONPATH = $Root

$helperDir = Join-Path $Root "helper_scripts"
$runsRoot = Join-Path $Root "runs"
$csvDir = Join-Path $Root "csv_sources"
$processed4 = Join-Path $Root "processed_4"
$pyannoteArchive = Join-Path $Root "pyannote_results"

New-Item -ItemType Directory -Path $helperDir -Force | Out-Null
New-Item -ItemType Directory -Path $processed4 -Force | Out-Null
New-Item -ItemType Directory -Path $pyannoteArchive -Force | Out-Null

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$runLog = Join-Path $helperDir "full_dataset_same_way_$timestamp.log"
$summaryPath = Join-Path $helperDir "full_dataset_same_way_$timestamp.summary.json"
$pyannoteReportPath = Join-Path $helperDir "pyannote_full_report_$timestamp.json"
$pyannoteProgressPath = Join-Path $helperDir "pyannote_progress_$timestamp.json"

Start-Transcript -Path $runLog -Force | Out-Null

try {
    $stages = @()

    if (-not $SkipPyannote) {
        $stages += Invoke-Stage -Name "pyannote_batch" -Action {
            & $PythonExe helper_scripts\run_pyannote_api.py `
                --batch `
                --runs-root $runsRoot `
                --audio-file $PyannoteAudioFile `
                --write-standard `
                --poll-interval-s 20 `
                --timeout-s 259200 `
                --timeout-http-s 120 `
                --log-file pyannote_api.log `
                --report-file $pyannoteReportPath
        }

        $stages += Invoke-Stage -Name "pyannote_progress_report" -Action {
            & $PythonExe helper_scripts\report_pyannote_progress.py `
                --runs-root $runsRoot `
                --out-json $pyannoteProgressPath
        }

        $stages += Invoke-Stage -Name "pyannote_archive_copy" -Action {
            $filesToCopy = @(
                "pyannote_job.json",
                "pyannote_job_result.json",
                "pyannote_job_output.json",
                "diarization_api.json",
                "asr_info_api.json",
                "transcript_api.json",
                "segments_detailed_api.json",
                "words_api.json",
                "asr_info.json",
                "transcript.json",
                "segments_detailed.json",
                "words.json"
            )

            $timings = @()
            $metaDirs = Get-ChildItem -Path $runsRoot -Recurse -Directory | Where-Object { $_.Name -eq "metadata" }
            foreach ($meta in $metaDirs) {
                $runDir = $meta.Parent.FullName
                $prefix = $runsRoot.TrimEnd("\")
                $relRun = $runDir.Substring($prefix.Length).TrimStart("\")
                $dstMeta = Join-Path $pyannoteArchive (Join-Path $relRun "metadata")
                New-Item -ItemType Directory -Path $dstMeta -Force | Out-Null

                foreach ($name in $filesToCopy) {
                    $src = Join-Path $meta.FullName $name
                    if (Test-Path -LiteralPath $src) {
                        Copy-Item -LiteralPath $src -Destination (Join-Path $dstMeta $name) -Force
                    }
                }

                $statePath = Join-Path $meta.FullName "pyannote_job.json"
                if (Test-Path -LiteralPath $statePath) {
                    try {
                        $j = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
                        $submitted = if ($j.submitted_at) { [datetimeoffset]::Parse($j.submitted_at) } else { $null }
                        $completed = if ($j.completed_at) { [datetimeoffset]::Parse($j.completed_at) } else { $null }
                        $updated = if ($j.updated_at) { [datetimeoffset]::Parse($j.updated_at) } else { $null }
                        $elapsed = if ($submitted -and $completed) {
                            [math]::Round(($completed - $submitted).TotalSeconds, 2)
                        } elseif ($submitted -and $updated) {
                            [math]::Round(($updated - $submitted).TotalSeconds, 2)
                        } else {
                            $null
                        }
                        $timings += [pscustomobject]@{
                            run_dir = $runDir
                            job_id = $j.job_id
                            status = $j.status
                            submitted_at = $j.submitted_at
                            completed_at = $j.completed_at
                            updated_at = $j.updated_at
                            elapsed_seconds = $elapsed
                        }
                    } catch {}
                }
            }

            $timingsPathCsv = Join-Path $pyannoteArchive "_timings.csv"
            $timingsPathJson = Join-Path $pyannoteArchive "_timings.json"
            $timings | Sort-Object run_dir | Export-Csv -NoTypeInformation -Encoding UTF8 $timingsPathCsv
            $timings | Sort-Object run_dir | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 $timingsPathJson
        }
    }

    if (-not $SkipSpeakerAnalysis) {
        $stages += Invoke-Stage -Name "speaker_analysis_batch" -Action {
            & $PythonExe helper_scripts\run_speaker_analysis.py `
                --batch `
                --runs-root $runsRoot `
                --csv-dir $csvDir `
                --force `
                --disable-named-turn-guard
        }
    }

    if (-not $SkipAudit) {
        $stages += Invoke-Stage -Name "audit_batch" -Action {
            & $PythonExe helper_scripts\run_audit.py `
                --batch `
                --runs-root $runsRoot `
                --csv-dir $csvDir `
                --force
        }
    }

    if (-not $SkipExtraction) {
        $stages += Invoke-Stage -Name "extraction_processed_4_batch" -Action {
            & $PythonExe helper_scripts\run_audio_extraction.py `
                --batch `
                --runs-root $runsRoot `
                --output-root $processed4 `
                --force `
                --quality-threshold 0.7 `
                --top-quality-fraction 0.75 `
                --max-total-minutes 5 `
                --min-segment-duration-s 4 `
                --acoustic-post-filter `
                --acoustic-min-speech-band-ratio 0.38 `
                --acoustic-min-voiced-ratio 0.28 `
                --acoustic-max-music-score 0.60 `
                --acoustic-min-speaker-consistency 0.60 `
                --acoustic-min-subject-similarity 0.45 `
                --denoise-output `
                --denoise-strength 0.65
        }
    }

    $summary = [pscustomobject]@{
        root = $Root
        runs_root = $runsRoot
        csv_dir = $csvDir
        processed_4 = $processed4
        pyannote_archive = $pyannoteArchive
        pyannote_report = $pyannoteReportPath
        pyannote_progress = $pyannoteProgressPath
        log = $runLog
        stages = $stages
    }
    $summary | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 $summaryPath
    Write-Host "Summary written to $summaryPath"
}
finally {
    Stop-Transcript | Out-Null
}
