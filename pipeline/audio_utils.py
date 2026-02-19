from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _resolve_binary(name: str, env_var: str) -> str:
    override = os.getenv(env_var)
    if override:
        if os.path.isdir(override):
            exe = f"{name}.exe" if os.name == "nt" else name
            candidate = os.path.join(override, exe)
            if os.path.exists(candidate):
                return candidate
        return override
    found = shutil.which(name) or shutil.which(f"{name}.exe")
    return found or name


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Command failed: {' '.join(cmd)}\nstderr: {result.stderr}")
        raise subprocess.CalledProcessError(
            result.returncode, cmd, output=result.stdout, stderr=result.stderr
        )


def _run_capture(cmd: list[str]) -> str:
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return result.stdout


def ffprobe_audio_info(input_path: str) -> Dict[str, Any]:
    ffprobe_bin = _resolve_binary("ffprobe", "FFPROBE_PATH")
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        input_path,
    ]
    raw = _run_capture(cmd)
    info = json.loads(raw)
    if not info.get("streams"):
        raise ValueError(f"No audio stream found in input: {input_path}")
    return info


def extract_audio_wav(
    input_path: str,
    output_wav: str,
    pcm_codec: str,
    sample_rate: Optional[int] = None,
    channels: Optional[int] = None,
) -> None:
    ffmpeg_bin = _resolve_binary("ffmpeg", "FFMPEG_PATH")
    cmd = [ffmpeg_bin, "-y", "-i", input_path, "-vn", "-map", "a:0", "-c:a", pcm_codec]
    if sample_rate is not None:
        cmd += ["-ar", str(sample_rate)]
    if channels is not None:
        cmd += ["-ac", str(channels)]
    cmd += [output_wav]
    _run(cmd)


def resample_audio_wav(
    input_wav: str,
    output_wav: str,
    sample_rate: int,
    channels: int,
) -> None:
    ffmpeg_bin = _resolve_binary("ffmpeg", "FFMPEG_PATH")
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        input_wav,
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        output_wav,
    ]
    _run(cmd)


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)
