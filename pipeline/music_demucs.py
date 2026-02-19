from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional


def separate_vocals(
    input_wav: str,
    output_dir: str,
    model_name: str = "htdemucs",
    two_stems: str = "vocals",
) -> str:
    """
    Runs Demucs to separate vocals. Returns path to vocals WAV.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        "-m",
        "demucs.separate",
        "-n",
        model_name,
        "--two-stems",
        two_stems,
        "-o",
        str(out_dir),
        input_wav,
    ]
    subprocess.run(cmd, check=True)

    # Demucs output: <out_dir>/<model>/<basename>/vocals.wav
    stem_dir = out_dir / model_name / Path(input_wav).stem
    vocals = stem_dir / f"{two_stems}.wav"
    if vocals.exists():
        return str(vocals)

    # Fallback: search for vocals.wav in output dir.
    for candidate in out_dir.rglob(f"{two_stems}.wav"):
        return str(candidate)
    raise FileNotFoundError("Demucs vocals stem not found.")
