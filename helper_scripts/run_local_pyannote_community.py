"""Run local pyannote community diarization and save outputs + timings."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import soundfile as sf

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None


def _resolve_audio_path(run_dir: Optional[str], audio_path: Optional[str], audio_file: str) -> Path:
    if audio_path:
        p = Path(audio_path)
        if not p.exists():
            raise FileNotFoundError(f"Audio not found: {p}")
        return p

    if not run_dir:
        raise ValueError("Provide either --audio-path or --run-dir.")

    p = Path(run_dir) / audio_file
    if not p.exists():
        raise FileNotFoundError(f"Audio not found: {p}")
    return p


def _default_output_dir(audio: Path, run_dir: Optional[str]) -> Path:
    root = Path("pyannote_results_local")
    if run_dir:
        rel = Path(run_dir)
        return root / rel / "metadata"
    return root / "single_audio" / audio.stem


def _as_annotation(diarization_output: Any) -> Any:
    if hasattr(diarization_output, "itertracks"):
        return diarization_output
    if hasattr(diarization_output, "speaker_diarization"):
        return diarization_output.speaker_diarization
    if isinstance(diarization_output, dict) and "speaker_diarization" in diarization_output:
        return diarization_output["speaker_diarization"]
    raise TypeError(f"Unsupported diarization output type: {type(diarization_output)!r}")


def _iter_turns(annotation: Any) -> Iterator[Tuple[Any, str]]:
    if hasattr(annotation, "itertracks"):
        for segment, _, speaker in annotation.itertracks(yield_label=True):
            yield segment, str(speaker)
        return

    # pyannote v4 examples expose an iterable of (turn, speaker) pairs.
    if hasattr(annotation, "__iter__"):
        for item in annotation:
            if not isinstance(item, tuple):
                continue
            if len(item) == 2:
                turn, speaker = item
                yield turn, str(speaker)
            elif len(item) == 3:
                turn, _, speaker = item
                yield turn, str(speaker)
        return

    raise TypeError(f"Unsupported diarization annotation type: {type(annotation)!r}")


def _collect_segments(diarization_output: Any) -> List[Dict[str, Any]]:
    annotation = _as_annotation(diarization_output)
    segments: List[Dict[str, Any]] = []
    for segment, speaker in _iter_turns(annotation):
        segments.append(
            {
                "start": float(segment.start),
                "end": float(segment.end),
                "duration": float(segment.end - segment.start),
                "speaker": str(speaker),
            }
        )
    segments.sort(key=lambda x: (x["start"], x["end"], x["speaker"]))
    return segments


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pyannote/speaker-diarization-community-1 locally.")
    parser.add_argument("--run-dir", default=None, help="Run directory under runs/... (optional).")
    parser.add_argument("--audio-path", default=None, help="Direct path to WAV file (optional).")
    parser.add_argument(
        "--audio-file",
        default="audio/audio_16k.wav",
        help="Relative audio path under run directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output metadata directory. Defaults to pyannote_results_local/.../metadata",
    )
    parser.add_argument(
        "--model",
        default="pyannote/speaker-diarization-community-1",
        help="HuggingFace model id.",
    )
    parser.add_argument("--hf-token-env", default="HF_TOKEN", help="Env var holding HF token.")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--min-speakers", type=int, default=None)
    parser.add_argument("--max-speakers", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if load_dotenv is not None:
        load_dotenv()

    token = os.getenv(args.hf_token_env) or os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        raise RuntimeError(
            f"No HuggingFace token found. Set {args.hf_token_env} or HUGGINGFACE_TOKEN (for gated models)."
        )

    audio_path = _resolve_audio_path(args.run_dir, args.audio_path, args.audio_file)
    out_dir = Path(args.output_dir) if args.output_dir else _default_output_dir(audio_path, args.run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    diar_json = out_dir / "diarization_local.json"
    timing_json = out_dir / "timing_local.json"
    rttm_path = out_dir / "diarization_local.rttm"

    if not args.overwrite and diar_json.exists() and timing_json.exists():
        print(json.dumps({"status": "skipped_existing", "output_dir": str(out_dir)}, indent=2))
        return

    try:
        import torch
        import torchaudio

        # Compatibility for speechbrain on newer torchaudio.
        if not hasattr(torchaudio, "list_audio_backends"):
            torchaudio.list_audio_backends = lambda: ["soundfile"]  # type: ignore[attr-defined]
        if not hasattr(torchaudio, "set_audio_backend"):
            torchaudio.set_audio_backend = lambda _: None  # type: ignore[attr-defined]
        if not hasattr(torchaudio, "get_audio_backend"):
            torchaudio.get_audio_backend = lambda: "soundfile"  # type: ignore[attr-defined]

        from pyannote.audio import Pipeline
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("pyannote.audio and torch are required in the active environment.") from exc

    info = sf.info(str(audio_path))
    audio_duration_s = float(info.frames) / float(info.samplerate)

    t0 = time.time()
    pipeline = Pipeline.from_pretrained(args.model, token=token)
    t1 = time.time()

    if args.device == "cuda":
        device = "cuda"
    elif args.device == "cpu":
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        pipeline.to(torch.device("cuda"))

    diarize_kwargs: Dict[str, Any] = {}
    if args.min_speakers is not None:
        diarize_kwargs["min_speakers"] = args.min_speakers
    if args.max_speakers is not None:
        diarize_kwargs["max_speakers"] = args.max_speakers

    t2 = time.time()
    diarization = pipeline(str(audio_path), **diarize_kwargs)
    t3 = time.time()

    segments = _collect_segments(diarization)
    annotation = _as_annotation(diarization)
    with rttm_path.open("w", encoding="utf-8") as f:
        if hasattr(annotation, "write_rttm"):
            annotation.write_rttm(f)

    load_seconds = t1 - t0
    infer_seconds = t3 - t2
    wall_seconds = t3 - t0
    rtf = infer_seconds / audio_duration_s if audio_duration_s > 0 else 0.0
    speed_x = (1.0 / rtf) if rtf > 0 else 0.0

    diar_payload: Dict[str, Any] = {
        "model": args.model,
        "audio_path": str(audio_path),
        "audio_duration_s": audio_duration_s,
        "device": device,
        "segments": segments,
    }
    timing_payload: Dict[str, Any] = {
        "model": args.model,
        "audio_path": str(audio_path),
        "output_dir": str(out_dir),
        "audio_duration_s": audio_duration_s,
        "load_seconds": load_seconds,
        "inference_seconds": infer_seconds,
        "wall_seconds": wall_seconds,
        "real_time_factor": rtf,
        "speed_x_realtime": speed_x,
        "num_segments": len(segments),
        "device": device,
    }

    diar_json.write_text(json.dumps(diar_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    timing_json.write_text(json.dumps(timing_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(timing_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
