from __future__ import annotations

from typing import Tuple


def _load_audio(path: str):
    try:
        import soundfile as sf
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "soundfile is required. Install with: pip install soundfile"
        ) from exc

    info = sf.info(path)
    subtype = info.subtype
    dtype = "float32"
    if subtype == "PCM_16":
        dtype = "int16"
    elif subtype in ("PCM_24", "PCM_32"):
        dtype = "int32"
    audio, sr = sf.read(path, dtype=dtype, always_2d=True)
    return audio, sr, subtype


def _write_audio(path: str, audio, sr: int, subtype: str) -> None:
    import soundfile as sf

    sf.write(path, audio, sr, subtype=subtype)


def denoise_with_deepfilternet(input_wav: str, output_wav: str) -> str:
    """
    Apply DeepFilterNet denoise while preserving sample rate and subtype.
    """
    try:
        from df.enhance import enhance, init_df
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "DeepFilterNet is required. Install with: pip install deepfilternet"
        ) from exc

    audio, sr, subtype = _load_audio(input_wav)
    model, df_state, _ = init_df()

    # Process each channel independently to preserve channel count.
    enhanced = audio.copy()
    for ch in range(audio.shape[1]):
        enhanced[:, ch] = enhance(model, df_state, audio[:, ch])

    _write_audio(output_wav, enhanced, sr, subtype)
    return output_wav
