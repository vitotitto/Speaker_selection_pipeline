from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AudioConfig:
    # Base audio extraction from video.
    pcm_codec: str = "pcm_s24le"
    # Model inputs (ASR/diarization) are usually 16 kHz mono.
    model_sample_rate: int = 16000
    model_channels: int = 1


@dataclass
class ASRConfig:
    backend: str = "faster-whisper"
    model_name: str = "large-v3"
    language: str = "en"
    device: str = "cuda"
    compute_type: str = "float16"
    beam_size: int = 5
    batch_size: int = 16
    vad_filter: bool = True
    skip: bool = False  # Skip ASR, only extract audio


@dataclass
class DiarizationConfig:
    enabled: bool = False
    hf_token_env: str = "HF_TOKEN"
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None


@dataclass
class MusicSeparationConfig:
    enabled: bool = False
    demucs_model: str = "htdemucs"
    two_stems: str = "vocals"


@dataclass
class DenoiseConfig:
    enabled: bool = False


@dataclass
class OverlapSeparationConfig:
    enabled: bool = False


@dataclass
class LLMConfig:
    enabled: bool = False
    provider: str = "gemini"
    model_name: str = "gemini-2.0-flash"
    api_key_env: str = "GEMINI_API_KEY"


@dataclass
class SegmentationConfig:
    max_gap_s: float = 1.0


@dataclass
class PipelineConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    diarization: DiarizationConfig = field(default_factory=DiarizationConfig)
    music: MusicSeparationConfig = field(default_factory=MusicSeparationConfig)
    denoise: DenoiseConfig = field(default_factory=DenoiseConfig)
    overlap: OverlapSeparationConfig = field(default_factory=OverlapSeparationConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    strict: bool = True
