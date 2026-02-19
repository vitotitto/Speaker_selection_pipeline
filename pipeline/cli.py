from __future__ import annotations

import argparse

from dotenv import load_dotenv

from .config import PipelineConfig
from .run import run_pipeline


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Single-video diarization + cleanup pipeline")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--out", required=True, help="Output directory for this run")
    args = parser.parse_args()

    config = PipelineConfig()
    run_pipeline(args.video, args.out, config)


if __name__ == "__main__":
    main()
