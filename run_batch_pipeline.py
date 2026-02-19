"""Batch process all media files through the v2 pipeline (faster-whisper large-v3).

Loads the ASR model once and processes videos sequentially.
Skips videos that already have a completed run.json.
"""
import json
import logging
import sys
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("batch_pipeline_v2.log"),
    ],
)
logger = logging.getLogger(__name__)

try:
    from pipeline.asr_faster_whisper import load_model
    from pipeline.config import PipelineConfig
    from pipeline.run import run_pipeline
except ImportError:
    logger.error("Could not import pipeline modules. Run from the project root.")
    sys.exit(1)

ALLOWED_EXTENSIONS = {
    ".mkv",
    ".mp4",
    ".avi",
    ".mov",
    ".webm",
    ".ts",
    ".m4v",
    ".m4a",
}


def get_video_files(root_dir: Path):
    videos = []
    if not root_dir.exists():
        logger.error(f"Input dir does not exist: {root_dir}")
        return videos
    for f in root_dir.rglob("*"):
        if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS:
            videos.append(f)
    videos.sort(key=lambda p: str(p).lower())
    return videos


def is_v2_complete(meta_dir: Path) -> bool:
    """Check if this run already has a completed v2 output."""
    run_json = meta_dir / "run.json"
    if not run_json.exists():
        return False
    try:
        data = json.loads(run_json.read_text(encoding="utf-8"))
        if data.get("pipeline_version", 0) < 2:
            return False

        # New format from pipeline/run.py writes explicit status.
        status = data.get("status")
        if status == "success":
            return True
        if status == "failed":
            return False

        # Legacy v2 runs (before status field): require full metadata set.
        required = (
            meta_dir / "asr_info.json",
            meta_dir / "transcript.json",
            meta_dir / "segments_detailed.json",
            meta_dir / "words.json",
        )
        return all(p.exists() for p in required)
    except (json.JSONDecodeError, OSError):
        return False


def main():
    load_dotenv()

    base_dir = Path(r"D:\medgemma_data\dementianet_updated_files")
    runs_dir = base_dir / "runs"

    input_dirs = [
        base_dir / "Dementia_raw_data",
        base_dir / "No_Dementia_raw_data",
    ]

    # Discover all media files
    all_videos = []
    for input_dir in input_dirs:
        logger.info(f"Scanning {input_dir}...")
        for v in get_video_files(input_dir):
            all_videos.append((v, input_dir))

    logger.info(f"Found {len(all_videos)} total media files.")

    # Count how many need processing
    to_process = []
    skipped = 0
    for video_path, root_data_dir in all_videos:
        relative_path = video_path.relative_to(root_data_dir)
        source = root_data_dir.name
        output_dir = runs_dir / source / relative_path.parent / video_path.stem
        meta_dir = output_dir / "metadata"

        if is_v2_complete(meta_dir):
            skipped += 1
            continue
        to_process.append((video_path, output_dir, source))

    logger.info(f"Skipping {skipped} already-completed v2 runs.")
    logger.info(f"Processing {len(to_process)} media files.")

    if not to_process:
        logger.info("Nothing to do.")
        return

    # Load model ONCE
    config = PipelineConfig()
    logger.info(f"Loading ASR model: {config.asr.model_name} on {config.asr.device} ({config.asr.compute_type})...")
    t0 = time.perf_counter()
    asr_model = load_model(
        model_name=config.asr.model_name,
        device=config.asr.device,
        compute_type=config.asr.compute_type,
    )
    logger.info(f"Model loaded in {time.perf_counter() - t0:.1f}s")

    # Process sequentially
    successes = 0
    failures = 0
    batch_start = time.perf_counter()

    for i, (video_path, output_dir, source) in enumerate(to_process):
        label = f"[{i + 1}/{len(to_process)}]"
        logger.info(f"{label} START: {video_path.name}")

        try:
            run_pipeline(str(video_path), str(output_dir), config, asr_model=asr_model)
            successes += 1
            logger.info(f"{label} SUCCESS: {video_path.name}")
        except Exception as e:
            failures += 1
            logger.error(f"{label} FAILED: {video_path.name}: {e}")
            logger.debug(traceback.format_exc())

    elapsed = time.perf_counter() - batch_start
    logger.info(
        f"Batch complete: {successes} success, {failures} failed, "
        f"{skipped} skipped. Total time: {elapsed / 60:.1f} min"
    )


if __name__ == "__main__":
    main()
