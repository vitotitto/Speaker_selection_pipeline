"""Batch process media files using a deterministic sharding strategy for distributed execution.

Run this script on multiple machines (e.g., Local + Colab) with different --shard_id
values to process the dataset in parallel without overlap.

Usage:
    # Machine 1 (Local):
    python run_distributed_pipeline.py --shard_id 0 --num_shards 2

    # Machine 2 (Colab):
    python run_distributed_pipeline.py --shard_id 1 --num_shards 2 --base_dir /content/drive/MyData
"""
import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("distributed_pipeline.log"),
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
    """Recursively find all supported media files."""
    videos = []
    if not root_dir.exists():
        logger.error(f"Input dir does not exist: {root_dir}")
        return videos
    for f in root_dir.rglob("*"):
        if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS:
            videos.append(f)
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
    parser = argparse.ArgumentParser(description="Distributed Video Processing Pipeline")
    parser.add_argument(
        "--shard_id", type=int, required=True, help="ID of this worker (0-indexed)"
    )
    parser.add_argument(
        "--num_shards", type=int, required=True, help="Total number of workers"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default=r"D:\medgemma_data\dementianet_updated_files",
        help="Root directory containing Dementia_raw_data and No_Dementia_raw_data",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print files that would be processed without running them",
    )
    args = parser.parse_args()

    load_dotenv()

    base_dir = Path(args.base_dir)
    runs_dir = base_dir / "runs"

    input_dirs = [
        base_dir / "Dementia_raw_data",
        base_dir / "No_Dementia_raw_data",
    ]

    # 1. Discover ALL videos first
    all_videos = []
    logger.info("Scanning for media files...")
    for input_dir in input_dirs:
        if not input_dir.exists():
            logger.warning(f"Directory not found: {input_dir}")
            continue
            
        for v in get_video_files(input_dir):
            # Store relative path for deterministic sorting across different machines
            rel_path = v.relative_to(base_dir)
            all_videos.append((str(rel_path), v, input_dir))

    # 2. Deterministic Sort (Critical for sharding)
    # Sort by the relative path string to ensure order is identical on Windows & Linux
    all_videos.sort(key=lambda x: x[0].replace("\\", "/").lower())

    total_videos = len(all_videos)
    logger.info(f"Found {total_videos} total media files.")

    # 3. Apply Sharding
    my_shard_videos = []
    for i, (rel_path_str, video_path, root_data_dir) in enumerate(all_videos):
        if i % args.num_shards == args.shard_id:
            my_shard_videos.append((video_path, root_data_dir))

    logger.info(
        f"Shard {args.shard_id}/{args.num_shards}: Assigned {len(my_shard_videos)} videos."
    )

    # 4. Filter already processed
    to_process = []
    skipped = 0
    
    for video_path, root_data_dir in my_shard_videos:
        # Reconstruct output paths
        relative_path = video_path.relative_to(root_data_dir)
        source = root_data_dir.name
        output_dir = runs_dir / source / relative_path.parent / video_path.stem
        meta_dir = output_dir / "metadata"

        if is_v2_complete(meta_dir):
            skipped += 1
            continue
        to_process.append((video_path, output_dir))

    logger.info(f"Skipping {skipped} already-completed runs.")
    logger.info(f"Ready to process {len(to_process)} videos.")

    if not to_process:
        logger.info("Nothing to do for this shard.")
        if args.dry_run:
             print("Dry run complete. No files to process.")
        return

    if args.dry_run:
        print(f"\n[DRY RUN] Shard {args.shard_id} would process:")
        for vp, _ in to_process:
            print(f"  - {vp.name}")
        return

    # 5. Load Model & Run
    config = PipelineConfig()
    logger.info(f"Loading ASR model: {config.asr.model_name}...")
    t0 = time.perf_counter()
    asr_model = load_model(
        model_name=config.asr.model_name,
        device=config.asr.device,
        compute_type=config.asr.compute_type,
    )
    logger.info(f"Model loaded in {time.perf_counter() - t0:.1f}s")

    successes = 0
    failures = 0
    batch_start = time.perf_counter()

    for i, (video_path, output_dir) in enumerate(to_process):
        label = f"[Shard {args.shard_id} | {i + 1}/{len(to_process)}]"
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
        f"Shard {args.shard_id} Complete: {successes} success, {failures} failed. "
        f"Total time: {elapsed / 60:.1f} min"
    )

if __name__ == "__main__":
    main()
