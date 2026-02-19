import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

from speaker_analysis.discovery import discover_runs, discover_single_run
from speaker_audit.config import SpeakerAuditConfig, LLMProviderConfig
from speaker_audit.runner import run_audit

def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Omniscient Speaker Audit Tool")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run-dir", type=str, help="Path to a single run directory")
    mode.add_argument("--batch", action="store_true", help="Process all discovered runs")
    
    parser.add_argument("--csv-dir", default="csv_sources", help="Directory containing patient metadata CSVs")
    parser.add_argument("--runs-root", default="runs", help="Root directory for runs")
    parser.add_argument("--force", action="store_true", help="Re-run audit even if speaker_audit.json exists")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("audit")
    
    helper_dir = Path(__file__).resolve().parent
    project_root = helper_dir.parent
    csv_dir = Path(args.csv_dir)
    runs_root = Path(args.runs_root)
    if not csv_dir.is_absolute():
        cwd_candidate = (Path.cwd() / csv_dir).resolve()
        project_candidate = (project_root / csv_dir).resolve()
        helper_candidate = (helper_dir / csv_dir).resolve()
        if cwd_candidate.exists():
            csv_dir = cwd_candidate
        elif project_candidate.exists():
            csv_dir = project_candidate
        else:
            csv_dir = helper_candidate
    if not runs_root.is_absolute():
        cwd_candidate = (Path.cwd() / runs_root).resolve()
        project_candidate = (project_root / runs_root).resolve()
        helper_candidate = (helper_dir / runs_root).resolve()
        if cwd_candidate.exists():
            runs_root = cwd_candidate
        elif project_candidate.exists():
            runs_root = project_candidate
        else:
            runs_root = helper_candidate
    
    config = SpeakerAuditConfig(skip_existing=not args.force)
    
    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.is_absolute():
            cwd_candidate = (Path.cwd() / run_dir).resolve()
            project_candidate = (project_root / run_dir).resolve()
            helper_candidate = (helper_dir / run_dir).resolve()
            if cwd_candidate.exists():
                run_dir = cwd_candidate
            elif project_candidate.exists():
                run_dir = project_candidate
            else:
                run_dir = helper_candidate
            
        run_info = discover_single_run(run_dir, runs_root)
        if run_info:
            logger.info(f"Auditing {run_info.video_stem}...")
            result = run_audit(run_info, config, csv_dir)
            logger.info(f"Result: {result.get('status')}")
        else:
            logger.error(f"Could not find run info for {run_dir}")
            
    elif args.batch:
        runs = discover_runs(runs_root)
        logger.info(f"Found {len(runs)} runs to audit.")
        for run_info in runs:
            try:
                result = run_audit(run_info, config, csv_dir)
                logger.info(f"Audited {run_info.video_stem}: {result.get('status')}")
            except Exception as e:
                logger.error(f"Failed to audit {run_info.video_stem}: {e}")

if __name__ == "__main__":
    main()
