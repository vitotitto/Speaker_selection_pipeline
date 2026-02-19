"""Submit and collect pyannoteAI diarization+transcription jobs for existing runs.

Default request settings match:
- model: precision-2 (or precision-1 when requested)
- transcription model: parakeet-tdt-0.6b-v3
- exclusive diarization: enabled
- confidence track: enabled
- turn-level confidence: enabled
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import mimetypes
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None


TERMINAL_STATUSES = {"succeeded", "failed", "canceled"}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def sanitize_component(value: str, fallback: str = "item") -> str:
    out: List[str] = []
    for ch in value:
        if ord(ch) < 128 and (ch.isalnum() or ch in "-_."):
            out.append(ch)
        else:
            out.append("_")
    cleaned = "".join(out).strip("._")
    return cleaned or fallback


@dataclass
class RunTarget:
    run_dir: Path
    metadata_dir: Path
    audio_path: Path
    source: str
    person: str
    timepoint: str
    video_stem: str

    @property
    def label(self) -> str:
        return f"{self.source}/{self.person}/{self.timepoint}/{self.video_stem}"

    @property
    def state_path(self) -> Path:
        return self.metadata_dir / "pyannote_job.json"

    @property
    def result_path(self) -> Path:
        return self.metadata_dir / "pyannote_job_result.json"

    @property
    def output_path(self) -> Path:
        return self.metadata_dir / "pyannote_job_output.json"


class PyannoteApiClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.pyannote.ai/v1",
        timeout_s: int = 120,
        max_retries: int = 5,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.session = requests.Session()

    def _request_json(
        self,
        method: str,
        url_or_path: str,
        *,
        expected: Tuple[int, ...],
        json_body: Optional[Dict[str, Any]] = None,
        auth: bool = True,
        headers: Optional[Dict[str, str]] = None,
        data: Any = None,
        parse_json: bool = True,
    ) -> Any:
        if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
            url = url_or_path
        else:
            url = f"{self.base_url}{url_or_path}"

        req_headers: Dict[str, str] = {}
        if auth:
            req_headers["Authorization"] = f"Bearer {self.api_key}"
        if headers:
            req_headers.update(headers)

        for attempt in range(self.max_retries):
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    headers=req_headers,
                    json=json_body,
                    data=data,
                    timeout=self.timeout_s,
                )
            except requests.RequestException as exc:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"{method} {url} request failed: {exc}") from exc
                time.sleep(min(2 ** attempt, 30))
                continue

            if response.status_code in expected:
                if not parse_json:
                    return None
                if not response.content:
                    return {}
                try:
                    return response.json()
                except ValueError as exc:
                    raise RuntimeError(
                        f"{method} {url} returned non-JSON response ({response.status_code})"
                    ) from exc

            retryable = response.status_code in {429, 500, 502, 503, 504}
            if retryable and attempt < self.max_retries - 1:
                retry_after = response.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    delay = int(retry_after)
                else:
                    delay = min(2 ** attempt, 60)
                time.sleep(delay)
                continue

            body_preview = response.text[:1500]
            raise RuntimeError(
                f"{method} {url} failed ({response.status_code}): {body_preview}"
            )

        raise RuntimeError(f"{method} {url} exhausted retries")

    def create_upload_url(self, media_url: str) -> str:
        payload = {"url": media_url}
        resp = self._request_json(
            "POST",
            "/media/input",
            expected=(201,),
            json_body=payload,
        )
        upload_url = resp.get("url")
        if not upload_url:
            raise RuntimeError(f"Upload URL missing in response: {resp}")
        return str(upload_url)

    def upload_file(self, upload_url: str, file_path: Path) -> None:
        content_type, _ = mimetypes.guess_type(str(file_path))
        if not content_type:
            content_type = "application/octet-stream"
        with open(file_path, "rb") as f:
            self._request_json(
                "PUT",
                upload_url,
                expected=(200, 201, 204),
                auth=False,
                headers={"Content-Type": content_type},
                data=f,
                parse_json=False,
            )

    def submit_diarization(self, payload: Dict[str, Any]) -> str:
        resp = self._request_json(
            "POST",
            "/diarize",
            expected=(200,),
            json_body=payload,
        )
        job_id = resp.get("jobId")
        if not job_id:
            raise RuntimeError(f"jobId missing in response: {resp}")
        return str(job_id)

    def get_job(self, job_id: str) -> Dict[str, Any]:
        resp = self._request_json(
            "GET",
            f"/jobs/{job_id}",
            expected=(200,),
        )
        if not isinstance(resp, dict):
            raise RuntimeError(f"Unexpected job response for {job_id}: {resp}")
        return resp


def parse_context(run_dir: Path, runs_root: Path) -> Optional[Tuple[str, str, str, str]]:
    try:
        rel = run_dir.resolve().relative_to(runs_root.resolve())
    except ValueError:
        return None
    parts = rel.parts
    if len(parts) < 4:
        return None
    source = parts[0]
    person = parts[1]
    timepoint = parts[2]
    video_stem = "/".join(parts[3:])
    return source, person, timepoint, video_stem


def pick_audio_path(run_dir: Path, preferred_audio_name: str) -> Optional[Path]:
    candidate_names = [preferred_audio_name]
    if preferred_audio_name != "audio_base.wav":
        candidate_names.append("audio_base.wav")
    if preferred_audio_name != "audio_16k.wav":
        candidate_names.append("audio_16k.wav")

    for name in candidate_names:
        path = run_dir / "audio" / name
        if path.exists() and path.is_file():
            return path
    return None


def discover_targets(
    runs_root: Path,
    preferred_audio_name: str,
    run_dir: Optional[Path] = None,
) -> List[RunTarget]:
    targets: List[RunTarget] = []
    run_dirs: List[Path] = []

    if run_dir is not None:
        run_dirs = [run_dir]
    else:
        for meta_dir in runs_root.rglob("metadata"):
            if meta_dir.is_dir():
                run_dirs.append(meta_dir.parent)

    seen: set[Path] = set()
    for candidate in run_dirs:
        if candidate in seen:
            continue
        seen.add(candidate)

        ctx = parse_context(candidate, runs_root)
        if ctx is None:
            continue
        audio_path = pick_audio_path(candidate, preferred_audio_name)
        if audio_path is None:
            continue
        source, person, timepoint, video_stem = ctx
        metadata_dir = candidate / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        targets.append(
            RunTarget(
                run_dir=candidate,
                metadata_dir=metadata_dir,
                audio_path=audio_path,
                source=source,
                person=person,
                timepoint=timepoint,
                video_stem=video_stem,
            )
        )

    targets.sort(key=lambda t: t.label.lower())
    return targets


def build_media_url(target: RunTarget) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    base_key = f"{target.source}/{target.person}/{target.timepoint}/{target.video_stem}"
    digest = hashlib.sha1(base_key.encode("utf-8")).hexdigest()[:12]

    source = sanitize_component(target.source, "source")
    person = sanitize_component(target.person, "person")
    timepoint = sanitize_component(target.timepoint, "timepoint")
    stem = sanitize_component(target.video_stem, "video")[:48]
    key = f"dementianet/{source}/{person}/{timepoint}/{stem}_{timestamp}_{digest}.wav"
    media_url = f"media://{key}"
    if len(media_url) > 255:
        media_url = f"media://dementianet/{digest}_{timestamp}.wav"
    return media_url


def build_request_payload(args: argparse.Namespace, media_url: str) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "url": media_url,
        "model": args.model,
        "exclusive": args.exclusive,
        "confidence": args.confidence,
        "turnLevelConfidence": args.turn_level_confidence,
        "transcription": args.transcription,
    }
    if args.webhook:
        payload["webhook"] = args.webhook
    if args.num_speakers is not None:
        payload["numSpeakers"] = args.num_speakers
    else:
        if args.min_speakers is not None:
            payload["minSpeakers"] = args.min_speakers
        if args.max_speakers is not None:
            payload["maxSpeakers"] = args.max_speakers
    if args.transcription:
        payload["transcriptionConfig"] = {"model": args.transcription_model}
    return payload


def select_primary_diarization(
    output: Dict[str, Any],
    prefer_exclusive: bool,
) -> List[Dict[str, Any]]:
    if prefer_exclusive:
        exclusive = output.get("exclusiveDiarization")
        if isinstance(exclusive, list) and exclusive:
            return exclusive
    diarization = output.get("diarization")
    if isinstance(diarization, list):
        return diarization
    return []


def build_transcript_segments(
    output: Dict[str, Any],
    prefer_exclusive: bool,
) -> List[Dict[str, Any]]:
    primary_diar = select_primary_diarization(output, prefer_exclusive)
    turns = output.get("turnLevelTranscription")
    diar_conf_by_key: Dict[Tuple[float, float, str], Dict[str, Any]] = {}

    for seg in primary_diar:
        conf = seg.get("confidence")
        if not isinstance(conf, dict):
            continue
        s = round(to_float(seg.get("start"), -1), 3)
        e = round(to_float(seg.get("end"), -1), 3)
        spk = str(seg.get("speaker", ""))
        diar_conf_by_key[(s, e, spk)] = conf

    if isinstance(turns, list) and turns:
        raw_segments = turns
    else:
        raw_segments = [
            {
                "start": seg.get("start"),
                "end": seg.get("end"),
                "text": "",
                "speaker": seg.get("speaker"),
            }
            for seg in primary_diar
        ]

    raw_segments.sort(key=lambda x: (to_float(x.get("start")), to_float(x.get("end"))))
    segments: List[Dict[str, Any]] = []
    for idx, seg in enumerate(raw_segments, start=1):
        start = round(to_float(seg.get("start")), 3)
        end = round(to_float(seg.get("end")), 3)
        if end <= start:
            continue
        speaker = str(seg.get("speaker", ""))
        row: Dict[str, Any] = {
            "id": idx,
            "start": start,
            "end": end,
            "text": str(seg.get("text", "")).strip(),
            "speaker": speaker,
        }
        conf_map = diar_conf_by_key.get((start, end, speaker))
        if conf_map:
            row["speaker_confidence"] = conf_map
        segments.append(row)
    return segments


def build_words(output: Dict[str, Any]) -> List[Dict[str, Any]]:
    words = output.get("wordLevelTranscription")
    if not isinstance(words, list):
        return []
    out: List[Dict[str, Any]] = []
    words_sorted = sorted(words, key=lambda x: (to_float(x.get("start")), to_float(x.get("end"))))
    for word in words_sorted:
        start = round(to_float(word.get("start")), 3)
        end = round(to_float(word.get("end")), 3)
        if end <= start:
            continue
        entry: Dict[str, Any] = {
            "word": str(word.get("text", "")).strip(),
            "start": start,
            "end": end,
        }
        if "speaker" in word:
            entry["speaker"] = str(word.get("speaker", ""))
        out.append(entry)
    return out


def build_segments_detailed(
    segments: List[Dict[str, Any]],
    words: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not segments:
        return []
    if not words:
        return [{**seg, "words": []} for seg in segments]

    detailed: List[Dict[str, Any]] = []
    w_idx = 0
    eps = 1e-6
    total_words = len(words)

    for seg in segments:
        seg_start = to_float(seg.get("start"))
        seg_end = to_float(seg.get("end"))
        seg_speaker = str(seg.get("speaker", ""))
        seg_words: List[Dict[str, Any]] = []

        while w_idx < total_words and to_float(words[w_idx].get("end")) <= seg_start + eps:
            w_idx += 1

        scan_idx = w_idx
        while scan_idx < total_words:
            word = words[scan_idx]
            w_start = to_float(word.get("start"))
            if w_start >= seg_end - eps:
                break
            midpoint = (w_start + to_float(word.get("end"))) / 2.0
            same_speaker = "speaker" not in word or not seg_speaker or word.get("speaker") == seg_speaker
            if seg_start - eps <= midpoint <= seg_end + eps and same_speaker:
                seg_words.append(word)
            scan_idx += 1

        detailed.append({**seg, "words": seg_words})

    return detailed


def infer_duration_seconds(output: Dict[str, Any], segments: List[Dict[str, Any]], words: List[Dict[str, Any]]) -> float:
    max_end = 0.0
    for seg in segments:
        max_end = max(max_end, to_float(seg.get("end")))
    for word in words:
        max_end = max(max_end, to_float(word.get("end")))
    for key in ("diarization", "exclusiveDiarization", "turnLevelTranscription", "wordLevelTranscription"):
        arr = output.get(key)
        if not isinstance(arr, list):
            continue
        for item in arr:
            max_end = max(max_end, to_float(item.get("end")))
    return round(max_end, 3)


def materialize_outputs(
    target: RunTarget,
    job: Dict[str, Any],
    request_payload: Dict[str, Any],
    *,
    write_standard: bool,
    prefer_exclusive: bool,
) -> None:
    output = job.get("output", {}) or {}
    segments = build_transcript_segments(output, prefer_exclusive=prefer_exclusive)
    words = build_words(output)
    detailed = build_segments_detailed(segments, words)
    duration_s = infer_duration_seconds(output, segments, words)

    transcript_data: Dict[str, Any] = {
        "language": "unknown",
        "language_probability": 0.0,
        "segments": segments,
    }
    segments_detailed_data: Dict[str, Any] = {"segments": detailed}
    words_data: Dict[str, Any] = {"words": words}
    asr_info_data: Dict[str, Any] = {
        "language": "unknown",
        "language_probability": 0.0,
        "duration": duration_s,
        "duration_after_vad": duration_s,
        "source": "pyannote_api",
        "transcription_model": (
            request_payload.get("transcriptionConfig", {}) or {}
        ).get("model"),
    }
    diarization_data: Dict[str, Any] = {
        "model": request_payload.get("model"),
        "exclusive_requested": bool(request_payload.get("exclusive", False)),
        "exclusive_used": bool(
            prefer_exclusive
            and isinstance(output.get("exclusiveDiarization"), list)
            and len(output.get("exclusiveDiarization", [])) > 0
        ),
        "diarization": select_primary_diarization(output, prefer_exclusive=prefer_exclusive),
        "confidence_track": output.get("confidence"),
        "warning": output.get("warning"),
        "error": output.get("error"),
    }

    write_json(target.output_path, output)
    write_json(target.metadata_dir / "diarization_api.json", diarization_data)
    write_json(target.metadata_dir / "asr_info_api.json", asr_info_data)
    write_json(target.metadata_dir / "transcript_api.json", transcript_data)
    write_json(target.metadata_dir / "segments_detailed_api.json", segments_detailed_data)
    write_json(target.metadata_dir / "words_api.json", words_data)
    write_json(target.result_path, job)

    if write_standard:
        write_json(target.metadata_dir / "asr_info.json", asr_info_data)
        write_json(target.metadata_dir / "transcript.json", transcript_data)
        write_json(target.metadata_dir / "segments_detailed.json", segments_detailed_data)
        write_json(target.metadata_dir / "words.json", words_data)


def has_completed_api_outputs(target: RunTarget, write_standard: bool) -> bool:
    if not target.output_path.exists():
        return False
    if write_standard:
        needed = [
            target.metadata_dir / "asr_info.json",
            target.metadata_dir / "transcript.json",
            target.metadata_dir / "segments_detailed.json",
            target.metadata_dir / "words.json",
        ]
    else:
        needed = [
            target.metadata_dir / "asr_info_api.json",
            target.metadata_dir / "transcript_api.json",
            target.metadata_dir / "segments_detailed_api.json",
            target.metadata_dir / "words_api.json",
        ]
    return all(path.exists() for path in needed)


def should_skip_target(
    target: RunTarget,
    args: argparse.Namespace,
) -> Tuple[bool, str]:
    state = read_json(target.state_path) or {}
    status = str(state.get("status", "")).lower()

    if args.force:
        return False, ""
    if status == "succeeded" and has_completed_api_outputs(target, write_standard=args.write_standard):
        return True, "already_succeeded"
    if status in {"failed", "canceled"} and not args.retry_failed:
        return True, f"prior_{status}"
    return False, ""


def update_state(target: RunTarget, data: Dict[str, Any]) -> None:
    write_json(target.state_path, data)


def main() -> None:
    if load_dotenv is not None:
        load_dotenv()

    parser = argparse.ArgumentParser(
        description="Run pyannoteAI diarization+transcription API for existing runs.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run-dir", type=str, help="Process a single run directory")
    mode.add_argument("--batch", action="store_true", help="Process all run directories under runs-root")
    mode.add_argument(
        "--run-list-file",
        type=str,
        help="Path to a UTF-8 text file with one run directory per line (relative or absolute).",
    )

    parser.add_argument("--runs-root", default="runs", help="Root runs directory")
    parser.add_argument("--audio-file", default="audio_base.wav", help="Preferred audio file name in run_dir/audio/")
    parser.add_argument("--api-key-env", default="PYANNOTE_API", help="Environment variable containing pyannote API key")
    parser.add_argument("--api-base-url", default="https://api.pyannote.ai/v1")
    parser.add_argument(
        "--model",
        default="precision-2",
        choices=["precision-1", "precision-2", "community-1"],
    )
    parser.add_argument(
        "--transcription-model",
        default="parakeet-tdt-0.6b-v3",
        choices=["parakeet-tdt-0.6b-v3", "faster-whisper-large-v3-turbo"],
    )
    parser.add_argument("--webhook", default=None, help="Optional webhook URL for completed jobs")
    parser.add_argument("--num-speakers", type=int, default=None)
    parser.add_argument("--min-speakers", type=int, default=None)
    parser.add_argument("--max-speakers", type=int, default=None)
    parser.add_argument("--poll-interval-s", type=int, default=30)
    parser.add_argument("--timeout-s", type=int, default=43200, help="Total polling timeout (seconds)")
    parser.add_argument("--timeout-http-s", type=int, default=120)
    parser.add_argument("--max-runs", type=int, default=None, help="Process at most N runs after filtering")
    parser.add_argument("--submit-only", action="store_true", help="Upload + submit jobs, do not poll")
    parser.add_argument("--poll-only", action="store_true", help="Poll previously submitted jobs from pyannote_job.json")
    parser.add_argument("--force", action="store_true", help="Ignore previous success state and resubmit")
    parser.add_argument("--retry-failed", action="store_true", help="Resubmit runs with prior failed/canceled state")
    parser.add_argument("--write-standard", action="store_true", help="Also write transcript/asr_info/words/segments_detailed")
    parser.add_argument(
        "--prefer-exclusive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer exclusive diarization in exports",
    )
    parser.add_argument("--transcription", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--exclusive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--confidence", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--turn-level-confidence", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-file", default="pyannote_api.log")
    parser.add_argument("--report-file", default=None, help="Optional path for summary JSON report")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(args.log_file)],
    )
    logger = logging.getLogger("pyannote_api")

    if args.num_speakers is not None and (args.min_speakers is not None or args.max_speakers is not None):
        logger.warning("num-speakers is set; min/max-speakers will be ignored.")

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key in env var: {args.api_key_env}")

    base_dir = Path(__file__).resolve().parent
    project_root = base_dir.parent
    runs_root = Path(args.runs_root)
    if not runs_root.is_absolute():
        cwd_candidate = (Path.cwd() / runs_root).resolve()
        project_candidate = (project_root / runs_root).resolve()
        helper_candidate = (base_dir / runs_root).resolve()
        if cwd_candidate.exists():
            runs_root = cwd_candidate
        elif project_candidate.exists():
            runs_root = project_candidate
        else:
            runs_root = helper_candidate

    single_run: Optional[Path] = None
    if args.run_dir:
        single_run = Path(args.run_dir)
        if not single_run.is_absolute():
            cwd_candidate = (Path.cwd() / single_run).resolve()
            project_candidate = (project_root / single_run).resolve()
            helper_candidate = (base_dir / single_run).resolve()
            if cwd_candidate.exists():
                single_run = cwd_candidate
            elif project_candidate.exists():
                single_run = project_candidate
            else:
                single_run = helper_candidate

    run_list_file: Optional[Path] = None
    if args.run_list_file:
        run_list_file = Path(args.run_list_file)
        if not run_list_file.is_absolute():
            cwd_candidate = (Path.cwd() / run_list_file).resolve()
            project_candidate = (project_root / run_list_file).resolve()
            helper_candidate = (base_dir / run_list_file).resolve()
            if cwd_candidate.exists():
                run_list_file = cwd_candidate
            elif project_candidate.exists():
                run_list_file = project_candidate
            else:
                run_list_file = helper_candidate
        if not run_list_file.exists():
            raise RuntimeError(f"Run list file not found: {run_list_file}")

    if run_list_file is not None:
        targets = []
        seen: set[Path] = set()
        lines = run_list_file.read_text(encoding="utf-8").splitlines()
        for line in lines:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            run_path = Path(raw)
            if not run_path.is_absolute():
                cwd_candidate = (Path.cwd() / run_path).resolve()
                project_candidate = (project_root / run_path).resolve()
                helper_candidate = (base_dir / run_path).resolve()
                if cwd_candidate.exists():
                    run_path = cwd_candidate
                elif project_candidate.exists():
                    run_path = project_candidate
                else:
                    run_path = helper_candidate
            run_path = run_path.resolve()
            if run_path in seen:
                continue
            seen.add(run_path)
            discovered = discover_targets(
                runs_root=runs_root,
                preferred_audio_name=args.audio_file,
                run_dir=run_path,
            )
            if discovered:
                targets.extend(discovered)
    else:
        targets = discover_targets(
            runs_root=runs_root,
            preferred_audio_name=args.audio_file,
            run_dir=single_run,
        )
    if args.max_runs is not None and args.max_runs >= 0:
        targets = targets[: args.max_runs]

    logger.info(f"Discovered {len(targets)} run(s) with audio available.")
    if not targets:
        return

    client = PyannoteApiClient(
        api_key=api_key,
        base_url=args.api_base_url,
        timeout_s=args.timeout_http_s,
    )

    summary: Dict[str, Any] = {
        "created_at": utc_now(),
        "runs_root": str(runs_root),
        "settings": {
            "model": args.model,
            "transcription_model": args.transcription_model,
            "transcription": args.transcription,
            "exclusive": args.exclusive,
            "confidence": args.confidence,
            "turn_level_confidence": args.turn_level_confidence,
            "write_standard": args.write_standard,
        },
        "counts": {
            "discovered": len(targets),
            "skipped": 0,
            "submitted": 0,
            "polling_existing": 0,
            "succeeded": 0,
            "failed": 0,
            "canceled": 0,
            "timeout": 0,
            "errors": 0,
        },
        "skips": [],
        "errors": [],
    }

    pending_jobs: Dict[str, Dict[str, Any]] = {}
    start_submit = time.time()

    for idx, target in enumerate(targets, start=1):
        skip, reason = should_skip_target(target, args)
        if skip:
            summary["counts"]["skipped"] += 1
            summary["skips"].append({"run": target.label, "reason": reason})
            logger.info(f"[{idx}/{len(targets)}] SKIP {target.label} ({reason})")
            continue

        existing_state = read_json(target.state_path) or {}
        existing_status = str(existing_state.get("status", "")).lower()
        existing_job_id = str(existing_state.get("job_id", "")).strip()

        if args.poll_only:
            if existing_job_id and existing_status not in TERMINAL_STATUSES:
                pending_jobs[existing_job_id] = {
                    "target": target,
                    "request_payload": existing_state.get("request", {}),
                }
                summary["counts"]["polling_existing"] += 1
                logger.info(f"[{idx}/{len(targets)}] POLL {target.label} ({existing_status})")
            else:
                summary["counts"]["skipped"] += 1
                summary["skips"].append({"run": target.label, "reason": "no_active_job_to_poll"})
                logger.info(f"[{idx}/{len(targets)}] SKIP {target.label} (no active job)")
            continue

        if existing_job_id and existing_status not in TERMINAL_STATUSES:
            pending_jobs[existing_job_id] = {
                "target": target,
                "request_payload": existing_state.get("request", {}),
            }
            summary["counts"]["polling_existing"] += 1
            logger.info(f"[{idx}/{len(targets)}] RESUME {target.label} ({existing_status})")
            continue

        try:
            media_url = build_media_url(target)
            request_payload = build_request_payload(args, media_url)
            upload_url = client.create_upload_url(media_url)
            client.upload_file(upload_url, target.audio_path)
            job_id = client.submit_diarization(request_payload)

            state = {
                "version": 1,
                "run": target.label,
                "run_dir": str(target.run_dir),
                "audio_path": str(target.audio_path),
                "media_url": media_url,
                "request": request_payload,
                "job_id": job_id,
                "status": "created",
                "submitted_at": utc_now(),
                "updated_at": utc_now(),
            }
            update_state(target, state)

            pending_jobs[job_id] = {
                "target": target,
                "request_payload": request_payload,
            }
            summary["counts"]["submitted"] += 1
            logger.info(f"[{idx}/{len(targets)}] SUBMITTED {target.label} (job={job_id})")
        except Exception as exc:
            summary["counts"]["errors"] += 1
            msg = f"{target.label}: {exc}"
            summary["errors"].append(msg)
            logger.error(f"[{idx}/{len(targets)}] ERROR submitting {msg}")

    logger.info(
        f"Submission phase done in {round(time.time() - start_submit, 1)}s. Pending jobs: {len(pending_jobs)}"
    )

    if args.submit_only:
        logger.info("Submit-only mode enabled; skipping polling.")
    else:
        start_poll = time.time()
        while pending_jobs:
            if args.timeout_s > 0 and (time.time() - start_poll) > args.timeout_s:
                logger.error("Polling timeout reached.")
                for job_id, item in list(pending_jobs.items()):
                    target = item["target"]
                    state = read_json(target.state_path) or {}
                    state["updated_at"] = utc_now()
                    state["status"] = str(state.get("status", "running"))
                    state["poll_timeout"] = True
                    update_state(target, state)
                    summary["counts"]["timeout"] += 1
                    del pending_jobs[job_id]
                break

            for job_id, item in list(pending_jobs.items()):
                target: RunTarget = item["target"]
                request_payload = item.get("request_payload", {})
                try:
                    job = client.get_job(job_id)
                    status = str(job.get("status", "")).lower()
                except Exception as exc:
                    summary["counts"]["errors"] += 1
                    msg = f"{target.label} (job={job_id}): {exc}"
                    summary["errors"].append(msg)
                    logger.error(f"Polling error: {msg}")
                    continue

                state = read_json(target.state_path) or {}
                prev_status = str(state.get("status", "")).lower()
                state["status"] = status
                state["updated_at"] = utc_now()
                update_state(target, state)

                if status != prev_status:
                    logger.info(f"{target.label} -> {status} (job={job_id})")

                if status not in TERMINAL_STATUSES:
                    continue

                if status == "succeeded":
                    try:
                        materialize_outputs(
                            target=target,
                            job=job,
                            request_payload=request_payload,
                            write_standard=args.write_standard,
                            prefer_exclusive=args.prefer_exclusive,
                        )
                        state = read_json(target.state_path) or {}
                        state["status"] = "succeeded"
                        state["completed_at"] = utc_now()
                        state["updated_at"] = utc_now()
                        update_state(target, state)
                        summary["counts"]["succeeded"] += 1
                    except Exception as exc:
                        summary["counts"]["errors"] += 1
                        msg = f"{target.label} materialize failed: {exc}"
                        summary["errors"].append(msg)
                        logger.error(msg)
                elif status == "failed":
                    summary["counts"]["failed"] += 1
                elif status == "canceled":
                    summary["counts"]["canceled"] += 1

                del pending_jobs[job_id]

            if pending_jobs:
                time.sleep(max(1, args.poll_interval_s))

    summary["finished_at"] = utc_now()
    summary["pending_after_finish"] = len(pending_jobs)

    if args.report_file:
        report_path = Path(args.report_file)
        if not report_path.is_absolute():
            report_path = (Path.cwd() / report_path).resolve()
    else:
        report_path = runs_root / f"_pyannote_api_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    write_json(report_path, summary)

    logger.info(
        "Done. "
        f"discovered={summary['counts']['discovered']}, "
        f"submitted={summary['counts']['submitted']}, "
        f"polling_existing={summary['counts']['polling_existing']}, "
        f"succeeded={summary['counts']['succeeded']}, "
        f"failed={summary['counts']['failed']}, "
        f"canceled={summary['counts']['canceled']}, "
        f"timeout={summary['counts']['timeout']}, "
        f"errors={summary['counts']['errors']}, "
        f"skipped={summary['counts']['skipped']}"
    )
    logger.info(f"Summary report: {report_path}")


if __name__ == "__main__":
    main()
