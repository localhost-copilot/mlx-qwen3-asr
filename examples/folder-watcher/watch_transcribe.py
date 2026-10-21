#!/usr/bin/env python3
"""Watch course folders and transcribe new media files to txt / srt.

Designed for the workflow: download a course with xiaoeknow-downloader, then
let this watcher turn every finished ``.ts`` / ``.mp3`` into a transcript next
to the media (or in a mirrored output tree).

The watcher is deliberately decoupled from the downloader:

* it only picks up files whose size has stopped changing for ``--stable`` sec;
* it keeps a JSON state file so restarts resume instead of redoing work;
* it reuses one loaded model + aligner for the whole run;
* it runs one transcription at a time (MLX is happiest that way).

Examples:
    python watch_transcribe.py --root "downloads/课程名" --dry-run
    python watch_transcribe.py --root "downloads/课程名"
    python watch_transcribe.py --root "/Volumes/backup/小鹅通课程" --once

Run it with the interpreter of the mlx-qwen3-asr virtualenv.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import fnmatch
import json
import os
import re
import signal
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from types import FrameType
from typing import Any, Dict, Iterable, List, Optional, TextIO

MEDIA_EXTS = {
    ".ts",
    ".mp4",
    ".mkv",
    ".mov",
    ".m4v",
    ".webm",
    ".avi",
    ".flv",
    ".wmv",
    ".mp3",
    ".m4a",
    ".aac",
    ".wav",
    ".flac",
    ".ogg",
    ".opus",
    ".wma",
}
SKIP_DIR_NAMES = {"@eaDir", "#recycle", "node_modules", "__pycache__", "lost+found"}
FORMAT_EXTS = {"txt": ".txt", "srt": ".srt", "vtt": ".vtt", "json": ".json", "tsv": ".tsv"}
DEFAULT_MODEL = "Qwen/Qwen3-ASR-1.7B"
DEFAULT_ALIGNER = "Qwen/Qwen3-ForcedAligner-0.6B"
DEFAULT_STATE = Path.home() / ".local" / "state" / "mlx-asr-watcher" / "state.json"
FMT = "%Y-%m-%d %H:%M:%S"

_stop = False


def _handle_signal(signum: int, _frame: Optional[FrameType]) -> None:
    global _stop
    _stop = True
    _log(f"收到信号 {signum}，当前文件完成后退出…")


def _log(message: str, *, log_file: Optional[TextIO] = None) -> None:
    line = f"[{time.strftime(FMT)}] {message}"
    print(line, flush=True)
    if log_file is not None:
        log_file.write(line + "\n")
        log_file.flush()


@dataclass
class Candidate:
    """A media file that may need transcribing."""

    path: Path
    root: Path
    size: int
    mtime: float

    @property
    def key(self) -> str:
        return str(self.path.resolve())


def _parse_formats(raw: str) -> List[str]:
    formats = [item.strip().lower() for item in raw.split(",") if item.strip()]
    if not formats:
        raise argparse.ArgumentTypeError("--formats 不能为空")
    unknown = [fmt for fmt in formats if fmt not in FORMAT_EXTS]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"不支持的格式: {', '.join(unknown)}；可选 {', '.join(FORMAT_EXTS)}"
        )
    return formats


def _is_excluded(path: Path, patterns: Iterable[str]) -> bool:
    text = str(path)
    return any(
        fnmatch.fnmatch(path.name, pattern) or fnmatch.fnmatch(text, pattern)
        for pattern in patterns
    )


def _iter_media(
    root: Path, min_bytes: int, excludes: Iterable[str] = ()
) -> Iterable[Candidate]:
    """Yield media files under ``root``, pruning noisy/hidden dirs."""
    excludes = list(excludes)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            d for d in dirnames if not d.startswith(".") and d not in SKIP_DIR_NAMES
        )
        for name in filenames:
            if name.startswith("."):
                continue
            path = Path(dirpath) / name
            if path.suffix.lower() not in MEDIA_EXTS:
                continue
            if excludes and _is_excluded(path, excludes):
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            if stat.st_size < min_bytes:
                continue
            yield Candidate(path=path, root=root, size=stat.st_size, mtime=stat.st_mtime)


def _load_state(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    files = data.get("files")
    if not isinstance(files, dict):
        files = {}
    return {"version": 1, "files": files}


def _save_state(path: Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def _output_targets(
    media: Path,
    root: Path,
    out_root: Optional[Path],
    formats: List[str],
) -> Dict[str, Path]:
    if out_root is None:
        base_dir = media.parent
    else:
        base_dir = out_root / media.parent.relative_to(root)
    return {fmt: base_dir / (media.stem + FORMAT_EXTS[fmt]) for fmt in formats}


def _outputs_current(media: Path, targets: Dict[str, Path]) -> bool:
    try:
        media_mtime = media.stat().st_mtime
    except OSError:
        return False
    for target in targets.values():
        try:
            stat = target.stat()
        except OSError:
            return False
        if stat.st_size <= 0 or stat.st_mtime < media_mtime - 2.0:
            return False
    return True


def _human_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _with_inferred_language(result: Any) -> Any:
    """Fill a missing language when the model output omits the language tag.

    Subtitle grouping joins tokens without spaces for CJK languages, so an
    empty language would produce "大 家 晚 上 好"-style SRT cues.
    """
    if getattr(result, "language", ""):
        return result
    text = getattr(result, "text", "") or ""
    cjk = sum(1 for ch in text if "\u3400" <= ch <= "\u9fff")
    if cjk >= 20 and cjk >= len(text) * 0.2:
        return replace(result, language="Chinese")
    return result


def _write_result(result: Any, targets: Dict[str, Path]) -> None:
    from mlx_qwen3_asr.writers import get_writer

    for fmt, target in targets.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".tmp")
        try:
            get_writer(fmt)(result, str(tmp))
            os.replace(tmp, target)
        finally:
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)


class Transcriber:
    """Owns the loaded model/aligner and turns one file into outputs."""

    def __init__(self, args: argparse.Namespace, log_file: Optional[TextIO]) -> None:
        self.args = args
        self.log_file = log_file
        self.session: Any = None
        self.aligner: Any = None
        self.needs_timestamps = any(fmt in {"srt", "vtt"} for fmt in args.formats)

    def _ensure_loaded(self) -> None:
        if self.session is not None:
            return
        import mlx.core as mx

        from mlx_qwen3_asr import Session
        from mlx_qwen3_asr.forced_aligner import ForcedAligner

        dtype = {
            "float16": mx.float16,
            "float32": mx.float32,
            "bfloat16": mx.bfloat16,
        }[self.args.dtype]

        started = time.time()
        _log(f"加载模型 {self.args.model} …", log_file=self.log_file)
        self.session = Session(model=self.args.model, dtype=dtype)
        if self.needs_timestamps:
            _log(f"加载对齐模型 {self.args.aligner} …", log_file=self.log_file)
            self.aligner = ForcedAligner(model_path=self.args.aligner, dtype=dtype)
        _log(f"模型就绪（{time.time() - started:.1f}s）", log_file=self.log_file)

    def transcribe(self, media: Path, targets: Dict[str, Path]) -> Dict[str, Any]:
        self._ensure_loaded()

        last_reported = -1
        started = time.time()

        def on_progress(event: Dict[str, Any]) -> None:
            nonlocal last_reported
            if event.get("event") != "chunk_completed":
                return
            percent = int(float(event.get("progress") or 0.0) * 100)
            if percent < last_reported + 10:
                return
            last_reported = percent
            processed = float(event.get("processed_audio_sec") or 0.0)
            total = float(event.get("audio_duration_sec") or 0.0)
            elapsed = time.time() - started
            _log(
                f"  {percent:3d}% · 音频 {_human_duration(processed)}/"
                f"{_human_duration(total)} · 已用 {_human_duration(elapsed)}",
                log_file=self.log_file,
            )

        result = self.session.transcribe(
            str(media),
            language=self.args.language or None,
            context=self.args.context,
            return_timestamps=self.needs_timestamps,
            return_chunks=True,
            forced_aligner=self.aligner,
            on_progress=on_progress,
        )
        result = _with_inferred_language(result)
        _write_result(result, targets)

        duration = 0.0
        if result.chunks:
            duration = float(result.chunks[-1].get("end") or 0.0)
        elif result.segments:
            duration = float(result.segments[-1].get("end") or 0.0)
        return {
            "language": result.language,
            "duration": duration,
            "elapsed": time.time() - started,
            "truncated": bool(result.truncated),
            "chars": len(result.text or ""),
        }


def _collect_candidates(
    args: argparse.Namespace,
    state: Dict[str, Any],
    observed: Dict[str, int],
) -> List[Candidate]:
    now = time.time()
    ready: List[Candidate] = []

    if getattr(args, "files", None):
        for path in args.files:
            try:
                stat = path.stat()
            except OSError:
                continue
            candidate = Candidate(
                path=path, root=path.parent, size=stat.st_size, mtime=stat.st_mtime
            )
            targets = _output_targets(candidate.path, candidate.root, args.out, args.formats)
            if not args.force:
                record = state["files"].get(candidate.key) or {}
                unchanged = (
                    record.get("size") == candidate.size
                    and record.get("mtime") == candidate.mtime
                )
                if unchanged and record.get("status") == "done" and _outputs_current(
                    candidate.path, targets
                ):
                    continue
                if _outputs_current(candidate.path, targets):
                    continue
            ready.append(candidate)
        ready.sort(key=lambda c: str(c.path))
        return ready

    for root in args.root:
        for candidate in _iter_media(root, args.min_bytes, args.exclude):
            if now - candidate.mtime < args.stable:
                observed[candidate.key] = candidate.size
                continue
            previous_size = observed.get(candidate.key)
            observed[candidate.key] = candidate.size
            if previous_size is None and args.require_two_scans:
                # First sighting: wait for the next poll to prove it is stable.
                continue
            if previous_size is not None and previous_size != candidate.size:
                continue

            record = state["files"].get(candidate.key) or {}
            targets = _output_targets(candidate.path, candidate.root, args.out, args.formats)
            unchanged = (
                record.get("size") == candidate.size
                and record.get("mtime") == candidate.mtime
            )
            if not args.force and unchanged:
                if record.get("status") == "ignored" and not args.include_ignored:
                    continue
                if record.get("status") == "done" and _outputs_current(candidate.path, targets):
                    continue
                attempts = int(record.get("attempts") or 0)
                if record.get("status") == "failed" and attempts >= args.max_attempts:
                    continue
            if not args.force and _outputs_current(candidate.path, targets):
                continue
            ready.append(candidate)

    if args.order == "newest":
        ready.sort(key=_recency_key, reverse=True)
    else:
        ready.sort(key=lambda c: (str(c.root), c.mtime, str(c.path)))
    return ready


def _recency_key(candidate: Candidate) -> tuple[int, int, int, float, str]:
    """Sort key for course recordings: year, month, lecture number, mtime.

    Course folders look like ``2026年10月内部直播``; using the date in the path
    is more reliable than the file mtime, which changes when files are copied.
    """
    match = re.search(r"(\d{4})年(\d{1,2})月", str(candidate.path))
    year, month = (int(match.group(1)), int(match.group(2))) if match else (0, 0)
    prefix = re.match(r"(\d+)", candidate.path.stem)
    number = int(prefix.group(1)) if prefix else 0
    return (year, month, number, candidate.mtime, str(candidate.path))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="监听课程目录，把新下载的音视频自动转成 txt / srt",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root",
        action="append",
        type=Path,
        default=[],
        help="要监听的目录，可重复传多次",
    )
    parser.add_argument(
        "--file",
        action="append",
        type=Path,
        default=[],
        help="只处理指定文件（可重复），配合 --once 供下载器等外部程序调用",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="输出根目录（镜像课程层级）；默认与媒体文件同目录",
    )
    parser.add_argument(
        "--formats",
        type=_parse_formats,
        default=_parse_formats("txt,srt"),
        help="输出格式，逗号分隔",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="ASR 模型")
    parser.add_argument("--aligner", default=DEFAULT_ALIGNER, help="词级时间戳对齐模型")
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="模型精度",
    )
    parser.add_argument("--language", default="", help="强制语言（如 Chinese）；默认自动识别")
    parser.add_argument("--context", default="", help="领域词提示，空格分隔")
    parser.add_argument("--stable", type=float, default=30.0, help="文件大小静止多少秒后开始")
    parser.add_argument(
        "--require-two-scans",
        action="store_true",
        help="必须在连续两次扫描中看到相同大小才开始（更保守）",
    )
    parser.add_argument("--poll", type=float, default=20.0, help="扫描间隔秒数")
    parser.add_argument("--min-bytes", type=int, default=10_000, help="小于该体积的文件忽略")
    parser.add_argument("--max-files", type=int, default=0, help="本轮最多处理几个文件，0 表示不限")
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="GLOB",
        help="跳过匹配的文件名或完整路径，可重复传",
    )
    parser.add_argument("--max-attempts", type=int, default=2, help="失败重试次数上限")
    parser.add_argument("--once", action="store_true", help="只扫描一轮，处理完退出")
    parser.add_argument("--dry-run", action="store_true", help="只列出待处理文件")
    parser.add_argument("--force", action="store_true", help="忽略状态与已有输出，强制重跑")
    parser.add_argument(
        "--include-ignored",
        action="store_true",
        help="重新处理 --baseline 标记为忽略的文件（已完成的仍会跳过）",
    )
    parser.add_argument(
        "--order",
        choices=["oldest", "newest"],
        default="oldest",
        help="处理顺序；newest 按路径中的年月和讲次从新到旧",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="把当前已有文件标记为忽略，只转写之后新增或变化的文件",
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=DEFAULT_STATE,
        help="状态文件，用于断点续跑",
    )
    parser.add_argument("--log", type=Path, default=None, help="额外写入日志文件")
    return parser


def _acquire_lock(state_path: Path) -> TextIO:
    lock_path = state_path.with_name(state_path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise SystemExit(f"已有另一个监听进程在运行（锁文件 {lock_path}）") from None
    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()))
    handle.flush()
    return handle


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    roots: List[Path] = []
    missing: List[Path] = []
    for root in args.root:
        resolved = root.expanduser().resolve()
        roots.append(resolved)
        if not resolved.is_dir():
            missing.append(resolved)
    for path in missing:
        if args.once:
            print(f"错误: 目录不存在: {path}", file=sys.stderr)
            return 2
        print(f"警告: 目录暂不可用（等待挂载）: {path}", file=sys.stderr)
    args.root = roots

    files: List[Path] = []
    for raw in args.file:
        path = raw.expanduser().resolve()
        if not path.is_file():
            print(f"错误: 文件不存在: {path}", file=sys.stderr)
            return 2
        files.append(path)
    args.files = files
    if not roots and not files:
        print("错误: 至少提供一个 --root 或 --file", file=sys.stderr)
        return 2

    if args.out is not None:
        args.out = args.out.expanduser()
        args.out.mkdir(parents=True, exist_ok=True)

    state_path = args.state.expanduser()
    log_file: Optional[TextIO] = None
    lock_handle: Optional[TextIO] = None
    try:
        if args.log is not None:
            args.log.parent.mkdir(parents=True, exist_ok=True)
            log_file = args.log.open("a", encoding="utf-8")
        lock_handle = _acquire_lock(state_path)

        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)

        state = _load_state(state_path)
        observed: Dict[str, int] = {}

        target_text = (
            "待处理文件: " + ", ".join(str(path) for path in args.files)
            if args.files
            else "监听目录: " + ", ".join(str(root) for root in roots)
        )
        _log(
            target_text + f" · 模型={args.model} · 格式={','.join(args.formats)}",
            log_file=log_file,
        )

        if args.baseline:
            marked = 0
            for root in args.root:
                for candidate in _iter_media(root, args.min_bytes, args.exclude):
                    record = state["files"].get(candidate.key) or {}
                    unchanged = (
                        record.get("size") == candidate.size
                        and record.get("mtime") == candidate.mtime
                    )
                    if unchanged and record.get("status") in {"ignored", "done"}:
                        continue
                    marked += 1
                    if args.dry_run:
                        continue
                    state["files"][candidate.key] = {
                        "size": candidate.size,
                        "mtime": candidate.mtime,
                        "status": "ignored",
                        "updated": time.time(),
                    }
            if args.dry_run:
                _log(f"[dry-run] 将把 {marked} 个已有文件标记为忽略。", log_file=log_file)
            else:
                _save_state(state_path, state)
                _log(
                    f"基线完成：{marked} 个已有文件被标记为忽略，之后只处理新增或变化的文件。",
                    log_file=log_file,
                )
            if args.once or args.dry_run:
                return 0

        transcriber = Transcriber(args, log_file)

        while True:
            ready = _collect_candidates(args, state, observed)

            if args.dry_run:
                listing = ready if args.max_files <= 0 else ready[: args.max_files]
                if not listing:
                    _log("没有待处理文件。", log_file=log_file)
                else:
                    _log(
                        f"共 {len(ready)} 个待处理文件"
                        + (f"（列出前 {len(listing)} 个）" if len(listing) < len(ready) else ""),
                        log_file=log_file,
                    )
                for candidate in listing:
                    targets = _output_targets(
                        candidate.path, candidate.root, args.out, args.formats
                    )
                    _log(
                        f"[待处理] {candidate.path} "
                        f"({candidate.size / 1024 / 1024:.1f} MB) → "
                        + ", ".join(str(t) for t in targets.values()),
                        log_file=log_file,
                    )
                return 0

            if args.max_files > 0:
                ready = ready[: args.max_files]

            if not ready:
                if args.once:
                    _log("没有待处理文件。", log_file=log_file)
                    return 0
                time.sleep(args.poll)
                continue

            for candidate in ready:
                if _stop:
                    _log("已停止。", log_file=log_file)
                    return 0
                record = state["files"].get(candidate.key) or {}
                attempts = int(record.get("attempts") or 0) + 1
                targets = _output_targets(
                    candidate.path, candidate.root, args.out, args.formats
                )
                _log(
                    f"开始: {candidate.path} "
                    f"({candidate.size / 1024 / 1024:.1f} MB, 第 {attempts} 次)",
                    log_file=log_file,
                )
                try:
                    stats = transcriber.transcribe(candidate.path, targets)
                except Exception as exc:  # keep the daemon alive on bad files
                    state["files"][candidate.key] = {
                        "size": candidate.size,
                        "mtime": candidate.mtime,
                        "status": "failed",
                        "attempts": attempts,
                        "error": f"{type(exc).__name__}: {exc}",
                        "updated": time.time(),
                    }
                    _save_state(state_path, state)
                    _log(f"失败: {candidate.path} · {type(exc).__name__}: {exc}", log_file=log_file)
                    continue

                state["files"][candidate.key] = {
                    "size": candidate.size,
                    "mtime": candidate.mtime,
                    "status": "done",
                    "attempts": attempts,
                    "model": args.model,
                    "language": stats["language"],
                    "duration": stats["duration"],
                    "elapsed": stats["elapsed"],
                    "chars": stats["chars"],
                    "truncated": stats["truncated"],
                    "outputs": [str(path) for path in targets.values()],
                    "updated": time.time(),
                }
                _save_state(state_path, state)
                _log(
                    f"完成: {candidate.path.stem} · {stats['language']} · "
                    f"音频 {_human_duration(stats['duration'])} · "
                    f"耗时 {_human_duration(stats['elapsed'])} · {stats['chars']} 字",
                    log_file=log_file,
                )

            if args.once:
                return 0
            time.sleep(args.poll)
    finally:
        if lock_handle is not None:
            with contextlib.suppress(OSError):
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            lock_handle.close()
        if log_file is not None:
            log_file.close()


if __name__ == "__main__":
    raise SystemExit(main())
