# Folder watcher — download → txt + srt automatically

Watch one or more course folders and transcribe every new audio/video file with
`mlx-qwen3-asr`. Output lands next to the media (default) or in a mirrored
folder tree. Built for the `xiaoeknow-downloader` workflow but works with any
folder of media.

```
课程名/
  子课程/
    章节/
      001_标题.ts          ← downloaded by the downloader
      001_标题.txt         ← written by this watcher
      001_标题.srt         ← written by this watcher
```

## Quick start

```bash
# from the repo root, using its .venv
./examples/folder-watcher/start-watcher.sh \
  --root "/Volumes/backup/小鹅通课程" --dry-run

# real run (keeps watching; Ctrl+C stops after the current file)
./examples/folder-watcher/start-watcher.sh \
  --root "/Volumes/backup/小鹅通课程"

# one pass over the existing library, then exit
./examples/folder-watcher/start-watcher.sh \
  --root "/Volumes/backup/小鹅通课程" --once

# process a single file (used by the downloader's auto-transcribe hook)
./examples/folder-watcher/start-watcher.sh \
  --file "/path/to/001_标题.ts" --formats txt,srt --once
```

Or call Python directly:

```bash
.venv/bin/python examples/folder-watcher/watch_transcribe.py \
  --root "/path/to/downloads" --formats txt,srt
```

## How it decides a file is ready

1. Extension is in the media list (`.ts`, `.mp4`, `.mp3`, `.m4a`, `.wav`, …).
2. Size is at least `--min-bytes` (default 10 KB).
3. The file has not changed for `--stable` seconds (default 30). This avoids
   transcribing a half-written HLS concat.
4. It is not already recorded as done with the same size/mtime, and its
   outputs either do not exist or are older than the media file.

`--require-two-scans` makes step 3 stricter: the watcher must see the same size
on two consecutive polls.

## Resume and retries

State lives in `~/.local/state/mlx-asr-watcher/state.json`:

- a finished file is skipped on the next run (even after a reboot);
- a failed file is retried at most `--max-attempts` times (default 2);
- `--force` ignores state and existing outputs and re-transcribes;
- `--include-ignored` re-processes files that a baseline marked as ignored,
  while still skipping files already finished;
- a `state.json.lock` file prevents two watchers from running at once.

## Only transcribe future downloads (recommended)

If the folder already contains a large library, mark it as a baseline once;
from then on only new or changed files are transcribed:

```bash
# one-time, after pointing --root at the download folder
./start-watcher.sh --root "/Volumes/backup/小鹅通课程" --baseline --once

# then run the watcher normally (or install the LaunchAgent)
./start-watcher.sh --root "/Volumes/backup/小鹅通课程"
```

Do not pass `--baseline` to the long-running process: files downloaded while
the watcher was stopped would be marked as ignored. A baseline is idempotent,
so re-running it later is harmless.

## Backfill an existing library

Files that a baseline marked as ignored are skipped; use
`--include-ignored` to transcribe them anyway. Completed files stay skipped,
so an interrupted backfill can be restarted with the same command.

```bash
# all live courses, newest month first, one pass
./start-watcher.sh \
  --root "/Volumes/backup/小鹅通课程/…/内部直播课" \
  --once --include-ignored --order newest

# process a batch at a time
./start-watcher.sh --root "/Volumes/backup/小鹅通课程" --once --max-files 5
```

`--order newest` sorts by the ``YYYY年M月`` folder name, then by the leading
lecture number, then by mtime. The default is oldest first.

Rough speed on an M-series Mac with 1.7B: about 8× real time for txt, less
with SRT because of alignment. A 1.5 h lecture takes ~12–15 minutes.

## Output formats

`--formats` accepts any of `txt,srt,vtt,json,tsv` (default `txt,srt`).

- `txt` — plain transcript, one file;
- `srt` / `vtt` — subtitles grouped from word-level timestamps; the first time
  you request these the watcher downloads the forced aligner
  (`Qwen/Qwen3-ForcedAligner-0.6B`, ~1.2 GB);
- `json` / `tsv` — structured output for downstream analysis.

## Performance notes

- One transcription at a time; the model and aligner stay loaded between files
  (roughly 1.2 GB for 0.6B, 3.4 GB for 1.7B, plus 1.2 GB for the aligner).
- 1.7B runs at roughly 8× real time on an M-series Mac: a 1.5 h lecture takes
  about 12 minutes, plus alignment for SRT.
- Want speed over accuracy? `--model Qwen/Qwen3-ASR-0.6B` or a quantized
  checkpoint such as `moona3k/mlx-qwen3-asr-1.7b-8bit`.
- `--language Chinese` can help on very short or noisy clips; long lectures
  auto-detect reliably.

## Start automatically at login (macOS)

```bash
./examples/folder-watcher/install-launchd.sh "/Volumes/backup/小鹅通课程"
```

This writes `~/Library/LaunchAgents/com.mlx-asr.watcher.plist` and loads it.
Logs go to `~/Library/Logs/mlx-asr-watcher.{out,err}.log`. Remove with:

```bash
launchctl bootout gui/$(id -u)/com.mlx-asr.watcher
rm ~/Library/LaunchAgents/com.mlx-asr.watcher.plist
```

The watcher idles when the target volume is not mounted; files that appear
after mounting are picked up on the next poll.

## Notes

- The watcher does not touch the downloader's process or port, so both can run
  at the same time. (The downloader uses port 8765; if you also run
  `mlx-qwen3-asr serve`, give it another port, e.g. `--port 8766`.)
- Only WAV works without ffmpeg; other formats need `brew install ffmpeg`.
- Files under hidden folders, `@eaDir`, `#recycle`, and `node_modules` are
  ignored.
