"""Make one small, browser-playable reference video per sentence for the Sign library.

Many ISL-CSLTR videos are MPEG-4 Part 2 at 1080p, which browsers cannot decode (the player shows a timer over a
black picture). This converts one clip per sentence to H.264, 480p, no audio, fast-start, into
data/reference_videos/<sentence id>.mp4 (a few hundred KB each). Needs ffmpeg on PATH.

    python scripts/make_reference_videos.py
"""

import shutil
import subprocess
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app.ml.dataset import load_sentences  # noqa: E402
from app.routers.data import _reference_video  # noqa: E402

OUT = BACKEND.parent / "data" / "reference_videos"


def main() -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        sys.exit("ffmpeg not found on PATH")
    OUT.mkdir(parents=True, exist_ok=True)
    done = 0
    for s in load_sentences():
        src = _reference_video(s["text"])
        if src is None:
            continue
        dst = OUT / f"{s['id']}.mp4"
        if not dst.exists():
            cmd = [ffmpeg, "-y", "-v", "error", "-i", str(src), "-an", "-vf", "scale=-2:480,fps=25",
                   "-c:v", "libx264", "-profile:v", "main", "-pix_fmt", "yuv420p", "-crf", "28", "-preset", "medium",
                   "-movflags", "+faststart", str(dst)]
            subprocess.run(cmd, check=True)
        done += 1
    print(f"{done} reference videos in {OUT}")


if __name__ == "__main__":
    main()
