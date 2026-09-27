"""Stream RGB frames to an MP4 and publish it after successful encoding."""

import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def video_writer(path, fps, width=832, height=480, lossless=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".mp4", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
    command = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "rawvideo",
               "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps),
               "-i", "-", "-an", "-c:v", "libx264", "-preset", "ultrafast",
               "-crf", "0" if lossless else "18", "-pix_fmt", "yuv420p", str(temporary)]
    process = None
    try:
        process = subprocess.Popen(command, stdin=subprocess.PIPE)
        yield process.stdin
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError(f"Video encoding failed: {path}")
        temporary.replace(path)
    finally:
        if process is not None:
            if process.poll() is None:
                process.terminate()
            if not process.stdin.closed:
                try:
                    process.stdin.close()
                except BrokenPipeError:
                    pass
            process.wait()
        temporary.unlink(missing_ok=True)
