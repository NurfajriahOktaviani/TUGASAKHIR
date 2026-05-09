"""Railway entrypoint for the YOLOv12 FastAPI service."""

from pathlib import Path
import sys

API_DIR = Path(__file__).resolve().parent / "yolov12"
sys.path.insert(0, str(API_DIR))

from api_v2 import APP_HOST, APP_PORT, app  # noqa: E402


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=APP_HOST,
        port=APP_PORT,
        reload=False,
        log_level="info",
        access_log=True,
    )
