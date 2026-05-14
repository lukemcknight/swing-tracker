import json
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from gemini_analyzer import (
    GeminiConfigError,
    GeminiResponseError,
    GeminiTimeoutError,
    analyze_swing_with_gemini,
)
from swing_analyzer import analyze_video

app = FastAPI(title="Swing Sensei Analyzer")
DEBUG_DIR = Path(__file__).resolve().parent / ".debug"
DEBUG_DIR.mkdir(exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/debug", StaticFiles(directory=DEBUG_DIR), name="debug")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze-swing")
async def analyze_swing(
    swing_id: str = Form(...),
    client_duration_ms: int | None = Form(None),
    trim_start_ms: int | None = Form(None),
    trim_end_ms: int | None = Form(None),
    rotation_degrees: int | None = Form(None),
    mirror_horizontal: bool = Form(False),
    video: UploadFile = File(...),
) -> dict:
    suffix = Path(video.filename or "swing.mp4").suffix or ".mp4"

    try:
        with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(await video.read())

        return analyze_video(
            swing_id=swing_id,
            video_path=temp_path,
            client_duration_ms=client_duration_ms,
            debug_dir=DEBUG_DIR,
            debug_url_path="/debug",
            trim_start_ms=trim_start_ms,
            trim_end_ms=trim_end_ms,
            rotation_degrees=rotation_degrees,
            mirror_horizontal=mirror_horizontal,
        )
    except RuntimeError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    finally:
        if "temp_path" in locals():
            temp_path.unlink(missing_ok=True)


@app.post("/ai-analysis")
async def ai_analysis(
    swing_id: str = Form(...),
    club: str = Form(...),
    analysis_json: UploadFile = File(...),
    client_duration_ms: int | None = Form(None),
    trim_start_ms: int | None = Form(None),
    trim_end_ms: int | None = Form(None),
    rotation_degrees: int | None = Form(None),
    mirror_horizontal: bool = Form(False),
    video: UploadFile = File(...),
) -> dict:
    suffix = Path(video.filename or "swing.mp4").suffix or ".mp4"

    try:
        analysis = json.loads((await analysis_json.read()).decode("utf-8"))
        if not isinstance(analysis, dict):
            raise ValueError("analysis_json must be an object.")
    except (UnicodeDecodeError, ValueError) as error:
        raise HTTPException(status_code=422, detail="analysis_json must be valid JSON.") from error

    try:
        with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(await video.read())

        return await analyze_swing_with_gemini(
            swing_id=swing_id,
            club=club,
            video_path=temp_path,
            analysis=analysis,
            client_duration_ms=client_duration_ms,
            trim_start_ms=trim_start_ms,
            trim_end_ms=trim_end_ms,
            rotation_degrees=rotation_degrees,
            mirror_horizontal=mirror_horizontal,
        )
    except GeminiConfigError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except GeminiTimeoutError as error:
        raise HTTPException(status_code=504, detail=str(error)) from error
    except (GeminiResponseError, RuntimeError) as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    finally:
        if "temp_path" in locals():
            temp_path.unlink(missing_ok=True)
