# Swing Sensei Analysis Backend

FastAPI service that accepts a swing video, samples frames with OpenCV, runs MediaPipe Pose, detects key swing positions, and returns `SwingAnalysis` JSON to the native iOS app.
It can also run an optional Gemini coaching pass against the trimmed swing video and existing pose analysis.

## Run

```bash
cd backend
/opt/homebrew/bin/python3.12 -m venv .venv312
source .venv312/bin/activate
pip install -r requirements.txt
MPLCONFIGDIR=.mplconfig XDG_CACHE_HOME=.cache uvicorn main:app --host 0.0.0.0 --port 8000
```

Add `--reload` while developing if your environment allows file watching.

For AI feedback, set a Gemini API key before starting the server:

```bash
export GEMINI_API_KEY="..."
export GEMINI_MODEL="gemini-2.5-pro"
export GEMINI_MEDIA_RESOLUTION="MEDIA_RESOLUTION_HIGH"
```

For the iOS simulator, keep the app backend URL at `http://127.0.0.1:8000`.
For a physical iPhone, set the backend URL in the app Settings tab to your Mac's LAN IP, for example `http://192.168.1.20:8000`.

## API

- `GET /health`
- `POST /analyze-swing`
  - `swing_id`: string form field
  - `client_duration_ms`: optional integer form field
  - `trim_start_ms`: optional integer form field
  - `trim_end_ms`: optional integer form field
  - `rotation_degrees`: optional integer form field (`0`, `90`, `180`, or `270`)
  - `mirror_horizontal`: optional boolean form field
  - `video`: uploaded movie file
- `POST /ai-analysis`
  - `swing_id`: string form field
  - `club`: selected club label (`D`, `3W`, `5H`, `7`, `PW`, `Other`, etc.)
  - `analysis_json`: uploaded `application/json` file part containing the existing `SwingAnalysis` JSON
  - `client_duration_ms`, `trim_start_ms`, `trim_end_ms`, `rotation_degrees`, `mirror_horizontal`: same edit metadata as `/analyze-swing`
  - `video`: uploaded movie file
