# Swing Sensei Native iOS Build Notes

The app is now native iOS first: SwiftUI for UI, AVFoundation for capture/playback, file-backed persistence for swing history, and the existing FastAPI/MediaPipe backend for pose analysis.

## App Surface

- Capture tab:
  - Native camera recording through `AVCaptureSession`.
  - Photos video import through `PhotosPicker`.
  - Imported or recorded videos are copied into app documents.
- Trim flow:
  - AVPlayer preview.
  - Generated thumbnail strip.
  - Draggable start/end handles.
  - Flip and rotate controls before analysis.
  - Required club selection before analysis.
  - Trim bounds and edit metadata sent with analysis upload.
- History tab:
  - Persistent swing cards.
  - Reopen completed analysis.
  - Resume trim or retry failed analysis.
- Analysis viewer:
  - AVFoundation playback.
  - Skeleton overlay and wrist trail.
  - Position chips, scrubber, frame step, and playback rates.
  - Score, quality warnings, swing thought, feedback cards, and debug links.
  - On-demand Gemini AI feedback when `GEMINI_API_KEY` is configured on the backend.
- Settings tab:
  - Backend URL stored with `@AppStorage`.

## Backend

The backend remains FastAPI + MediaPipe and exposes:

- `GET /health`
- `POST /analyze-swing`
- `POST /ai-analysis`

The native app posts `swing_id`, `client_duration_ms`, optional `trim_start_ms`, optional `trim_end_ms`, and the video file.

## Verify

```bash
cd ios-native
CLANG_MODULE_CACHE_PATH=.build/ModuleCache swift test
```

```bash
xcodebuild -project ios-native/SwingSenseiNative.xcodeproj \
  -scheme SwingSenseiNative \
  -configuration Debug \
  -destination generic/platform=iOS \
  CODE_SIGNING_ALLOWED=NO \
  -derivedDataPath ios-native/.derived \
  build
```

```bash
cd backend
pytest
```
