# Swing Sensei Native iOS

SwiftUI + AVFoundation iOS app for recording, trimming, analyzing, and reviewing golf swing videos.

## Run

1. Start the backend from the repo root:

   ```bash
   cd backend
   source .venv312/bin/activate
   MPLCONFIGDIR=.mplconfig XDG_CACHE_HOME=.cache uvicorn main:app --host 0.0.0.0 --port 8000
   ```

   To enable AI Analysis, export `GEMINI_API_KEY` before starting the backend. Optional overrides are `GEMINI_MODEL` and `GEMINI_MEDIA_RESOLUTION`.

2. Open the native project:

   ```bash
   open ios-native/SwingSenseiNative.xcodeproj
   ```

3. Run the `SwingSenseiNative` scheme on an iOS 17+ simulator or device.

The simulator default backend URL is `http://127.0.0.1:8000`. For a physical device, use your Mac's LAN IP in the app's Settings tab.

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

## Scope

- Native camera recording with countdown, haptics, and tap-to-stop.
- Photos video selection.
- Persistent swing history in app documents.
- Pre-analysis edit view with AVPlayer, generated thumbnails, trim handles, flip, and rotate.
- Required club selection before analysis.
- Multipart backend upload with trim bounds and video edit metadata.
- Native AVFoundation analysis viewer with phase chips, scrubber, frame stepping, speed controls, skeleton overlay, wrist trail, score, quality banner, swing thought, feedback cards, and on-demand Gemini AI feedback.
