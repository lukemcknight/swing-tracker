# Swing Sensei

Native iOS golf swing analysis app backed by a local FastAPI/MediaPipe analysis service.

The app is now SwiftUI + AVFoundation. The previous Expo/React Native app surface has been removed.

## Run the Backend

```bash
cd backend
/opt/homebrew/bin/python3.12 -m venv .venv312
source .venv312/bin/activate
pip install -r requirements.txt
MPLCONFIGDIR=.mplconfig XDG_CACHE_HOME=.cache uvicorn main:app --host 0.0.0.0 --port 8000
```

## Run the iOS App

```bash
open ios-native/SwingSenseiNative.xcodeproj
```

Run the `SwingSenseiNative` scheme on an iOS 17+ simulator or device.

- Simulator backend URL: `http://127.0.0.1:8000`
- Physical device backend URL: use your Mac's LAN IP, for example `http://192.168.1.20:8000`

The backend URL is editable in the app's Settings tab.

## Native App Features

- Capture tab with native camera recording and Photos video import.
- Persistent swing library stored in app documents.
- Pre-analysis edit workflow with AVPlayer preview, thumbnail trim strip, flip, and rotate controls.
- Multipart upload to `POST /analyze-swing` with trim bounds and video edit metadata.
- AVFoundation analysis viewer with skeleton overlay, wrist trail, position chips, scrubber, frame stepping, speed controls, score, quality warnings, and feedback cards.
- History tab for reopening saved swings after app relaunch.

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
