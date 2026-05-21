# Slice-Fixer Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Strip the app down to the slice-fixer MVP by removing the Arcade, Play, launch-monitor, and Rounds features, leaving a focused Capture → Analyze → Club Path flow.

**Architecture:** All Arcade + Rounds types live in one file (`Sources/SwingSenseiCore/ArcadeCourse.swift`, 984 lines). The consumers are `PlayView.swift`, `ArcadeDebugLabView.swift`, a cluster of members in `AppViewModel.swift`, and a block of `SwingSenseiCoreTests.swift`. `LaunchMonitorReadoutView.swift` is consumed only by `ContentView.swift`. Removing the consumers, then the definitions, in dependency order keeps the build green at every step.

**Tech Stack:** Swift, SwiftUI, Xcode project (`ios-native/SwingSenseiNative.xcodeproj`). Build = Xcode Cmd+B; tests = Xcode Cmd+U.

**Preservation:** The full pre-cleanup state is committed at `120eb5a` and bookmarked by branch `parked/pre-slice-cleanup`. All removed code is recoverable from there.

**Note on Xcode steps:** Deleting a `.swift` file requires removing its reference from the Xcode project. After deleting a file on disk, the file shows red in Xcode's Project Navigator — right-click it → Delete → Remove Reference. Each task's build step covers this.

---

### Task 1: Delete the orphaned Arcade views

`PlayView.swift` and `ArcadeDebugLabView.swift` are not presented from anywhere in the app — they are pure consumers of Arcade symbols, so removing them cannot break other code.

**Files:**
- Delete: `ios-native/SwingSenseiNative/PlayView.swift`
- Delete: `ios-native/SwingSenseiNative/ArcadeDebugLabView.swift`

- [ ] **Step 1: Delete the two files**

```bash
git rm ios-native/SwingSenseiNative/PlayView.swift ios-native/SwingSenseiNative/ArcadeDebugLabView.swift
```

- [ ] **Step 2: Remove the Xcode references**

In Xcode Project Navigator, `PlayView.swift` and `ArcadeDebugLabView.swift` now show red. Select both → right-click → Delete → Remove Reference.

- [ ] **Step 3: Build**

Run: Xcode Cmd+B.
Expected: Build succeeds. (These files had no consumers, so nothing else changes.)

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "Remove orphaned Arcade/Play views"
```

---

### Task 2: Rewire ContentView and delete LaunchMonitorReadoutView

`ContentView` wraps `AnalysisViewer` in `LaunchMonitorReadoutView`. Promote `AnalysisViewer` to be presented directly; it gains responsibility for its own dismissal via `onClose`.

**Files:**
- Modify: `ios-native/SwingSenseiNative/ContentView.swift` (the `.fullScreenCover` block, lines 58–91)
- Delete: `ios-native/SwingSenseiNative/LaunchMonitorReadoutView.swift`

- [ ] **Step 1: Replace the fullScreenCover body**

In `ContentView.swift`, replace this:

```swift
            {
                LaunchMonitorReadoutView(
                    swing: swing,
                    analysis: analysis,
                    onDismiss: { viewModel.viewerSelection = nil }
                ) {
                    AnalysisViewer(
                        swingID: swing.id,
                        videoURL: swing.videoURL,
                        analysis: analysis,
                        club: swing.club,
                        aiAnalysis: swing.aiAnalysis,
                        videoEditState: swing.videoEditState ?? .identity,
                        onClubChange: { club in
                            viewModel.updateClub(swingID: swing.id, club: club)
                        },
                        onRequestAIFeedback: { club in
                            try await viewModel.requestAIFeedback(
                                swingID: swing.id,
                                club: club,
                                baseURLString: baseURLString
                            )
                        },
                        onClose: {}
                    )
                }
            } else {
```

with this:

```swift
            {
                AnalysisViewer(
                    swingID: swing.id,
                    videoURL: swing.videoURL,
                    analysis: analysis,
                    club: swing.club,
                    aiAnalysis: swing.aiAnalysis,
                    videoEditState: swing.videoEditState ?? .identity,
                    onClubChange: { club in
                        viewModel.updateClub(swingID: swing.id, club: club)
                    },
                    onRequestAIFeedback: { club in
                        try await viewModel.requestAIFeedback(
                            swingID: swing.id,
                            club: club,
                            baseURLString: baseURLString
                        )
                    },
                    onClose: { viewModel.viewerSelection = nil }
                )
            } else {
```

- [ ] **Step 2: Delete LaunchMonitorReadoutView**

```bash
git rm ios-native/SwingSenseiNative/LaunchMonitorReadoutView.swift
```

- [ ] **Step 3: Remove the Xcode reference**

In Xcode, `LaunchMonitorReadoutView.swift` shows red → right-click → Delete → Remove Reference.

- [ ] **Step 4: Build**

Run: Xcode Cmd+B.
Expected: Build succeeds.

- [ ] **Step 5: Verify the flow**

Run the app in the Simulator. Analyze a swing (or open an existing complete swing from the Swings tab). Expected: the analysis viewer opens directly (no launch-monitor stats screen first), and its close control dismisses back to the app.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "Present AnalysisViewer directly, remove LaunchMonitorReadoutView"
```

---

### Task 3: Strip Arcade/Round code from AppViewModel

Remove the Rounds/Arcade cluster from `AppViewModel.swift`. After this task `AppViewModel` no longer references any Arcade or Round symbol, but `ArcadeCourse.swift` still exists and still compiles on its own — the build stays green.

**Files:**
- Modify: `ios-native/SwingSenseiNative/AppViewModel.swift`

- [ ] **Step 1: Delete the `PendingRoundShot` struct**

Remove the entire top-level struct (currently lines 15–19):

```swift
private struct PendingRoundShot: Equatable {
    let roundID: String
    let swingID: String
    let ballMode: ArcadeBallMode
}
```

- [ ] **Step 2: Delete the Round/Arcade stored properties**

Remove these `@Published` properties: `rounds`, `activeRoundID`, `pendingRoundShotSwingID`. Remove these private properties: `roundRepository`, `shotEngine`, and the `pendingRoundShot` property with its `didSet`. The kept properties are `swings`, `trimSelection`, `viewerSelection`, `processingMessage`, `appMessage`, and `repository`.

- [ ] **Step 3: Simplify the initializer**

Replace the initializer with:

```swift
    init(repository: SwingRepository = SwingRepository()) {
        self.repository = repository
    }
```

- [ ] **Step 4: Simplify `loadData()`**

Replace it with:

```swift
    func loadData() {
        loadSwings()
    }
```

- [ ] **Step 5: Delete the Round-only methods and computed properties**

Delete these members entirely: `loadRounds()`, `activeRound`, `activeHole`, `recentCompletedRounds`, `startStarterRound()`, `continueRound(_:)`, `undoLastRoundShot()`, `restartActiveRound()`, `abandonActiveRound()`, `retryPendingRoundShot(baseURLString:)`, `discardPendingRoundShot()`, `finishRecordedRoundShotVideo(_:club:ballMode:)`, `save(_ round:)`, and `applyRoundShot(roundID:swing:club:ballMode:analysis:)`.

- [ ] **Step 6: Remove the pending-round block from `saveTrimAndAnalyze`**

In `saveTrimAndAnalyze`, after `try save(swing)` (the one following `swing.status = .complete`), remove this block:

```swift
            if let pendingRoundShot, pendingRoundShot.swingID == swing.id {
                self.pendingRoundShot = nil
                do {
                    try applyRoundShot(
                        roundID: pendingRoundShot.roundID,
                        swing: swing,
                        club: club,
                        ballMode: pendingRoundShot.ballMode,
                        analysis: analysis
                    )
                    processingMessage = nil
                    UINotificationFeedbackGenerator().notificationOccurred(.success)
                    return
                } catch {
                    appMessage = AppMessage(text: "Swing analyzed, but the shot could not be added to the round: \(error.localizedDescription)")
                }
            }

```

The following three lines (`processingMessage = nil`, `viewerSelection = SwingSelection(id: swing.id)`, `UINotificationFeedbackGenerator().notificationOccurred(.success)`) remain.

- [ ] **Step 7: Simplify `cancelTrimSelection()`**

Replace it with:

```swift
    func cancelTrimSelection() {
        trimSelection = nil
    }
```

- [ ] **Step 8: Build**

Run: Xcode Cmd+B.
Expected: Build succeeds. If the compiler flags a remaining Arcade/Round reference in `AppViewModel.swift`, remove that line — it belongs to the deleted cluster.

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "Strip Rounds/Arcade code from AppViewModel"
```

---

### Task 4: Remove Arcade/Round tests

`SwingSenseiCoreTests.swift` has 14 test methods exercising `ArcadeShotEngine` / `RoundRecord` / `RoundRepository`. They must go before `ArcadeCourse.swift` is deleted, or the test target will not compile.

**Files:**
- Modify: `ios-native/Tests/SwingSenseiCoreTests/SwingSenseiCoreTests.swift`

- [ ] **Step 1: Delete the 14 Arcade/Round test methods**

Delete these test methods in full (each `func ... { ... }` body):
`testArcadeShotEngineUsesBallSpeedEstimate`, `testArcadeShotEngineFallsBackToClubSpeedThenDefaultCarry`, `testArcadeShotEngineWidensDispersionForLowConfidence`, `testArcadeShotEngineAppliesLandingZoneBias`, `testArcadeShotEngineLateralJitterDiffersBetweenSwingsButIsDeterministic`, `testArcadeShotEngineCenterShotsDriftOnlyOnLowConfidence`, `testRoundRecordUndoLastShotRevertsBallAndStrokes`, `testRoundRecordRestartHolesClearsAllStrokes`, `testRoundRecordAbandonMarksRoundComplete`, `testArcadeShotEngineFoamBallIgnoresBallSpeedForCarry`, `testArcadeShotEngineNoBallIgnoresLandingZoneAndForcesLowConfidence`, `testArcadeShotEngineAutoFinishesOnGreen`, `testRoundShotDecodesLegacyJSONWithoutBallMode`, `testRoundRepositoryMissingFileAndPersistence`.

- [ ] **Step 2: Remove now-unused test helpers**

Run: `grep -n "makeShotEstimate\|makeTemporaryDirectory" ios-native/Tests/SwingSenseiCoreTests/SwingSenseiCoreTests.swift`
If a helper is now referenced only by its own definition (zero call sites remain), delete the helper. If it is still called by a kept test, leave it.

- [ ] **Step 3: Build the test target**

Run: Xcode Cmd+B (or select the test target and build).
Expected: Build succeeds. Any remaining error naming an Arcade/Round symbol points to a test method or helper missed in Steps 1–2 — remove it.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "Remove Arcade/Round unit tests"
```

---

### Task 5: Delete ArcadeCourse.swift and verify

With every consumer gone, the 984-line `ArcadeCourse.swift` (all Arcade + Round type definitions) can be deleted.

**Files:**
- Delete: `ios-native/Sources/SwingSenseiCore/ArcadeCourse.swift`

- [ ] **Step 1: Delete the file**

```bash
git rm ios-native/Sources/SwingSenseiCore/ArcadeCourse.swift
```

- [ ] **Step 2: Remove the Xcode reference**

In Xcode, `ArcadeCourse.swift` shows red → right-click → Delete → Remove Reference.

- [ ] **Step 3: Confirm no references remain**

Run: `grep -rn "Arcade\|RoundRecord\|RoundRepository\|RoundStatus\|HoleState" ios-native/Sources ios-native/SwingSenseiNative ios-native/Tests`
Expected: no output. Any hit is a missed reference — remove it.

- [ ] **Step 4: Build and test**

Run: Xcode Cmd+B, then Cmd+U.
Expected: Build succeeds; all remaining tests pass.

- [ ] **Step 5: Verify the focused app**

Run the app in the Simulator. Expected: three tabs (Capture, Swings, Settings); analyzing a swing goes Capture → Trim → Analyze → AnalysisViewer with the club path. No Arcade, Play, or launch-monitor surfaces anywhere.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "Remove Arcade/Rounds subsystem (ArcadeCourse.swift)"
```

---

## Out of scope

- Session debug scaffolding (`print()` logging in `ClubHeadDetectionService`/`AppViewModel`, the diagnostic Core functions `plausibilityReport`/`scoreReport`, the "Pass 4 (temporary)" trim). That is a separate cleanup, tracked independently.
- The slice-diagnosis feature itself — covered by `docs/superpowers/specs/2026-05-20-slice-diagnosis-design.md`, to be planned after this cleanup lands.
