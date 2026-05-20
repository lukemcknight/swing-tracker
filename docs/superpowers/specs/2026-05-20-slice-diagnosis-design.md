# Slice Diagnosis — Design

**Date:** 2026-05-20
**Status:** Approved design, pending implementation plan

## Purpose

Turn the on-device clubhead detector from a *visualization* (the Club Path
overlay) into an *answer*: tell a golfer whether their swing path promotes a
slice, and what to do about it.

A club-path overlay alone shows the path but does not diagnose a problem. This
feature reads the detected path and delivers a plain-language verdict plus a
corrective cue.

## Scope and constraints

- **Swing path only.** A slice is caused by the clubface being open relative to
  the swing path at impact. The detector produces a clubhead *position* per
  frame, so we can measure the *path* (out-to-in vs in-to-out) but not the
  *clubface angle*. The diagnosis is therefore path-based: an out-to-in path is
  the classic slice pattern. Clubface estimation is explicitly out of scope.
- **Down-the-line captures only.** Swing path is only readable from a
  down-the-line camera angle, where an over-the-top move is visible as the
  downswing track coming down outside the backswing track. From a face-on
  capture the in/out direction runs toward/away from the lens and is not
  measurable. On non-DTL captures the feature is hidden.
- **No new Vision request or network call.** The diagnosis is pure geometry on
  the clubhead path already produced on-device by the CoreML detector. It runs
  in the same pass, right after `cleanedAndBridged`.

## Architecture

A new pure-logic component, `SwingPathDiagnoser`, in `SwingSenseiCore` alongside
`ClubPathStabilizer`. No platform dependencies; fully unit-testable.

Data flow:

1. **Backend** produces the base `SwingAnalysis` (pose keypoints, metrics).
2. **Apple Vision + CoreML** (`ClubHeadDetector` / `ClubHeadDetectionService`)
   populate per-frame clubhead positions, then `cleanedAndBridged` cleans them.
3. **`SwingPathDiagnoser`** runs pure geometry on the cleaned clubhead path and
   returns a `SwingPathDiagnosis`.
4. `ClubHeadDetectionService` attaches the result to a new optional field on
   `SwingAnalysis`: `swingPathDiagnosis: SwingPathDiagnosis?`.

`swingPathDiagnosis` is optional so the backend's JSON still decodes unchanged;
the iOS pipeline populates it locally.

## The measurement — signed loop area

The clubhead path from address through impact forms a loop: up the backswing,
down the downswing, the two tracks enclosing a region. The rotational direction
of that loop is the diagnosis:

- Downswing comes down *outside* the backswing track → over-the-top → out-to-in
- Downswing drops *inside* the backswing track → in-to-out
- Downswing retraces the backswing → on-plane → neutral

The signed area of the path polygon (shoelace formula) encodes this directly:

```
A = ½ · Σ (x_i · y_{i+1} − x_{i+1} · y_i)
```

- **Sign of A** → loop direction → out-to-in vs in-to-out.
- **|A|, normalized by the path's bounding-box area** → severity, scale-
  independent.

This needs no pose keypoints and no "which way is the body" logic — the signed
area inherently encodes direction. One calibration step during implementation:
confirm which sign corresponds to out-to-in, using a known down-the-line clip.

## Classification

Three buckets from the normalized signed area:

| Bucket      | Condition                              |
|-------------|----------------------------------------|
| Neutral     | `|normalized area| < neutralThreshold` |
| Out-to-in   | area beyond threshold, slice-side sign |
| In-to-out   | area beyond threshold, other sign      |

`neutralThreshold` starts at 0.05 and is tuned against the test clips during
implementation. Severity magnitude is retained on the result for future use;
the MVP only acts on the three buckets.

## Output

```swift
public struct SwingPathDiagnosis: Codable, Equatable {
    public enum PathDirection: String, Codable { case outToIn, neutral, inToOut }
    public let direction: PathDirection
    public let severity: Double      // normalized |signed area|
    public let verdict: String       // plain-language headline
    public let cue: String?          // corrective cue; nil for neutral
}
```

Verdict and cue text are canned per bucket. Examples (final wording during
implementation):

- **Out-to-in:** verdict "Out-to-in path — the classic slice pattern." cue
  "Feel the club drop behind you as you start down, rather than throwing it out
  toward the ball."
- **Neutral:** verdict "On-plane path — no slice tendency here." cue `nil`.
- **In-to-out:** verdict "In-to-out path — promotes a draw, or a hook/push if
  exaggerated." cue a mild cue or `nil`.

## Confidence gate

The diagnosis requires both the backswing and downswing tracks to be detected
well enough to form a meaningful loop. Gate conditions — any failure →
`swingPathDiagnosis = nil`:

- Capture is not down-the-line (read the existing capture-angle signal from
  `analysisQuality`; if not cleanly available, resolve during implementation).
- Too few detected clubhead points spanning the backswing (address→top) and the
  downswing (top→impact) portions.
- Degenerate path: too few points overall, or a zero-area bounding box.

When `nil`, the Club Path screen shows no verdict — consistent with the existing
"Partial" handling.

## UI

On the Club Path screen (the club-path section of `AnalysisViewer`): when
`swingPathDiagnosis` is non-nil, show a verdict card near the path overlay — the
verdict headline plus the cue — using the same visual language as the existing
"Path confidence" card. When `nil`, show nothing extra.

## Error handling

- Sparse detection, non-DTL capture, or a degenerate path → `nil` diagnosis,
  handled gracefully (no verdict shown, no crash).
- The component is total: any input array of frames yields either a valid
  diagnosis or `nil`, never an error.

## Testing

- **Unit tests** in `SwingSenseiCoreTests` for `SwingPathDiagnoser`:
  - Synthetic over-the-top loop → classified out-to-in.
  - Synthetic in-to-out loop → classified in-to-out.
  - Synthetic on-plane path (near-zero area) → classified neutral.
  - Degenerate inputs (too few points, zero bounding box) → `nil`.
- **Sign-convention calibration:** one known out-to-in down-the-line clip used
  to lock which signed-area sign maps to out-to-in.
- **Integration sanity:** run the 8 held-out test clips and eyeball that the
  verdicts are reasonable.

## Out of scope

- Clubface angle estimation.
- Face-on capture support.
- Ball-flight integration.
- Severity sub-grading ("slight" vs "strong") in the verdict text.
