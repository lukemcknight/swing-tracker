# Clubhead labeling spec

How to draw `clubhead` boxes for the SwingSensei detector. Consistency here is
what fixes loose boxes (the low varied-IoU score). Apply every rule the same
way on every frame, in training data and test data alike.

## What "clubhead" means

The box encloses the **club head only** — the mass at the end of the shaft that
strikes the ball. It does **not** include the shaft or the grip.

- The box is the tightest axis-aligned rectangle that contains the whole head.
- For irons and wedges: the blade/face plus the hosel where it meets the head.
- For drivers/woods: the whole head volume.
- Include the head's full visible extent — toe, heel, sole, crown.

## The motion-blur rule (decided convention)

During the fast parts of the swing the clubhead is a blurred streak.

**Rule: box the full visible clubhead including its motion-blur streak.** The
rectangle covers the entire smeared region the head occupies in that frame.

Rationale: it is the most consistent, least subjective call and it matches what
is actually on screen. (This convention is flagged for revisit in phase 2 if it
turns out to drive path wobble.)

## Partial occlusion

If the clubhead is partially hidden (behind the body, ball, or frame edge):

- Box only the **visible** portion.
- If essentially all of the head is hidden, leave the frame **unannotated** —
  it becomes a negative frame.

## Negative frames

Leave a frame unannotated when no clubhead is meaningfully visible — for
example, follow-through frames where the club has left the frame, or non-swing
footage. Negative frames are first-class training data against phantom
detections; do not skip or delete them.

## One box per frame

There is exactly one clubhead. Never draw more than one `clubhead` box on a
frame. If two candidates seem plausible, pick the real club head and box only
that.
