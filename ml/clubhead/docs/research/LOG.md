# Clubhead detector research log

Append-only. Each entry is one verified, genuinely new finding. Entries are
off-limits for repeat research in later runs — read this whole file before
starting new research.

---

## 2026-10-23 — DeFMO (fast-moving-object deblurring, MIT) as a motion-blur synthesis source

**What it is:** [DeFMO](https://github.com/rozumden/DeFMO) ("Deblurring and
Shape Recovery of Fast Moving Objects", Rozumnyi et al., CVPR 2021). A
network + training pipeline for recovering sharp sub-frame appearance and
shape of small, fast-moving blurred objects (ball-sized objects filmed at
normal frame rates) from a single blurred frame plus a clean background
frame. The authors explicitly cite sports (ball/rotation speed measurement)
as a target application — this is squarely the "small fast blurred object"
problem class our clubhead failure mode falls into.

**URL:** https://github.com/rozumden/DeFMO (code + training pipeline).
Pretrained checkpoint: https://polybox.ethz.ch/index.php/s/M06QR8jHog9GAcF
(linked from the repo README, confirmed live).

**Licence:** Verified by fetching `LICENSE` directly from the repo
(`raw.githubusercontent.com/rozumden/DeFMO/master/LICENSE`) — full MIT
License text, copyright Denys Rozumnyi 2022. **Commercial use of the code
itself is permitted.**

Important carve-out: the repo's *synthetic training-data generation*
pipeline (Blender 2.79b renderer) composites ShapeNetCore.v2 3D models onto
VOT/Sports-1M backgrounds to produce its training set, and the README states
the pre-generated dataset "can't be shared due to ShapeNet licensing."
ShapeNet's own terms are research-use-only. **Any reuse of DeFMO that
pulls in ShapeNet assets (their 3D object models) is not commercially
clean.** The MIT-licensed code and the general blur-compositing technique
are fine to reuse; ShapeNet's actual 3D model library is not.

**Which failure mode:** Motion blur, not camouflage.

**Why it's relevant to this model specifically:** It does not fix our
problem out of the box — it's a different architecture (temporal
super-resolution of a blurred ball-like object given a clean background
plate) solving a different task (deblur-and-render, not detect-a-box), and
it needs a clean background frame per shot, which handheld swing footage
mostly doesn't give you (a locked-off tripod/simulator-bay camera would).
But its *blur-kernel compositing approach* — rendering a sharp object,
applying a physically-plausible motion-blur kernel, and compositing onto a
real background with a correctly expanded ground-truth box — is directly
exploitable without touching ShapeNet or the trained network at all: take
our own sharp, correctly-labelled clubhead crops (not ShapeNet CAD models),
apply a motion-blur kernel at various swing speeds/angles, and composite
them onto real backgrounds to synthesize additional blurred training
frames with correctly-elongated boxes. That directly targets the measured
gap (median labelled-box elongation 1.60, p90 3.01 — most "blurred" labels
aren't actually blurred) without needing new blurred footage.

Running the actual DeFMO network as an on-device *inference-time*
deblurring preprocessor is a separate, much weaker idea and I'm flagging it
as low-payoff: it's a full encoder-decoder temporal-super-resolution model,
not designed for CoreML/mobile export, it needs a background plate per
frame (unavailable for a panning handheld shot), and it would add
significant latency to a pipeline whose whole point is running YOLO11n
on-device in real time. Don't pursue that part.

**Effort vs. payoff:** Low-to-medium effort, medium payoff. No need to
train or run DeFMO at all — just borrow the "sharp crop + blur kernel +
composite + expand box" idea and build a small in-house augmentation step
(this could be a few dozen lines: synthetic motion blur via directional PSF
convolution on existing clubhead crops, already a known technique
independent of DeFMO). The genuinely new piece from this search is
confirmation that a published, MIT-licensed, peer-reviewed implementation
of exactly this blur-compositing idea exists and is inspectable for
reference — not a drop-in fix. Recommend: prototype the synthetic-blur
augmentation in-house; do not invest engineering time in running DeFMO's
network itself on-device.

**Commercial-use verdict:** DeFMO code — yes (MIT). DeFMO's own synthetic
training dataset / ShapeNet-based renderer — no, research-only. Our own
re-implementation of the blur-compositing idea using our own crops — yes
(no third-party assets involved).
