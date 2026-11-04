# Clubhead detector research log

Append-only. One verified entry per run. Entries already here are off-limits
for future runs — do not repeat a topic already logged, even if the specific
source differs.

---

## 2026-11-04 — SoccerSynth-Detection: synthetic camera-motion-blur pretraining measurably helps a YOLOv8n-family detector transfer to real blurred frames

**What it is:** A synthetic dataset + generator for soccer-player detection,
built specifically to include "simulated camera motion blur" alongside random
lighting/textures, released alongside the paper "SoccerSynth-Detection: A
Synthetic Dataset for Soccer Player Detection." The authors pretrained
YOLOv8n on the synthetic (blurred) data and report that, on transfer to real
motion-blurred frames, it outperformed models trained only on real data
(SoccerNet-Tracking / SportsMOT).

**URL:** https://github.com/open-starlab/SoccerSynth-Detection

**Licence:** Apache License 2.0 (`LICENSE.md` in the repo, confirmed via
GitHub's own license detection and by fetching the file). Apache-2.0 is a
standard permissive licence — commercial use is explicitly permitted;
conditions are limited to preserving copyright/attribution notices and
marking changed files. **Commercial use: permitted**, including by this
project.

**Verification performed:** Fetched the GitHub repo page and the raw
`LICENSE.md` and `README.md` directly (network access to `arxiv.org`,
`huggingface.co`, and `universe.roboflow.com` is blocked for this session's
environment, so the arXiv PDF itself (2501.09281) could not be read — only
the GitHub repo, which mirrors the abstract in its README). Confirmed: repo
is public, non-archived, Apache-2.0 licensed, and has working download links
(Google Drive) for the dataset, the Windows generator binary, and the
pretraining/transfer experiment result folders. Could not independently
verify the Google Drive links still resolve (Google Drive is outside this
session's allowed network domains) or read the exact blur-synthesis method
(PSF convolution vs. a simpler directional filter) — that detail is in the
paper body, which was unreachable.

**Which failure mode this addresses:** Motion blur, primarily. It is
evidence — not proof, since the domain is soccer players, not golf clubs —
that the general technique the v2 design doc already commits to (synthetic
motion-blur augmentation before training a YOLO-family nano model) is
empirically worth the investment, specifically for the blur regime this
project has almost no real labelled examples of (median box elongation 1.60).
It does not address camouflage at all.

**Why it matters for this model specifically:** The design spec says blur
augmentation will come from Albumentations' generic `MotionBlur` transform
(a randomized kernel blur, not object-motion-aware). This paper's result
suggests that's probably not the ceiling — a renderer-level simulation of
*actual camera/object motion* (not just a blurred kernel applied to an
otherwise-sharp crop) produces training signal strong enough to move a
YOLOv8n-class detector's real-world blur performance, which is exactly the
untested gap this project has (zero genuinely blurred clubheads in training,
indoor/simulator footage never measured). It's a vote of confidence for
building a purpose-made blurred-clubhead synthesis step (e.g., compositing
sharp clubhead crops onto backgrounds with a PSF kernel derived from the
labelled/estimated club speed at that swing phase) rather than relying on
Albumentations' default knob.

**Effort vs. payoff:** Low effort to use as *evidence/justification* (already
done — this entry). High effort and uncertain payoff to actually reuse their
asset: the generator is a soccer-player-specific Windows binary with no
visible source in the repo, so it cannot be re-skinned for golf clubs, and
the dataset's object class (people) doesn't transfer. The real payoff isn't
"use their data," it's "their result justifies spending phase-2 effort on a
custom motion-blur compositing step for clubhead crops instead of trusting
Albumentations' generic `MotionBlur`." Medium-low payoff overall, but
genuinely new and directly load-bearing for the blur failure mode, which the
log must not let the camouflage finding monopolize.
