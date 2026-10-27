# Clubhead detector — research log

One verified entry per run. Nothing below has been acted on yet; this is a
sourcing log, not a backlog. Entries are append-only — do not edit past
entries except to fix a factual error, and note the correction inline.

**Methodological note (2026-10-27):** this session's outbound network access
is restricted to `github.com` / `raw.githubusercontent.com`. `arxiv.org`,
`universe.roboflow.com`, `huggingface.co`, `paperswithcode.com`,
`*.github.io`, and even `en.wikipedia.org` all fail DNS resolution through
the fetch tool. WebSearch still works and surfaces non-GitHub sources, but
only claims backed by a page this session could actually load count as
"verified" below. Anything found only via a search snippet is flagged as
unverified and excluded. Future runs should check whether this restriction
still holds — if the environment's network policy has been widened, dataset
sources on Roboflow Universe and papers on arXiv become checkable directly.

---

## 2026-10-27 — Video camouflaged-object-detection via motion correlation (SLT-Net / MoCA-Mask)

**What it is:** Cheng et al., "Implicit Motion Handling for Video Camouflaged
Object Detection" (CVPR 2022) — a two-stage architecture for finding
camouflaged objects in video. A short-term module takes a *pair of
consecutive frames* and builds a dense correlation volume between them
(implicit motion estimation, no explicit optical flow) to predict the mask
in the reference frame; a long-term module then takes a sequence of T
short-term predictions and refines them with a spatiotemporal transformer
for consistency across time. This is architecturally the same idea the task
brief points at for the camouflage failure mode: use inter-frame motion to
find an object that appearance alone can't separate from clutter.

**URL:** https://github.com/XuelianCheng/SLT-Net (code), dataset release
MoCA-Mask linked from the same README (Google Drive; link present in the
page but not independently verified as live — Drive links aren't fetchable
through this session's network restriction).

**Licence — verbatim:** none. I fetched the repo page directly: there is no
`LICENSE` file in the file listing, no licence badge, and no licence section
in the README. The only usage text is a citation request ("If you find this
code useful, please consider to cite our work"). No licence stated means
default copyright applies — **commercial use is not permitted** without
contacting the authors. The underlying MoCA dataset it trains/evaluates on
("Betrayed by Motion", Lamdouar et al., ACCV 2020) is a separate academic
release with its own terms, not checked here (its project page is on
`*.github.io`, unreachable this session) — assume research-only until shown
otherwise.

**Which failure mode:** camouflage, primarily. Not motion blur — see caveat
below.

**Why it's relevant, and why it doesn't just transfer:** The mechanism (pair
two consecutive frames, let the model find what moved) is directionally
right for our camouflage failures — a dark clubhead against dark clothing
or foliage has near-zero single-frame signal but is the only moving thing
between two frames. But MoCA's whole premise is the opposite motion regime
from ours: camouflaged animals are detectable *because they move slowly and
subtly* relative to the background over many frames; a golf clubhead at
impact does the opposite — it displaces by a large fraction of the frame
between consecutive samples and blurs. A correlation volume tuned to
resolve slow, small displacements is not obviously the right tool for large,
fast ones, and the paper gives no evidence either way for that regime.
Separately, this approach outputs per-pixel segmentation masks trained on
dense mask labels, not boxes — our labeling spec and eval harness are
box-based, so adopting it as-is would mean a parallel mask-labeling effort,
not a drop-in detector change.

**Effort vs. payoff:** High effort, uncertain payoff, if taken as "port this
repo." Reimplementing and retraining a two-stream correlation + transformer
architecture on our box-labeled data, with no commercially-usable pretrained
weights or dataset to start from, is a multi-week undertaking with no
golf-domain evidence it even applies to fast-motion camouflage rather than
slow-motion camouflage. The one cheap, low-risk idea worth lifting from this
without touching the repo at all: add a frame-differencing or stacked
consecutive-frame channel as extra input to the existing YOLO11n detector
(no architecture change, no new dataset, just feed frame *t-1* alongside
frame *t*) and see empirically whether that recovers any of the zero-candidate
camouflage frames — this is untested and not yet verified to work, but it is
a half-day experiment against the existing held-out set rather than a new
research program. Logging the idea here; it is not yet a validated finding.
