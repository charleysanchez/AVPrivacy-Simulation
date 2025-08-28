# Real‑Time Face Anonymization for AV/XR (Updated)

This README documents the **updated** anonymization pipeline and how to run it locally, in CARLA, and on an NVIDIA Jetson with RealSense. It replaces the prior person‑detector + heavy anonymizer with a **face‑specialized detector** and a **fast, non‑invertible mosaic+noise** method that preserves privacy at real‑time speeds.

---

## What changed vs. prior work

**Before (summary):**
- Person detection via `fasterrcnn_resnet50_fpn` (COCO) → coarse boxes; misses faces when only heads are visible.
- Depth‑guided thresholding inside boxes; single path required depth to be reliable.
- Multi‑stage anonymizers (noise, smoothing, color transforms, resampling) → heavier compute.

**Now (this repo):**
- **Face detector:** InsightFace **SCRFD** via `insightface.app.FaceAnalysis` with ONNX Runtime (prefers **TensorRT → CUDA → CPU**). ~**16 ms** / frame at **640×480** on Jetson Orin Nano.
- **Flexible masks:** depth‑guided **or** box‑only. Box‑only supports **padding** and **oval (elliptical)** fills; both paths apply morphological **dilation** to reduce boundary leaks.
- **Anonymizer:** **single‑pass mosaic (pixelation) for the whole frame** + **bounded random noise** applied **only under the mask**. Non‑invertible in practice and ~**20 ms** / frame on Jetson Orin Nano.
- **Depth privacy:** optional Gaussian noise added **only** where the mask is 1 (z‑anonymization).
- **Evaluation:** alignment with SAM masks; per‑environment and per‑facecount **Dice** and **Recall** aggregates; FPS + latency reporting.

---

## Installation

### CPU/GPU desktop (conda recommended)
```bash
conda create -n av-privacy python=3.10 -y
conda activate av-privacy

pip install insightface onnxruntime-gpu opencv-python-headless numpy torch torchvision
# If GPU not available or you want CPU only:
# pip install onnxruntime
```

---

## CARLA (synthetic data)

**Why:** generate controlled RGB‑D sequences (crowds, severe angles) without collecting real faces.

**Run CARLA headless (example):**
```bash
conda activate python37-911          # environment used by prior team
./force_nvidia_egl.sh ./CarlaUE4.sh -RenderOffscreen
```


---

## Quickstart: use the pipeline class

```python
from av_privacy_masker import AVPrivacyMasker
import cv2

mp = AVPrivacyMasker(
    device="cuda",              # or "cpu"
    conf_thresh=0.5,
    anon_block=24, anon_noise=20,
    dilate_kernel=13,
    det_size=(640, 640),
    verbose=False,
    enable_depth_anon=False     # set True to add noise to depth where masked
)

bgr = cv2.imread("sample_rgb.jpg")         # 640×480 recommended
depth = cv2.imread("sample_depth.png", -1) # optional (z16 or float), can be None

rgb_anon, depth_anon, mask, boxes, times = mp.process_arrays(
    bgr, depth,
    use_depth_for_mask=bool(depth is not None),
    pad_ratio=0.25,
    oval_boxes=True,
)
print("Timings (s):", times)
cv2.imwrite("out_anon.jpg", rgb_anon)
```

### Why the anonymizer is fast *and* robust
- We **mosaic once** for the entire frame, then **copy only masked pixels**. No per‑pixel recomputation outside the mask.
- We inject **bounded random noise** within the masked region to make pixelation **non‑invertible in practice**.
- The result hides identity while preserving scene context for downstream AV/XR perception.

---


---

## Evaluation (Dice / Recall) vs. SAM

We compare our masks against SAM‑generated masks for the same sessions. The evaluation script supports strict alignment (start/stop frames, small shift search), environment mapping, and per‑facecount aggregation.

- **Dice**: \( 2|A∩B| / (|A|+|B|) \) — overlap similarity of our mask vs. SAM’s.
- **Recall**: \( |A∩B| / |B| \) — how much of SAM’s mask we covered.
- Aggregates are reported by **environment** (Lab/Hallway/Outside) and **facecount** (1 vs. 2).

---

## Troubleshooting

- **“No masks produced”**: ensure sessions contain detectable faces; try lowering `conf_thresh` (e.g., `0.3`) and confirm image size is **640×480**.
- **ONNX/TensorRT provider not applied**: check `onnxruntime-gpu` is installed and that TensorRT is available; the code falls back to CUDA → CPU automatically.
- **Depth unreliable**: set `use_depth_for_mask=False` in `process_arrays`; box‑only masks with **padding** and **oval** coverage are robust for faces.
- **FPS dips**: reduce `det_size` to `512`, lower `anon_block` noise, and ensure you’re not saving every frame to disk.

---

## Reusing the code (for the next contributor)

- `AVPrivacyMasker.detect_faces` returns face **[x1,y1,x2,y2]** boxes.
- `build_mask_numpy` uses **depth statistics** around each face to perform fast thresholding and ROI masking; `build_mask_from_boxes` creates **depth‑free** masks with padding and oval fill.
- `fast_pixelate` performs the **single‑mosaic + masked copy** with optional random noise to resist inversion.
- The public API (`process_arrays`, `process_paths`) remains stable regardless of mask path (depth vs. box‑only).

---

## License & attribution

- Face detector: [InsightFace / SCRFD](https://github.com/deepinsight/insightface).
- SAM used **only for evaluation** (ground‑truth‑like masks).
- CARLA simulator for synthetic data.
- This repo is for research/educational use; review licenses of upstream dependencies before commercial use.
