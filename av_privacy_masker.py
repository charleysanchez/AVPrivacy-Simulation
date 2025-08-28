#!/usr/bin/env python3
# av_privacy_masker.py

import os
import time
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from insightface.app import FaceAnalysis




class AVPrivacyMasker:
    """
    Face-based anonymization pipeline:
      - Detect faces (InsightFace SCRFD via FaceAnalysis)
      - Build depth-based mask per face
      - Anonymize RGB inside mask (pixelate + noise)
      - Add noise to depth inside mask
      - Optional save to disk

    Typical use:
        mp = AVPrivacyMasker(device="cuda")
        rgb_anon, depth_anon, mask_np, boxes, timings = mp.process_paths(
            img_path, depth_path, out_rgb="rgb_out.png", out_depth="depth_out.png", save=True
        )
    """

    def __init__(
        self,
        device: Optional[str] = None,
        conf_thresh: float = 0.5,
        window_size: int = 10,
        depth_std_mult: float = 75.0,
        anon_block: int = 16,
        anon_noise: int = 20,
        dilate_kernel: int = 31,
        detector_name: str = "buffalo_s",
        det_size: Tuple[int, int] = (640, 640),
        verbose: bool = False,
        enable_depth_anon: bool = False,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.conf_thresh = conf_thresh
        self.window_size = window_size
        self.depth_std_mult = depth_std_mult
        self.anon_block = anon_block
        self.anon_noise = anon_noise
        self.dilate_kernel = dilate_kernel
        self.verbose = verbose
        self.enable_depth_anon = enable_depth_anon

        # Init detector
        self.detector = FaceAnalysis(name=detector_name, allowed_modules=["detection"])
        self.detector.prepare(
            ctx_id=0 if self.device == "cuda" and torch.cuda.is_available() else -1,
            det_size=det_size,
        )

                # ---- Prefer TensorRT EP (then CUDA, then CPU) ----
        import onnxruntime as ort
        import os

        trt_opts = {
            "trt_fp16_enable": True,                 # big speedup on Orin
            "trt_int8_enable": False,                # unless you have calibration
            "trt_engine_cache_enable": True,         # cache engines on disk
            "trt_engine_cache_path": os.path.expanduser("~/.ort_trt_cache"),
            # Optional if you hit mem issues:
            # "trt_max_workspace_size": str(1<<30),   # 1GB
        }
        cuda_opts = {
            "device_id": "0",
            "cudnn_conv_use_max_workspace": "1",
        }

        provider_names, provider_opts = [], []
        if "TensorrtExecutionProvider" in ort.get_available_providers():
            provider_names.append("TensorrtExecutionProvider")
            provider_opts.append(trt_opts)
        provider_names += ["CUDAExecutionProvider", "CPUExecutionProvider"]
        provider_opts  += [cuda_opts, {}]

        # Switch providers for all InsightFace model sessions
        for key, model in self.detector.models.items():
            sess = getattr(model, "session", None)
            if sess is None:
                continue
            try:
                sess.set_providers(provider_names, provider_opts)
                if self.verbose:
                    print(f"[ORT] {key} providers -> {sess.get_providers()}")
            except Exception as e:
                if self.verbose:
                    print(f"[ORT] Could not set providers for {key}: {e}")

        # Prebuild dilation kernel
        k = max(1, int(self.dilate_kernel) // 2 * 2 + 1)  # force odd
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)).astype(np.uint8)

    # ----------------- Detection -----------------

    def detect_faces(self, bgr_np: np.ndarray) -> List[List[int]]:
        """Return list of [x1,y1,x2,y2] for faces above threshold."""
        faces = self.detector.get(bgr_np)
        boxes = []
        for f in faces:
            if f.det_score >= self.conf_thresh:
                x1, y1, x2, y2 = map(int, f.bbox)
                boxes.append([x1, y1, x2, y2])
        if self.verbose:
            print(f"[detect_faces] boxes={boxes}")
        return boxes

    # ------------- Depth profiling + mask ----------

    def _calc_depth_profile(self, depth_np: np.ndarray, box: List[int]) -> Optional[Dict]:
        """Mean/std window around box center → threshold for that person."""
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        hw = self.window_size // 2
        y0, y1w = max(cy - hw, 0), min(cy + hw + 1, depth_np.shape[0])
        x0, x1w = max(cx - hw, 0), min(cx + hw + 1, depth_np.shape[1])
        win = depth_np[y0:y1w, x0:x1w].reshape(-1)

        # Support z16 (uint16) and float depths
        if np.issubdtype(win.dtype, np.integer):
            valid = win > 0
            win = win[valid].astype(np.float32, copy=False)
        else:
            win = win[~np.isnan(win)]

        if win.size == 0:
            return None
        m, s = float(win.mean()), float(win.std())
        return {"mean": m, "th": s * self.depth_std_mult, "box": box}

    def _segment_mask_single(self, depth_t: torch.Tensor, prof: Dict) -> torch.Tensor:
        """
        Create mask for a single person:
        - threshold whole frame vs mean (fast) but only keep inside box.
        """
        m = torch.tensor(prof["mean"], device=depth_t.device, dtype=depth_t.dtype)
        th = torch.tensor(prof["th"], device=depth_t.device, dtype=depth_t.dtype)
        diff = torch.abs(depth_t - m)
        mask = (diff <= th).to(torch.uint8)

        x1, y1, x2, y2 = prof["box"]
        H, W = depth_t.shape[-2], depth_t.shape[-1]
        # clamp:
        x1c, x2c = max(0, min(x1, W)), max(0, min(x2, W))
        y1c, y2c = max(0, min(y1, H)), max(0, min(y2, H))
        out = torch.zeros_like(mask)
        if x2c > x1c and y2c > y1c:
            out[y1c:y2c, x1c:x2c] = mask[y1c:y2c, x1c:x2c]
        return out

    def build_mask(
        self, depth_np: np.ndarray, boxes: List[List[int]]
    ) -> np.ndarray:
        """
        Combine per-face masks into a single HxW uint8 mask (0/1), then dilate.
        """
        depth_t = torch.from_numpy(depth_np).to(self.device)
        combined = torch.zeros_like(depth_t, dtype=torch.uint8)
        for box in boxes:
            prof = self._calc_depth_profile(depth_np, box)
            if prof is None:
                if self.verbose:
                    print(f"[build_mask] empty depth window for {box}")
                continue
            m = self._segment_mask_single(depth_t, prof)
            combined = torch.logical_or(combined.bool(), m.bool()).to(torch.uint8)

        mask_np = combined.detach().cpu().numpy().astype(np.uint8)
        if self.dilate_kernel > 0:
            mask_np = cv2.dilate(mask_np, self.kernel)
        if self.verbose:
            print(f"[build_mask] mask pixels {mask_np.sum()} / {mask_np.size}")
        return mask_np

        # Pure-numpy depth mask builder (avoids torch transfers)
    def build_mask_numpy(self, depth_np, boxes, kernel, calc_profile_fn):
        H, W = depth_np.shape
        combined = np.zeros((H, W), np.uint8)
        for box in boxes:
            prof = calc_profile_fn(depth_np, box)
            if prof is None:
                continue
            x1, y1, x2, y2 = prof["box"]
            y1, y2 = max(0, y1), min(H, y2)
            x1, x2 = max(0, x1), min(W, x2)
            if x2 <= x1 or y2 <= y1:
                continue
            roi = depth_np[y1:y2, x1:x2]

            # Validity mask: integers (z16) use >0, floats use ~NaN
            if roi.dtype.kind in ("i", "u"):
                valid = roi > 0
                roi_f = roi.astype(np.float32, copy=False)
            else:
                valid = ~np.isnan(roi)
                roi_f = roi  # already float

            m, th = prof["mean"], prof["th"]
            sel = np.zeros_like(roi, dtype=np.uint8)
            if valid.any():
                diff = np.abs(roi_f[valid] - m)
                sel_valid = (diff <= th).astype(np.uint8)
                sel[valid] = sel_valid
            combined[y1:y2, x1:x2] |= sel

        if kernel is not None:
            combined = cv2.dilate(combined, kernel)
        return combined

    # ---------------- Anonymization ----------------

    def anonymize_rgb(self, bgr_np: np.ndarray, mask_np: np.ndarray) -> np.ndarray:
        """
        Pixelate + noise inside mask (GPU). Input/Output are BGR uint8.
        """
        rgb = cv2.cvtColor(bgr_np, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).to(self.device).permute(2, 0, 1).contiguous().float() / 255.0
        C, H, W = t.shape

        block = max(1, self.anon_block if self.anon_block > 0 else W // 16)
        pooled = F.avg_pool2d(t.unsqueeze(0), block, block, ceil_mode=True)
        pix = F.interpolate(pooled, size=(H, W), mode="nearest").squeeze(0)
        noise = (torch.rand_like(pix) * 2 - 1) * (self.anon_noise / 255.0)
        pix = (pix + noise).clamp(0, 1)

        m = torch.from_numpy(mask_np.astype(bool)).to(self.device).to(pix.dtype).unsqueeze(0)
        out = pix * m + t * (1 - m)
        out_np = (out * 255).byte().permute(1, 2, 0).detach().cpu().numpy()
        return cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)

    def fast_pixelate(self, bgr, mask, block=16, noise=20):
        # ---- image dims drive everything ----
        H, W = bgr.shape[:2]

        # ---- sanitize mask to (H, W) boolean ----
        if mask.ndim == 3:
            mask = mask[..., 0]
        if mask.shape != (H, W):
            mask = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        m = (mask > 0)

        # ---- block size (accept int or (bw,bh)) ----
        if isinstance(block, (tuple, list)):
            bw, bh = int(block[0]), int(block[1])
        else:
            bw = bh = int(block)
        bw = max(1, min(W, bw))
        bh = max(1, min(H, bh))

        # target downsample size = ceil(W/bw) × ceil(H/bh)
        small_w = max(1, (W + bw - 1) // bw)
        small_h = max(1, (H + bh - 1) // bh)

        # ---- build mosaic from full frame, then upsample back to H×W ----
        small = cv2.resize(bgr, (small_w, small_h), interpolation=cv2.INTER_AREA)
        pix   = cv2.resize(small, (W, H),       interpolation=cv2.INTER_NEAREST)

        # ---- optional noise ----
        if noise > 0:
            n = np.random.randint(-noise, noise + 1, bgr.shape, dtype=np.int16)
            pix = np.clip(pix.astype(np.int16) + n, 0, 255).astype(np.uint8)

        # ---- copy only inside mask ----
        out = bgr.copy()
        out[m] = pix[m]
        return out  # ALWAYS (H, W, 3)

    def anonymize_depth(self, depth_np: np.ndarray, mask_np: np.ndarray, noise_sigma: float = 10.0) -> np.ndarray:
        """
        Add Gaussian noise ONLY where mask==1. Returns float32 array (same shape).
        """
        d = depth_np.copy()
        m = mask_np.astype(bool)
        if m.any():
            d[m] += np.random.normal(0.0, noise_sigma, size=m.sum())
        return d


    def build_mask_from_boxes(
        self,
        boxes: List[List[int]],
        image_shape: Tuple[int, int],
        pad_ratio: float = 0.25,
        oval: bool = True,
    ) -> np.ndarray:
        """
        Make an HxW uint8 mask (0/1) from face boxes only (no depth).
        - pad_ratio expands each box by % of its size.
        - oval=True draws ellipse; False draws rectangle.
        """
        H, W = image_shape[:2]
        mask = np.zeros((H, W), np.uint8)

        for (x1, y1, x2, y2) in boxes:
            w = x2 - x1
            h = y2 - y1
            if w <= 0 or h <= 0:
                continue
            # pad & clamp
            px = int(w * pad_ratio)
            py = int(h * pad_ratio)
            xa = max(0, x1 - px)
            ya = max(0, y1 - py)
            xb = min(W, x2 + px)
            yb = min(H, y2 + py)

            if oval:
                # ellipse centered on the (padded) box
                cx = (xa + xb) // 2
                cy = (ya + yb) // 2
                ax = max(1, (xb - xa) // 2)
                ay = max(1, (yb - ya) // 2)
                cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 1, thickness=-1)
            else:
                cv2.rectangle(mask, (xa, ya), (xb, yb), 1, thickness=-1)

        # optional dilation like your depth flow
        if self.dilate_kernel > 0:
            mask = cv2.dilate(mask, self.kernel)
        return mask

    # ---------------- High-level API ----------------

    def process_arrays(
        self,
        bgr_np: np.ndarray,
        depth_np: Optional[np.ndarray] = None,      # now optional
        out_rgb: Optional[str] = None,
        out_depth: Optional[str] = None,
        save: bool = False,
        depth_view_normalize: bool = True,
        use_depth_for_mask: bool = True,            # NEW: choose mask source
        pad_ratio: float = 0.25,                    # NEW: for bbox mask
        oval_boxes: bool = True,                    # NEW: ellipse vs rectangle
    ) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, List[List[int]], Dict[str, float]]:

        timings: Dict[str, float] = {}
        t0 = time.perf_counter()

        # 1) detect
        boxes = self.detect_faces(bgr_np)
        timings["detect_s"] = time.perf_counter() - t0

        # 2) mask
        t1 = time.perf_counter()
        if use_depth_for_mask and depth_np is not None:
            # existing depth-based masking
            mask_np = self.build_mask_numpy(depth_np, boxes, self.kernel, self._calc_depth_profile)
        else:
            # NEW: bbox-only masking (no depth needed)
            mask_np = self.build_mask_from_boxes(boxes, bgr_np.shape, pad_ratio=pad_ratio, oval=oval_boxes)
        timings["mask_s"] = time.perf_counter() - t1

        # 3) anonymize RGB (fast path)
        t2 = time.perf_counter()
        rgb_anon = self.fast_pixelate(bgr_np, mask_np, block=self.anon_block, noise=self.anon_noise)

        depth_anon = None
        if self.enable_depth_anon and depth_np is not None:
            depth_anon = self.anonymize_depth(depth_np, mask_np, noise_sigma=10.0)
        timings["anonymize_s"] = time.perf_counter() - t2

        # 4) save (optional)
        if save and out_rgb:
            cv2.imwrite(out_rgb, rgb_anon)
        if save and out_depth and self.enable_depth_anon and depth_anon is not None:
            if depth_view_normalize:
                view = cv2.normalize(depth_anon, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                cv2.imwrite(out_depth, view)
            else:
                raw16 = np.clip(depth_anon, 0, 65535).astype(np.uint16)
                cv2.imwrite(out_depth, raw16)

        timings["total_s"] = time.perf_counter() - t0
        return rgb_anon, depth_anon, mask_np, boxes, timings

    def process_paths(
        self,
        img_path: str,
        depth_path: str,
        out_rgb: Optional[str] = None,
        out_depth: Optional[str] = None,
        save: bool = False,
        depth_view_normalize: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[List[int]], Dict[str, float]]:
        """Load files and call process_arrays()."""
        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f"Could not read depth: {depth_path}")
        depth = depth.astype(np.float32)

        return self.process_arrays(
            bgr, depth, out_rgb=out_rgb, out_depth=out_depth, save=save,
            depth_view_normalize=depth_view_normalize,
        )