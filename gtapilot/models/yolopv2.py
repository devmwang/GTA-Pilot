# Attribution: YOLOPv2 (CAIC-AD)
# This module adapts concepts/components from the YOLOPv2 project for panoptic driving perception.
# Source: https://github.com/CAIC-AD/YOLOPv2

from dataclasses import dataclass
import os
from typing import Any, List, Optional, Tuple, Sequence, cast
from pathlib import Path
from contextlib import nullcontext

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from gtapilot.ipc.vision_ipc import VisionIPCSubscriber
from gtapilot.ipc.messaging import PubMaster, Message

# GPU NMS
try:
    from torchvision.ops import nms as tv_nms
except Exception:
    tv_nms = None


if torch.cuda.is_available():
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass


def _make_divisible(value: int, divisor: int) -> int:
    r = value % divisor
    return value if r == 0 else value + (divisor - r)


def _choose_net_hw(
    width: int, height: int, max_long_side: int = 1280, stride: int = 32
) -> Tuple[int, int]:
    """
    Choose a network input (H, W) close to max_long_side while respecting stride multiples.
    Keeps original aspect ratio. For 1920x1080, this returns approximately 736x1280.
    """
    if width <= 0 or height <= 0:
        return 736, 1280  # sensible default for 16:9

    if width >= height:
        target_w = max_long_side
        target_h = int(round(height * (target_w / float(width))))
    else:
        target_h = max_long_side
        target_w = int(round(width * (target_h / float(height))))

    target_h = _make_divisible(target_h, stride)
    target_w = _make_divisible(target_w, stride)
    return int(target_h), int(target_w)


def _compute_letterbox_params(
    width: int, height: int, input_hw: Tuple[int, int]
) -> Tuple[float, int, int, int, int, int, int, int, int]:
    """
    Compute letterbox scale and integer paddings.
    Returns (scale, new_w, new_h, top, bottom, left, right, W0, H0)
    """
    H_in, W_in = int(input_hw[0]), int(input_hw[1])
    H0, W0 = int(height), int(width)
    if H0 <= 0 or W0 <= 0:
        return 1.0, W_in, H_in, 0, 0, 0, 0, W0, H0
    scale = min(W_in / float(max(W0, 1)), H_in / float(max(H0, 1)))
    new_w = int(round(W0 * scale))
    new_h = int(round(H0 * scale))
    padw = (W_in - new_w) / 2.0
    padh = (H_in - new_h) / 2.0
    top = int(round(padh))
    bottom = int(round(H_in - new_h - top))
    left = int(round(padw))
    right = int(round(W_in - new_w - left))
    return scale, new_w, new_h, top, bottom, left, right, W0, H0


def _mask_from_logits(
    logits: torch.Tensor,
    input_hw: Tuple[int, int],
    orig_wh: Tuple[int, int],
    pad_tblr: Tuple[int, int, int, int],
    positive_class: int = 1,
    thresh: float = 0.5,
    close_k: int = 3,
    dilate_k: int = 1,
) -> np.ndarray:
    """
    Convert raw segmentation logits to original-resolution binary mask.
    Performs upsample to input canvas, crop (remove padding), and final resize on GPU.
    """
    t = logits.detach()
    # Ensure [C,h,w]
    if t.dim() == 4:
        t = t[0]
    if t.dim() == 2:
        pos = t
    elif t.dim() == 3:
        C = t.shape[0]
        idx = min(positive_class, max(0, C - 1))
        pos = t[idx]
    else:
        W0, H0 = int(orig_wh[0]), int(orig_wh[1])
        return np.zeros((H0, W0), dtype=np.uint8)

    # If logits, apply sigmoid; avoid host sync by using GPU scalar conditions
    pos_min, pos_max = pos.min(), pos.max()
    pos = torch.where((pos_min < 0) | (pos_max > 1), pos.sigmoid(), pos)

    # Threshold → {0,1} float for interpolation
    mask_small = (pos >= float(thresh)).to(dtype=torch.float32)
    # [h,w] → [1,1,h,w]
    mask_small = mask_small.unsqueeze(0).unsqueeze(0)

    H_in_g, W_in_g = int(input_hw[0]), int(input_hw[1])
    # Upsample to input canvas
    mask_canvas = F.interpolate(mask_small, size=(H_in_g, W_in_g), mode="nearest")

    # Crop padding
    top, bottom, left, right = pad_tblr
    y1 = max(0, min(int(top), H_in_g))
    y2 = max(0, min(int(H_in_g - bottom), H_in_g))
    x1 = max(0, min(int(left), W_in_g))
    x2 = max(0, min(int(W_in_g - right), W_in_g))
    if x2 <= x1 or y2 <= y1:
        cropped = mask_canvas.new_zeros((1, 1, 1, 1))
    else:
        cropped = mask_canvas[:, :, y1:y2, x1:x2]

    # Resize to original resolution
    W0, H0 = int(orig_wh[0]), int(orig_wh[1])
    resized = F.interpolate(cropped, size=(H0, W0), mode="nearest")
    # To uint8 0/255 on CPU
    out = (resized.squeeze().clamp(0, 1) * 255.0).to(dtype=torch.uint8)
    mask = out.detach().cpu().numpy()

    # morphology for cleanups (on CPU to match prior behavior)
    if close_k and close_k > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    if dilate_k and dilate_k > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k))
        mask = cv2.dilate(mask, k, iterations=1)

    return mask


@dataclass
class DetBox:
    xyxy: Tuple[int, int, int, int]
    conf: float
    cls_id: int


class YOLOPv2:
    """
    Adapter for YOLOPv2 panoptic model.

    Returns a tuple: ([pred_scales, anchor_grid], drivable_logits, lane_logits)
    where pred_scales is a list of 3 tensors [B, A, Sy, Sx, 5+nc],
    anchor_grid is a list of 3 tensors [1, A, Sy, Sx, 2] (anchors*stride in px).
    """

    def __init__(
        self,
        conf_thres: float = 0.25,
        iou_thres: float = 0.50,
        lane_thresh: float = 0.5,
        driveable_area_thresh: float = 0.5,
        px_import_name: str = "yolopv2",
        px_entry_attr: str = "load_model",
        px_weights: Optional[str] = None,
        res_height=864,  # 32 * 3 * 9
        res_width=1536,  # 32 * 3 * 16
    ):
        # Always prefer CUDA if available, regardless of requested device
        prefer_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(prefer_device)
        self.conf_thres = float(conf_thres)
        self.iou_thres = float(iou_thres)
        self.lane_thresh = float(lane_thresh)
        self.driveable_area_thresh = float(driveable_area_thresh)
        self.input_hw = (res_height, res_width)

        self.model: Any
        self._is_jit: bool = False
        self._did_debug: bool = False  # one-time shape probe
        self._try_load(px_import_name, px_entry_attr, px_weights)

        try:
            self.model = self.model.to(self.device)
        except Exception as e:
            print(f"[YOLOPv2] Warning: model.to({self.device}) failed: {e}")

        try:
            self.model.eval()
        except Exception:
            pass

    def _try_load(
        self, import_name: str, entry_attr: str, weights: Optional[str]
    ) -> None:
        if weights and os.path.isfile(weights) and weights.lower().endswith(".pt"):
            try:
                self.model = torch.jit.load(weights, map_location=str(self.device))
                self._is_jit = True
                return
            except Exception as e:
                raise RuntimeError(
                    f"Failed to torch.jit.load() weights: {weights}"
                ) from e

        try:
            self.model = torch.hub.load(
                import_name, "yolopv2", pretrained=True, trust_repo=True
            )
            return
        except Exception:
            pass

        try:
            mod = __import__(import_name, fromlist=["*"])
        except Exception as e:
            raise RuntimeError(
                f"Could not import YOLOPv2 module '{import_name}'. "
                "Pass --px-weights=<path/to/yolopv2.pt> (recommended), or "
                "use --px-import-name=<your.module> and ensure the repo is on PYTHONPATH via --px-repo-dir."
            ) from e

        if hasattr(mod, entry_attr):
            loader = getattr(mod, entry_attr)
            try:
                self.model = loader(weights, device=str(self.device))
            except Exception:
                self.model = loader(weights)
            return

        for name in ("build_model", "get_model", "Model", "get_net"):
            if hasattr(mod, name):
                ctor = getattr(mod, name)
                try:
                    self.model = ctor(weights)
                except Exception:
                    self.model = ctor()
                return

        if callable(mod):
            self.model = mod()
            return

        raise RuntimeError(
            f"Could not find a model entrypoint in module '{import_name}'. "
            f"Looked for '{entry_attr}', 'build_model', 'get_model', 'get_net', 'Model'."
        )

    # ---- utils
    @staticmethod
    def _flatten_det(pred: Any) -> torch.Tensor:
        import numpy as _np
        import torch as _t

        def _to_tensor(x: Any) -> _t.Tensor:
            if isinstance(x, _t.Tensor):
                return x
            if isinstance(x, _np.ndarray):
                return _t.from_numpy(x)
            raise RuntimeError(f"Unsupported detection element type: {type(x)}")

        def _to_BNC(t: _t.Tensor) -> _t.Tensor:
            if t.dim() == 3:
                # [B, N, 5+nc]
                return t
            if t.dim() == 5:
                # [B, A, Sy, Sx, 5+nc] → [B, N, 5+nc]
                B, A, Sy, Sx, no = t.shape
                return t.view(B, A * Sy * Sx, no)
            raise RuntimeError(
                f"Unsupported detection tensor shape {tuple(t.shape)}; expected [B,N,5+nc] or [B,A,Sy,Sx,5+nc]."
            )

        if isinstance(pred, dict):
            for k in ("det", "pred", "outputs", "boxes", "head", "yolo_head"):
                if k in pred:
                    return YOLOPv2._flatten_det(pred[k])
            raise RuntimeError(
                f"Unsupported detection dict keys: {list(pred.keys())}; expected one of det/pred/outputs/boxes/head/yolo_head."
            )

        if isinstance(pred, (_t.Tensor, _np.ndarray)):
            return _to_BNC(_to_tensor(pred))

        if isinstance(pred, (list, tuple)):
            # Concatenate well-formed tensors
            parts: List[_t.Tensor] = []
            for p in pred:
                if isinstance(p, (_t.Tensor, _np.ndarray)):
                    parts.append(_to_BNC(_to_tensor(p)))
                else:
                    raise RuntimeError(
                        "Unsupported nested container in detection outputs; expected list/tuple of tensors."
                    )
            if not parts:
                raise RuntimeError("Empty detection outputs list.")
            cset = {t.shape[-1] for t in parts}
            if len(cset) != 1:
                raise RuntimeError(
                    f"Detection parts disagree on channel size: {sorted(cset)}"
                )
            return _t.cat(parts, dim=1)

        raise RuntimeError(f"Unsupported detection output type: {type(pred)}")

    def _decode_yolov5_ts(
        self,
        pred_scales: Sequence[torch.Tensor],
        anchor_grid: Sequence[torch.Tensor],
        input_hw: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Decode YOLOPv2/YOLOv5 TorchScript outputs to [B, N, 5+nc] in pixels on the
        letterboxed canvas. Supports two per-scale formats:

        A) Raw head maps:   p.shape == [B, 255, Ny, Nx]  (na=3, no=5+nc, 255=na*no)
        B) Already reshaped p.shape == [B, A, Ny, Nx, 5+nc]

        anchor_grid[i].shape == [1, A, 1, 1, 2]  (anchors×stride in px), broadcasted.
        """
        H_in, W_in = input_hw
        outs: List[torch.Tensor] = []

        for i, (p, ag) in enumerate(zip(pred_scales, anchor_grid)):
            if p.dim() == 4:
                # Format A: [B, 255, Ny, Nx] → [B, A, Ny, Nx, no]
                B, C, Ny, Nx = p.shape
                A = ag.shape[1]  # usually 3
                assert C % A == 0, f"det[{i}] channels {C} not divisible by anchors {A}"
                no = C // A  # 5+nc
                nc = max(0, no - 5)
                # reshape and move "no" to last
                p = p.view(B, A, no, Ny, Nx).permute(0, 1, 3, 4, 2).contiguous()
            elif p.dim() == 5:
                # Format B: already [B, A, Ny, Nx, 5+nc]
                B, A, Ny, Nx, no = p.shape
                nc = max(0, no - 5)
            else:
                raise RuntimeError(
                    f"Unexpected det tensor shape {tuple(p.shape)} at scale {i}"
                )

            device = p.device
            # YOLOv5 export decode
            ps = p.sigmoid()

            # build grid in cell coords (cache per (Ny,Nx,device))
            cache_key = (
                Ny,
                Nx,
                device.type,
                device.index if hasattr(device, "index") else None,
            )
            grid = (
                getattr(self, "_grid_cache", {}).get(cache_key)
                if hasattr(self, "_grid_cache")
                else None
            )
            if grid is None or grid.device != device:
                yv, xv = torch.meshgrid(
                    torch.arange(Ny, device=device),
                    torch.arange(Nx, device=device),
                    indexing="ij",
                )
                grid = torch.stack((xv, yv), dim=-1).view(1, 1, Ny, Nx, 2).float()
                if not hasattr(self, "_grid_cache"):
                    self._grid_cache = {}
                self._grid_cache[cache_key] = grid

            # anisotropic strides from letterbox size
            sx = W_in / float(Nx)
            sy = H_in / float(Ny)

            # centers: (sigmoid*2 - 0.5 + grid) * stride
            xy = ps[..., 0:2] * 2.0 - 0.5
            xy = xy + grid
            xy[..., 0] = xy[..., 0] * sx
            xy[..., 1] = xy[..., 1] * sy

            # sizes: (sigmoid*2)**2 * anchor_grid  (ag already pixels; broadcasts to [1,A,Ny,Nx,2])
            wh = (ps[..., 2:4] * 2.0) ** 2 * ag

            obj = ps[..., 4:5]
            cls = ps[..., 5:]  # may be empty if single-class

            out = torch.cat((xy, wh, obj, cls), dim=-1)  # [B, A, Ny, Nx, 5+nc]
            outs.append(out.view(B, -1, no))  # [B, A*Ny*Nx, 5+nc]

            # One-time debug — stride and channels sanity
            if not self._did_debug:
                print(
                    f"[YOLOPv2:decode] scale {i}: A={A} Ny={Ny} Nx={Nx} no={no} nc={nc} sx={sx:.1f} sy={sy:.1f}"
                )

        self._did_debug = True
        return torch.cat(outs, dim=1)  # [B, N, 5+nc]

    def _debug_dump_shapes(self, det_out: Any, da_logits: Any, ll_logits: Any) -> None:
        if self._did_debug:
            return
        doit = os.environ.get("FUSION_DEBUG", "0") == "1"
        if not doit:
            doit = True
        if not doit:
            return

        def _shape(x):
            if isinstance(x, torch.Tensor):
                return tuple(x.shape)
            if isinstance(x, (list, tuple)):
                return [_shape(xx) for xx in x]
            if isinstance(x, dict):
                return {k: _shape(v) for k, v in x.items()}
            return type(x).__name__

        self._did_debug = True

    @torch.inference_mode()
    def infer(self, rgb: np.ndarray) -> Tuple[List[DetBox], np.ndarray, np.ndarray]:
        H0, W0 = rgb.shape[:2]
        if self.input_hw is None:
            self.input_hw = _choose_net_hw(W0, H0)

        scale, new_w, new_h, top, bottom, left, right, W0, H0 = (
            _compute_letterbox_params(W0, H0, self.input_hw)
        )

        x_t = torch.tensor(rgb, device=self.device)
        x_t = (
            x_t.permute(2, 0, 1)
            .contiguous()
            .unsqueeze(0)
            .to(dtype=torch.float32)
            .div(255.0)
        )
        # Resize to (new_h, new_w)
        if new_h != H0 or new_w != W0:
            x_t = F.interpolate(
                x_t, size=(new_h, new_w), mode="bilinear", align_corners=False
            )
        # Pad to (H_in, W_in)
        if top or bottom or left or right:
            x_t = F.pad(x_t, (left, right, top, bottom), value=114.0 / 255.0)
        padw, padh = (left + right) / 2.0, (top + bottom) / 2.0
        pad_tblr = (top, bottom, left, right)
        gain = float(scale)

        # Forward (optional mixed precision on CUDA)
        use_autocast = self.device.type == "cuda"
        with (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if use_autocast
            else nullcontext()
        ):
            out = self.model(x_t)

        # ---- normalize to (det_out, da_logits, ll_logits) WITHOUT unwrapping the tuple
        if isinstance(out, (list, tuple)) and len(out) >= 3:
            det_out, da_logits, ll_logits = out[0], out[1], out[2]
            # IMPORTANT: do NOT do "pred_scales, _ = det_out; det_out = pred_scales"
            # We need anchor_grid later; keep the 2-tuple intact.
        elif isinstance(out, dict):
            det_out = out.get("det") or out.get("pred") or out.get("det_out")
            da_logits = out.get("da") or out.get("drivable") or out.get("da_logits")
            ll_logits = out.get("ll") or out.get("lane") or out.get("ll_logits")
            if det_out is None or da_logits is None or ll_logits is None:
                raise RuntimeError("YOLOPv2 forward dict missing det/da/ll.")
        else:
            raise RuntimeError("YOLOPv2 forward returned an unexpected type/shape.")

        # one-time shapes
        self._debug_dump_shapes(det_out, da_logits, ll_logits)

        # --- SEGMENTATION (GPU upsample/crop path)
        da_mask = _mask_from_logits(
            da_logits,
            self.input_hw,
            (W0, H0),
            pad_tblr,
            positive_class=1,
            thresh=self.driveable_area_thresh,
            close_k=3,
            dilate_k=1,
        )
        ll_mask = _mask_from_logits(
            ll_logits,
            self.input_hw,
            (W0, H0),
            pad_tblr,
            positive_class=1,
            thresh=self.lane_thresh,
            close_k=5,
            dilate_k=2,
        )

        # --- DETECTION
        boxes: List[DetBox] = []
        det_pred_bnc: Optional[torch.Tensor] = None

        # Preferred TorchScript decode path: ([pred_scales], [anchor_grid])
        if self._is_jit and isinstance(det_out, (list, tuple)) and len(det_out) == 2:
            pred_scales, anchor_grid = det_out
            if isinstance(pred_scales, (list, tuple)) and isinstance(
                anchor_grid, (list, tuple)
            ):
                det_pred_bnc = self._decode_yolov5_ts(
                    cast(Sequence[torch.Tensor], pred_scales),
                    cast(Sequence[torch.Tensor], anchor_grid),
                    self.input_hw,
                )

        # Fallbacks
        if det_pred_bnc is None:
            # Case 1: already NMSed [B,M,6] (x1,y1,x2,y2,conf,cls)
            if (
                isinstance(det_out, torch.Tensor)
                and det_out.dim() == 3
                and det_out.shape[-1] == 6
            ):
                p = det_out[0].clone()
                coords = p[:, :4]
                coords[:, [0, 2]] -= padw
                coords[:, [1, 3]] -= padh
                coords[:, :4] /= gain
                coords[:, 0].clamp_(0, W0 - 1)
                coords[:, 1].clamp_(0, H0 - 1)
                coords[:, 2].clamp_(0, W0 - 1)
                coords[:, 3].clamp_(0, H0 - 1)
                p[:, :4] = coords
                p_np = p.cpu().numpy()
                for x1, y1, x2, y2, cf, cls_id in p_np:
                    if cf >= self.conf_thres and x2 > x1 and y2 > y1:
                        boxes.append(
                            DetBox(
                                (int(x1), int(y1), int(x2), int(y2)),
                                float(cf),
                                int(cls_id),
                            )
                        )
                return boxes, da_mask, ll_mask

            # Case 2: raw-style [B, N, C]
            det_pred_bnc = self._flatten_det(det_out).to(self.device)

        dp_t = det_pred_bnc[0].detach()
        C = dp_t.shape[1]
        # accept 5 (single-class) or 5+nc
        if C < 5:
            return boxes, da_mask, ll_mask

        cxcy = dp_t[:, 0:2]
        wh = dp_t[:, 2:4]
        obj = dp_t[:, 4:5]
        cls_logits = dp_t[:, 5:]  # may be empty

        # normalize obj/cls if logits without host syncs
        obj_min, obj_max = obj.min(), obj.max()
        obj = torch.where((obj_min < 0) | (obj_max > 1), obj.sigmoid(), obj)

        if cls_logits.numel() > 0:
            cls_min, cls_max = cls_logits.min(), cls_logits.max()
            cls_prob = torch.where(
                (cls_min < 0) | (cls_max > 1), cls_logits.sigmoid(), cls_logits
            )
            mult = obj * cls_prob
            scores = mult.max(dim=1).values
            cls_ids = mult.argmax(dim=1)
        else:
            scores = obj.reshape(-1)
            cls_ids = torch.zeros_like(scores, dtype=torch.long)

        # letterboxed canvas → xyxy (torch)
        x1y1 = cxcy - wh / 2.0
        x2y2 = cxcy + wh / 2.0
        xyxy = torch.cat([x1y1, x2y2], dim=1)

        keep = scores >= self.conf_thres
        if not torch.any(keep):
            return boxes, da_mask, ll_mask
        xyxy = xyxy[keep]
        scores_k = scores[keep]
        cls_ids_k = cls_ids[keep]

        # un-letterbox to original frame
        xyxy[:, [0, 2]] -= float(padw)
        xyxy[:, [1, 3]] -= float(padh)
        xyxy[:, :4] /= max(float(gain), 1e-9)
        xyxy[:, 0].clamp_(0, W0 - 1)
        xyxy[:, 1].clamp_(0, H0 - 1)
        xyxy[:, 2].clamp_(0, W0 - 1)
        xyxy[:, 3].clamp_(0, H0 - 1)

        # per-class NMS (prefer torchvision on torch tensors)
        kept: List[DetBox] = []
        if tv_nms is not None:
            for c in torch.unique(cls_ids_k).tolist():
                idx = torch.where(cls_ids_k == int(c))[0]
                if idx.numel() == 0:
                    continue
                keep_idx = tv_nms(xyxy[idx], scores_k[idx], float(self.iou_thres))
                sel = idx[keep_idx]
                for j in sel.tolist():
                    x1, y1, x2, y2 = xyxy[j].round().to(torch.int64).tolist()
                    if x2 > x1 and y2 > y1:
                        kept.append(
                            DetBox(
                                (int(x1), int(y1), int(x2), int(y2)),
                                float(scores_k[j].item()),
                                int(c),
                            )
                        )
        else:
            # Fallback to NumPy CPU NMS
            xyxy_np = xyxy.detach().cpu().numpy()
            scores_np = scores_k.detach().cpu().numpy()
            cls_np = cls_ids_k.detach().cpu().numpy()
            for c in np.unique(cls_np):
                idx = np.where(cls_np == c)[0]
                if idx.size == 0:
                    continue
                keep_idx = _nms_xyxy(xyxy_np[idx], scores_np[idx], self.iou_thres)
                for j in keep_idx:
                    x1, y1, x2, y2 = xyxy_np[idx][j].astype(int)
                    if x2 > x1 and y2 > y1:
                        kept.append(
                            DetBox(
                                (int(x1), int(y1), int(x2), int(y2)),
                                float(scores_np[idx][j]),
                                int(c),
                            )
                        )

        boxes.extend(kept)

        return boxes, da_mask, ll_mask


# IOU util
def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:4], b[None, :, 2:4])
    wh = np.clip(br - tl, 0, None)
    inter = wh[:, :, 0] * wh[:, :, 1]

    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]) + 1e-9
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) + 1e-9

    return inter / (area_a[:, None] + area_b[None, :] - inter)


# NMS util
def _nms_xyxy(
    boxes_xyxy: np.ndarray, scores: np.ndarray, iou_thres: float
) -> List[int]:
    order = scores.argsort()[::-1]
    keep: List[int] = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        if order.size == 1:
            break

        ious = _iou_xyxy(boxes_xyxy[i : i + 1], boxes_xyxy[order[1:]])[0]
        order = order[1:][ious <= iou_thres]

    return keep


def _draw_overlay(
    frame_bgr: np.ndarray,
    boxes: List[DetBox],
    da_mask: Optional[np.ndarray],
    ll_mask: Optional[np.ndarray],
    alpha: float = 0.3,
) -> np.ndarray:
    out = frame_bgr.copy()
    if da_mask is not None and da_mask.size:
        da_col = np.zeros_like(out)
        da_col[:, :, 1] = da_mask  # green
        out = cv2.addWeighted(out, 1.0, da_col, alpha, 0)
    if ll_mask is not None and ll_mask.size:
        ll_col = np.zeros_like(out)
        ll_col[:, :, 2] = ll_mask  # red
        out = cv2.addWeighted(out, 1.0, ll_col, alpha, 0)
    for b in boxes:
        x1, y1, x2, y2 = b.xyxy
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(
            out,
            f"{b.cls_id}:{b.conf:.2f}",
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return out


def main(**kwargs):
    """
    Subscribe to the Vision IPC, run YOLOPv2 per frame, and optionally visualize overlays.
    Model expects RGB HxWx3 frames.
    """
    # Always load weights from repo root: weights/YOLOPv2.pt
    root_dir = Path(__file__).resolve().parents[2]
    weights_file = root_dir / "weights" / "YOLOPv2.pt"
    px_weights: Optional[str] = str(weights_file) if weights_file.is_file() else None
    px_import_name: str = kwargs.get("px_import_name", "yolopv2")
    px_entry_attr: str = kwargs.get("px_entry_attr", "load_model")
    conf_thres: float = float(kwargs.get("conf_thres", 0.25))
    iou_thres: float = float(kwargs.get("iou_thres", 0.50))

    model = YOLOPv2(
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        px_import_name=px_import_name,
        px_entry_attr=px_entry_attr,
        px_weights=px_weights,
    )

    subscriber = VisionIPCSubscriber(latestOnly=True)
    pub = PubMaster()

    try:
        while True:
            rgb = subscriber.receive_frame(blocking=True)
            if rgb is None:
                continue

            boxes, da_mask, ll_mask = model.infer(rgb)

            # Publish results via messaging IPC
            try:
                pub.publish("lane_lines", Message("lane_lines", {"mask": ll_mask}))

                pub.publish(
                    "driveable_area",
                    Message("driveable_area", {"mask": da_mask}),
                )

                boxes_payload = [
                    {
                        "x1": int(b.xyxy[0]),
                        "y1": int(b.xyxy[1]),
                        "x2": int(b.xyxy[2]),
                        "y2": int(b.xyxy[3]),
                        "conf": float(b.conf),
                        "cls_id": int(b.cls_id),
                    }
                    for b in boxes
                ]
                pub.publish(
                    "vehicle_bounding_boxes",
                    Message("vehicle_bounding_boxes", {"boxes": boxes_payload}),
                )
            except Exception:
                pass

    except KeyboardInterrupt:
        pass
    finally:
        subscriber.close()
        try:
            pub.close()
        except Exception:
            pass
