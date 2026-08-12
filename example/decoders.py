"""YOLOv3 헤드 디코더 두 종류 (numpy 구현).

원본과 Nota 프루닝 모델은 **박스 크기 디코딩 공식이 다릅니다.**

    표준 YOLOv3   wh = exp(t) * anchor
    Nota 프루닝    wh = 2 * (tanh(t/2 - 0.549306) + 1) * anchor

`tanh` 쪽은 폭발하지 않아 저정밀 양자화에 유리합니다.
프루닝 모델의 헤드를 표준 디코더에 먹이면 박스가 전부 틀어지므로 반드시 구분해야 합니다.

출처: demo_pruning/detection/yolo_image_pruning.py 의
      `yolo_fastest_head_decode` + `nms` 를 numpy 로 옮긴 것 (torch/torchvision 불필요).
"""
from __future__ import annotations

import numpy as np

INPUT_SIZE = 608

# 세 스테이지 앵커 — 표준 YOLOv3 COCO 앵커와 동일한 값입니다.
# 순서는 헤드 해상도 [76, 38, 19] 에 대응 (stride 8 / 16 / 32).
ANCHORS = (
    np.array([[10.0, 13.0], [16.0, 30.0], [33.0, 23.0]], np.float32),
    np.array([[30.0, 61.0], [62.0, 45.0], [59.0, 119.0]], np.float32),
    np.array([[116.0, 90.0], [156.0, 198.0], [373.0, 326.0]], np.float32),
)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def _wh_nota(t: np.ndarray) -> np.ndarray:
    """Nota 프루닝: 2·(tanh(t/2 − 0.549306) + 1) ≡ 4·σ(t − ln3). 상한 4.0."""
    return 2.0 * (np.tanh(t / 2.0 - 0.549306) + 1.0)


def _wh_std(t: np.ndarray) -> np.ndarray:
    """표준 YOLOv3: exp(t). 상한 없음 — 클립은 float32 오버플로 방지용."""
    return np.exp(np.clip(t, -80, 80))


def _decode(heads: list[np.ndarray], wh_fn, score_thresh: float,
            input_size: int) -> np.ndarray:
    out = []
    for idx, p in enumerate(heads):
        _, _, H, W = p.shape
        stride = input_size // H
        na = ANCHORS[idx].shape[0]                       # 3
        p = p.reshape(1, na, -1, H, W).transpose(0, 1, 3, 4, 2)   # (1,3,H,W,85)

        yv, xv = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        grid = np.stack((xv, yv), -1).astype(np.float32)          # (H,W,2)

        xy = (_sigmoid(p[..., 0:2]) + grid) * stride
        wh = wh_fn(p[..., 2:4]) * ANCHORS[idx].reshape(1, na, 1, 1, 2)
        rest = _sigmoid(p[..., 4:])                                # obj + 80 cls

        out.append(np.concatenate([xy, wh, rest], -1).reshape(-1, 85))

    pred = np.concatenate(out, 0)
    obj = pred[:, 4]
    cls_conf = pred[:, 5:].max(1)
    cls_id = pred[:, 5:].argmax(1)
    score = obj * cls_conf

    keep = score >= score_thresh
    if not keep.any():
        return np.zeros((0, 6), np.float32)
    cx, cy, w, h = pred[keep, 0], pred[keep, 1], pred[keep, 2], pred[keep, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    return np.concatenate([boxes, score[keep, None], cls_id[keep, None].astype(np.float32)], 1)


def decode_nota(heads: list[np.ndarray], score_thresh: float = 0.001,
                input_size: int = INPUT_SIZE) -> np.ndarray:
    """Nota 프루닝 모델용 디코더 (wh = 4·σ(t − ln3) · anchor).

    heads: (1,255,H,W) 3개, **해상도 큰 순서** [76, 38, 19]
    반환 : (N, 6) = x1, y1, x2, y2, score, class_id   (input_size 좌표계)
    """
    return _decode(heads, _wh_nota, score_thresh, input_size)


def decode_std(heads: list[np.ndarray], score_thresh: float = 0.001,
               input_size: int = INPUT_SIZE) -> np.ndarray:
    """표준 YOLOv3 디코더 (wh = exp(t) · anchor).

    후처리가 그래프에 내장되지 않은 원본 계열 모델용입니다. 배포본처럼 후처리가
    별도 `.back` ONNX 로 분리돼 있으면 그쪽을 쓰는 편이 정확합니다.
    인자·반환은 `decode_nota` 와 같습니다.
    """
    return _decode(heads, _wh_std, score_thresh, input_size)


def nms_per_class(det: np.ndarray, iou_thresh: float = 0.45,
                  max_det: int = 100) -> np.ndarray:
    """클래스별 NMS (torchvision.ops.batched_nms 와 동일한 동작)."""
    if len(det) == 0:
        return det
    keep_all = []
    for c in np.unique(det[:, 5]):
        idx = np.where(det[:, 5] == c)[0]
        b, s = det[idx, :4], det[idx, 4]
        order = s.argsort()[::-1]
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        while order.size:
            i = order[0]
            keep_all.append(idx[i])
            if order.size == 1:
                break
            xx1 = np.maximum(b[i, 0], b[order[1:], 0])
            yy1 = np.maximum(b[i, 1], b[order[1:], 1])
            xx2 = np.minimum(b[i, 2], b[order[1:], 2])
            yy2 = np.minimum(b[i, 3], b[order[1:], 3])
            inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
            iou = inter / (area[i] + area[order[1:]] - inter + 1e-9)
            order = order[1:][iou <= iou_thresh]
    keep_all = np.array(keep_all, int)
    out = det[keep_all]
    return out[out[:, 4].argsort()[::-1]][:max_det]
