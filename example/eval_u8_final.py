#!/usr/bin/env python3
"""동일 조건 uint8 세 모델의 COCO mAP — 확정 표용 (모델당 프로세스 1개).

`model/u8/*_u8_maD20.onnx` 는 셋 다

    같은 fp32 계보(Nota 학습 YOLOv3) → 같은 캘리브 데이터(배포 20장)
    → 같은 방식(MovingAverage, QOperator, per-tensor uint8)

로 만들었으므로, 이 셋의 비교만이 **프루닝 손실**을 정직하게 측정합니다.
배포 uint8 파일들은 원본만 계보가 달라 쓸 수 없습니다
([docs/model-lineage-warning.md]).

    python eval_u8_final.py original            # 전체 5,000장
    python eval_u8_final.py pruned40 --limit 500
"""
from __future__ import annotations

import argparse
import contextlib
import io as _io
import json
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from decoders import decode_nota, nms_per_class
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

HOME = Path.home()
COCO_IMG = HOME / "datasets/coco/images/val2017"
COCO_ANN = HOME / "datasets/coco/annotations/instances_val2017.json"
U8 = HOME / "workspace_docs/model/u8"
BACK = HOME / ("demo_1127/yolov3/yolov3_d53_mstrain-608_273e_coco_optim_opset12"
               ".neubla_u8_lwq_movingaverage.onnx.back")
YV = HOME / "yolo_val"
SIZE = 608

# .back 그래프마다 입력 이름이 다릅니다 — 해상도(76/38/19) → 입력 이름
BACK_IN = {
    str(BACK): {19: "onnx::Transpose_684", 38: "onnx::Transpose_688", 76: "onnx::Transpose_692"},
    str(YV / "yolov3-mod-precomile-quantize.onnx.back"):
        {76: "convolution_output", 38: "convolution_output1", 19: "convolution_output2"},
}

# 이름 → (모델, 디코더, .back 경로).  exp = .back 그래프 · tanh = Nota numpy 디코더
TARGETS = {
    "original": (U8 / "original_u8_maD20.onnx", "exp", BACK),
    "pruned40": (U8 / "pruned40_u8_maD20.onnx", "tanh", None),
    "pruned50": (U8 / "pruned50_u8_maD20.onnx", "tanh", None),
    # Neubla 자체 양자화본 — 제 fp32 원본과 같은 계보(첫 Conv 상관 +0.9999).
    # 우리 파이프라인으로 재면 "양자화 품질" 만의 차이를 분리할 수 있습니다.
    # CPU 로는 후처리가 그래프에 내장되어 있어 dets/labels 가 바로 나옵니다(builtin).
    "neubla_q": (YV / "yolov3-mod-precomile-quantize.onnx", "builtin", None),
    # fp32 원본 — uint8 과 완전히 같은 경로로 재기 위한 타깃
    "fp32_orig": (HOME / "workspace_docs/model/yolov3_original_fp32.onnx", "exp", BACK),
    "fp32_p40":  (HOME / "workspace_docs/model/yolov3_pruned40_fp32.onnx", "tanh", None),
    "fp32_p50":  (HOME / "workspace_docs/model/yolov3_pruned50_fp32.onnx", "tanh", None),
}
FP32_REF = {"original": 38.26, "pruned40": 30.19, "pruned50": 26.13, "neubla_q": 38.26,
            "fp32_orig": 38.26, "fp32_p40": 30.19, "fp32_p50": 26.13}


def letterbox(bgr, size=SIZE, color=114):
    """ultralytics `val.py` 와 동일한 전처리 — 종횡비 유지 + 회색 패딩.

    val 경로는 `scaleup=False` 라 원본보다 키우지 않습니다.
    반환: (letterbox 이미지, 스케일 r, 좌상단 패딩 (dw, dh))
    """
    h, w = bgr.shape[:2]
    r = min(size / h, size / w, 1.0)
    nw, nh = round(w * r), round(h * r)
    dw, dh = (size - nw) / 2, (size - nh) / 2
    if (w, h) != (nw, nh):
        bgr = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    out = cv2.copyMakeBorder(bgr, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=(color,) * 3)
    return out, r, (left, top)


def sess(p):
    so = ort.SessionOptions(); so.log_severity_level = 3
    return ort.InferenceSession(str(p), so, providers=["CPUExecutionProvider"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", choices=sorted(TARGETS))
    ap.add_argument("--limit", type=int, default=0, help="0 = 전체 5000장")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--letterbox", action="store_true",
                    help="ultralytics 표준 전처리(종횡비 유지 + 패딩). "
                         "기본은 단순 resize — 표준 대비 약 1.9 AP 낮게 나옵니다")
    args = ap.parse_args()

    path, dec, backp = TARGETS[args.target]
    with contextlib.redirect_stdout(_io.StringIO()):
        coco = COCO(str(COCO_ANN))
    cat_ids = sorted(coco.getCatIds()); all_ids = sorted(coco.getImgIds())
    rng = np.random.default_rng(args.seed)
    img_ids = (sorted(rng.choice(all_ids, args.limit, replace=False).tolist())
               if args.limit and args.limit < len(all_ids) else all_ids)

    m = sess(path); in_name = m.get_inputs()[0].name
    back = sess(backp) if dec == "exp" else None
    bin_ = BACK_IN[str(backp)] if dec == "exp" else None
    print(f"▶ {args.target}  ({path.name}, 디코더 {dec}, {len(img_ids)}장, "
          f"전처리 {'letterbox' if args.letterbox else 'resize'})", flush=True)

    results, t0 = [], time.perf_counter()
    for i, iid in enumerate(img_ids, 1):
        f = coco.loadImgs(iid)[0]
        bgr = cv2.imread(str(COCO_IMG / f["file_name"]))
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        if args.letterbox:
            lb, r, (padx, pady) = letterbox(bgr)
            src = lb
        else:
            src = cv2.resize(bgr, (SIZE, SIZE))
        x = (cv2.cvtColor(src, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
             ).transpose(2, 0, 1)[None]
        out = m.run(None, {in_name: x})
        if dec == "builtin":
            dd, ll = out
            det = np.concatenate([dd[0][:, :5], ll[0][:, None].astype(np.float32)], 1)
        else:
            by = {o.shape[-1]: o for o in out}
        if dec == "exp":
            dd, ll = back.run(None, {bin_[19]: by[19], bin_[38]: by[38], bin_[76]: by[76]})
            det = np.concatenate([dd[0][:, :5], ll[0][:, None].astype(np.float32)], 1)
        elif dec == "tanh":
            det = nms_per_class(decode_nota([by[76], by[38], by[19]], score_thresh=0.001))

        for x1, y1, x2, y2, s, c in det:
            if s <= 0.001:
                continue
            if args.letterbox:                       # 패딩 제거 후 스케일 복원
                x1, x2 = (x1 - padx) / r, (x2 - padx) / r
                y1, y2 = (y1 - pady) / r, (y2 - pady) / r
                x1, x2 = min(max(x1, 0), w), min(max(x2, 0), w)
                y1, y2 = min(max(y1, 0), h), min(max(y2, 0), h)
            else:
                x1, x2 = x1 * w / SIZE, x2 * w / SIZE
                y1, y2 = y1 * h / SIZE, y2 * h / SIZE
            results.append({"image_id": iid, "category_id": cat_ids[int(c)],
                            "bbox": [float(x1), float(y1),
                                     float(x2 - x1), float(y2 - y1)],
                            "score": float(s)})
        if i % 250 == 0:
            el = time.perf_counter() - t0
            print(f"   {i}/{len(img_ids)}  {el/i*1000:.0f} ms/장  "
                  f"남은 {(len(img_ids)-i)*el/i/60:.0f}분", flush=True)

    with contextlib.redirect_stdout(_io.StringIO()):
        e = COCOeval(coco, coco.loadRes(results), "bbox")
        e.params.imgIds = img_ids; e.evaluate(); e.accumulate(); e.summarize()
    s = e.stats * 100
    ref = FP32_REF[args.target]
    print(f"  → AP {s[0]:.2f}  AP50 {s[1]:.2f}  AP75 {s[2]:.2f}  "
          f"APs {s[3]:.2f}  APm {s[4]:.2f}  APl {s[5]:.2f}", flush=True)
    if not args.limit and not args.letterbox:
        print(f"     fp32 {ref:.2f} → 양자화 손실 {ref - s[0]:+.2f}", flush=True)
    print(f"RESULT\t{args.target}\t{s[0]:.2f}\t{s[1]:.2f}\t{s[2]:.2f}"
          f"\t{s[3]:.2f}\t{s[4]:.2f}\t{s[5]:.2f}")

    tag = args.target + ("_lb" if args.letterbox else "")
    Path(f"map_u8_final_{tag}.json").write_text(json.dumps(
        {"target": args.target, "letterbox": args.letterbox,
         "n_images": len(img_ids), "fp32_ref": ref,
         "AP": s[0], "AP50": s[1], "AP75": s[2],
         "APs": s[3], "APm": s[4], "APl": s[5]}, indent=2))


if __name__ == "__main__":
    main()
