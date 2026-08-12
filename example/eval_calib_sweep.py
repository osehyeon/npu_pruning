#!/usr/bin/env python3
"""캘리브레이션 장수 × 모델 격자 mAP — 교란변수 판별용.

기존 uint8 모델들은 서로 다른 시점·설정으로 양자화되어 있었습니다.
이 스크립트는 **동일 이미지로 재양자화한 9개**(3모델 × calib 1/20/100)를
같은 평가 이미지로 재서, 앞서 관측한 격차가 진짜인지 판별합니다.

    원본이 캘리브레이션 부족으로 손해 본 것이라면
    → calib 100 에서 원본 손실이 크게 줄어야 합니다.

    python eval_calib_sweep.py --limit 500
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
U8_DIR = HOME / "workspace_docs/model/u8"
BACK = HOME / ("demo_1127/yolov3/yolov3_d53_mstrain-608_273e_coco_optim_opset12"
               ".neubla_u8_lwq_movingaverage.onnx.back")
SIZE = 608

# 재양자화 모델은 전부 raw 헤드를 냅니다(fp32 원본에서 양자화했으므로).
#   original → exp 디코더(.back) · pruned → tanh 디코더
DECODER = {"original": "exp", "pruned40": "tanh", "pruned50": "tanh"}
FP32_REF = {"original": 38.26, "pruned40": 30.19, "pruned50": 26.13}   # 전체 val2017 기준


def sess(p):
    so = ort.SessionOptions(); so.log_severity_level = 3
    return ort.InferenceSession(str(p), so, providers=["CPUExecutionProvider"])


def evaluate(path, dec, back, coco, img_ids, cat_ids, tag):
    m = sess(path); inp = m.get_inputs()[0].name
    res, t0all = [], time.perf_counter()
    for i, iid in enumerate(img_ids, 1):
        f = coco.loadImgs(iid)[0]
        bgr = cv2.imread(str(COCO_IMG / f["file_name"]))
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        x = (cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (SIZE, SIZE))
             .astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        out = m.run(None, {inp: x})
        by = {o.shape[-1]: o for o in out}
        if dec == "exp":
            d, l = back.run(None, {"onnx::Transpose_684": by[19],
                                   "onnx::Transpose_688": by[38],
                                   "onnx::Transpose_692": by[76]})
            det = np.concatenate([d[0][:, :5], l[0][:, None].astype(np.float32)], 1)
        else:
            det = nms_per_class(decode_nota([by[76], by[38], by[19]], score_thresh=0.001))
        sx, sy = w / SIZE, h / SIZE
        for x1, y1, x2, y2, s, c in det:
            if s > 0.001:
                res.append({"image_id": iid, "category_id": cat_ids[int(c)],
                            "bbox": [float(x1*sx), float(y1*sy),
                                     float((x2-x1)*sx), float((y2-y1)*sy)],
                            "score": float(s)})
        if i % 200 == 0:
            print(f"    {tag} {i}/{len(img_ids)}", flush=True)
    if not res:
        return None
    with contextlib.redirect_stdout(_io.StringIO()):
        e = COCOeval(coco, coco.loadRes(res), "bbox")
        e.params.imgIds = img_ids; e.evaluate(); e.accumulate(); e.summarize()
    return {"AP": e.stats[0]*100, "AP50": e.stats[1]*100,
            "sec": time.perf_counter() - t0all}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--method", choices=("mm", "ma", "both"), default="both",
                    help="mm=MinMax · ma=MovingAverage(배포 조건)")
    ap.add_argument("--out", default="calib_sweep.json")
    args = ap.parse_args()

    with contextlib.redirect_stdout(_io.StringIO()):
        coco = COCO(str(COCO_ANN))
    cat_ids = sorted(coco.getCatIds()); all_ids = sorted(coco.getImgIds())
    rng = np.random.default_rng(args.seed)
    img_ids = (sorted(rng.choice(all_ids, args.limit, replace=False).tolist())
               if args.limit and args.limit < len(all_ids) else all_ids)
    back = sess(BACK)

    print(f"평가 이미지 {len(img_ids)}장 (seed={args.seed})\n")
    methods = ("mm", "ma") if args.method == "both" else (args.method,)
    grid = {}
    for meth in methods:
      for model in ("original", "pruned40", "pruned50"):
        for n in (1, 20, 100):
            p = U8_DIR / f"{model}_u8_{meth}{n}.onnx"
            if not p.exists():
                continue
            tag = f"{meth}/{model}/calib{n}"
            print(f"▶ {tag}")
            r = evaluate(p, DECODER[model], back, coco, img_ids, cat_ids, tag)
            grid[tag] = r
            print(f"  → AP {r['AP']:.2f}  AP50 {r['AP50']:.2f}  ({r['sec']:.0f}s)\n", flush=True)

    NAME = {"mm": "MinMax", "ma": "MovingAverage(배포)"}
    for meth in methods:
        print("=" * 70)
        print(f"[{NAME[meth]}]  {'모델':<11}{'fp32':>8}{'calib 1':>10}{'calib 20':>10}{'calib 100':>11}")
        print("-" * 70)
        for model in ("original", "pruned40", "pruned50"):
            row = [grid.get(f"{meth}/{model}/calib{n}") for n in (1, 20, 100)]
            cells = "".join(f"{r['AP']:>10.2f}" if r else f"{'—':>10}" for r in row)
            print(f"{'':<11}{model:<11}{FP32_REF[model]:>8.2f}{cells}")
        print(f"{'':<11}양자화 손실 (fp32 − uint8)")
        for model in ("original", "pruned40", "pruned50"):
            row = [grid.get(f"{meth}/{model}/calib{n}") for n in (1, 20, 100)]
            cells = "".join(f"{FP32_REF[model]-r['AP']:>+10.2f}" if r else f"{'—':>10}" for r in row)
            print(f"{'':<11}{model:<11}{'':>8}{cells}")
    print("=" * 70)
    Path(args.out).write_text(json.dumps(
        {"n_images": len(img_ids), "fp32_ref": FP32_REF, "grid": grid}, indent=2, ensure_ascii=False))
    print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()
