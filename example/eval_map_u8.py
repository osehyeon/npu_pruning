#!/usr/bin/env python3
"""uint8 양자화 모델의 COCO mAP — 실제 NPU 에 올라가는 모델의 정확도.

fp32 측정([eval_map.py])이 **프루닝만의 손실**을 보는 것이라면,
이 스크립트는 **프루닝 + 양자화 합계 손실**을 봅니다.

모델마다 출력 형태가 다릅니다:
  원본 uint8   dets/labels 를 직접 출력 (후처리가 그래프 안에 들어있음)
  프루닝 uint8 raw 헤드 3개 → Nota tanh 디코더 필요

    python eval_map_u8.py --limit 500
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
SIZE = 608

# (이름, 경로, 출력형태)   builtin = dets/labels 내장 · heads = raw 헤드 + tanh 디코더
MODELS = [
    ("원본 u8",   HOME / "demo_1127/yolov3/yolov3_d53_mstrain-608_273e_coco_optim_opset12"
                        ".neubla_u8_lwq_movingaverage.onnx", "builtin"),
    ("40% u8",    HOME / "demo_0211/pruning_model/40.onnx", "heads"),
    ("50% u8",    HOME / "demo_0211/pruning_model/50.onnx", "heads"),
]


def sess(p) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(str(p), so, providers=["CPUExecutionProvider"])


def evaluate(name, path, kind, coco, img_ids, cat_ids) -> dict:
    m = sess(path)
    in_name = m.get_inputs()[0].name
    results, t_total = [], 0.0

    for i, iid in enumerate(img_ids, 1):
        info = coco.loadImgs(iid)[0]
        bgr = cv2.imread(str(COCO_IMG / info["file_name"]))
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        x = (cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (SIZE, SIZE))
             .astype(np.float32) / 255.0).transpose(2, 0, 1)[None]

        t0 = time.perf_counter()
        out = m.run(None, {in_name: x})
        t_total += time.perf_counter() - t0

        if kind == "builtin":                                  # dets(1,100,5) · labels(1,100)
            d, l = out[0], out[1]
            det = np.concatenate([d[0][:, :5], l[0][:, None].astype(np.float32)], 1)
        else:                                                  # raw 헤드 → tanh 디코더
            by = {o.shape[-1]: o for o in out}
            det = nms_per_class(decode_nota([by[76], by[38], by[19]], score_thresh=0.001))

        sx, sy = w / SIZE, h / SIZE
        for x1, y1, x2, y2, score, cls in det:
            if score <= 0.001:
                continue
            results.append({"image_id": iid, "category_id": cat_ids[int(cls)],
                            "bbox": [round(float(x1 * sx), 2), round(float(y1 * sy), 2),
                                     round(float((x2 - x1) * sx), 2), round(float((y2 - y1) * sy), 2)],
                            "score": round(float(score), 5)})

        if i % 100 == 0:
            print(f"  [{name}] {i}/{len(img_ids)}  ({t_total / i * 1000:.0f} ms/장)", flush=True)

    if not results:
        return {"name": name, "error": "검출 0개"}
    with contextlib.redirect_stdout(_io.StringIO()):
        e = COCOeval(coco, coco.loadRes(results), "bbox")
        e.params.imgIds = img_ids
        e.evaluate(); e.accumulate(); e.summarize()
    s = e.stats
    return {"name": name, "AP": s[0], "AP50": s[1], "AP75": s[2],
            "APs": s[3], "APm": s[4], "APl": s[5], "ms": t_total / len(img_ids) * 1000}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="map_results_u8.json")
    args = ap.parse_args()

    with contextlib.redirect_stdout(_io.StringIO()):
        coco = COCO(str(COCO_ANN))
    cat_ids = sorted(coco.getCatIds())
    all_ids = sorted(coco.getImgIds())
    rng = np.random.default_rng(args.seed)
    img_ids = (sorted(rng.choice(all_ids, args.limit, replace=False).tolist())
               if args.limit and args.limit < len(all_ids) else all_ids)

    print(f"uint8 모델 · 이미지 {len(img_ids)}장 (seed={args.seed})\n")
    rows = []
    for name, path, kind in MODELS:
        if not path.exists():
            print(f"{name}: 파일 없음"); continue
        print(f"▶ {name}  ({path.stat().st_size/1e6:.1f} MB, {kind})")
        r = evaluate(name, path, kind, coco, img_ids, cat_ids)
        rows.append(r)
        if "error" not in r:
            print(f"  → AP {r['AP']*100:.2f}  AP50 {r['AP50']*100:.2f}  ({r['ms']:.0f} ms/장)\n", flush=True)

    print("=" * 68)
    print(f"{'모델':<11}{'AP':>8}{'AP50':>8}{'AP75':>8}{'APs':>7}{'APm':>7}{'APl':>7}")
    print("-" * 68)
    for r in rows:
        if "error" in r:
            print(f"{r['name']:<11} {r['error']}"); continue
        print(f"{r['name']:<11}{r['AP']*100:8.2f}{r['AP50']*100:8.2f}{r['AP75']*100:8.2f}"
              f"{r['APs']*100:7.2f}{r['APm']*100:7.2f}{r['APl']*100:7.2f}")
    print("=" * 68)
    Path(args.out).write_text(json.dumps({"n_images": len(img_ids), "results": rows}, indent=2))
    print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()
