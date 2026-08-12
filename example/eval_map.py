#!/usr/bin/env python3
"""YOLOv3 원본 / 40% / 50% 프루닝 모델의 COCO mAP 비교.

세 모델에 **같은 이미지 집합**을 먹여 쌍대 비교합니다. 절대 mAP 는 표본 크기에
따라 흔들리지만, 같은 표본을 쓰므로 모델 간 차이는 훨씬 안정적입니다.

    python eval_map.py                 # 기본 500장
    python eval_map.py --limit 2000
    python eval_map.py --limit 0       # 전체 5000장

추론 경로:
    이미지 → resize 608 → float 모델 → 헤드 3개 → back ONNX(앵커 디코딩+NMS) → dets/labels
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
BACK = HOME / ("demo_1127/yolov3/yolov3_d53_mstrain-608_273e_coco_optim_opset12"
               ".neubla_u8_lwq_movingaverage.onnx.back")
MODEL_DIR = HOME / "workspace_docs/model"

# 모델마다 헤드 디코딩 공식이 다릅니다 — decoders.py 참조
#   exp  : 표준 YOLOv3   wh = exp(t) * anchor          (원본 .back 서브그래프)
#   tanh : Nota 프루닝    wh = 2*(tanh(t/2-0.549)+1)*anchor
MODELS = [
    ("original", MODEL_DIR / "yolov3_original_fp32.onnx", "exp"),
    ("pruned40", MODEL_DIR / "yolov3_pruned40_fp32.onnx", "tanh"),
    ("pruned50", MODEL_DIR / "yolov3_pruned50_fp32.onnx", "tanh"),
]
SIZE = 608


def sess(path: Path) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def evaluate(name: str, model_path: Path, decoder: str, coco: COCO, img_ids: list[int],
             back: ort.InferenceSession, cat_ids: list[int]) -> dict:
    m = sess(model_path)
    in_name = m.get_inputs()[0].name
    results, t_total = [], 0.0

    for i, iid in enumerate(img_ids, 1):
        info = coco.loadImgs(iid)[0]
        bgr = cv2.imread(str(COCO_IMG / info["file_name"]))
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        x = (cv2.resize(rgb, (SIZE, SIZE)).astype(np.float32) / 255.0).transpose(2, 0, 1)[None]

        t0 = time.perf_counter()
        heads = m.run(None, {in_name: x})
        t_total += time.perf_counter() - t0

        by_res = {hd.shape[-1]: hd for hd in heads}          # 19 / 38 / 76 으로 매칭
        if decoder == "exp":
            d, l = back.run(None, {"onnx::Transpose_684": by_res[19],
                                   "onnx::Transpose_688": by_res[38],
                                   "onnx::Transpose_692": by_res[76]})
            det = np.concatenate([d[0][:, :5], l[0][:, None].astype(np.float32)], 1)
        else:
            det = nms_per_class(decode_nota([by_res[76], by_res[38], by_res[19]],
                                            score_thresh=0.001))

        # 608x608 좌표 → 원본 이미지 좌표 (letterbox 아닌 단순 resize)
        sx, sy = w / SIZE, h / SIZE
        for x1, y1, x2, y2, score, cls in det:
            if score <= 0.001:                                # COCO 관례: 낮은 점수도 포함
                continue
            results.append({
                "image_id": iid,
                "category_id": cat_ids[int(cls)],
                "bbox": [round(float(x1 * sx), 2), round(float(y1 * sy), 2),
                         round(float((x2 - x1) * sx), 2), round(float((y2 - y1) * sy), 2)],
                "score": round(float(score), 5),
            })

        if i % 100 == 0:
            print(f"  [{name}] {i}/{len(img_ids)}  ({t_total / i * 1000:.0f} ms/장)", flush=True)

    if not results:
        return {"name": name, "error": "검출 0개"}

    with contextlib.redirect_stdout(_io.StringIO()):
        dt = coco.loadRes(results)
        e = COCOeval(coco, dt, "bbox")
        e.params.imgIds = img_ids
        e.evaluate(); e.accumulate(); e.summarize()

    s = e.stats
    return {"name": name, "AP": s[0], "AP50": s[1], "AP75": s[2],
            "APs": s[3], "APm": s[4], "APl": s[5],
            "ms": t_total / len(img_ids) * 1000, "ndet": len(results)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=500, help="이미지 수 (0 = 전체)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="map_results.json")
    ap.add_argument("--only", default=None, help="특정 모델만 (예: original)")
    args = ap.parse_args()

    with contextlib.redirect_stdout(_io.StringIO()):
        coco = COCO(str(COCO_ANN))
    cat_ids = sorted(coco.getCatIds())                        # 80개, 인덱스 0-79 → COCO id
    all_ids = sorted(coco.getImgIds())

    if args.limit and args.limit < len(all_ids):
        rng = np.random.default_rng(args.seed)
        img_ids = sorted(rng.choice(all_ids, args.limit, replace=False).tolist())
    else:
        img_ids = all_ids

    print(f"이미지 {len(img_ids)}장 (전체 {len(all_ids)}장 중, seed={args.seed})\n")
    back = sess(BACK)

    models = [m for m in MODELS if not args.only or m[0] == args.only]
    rows = []
    for name, path, dec in models:
        print(f"▶ {name}  ({path.stat().st_size / 1e6:.1f} MB, 디코더 {dec})")
        r = evaluate(name, path, dec, coco, img_ids, back, cat_ids)
        rows.append(r)
        if "error" not in r:
            print(f"  → AP {r['AP']*100:.2f}  AP50 {r['AP50']*100:.2f}  "
                  f"AP75 {r['AP75']*100:.2f}  ({r['ms']:.0f} ms/장)\n", flush=True)

    base = rows[0]
    print("=" * 74)
    print(f"{'모델':<11}{'AP':>7}{'AP50':>8}{'AP75':>8}{'APs':>7}{'APm':>7}{'APl':>7}{'ms':>7}")
    print("-" * 74)
    for r in rows:
        if "error" in r:
            print(f"{r['name']:<11} {r['error']}"); continue
        print(f"{r['name']:<11}{r['AP']*100:7.2f}{r['AP50']*100:8.2f}{r['AP75']*100:8.2f}"
              f"{r['APs']*100:7.2f}{r['APm']*100:7.2f}{r['APl']*100:7.2f}{r['ms']:7.0f}")
    print("-" * 74)
    for r in rows[1:]:
        if "error" in r:
            continue
        print(f"{r['name']:<11} 원본 대비  AP {(r['AP']-base['AP'])*100:+.2f}  "
              f"AP50 {(r['AP50']-base['AP50'])*100:+.2f}  속도 {base['ms']/r['ms']:.2f}×")
    print("=" * 74)

    Path(args.out).write_text(json.dumps(
        {"n_images": len(img_ids), "seed": args.seed, "results": rows}, indent=2))
    print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()
