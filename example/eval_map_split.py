#!/usr/bin/env python3
"""임의의 COCO 분할에서 원본/40%/50% 프루닝 모델을 동일 조건으로 평가.

배경 — 두 모델 계열의 held-out 집합이 서로 다릅니다.

    원본(Darknet)      학습 trainvalno5k = train2014 + val2014 − 5k(pjreddie)
    프루닝본(Nota)     학습 train2017    = train2014 + val2014 − val2017

    → 5k 로 재면 프루닝본이 유리(87%가 학습 데이터), val2017 로 재면 원본이 유리.
      두 집합의 교집합 643장만이 양쪽 모두에게 held-out 입니다.

전처리·디코더·NMS 는 세 모델에 동일하게 적용합니다 (디코딩 공식만 계열별로 다름).

    python eval_map_split.py --ann <json> --imgdir <dir> --ids-file <txt> --out r.json
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
from decoders import decode_nota, decode_std, nms_per_class
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

MODEL_DIR = Path.home() / "workspace_docs/model"
MODELS = [
    ("original", MODEL_DIR / "yolov3_original_fp32.onnx", "exp"),
    ("pruned40", MODEL_DIR / "../paper/model/final/final_r40_sc64.onnx", "nota"),
    ("pruned50", MODEL_DIR / "../paper/model/final/final_r50_sc64.onnx", "nota"),
    # p40ep10 재현본은 논문에 쓰지 않아 fgcs/archive/neubla/models-not-in-paper/model/ 로 옮겼다 (2026-09-16)
    # ("p40ep10",  MODEL_DIR / "yolov3_p40_ep10_fp32.onnx", "exp"),
]


def preprocess(bgr: np.ndarray, size: int, mode: str):
    """반환: (입력텐서, 역변환 파라미터). 역변환은 (scale_x, scale_y, pad_x, pad_y)."""
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if mode == "letterbox":
        s = min(size / w, size / h)
        nw, nh = int(round(w * s)), int(round(h * s))
        canvas = np.full((size, size, 3), 128, np.uint8)
        px, py = (size - nw) // 2, (size - nh) // 2
        canvas[py:py + nh, px:px + nw] = cv2.resize(rgb, (nw, nh))
        x = canvas.astype(np.float32) / 255.0
        inv = (1.0 / s, 1.0 / s, px, py)
    else:                                            # plain resize
        x = cv2.resize(rgb, (size, size)).astype(np.float32) / 255.0
        inv = (w / size, h / size, 0, 0)
    return x.transpose(2, 0, 1)[None], inv


def evaluate(name, path, dec, coco, img_ids, cat_ids, imgdir, size, mode):
    so = ort.SessionOptions(); so.log_severity_level = 3
    m = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    in_name = m.get_inputs()[0].name
    decode = decode_std if dec == "exp" else decode_nota
    results, t = [], 0.0

    for i, iid in enumerate(img_ids, 1):
        info = coco.loadImgs(iid)[0]
        bgr = cv2.imread(str(Path(imgdir) / info["file_name"]))
        if bgr is None:
            continue
        x, (sx, sy, px, py) = preprocess(bgr, size, mode)
        t0 = time.perf_counter()
        heads = m.run(None, {in_name: x})
        t += time.perf_counter() - t0
        by = {hd.shape[-1]: hd for hd in heads}
        det = nms_per_class(decode([by[size // 8], by[size // 16], by[size // 32]],
                                   score_thresh=0.001, input_size=size))
        for x1, y1, x2, y2, sc, cl in det:
            X1, Y1 = (x1 - px) * sx, (y1 - py) * sy
            X2, Y2 = (x2 - px) * sx, (y2 - py) * sy
            results.append({"image_id": iid, "category_id": cat_ids[int(cl)],
                            "bbox": [round(float(X1), 2), round(float(Y1), 2),
                                     round(float(X2 - X1), 2), round(float(Y2 - Y1), 2)],
                            "score": round(float(sc), 5)})
        if i % 200 == 0:
            print(f"  [{name}] {i}/{len(img_ids)} ({t/i*1000:.0f} ms/장)", flush=True)

    if not results:
        return {"name": name, "error": "검출 0"}, []
    with contextlib.redirect_stdout(_io.StringIO()):
        e = COCOeval(coco, coco.loadRes(results), "bbox")
        e.params.imgIds = img_ids
        e.evaluate(); e.accumulate(); e.summarize()
    s = e.stats
    return ({"name": name, "AP": s[0]*100, "AP50": s[1]*100, "AP75": s[2]*100,
             "APs": s[3]*100, "APm": s[4]*100, "APl": s[5]*100,
             "ndet": len(results), "ms": t/len(img_ids)*1000}, results)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True)
    ap.add_argument("--imgdir", required=True)
    ap.add_argument("--ids-file", default=None)
    ap.add_argument("--size", type=int, default=608)
    ap.add_argument("--preproc", choices=["resize", "letterbox"], default="resize")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--only", default=None, help="특정 모델만 (original/pruned40/pruned50)")
    ap.add_argument("--save-dets", default=None, help="검출 JSON 저장 경로 접두어")
    args = ap.parse_args()

    with contextlib.redirect_stdout(_io.StringIO()):
        coco = COCO(args.ann)
    cat_ids = sorted(coco.getCatIds())
    all_ids = sorted(coco.getImgIds())
    if args.ids_file:
        want = {int(v) for v in Path(args.ids_file).read_text().split()}
        img_ids = sorted(set(all_ids) & want)
    else:
        img_ids = all_ids
    print(f"[{args.tag}] 이미지 {len(img_ids)}장 · {args.size}px · {args.preproc}\n", flush=True)

    rows = []
    for name, path, dec in MODELS:
        if args.only and name != args.only:
            continue
        print(f"▶ {name} ({dec})", flush=True)
        r, dets = evaluate(name, path, dec, coco, img_ids, cat_ids, args.imgdir, args.size, args.preproc)
        if args.save_dets:
            Path(f"{args.save_dets}_{name}.json").write_text(json.dumps(dets))
            print(f"  검출 저장 → {args.save_dets}_{name}.json ({len(dets):,}개)", flush=True)
        rows.append(r)
        if "error" not in r:
            print(f"  → AP {r['AP']:.2f}  AP50 {r['AP50']:.2f}  AP75 {r['AP75']:.2f}\n", flush=True)

    print("=" * 66)
    print(f"{'모델':<11}{'AP':>8}{'AP50':>8}{'AP75':>8}{'APs':>7}{'APm':>7}{'APl':>7}")
    for r in rows:
        if "error" in r:
            print(f"{r['name']:<11} {r['error']}"); continue
        print(f"{r['name']:<11}{r['AP']:8.2f}{r['AP50']:8.2f}{r['AP75']:8.2f}"
              f"{r['APs']:7.2f}{r['APm']:7.2f}{r['APl']:7.2f}")
    base = rows[0]
    print("-" * 66)
    for r in rows[1:]:
        if "error" not in r and "error" not in base:
            print(f"{r['name']:<11} 원본 대비 AP {r['AP']-base['AP']:+.2f}")
    if args.out:
        Path(args.out).write_text(json.dumps(
            {"tag": args.tag, "ann": args.ann, "n_images": len(img_ids),
             "size": args.size, "preproc": args.preproc, "results": rows},
            indent=2, ensure_ascii=False))
        print(f"→ {args.out}")


if __name__ == "__main__":
    main()
