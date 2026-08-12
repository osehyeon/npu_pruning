#!/usr/bin/env python3
"""ONNX Model Zoo `yolov3-12.onnx` 를 **손대지 않고** COCO val2017 로 평가.

이 파일은 디코딩·NMS 가 그래프 안에 들어 있어서 우리 디코더가 전혀 개입하지
않습니다. 따라서 이 스크립트의 결과는 **평가 하네스(이미지 로딩·전처리·좌표·
카테고리 매핑·pycocotools 호출)만** 검증합니다.

그래프에 박힌 NMS 파라미터:
    max_output_size 20/클래스 · iou_threshold 0.5 · score_threshold 0.6
score 0.6 은 COCO 관례(0.001)보다 훨씬 높아 AP 가 과소평가됩니다.
Zoo 모델 카드의 공개값(416 에서 mAP 0.2874)이 이 조건에서 나온 값이므로,
**우리가 그 값을 재현하면 하네스는 정상**입니다.

    python eval_map_zoo.py --limit 100          # 배관 확인
    python eval_map_zoo.py --limit 0 --size 416 # 공개값 대조 (전수)
"""
from __future__ import annotations

import argparse
import contextlib
import io as _io
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

HOME = Path.home()
COCO_IMG = HOME / "datasets/coco/images/val2017"
COCO_ANN = HOME / "datasets/coco/annotations/instances_val2017.json"
ZOO = HOME / ("workspace_docs/YOLOv3_Antara_재현패키지/전처리/구글드라이브/yolov3-12.onnx")


def letterbox(img: Image.Image, size: int) -> np.ndarray:
    """keras-yolo3 원본과 동일 — BICUBIC 축소 후 (128,128,128) 패딩."""
    iw, ih = img.size
    scale = min(size / iw, size / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    new = Image.new("RGB", (size, size), (128, 128, 128))
    new.paste(img.resize((nw, nh), Image.BICUBIC), ((size - nw) // 2, (size - nh) // 2))
    x = np.asarray(new, dtype=np.float32) / 255.0
    return np.transpose(x, (2, 0, 1))[None]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100, help="이미지 수 (0 = 전체 5000)")
    ap.add_argument("--size", type=int, default=416)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--model", default=str(ZOO),
                    help="기본은 Zoo 원본. NMS 상수를 표준값으로 바꾼 사본을 줄 수도 있음")
    ap.add_argument("--topk", type=int, default=100,
                    help="이미지당 상위 N개만 제출 (COCO maxDets 관례)")
    ap.add_argument("--ids-file", default=None,
                    help="평가할 image_id 목록 파일 (한 줄에 하나) — 외부 구현과 동일 표본 대조용")
    args = ap.parse_args()
    model_path = Path(args.model)

    with contextlib.redirect_stdout(_io.StringIO()):
        coco = COCO(str(COCO_ANN))
    cat_ids = sorted(coco.getCatIds())
    all_ids = sorted(coco.getImgIds())
    if args.ids_file:
        want = {int(x) for x in Path(args.ids_file).read_text().split()}
        img_ids = sorted(set(all_ids) & want)
        print(f"ids-file: {len(want)}개 요청 → {len(img_ids)}개 매칭")
    elif args.limit and args.limit < len(all_ids):
        rng = np.random.default_rng(args.seed)
        img_ids = sorted(rng.choice(all_ids, args.limit, replace=False).tolist())
    else:
        img_ids = all_ids

    so = ort.SessionOptions()
    so.log_severity_level = 3
    m = ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])
    print(f"모델 {model_path.name}  입력 {[i.name for i in m.get_inputs()]}")
    print(f"이미지 {len(img_ids)}장 · {args.size}x{args.size} letterbox · 그래프 내장 NMS\n",
          flush=True)

    results, t_total, n_empty = [], 0.0, 0
    for i, iid in enumerate(img_ids, 1):
        info = coco.loadImgs(iid)[0]
        img = Image.open(COCO_IMG / info["file_name"]).convert("RGB")
        x = letterbox(img, args.size)
        shape = np.array([[img.size[1], img.size[0]]], dtype=np.float32)   # (h, w)

        t0 = time.perf_counter()
        boxes, scores, indices = m.run(None, {"input_1": x, "image_shape": shape})
        t_total += time.perf_counter() - t0

        idx = np.asarray(indices)
        if idx.ndim == 3:                       # [1, N, 3] 형태도 대응
            idx = idx[0]
        if idx.size == 0:
            n_empty += 1
            continue
        per_img = []
        for b_i, c_i, box_i in idx:
            y1, x1, y2, x2 = boxes[b_i, box_i]          # 원본 이미지 좌표 (y1,x1,y2,x2)
            per_img.append({
                "image_id": iid,
                "category_id": cat_ids[int(c_i)],
                "bbox": [round(float(x1), 2), round(float(y1), 2),
                         round(float(x2 - x1), 2), round(float(y2 - y1), 2)],
                "score": round(float(scores[b_i, c_i, box_i]), 5),
            })
        per_img.sort(key=lambda d: -d["score"])          # COCO maxDets 관례
        results.extend(per_img[:args.topk])

        if i % 200 == 0:
            print(f"  {i}/{len(img_ids)}  ({t_total / i * 1000:.0f} ms/장, "
                  f"검출 {len(results)}개)", flush=True)

    print(f"\n검출 총 {len(results)}개 · 검출 0인 이미지 {n_empty}장")
    if not results:
        print("검출이 없습니다."); return

    with contextlib.redirect_stdout(_io.StringIO()):
        dt = coco.loadRes(results)
        e = COCOeval(coco, dt, "bbox")
        e.params.imgIds = img_ids
        e.evaluate(); e.accumulate(); e.summarize()
    s = e.stats
    out = {"model": "onnx_zoo_yolov3-12", "size": args.size, "n_images": len(img_ids),
           "AP": s[0] * 100, "AP50": s[1] * 100, "AP75": s[2] * 100,
           "APs": s[3] * 100, "APm": s[4] * 100, "APl": s[5] * 100,
           "ndet": len(results), "ms": t_total / len(img_ids) * 1000}
    print(f"\nAP {out['AP']:.2f}  AP50 {out['AP50']:.2f}  AP75 {out['AP75']:.2f}  "
          f"APs {out['APs']:.2f}  APm {out['APm']:.2f}  APl {out['APl']:.2f}")
    print(f"공개값 대조: Zoo 모델 카드 416 = 28.74")
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(f"→ {args.out}")


if __name__ == "__main__":
    main()
