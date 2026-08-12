#!/usr/bin/env python3
"""임의의 fp32 ONNX 를 배포 설정으로 uint8 양자화하고 fp32/uint8 을 같은 분할로 평가.

양자화 설정은 배포 파이프라인(`demo_pruning/utils/onnxquant.py`)과 동일합니다:

    QuantFormat.QOperator · per_channel=False · uint8 가중치/활성화

캘리브레이션 이미지는 **평가셋과 겹치지 않는 곳**에서 뽑습니다. 평가셋 이미지를
캘리브레이션에 쓰면 그 자체가 오염입니다.

    python quantize_and_eval.py --model x.onnx --tag p40_ep10 \
        --ann <val2014.json> --imgdir <val2014> --ids-file ids_5k.txt \
        --calib-list trainvalno5k.txt --calib 20
"""
from __future__ import annotations

import argparse, contextlib, io as _io, json, time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod,
                                      QuantFormat, QuantType, quantize_static)
from decoders import decode_std, nms_per_class
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

SIZE = 608


class Reader(CalibrationDataReader):
    def __init__(self, files, in_name):
        self.files, self.in_name, self.i = files, in_name, 0

    def get_next(self):
        if self.i >= len(self.files):
            return None
        p = self.files[self.i]; self.i += 1
        bgr = cv2.imread(p)
        x = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (SIZE, SIZE))
        return {self.in_name: (x.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]}


def evaluate(model, coco, ids, cat_ids, imgdir, tag):
    so = ort.SessionOptions(); so.log_severity_level = 3
    m = ort.InferenceSession(str(model), so, providers=["CPUExecutionProvider"])
    inm = m.get_inputs()[0].name
    res, t = [], 0.0
    for i, iid in enumerate(ids, 1):
        info = coco.loadImgs(iid)[0]
        bgr = cv2.imread(str(Path(imgdir) / info["file_name"]))
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        x = (cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (SIZE, SIZE))
             .astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        t0 = time.perf_counter(); heads = m.run(None, {inm: x}); t += time.perf_counter() - t0
        by = {a.shape[-1]: a for a in heads}
        det = nms_per_class(decode_std([by[76], by[38], by[19]], score_thresh=0.001))
        sx, sy = w / SIZE, h / SIZE
        for x1, y1, x2, y2, s, c in det:
            res.append({"image_id": iid, "category_id": cat_ids[int(c)],
                        "bbox": [round(float(x1*sx), 2), round(float(y1*sy), 2),
                                 round(float((x2-x1)*sx), 2), round(float((y2-y1)*sy), 2)],
                        "score": round(float(s), 5)})
        if i % 500 == 0:
            print(f"  [{tag}] {i}/{len(ids)} ({t/i*1000:.0f} ms/장)", flush=True)
    with contextlib.redirect_stdout(_io.StringIO()):
        e = COCOeval(coco, coco.loadRes(res), "bbox"); e.params.imgIds = ids
        e.evaluate(); e.accumulate(); e.summarize()
    s = e.stats
    return {"tag": tag, "AP": s[0]*100, "AP50": s[1]*100, "AP75": s[2]*100,
            "APs": s[3]*100, "APm": s[4]*100, "APl": s[5]*100,
            "ndet": len(res), "ms": t/len(ids)*1000}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ann", required=True)
    ap.add_argument("--imgdir", required=True)
    ap.add_argument("--ids-file", required=True)
    ap.add_argument("--calib-list", required=True, help="캘리브레이션 후보 이미지 목록 (평가셋과 무관해야 함)")
    ap.add_argument("--calib", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--calib-files", default=None,
                    help="캘리브 파일명 목록(줄당 하나). 주면 무작위 추출 대신 이걸 사용 — 재현용")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model = Path(args.model)
    u8 = model.with_name(model.stem + f"_u8ma{args.calib}.onnx")

    if not u8.exists():
        pool = [l.strip() for l in open(args.calib_list) if l.strip()]
        if args.calib_files:
            # 산출물 고정 방식 — 절차(시드)가 아니라 목록 자체를 정의로 삼는다.
            # np.random.default_rng 는 NumPy 버전 간 스트림 호환을 보장하지 않으므로
            # 시드만으로는 재현되지 않는다.
            want = {l.strip() for l in open(args.calib_files) if l.strip()}
            by_name = {Path(p).name: p for p in pool}
            missing = want - by_name.keys()
            if missing:
                raise SystemExit(f"캘리브 목록 중 {len(missing)}장을 pool 에서 못 찾음: {sorted(missing)[:3]}")
            files = [by_name[n] for n in sorted(want)]
            print(f"캘리브 목록 지정: {len(files)}장 ({args.calib_files})", flush=True)
        else:
            rng = np.random.default_rng(args.seed)
            files = [pool[i] for i in rng.choice(len(pool), args.calib, replace=False)]
        rec = model.with_name(model.stem + f"_calib{len(files)}.txt")
        rec.write_text("\n".join(Path(f).name for f in files) + "\n")
        print(f"사용한 캘리브 목록 저장 → {rec.name}", flush=True)
        so = ort.SessionOptions(); so.log_severity_level = 3
        in_name = ort.InferenceSession(str(model), so,
                                       providers=["CPUExecutionProvider"]).get_inputs()[0].name
        print(f"양자화: 캘리브 {args.calib}장 · MovingAverage · QOperator · per_channel=False", flush=True)
        t0 = time.perf_counter()
        quantize_static(str(model), str(u8),
                        calibration_data_reader=Reader(files, in_name),
                        quant_format=QuantFormat.QOperator,
                        per_channel=False,
                        weight_type=QuantType.QUInt8,
                        activation_type=QuantType.QUInt8,
                        calibrate_method=CalibrationMethod.MinMax)
        print(f"  → {u8.name}  ({u8.stat().st_size/1e6:.1f} MB, {time.perf_counter()-t0:.0f}s)", flush=True)

    with contextlib.redirect_stdout(_io.StringIO()):
        coco = COCO(args.ann)
    cat_ids = sorted(coco.getCatIds())
    ids = sorted(set(coco.getImgIds()) & {int(x) for x in open(args.ids_file).read().split()})
    print(f"평가 {len(ids)}장\n", flush=True)

    rows = [evaluate(model, coco, ids, cat_ids, args.imgdir, f"{args.tag} fp32"),
            evaluate(u8, coco, ids, cat_ids, args.imgdir, f"{args.tag} uint8")]
    print("\n" + "=" * 60)
    for r in rows:
        print(f"{r['tag']:<20} AP {r['AP']:6.2f}  AP50 {r['AP50']:6.2f}  AP75 {r['AP75']:6.2f}  {r['ms']:.0f} ms")
    print("-" * 60)
    print(f"{'양자화 손실':<20} {rows[1]['AP']-rows[0]['AP']:+6.2f}")
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2, ensure_ascii=False))
        print(f"→ {args.out}")


if __name__ == "__main__":
    main()
