#!/usr/bin/env python3
"""세 모델을 **동일한 설정**으로 재양자화 — 교란변수 제거용.

기존 워크스페이스의 uint8 모델들은 서로 다른 시점·다른 설정으로 양자화되어
있어서 "양자화 손실" 비교가 오염되어 있습니다:

    원본 u8   demo_1127/...neubla_u8_lwq_movingaverage.onnx   출처·설정 불명
    40% u8    demo_0211/pruning_model/40.onnx                 별개 실행 (5월)

이 스크립트는 셋을 같은 캘리브레이션 이미지·같은 장수·같은 방식으로 다시 양자화합니다.

    python requantize.py --calib 20
    python requantize.py --calib 1 --calib 20 --calib 100   (여러 개면 각각 생성)

설정은 배포 파이프라인(demo_pruning/utils/onnxquant.py)과 동일합니다:
    QuantFormat.QOperator · per_channel=False · uint8 가중치/활성화
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod,
                                      QuantFormat, QuantType, quantize_static)

HOME = Path.home()
MODEL_DIR = HOME / "workspace_docs/model"
OUT_DIR = HOME / "workspace_docs/model/u8"
CALIB_DIR = HOME / "datasets/coco/images/val2017"     # 고정 시드로 뽑아 셋이 동일하게 사용
SIZE = 608

DRIVE = HOME / "workspace_docs/YOLOv3_Antara_재현패키지/전처리/구글드라이브"
MODELS = [
    ("original", MODEL_DIR / "yolov3_original_fp32.onnx", "input.1"),
    ("pruned40", MODEL_DIR / "yolov3_pruned40_fp32.onnx", "input_1"),
    ("pruned50", MODEL_DIR / "yolov3_pruned50_fp32.onnx", "input_1"),
]
# staircase 스윕 (50% × sc 8/16/32/64) — 하드웨어 정렬 실험용
SWEEP50 = [(f"sw50sc{sc}", DRIVE / f"sweep_12종/50/compressed_{sc}_yolov3-12_sim.onnx", None)
           for sc in (8, 16, 32, 64)]
SWEEP_ALL = [(f"sw{r}sc{sc}", DRIVE / f"sweep_12종/{r}/compressed_{sc}_yolov3-12_sim.onnx", None)
             for r in (10, 30, 50) for sc in (8, 16, 32, 64)]
# 우리 재학습본 (프루닝 40% + trainvalno5k 10 에포크)
OURS = [("p40ep10", MODEL_DIR / "p40_ep10_clean.onnx", "images")]


def pick_calib_images(n: int, seed: int = 1234, d: Path | None = None) -> list[str]:
    """세 모델이 **같은 이미지**를 쓰도록 고정 선택."""
    d = d or CALIB_DIR
    files = sorted(p.name for p in d.glob("*.jpg"))
    if len(files) <= n:                       # 폴더가 작으면 전부 사용 (배포 재현용)
        return [str(d / f) for f in files]
    rng = np.random.default_rng(seed)
    return [str(d / f) for f in rng.choice(files, n, replace=False)]


def preprocess(path: str) -> np.ndarray:
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (SIZE, SIZE)).astype(np.float32) / 255.0
    return np.expand_dims(img.transpose(2, 0, 1), 0)


class Reader(CalibrationDataReader):
    def __init__(self, files: list[str], in_name: str):
        self.it = iter([{in_name: preprocess(f)} for f in files])

    def get_next(self):
        return next(self.it, None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", type=int, action="append", default=None,
                    help="캘리브레이션 장수 (여러 번 지정 가능)")
    ap.add_argument("--method", choices=("MinMax", "MovingAverage"), default="MinMax",
                    help="MovingAverage = 배포 파이프라인과 동일 (MinMax + EMA 0.01)")
    ap.add_argument("--calib-dir", default=None,
                    help="캘리브레이션 이미지 폴더 (기본: COCO val2017 무작위). "
                         "배포 재현: ~/demo_pruning/utils/calib_dataset")
    ap.add_argument("--tag", default="", help="출력 파일명 접미사")
    ap.add_argument("--sweep50", action="store_true", help="50% staircase 스윕 4종만 처리")
    ap.add_argument("--sweep-all", action="store_true", help="스윕 12종 전부")
    ap.add_argument("--ours", action="store_true", help="우리 재학습본")
    args = ap.parse_args()
    counts = args.calib or [20]

    # Neubla devkit 의 'MovingAverage' 는 별도 방식이 아니라 MinMax + 지수이동평균입니다:
    #   calibration_method = MinMax
    #   extra_options = {"CalibMovingAverage": True, "CalibMovingAverageConstant": 0.01}
    # EMA 계수 0.01 이라 이상치가 min/max 범위를 넓히지 못합니다.
    extra = ({"CalibMovingAverage": True, "CalibMovingAverageConstant": 0.01}
             if args.method == "MovingAverage" else {})
    suffix = ("ma" if args.method == "MovingAverage" else "mm") + args.tag

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for n in counts:
        files = pick_calib_images(n, d=Path(args.calib_dir) if args.calib_dir else None)
        print(f"\n════ {args.method} · 캘리브레이션 {n}장 (동일 이미지 집합) ════")
        _sel = OURS if args.ours else (SWEEP_ALL if args.sweep_all else (SWEEP50 if args.sweep50 else MODELS))
        for tag, path, in_name in _sel:
            if in_name is None:
                import onnx as _o; in_name = _o.load(str(path)).graph.input[0].name
            out = OUT_DIR / f"{tag}_u8_{suffix}{n}.onnx"
            if out.exists():
                print(f"  {tag:<10} 이미 존재 — 건너뜀"); continue
            t0 = time.time()
            quantize_static(
                model_input=str(path), model_output=str(out),
                calibration_data_reader=Reader(files, in_name),
                quant_format=QuantFormat.QOperator,
                per_channel=False,
                weight_type=QuantType.QUInt8,
                activation_type=QuantType.QUInt8,
                calibrate_method=CalibrationMethod.MinMax,
                extra_options=extra,
            )
            print(f"  {tag:<10} → {out.name}  "
                  f"({out.stat().st_size/1e6:.1f} MB, {time.time()-t0:.0f}s)", flush=True)
    for f in ("augmented_model.onnx",):
        if os.path.exists(f):
            os.remove(f)
    print(f"\n저장 위치: {OUT_DIR}")


if __name__ == "__main__":
    main()
