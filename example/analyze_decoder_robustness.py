#!/usr/bin/env python3
"""헤드 파라미터화가 양자화 견고성에 미치는 영향 — 오차 전파 분석.

원본과 Nota 프루닝 모델은 **채널 수와 헤드 디코딩이 동시에** 다릅니다.
따라서 "프루닝이 양자화에 강하다"고 말할 수 없습니다.

이 스크립트는 재학습 없이 **메커니즘만 분리**합니다:

  실제 모델의 wh 텐서를 uint8 로 양자화했을 때 생기는 오차를
    ① exp(t)              (표준 YOLOv3)
    ② 4·σ(t − ln3)        (Nota, = 2(tanh(t/2−0.549)+1))
  두 경로로 전파해 **박스 크기 상대오차 분포**를 비교합니다.

    python analyze_decoder_robustness.py --images 100
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

HOME = Path.home()
SIZE = 608
MODELS = {
    "원본":       HOME / "workspace_docs/model/yolov3_original_fp32.onnx",
    "40% 프루닝": HOME / "workspace_docs/model/yolov3_pruned40_fp32.onnx",
}


def dec_exp(t):
    return np.exp(np.clip(t, -60, 60))


def dec_nota(t):
    return 4.0 / (1.0 + np.exp(-np.clip(t - np.log(3.0), -60, 60)))


def quantize_dequantize(x: np.ndarray, bits: int = 8):
    """per-tensor asymmetric uint8 — 배포 파이프라인과 동일한 방식."""
    lo, hi = float(x.min()), float(x.max())
    n = (1 << bits) - 1
    scale = (hi - lo) / n if hi > lo else 1.0
    zp = round(-lo / scale)
    q = np.clip(np.round(x / scale) + zp, 0, n)
    return (q - zp) * scale, scale


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=100)
    ap.add_argument("--out", default="decoder_robustness.json")
    args = ap.parse_args()

    imgs = sorted(glob.glob(str(HOME / "datasets/coco/images/val2017/*.jpg")))[: args.images]
    so = ort.SessionOptions(); so.log_severity_level = 3

    report = {}
    print(f"이미지 {len(imgs)}장 · wh 채널만 추출해 uint8 양자화 오차를 전파합니다\n")

    for name, path in MODELS.items():
        m = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        inp = m.get_inputs()[0].name
        wh_all = []
        for p in imgs:
            x = (cv2.resize(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB), (SIZE, SIZE))
                 .astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
            for h in m.run(None, {inp: x}):
                # (1,255,H,W) → (1,3,85,H,W) 에서 wh 채널(2:4)만
                a = h.reshape(1, 3, 85, h.shape[-2], h.shape[-1])[:, :, 2:4]
                wh_all.append(a.ravel())
        wh = np.concatenate(wh_all).astype(np.float32)

        # 배포와 동일한 per-tensor uint8 양자화를 wh 텐서에 적용
        wh_q, scale = quantize_dequantize(wh)
        err_t = wh_q - wh                                   # 양자화로 생긴 t 오차

        rows = {}
        for dname, f in (("exp", dec_exp), ("nota", dec_nota)):
            ref, got = f(wh), f(wh_q)
            rel = np.abs(got - ref) / np.maximum(np.abs(ref), 1e-9)   # 박스 크기 상대오차
            rows[dname] = {
                "배수_최대": float(ref.max()),
                "상대오차_평균": float(rel.mean()),
                "상대오차_p99": float(np.percentile(rel, 99)),
                "상대오차_최대": float(rel.max()),
                "절대오차_최대_배수": float(np.abs(got - ref).max()),
            }

        report[name] = {
            "wh_min": float(wh.min()), "wh_max": float(wh.max()),
            "양자화_스텝": float(scale), "t_오차_최대": float(np.abs(err_t).max()),
            "decoders": rows,
        }

        print(f"── {name}")
        print(f"   wh 범위 [{wh.min():7.2f}, {wh.max():6.2f}]   "
              f"양자화 스텝 {scale:.4f}   t 오차 최대 {np.abs(err_t).max():.4f}")
        for dname in ("exp", "nota"):
            r = rows[dname]
            print(f"   {dname:<5} 최대배수 {r['배수_최대']:>12.1f}×   "
                  f"상대오차 평균 {r['상대오차_평균']*100:6.2f}%  "
                  f"p99 {r['상대오차_p99']*100:7.2f}%  최대 {r['상대오차_최대']*100:9.1f}%")
        print()

    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"저장: {args.out}")

    # 핵심 대비 한 줄
    print("\n" + "=" * 72)
    for name, r in report.items():
        e, n = r["decoders"]["exp"], r["decoders"]["nota"]
        print(f"{name:<10} exp 상대오차 p99 {e['상대오차_p99']*100:8.2f}%  vs  "
              f"nota {n['상대오차_p99']*100:6.2f}%   "
              f"({e['상대오차_p99']/max(n['상대오차_p99'],1e-12):.1f}배)")
    print("=" * 72)


if __name__ == "__main__":
    main()
