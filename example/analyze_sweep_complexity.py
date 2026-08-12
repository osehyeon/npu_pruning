#!/usr/bin/env python3
"""프루닝율 × staircase 별 연산량·파라미터·CPU 지연 측정.

드라이브에 남아 있는 sweep 12종(10/30/50 % × sc 8/16/32/64)과 250514 3종(30/40/45 %),
그리고 최종본을 같은 기준으로 잽니다.

MACs 는 Conv 별 out×in×k×k×H_out×W_out 합입니다 (608 입력 기준).

    python analyze_sweep_complexity.py --latency 20
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

DRIVE = Path("/home/osehn/workspace_docs/YOLOv3_Antara_재현패키지/전처리/구글드라이브")
MODEL = Path("/home/osehn/workspace_docs/model")

TARGETS = [("원본", 0, None, DRIVE / "yolov3-12.onnx")]
for r in (10, 30, 50):
    for sc in (8, 16, 32, 64):
        TARGETS.append((f"sweep {r}%", r, sc,
                        DRIVE / f"sweep_12종/{r}/compressed_{sc}_yolov3-12_sim.onnx"))
for r in (30, 40, 45):
    TARGETS.append((f"250514 {r}%", r, 64,
                    DRIVE / f"250514_comp/{r}_compressed_64_yolov3-12_sim.onnx"))
TARGETS += [("최종 40%", 40, 64, MODEL / "yolov3_pruned40_fp32.onnx"),
            ("최종 50%", 50, 64, MODEL / "yolov3_pruned50_fp32.onnx")]


def stats(path, size=608):
    m = onnx.load(str(path))
    init = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    params = sum(a.size for a in init.values())

    # 공간 크기 추적: Conv stride 와 Resize(업샘플)만 반영
    hw, macs, ch = size, 0, []
    spatial = {}
    for n in m.graph.node:
        if n.op_type == "Conv" and len(n.input) > 1 and n.input[1] in init:
            w = init[n.input[1]]
            st = 1
            for a in n.attribute:
                if a.name == "strides":
                    st = a.ints[0]
            hw = hw // st
            o, c, kh, kw = w.shape
            macs += o * c * kh * kw * hw * hw
            ch.append(o)
            spatial[n.output[0]] = hw
        elif n.op_type in ("Resize", "Upsample"):
            hw *= 2
    del m, init
    return params, macs, sum(ch), len(ch)


def latency(path, n=20, size=608):
    import onnxruntime as ort
    so = ort.SessionOptions(); so.log_severity_level = 3
    s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    inp = s.get_inputs()[0]
    feed = {inp.name: np.random.rand(1, 3, size, size).astype(np.float32)}
    for o in s.get_inputs()[1:]:
        feed[o.name] = np.array([[size, size]], np.float32)
    for _ in range(3):
        s.run(None, feed)
    t = []
    for _ in range(n):
        t0 = time.perf_counter(); s.run(None, feed); t.append(time.perf_counter() - t0)
    return float(np.mean(t) * 1000), float(np.std(t) * 1000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latency", type=int, default=0, help="CPU 지연 측정 반복 수 (0=생략)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base = None
    rows = []
    print(f"{'모델':<13}{'ratio':>6}{'sc':>5}{'params(M)':>11}{'MACs(G)':>10}"
          f"{'채널':>8}{'MAC비':>8}" + (f"{'CPU ms':>10}{'±':>7}" if args.latency else ""))
    print("-" * (61 + (17 if args.latency else 0)))
    for name, r, sc, p in TARGETS:
        if not p.exists():
            print(f"{name:<13} 파일 없음: {p.name}"); continue
        pa, ma, chs, nc = stats(p)
        if base is None:
            base = ma
        row = {"name": name, "ratio": r, "staircase": sc, "params": pa,
               "macs": ma, "channels": chs, "n_conv": nc, "mac_ratio": ma / base}
        line = (f"{name:<13}{r:>6}{str(sc or '-'):>5}{pa/1e6:>11.2f}{ma/1e9:>10.1f}"
                f"{chs:>8}{ma/base:>8.3f}")
        if args.latency:
            mu, sd = latency(p, args.latency)
            row["cpu_ms"], row["cpu_sd"] = mu, sd
            line += f"{mu:>10.1f}{sd:>7.1f}"
        print(line, flush=True)
        rows.append(row)

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2, ensure_ascii=False))
        print(f"\n→ {args.out}")


if __name__ == "__main__":
    main()
