#!/usr/bin/env python3
"""스윕 12종의 GPU 지연 측정 — 정렬 효과가 하드웨어별로 다른지 확인.

CPU 에서는 raw MACs 가 지연을 잘 설명하고(r 0.947) staircase 64 가 가장 느리다.
NPU 에서는 반대로 staircase 64 가 가장 빠르다. GPU 는 그 사이 어디인가.

    python bench_gpu_sweep.py --provider cuda --repeat 30
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np
import onnxruntime as ort

SIZE = 608


def latency(path: Path, provider: str, repeat: int, warmup: int = 10, batch: int = 1):
    so = ort.SessionOptions()
    so.log_severity_level = 3
    prov = {"cuda": "CUDAExecutionProvider", "trt": "TensorrtExecutionProvider",
            "cpu": "CPUExecutionProvider"}[provider]
    s = ort.InferenceSession(str(path), so, providers=[prov, "CPUExecutionProvider"])
    used = s.get_providers()[0]
    inp = s.get_inputs()[0]
    shp = inp.shape
    dyn = not isinstance(shp[0], int)
    b = batch if dyn else 1
    feed = {inp.name: np.random.rand(b, 3, SIZE, SIZE).astype(np.float32)}
    for o in s.get_inputs()[1:]:
        feed[o.name] = np.array([[SIZE, SIZE]] * b, np.float32)
    for _ in range(warmup):
        s.run(None, feed)
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        s.run(None, feed)
        ts.append((time.perf_counter() - t0) * 1000)
    ts = np.array(ts)
    return float(ts.mean() / b), float(ts.std() / b), f"{used} b={b}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="스윕 ONNX 가 있는 디렉터리")
    ap.add_argument("--provider", choices=("cuda", "trt", "cpu"), default="cuda")
    ap.add_argument("--repeat", type=int, default=30)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.dir)
    rows = []
    print(f"{'ratio':>6}{'sc':>5}{'평균 [ms]':>12}{'표준편차':>10}  provider")
    print("-" * 48)
    for r in (10, 30, 50):
        for sc in (8, 16, 32, 64):
            p = root / f"{r}/compressed_{sc}_yolov3-12_sim.onnx"
            if not p.exists():
                print(f"{r:>6}{sc:>5}  파일 없음"); continue
            mu, sd, used = latency(p, args.provider, args.repeat, batch=args.batch)
            print(f"{r:>6}{sc:>5}{mu:>12.2f}{sd:>10.2f}  {used}", flush=True)
            rows.append({"ratio": r, "sc": sc, "ms": mu, "sd": sd, "provider": used})

    if len(rows) >= 4:
        # NPU / CPU 실측과 상관 비교
        NPU = {(10,8):54.3,(10,16):54.2,(10,32):53.7,(10,64):53.7,
               (30,8):50.3,(30,16):48.8,(30,32):45.9,(30,64):46.2,
               (50,8):45.2,(50,16):45.4,(50,32):45.5,(50,64):40.8}
        CPU = {(10,8):180.4,(10,16):223.4,(10,32):229.7,(10,64):227.8,
               (30,8):182.6,(30,16):185.1,(30,32):187.8,(30,64):187.5,
               (50,8):129.7,(50,16):133.1,(50,32):134.8,(50,64):145.6}
        g = np.array([x["ms"] for x in rows])
        n = np.array([NPU[(x["ratio"], x["sc"])] for x in rows])
        c = np.array([CPU[(x["ratio"], x["sc"])] for x in rows])
        print(f"\nGPU 대 NPU  피어슨 r = {np.corrcoef(g, n)[0,1]:+.4f}")
        print(f"GPU 대 CPU  피어슨 r = {np.corrcoef(g, c)[0,1]:+.4f}")
        for r in (10, 30, 50):
            sub = [x for x in rows if x["ratio"] == r]
            if len(sub) == 4:
                fast = min(sub, key=lambda x: x["ms"])
                print(f"  {r}% 최속 staircase = {fast['sc']}  ({fast['ms']:.2f} ms)")

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2, ensure_ascii=False))
        print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()
