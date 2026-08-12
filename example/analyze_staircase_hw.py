#!/usr/bin/env python3
"""staircase 와 하드웨어의 관계 — 64 채널 정렬 가설 검증.

가설: Antara NPU 는 활성값·가중치 버퍼를 채널 64 배수로 패딩한다.
     그렇다면 실제 비용은 raw MACs 가 아니라 **패딩된 MACs** 여야 하고,
     채널이 이미 64 배수인 staircase-64 모델은 낭비가 0 이 된다.

        raw    MACs = Σ  Cout · Cin · k² · H · W
        padded MACs = Σ ⌈Cout/64⌉·64 · ⌈Cin/64⌉·64 · k² · H · W

3월 NPU latency 스윕(연계 문서 E02)과 상관을 비교한다.
raw 보다 padded 가 실측을 잘 설명하면 가설이 지지된다.
"""
from __future__ import annotations

import json, math
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

DRIVE = Path("/home/osehn/workspace_docs/YOLOv3_Antara_재현패키지/전처리/구글드라이브")

# 연계 문서 E02 — 2025.03.25 NPU latency 스윕 (초)
NPU_LAT = {
    (10, 8): 0.0562, (10, 16): 0.0561, (10, 32): 0.0561, (10, 64): 0.0558,
    (30, 8): 0.0524, (30, 16): 0.0505, (30, 32): 0.0466, (30, 64): 0.0474,
    (50, 8): 0.0462, (50, 16): 0.0463, (50, 32): 0.0463, (50, 64): 0.0413,
}
# 2026-08-12 우리 실측 (dbgprint OFF · maD20 · lw · 총 ms)
OURS_LAT = {
    (10, 8): 54.3, (10, 16): 54.2, (10, 32): 53.7, (10, 64): 53.7,
    (30, 8): 50.3, (30, 16): 48.8, (30, 32): 45.9, (30, 64): 46.2,
    (50, 8): 45.2, (50, 16): 45.4, (50, 32): 45.5, (50, 64): 40.8,
}

TARGETS = []
for r in (10, 30, 50):
    for sc in (8, 16, 32, 64):
        TARGETS.append((r, sc, DRIVE / f"sweep_12종/{r}/compressed_{sc}_yolov3-12_sim.onnx"))

ALIGN = 64


def analyze(path, size=608):
    m = onnx.load(str(path))
    init = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    hw = size
    raw = pad = 0
    act_raw = act_pad = 0
    chans = []
    for n in m.graph.node:
        if n.op_type == "Conv" and len(n.input) > 1 and n.input[1] in init:
            w = init[n.input[1]]
            st = next((a.ints[0] for a in n.attribute if a.name == "strides"), 1)
            hw = hw // st
            o, c, kh, kw = w.shape
            op = math.ceil(o / ALIGN) * ALIGN
            cp = math.ceil(c / ALIGN) * ALIGN
            raw += o * c * kh * kw * hw * hw
            pad += op * cp * kh * kw * hw * hw
            act_raw += o * hw * hw
            act_pad += op * hw * hw
            chans.append(o)
        elif n.op_type in ("Resize", "Upsample"):
            hw *= 2
    del m, init
    n64 = sum(1 for c in chans if c % ALIGN == 0)
    return dict(raw=raw, pad=pad, act_raw=act_raw, act_pad=act_pad,
                n_conv=len(chans), n_aligned=n64, channels=chans)


rows = []
base = None
print(f"{'ratio':>6}{'sc':>5}{'raw MACs(G)':>13}{'pad MACs(G)':>13}{'낭비%':>8}"
      f"{'64배수 conv':>12}{'활성 패딩%':>11}{'NPU s':>9}")
print("-" * 78)
for r, sc, p in TARGETS:
    if not p.exists():
        print(f"{r:>6}{str(sc or '-'):>5}  파일 없음"); continue
    a = analyze(p)
    if base is None:
        base = a
    waste = 100 * (a["pad"] / a["raw"] - 1)
    awaste = 100 * (a["act_pad"] / a["act_raw"] - 1)
    lat = OURS_LAT.get((r, sc))
    print(f"{r:>6}{str(sc or '-'):>5}{a['raw']/1e9:>13.1f}{a['pad']/1e9:>13.1f}"
          f"{waste:>8.1f}{a['n_aligned']}/{a['n_conv']:<8}{awaste:>11.1f}"
          f"{lat if lat else 0:>9.4f}")
    rows.append(dict(ratio=r, sc=sc, **{k: v for k, v in a.items() if k != "channels"},
                     npu_s=lat, npu_2503=NPU_LAT.get((r, sc))))

# 상관 분석
have = [x for x in rows if x["npu_s"]]
lat = np.array([x["npu_s"] for x in have])
old = np.array([x["npu_2503"] for x in have], float)
print(f"\n우리 실측 ↔ 2025-03 원측정  피어슨 r = {float(np.corrcoef(lat, old)[0,1]):+.4f}\n")
for key, label in (("raw", "raw MACs"), ("pad", "padded MACs"),
                   ("act_raw", "활성 raw"), ("act_pad", "활성 padded")):
    v = np.array([x[key] for x in have], float)
    r1 = float(np.corrcoef(v, lat)[0, 1]); r2 = float(np.corrcoef(v, old)[0, 1])
    print(f"{label:>14}  우리 r={r1:+.4f}   2025-03 r={r2:+.4f}")

Path("staircase_hw.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
print("\n→ staircase_hw.json")
