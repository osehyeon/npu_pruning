#!/usr/bin/env python3
"""교차 측정 — 모델을 번갈아 재어 보드 드리프트를 상쇄한다.

    python measure_interleaved.py egs 10 20 sw50sc8 sw50sc16 sw50sc32 sw50sc64
                                  조합 라운드 라운드당반복 모델...
"""
import sys, time
from pathlib import Path
import numpy as np, cv2, onnx, onnxruntime as ort
from NeublaDriver import NeublaDriver

H = Path.home(); EX = H/"workspace_docs/example"; SIZE = 608
combo, ROUNDS, REPS = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
TAGS = sys.argv[4:]
so = ort.SessionOptions(); so.log_severity_level = 3

img = cv2.imread(str(H/"datasets/coco/images/val2017/000000000139.jpg"))
x = (cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (SIZE, SIZE)).astype(np.float32)/255
     ).transpose(2,0,1)[None]

def quantized(tag):
    mp = H/f"workspace_docs/model/u8/{tag}_u8_maD20.onnx"
    m = onnx.load(str(mp)); n = m.graph.input[0].name; q0 = m.graph.node[0].output[0]
    f = Path(f"/tmp/{tag}.front.onnx")
    if not f.exists(): onnx.utils.extract_model(str(mp), str(f), [n], [q0])
    return ort.InferenceSession(str(f), so, providers=["CPUExecutionProvider"]).run(None, {n: x})[0].tobytes()

Q = {t: quantized(t) for t in TAGS}
d = NeublaDriver(); assert d.Init() == 0
rounds = {t: [] for t in TAGS}          # 라운드별 평균

for r in range(ROUNDS):
    for t in TAGS:
        for _ in range(8):
            if d.LoadModel(str(EX/f"{t}_{combo}.o")) == 0: break
        else:
            print(f"FAIL load {t}"); sys.exit(1)
        q = Q[t]; v = []
        for i in range(REPS + 2):
            t0 = time.perf_counter()
            d.SendInput(q, 3*SIZE*SIZE); d.Launch(); d.Wait(); d.ReceiveOutputs()
            dt = (time.perf_counter()-t0)*1000
            if i >= 2 and dt < 200: v.append(dt)
        rounds[t].append(np.mean(v))
    print("  라운드 %d/%d  " % (r+1, ROUNDS)
          + " ".join(f"{t}:{rounds[t][-1]:.2f}" for t in TAGS), flush=True)

ref = TAGS[-1]
R = {t: np.array(rounds[t]) for t in TAGS}
print(f"\n조합 {combo} · 라운드 {ROUNDS} · 라운드당 유효 반복 최대 {REPS}")
print("라운드 평균을 짝지어 비교한다. 같은 라운드 안의 반복은 독립이 아니므로")
print("라운드를 표본 단위로 삼는다.\n")
print(f"{'model':10}{'mean [ms]':>12}{'라운드 std':>12}{'vs '+ref:>12}{'paired t':>10}")
for t in TAGS:
    a = R[t]
    if t == ref:
        print(f"{t:10}{a.mean():12.3f}{a.std(ddof=1):12.3f}{'---':>12}{'---':>10}")
        continue
    dif = a - R[ref]
    tstat = dif.mean() / (dif.std(ddof=1)/len(dif)**.5)
    print(f"{t:10}{a.mean():12.3f}{a.std(ddof=1):12.3f}"
          f"{100*dif.mean()/a.mean():+11.2f}%{tstat:10.2f}")
print(f"\n자유도 {ROUNDS-1} · |t| 가 2.26 을 넘으면 5 % 수준에서 유의")
