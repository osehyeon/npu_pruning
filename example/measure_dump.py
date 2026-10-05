#!/usr/bin/env python3
"""정상 구간만 취해 지연을 산출한다. 일부 바이너리는 격 회차마다 3초 타임아웃이 섞인다."""
import sys, time
from pathlib import Path
import numpy as np, cv2, onnx, onnxruntime as ort
from NeublaDriver import NeublaDriver
H=Path.home(); EX=H/"workspace_docs/example"; SIZE=608
tag, combo = sys.argv[1], sys.argv[2]
so=ort.SessionOptions(); so.log_severity_level=3
img=cv2.imread(str(H/"datasets/coco/images/val2017/000000000139.jpg"))
x=(cv2.resize(cv2.cvtColor(img,cv2.COLOR_BGR2RGB),(SIZE,SIZE)).astype(np.float32)/255
   ).transpose(2,0,1)[None]
mp=H/f"workspace_docs/model/u8/{tag}_u8_maD20.onnx"
m=onnx.load(str(mp)); n=m.graph.input[0].name; q0=m.graph.node[0].output[0]
f=Path(f"/tmp/{tag}.front.onnx")
if not f.exists(): onnx.utils.extract_model(str(mp),str(f),[n],[q0])
q=ort.InferenceSession(str(f),so,providers=["CPUExecutionProvider"]).run(None,{n:x})[0].tobytes()
d=NeublaDriver(); d.Init()
stem = "original_maD20" if (tag=="original" and combo=="lw") else tag
for t in range(8):
    if d.LoadModel(str(EX/f"{stem}_{combo}.o"))==0: break
else:
    print(f"FAIL\t{tag}\t{combo}"); sys.exit(1)
ts=[]
for i in range(200):
    t0=time.perf_counter(); d.SendInput(q,3*SIZE*SIZE); d.Launch(); d.Wait(); d.ReceiveOutputs()
    ts.append((time.perf_counter()-t0)*1000)
a=np.array(ts[2:])
np.save("/tmp/raw_timings.npy", a)
good=a[a<200]; bad=a[a>=200]
print(f"RESULT\t{tag}\t{combo}\t{good.mean():.2f}\t{good.std():.2f}\t{good.mean()-0.5:.2f}")
print(f"INFO\t정상 {len(good)}/{len(a)}" + (f" · 이상 중앙값 {np.median(bad):.0f} ms" if len(bad) else ""))
