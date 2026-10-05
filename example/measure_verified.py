#!/usr/bin/env python3
"""회차마다 다른 입력을 넣고 출력이 갱신되는지 확인하면서 지연을 잰다.

    python measure_verified.py <tag> <combo> [reps]

출력이 직전 회차와 같으면 낡은 결과로 본다. 입력이 다르므로 같을 수 없다.
"""
import sys, time, hashlib, json
from pathlib import Path
import numpy as np, cv2, onnx, onnxruntime as ort
from NeublaDriver import NeublaDriver

H=Path.home(); EX=H/"workspace_docs/example"; SIZE=608
tag, combo = sys.argv[1], sys.argv[2]
REPS = int(sys.argv[3]) if len(sys.argv)>3 else 200
IMGDIR = H/"datasets/coco/images/val2017"

so=ort.SessionOptions(); so.log_severity_level=3
mp=H/f"workspace_docs/model/u8/{tag}_u8_maD20.onnx"
m=onnx.load(str(mp)); n=m.graph.input[0].name; q0=m.graph.node[0].output[0]
f=Path(f"/tmp/{tag}.front.onnx")
if not f.exists(): onnx.utils.extract_model(str(mp),str(f),[n],[q0])
front=ort.InferenceSession(str(f),so,providers=["CPUExecutionProvider"])

# 서로 다른 입력 8 종을 미리 양자화해 둔다
files=sorted(IMGDIR.glob("*.jpg"))[:8]
Q=[]
for p in files:
    img=cv2.imread(str(p))
    x=(cv2.resize(cv2.cvtColor(img,cv2.COLOR_BGR2RGB),(SIZE,SIZE)).astype(np.float32)/255
       ).transpose(2,0,1)[None]
    Q.append(front.run(None,{n:x})[0].tobytes())
print(f"입력 {len(Q)} 종 준비", flush=True)

d=NeublaDriver(); d.Init()
stem = "original_maD20" if (tag=="original" and combo=="lw") else tag
for t in range(8):
    if d.LoadModel(str(EX/f"{stem}_{combo}.o"))==0: break
else:
    print(f"FAIL\t{tag}\t{combo}"); sys.exit(1)

ts=[]; hs=[]; idx=[]
for i in range(REPS):
    k=i%len(Q)
    t0=time.perf_counter()
    d.SendInput(Q[k], 3*SIZE*SIZE); d.Launch(); d.Wait(); raw=d.ReceiveOutputs()
    dt=(time.perf_counter()-t0)*1000
    blob=b"".join(np.asarray(o).tobytes() for o in raw) if isinstance(raw,(list,tuple)) else np.asarray(raw).tobytes()
    ts.append(dt); hs.append(hashlib.md5(blob).hexdigest()[:12]); idx.append(k)

ts=np.array(ts); ts_,hs_,idx_ = ts[2:], hs[2:], idx[2:]
# 입력별 기준 해시 — 각 입력이 처음 낸 값
ref={}
for k,h in zip(idx_,hs_): ref.setdefault(k,h)
mismatch=[i for i,(k,h) in enumerate(zip(idx_,hs_)) if h!=ref[k]]
stale=[i for i in range(1,len(hs_)) if hs_[i]==hs_[i-1]]

good=ts_[ts_<200]
print(f"RESULT\t{tag}\t{combo}\t{good.mean():.2f}\t{good.std():.2f}")
print(f"INFO\t회차 {len(ts_)}  정상 {len(good)}  느린회차 {len(ts_)-len(good)}")
print(f"VERIFY\t고유 출력 {len(set(hs_))} (기대 {len(Q)})  "
      f"입력별 불일치 {len(mismatch)}  직전과 동일 {len(stale)}")
if mismatch: print(f"  불일치 회차 앞 10: {mismatch[:10]}")
if stale:    print(f"  낡은 회차 앞 10: {stale[:10]}")
np.save("/tmp/verified_ts.npy", ts_)
json.dump({"idx":idx_,"hash":hs_}, open("/tmp/verified_meta.json","w"))
