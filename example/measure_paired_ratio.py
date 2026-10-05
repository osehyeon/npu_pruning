#!/usr/bin/env python3
"""staircase 4종을 한 라운드 안에서 번갈아 재어 공통 드리프트를 상쇄한다.

    python measure_paired.py <ratio> <combo> [rounds] [reps]
"""
import sys, time, json, hashlib
from pathlib import Path
import numpy as np, cv2, onnx, onnxruntime as ort
from NeublaDriver import NeublaDriver

H=Path.home(); EX=H/"workspace_docs/example"; SIZE=608
ratio, combo = sys.argv[1], sys.argv[2]
ROUNDS = int(sys.argv[3]) if len(sys.argv)>3 else 10
REPS   = int(sys.argv[4]) if len(sys.argv)>4 else 20
SC=None
TAGS=["original","sw10sc64","sw30sc64","sw50sc64"]

so=ort.SessionOptions(); so.log_severity_level=3
imgs=sorted((H/"datasets/coco/images/val2017").glob("*.jpg"))[:2]
X=[(cv2.resize(cv2.cvtColor(cv2.imread(str(p)),cv2.COLOR_BGR2RGB),(SIZE,SIZE))
    .astype(np.float32)/255).transpose(2,0,1)[None] for p in imgs]

Q={}
for tag in TAGS:
    mp=H/f"workspace_docs/model/u8/{tag}_u8_maD20.onnx"
    m=onnx.load(str(mp)); n=m.graph.input[0].name; q0=m.graph.node[0].output[0]
    f=Path(f"/tmp/{tag}.front.onnx")
    if not f.exists(): onnx.utils.extract_model(str(mp),str(f),[n],[q0])
    s=ort.InferenceSession(str(f),so,providers=["CPUExecutionProvider"])
    Q[tag]=[s.run(None,{n:x})[0].tobytes() for x in X]
print(f"입력 준비 완료 · {len(TAGS)} 모델", flush=True)

d=NeublaDriver(); assert d.Init()==0
ts={t:[] for t in TAGS}; rnd={t:[] for t in TAGS}; hs={t:set() for t in TAGS}
for r in range(ROUNDS):
    for tag in TAGS:
        stem = "original_maD20" if (tag=="original" and combo=="lw") else tag
        for _ in range(8):
            if d.LoadModel(str(EX/f"{stem}_{combo}.o"))==0: break
        else: print(f"FAIL load {tag}"); sys.exit(1)
        v=[]
        for i in range(REPS+2):
            k=i%2; t0=time.perf_counter()
            d.SendInput(Q[tag][k],3*SIZE*SIZE); d.Launch(); d.Wait(); raw=d.ReceiveOutputs()
            dt=(time.perf_counter()-t0)*1000
            if i>=2 and dt<200:
                v.append(dt)
                blob=b"".join(np.asarray(o).tobytes() for o in raw) if isinstance(raw,(list,tuple)) else np.asarray(raw).tobytes()
                hs[tag].add(hashlib.md5(blob).hexdigest()[:8])
        ts[tag]+=v; rnd[tag].append(np.mean(v))
    print(f"  라운드 {r+1}/{ROUNDS}  " + "  ".join(f"{t}:{rnd[t][-1]:.2f}" for t in TAGS), flush=True)

print(f"\n=== ratio {ratio} % · {combo} · 라운드 {ROUNDS} × {REPS} 회 ===")
print(f"{'sc':>4}{'평균':>9}{'표본std':>9}{'라운드std':>10}{'표준오차':>9}{'고유출력':>9}")
R={}
for s,tag in zip(TAGS,TAGS):
    a=np.array(ts[tag]); b=np.array(rnd[tag]); R[s]=b
    print(f"{str(s):>10}{a.mean():9.2f}{a.std(ddof=1):9.3f}{b.std(ddof=1):10.3f}"
          f"{b.std(ddof=1)/np.sqrt(len(b)):9.3f}{len(hs[tag]):9d}")
print(f"\n원본 대비 지연 감소 · 라운드별로 짝지어 계산")
for s in TAGS[1:]:
    dif=(R["original"]-R[s])/R["original"]*100
    t=dif.mean()/(dif.std(ddof=1)/np.sqrt(len(dif)))
    print(f"  {s:<10} {dif.mean():+6.2f} % ± {dif.std(ddof=1)/np.sqrt(len(dif)):.2f}   paired t {t:+6.2f}")
json.dump({t:ts[t] for t in TAGS}, open(f"/tmp/pairedR_{combo}.json","w"))
