#!/usr/bin/env python3
"""컴파일 옵션 격자 실측. LoadModel 간헐 실패에 재시도를 넣는다.

    python measure_grid.py sw50sc64 --combos lw lwhalf egs egshalf
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np, cv2, onnx, onnxruntime as ort
from NeublaDriver import NeublaDriver

HOME = Path.home(); EX = HOME / "workspace_docs/example"; SIZE = 608
IMAGE = HOME / "datasets/coco/images/val2017/000000000139.jpg"
REPEAT = 30

def sess(p):
    so = ort.SessionOptions(); so.log_severity_level = 3
    return ort.InferenceSession(str(p), so, providers=["CPUExecutionProvider"])

def front_of(mp: Path):
    m = onnx.load(str(mp)); n = m.graph.input[0].name; q = m.graph.node[0].output[0]
    f = Path(f"/tmp/{mp.stem}.front.onnx")
    if not f.exists(): onnx.utils.extract_model(str(mp), str(f), [n], [q])
    return sess(f), n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("--combo", required=True)
    a = ap.parse_args()
    stem = "original_maD20" if (a.model == "original" and a.combo == "lw") else a.model
    onnx_p = HOME / f"workspace_docs/model/u8/{a.model}_u8_maD20.onnx"
    bin_p = EX / f"{stem}_{a.combo}.o"
    if not bin_p.exists(): print(f"NONE\t{a.model}\t{a.combo}"); return

    bgr = cv2.imread(str(IMAGE)); rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    x = (cv2.resize(rgb, (SIZE, SIZE)).astype(np.float32)/255.0).transpose(2,0,1)[None]
    front, in_name = front_of(onnx_p)
    q = front.run(None, {in_name: x})[0].tobytes()

    d = NeublaDriver()
    assert d.Init() == 0
    for t in range(8):
        if d.LoadModel(str(bin_p)) == 0: break
        time.sleep(1)
    else:
        print(f"FAIL\t{a.model}\t{a.combo}"); return
    ts, lw = [], []
    for i in range(REPEAT + 2):
        t0 = time.perf_counter(); d.SendInput(q, 3*SIZE*SIZE)
        t1 = time.perf_counter(); d.Launch(); d.Wait()
        t2 = time.perf_counter(); d.ReceiveOutputs()
        t3 = time.perf_counter()
        if i >= 2: ts.append((t3-t0)*1000); lw.append((t2-t1)*1000)
    print(f"RESULT\t{a.model}\t{a.combo}\t{np.mean(ts):.2f}\t{np.std(ts):.2f}\t{np.mean(lw):.2f}")

if __name__ == "__main__": main()
