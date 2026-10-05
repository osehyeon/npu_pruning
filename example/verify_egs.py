#!/usr/bin/env python3
"""NPU 바이너리의 출력을 CPU ONNX 기준과 직접 대조한다.

스윕 12종은 재학습 전 모델이라 mAP 로는 검증할 수 없다(정상 상태에서도 AP≈0).
따라서 출력 텐서를 수치로 비교한다. 벤더 verify 가 하려던 일과 같다.

    python verify_egs.py sw50sc64_lw.o sw50sc64_egs.o --onnx sw50sc64
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, cv2, onnx, onnxruntime as ort
from onnx import numpy_helper
from NeublaDriver import NeublaDriver

HOME = Path.home(); EX = HOME / "workspace_docs/example"; SIZE = 608
IMAGE = HOME / "datasets/coco/images/val2017/000000000139.jpg"

def sess(p):
    so = ort.SessionOptions(); so.log_severity_level = 3
    return ort.InferenceSession(str(p), so, providers=["CPUExecutionProvider"])

def front_of(model_path: Path):
    m = onnx.load(str(model_path)); in_name = m.graph.input[0].name
    q_out = m.graph.node[0].output[0]
    f = Path(f"/tmp/{model_path.stem}.front.onnx")
    if not f.exists(): onnx.utils.extract_model(str(model_path), str(f), [in_name], [q_out])
    return sess(f), in_name

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bins", nargs="+")
    ap.add_argument("--onnx", required=True, help="model/u8 의 stem · 예 sw50sc64")
    a = ap.parse_args()
    onnx_path = HOME / f"workspace_docs/model/u8/{a.onnx}_u8_maD20.onnx"

    bgr = cv2.imread(str(IMAGE)); rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    x = (cv2.resize(rgb, (SIZE, SIZE)).astype(np.float32) / 255.0).transpose(2, 0, 1)[None]

    front, in_name = front_of(onnx_path)
    q = front.run(None, {in_name: x})[0].tobytes()
    ref = [o.astype(np.float32) for o in sess(onnx_path).run(None, {in_name: x})]
    print(f"CPU 기준  {onnx_path.name}")
    for o in ref: print(f"  {str(o.shape):>22}  min {o.min():>9.3f}  max {o.max():>9.3f}")

    for b in a.bins:
        p = EX / b
        if not p.exists(): print(f"\n{b}: 없음"); continue
        d = NeublaDriver()
        try:
            if d.Init() != 0: print(f"\n{b}: Init 실패"); continue
            for _try in range(6):
                if d.LoadModel(str(p)) == 0: break
                print(f"  LoadModel 재시도 {_try+1}/6")
                import time as _t; _t.sleep(1)
            else:
                print(f"\n{b}: LoadModel 6회 실패"); continue
            assert d.SendInput(q, 3 * SIZE * SIZE) == 0
            assert d.Launch() == 0 and d.Wait() == 0
            raw = d.ReceiveOutputs()
        finally:
            pass   # Close() 는 라이브러리 이중 해제 버그가 있어 부르지 않는다
        bufs = list(raw) if isinstance(raw, (list, tuple)) else [raw]
        arrs = [np.frombuffer(b, dtype=np.uint8) for b in bufs]
        print(f"  버퍼 {len(arrs)}개 · 크기 {[a.size for a in arrs]}")
        print(f"  기대 원소 {[int(np.prod(o.shape)) for o in ref]}")
        ok = True
        for o in ref:
            n = int(np.prod(o.shape))
            cand = [a for a in arrs if abs(a.size - n) <= 256]
            if not cand:
                print(f"  {str(o.shape):>22}  대응 버퍼 없음"); ok = False; continue
            seg = cand[0][:n].astype(np.float32)
            r = o.reshape(-1)
            rn = (r - r.min()) / (r.ptp() + 1e-9); sn = (seg - seg.min()) / (seg.ptp() + 1e-9)
            c = float(np.corrcoef(rn, sn)[0, 1]) if seg.std() > 0 else 0.0
            print(f"  {str(o.shape):>22}  상관 {c:+.4f}  NPU min {seg.min():>4.0f} max {seg.max():>4.0f}")
            ok &= c > 0.99
        print("  →", "정상" if ok else "불일치")

if __name__ == "__main__": main()
