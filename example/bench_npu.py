#!/usr/bin/env python3
"""원본 / 40% / 50% 프루닝 모델의 NPU 실행 시간 비교.

프루닝 모델은 헤드 shape 은 같지만(255×19/38/76) **역양자화 노드 이름이 다르므로**
모델 파일에서 직접 읽어옵니다.

    ./run_bench.sh                      # 세 모델 (권장)
    ./run_bench.sh 40.o                 # 하나만

주의: **한 프로세스에서 두 번째 모델을 로드하면 DMA 오류로 보드가 망가집니다.**
Close() 를 불러도 마찬가지입니다. 그래서 run_bench.sh 가 모델마다 프로세스를 새로 띄웁니다.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper
from NeublaDriver import NeublaDriver

HOME = Path.home()
SIZE = 608
IMAGE = HOME / "demo_1127/yolov3/dog.jpg"
REPEAT = 10

# (이름, 양자화 ONNX, NPU 바이너리)
TARGETS = [
    ("원본",      HOME / "demo_1127/yolov3/yolov3_d53_mstrain-608_273e_coco_optim_opset12"
                         ".neubla_u8_lwq_movingaverage.onnx",
                  HOME / "demo_1127/yolov3/yolov3_lw.o"),
    ("40% 프루닝", HOME / "demo_0211/pruning_model/40.onnx", HOME / "demo_0211/pruning_model/40.o"),
    ("50% 프루닝", HOME / "demo_0211/pruning_model/50.onnx", HOME / "demo_0211/pruning_model/50.o"),
    ("40% lw+half", HOME / "demo_0211/pruning_model/40.onnx",
                    HOME / "workspace_docs/example/40_lwhalf.o"),
    ("40% egs",     HOME / "demo_0211/pruning_model/40.onnx",
                    HOME / "workspace_docs/example/40_egs.o"),
    ("50% egs",     HOME / "demo_0211/pruning_model/50.onnx",
                    HOME / "workspace_docs/example/50_egs.o"),
    ("원본 egs",    HOME / "demo_1127/yolov3/yolov3_d53_mstrain-608_273e_coco_optim_opset12"
                          ".neubla_u8_lwq_movingaverage.onnx",
                    HOME / "workspace_docs/example/orig_egs.o"),
    ("R 원본 lw",  HOME / "workspace_docs/model/u8/original_u8_maD20.onnx",
                   HOME / "workspace_docs/example/original_maD20_lw.o"),
    ("R 40% lw",   HOME / "workspace_docs/model/u8/pruned40_u8_maD20.onnx",
                   HOME / "workspace_docs/example/pruned40_maD20_lw.o"),
    ("R 50% lw",   HOME / "workspace_docs/model/u8/pruned50_u8_maD20.onnx",
                   HOME / "workspace_docs/example/pruned50_maD20_lw.o"),
    ("sw50 sc8 lw", HOME / "workspace_docs/model/u8/sw50sc8_u8_maD20.onnx",
                    HOME / "workspace_docs/example/sw50sc8_lw.o"),
    ("sw50 sc16 lw", HOME / "workspace_docs/model/u8/sw50sc16_u8_maD20.onnx",
                    HOME / "workspace_docs/example/sw50sc16_lw.o"),
    ("sw50 sc32 lw", HOME / "workspace_docs/model/u8/sw50sc32_u8_maD20.onnx",
                    HOME / "workspace_docs/example/sw50sc32_lw.o"),
    ("sw50 sc64 lw", HOME / "workspace_docs/model/u8/sw50sc64_u8_maD20.onnx",
                    HOME / "workspace_docs/example/sw50sc64_lw.o"),
    ("sw10 sc8 lw", HOME / "workspace_docs/model/u8/sw10sc8_u8_maD20.onnx",
                    HOME / "workspace_docs/example/sw10sc8_lw.o"),
    ("sw10 sc16 lw", HOME / "workspace_docs/model/u8/sw10sc16_u8_maD20.onnx",
                    HOME / "workspace_docs/example/sw10sc16_lw.o"),
    ("sw10 sc32 lw", HOME / "workspace_docs/model/u8/sw10sc32_u8_maD20.onnx",
                    HOME / "workspace_docs/example/sw10sc32_lw.o"),
    ("sw10 sc64 lw", HOME / "workspace_docs/model/u8/sw10sc64_u8_maD20.onnx",
                    HOME / "workspace_docs/example/sw10sc64_lw.o"),
    ("sw30 sc8 lw", HOME / "workspace_docs/model/u8/sw30sc8_u8_maD20.onnx",
                    HOME / "workspace_docs/example/sw30sc8_lw.o"),
    ("sw30 sc16 lw", HOME / "workspace_docs/model/u8/sw30sc16_u8_maD20.onnx",
                    HOME / "workspace_docs/example/sw30sc16_lw.o"),
    ("sw30 sc32 lw", HOME / "workspace_docs/model/u8/sw30sc32_u8_maD20.onnx",
                    HOME / "workspace_docs/example/sw30sc32_lw.o"),
    ("sw30 sc64 lw", HOME / "workspace_docs/model/u8/sw30sc64_u8_maD20.onnx",
                    HOME / "workspace_docs/example/sw30sc64_lw.o"),
    ("우리 40%ep10 lw", HOME / "workspace_docs/model/u8/p40ep10_u8_maD20.onnx",
                    HOME / "workspace_docs/example/p40ep10_lw.o"),
    ("sw50 sc64 egs", HOME / "workspace_docs/model/u8/sw50sc64_u8_maD20.onnx",
                    HOME / "workspace_docs/example/sw50sc64_egs.o"),
]

COCO80 = ("person bicycle car motorcycle airplane bus train truck boat traffic_light "
          "fire_hydrant stop_sign parking_meter bench bird cat dog horse sheep cow").split()


def sess(path) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def build_front(model_path: Path):
    """input → *_quantized 서브그래프. 입력 이름은 모델마다 다릅니다(input / input_1)."""
    m = onnx.load(str(model_path))
    in_name = m.graph.input[0].name
    q_out = m.graph.node[0].output[0]            # 첫 노드가 QuantizeLinear
    front = Path(f"/tmp/{model_path.stem}.front.onnx")
    if not front.exists():
        onnx.utils.extract_model(str(model_path), str(front), [in_name], [q_out])
    return sess(front), in_name


def dequant_params(model_path: Path):
    """출력 3개를 만드는 DequantizeLinear 에서 scale / zero_point 를 해상도별로 수집."""
    m = onnx.load(str(model_path))
    init = {i.name: i for i in m.graph.initializer}
    shape = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim] for v in m.graph.output}
    out = {}
    for n in m.graph.node:
        if n.op_type == "DequantizeLinear" and n.output[0] in shape:
            res = shape[n.output[0]][-1]                       # 19 / 38 / 76
            out[res] = (float(numpy_helper.to_array(init[n.input[1]])),
                        int(numpy_helper.to_array(init[n.input[2]])))
    return out


def main() -> None:
    bgr = cv2.imread(str(IMAGE))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    x = (cv2.resize(rgb, (SIZE, SIZE)).astype(np.float32) / 255.0).transpose(2, 0, 1)[None]

    only = sys.argv[1] if len(sys.argv) > 1 else None
    targets = [t for t in TARGETS if not only or only in str(t[2])]

    rows = []
    for name, onnx_path, bin_path in targets:
        if not bin_path.exists():
            print(f"{name}: 바이너리 없음 {bin_path}"); continue

        front, in_name = build_front(onnx_path)
        dq = dequant_params(onnx_path)
        q = front.run(None, {in_name: x})[0].tobytes()

        d = NeublaDriver()
        try:
            assert d.Init() == 0, "Init 실패"
            if d.LoadModel(str(bin_path)) != 0:
                print(f"{name}: LoadModel 실패 — 재시도하거나 hostreset 필요"); continue

            ts, ph = [], []
            for i in range(REPEAT + 1):                      # 1회는 워밍업
                t0 = time.perf_counter()
                assert d.SendInput(q, 3 * SIZE * SIZE) == 0
                t1 = time.perf_counter()
                assert d.Launch() == 0
                assert d.Wait() == 0                          # 실제 연산 구간
                t2 = time.perf_counter()
                raw = d.ReceiveOutputs()
                t3 = time.perf_counter()
                if i:
                    ts.append((t3 - t0) * 1000)
                    ph.append([(t1-t0)*1000, (t2-t1)*1000, (t3-t2)*1000])
        finally:
            d.Close()

        # 정확성 확인 — 역양자화 후 Nota 디코더로 검출을 뽑아 봅니다
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from decoders import decode_nota, nms_per_class
            heads = []
            for res, buf in zip((19, 38, 76), raw):
                sc, zp = dq.get(res, (1.0, 0))
                n = 255 * res * res
                a = (np.frombuffer(buf, np.uint8)[:n].astype(np.float32) - zp) * sc
                heads.append(a.reshape(1, 255, res, res))
            det = nms_per_class(decode_nota([heads[2], heads[1], heads[0]], score_thresh=0.4))
            top = [(COCO80[int(c)] if int(c) < len(COCO80) else int(c), round(float(sv), 2))
                   for *_, sv, c in det[:4]]
            print(f"{'':10} 검출 {len(det):2d}개 → {top}", flush=True)
        except Exception as e:
            print(f"{'':10} 검출 확인 실패: {e}", flush=True)

        ts = np.array(ts); ph = np.array(ph).mean(0)
        rows.append((name, ts.mean(), ts.std(), ts.min(), bin_path.stat().st_size / 1e6, ph))
        print(f"{name:<10} 총 {ts.mean():6.1f} ms  ±{ts.std():4.1f}   "
              f"[SendInput {ph[0]:5.1f} | Launch+Wait {ph[1]:5.1f} | Receive {ph[2]:5.1f}]  "
              f"출력 {sum(len(b) for b in raw)/1e6:.2f} MB", flush=True)

    for n, mu, sd, mn, sz, ph in rows:                  # 집계는 run_bench.sh 가 함
        print(f"RESULT\t{n}\t{mu:.2f}\t{sd:.2f}\t{mn:.2f}\t{sz:.1f}\t{ph[0]:.2f}\t{ph[1]:.2f}\t{ph[2]:.2f}")


if __name__ == "__main__":
    main()
