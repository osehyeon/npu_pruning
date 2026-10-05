#!/usr/bin/env python3
"""TABLE 3 의 channels, MACs, padded channels.

2026-09-17 · padded MACs(채널을 64 로 올려 계산한 연산량)와 그 상관표는 뺐다. paper/data/decode_o.py
가 .o 의 명령을 직접 해독해 보니 컴파일된 명령의 MACs 합이 13 종 모델 전부에서 저장된 MACs 와
정확히 같았다(비 1.0000). 입력 채널은 원형 그대로 명령에 실리고 출력 채널만 64 또는 나머지로
쪼개질 뿐, 64 로 올려 계산하는 연산은 어디에도 없다. padded MACs 는 산출물에 없는 모형값이므로
논문에 싣지 않는다. 이 개념을 다뤘던 설명 페이지와 상관 분석은
archive/neubla/padded-macs-concept/ 로 옮겼다. padded channels(타일 수 × 64)는 남긴다 — 이
값은 스케줄 기록의 타일 수, DRAM 버퍼 크기와 정확히 일치하는 실측값이다.

층마다 필요한 것은 (Cout, Cin, k, H, W) 다섯 값이다. 채널과 커널은 Conv 가중치 텐서의
모양 [Cout, Cin, kh, kw] 에서, 출력 특징맵의 높이와 폭은 onnx.shape_inference 가 채운
value_info 에서 읽는다. 손으로 전파하는 계산은 없다.

fp32 파일을 읽는 이유. uint8 파일은 QLinearLeakyRelu, QLinearAdd, QLinearConcat 이
onnx 패키지에 shape inference 가 없는 contrib 연산자라 첫 층 뒤로 크기가 채워지지 않는다.
양자화는 텐서 모양을 바꾸지 않으므로 fp32 파일의 모양이 곧 uint8 파일의 모양이다.
--check 를 주면 uint8 파일에서 stride 와 pad 로 직접 전파한 값과 대조한다.
2026-09-16 에 13 종 모두 75 층이 같음을 확인했다.

    python3 align_by_combo.py            # TABLE 3 값
    python3 align_by_combo.py --check    # uint8 직접 전파와 대조
"""
import math, sys
from pathlib import Path
import onnx
from onnx import shape_inference

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] / "npu_pruning"                  # 이 저장소 안의 실험 폴더
MODELS = ["original"] + [f"sw{r}sc{s}" for r in (10, 30, 50) for s in (8, 16, 32, 64)]


def fp32_path(m):
    if m == "original":
        return ROOT / "model/yolov3_original_fp32.onnx"
    r, s = m[2:].split("sc")
    return ROOT / f"paper/model/sweep/sweep_r{r}_sc{s}.onnx"


def conv_layers(path):
    """[(Cout, Cin, k, H, W)]. 모양은 전부 ONNX 에서 읽는다."""
    m = shape_inference.infer_shapes(onnx.load(str(path)))
    g = m.graph
    init = {t.name: t for t in g.initializer}
    shape = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
             for v in list(g.value_info) + list(g.output)}
    out = []
    for n in g.node:
        if n.op_type != "Conv":
            continue
        co, ci, kh, kw = init[n.input[1]].dims          # 가중치 [Cout, Cin, kh, kw]
        _, _, H, W = shape[n.output[0]]                 # shape inference 가 채운 출력 크기
        out.append((co, ci, kh, H, W))
    return out


def conv_layers_u8_manual(path):
    """대조용. uint8 파일에서 stride 와 pad 로 출력 크기를 직접 전파한다."""
    PASS = {"QLinearLeakyRelu", "QLinearAdd", "DequantizeLinear", "QuantizeLinear", "QLinearConcat"}
    g = onnx.load(str(path)).graph
    init = {t.name: t for t in g.initializer}
    hw = {g.input[0].name: (608, 608)}
    out = []
    for n in g.node:
        if n.op_type == "QLinearConv":
            H, W = hw[n.input[0]]
            co, ci, kh, kw = init[n.input[3]].dims
            at = {a.name: a for a in n.attribute}
            s = at["strides"].ints[0] if "strides" in at else 1
            p = at["pads"].ints[0] if "pads" in at else 0
            oh = (H + 2 * p - kh) // s + 1
            ow = (W + 2 * p - kw) // s + 1
            hw[n.output[0]] = (oh, ow)
            out.append((co, ci, kh, oh, ow))
        elif n.op_type == "Resize":
            H, W = hw[n.input[0]]
            hw[n.output[0]] = (H * 2, W * 2)
        elif n.op_type in PASS:
            src = next((i for i in n.input if i in hw), None)
            if src:
                hw[n.output[0]] = hw[src]
    return out


def macs(layers, A):
    return sum(math.ceil(co / A) * A * math.ceil(ci / A) * A * k * k * H * W
               for co, ci, k, H, W in layers)


def channels(layers, A):
    return sum(math.ceil(co / A) * A for co, ci, k, H, W in layers)


L = {m: conv_layers(fp32_path(m)) for m in MODELS}

if "--check" in sys.argv:
    for m in MODELS:
        u8 = conv_layers_u8_manual(ROOT / f"model/u8/{m}_u8_maD20.onnx")
        print(f"{m:10} {'같음' if u8 == L[m] else '다름'}  Conv {len(L[m])}")
    sys.exit(0)

print("TABLE 3.  channels · MACs(G) · padded channels  A = 64")
for m in MODELS:
    l = L[m]
    raw = macs(l, 1)
    print(f"{m:10} {channels(l,1):>7,} {raw/1e9:7.2f} {channels(l,64):>7,}")
