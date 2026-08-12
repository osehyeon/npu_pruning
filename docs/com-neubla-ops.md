# `com.neubla` 커스텀 연산자

> 작성일: 2026-08-02 · 관련: [onnx-constraints.md](./onnx-constraints.md), [neubla-isa.md](./neubla-isa.md)
> ONNX Runtime EP 커널 등록표와 **실제 EP 산출물 ONNX**에서 확인했습니다.

---

## 핵심 — 손으로 쓰는 연산자가 아니라 툴이 만들어내는 것

`com.neubla` 노드는 **모델에 직접 넣는 게 아닙니다.** 두 단계에서 자동 생성됩니다.

| 생성 시점 | 무엇이 나오나 | 이유 |
|---|---|---|
| **① 양자화** (`quantize.py`) | `QLinearRelu` 등 | ORT 표준/MS 도메인에 QLinear 판이 **없는 연산**을 자체 정의로 대체 |
| **② EP 그래프 융합** (`GetCapability` → `Compile`) | `AntaraFusedConv`, `QLinearRescale` | 여러 노드를 하나로 합치고 스케일을 맞춤 |

작성자는 **`ai.onnx` + `com.microsoft` 표준으로만** 모델을 쓰면 됩니다.

증거는 두 파일을 비교하면 바로 나옵니다.

**① 사람이 만든 양자화 모델** — `yolov3_..._u8_lwq_movingaverage.onnx`

```
ai.onnx        QLinearConv 75 · Gather 43 · Reshape 36 · Slice 18 · Add 16 ·
               Mul 13 · Concat 12 · Sigmoid 9 · Sub 9 · TopK 4 · Resize 2 ·
               NonMaxSuppression 1 · ScatterND 1 · ...
com.microsoft  QLinearLeakyRelu 72 · QLinearAdd 23 · QLinearConcat 2
com.neubla     ← 하나도 없음
```

**② EP가 뽑아낸 NPU 서브그래프** — `NeublaExecutionProvider_NEUBLA_0_0_yolov3.onnx`

```
com.neubla     AntaraFusedConv 72 · QLinearRescale 2
ai.onnx        QLinearConv 3
com.microsoft  QLinearConcat 2
```

**75개 QLinearConv 중 72개가 `AntaraFusedConv`로 바뀌었습니다.**
남은 3개는 YOLOv3의 최종 출력 conv — 뒤에 활성화가 없어서 융합할 게 없는 것들입니다.

즉 파이프라인은 이렇습니다:

```
표준 ONNX (ai.onnx + com.microsoft)
   → ORT가 로드
   → NeublaExecutionProvider.GetCapability()  : 가져갈 노드 판별
   → 그래프 융합                               : com.neubla 노드 생성  ← 여기
   → 서브그래프 ONNX 출력
   → nbc 컴파일                                 → .o
```

**작성자는 `ai.onnx` / `com.microsoft` 로만 쓰면 됩니다.** `com.neubla`는 EP의 내부 표현입니다.

---

## 등록된 `com.neubla` 커널 9개

`libonnxruntime_providers_neubla.a` 심볼에서 추출. 전부 `ver 1`입니다.

| 커널 | dtype | 성격 |
|---|---|---|
| **`AntaraFusedConv`** | u8+i32, u16+i32, i16+i64 | **융합 conv — 실제로 쓰이는 유일한 핵심 노드** |
| **`QLinearRescale`** | u8, u16, i16 | 양자화 스케일 재조정 |
| `QLinearConvTranspose` | u8+i32, u16+i32, i16+i64 | ConvTranspose (표준에 QLinear판이 없어 자체 정의) |
| `QLinearRelu` | u8 | 표준/MS 도메인에 없어 자체 정의 |
| `QLinearClip` | u8, u16, i16 | 〃 |
| `QLinearGelu` | u8 | 〃 |
| `QLinearHardSigmoid` | u8 | 〃 |
| `QLinearSqrt` | u8 | 〃 |
| `QLinearReduceMean` | u8 | 〃 |

아래 7개(`QLinearConvTranspose` ~ `QLinearReduceMean`)는 **ONNX 표준에도 `com.microsoft`에도
QLinear 버전이 없는 연산**이라 Neubla가 자체 정의한 것입니다.

### 실제 모델에서 관측된 것

워크스페이스의 ONNX 40개를 전수 스캔한 결과입니다.

| 노드 | 도메인 | 등장 모델 |
|---|---|---|
| `AntaraFusedConv` | **com.neubla** | EP 서브그래프 2개 |
| `QLinearRescale` | **com.neubla** | EP 서브그래프 2개 |
| `QLinearRelu` | **com.neubla** | `resnet50-..._u8_lwq_percentile.onnx` 에 **49개** |
| `QLinearConv` | ai.onnx | 23개 모델 |
| `QLinearConcat` | com.microsoft | 22개 모델 |
| `QLinearAdd` | com.microsoft | 21개 모델 |
| `QLinearLeakyRelu` | com.microsoft | 20개 모델 |
| `QGemm` · `QLinearGlobalAveragePool` | com.microsoft | 1개 모델 |

**`QLinearAdd` 는 `com.microsoft` 입니다** — 커널 등록도 `kMSDomain` 하나뿐이고,
실제 모델 21개 전부 `com.microsoft` 입니다. 다만 EP 서브그래프에서는
`AntaraFusedConv` 의 `fused_op` 로 흡수되어 사라지므로, 최종적으로는
**`com.neubla` 노드 안에 들어갑니다**(아래 융합 표 참조).

`QLinearRelu` 가 나오는 ResNet50 파일은 EP 산출물이 아니라 **양자화 산출물**입니다.
즉 `com.neubla` 노드는 EP 이전 단계에서도 생깁니다.

```
resnet50-0676ba61_opset12.neubla_u8_lwq_percentile.onnx
  ai.onnx        QLinearConv 53 · QuantizeLinear 1 · MaxPool 1 · Flatten 1 · ArgMax 1
  com.microsoft  QLinearAdd 16 · QLinearGlobalAveragePool 1 · QGemm 1
  com.neubla     QLinearRelu 49
```

`QLinearRelu` 노드는 입력 5개(`x_quantized`, `x_scale`, `x_zp`, `y_scale`, `y_zp`), attribute 없음입니다.

나머지 6개(`QLinearConvTranspose` `QLinearClip` `QLinearGelu` `QLinearHardSigmoid`
`QLinearSqrt` `QLinearReduceMean`)는 어느 모델에도 등장하지 않아 실동작 미검증입니다.

---

## `AntaraFusedConv` — 실체

### 노드 예시 (서브그래프에서 그대로 발췌)

```
[com.neubla] AntaraFusedConv   name=/model/n_conv2d_1/Conv_quant

inputs (11):
   0  input_1_quantized                            ← 활성화
   1  input_1_scale                        (init)
   2  input_1_zero_point                   (init)
   3  model.n_conv2d_1.weight_quantized    (init)  ← 가중치
   4  model.n_conv2d_1.weight_scale        (init)
   5  model.n_conv2d_1.weight_zero_point   (init)
   6  .../Conv_output_0_scale              (init)  ← conv 출력 스케일
   7  .../Conv_output_0_zero_point         (init)
   8  model.n_conv2d_1.bias_quantized      (init)  ← i32 bias
   9  /model/LeakyRelu_output_0_scale      (init)  ← 최종(활성화 후) 스케일
  10  /model/LeakyRelu_output_0_zero_point (init)

outputs: [/model/LeakyRelu_output_0_quantized]

attributes:
  fused_op          = ['QLinearConv', 'BiasAdd', 'QLinearLeakyRelu']
  activation_params = [0.10000000149011612]        ← LeakyRelu alpha
  auto_pad          = NOTSET
  kernel_shape      = [3, 3]
  pads              = [1, 1, 1, 1]
  strides           = [1, 1]
  dilations         = [1, 1]
  group             = 1
```

입력 11개 = **활성화 3 (양자화값·scale·zp) + 가중치 3 + conv출력 양자화 2 + bias 1 + 최종 양자화 2**.
`fused_op` 리스트가 무엇이 합쳐졌는지 그대로 알려줍니다.

### 관측된 융합 조합

YOLOv3 서브그래프의 72개 전부를 세면:

| 횟수 | `fused_op` | 추가 attribute |
|---:|---|---|
| **47** | `QLinearConv + BiasAdd + QLinearLeakyRelu` | — |
| **23** | `QLinearConv + BiasAdd + QLinearLeakyRelu + QLinearAdd` | — |
| **2** | `QLinearConv + BiasAdd + QLinearLeakyRelu + Resize` | `mode`, `coordinate_transformation_mode`, `scales` |

원본 모델의 `QLinearAdd` 23개, `Resize` 2개와 **정확히 일치**합니다.
즉 **residual add와 upsample이 앞선 conv 노드로 완전히 흡수**됩니다.

이게 [ISA](./neubla-isa.md)에서 본 `conv2d_bias{f}` 직후 `prelu` 융합 쌍의 정체입니다.

### `fuse.cc` 가 융합 대상으로 아는 연산

```
QLinearConv   QLinearConvTranspose   BiasAdd
QLinearLeakyRelu   QLinearRelu   QLinearClip
QLinearAdd   Resize   Concat
```

내부 플래그 이름도 융합 구조를 그대로 드러냅니다:

```
BIAS_ADDED_BIT              bias 합쳐짐
ACTIVATION_BIT              활성화 합쳐짐
ADDED_ACTIVATION_ELSUM_BIT  활성화 + elementwise sum(residual) 합쳐짐
ASHIFT_BIT / BSHIFT_BIT / ASHIFTED_OUT_BIT / BISHIFTED_BIT   양자화 시프트
CONV_OUT_BIT_4 / _8 / _16   conv 출력 비트폭
QUINT8 / QUINT16 / QINT8 / QINT16 / QINT32
AVGPOOL2D · CONVOLUTION · CONCAT · COMPARE · ARGMAX · ARGMIN
```

`CONV_OUT_BIT_4` 가 있는 걸 보면 **4비트 출력 경로도 하드웨어에 존재**하는 모양입니다
(현재 파이프라인은 8비트만 씁니다).

---

## `QLinearRescale` — 스케일 맞추기

```
[com.neubla] QLinearRescale   name=/model/Add_18_output_0_quantized_rescale

inputs (5):
   0  /model/Add_18_output_0_quantized       ← 입력 텐서
   1  /model/Add_18_output_0_scale     (init) ← 현재 스케일
   2  /model/Add_18_output_0_zero_point(init)
   3  /model/Concat_2_output_0_scale   (init) ← 목표 스케일
   4  /model/Concat_2_output_0_zero_point(init)

outputs: [/model/Add_18_output_0_quantized_rescaled]
attributes: (없음)
```

**양자화 스케일이 서로 다른 텐서를 `Concat` 하기 직전에 삽입**됩니다.
per-tensor 양자화라서 텐서마다 scale/zero-point가 다르고, 이어붙이려면 한쪽을 맞춰야 하기 때문입니다.

YOLOv3에 딱 2개 — `QLinearConcat` 2개와 짝입니다.

---

## 이 사실이 실무에 주는 의미

1. **`com.neubla` 노드를 직접 만들 필요도, 만들 수도 없습니다.** 모델은 `ai.onnx` + `com.microsoft` 표준으로 쓰고, 융합은 EP에 맡기면 됩니다.

2. **융합이 성능의 핵심입니다.** conv 하나에 bias·활성화·residual·upsample까지 흡수되어 SRAM 왕복이 사라집니다. **융합이 깨지는 그래프 형태를 피하는 것**이 최적화 포인트입니다:
   - conv 출력이 **여러 노드에 분기**하면 융합이 어려워집니다
   - conv와 활성화 사이에 다른 노드가 끼면 융합이 깨집니다
   - 활성화는 **LeakyRelu / Relu / Clip** 을 쓰세요 — `fuse.cc`가 아는 목록입니다

3. **`Resize`도 conv에 융합됩니다.** upsample을 별도 노드로 두는 것을 두려워할 필요 없습니다 — 단, 바로 앞이 conv일 때만입니다.

4. **서브그래프 ONNX를 열어보면 융합 결과를 직접 확인할 수 있습니다.** `fused_op` attribute를 세어보면 어디서 융합이 실패했는지 바로 보입니다.

```python
import onnx, collections
from onnx import helper
m = onnx.load("NeublaExecutionProvider_NEUBLA_0_0_yourmodel.onnx")
c = collections.Counter()
for n in m.graph.node:
    if n.op_type == "AntaraFusedConv":
        for a in n.attribute:
            if a.name == "fused_op":
                c[tuple(x.decode() for x in helper.get_attribute_value(a))] += 1
    else:
        c[(n.domain or "ai.onnx", n.op_type)] += 1
for k, v in c.most_common(): print(v, k)
```

---

## 확인하지 못한 것

- `QLinearClip` / `QLinearGelu` / `QLinearHardSigmoid` / `QLinearSqrt` / `QLinearReduceMean` / `QLinearConvTranspose` 는 **워크스페이스의 어느 모델에도 등장하지 않아 실동작 미검증**입니다. 커널은 등록되어 있습니다. (`QLinearRelu` 는 ResNet50 에서 확인됨)
- 양자화기가 어떤 규칙으로 `com.microsoft` 와 `com.neubla` 중 하나를 고르는지 — `Relu` 는 `com.neubla`, `Add`/`LeakyRelu`/`Concat` 은 `com.microsoft` 입니다. ORT 표준 contrib op 존재 여부로 갈리는 것으로 보이나 규칙을 코드로 확인하지는 못했습니다.
- 융합 실패 조건의 정확한 규칙 — `fuse.cc` 로직을 더 읽어야 합니다
- `CONV_OUT_BIT_4` (4비트 출력) 를 실제로 쓸 수 있는지, 어떤 경로로 켜는지
- `AntaraFusedConv` 가 받을 수 있는 `fused_op` 조합의 전체 집합 — 위 3가지는 YOLOv3에서 관측된 것뿐입니다
