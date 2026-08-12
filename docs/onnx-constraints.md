# NPU 컴파일용 ONNX 작성 제약

> 작성일: 2026-08-02 · 관련: [neubla-isa.md](./neubla-isa.md)
> 공식 문서는 없습니다. ONNX Runtime EP 커널 등록표, 전처리 스크립트, 양자화 설정에서 복원했습니다.

---

## 먼저 — 병목은 하드웨어가 아니라 EP입니다

`nbc`가 직접 ONNX를 다 소화하는 게 아닙니다. 실제 경로는 이렇습니다:

```
내 ONNX  →  ONNX Runtime  →  NeublaExecutionProvider 가 그래프를 분할
                              (지원하는 노드만 떼어감)
                           →  NeublaExecutionProvider_NEUBLA_0_0_*.onnx  (서브그래프)
                           →  nbc  →  gen_inline_assembly.o
```

**그래서 "무엇을 쓸 수 있나"는 EP의 커널 등록표가 결정합니다.**
EP가 못 가져가는 노드는 그냥 CPU에 남고, NPU 구간이 잘게 쪼개져 성능이 무너집니다.

> 참고로 [ISA](./neubla-isa.md)에는 `sigmoid` `tanh` `transpose` `pad` `upsample` `softmax` 계열이
> 다 있지만, **EP가 노출하지 않습니다.** 하드웨어가 되는 것과 쓸 수 있는 것은 다릅니다.

---

## EP가 실제로 가져가는 연산 — 전부 23개

`libonnxruntime_providers_neubla.a` 의 커널 등록 심볼에서 추출한 완전한 목록입니다.

| 커널 | 도메인 | opset | 지원 dtype | 원래 ONNX 연산 |
|---|---|---:|---|---|
| `QLinearConv` | Onnx | 10 | u8/i32, u16/i32, i16/i64 | **Conv** |
| `QLinearConvTranspose` | Neubla | 1 | u8/i32, u16/i32, i16/i64 | ConvTranspose |
| `AntaraFusedConv` | Neubla | 1 | u8/i32, u16/i32, i16/i64 | *(컴파일러가 만드는 융합 노드)* |
| `QGemm` | MS | 1 | u8, u16, i16 | **Gemm** |
| `QLinearAdd` | MS | 1 | u8, u16, i16 | **Add** |
| `QLinearConcat` | MS | 1 | **u8 만** | **Concat** |
| `QLinearRelu` | Neubla | 1 | u8 | Relu |
| `QLinearLeakyRelu` | MS | 1 | u8, u16, i16 | **LeakyRelu** |
| `QLinearClip` | Neubla | 1 | u8, u16, i16 | Clip |
| `QLinearGelu` | Neubla | 1 | **u8 만** | Gelu |
| `QLinearHardSigmoid` | Neubla | 1 | **u8 만** | HardSigmoid |
| `QLinearSqrt` | Neubla | 1 | **u8 만** | Sqrt |
| `QLinearReduceMean` | Neubla | 1 | **u8 만** | ReduceMean |
| `QLinearAveragePool` | MS | 1 | u8, u16, i16 | AveragePool |
| `QLinearGlobalAveragePool` | MS | 1 | u8, u16, i16 | GlobalAveragePool |
| `MaxPool` | Onnx | **12** | u8, u16, i16 | MaxPool |
| `Reshape` | Onnx | **5** | u8, u16, i16 | Reshape |
| `Resize` | Onnx | **11** | **u8, i8 만** | Resize |
| `SpaceToDepth` | Onnx | **11** | u8, u16, i16 | SpaceToDepth |
| `DepthToSpace` | Onnx | **11** | u8, u16, i16 | DepthToSpace |
| `QLinearRescale` | Neubla | 1 | u8, u16, i16 | *(스케일 변환, 내부용)* |
| `MemcpyFromHost` / `MemcpyToHost` | Onnx | 1 | — | *(데이터 이동, 내부용)* |

### 여기 없는 것은 전부 CPU로 떨어집니다

자주 쓰는데 **지원 안 되는** 것들:

```
Mul  Sub  Div  Pow           ← Add 만 됩니다
Sigmoid  Tanh  Softmax  Elu  Swish/SiLU  Mish
Transpose  Pad  Slice  Split  Gather  Flatten  Squeeze/Unsqueeze
Upsample(deprecated)  BatchNormalization  InstanceNorm  LayerNorm
LSTM  GRU  Attention  MatMul(단독)
```

**`Sigmoid`/`SiLU`가 없다는 건 YOLOv5 이후 계열이 그대로는 안 올라간다는 뜻입니다.**
YOLOv3가 쓰이는 이유가 여기 있습니다 — Conv + LeakyRelu + Add + Concat + Resize 로만 이루어져 있습니다.

---

## Conv 작성 규칙

### `auto_pad` 는 쓸 수 없습니다 — 명시적 `pads` 로 바꿔야 합니다

전처리 스크립트에 그대로 박혀 있습니다 (`yolov3-preprocessing/update_attribute.py`):

```python
for node in graph.node:
    if node.op_type == "Conv":
        updated_attributes = update_attribute(updated_attributes, "auto_pad", "NOTSET")
        pads = determine_padding(kernel_shape)          # 3×3 → [1,1,1,1],  1×1 → [0,0,0,0]
        updated_attributes = update_attribute(updated_attributes, "pads", pads)
```

즉 **`auto_pad`를 `NOTSET`으로 강제하고 `pads`를 직접 계산해 넣습니다.**
`SAME_UPPER` / `SAME_LOWER` / `VALID` 를 쓴 모델은 이 변환을 먼저 거쳐야 합니다.

### 그 밖에

| 항목 | 제약 |
|---|---|
| **가중치 레이아웃** | `MCRSm` / `MCRScm` (컴파일러가 재배열). 입력은 `NCHW`, 내부는 `NCHWc` |
| **bias dtype** | u8/u16 활성화 → **i32 bias**, i16 활성화 → **i64 bias** (커널 등록에 고정) |
| **group** | 값 자체는 받지만 ISA에 `dwconv2d`가 있어 depthwise는 됩니다. group conv 일반형은 미확인 |
| **dilations / strides** | 명령어에 `d(...)`, `s(...)` 필드가 있어 지원됩니다 |

---

## 양자화 설정 — 여기서 벗어나면 안 됩니다

`quantize.py` / `quant_model.py` 의 설정이 그대로 하드 제약입니다:

```python
cfg.bit          = 8
cfg.per_channel  = False          # ← per-tensor 만. per-channel 불가
cfg.symmetric    = False          # ← asymmetric (uint8, zero-point 있음)
cfg.calib_method = "MovingAverage"
```

모델 파일명이 이 설정을 그대로 반영합니다:
`yolov3_..._opset12.neubla_u8_lwq_movingaverage.onnx`
(`u8` = uint8, `lwq` = layer-wise quant = per-tensor, `movingaverage` = 캘리브레이션 방식)

**per-channel 양자화를 쓰면 안 됩니다.** 정확도가 아쉬워도 per-tensor로 가야 합니다.

캘리브레이션 이미지는 예제에서 **1장**만 씁니다(`calib_img_cnt = 1`). 늘려도 되지만
`MovingAverage` 방식이라 순서에 영향을 받습니다.

---

## 전처리 파이프라인 — 이 순서를 지켜야 합니다

`yolov3-preprocessing/script.sh` 가 정본입니다.

```bash
# 1. BatchNormalization 제거 (Conv 로 folding)
onnxsim model.onnx model-onnxsim.onnx

# 2. 입력 텐서 크기 고정 — 동적 차원 불가
python3 shape_inference.py        # batch=1, C=3, H=608, W=608 로 못박고 shape_inference

# 3. Conv 속성 변경 — auto_pad → NOTSET + 명시적 pads
python3 update_attribute.py

# 4. NPU 로 보낼 구간만 잘라내기
python3 extract_model.py --model_input model-attribute.onnx \
                         --model_output model-backbone.onnx \
                         --node_input  input_1 \
                         --node_output convolution_output convolution_output1 convolution_output2

# 5. 양자화
python3 quantize.py
```

그리고 opset 변환 (`neubla_dev/scripts/prep_model.py`):

```python
model = onnx.version_converter.convert_version(model, 12)   # ← opset 12
model, ret = simplify(model)
```

### 각 단계가 왜 필요한가

| 단계 | 이유 |
|---|---|
| **onnxsim** | `BatchNormalization` 커널이 EP에 없습니다. Conv로 folding 해야 합니다 |
| **shape 고정** | 동적 차원(`batch`, `?`)이 있으면 타일링을 계산할 수 없습니다. **batch=1 고정** |
| **auto_pad 제거** | 위 참조 |
| **extract_model** | 전처리(정규화)·후처리(NMS, 앵커 디코딩)는 EP가 못 가져갑니다. 잘라내서 CPU에 남깁니다 |
| **opset 12** | `QLinearConv`가 opset 10, `MaxPool`이 12, `Resize`/`SpaceToDepth`가 11. **12로 맞추면 전부 커버** |

> opset 표기가 자료마다 다릅니다 — `prep_model.py`는 12, `0521_nota/readme.md`는 13이라고 적혀 있습니다.
> 모델 파일명이 `opset12`이므로 **12를 기준**으로 보는 게 안전합니다.

---

## 모델 구조 자체의 제약

| 항목 | 값 | 출처 |
|---|---|---|
| **배치 크기** | **1 고정** | `shape_inference.py`, 명령어의 타일 형상 |
| **출력 텐서 개수** | **최대 5개** | 펌웨어 `MAX_OFM_COUNT = 5U` |
| **SRAM** | **2 MB** | `nbc -sram 2097152`. 레이어 하나가 여기 안 들어가면 타일로 쪼개집니다 |
| **입력 해상도** | 제한은 없으나 SRAM 타일링에 직결 | 608×608 검증됨 |

SRAM 2MB가 실질적 병목입니다. 컴파일된 YOLOv3에서 `subtile` 이 가장 많이 등장하는 이유입니다
(표본 107회 — 전체 명령의 1/3). 채널 수가 크거나 해상도가 높으면 타일 수가 급증합니다.

`nbc` 로 탐색 폭을 조절할 수 있습니다:

```
-try_all_tiles      타일 개수를 전부 시도 (기본은 유효값 찾으면 중단)
-egs                exhaustive group search
-enable-lw-group    layer-wise 그룹화     → yolov3_lw.o
-tx_half_ofm_step   TX 명령을 half OFM step 으로 분할 → yolov3_half.o
```

---

## 요약 체크리스트

ONNX를 만들 때:

- [ ] **opset 12** 로 변환
- [ ] `onnxsim` 으로 **BatchNormalization 제거**
- [ ] **입력 shape 완전 고정**, batch = 1
- [ ] Conv의 **`auto_pad` 제거 → 명시적 `pads`**
- [ ] 연산을 **위 23개 커널 안에서만** 구성 (특히 `Sigmoid`/`Mul`/`Transpose`/`Split` 회피)
- [ ] 활성화는 **`Relu` / `LeakyRelu` / `Clip`** 위주로 (`Gelu`/`HardSigmoid`/`Sqrt`는 u8 전용)
- [ ] 전·후처리는 **모델에서 분리**해 CPU에 남기기
- [ ] 양자화는 **uint8 · per-tensor · asymmetric**
- [ ] 출력 텐서 **5개 이하**

---

## 확인하지 못한 것

- **`group` 속성의 일반 group conv 지원 여부** — ISA에 `dwconv2d`가 있어 depthwise는 되지만, group=N(중간값) 케이스는 검증 안 됨
- **`Resize` 의 허용 모드** — `coordinate_transformation_mode`, `nearest_mode` 중 어떤 조합이 되는지. ISA의 `upsample` 명령과 `AttrUpsample` 오퍼랜드 클래스는 존재
- **`MaxPool` 의 `ceil_mode` / `dilations` / `storage_order`** 허용 여부
- **최대 커널 크기 / 채널 수** 상한 — SRAM과 TX array(64×32)에서 오는 제약이 있을 텐데 수치 미확인
- EP가 노드를 거부할 때의 진단 방법 — `IsNodeSupported` / `GetSupportedNodes` 심볼은 있으나 로그 레벨 미확인
