# ONNX → `.o` 실제 파이프라인

> 작성일: 2026-08-02 · 관련: [onnx-constraints.md](./onnx-constraints.md), [nbc-compile-options.md](./nbc-compile-options.md)
> 근거: `demo_pruning/*.sh`, `demo_pruning/utils/onnxquant.py`, `demo_0211/nbc/neubla/{nn_test,util}.py`

앞선 문서들이 "무엇이 제약인가"를 다뤘다면, 이 문서는 **실제로 어떤 명령이 돌아가는가**입니다.

---

## 전체 그림

```
학습 산출물 ONNX
   │
   │  ⓪ opset 12 변환        prep_model.py — onnx.version_converter
   │  ① onnxsim              BN folding + 상수 폴딩
   │  ② shape 고정            shape_inference.py — batch=1, 608×608 못박기
   │  ③ Conv 속성 수정        update_attribute.py — auto_pad 제거 + 명시적 pads
   │  ④ 전후처리 분리         extract_model.py — 백본만 남기고 NMS 등은 CPU 로
   ↓
float32 ONNX          ← workspace_docs/model/ 의 파일들이 여기 상태입니다
   │
   │  ⑤ onnxquant.sh  →  utils/onnxquant.py
   │     ORT quantize_static · QOperator · uint8 · per-tensor · 캘리브 20장
   ↓
uint8 ONNX
   │
   │  ⑥ compile.sh  →  dev/set_env.sh  →  neubla/nn_test.py  →  util.Inference()
   │     ONNX Runtime + NeublaExecutionProvider 세션 생성만으로 컴파일이 일어남
   │       └ EP 가 그래프 분할 → 서브그래프 → nbc → clang → asm
   ↓
gen_inline_assembly.o   →   이름 바꿔서 model_object/<모델>.o
```

**놀라운 점: 별도의 "컴파일 명령"이 없습니다.**
`ort.InferenceSession(...)` 을 만드는 것만으로 EP 가 내부에서 `nbc`/`clang`/`asm` 을 호출합니다.

---

## ⓪~④ 양자화 이전 전처리

정본은 `yolov3-preprocessing/script.sh` 입니다. 단계별 도구가 흩어져 있습니다:

| 단계 | 스크립트 | 하는 일 |
|---|---|---|
| ⓪ opset | `neubla_dev/scripts/prep_model.py` | `onnx.version_converter.convert_version(m, 12)` |
| ① onnxsim | `demo_0211/onnxsim.sh` · `prep_model.py` | `onnxsim in.onnx out.onnx` |
| ② shape 고정 | `yolov3-preprocessing/shape_inference.py` | dim_value 직접 대입 후 `infer_shapes` |
| ③ Conv 속성 | `demo_0211/utils/update_attribute.py` | `auto_pad` → `NOTSET`, `pads` 계산해 삽입 |
| ④ 분리 | `yolov3-preprocessing/extract_model.py` | `onnx.utils.extract_model` |

### 각 단계가 왜 필요한가

| 단계 | 이유 |
|---|---|
| **opset 12** | `QLinearConv` 가 opset 10, `MaxPool` 12, `Resize`/`SpaceToDepth` 11 — 12 로 맞추면 전부 커버 |
| **onnxsim** | EP 에 `BatchNormalization` 커널이 **없습니다.** Conv 로 folding 해야 함 |
| **shape 고정** | 동적 차원이 있으면 SRAM 2 MB 타일링을 계산할 수 없음. **batch=1 필수** |
| **auto_pad 제거** | 컴파일러가 `SAME_UPPER` 같은 동적 패딩 해석을 못 함 |
| **전후처리 분리** | NMS·앵커 디코딩은 EP 가 못 가져감 ([지원 연산 23개](./onnx-constraints.md)) |

### `update_attribute.py` 실제 코드

```python
for node in graph.node:
    if node.op_type == "Conv":
        updated = update_attribute(updated, "auto_pad", "NOTSET")
        pads    = determine_padding(kernel_shape)   # 3×3 → [1,1,1,1] · 1×1 → [0,0,0,0]
        updated = update_attribute(updated, "pads", pads)
```

### 현재 모델은 이미 끝난 상태입니다

`workspace_docs/model/` 의 세 파일을 실측한 결과:

```
노드 174개 · BatchNormalization 0개          ← onnxsim 완료
auto_pad  : 75개 Conv 전부 "속성 없음"        ← 제거 완료
pads      : (1,1,1,1) 38개 · (0,0,0,0) 37개  ← 3×3 / 1×1 커널과 정확히 대응
입력      : [1, 3, 608, 608] 정적            ← shape 고정 완료
출력      : raw 헤드 3개                      ← 전후처리 분리 완료
```

> `auto_pad` 가 **속성 자체로 없는** 이유 — ONNX 에서 `NOTSET` 은 기본값이라 저장 시 생략됩니다.
> 명시적 `pads` 가 있으므로 결과는 동일합니다.

**따라서 이 파일들은 ⑤ 양자화부터 바로 시작하면 됩니다.**

> ④ 는 조금 다릅니다 — 현재 모델은 출력이 raw 헤드 3개라 **이미 백본만 남은 상태**입니다.
> 그래서 mAP 측정 시 디코더를 따로 구현해야 했습니다
> ([pruning-map-comparison.md](./pruning-map-comparison.md), `example/decoders.py`).

---

## ⑤ 양자화

`demo_pruning/onnxquant.sh`

```bash
python3 utils/onnxquant.py \
    --input-model  model_float32/${MODEL_NAME}.onnx \
    --output-model model_uint8/${MODEL_NAME}.onnx \
    --calib-dataset utils/calib_dataset \
    --calib-count  20 \
    --input-name   ${INPUT_NAME}
rm -rf augmented_model.onnx
```

`utils/onnxquant.py` 의 핵심:

```python
quantize_static(
    model_input  = ...,
    model_output = ...,
    calibration_data_reader = ...,
    quant_format   = QuantFormat.QOperator,     # QDQ 아님 — QLinear* 연산자로 직접
    per_channel    = False,                     # per-tensor
    weight_type    = QuantType.QUInt8,
    activation_type= QuantType.QUInt8,
)
```

캘리브레이션 전처리 — **추론 때와 동일해야 합니다**:

```python
img = cv2.imread(p)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # BGR → RGB
img = cv2.resize(img, (608, 608))            # letterbox 아님, 단순 resize
img = img.astype(np.float32) / 255.0
img = np.transpose(img, (2, 0, 1))           # HWC → NCHW
img = np.expand_dims(img, axis=0)
```

> **`QuantFormat.QOperator` 가 중요합니다.** QDQ 형식이면 `QuantizeLinear`/`DequantizeLinear`
> 쌍이 흩뿌려지는데, EP 커널은 `QLinearConv` 같은 **QOperator 형식**을 받습니다.

### 모델마다 입력 이름이 다릅니다

```bash
./onnxquant.sh yolov3                input.1     # 원본
./onnxquant.sh compressed_yolov3_40% input_1     # 프루닝
```

`input.1` vs `input_1` — 내보낸 시점이 달라 이름이 어긋나 있습니다. 스크립트가 이걸 인자로 받는 이유입니다.

### 캘리브레이션 장수가 자료마다 다릅니다

| 스크립트 | `calib-count` |
|---|---:|
| `demo_pruning/onnxquant.sh` | **20** |
| `yolov3-preprocessing/quantize.py` | 1 |
| `neubla_dev/scripts/quant_model.py` | 1 |

20장 쪽이 나중에 정리된 것으로 보입니다.
`MovingAverage` 캘리브레이션은 **순서에 영향을 받으므로** 장수를 바꾸면 결과가 달라집니다.

---

## ⑥ 컴파일

`demo_pruning/compile.sh`

```bash
cd ../demo_0211/nbc/dev
source set_env.sh                    # ← 툴체인 환경

cd ../neubla
python3 nn_test.py \
    --model ../../../demo_pruning/model_uint8/${MODEL_NAME}.onnx \
    --input_name ${INPUT_NAME}

mv gen_inline_assembly.o ../../../demo_pruning/model_object/${MODEL_NAME}.o
```

### `set_env.sh` — 툴체인 환경

```bash
pyenv global 3.9.10                                     # ← Python 3.9 고정
export PATH=$TOP/tmp/neubla_mlp-{build,install}/bin:$PATH
export LD_LIBRARY_PATH=.../lib:.../cuda-11.8/lib64:.../libtorch/lib:...
export NEUBLA_BIN=$TOP/tmp/neubla_mlp-install/bin
export NISS_BUILD=$TOP/tmp/neubla_mlp-install/lib
```

`workspace_docs/onnxruntime-neubla/activate.sh` 가 이것과 같은 일을 합니다
(Python 3.9, 같은 라이브러리 경로) — 그때 CUDA 11.8 경로가 필요했던 이유가 여기 있습니다.

### `nn_test.py` — 20줄짜리 껍데기

```python
util.Inference(args.name, model_path, {args.input_name: input_path}, gs="lw")
                                                                     ^^^^^^^
```

**`gs="lw"` 가 `yolov3_lw.o` 의 정체입니다.**

### `util.Inference()` — 여기서 진짜 컴파일이 일어납니다

```python
ep_options = {
    "model":  name,
    "mode":   "niss",                  # 실행 모드 (niss | nrcl)
    "nbc":            cfg.nbc_exec_path,     # Neubla 백엔드 컴파일러
    "nbc.groupsearch": gs,                   # (egs | lw), 생략 시 built-in group
    "nbc.tx_ofm_step": tx_ofm_step,          # single (64ch/tx) | half (32ch/tx)
    "clang":          cfg.clang_exec_path,
    "asm":            cfg.asm_exec_path,     # Neubla 어셈블러
}

sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
sess_options.log_severity_level = 0                     # 전체 로그

ort_session = ort.InferenceSession(
    model_name, sess_options,
    providers=[("NeublaExecutionProvider", ep_options), "CPUExecutionProvider"]
)
outputs = ort_session.run(None, inputs)                 # 실행까지 해서 결과 비교용 데이터 생성
```

세션을 만드는 순간 EP 가 [`GetCapability` → 융합 → `Compile`](./com-neubla-ops.md) 을 돌리고,
그 안에서 `nbc` → `clang` → `asm` 이 차례로 불려 **`gen_inline_assembly.o`** 가 떨어집니다.

### 컴파일 옵션이 EP 옵션으로 전달됩니다

앞 문서에서 본 `nbc` CLI 플래그가 여기서는 **문자열 키**로 들어갑니다:

| EP 옵션 | 값 | 대응하는 `nbc` 플래그 | 산출물 |
|---|---|---|---|
| `nbc.groupsearch` | 생략 | (built-in group) | `yolov3_default.o` |
| `nbc.groupsearch` | `"lw"` | `-enable-lw-group` | `yolov3_lw.o` |
| `nbc.groupsearch` | `"egs"` | `-egs` | *(만들어진 적 없음)* |
| `nbc.tx_ofm_step` | `"single"` (기본) | — | 64 ch / tx |
| `nbc.tx_ofm_step` | `"half"` | `-tx_half_ofm_step` | `yolov3_half.o` |

**즉 세 변형은 이 두 옵션의 조합입니다.** `nn_test.py` 를 고치면 바로 다른 조합을 뽑을 수 있습니다:

```python
util.Inference(name, model_path, inputs, gs=None,  tx_ofm_step="half")   # half
util.Inference(name, model_path, inputs, gs="lw",  tx_ofm_step="half")   # lw + half (미시도)
util.Inference(name, model_path, inputs, gs="egs", tx_ofm_step="half")   # egs + half (미시도)
```

[실측](./nbc-compile-options.md)으로는 `half` 가 가장 빨랐으므로(60.0 ms), **`egs` + `half` 조합**이
다음 후보로 유력합니다.

### 두 모드를 동시에 돌립니다

```python
args = [
    (model_path, {..., "mode": "niss", "nbc": ..., "clang": ...}, inputs),   # 컴파일 + 시뮬
    (model_path, {"mode": "NRCL"}, inputs),                                  # 레퍼런스 실행
]
ths = [ThreadWithReturnValue(target=run, args=arg) for arg in args]           # 스레드 2개 병렬
...
verify.verify(trace_name, ref_path, elf_file_path, skip_nodes)               # 결과 대조
```

- **`niss`** — 컴파일하고 시뮬레이터로 실행 (`./niss/*.bin` 에 레이어별 출력)
- **`NRCL`** — 레퍼런스 구현으로 실행 (`./nrcl/*.bin`)
- 마지막에 `verify.verify()` 로 **레이어별 출력을 대조**해 컴파일이 맞았는지 검증

즉 `.o` 만 뽑는 게 아니라 **정확성 검증까지 한 번에** 합니다.
`neubla_dev_env/neubla_nene/`, `neubla_quant/nrcl/` 에 남아 있는 `.bin` 들이 그 산출물입니다.

`skip_nodes` 에 하드코딩된 이름이 있는 걸 보면 일부 노드는 대조에서 제외합니다:

```python
skip_nodes = ["input_608_quantized", "input_556_quantized", "_Concat_3_output_0_quantized"]
# 주석: for yolov3 egs best / for sidnet egs best
```

> 주석의 **`sidnet`** 은 이 워크스페이스에 없는 다른 모델입니다 —
> YOLOv3 외에도 컴파일해 본 이력이 있다는 뜻입니다.

---

## 요약 — 실행 순서

전처리(⓪~④)가 끝난 float32 ONNX 를 `demo_pruning/model_float32/` 에 두면
나머지는 스크립트 한 줄입니다.

```bash
# 한 번에 (양자화 + 컴파일)
cd ~/demo_pruning
./onnxquant_compile.sh compressed_yolov3_40%

# 또는 나눠서
./onnxquant.sh compressed_yolov3_40% input_1     # float32 → uint8
./compile.sh   compressed_yolov3_40% input_1     # uint8 → .o
```

산출물: `demo_pruning/model_object/compressed_yolov3_40%.o`

---

## 확인하지 못한 것

- `cfg.nbc_exec_path` 등이 정의된 `cfg` 모듈 내용 — 경로만 담고 있을 것으로 보입니다
- `mode` 의 `niss` / `nrcl` 이 정확히 무엇의 약자인지 (NISS = 시뮬레이터, NRCL = reference C library 로 추정)
- `verify.verify()` 의 판정 기준 (허용 오차)
- **컴파일 소요 시간** — 기록이 없습니다. `egs` 는 전수 탐색이라 오래 걸릴 수 있습니다
- `sidnet` 모델의 정체
