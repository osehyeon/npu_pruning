# YOLOv3 float32 모델 (원본 / 프루닝)

> 작성일: 2026-08-02 · 관련: [../docs/onnx-constraints.md](../docs/onnx-constraints.md)

Nota AI 가 압축한 YOLOv3 를 압축률별로 모아둔 것입니다. **전부 양자화 전 float32** 입니다.

> 2026-09-16 · 프루닝 모델 둘은 `paper/model/final/` 의 파일과 바이트까지 같아 이 폴더의 사본을 지웠습니다.
> 이 폴더에 실물로 남은 fp32 는 BN 을 접은 `yolov3_original_fp32.onnx` 하나이고, 접지 않은 Model Zoo 원본은
> `paper/model/parent/yolov3-12_bn-unfolded.onnx` 입니다. 스크립트는 전부 새 경로로 고쳤습니다.

---

## 내용

| 파일 | 크기 | Conv | 총 채널 | 원본 대비 | md5 (앞 12자리) |
|---|---:|---:|---:|---:|---|
| `yolov3_original_fp32.onnx` | 247.7 MB | 75 | 27,069 | 100.0 % | `c4964902f76f` |
| `../paper/model/final/final_r40_sc64.onnx` | 81.5 MB | 75 | 17,021 | **62.9 %** | `d26ca1453abc` |
| `../paper/model/final/final_r50_sc64.onnx` | 59.0 MB | 75 | 14,781 | **54.6 %** | `1e5ca97e3a32` |

셋 다 **동일한 형태**입니다 — opset 12 · IR 7 · 노드 174개 · Conv 75개 ·
입력 `[1, 3, 608, 608]` · 출력 `356` `342` `309` (YOLO 헤드 3개).
구조는 같고 **채널 수만 줄어든** 구조적 프루닝입니다.

> 파일명의 40 % / 50 % 는 **압축률 표기**이고, 실제 채널 감소는 37.1 % / 45.4 % 입니다.
> 파라미터 수 기준일 가능성이 높습니다 (파일 크기는 각각 32.9 % / 23.8 % 로 줄어듦).

---

## 출처

```
yolov3_original_fp32.onnx  ←  demo_pruning/model_float32/yolov3.onnx
                               (= ONNX Model Zoo yolov3-12.onnx 의 BN 접기 판본.
                                원본 다운로드: https://github.com/onnx/models/blob/main/validated/vision/object_detection_segmentation/yolov3/README.md)
paper/model/final/final_r40_sc64.onnx  ←  demo_pruning/model_float32/compressed_yolov3_40%.onnx
paper/model/final/final_r50_sc64.onnx  ←  demo_0211/nota_model_0523/compressed_yolov3_50%.onnx
```

### 왜 이 파일들을 골랐나

워크스페이스에 같은 모델의 사본이 여러 벌 있는데 **바이트 단위로는 전부 다릅니다.**
크게 두 계열로 갈립니다:

| | 노드 | 특징 |
|---|---:|---|
| **174 노드** ← **이걸 담았습니다** | 174 | onnxsim 으로 상수 폴딩 완료. **shape 이 정적** |
| 206 노드 | 206 | `Shape` `Slice` `Floor` `Cast` `Unsqueeze` `Constant` 가 남아 있음 — upsample 크기를 입력 shape 에서 동적으로 계산 |

**NPU 파이프라인은 정적 shape 을 요구**하므로 174 노드 계열이 바로 쓸 수 있는 형태입니다
(출력 텐서는 양쪽 동일).

206 노드 계열이 필요하면:

```
원본  Downloads/250527_original_compressed_64_original_yolov3-12_sim.onnx   ← 이름과 달리 원본
40%   demo_0211/nota_model_0521/40%pruned_yolov3.onnx
50%   demo_0211/nota_model_0521/50%pruned_yolov3.onnx
```

---

## 프루닝 단위 — `step64` 가 핵심입니다

Nota 의 원본 파일명이 `yolov3_neubla_pruned40%_**step64**_retrain.onnx` 인데,
이 `64` 는 **채널을 64 의 배수 단위로 깎았다**는 뜻입니다.

같은 워크스페이스에 `compressed_8` 변형도 있어 비교하면 차이가 분명합니다:

| | 총 채널 | 원본 대비 | 64 배수가 아닌 Conv |
|---|---:|---:|---:|
| 원본 | 27,069 | 100.0 % | 5 / 75 |
| `compressed_8` | 24,965 | 92.2 % | **31 / 75** ← 정렬 깨짐 |
| `compressed_64` = 40 % | 17,021 | 62.9 % | 5 / 75 ← 정렬 유지 |

채널 변화를 직접 보면:

```
       원본   step8   step64
#26    512  →  496  →   384      (16개씩 vs 128개씩)
#27    256  →  256  →   192
```

**왜 64인가** — NPU 의 TX array 가 **64 × 32** 이고 SRAM word 도 **64 B** 입니다
([antara-npu-internals.png](../docs/antara-npu-internals.png)).
496 채널로 깎아봐야 하드웨어는 512칸을 쓰므로 **연산량은 그대로고 정확도만 손해**입니다.
이후 산출물이 전부 `step64` 계열인 이유입니다.

---

## 이 다음 단계

NPU 로 올리려면 [onnx-constraints.md](../docs/onnx-constraints.md) 의 파이프라인을 그대로 밟습니다.
이 파일들은 이미 opset 12 · onnxsim · 정적 shape 을 만족하므로 **양자화부터 시작**하면 됩니다.

```
① 양자화        uint8 · per-tensor · asymmetric · MovingAverage
② 전후처리 분리  extract_model 로 백본만 추출
③ nbc 컴파일    -tx_half_ofm_step 권장 (실측 최고, docs/nbc-compile-options.md)
④ llc -march=neubla → .o
```

40 % / 50 % 는 **양자화·컴파일 산출물이 이미 존재**합니다:

```
40%  uint8 20.5 MB  demo_pruning/model_uint8/compressed_yolov3_40%.onnx
                    demo_0211/pruning_model/40.o        ← NPU 바이너리
50%  uint8 14.9 MB  demo_0211/pruning_model/50.onnx
                    demo_0211/pruning_model/50.o        ← NPU 바이너리
```

> 프루닝 모델로 `run_npu.sh` 를 쓰려면 **`run_yolov3.py` 의 상수를 바꿔야 합니다** —
> 채널 수가 달라 `TARGET_DEQUANT_LAYER_NAMES` 와 `YOLO_HEAD_SPECS` 가 원본과 다릅니다.
