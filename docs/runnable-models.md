# Antara NPU 로 돌릴 수 있는 모델

> 작성일: 2026-08-02 · 근거: [onnx-constraints.md](./onnx-constraints.md), [com-neubla-ops.md](./com-neubla-ops.md)

EP 커널 23개를 기준으로 대표 모델들을 분류했습니다.

---

## 판정 기준

**쓸 수 있는 연산** (EP 커널 전부)

```
Conv  ConvTranspose  Gemm                    (QLinearConv / QLinearConvTranspose / QGemm)
Add   Concat                                 (QLinearAdd / QLinearConcat)
Relu  LeakyRelu  Clip  Gelu  HardSigmoid     활성화
Sqrt  ReduceMean
MaxPool  AveragePool  GlobalAveragePool
Reshape  Resize  SpaceToDepth  DepthToSpace
```

**없어서 발목 잡는 것** — 이 넷이 사실상 모든 판정을 좌우합니다

| 없는 연산 | 무엇이 죽는가 |
|---|---|
| **`Mul`** | SE 블록, SiLU/Swish, hard-swish, 모든 게이팅 |
| **`Sigmoid`** | SiLU, 어텐션 |
| **`Softmax` · `LayerNorm` · `MatMul`** | 트랜스포머 전부 |
| **`Transpose`** | channel shuffle, 트랜스포머 |

그 밖에 `Sub` `Div` `Pow` `Tanh` `Split` `Slice` `Gather` `Pad` `LSTM` 도 없습니다.

**구조 제약**: batch=1 고정 · 동적 shape 불가 · 출력 텐서 5개 이하 · uint8 per-tensor 양자화 · SRAM 2 MB

> **전·후처리는 CPU 로 빼면 됩니다.** YOLOv3 가 그렇게 돕니다 —
> 앵커 디코딩·NMS 는 back ONNX 로 분리해 CPU 에서 실행합니다.
> 그래서 **백본이 통과하는지**가 실질적 판정 기준입니다.

---

## ✅ 실증됨 — 컴파일된 바이너리가 있음

| 모델 | 근거 |
|---|---|
| **YOLOv3** (Darknet-53) | `demo_1127/yolov3/yolov3_{default,half,lw}.o` · 오늘 실행 확인 (60~96 ms) |
| **ResNet-50** | `resnet50_demo/resnet/resnet50_neubla_ori_lw.o` + 데모 일체 |

ResNet-50 양자화 모델의 실제 구성 — **전부 지원 범위 안**입니다:

```
QLinearConv 53 · QLinearRelu 49 · QLinearAdd 16
QLinearGlobalAveragePool 1 · MaxPool 1 · QGemm 1
Flatten 1 · ArgMax 1        ← 이 둘은 CPU 로 떨어짐 (문제 없음)
```

---

## ✅ 될 가능성 높음 — 연산만 보면 전부 통과

### 분류 (Classification)

| 모델 | 논문 | 쓰는 연산 | 비고 |
|---|---|---|---|
| **ResNet-18/34/101/152** | He 2016 | Conv·ReLU·Add·MaxPool·GAP·Gemm | R50 과 동일 구조 |
| **VGG-16/19** | Simonyan 2014 | Conv·ReLU·MaxPool·Gemm | 가중치 138 M → DRAM 부담 큼 |
| **SqueezeNet 1.0/1.1** | Iandola 2016 | Conv·ReLU·MaxPool·Concat | 5 MB 급, 가장 가벼움 |
| **Inception-v3** | Szegedy 2016 | Conv·ReLU·Pool·Concat | BN 은 onnxsim 이 folding |
| **DenseNet-121/169** | Huang 2017 | Conv·ReLU·Concat·AvgPool | Concat 매우 많음 → SRAM 압박 |
| **MobileNetV1** | Howard 2017 | dwConv·ReLU6(=Clip) | ISA 에 `dwconv2d` 있음 |
| **MobileNetV2** | Sandler 2018 | dwConv·ReLU6·Add | inverted residual, **`Mul` 안 씀** |
| **LeNet-5** | LeCun 1998 | Conv·Pool·Gemm | 자명하게 통과 |

### 검출 · 분할 (백본 + CPU 헤드)

| 모델 | 논문 | 비고 |
|---|---|---|
| **YOLOv3-tiny** | Redmon 2018 | YOLOv3 와 동일 연산, 훨씬 가벼움 |
| **U-Net** | Ronneberger 2015 | `QLinearConvTranspose` 지원 · skip 은 Concat |
| **SSD (VGG/ResNet 백본)** | Liu 2016 | 백본만 NPU, 앵커·NMS 는 CPU |
| **RetinaNet · Faster R-CNN** | Lin 2017 / Ren 2015 | 〃 (RoIAlign 은 CPU) |
| **FCN · DeepLabv3** | Long 2015 / Chen 2017 | ASPP 는 Conv+Concat · upsample 은 `Resize` |

> 이 목록은 **연산자 관점 판정**입니다. SRAM 2 MB 타일링과 가중치 크기 때문에
> 실제로는 컴파일이 실패하거나 느릴 수 있습니다. `nbc` 로 직접 돌려봐야 확정됩니다.

---

## ❌ 안 되는 것 — 이유가 명확함

### `Mul` 이 없어서

| 모델 | 막히는 지점 |
|---|---|
| **EfficientNet B0~B7** (Tan 2019) | SiLU(`x * sigmoid(x)`) + SE 블록(채널별 `Mul`) — **이중으로** 막힘 |
| **MobileNetV3** (Howard 2019) | hard-swish = `x * ReLU6(x+3)/6` |
| **SENet · SE-ResNet · RegNetY** (Hu 2018) | SE 블록의 채널 재조정 `Mul` |
| **YOLOv5 / v6 / v7 / v8 / v9** | SiLU 활성화 |
| **YOLOv4** | Mish = `x * tanh(softplus(x))` |

> **이 워크스페이스가 계속 YOLOv3 를 쓰는 이유입니다.**
> YOLOv5 이후는 전부 SiLU 라 그대로는 못 올립니다.

### `Softmax` · `LayerNorm` · `MatMul` · `Transpose` 가 없어서

| 모델 | |
|---|---|
| **ViT · DeiT · Swin Transformer** | 어텐션 전체 |
| **BERT · GPT 계열** | 〃 |
| **DETR** | 〃 |
| **ConvNeXt** (Liu 2022) | LayerNorm (GELU 는 지원되는데 LayerNorm 이 없음) |

> `QLinearGelu` 가 있는 걸 보면 트랜스포머를 염두에 뒀던 것 같은데,
> Softmax·LayerNorm·MatMul 이 없어 실제로는 못 돌립니다.

### 기타

| 모델 | 막히는 지점 |
|---|---|
| **ShuffleNet V1/V2** | channel shuffle = Reshape+**Transpose**+Reshape |
| **LSTM · GRU 계열** | 순환 연산 자체가 없음 |
| **Mask R-CNN** | RoIAlign (헤드를 CPU 로 빼면 백본은 가능) |
| **AlexNet** | 원본의 LRN. LRN 없는 변형이면 통과 |

---

## 추천 순서

지금 워크스페이스 자산으로 바로 해볼 만한 것부터:

1. **ResNet-18 / ResNet-34** — R50 이 이미 되므로 거의 확실. 더 가벼워 SRAM 여유
2. **MobileNetV2** — 경량 모델이 실제로 이득인지 확인하는 좋은 지표. dwconv 성능이 관건
3. **SqueezeNet** — 5 MB 급이라 가중치 전송 병목이 사라진 조건에서의 상한 측정용
4. **YOLOv3-tiny** — 검출 파이프라인을 그대로 재사용 가능
5. **U-Net** — `QLinearConvTranspose` 가 실제로 동작하는지 확인 (미검증 커널)

경로는 [onnx-constraints.md](./onnx-constraints.md) 의 전처리 파이프라인 그대로입니다:
opset 12 → onnxsim → shape 고정 → auto_pad 제거 → 전후처리 분리 → uint8 per-tensor 양자화.

---

## 확인하지 못한 것

- **group conv 일반형** (`group` 이 1 도 아니고 채널 수도 아닌 경우) — ResNeXt · 일부 경량 모델이 여기 걸림. ISA 에 `dwconv2d` 는 있지만 중간값 group 은 미검증
- **`QLinearConvTranspose`** — 커널은 등록돼 있으나 실제 모델에서 동작 확인 못 함 (U-Net 이 시금석)
- **`Resize` 의 허용 모드** — `coordinate_transformation_mode` / `nearest_mode` 조합
- 위 "될 가능성 높음" 목록 중 **실제로 컴파일·실행해 본 것은 하나도 없습니다.** 연산자 판정일 뿐입니다
