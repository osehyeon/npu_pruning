# 관련 연구와 기여 위치

> 작성일 2026-08-24 · 조사자 메모
> 목적은 심사자가 "이미 알려진 것 아닌가" 라고 할 지점을 먼저 찾아 막는 것임

---

## 1. 우리 현상은 이미 문헌에 있음

### staircase 지연 — 같은 용어를 씀

**HALP · NeurIPS 2022 · Structural Pruning via Latency-Saliency Knapsack**
arXiv 2110.10811 의 제목이 `HALP: Hardware-Aware Latency Pruning` 이고 이것이 NeurIPS 2022 에
위 제목으로 게재됨. arXiv 2210.06659 가 게재본. 2026-09-10 원문 확인.
**전에 ICLR 2022 로 적었던 것은 틀림.** OpenReview 에 ICLR 투고 기록이 있어 생긴 오기임

> GPU usually imposes **staircase-shaped latency patterns** for convolutional operators
> with varying channels, which inevitably occur per varying pruning rate.
> This imposes a constraint that **pruning needs to be done in groups** to achieve
> latency improvement.

그리고 우리가 쓰는 `staircase` 파라미터와 정확히 같은 개념을 정의함.

> We refer to the difference of neurons counts between two latency cliffs of the
> staircase-patterned latency as the **latency step size**. In our method, we group
> `s` channels in a layer as an entirety, where the value of `s` is equal to the
> latency step size.

**HALP 의 `s` 가 Nota 도구의 `staircase` 임.** 이름만 다름.

측정 환경은 판본마다 다름. 2026-09-10 에 두 판본 원문을 대조함.

    arXiv 2021    TITAN V · V100 · Jetson TX2 · RTX 3080
    NeurIPS 2022  TITAN V · V100 · Jetson TX2 · Jetson Xavier · Intel CPU

RTX 3080 은 arXiv 판본의 TensorRT 부록 실험에 나옴. TITAN V 가 int8 Tensor Core 가속을
지원하지 않아 양자화 결과를 RTX 3080 에서만 보고한다고 적혀 있음. 게재본에는 이 절이 없음.
**게재본을 인용하면서 "전부 GPU" 라고 쓰면 안 됨. Intel CPU 가 포함됨.**

게재본에는 층마다 다른 group size 를 쓴다고 명시됨. ResNet-50 에서 23 개 층에 32,
20 개 층에 64, 10 개 층에 128 이고 입력에 가까울수록 작은 값임.
**본 연구의 staircase 는 전 층 단일 값이므로 이 차이를 관련 연구에서 다루어야 함.**

### 원 관측

**Radu et al. · Performance Aware CNN Channel Pruning for Embedded GPUs**
IISWC 2019, pp. 24--34. HALP 의 참고문헌 [51] 에서 확인. arXiv 2002.08697 은 프리프린트임

채널 수를 줄이며 지연을 재면 평탄하다가 특정 지점에서 뚝 떨어지는 계단이 나타나고,
**각 계단의 오른쪽이 최적 지점**이라고 관측함.

**Towards Latency-aware DNN Optimization with GPU Runtime Analysis and Tail Effect Elimination**
arXiv 2011.03897

> the inference latency of convolutional layers exhibits a **staircase pattern** along
> with channel number due to the **GPU tail effect**

> blindly DNN structure optimization may not always benefit the latency

### 정렬 패딩 — 벤더 문서에 명시돼 있음

| 하드웨어 | 정렬 | 출처 |
|---|---|---|
| Google TPU | 128 · 부차원 8 | Cloud TPU performance guide · OpenXLA tiled layout |
| Intel oneDNN | 16 · AVX512 · 8 · SSE4.1 | Understanding Memory Formats |
| NVIDIA INT8 | 16 · cuDNN 7.6.3+ 자동 패딩 | Convolutional Layers User's Guide |

oneDNN 문서의 설명이 우리와 동일함.

> round the channels up to make them multiples of the block size and **pad the
> resulting tail with zeros**

---

## 2. 그래서 이렇게 주장하면 안 됨

```
✗  채널 정렬로 패딩이 생긴다는 것을 발견했다
✗  압축률과 지연이 비례하지 않음을 발견했다
✗  staircase 라는 개념을 제안한다
```

전부 선행 연구가 이미 말한 것임. 특히 HALP 는 용어까지 같음.

---

## 3. 남는 기여

### 3.1 GPU 가 아니라 NPU 이고 기전이 다름

```
GPU   tail effect · SM 점유율 양자화 → staircase
NPU   SRAM 워드 64바이트 · OFM 타일 64채널 → padding
```

**원인이 다름.** GPU 는 스케줄링 단위, NPU 는 메모리 워드임.
NPU 에서의 정량 보고는 문헌이 얇음.

### 3.2 경향이 반대임

HALP 는 GPU 에서 이렇게 관측함.

> the staircase pattern becomes **less obvious as the number of input channel reduces**
> and the GPU has sufficient capacity for the reduced computation. This happens during
> pruning, especially for **large prune ratios**. In such a case, the FLOP count reflects
> the latency more accurately

**우리 NPU 는 반대임.** 압축률이 높을수록 정렬 효과가 커짐.

| 목표 압축률 | sc8 | sc64 | 차이 |
|---|---|---|---|
| 10 % | 54.3 | 53.7 | 1.1 % |
| 30 % | 50.3 | 46.2 | 8.2 % |
| 50 % | 45.2 | 40.8 | **9.7 %** |

GPU 는 압축하면 여유가 생겨 계단이 흐려지는데, NPU 는 워드 단위가 고정이라
채널이 줄수록 패딩 비율이 커짐. 낭비율로도 확인됨.

```
원본        23.8 %
50 % / sc8  62.1 %
50 % / sc64 40.2 %
```

**직접 대비되는 반례라 서술 가치가 있음.**

정렬된 Conv 수와 낭비율로 기전이 드러남.

| 목표 | sc | 낭비율 | 64배수 Conv | NPU ms | sc8 대비 |
|---|---|---|---|---|---|
| 원본 | — | 24.0 % | 70 / 75 | 58.4 | — |
| 10 % | 8 | 28.3 % | 44 / 75 | 54.3 | 0.0 % |
| | 16 | 28.2 % | 46 / 75 | 54.2 | 0.2 % |
| | 32 | 26.6 % | 60 / 75 | 53.7 | 1.1 % |
| | 64 | 25.5 % | 70 / 75 | 53.7 | 1.1 % |
| 30 % | 8 | 38.7 % | 25 / 75 | 50.3 | 0.0 % |
| | 16 | 39.2 % | 34 / 75 | 48.8 | 3.0 % |
| | 32 | 32.7 % | 58 / 75 | 45.9 | 8.7 % |
| | 64 | 30.9 % | 70 / 75 | 46.2 | 8.2 % |
| 50 % | 8 | **62.4 %** | **18 / 75** | 45.2 | 0.0 % |
| | 16 | 58.0 % | 32 / 75 | 45.4 | −0.4 % |
| | 32 | 52.3 % | 52 / 75 | 45.5 | −0.7 % |
| | 64 | **40.5 %** | **70 / 75** | 40.8 | **9.7 %** |

**원본은 이미 70 / 75 가 64 배수임.** 채널이 2의 거듭제곱으로 설계돼 있기 때문임.
프루닝이 이 정렬을 깨뜨림. sc8 은 50 % 에서 18 / 75 까지 떨어지고 낭비율이 62.4 %
로 치솟음.

**staircase 64 만 원본의 정렬을 보존함.** 압축률과 무관하게 70 / 75 를 유지함.

압축률이 오를수록 sc8 의 낭비가 커지므로 정렬 이득도 같이 커짐.
`1.1 % → 8.2 % → 9.7 %` 임. GPU 에서 계단이 흐려진다는 HALP 의 관측과 반대임.

**50 % 에서 sc16 과 sc32 는 sc8 보다 오히려 느림.** 낭비율은 낮은데 지연이 큼.
정렬만으로 설명되지 않는 구간이 있으며, 64 배수에 완전히 도달해야 이득이 나타남.

### 3.3 도구가 파라미터를 노출하지만 정답을 알려주지 않음

HALP 는 `s` 를 실측으로 학습해 정함. 상용 도구는 `staircase` 를 사용자가 고르게 함.

```
제공 값     8 · 16 · 32 · 64
정답        64 하나뿐
사용자      대응을 알 수 없음
```

**압축률만 보면 sc8 이 최선이고 sc64 가 최악임.** 실제 속도는 정반대임.

| 지표 | 50 % 에서 sc64 |
|---|---|
| 채널 총합 | 최대 · 불리 |
| MACs | 최대 · 불리 |
| fp32 크기 | 최대 · 불리 |
| uint8 크기 | 최대 · 불리 |
| 바이너리 크기 | **최소 · 유리** |
| NPU 지연 | **최소 · 유리** |

**지표와 실측이 정면으로 역전되는 사례**를 전 구간 격자로 제시할 수 있음.

### 3.4 컴파일러 내부 트레이스로 기전을 짚음

문헌은 대부분 블랙박스 지연 측정임. 우리는 컴파일러가 남기는 스케줄 트레이스를
유닛별로 분해했음.

```
padded MACs   실측과 r = 0.9315
컴파일러 추정  r = 0.9440
TX0 · 연산     r = 0.9753
```

그리고 `lw → egs` 에서 무엇이 줄어드는지 분리했음.

```
MEM 계열   3.64배 감소
TX 계열    0.99배 · 변화 없음
```

**`egs` 의 이득이 전부 메모리에서 나옴을 수치로 보임.** 이건 벤더 툴체인 접근 없이는
못 하는 분석임.

### 3.5 압축보다 컴파일 옵션이 큼

```
프루닝 50 %       1.43배
컴파일 lw → egs   1.73배
half 추가         1.26배
합쳐서            2.18배
```

**모델을 절반으로 줄이는 것보다 컴파일 설정을 바꾸는 것이 효과가 큼.**

FUPRUNER 계열이 융합과 프루닝을 함께 최적화하나, **둘의 상대적 크기를 격자로
비교한 보고는 못 찾았음.** 압축 논문은 대개 컴파일 설정을 고정한 채 보고함.

### 3.6 시스템 층위의 포화

선행 논문의 단계 분해를 그대로 쓰면 CPU 단계 15.20 ms 가 바닥임.
NPU 가 그 아래로 내려가면 더 압축해도 처리량이 안 늘어남.

```
필요한 가속   25.76 / 15.20 = 1.70배
얻은 가속     1.43배
```

**두 층위에서 같은 구조가 반복됨.** NPU 안에서는 메모리가, 시스템에서는 CPU 가
줄지 않는 부분으로 남음.

---

## 4. 초록에 넣을 문장 초안

> 구조적 프루닝의 압축률과 실제 지연이 비례하지 않는다는 것은 알려져 있다.
> 본 연구는 상용 NPU 와 벤더 압축 도구를 대상으로, 도구가 노출하는 채널 정렬
> 파라미터가 하드웨어의 메모리 워드 단위와 어긋날 때 무엇이 일어나는지를
> 12종 격자 실측으로 보인다. 파일 크기와 MACs 기준으로 가장 작은 모델이 가장
> 느렸으며, 정렬을 맞춘 모델은 연산량이 10 % 많음에도 11 % 빨랐다.
> 컴파일러 스케줄 트레이스를 유닛별로 분해하여 이 차이가 메모리 이동에서
> 비롯됨을 확인하였고, 압축보다 컴파일러 그룹 탐색 옵션의 영향이 더 크다는
> 것을 함께 보고한다.

---

## 5. 인용 목록

| 문헌 | 용도 |
|---|---|
| HALP · NeurIPS 2022 | staircase 개념 · latency step size · 층별 group size |
| Radu et al. · IISWC 2019 | staircase 원 관측 · 임베디드 GPU |
| Tail Effect · arXiv 2011.03897 | GPU tail effect 가 원인임 |
| FUPRUNER · arXiv 2010.16165 | 융합과 프루닝의 결합 |
| Cloud TPU performance guide | 128 · 8 정렬과 패딩 |
| OpenXLA tiled layout | 타일 레이아웃 |
| oneDNN Understanding Memory Formats | nChw16c · 제로 패딩 |
| NVIDIA Convolutional Layers User's Guide | INT8 16 정렬 · 자동 패딩 |
| Sensors 2025, 25, 1376 | 선행 논문 · 시스템 축 |

**미확인이라 인용하지 않을 것.** ARM Ethos-U 블록 깊이 · Qualcomm Hexagon 정렬 단위.
