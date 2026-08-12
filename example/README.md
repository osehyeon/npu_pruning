# YOLOv3 실행 예제

> 작성일: 2026-08-02 · 관련: [../docs/why-docker-for-demos.md](../docs/why-docker-for-demos.md)

`demo_1127/yolo_demo.py`(480줄, 비디오·카메라·GUI 포함)에서 **핵심 경로만 뽑아
스스로 진단 가능하게** 다시 쓴 이미지 한 장짜리 예제입니다.

---

## 바로 해보기

```bash
cd ~/workspace_docs/example

# 1) 무엇이 막고 있는지 먼저 확인 (하드웨어 안 건드림)
python3 run_yolov3.py --mode check

# 2) NPU 없이 CPU 로만 실행 — 지금 바로 됩니다
python3 run_yolov3.py --mode cpu
```

### 검증된 결과

```
[  input] dog.jpg  768x576 → 608x608
[    cpu] 추론 완료 98 ms
[    cpu] 검출 3개 (conf ≥ 0.5) · 98.2 ms → result.jpg
```

`result.jpg` 에 `bicycle 1.00` · `dog 0.99` · `truck 0.86` 이 그려집니다.

---

## 모드

| 모드 | 하는 일 | 하드웨어 |
|---|---|:---:|
| `check` | 환경·자산 점검만. 무엇이 막는지 전부 나열 | 불필요 |
| `cpu` | 원본 양자화 ONNX 를 CPU 로 통째로 실행 (기준값) | 불필요 |
| `npu` | front(CPU) → NPU → back(CPU) 3단 실행 | **필요** |
| `both` | 둘 다 실행하고 라벨 일치도·속도 비교 | **필요** |

```bash
python3 run_yolov3.py --mode cpu --image 내사진.jpg --out 결과.jpg --conf 0.3
```

---

## 추론 경로

`--mode npu` 는 원본 데모와 동일하게 3단으로 나눠 돕니다.

```
이미지 ─[전처리]→ float32 NCHW 608×608
          ↓
     front ONNX (CPU)          input → input_quantized       uint8 로 양자화
          ↓
     NPU (yolov3_lw.o)         SendInput → Launch → ReceiveOutputs
          ↓                    uint8 버퍼 3개 (19², 38², 76² 헤드)
     [역양자화]                 (x - zero_point) * scale
          ↓
     back ONNX (CPU)           3개 헤드 → dets, labels   (앵커 디코딩 + NMS)
          ↓
     [그리기] → result.jpg
```

`front`/`back` 서브그래프는 `onnx.utils.extract_model` 로 원본에서 잘라내며,
이미 있으면 재사용합니다(`*.onnx.front`, `*.onnx.back`).

역양자화 파라미터는 원본 ONNX 의 `DequantizeLinear` 노드 세 개에서 읽습니다:

```
onnx::Transpose_684_DequantizeLinear
onnx::Transpose_688_DequantizeLinear
onnx::Transpose_692_DequantizeLinear
```

---

## NPU 모드 — 동작 확인 완료

전용 Python 3.8 환경(`.venv38/`)을 이 폴더 안에 만들어 두었습니다.
udev 규칙도 적용되어 있어 **sudo 없이 바로 됩니다.**

```bash
./run_npu.sh            # --mode npu
./run_npu.sh both       # CPU 와 나란히 비교
```

### 실측 결과

```
[    npu] 드라이버 초기화 완료
[    npu] 모델 적재 완료 (yolov3_lw.o)
[    npu] NPU 실행 90.6 ms  (출력 3개 버퍼)

[    cpu] 검출 3개 (conf ≥ 0.5) · 77.5 ms → result.cpu.jpg
[    npu] 검출 3개 (conf ≥ 0.5) · 90.6 ms → result.npu.jpg
[compare] 라벨 일치 100/100 · 속도 CPU 77 ms vs NPU 90.6 ms
```

양쪽 모두 `bicycle 1.00` · `dog 0.99` · `truck 0.86` — **CPU 와 결과가 완전히 일치**합니다.

> **NPU 가 CPU 보다 느린 것은 공정한 비교가 아닙니다.**
> NPU 쪽 91 ms 에는 `SendInput`(1.1 MB 전송) + `Launch` + `Wait` + `ReceiveOutputs` 가
> 전부 들어 있고, 앞뒤의 front/back CPU 세션 시간은 빠져 있습니다.
> 상대는 i7-12700 이고요. 원본 데모가 비디오에서 warmup 파이프라인을 쓰는 이유가
> 여기 있습니다 — 전송과 연산을 겹쳐 지연을 숨깁니다. 단발 추론으로는 그 이점이 안 나옵니다.

### 드라이버 호출 순서 — 두 번 데였습니다

`NeublaDriver` API 는 7개입니다: `Init` `LoadModel` `SendInput` `Launch` `Wait` `ReceiveOutputs` `Close`.

**① `Wait()` 를 빼면 쓰레기값이 나옵니다**

`Launch()` 는 비동기입니다. 완료를 기다리지 않고 `ReceiveOutputs()` 를 부르면
초기화되지 않은 버퍼가 돌아와 **검출 100개가 전부 score 1.00** 인 결과가 나옵니다
(NPU 실행 시간도 0.7 ms 로 찍혀 바로 알아챌 수 있습니다).

원본 데모의 **비디오 루프**는 `Wait()` 없이 도는데, 그건 시작 전 warmup 으로
`SendInput`+`Launch` 를 한 번 더 해서 **한 프레임 밀린 파이프라인**으로 돌리기 때문입니다.
단일 이미지에서는 `Wait()` 가 맞습니다.

**② `Close()` 를 빼면 다음 실행이 실패합니다**

```
Failed to program DMA0 engine. err: 1
Failed to sg_dma_write (0xa00001000, 217248)      ← 0x350a0 = .text 크기
Failed to copy antara object file [ret: -5]
```

성공한 실행이 `Close()` 없이 끝나면 보드에 상태가 남아 **다음 `LoadModel` 이 DMA 단계에서 죽습니다.**
성공/실패가 번갈아 나타나면 이걸 의심하세요. 예외가 나도 닫히도록 `try/finally` 로 감싸야 합니다.

### udev 규칙 (이미 적용됨)

이 장비에는 설치 완료되어 `/dev/plda0` 이 `crw-rw-rw-` 입니다.
새 장비에 옮길 때는 한 번 해주면 됩니다:

```bash
sudo cp ~/workdir/antara-software/antara-pcie-sdk/PCIeDriver/51-plda-pcie-udev.rules \
        /etc/udev/rules.d/
sudo udevadm control --reload-rules
# 드라이버 재로드 후 /dev/plda0 이 0666 으로 생성됩니다
```

### 보드가 응답하지 않으면

리셋은 **반드시 호스트에서 먼저** 해야 합니다
([why-hostreset-flow.md](../docs/why-hostreset-flow.md)):

```bash
cd ~/workdir && sudo ./hostreset.sh
cd ~/workdir/antara-software/antara-pcie-sdk && sudo ./run.sh
```

### `.venv38` 은 어떻게 만들었나

기존 conda 환경을 건드리지 않으려고 폴더 안에 독립 venv 를 만들었습니다.
날려도 아래 두 줄로 복구됩니다.

```bash
~/miniforge3/envs/test/bin/python -m venv .venv38          # 3.8.20 기반
.venv38/bin/pip install "numpy<2" opencv-python-headless onnx onnxruntime
```

`opencv-python-headless` 를 쓴 이유는 이 예제가 GUI 를 안 띄우기 때문입니다(X11 불필요).
`ultralytics` 도 불필요합니다 — COCO 클래스 이름을 스크립트에 내장했습니다.

---

## 원본 데모와 다른 점

| | `demo_1127/yolo_demo.py` | 이 예제 |
|---|---|---|
| 입력 | 비디오 · 카메라 · 이미지 | 이미지 한 장 |
| 출력 | `cv2.imshow` GUI + 파일 | 파일만 (X11 불필요) |
| CPU 기준 실행 | 없음 | `--mode cpu` |
| 사전 점검 | 없음 (실패하면 예외) | `--mode check` |
| `ultralytics` | 필수 | 불필요 |
| 줄 수 | 480 | ~300 (주석 포함) |

성능 측정이나 비디오 데모가 필요하면 원본을 쓰는 게 낫습니다.
이 예제는 **경로를 이해하고 문제를 좁히는 용도**입니다.

---

## 확인된 것

**전 구간 실측 완료**입니다.

| 항목 | 결과 |
|---|---|
| `--mode check` | python 3.10 · 3.8 양쪽에서 진단 정확 |
| `--mode cpu` | bicycle 1.00 · dog 0.99 · truck 0.86 · 77~98 ms |
| `--mode npu` | **동일한 검출** · NPU 구간 90.6 ms |
| `--mode both` | 라벨 일치 **100/100** |
| 연속 실행 | `Close()` 추가 후 2회 연속 성공 확인 |

중간 텐서 실측값 (문제 생겼을 때 대조용):

```
front  ['input'] → ['input_quantized']    uint8 (1,3,608,608) = 1,108,992 B
                                           = 3 × 608 × 608, SendInput 인자와 일치
back   3개 헤드 → ['dets','labels']        (1,100,5) · (1,100)
역양자화 scale/zero_point                   684: 0.11476/183
                                           688: 0.12135/188
                                           692: 0.12158/196
NPU 출력                                    uint8 버퍼 3개
```

### 남은 것

- 비디오·카메라 입력 — 원본 `demo_1127/yolo_demo.py` 를 쓰세요
- warmup 파이프라인 방식 — 이 예제는 단일 이미지라 `Wait()` 동기 방식만 씁니다
- `yolov3_default.o` / `yolov3_half.o` 변형 비교 — `--binary` 로 바꿔가며 시도 가능
