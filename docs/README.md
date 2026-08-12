# `/home/osehn` 워크스페이스 정리

> 작성일: 2026-08-02 · 대상: `/home/osehn` 홈 디렉토리 전체
> 각 폴더를 훑어보고 "이게 뭘 위한 파일인가"를 정리한 문서입니다.

---

## 한 줄 요약

**Neubla NPU(Antara 보드) 위에서 YOLOv3 객체탐지 모델을 돌리기 위한 작업 공간**입니다.
ONNX 모델을 **단순화(onnxsim) → 양자화(quantization) → 프루닝(Nota AI 압축) → NPU 컴파일(nbc) → 보드 실행/데모/성능측정**하는
전 과정의 스크립트·모델·데모가 시기별/실험별로 쌓여 있습니다.

핵심 등장인물:

| 이름 | 정체 | 흔적 |
|---|---|---|
| **Neubla** | NPU 하드웨어/컴파일러 벤더 | `NeublaDriver.*.so`, `nbc` 컴파일러, `neubla/demo_1022` 도커 이미지 |
| **Antara** | Neubla NPU 보드 (PCIe/EP 모드) | `antara-software`, `antara_ne_run`, `ST-Antara EP mode build guide.pdf` |
| **Nota AI** | 모델 경량화(프루닝) 파트너 | `compressed_yolov3_40%.onnx`, `0521_nota`, `nota/` |
| **ETRI** | 폴더 소유 그룹(`etri:antara`)으로 등장 | `AIComps/` |

전형적인 파이프라인:

```
yolov3.onnx (fp32)
   └─ onnxsim.sh          → 그래프 단순화 + shape inference (opset 13)
   └─ onnxquant.sh        → ONNXRuntime 정적 양자화 (uint8, calib_dataset 사용)
   └─ Nota 프루닝          → compressed_yolov3_40% / 50%
   └─ nbc (Neubla 컴파일)  → *.o  (gen_inline_assembly.o → yolov3_lw.o 등)
   └─ NeublaDriver + ORT  → 보드에서 추론 (앞단/뒷단은 CPU, 본체는 NPU)
```

추론 구조가 특이한데, `demo_1127/yolo_demo.py`를 보면
**front(전처리·양자화) ONNX → NPU 바이너리(.o) → back(역양자화·후처리) ONNX** 3단으로 쪼개서
가운데만 NPU가 실행하는 방식입니다.

---

## 폴더별 정리

### 1) 최신 데모 (실제로 돌려보는 것들)

| 폴더 | 크기 | 용도 |
|---|---|---|
| `demo_1127/` | 933M | **가장 최신 데모(2025-11-27).** 원본/프루닝 YOLOv3 각각의 이미지·비디오 데모. Dockerfile + `docker_run.sh` 로 컨테이너 실행 |
| `demo_1022/` | 2.2G | 그 이전 버전 데모(10/22). `neubla/demo_1022` 도커 이미지가 여기서 나옴 |
| `demo_0211/` | 41G | 가장 방대한 실험 폴더. `onnxsim / onnxquant / compile / run / latency` 전 단계 스크립트 + 각 단계별 모델 디렉토리가 다 들어있음 |
| `demo_archive/` | 1.3G | 위 파이프라인을 정리해 문서화한 아카이브. `readme.md`에 Todo 형태로 실행 순서가 적혀 있음 |
| `demodir_origin/` | 7.0G | 원본 데모 배포본. 카메라 데모(`yolo_demo_cam.py`), SSH 데모, USB 리셋 스크립트 등 보드 붙여서 쓰던 것들 |

### 2) 압축/양자화 실험

| 폴더 | 크기 | 용도 |
|---|---|---|
| `demo_pruning/` | 897M | 프루닝 모델(40%) 양자화 + 컴파일 데모. `README.md`에 conda 환경 + `run_onnxquant.sh` 사용법 |
| `demo_quant/` | 589M | 위에서 양자화만 떼어낸 최소 버전 |
| `yolov3-preprocessing/` | 99M | 양자화 **전처리 도구 모음**. 모델 추출/속성 수정/shape inference/캘리브레이션 → `yolov3-12-quantized.onnx` 생성 |
| `utils/` | 188K | 공용 유틸 (`onnxquant.py`, `calib_dataset`, requirements) |

### 3) Nota AI 관련

| 폴더 | 크기 | 용도 |
|---|---|---|
| `nota/` | 15G | Nota와 주고받은 작업들의 총집합. `nota`, `nota_2`, `nota_no_model`, `nota_yolov3`, `yolov3_nota` 등 반복 시도 흔적. `model/`에 원본 vs 40% 프루닝 모델 |
| `0521_nota/` | 552M | 2025-05-21 Nota 전달용 **YOLOv3 ONNX 추론 패키지**(zip 포함). 홈의 `readme.md`가 이 패키지 설명서 |
| `nota_yolov3_inference/` | 496M | 위 패키지의 압축 해제본 (`__MACOSX/`는 macOS zip 잔재) |
| `result/`, `sample_image/`, `yolo_image.py`, `readme.md` | - | **홈에 흩어져 있지만 위 패키지의 일부.** 원본/40%/50% 프루닝 int8 모델 추론 결과가 `result/`에 있음 |

### 4) Neubla 런타임 / 드라이버 / SDK

| 폴더 | 크기 | 용도 |
|---|---|---|
| `neubla_dev_env/` | 51G | Neubla 개발 환경 스냅샷 (컨테이너 루트를 통째로 뜬 형태: `opt/`, `usr/`, `tmp/`). `nbc` 컴파일러가 여기 있음 |
| `neubla_dev/` | 554M | 개발 킷 (드라이버·양자화 devkit·도커 이미지·스크립트) |
| `neubla_driver_origin/` | 131M | **Neubla Runtime 예제 원본.** README에 nbc 컴파일 옵션(`-tx_half_ofm_step`, `-enable-lw-group`)과 보드 배포 절차가 정리되어 있어 가장 참고할 만함 |
| `neubla/` | 40K | 껍데기 (`nn_test.py`, `verify.py` 등 스크립트만, `niss`/`nrcl`은 비어 있음) |
| `workdir/` | 4.4G | Antara 보드 소프트웨어(`antara-software`: PCIe SDK, 부트로더, 툴체인) + 도커 빌드 스크립트 |
| `tests/` | 129M | 런타임 테스트 조각 (`yolo_run.cpp`, `runtime.py`) |

### 5) 성능 측정 / 검증

| 폴더 | 크기 | 용도 |
|---|---|---|
| `yolo_table/` | 1.2G | **GPU vs NPU 속도 비교표 생성.** `yolo_table_gpu/log/`에 GTX 1060 / i7-12700 / AGX 별 fp32·uint8 1000회 측정 로그 |
| `yolo_val/` | 1.1G | COCO **정확도(mAP) 검증**. ultralytics `val.py` 기반, `val_antara.sh`로 보드에서 검증 |
| `yolo_demo/` | 1.1G | 초기 데모 실험장. `yolo_image_v2/v3`, `yolo_imshow_*` 등 버전별 시도가 그대로 남아있음 |

### 6) 모델 / 데이터 / 기타

| 항목 | 크기 | 용도 |
|---|---|---|
| `AIComps/` | 52G | ETRI 과제 폴더(그룹 `etri:antara`). `model_zoo/`에 yolov3-10/12/int8 ONNX, `project/`에 Neubla용 양자화 모델 |
| `datasets/` | 1.3G | `coco`, `coco128` (캘리브레이션·검증용) |
| `yolov3/` | 369M | ultralytics YOLOv3 원본 리포지토리 클론 (`yolov3.pt`, `yolov3.onnx`) |
| `resnet50_demo/` | 3.8G | YOLO 말고 **ResNet50** NPU 데모 (같은 구조: Dockerfile + NeublaDriver + runtime) |
| `neubla_dev_env.tar.gz` | **13GB** | `neubla_dev_env/` 의 원본 tarball. 압축 해제본이 이미 있으므로 **정리 1순위 후보** |
| `D` | 1.8K | 이름만 `D`인 **Dockerfile**. ubuntu:22.04 + pyenv 3.8 + onnx/onnxruntime/ultralytics 빌드 |
| `paper.pptx` / `paper.odp` | 1.2M | 발표 자료 |
| `ST-Antara EP mode build guide-*.pdf` | 8.5M | Antara 보드 EP(Endpoint) 모드 빌드 가이드 |
| `ne_out_0-0.bin` | 704B | NPU 출력 덤프 조각 |
| `yolo_image.py` | 15K | 홈 루트의 추론 스크립트 (0521 패키지 소속) |

---

## 관찰 / 제안

1. **같은 파이프라인이 6~7벌 복제되어 있습니다.** `demo_0211 / demo_1022 / demo_1127 / demo_archive / demodir_origin / demo_pruning / demo_quant` 는 대부분 같은 스크립트의 시점별 스냅샷입니다. 최신은 `demo_1127`, 문서화가 제일 잘 된 건 `demo_archive`입니다.

2. **용량 상위 5개가 전체의 대부분을 차지합니다.**
   - `AIComps` 52G · `neubla_dev_env` 51G · `demo_0211` 41G · `nota` 15G · `neubla_dev_env.tar.gz` 13G
   - 특히 `neubla_dev_env.tar.gz`(13GB)는 압축 해제본이 옆에 그대로 있으므로 바로 회수 가능한 공간입니다.

3. **홈 루트가 프로젝트 폴더처럼 쓰이고 있습니다.** `readme.md`, `yolo_image.py`, `result/`, `sample_image/`, `utils/` 는 `nota_yolov3_inference` 패키지의 일부인데 홈 루트에 풀려 있습니다. 한 폴더로 묶으면 깔끔해집니다.

4. **`AIComps/`는 소유자가 `etri:antara`** 입니다. 개인 작업물이 아니라 과제 공유 폴더일 가능성이 높으니 정리 시 주의가 필요합니다.

5. **처음 파악할 때 읽으면 좋은 파일 순서**
   1. `demo_archive/readme.md` — 전체 파이프라인 실행 순서
   2. `neubla_driver_origin/README.md` — nbc 컴파일 옵션과 보드 배포 절차
   3. `demo_1127/yolo_demo.py` — front/NPU/back 3단 추론의 실제 구현
   4. `readme.md` (홈 루트) — Nota 전달용 추론 패키지 사용법
