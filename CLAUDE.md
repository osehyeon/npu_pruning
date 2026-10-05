# workspace_docs

YOLOv3 구조적 프루닝 + INT8 양자화를 Neubla Antara NPU 에서 평가한 기록.
논문 초안은 `paper.md`, 근거는 `docs/`, 실행 코드는 `example/`.

## 먼저 알아야 할 함정 세 가지

이 셋은 전부 실제로 틀린 결론을 만들었던 것들입니다. 작업 전에 읽으세요.

### 1. 모델 계보 — 배포 원본은 프루닝 모델의 부모가 아닙니다

워크스페이스에 **서로 다른 두 YOLOv3** 가 같은 이름으로 섞여 있습니다.

```
배포 원본 u8  = mmdetection yolov3_d53_mstrain-608_273e_coco 계열
프루닝 부모   = Nota 학습 PyTorch YOLOv3 계열
첫 Conv (32,3,3,3) 가중치 상관 −0.05      ← 무관한 모델
```

아키텍처는 동일(Conv 75개 · 채널 27,069)해서 겉으로는 구분이 안 됩니다.
**계보를 가로지르는 AP 비교는 무효입니다.** 이걸 놓쳐서 *"프루닝이 거의 무손실(−0.47)"* 이라는
결론이 세 번 나왔고 세 번 다 틀렸습니다. 실제 손실은 **−9.6** 입니다.

→ `docs/model-lineage-warning.md`

### 2. 디코더 — 모델마다 wh 디코딩 공식이 다릅니다

```
표준 YOLOv3 (원본)   wh = exp(t) · anchor
Nota 프루닝          wh = 4·σ(t − ln3) · anchor      (상한 4.0)
```

프루닝 모델의 wh 는 최대 20.68 까지 튑니다. `exp` 를 씌우면 앵커의 9억 배 박스가 나오고
AP 가 33.97 → 12.70 으로 무너집니다. `example/decoders.py` 에 둘 다 구현돼 있으니
**모델에 맞는 디코더를 쓰세요.**

### 3. `o_dbg=8` — 커널 로그가 디스크를 채웁니다

PLDA 드라이버를 디버그 모드로 올리면 추론 1회마다 로그가 쌓입니다.
5,000장 측정을 돌렸다가 `kern.log`+`syslog` 가 15.5 GB 로 불어나 루트 파티션이
100 % 가 됐고 측정이 통째로 죽었습니다. **디스크가 차면 `df` 조차 실행되지 않습니다.**

`~/workdir/load_driver.sh` 는 이제 기본이 `o_dbg=0` 입니다. 디버깅할 때만 `-d`.

→ `docs/after-reboot.md`

## 문서 권위 순서

| 주제 | 정답 문서 |
|---|---|
| mAP · 프루닝 손실 · NPU 실측 | **`docs/pruning-map-comparison.md`** |
| 모델 계보 대조 | `docs/model-lineage-warning.md` |
| 재부팅 · 드라이버 · 보드 복구 | `docs/after-reboot.md` |
| 컴파일 옵션(`lw`/`egs`) | `docs/nbc-compile-options.md` |
| 논문 초안 | `paper.md` |

**`YOLOv3_Antara_재현패키지/` 는 2026-07-29 시점 스냅샷이라 결과보다 뒤처져 있습니다.**
그 안의 `전처리/재현/mAP_측정_계획.md` 는 *"보류 중, 환경 준비 단계"* 라고 적혀 있지만
mAP 측정은 2026-08-02~04 에 완료됐습니다. 계획서가 아니라 위 표를 믿으세요.

## 실행

```bash
cd ~/workspace_docs/example
~/miniforge3/envs/nota/bin/python eval_map.py --limit 500      # fp32 CPU
./run_u8_final.sh                                              # uint8 CPU (.venv38 사용)
LIMIT=0 ./run_map_npu.sh 40_egs                                # NPU 실측 (보드 필요)
./run_bench.sh orig_egs.o 40_egs.o 50_egs.o                    # NPU 속도만
```

`run_bench.sh` 는 인자 없이 돌리면 `orig_egs.o 40_egs.o 50_egs.o` 를 씁니다.
**첫 번째 바이너리가 가속 배수의 기준**이 되므로 순서를 바꿀 때 주의하세요.

파이썬 환경이 두 개이고 **역할이 다릅니다**:

| 환경 | ORT | 용도 |
|---|---|---|
| `~/miniforge3/envs/nota/bin/python` | **1.12.1** | fp32 평가 · **재양자화**(`requantize.py`) — 배포 당시 버전이라 `CalibMovingAverage` 조건을 그대로 재현 |
| `example/.venv38/bin/python` | **1.19.2** | uint8 평가(`eval_u8_final.py`, `run_u8_final.sh`) |

`torch` 불필요. Nota 디코더는 numpy 로 이식돼 있습니다 (`example/decoders.py`).

## 이 디렉토리 밖 의존

cwd 를 `workspace_docs` 로 두면 아래는 밖입니다. 필요하면 `/add-dir` 하세요.

```
~/demo_1127     원본 u8 모델 · PYTHONPATH 로 필요
~/demo_0211     40.onnx / 50.onnx / nbc 컴파일러
~/demo_pruning  캘리브레이션 데이터셋
~/workdir       load_driver.sh · hostreset.sh · PCIe SDK
```

## 미완

- ~~`yolov3_original_fp32.onnx` 의 공개 출처 미확인~~ → **2026-08-26 확정.**
  ONNX Model Zoo `yolov3-12` 의 BatchNorm 접기 판본이며 Conv 75 개 전부 비트 동일.
  따라서 계보는 Darknet 이고 held-out 은 `5k` 입니다.
- 배포 원본 uint8 의 fp32 소스 — 워크스페이스 밖. mmdet 계열이라 프루닝 실험과는 무관.

(R 계열 NPU 재측정은 2026-08-09 완료 — `docs/pruning-map-comparison.md` 의 R 계열 절)

## 작업 규칙

- 수치를 옮길 때는 **표본 수(500장/5,000장)와 전처리(resize/letterbox)를 반드시 함께** 적으세요.
  같은 모델이 조건에 따라 26~41 사이에서 움직입니다.
- 새 측정 결과는 `example/*.json` 에 저장되고, 해석은 `docs/pruning-map-comparison.md` 에 씁니다.
