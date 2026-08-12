# COCO 분할 오염 — Darknet 가중치를 val2017 로 평가하면 안 됩니다

2026-08-10 확인. **계보 혼동(`model-lineage-warning.md`)보다 영향이 큽니다.**

## 한 줄 요약

원본 YOLOv3(Darknet)와 Nota 프루닝본은 **서로 다른 held-out 집합**을 갖습니다.
어느 쪽 집합으로 재느냐에 따라 **프루닝 손실이 −8.07 에서 +8.85 까지 뒤집힙니다.**

## 왜 어긋나는가

```
원본 Darknet YOLOv3
    학습 trainvalno5k = train2014 + val2014 − 5k(pjreddie)     → held-out: 5k
Nota 프루닝 + 재학습
    학습 train2017    = train2014 + val2014 − val2017          → held-out: val2017
```

두 진영이 val2014 에서 **서로 다른 5,000장**을 held-out 으로 잡았습니다.

```
pjreddie 5k.part  ∩  COCO val2017  =  643장
```

무작위 5k 를 독립적으로 두 번 뽑았을 때 기대 겹침이 `5000²/40504 ≈ 617` 이고 관측이 **643**
입니다 — 서로 무관한 추출입니다.

**결과: val2017 의 4,357장(87.1 %)이 Darknet 의 학습 데이터입니다.**

## 실측 — 같은 세 모델, 같은 코드, 데이터만 교체

`example/eval_map_split.py`, 608px, plain resize, score 0.001 · NMS 0.45 · 상위 100.

| 평가 집합 | original | pruned40 | pruned50 | Δ40 | Δ50 |
|---|---:|---:|---:|---:|---:|
| val2017 5,000 *(원본만 오염)* | 38.26 | 30.19 | 26.13 | **−8.07** | **−12.13** |
| **643 교집합** *(양쪽 깨끗)* | **34.16** | **32.20** | **29.21** | **−1.95** | **−4.94** |
| 5k 5,000 *(프루닝본만 오염)* | 31.51 | 40.36 | 36.11 | **+8.85** | **+4.60** |

세 표 전부 "COCO 5,000장(또는 643장) 전수 · 표준 COCO 프로토콜"이라고 적을 수 있습니다.
**프루닝 손실의 정답은 −1.95 / −4.94 입니다.**

5k 에서 프루닝본이 원본보다 8.85 높게 나오는 것이 오염의 명백한 증거입니다 —
프루닝이 모델을 좋게 만들 수는 없습니다.

## 외부 기준과의 정합

깨끗한 5k 에서 원본을 재면 문헌값 대역에 들어옵니다.

| 출처 | 값 | 조건 |
|---|---:|---|
| YOLOv3 논문 (Redmon) | 33.0 | test-dev |
| ultralytics archive README | 33.1 | 5k(2014) |
| 우리 재현, archive 코드 | **32.9** | 5k, `--iou-thres 0.7` |
| 우리 파이프라인 | **31.51** | 5k, 정사각 resize |
| 우리 파이프라인 | 38.26 | **val2017 (오염)** |

문헌의 33.0 은 옳았고 재현됩니다. 재현되지 않은 건 우리 쪽이었습니다.
40 대가 나온 것은 파이프라인 결함이 아니라 **평가셋 오염**이었습니다.

## 재현 방법

```bash
# 1. pjreddie 의 5k 목록 (원 URL 은 죽었고 Wayback 에 남아 있습니다)
curl -sSL -o 5k.part \
  "https://web.archive.org/web/20240619191224id_/https://pjreddie.com/media/files/coco/5k.part"

# 2. COCO 2014 (5k 는 전부 val2014 소속)
curl -O http://images.cocodataset.org/zips/val2014.zip
curl -O http://images.cocodataset.org/annotations/annotations_trainval2014.zip

# 3. 세 집합 평가
python eval_map_split.py --ann <instances_val2014.json> --imgdir <val2014> \
                         --ids-file ids_5k.txt  --tag "5k"  --out map_split_5k.json
python eval_map_split.py --ann <instances_val2014.json> --imgdir <val2014> \
                         --ids-file ids_643.txt --tag "643" --out map_split_643.json
```

`ids_643.txt` 는 `5k.part` 의 ID 와 `instances_val2017.json` 의 ID 교집합입니다.

## 영향받는 기존 결론

| 항목 | 판정 |
|---|---|
| 프루닝 손실 −6.84 (NPU R 계열) | **무효** — val2017 기준 |
| 프루닝 손실 −9.6 (`CLAUDE.md` 함정 1) | **무효** — val2017 기준 |
| 원본 fp32 40.75 / uint8 38.15 | **무효** — 87 % 학습 데이터 |
| R 계열 NPU 실측 35.47 / 28.63 / 24.75 | **재측정 필요** |
| 양자화 손실 −1.5 ~ −2.6 | 동일 모델 내부 비교 — 영향 작음 |
| `egs` 컴파일 무손실 | **영향 없음** — 같은 모델·같은 데이터 |
| 계보 확정 (sha256 · 비트 동일) | **영향 없음** |
| 프루닝 마스크 복원 (L2 상위 k · 64 정렬) | **영향 없음** |

## 규칙

1. **Darknet 계열 가중치를 COCO val2017 로 평가하지 마세요.** 87 %가 학습 데이터입니다.
2. **Nota 재학습본을 5k 로 평가하지 마세요.** 87 %가 학습 데이터입니다.
3. 두 계열을 **비교**할 때는 `643장 교집합`만 유효합니다.
4. 원본 **절대값**은 `5k 5,000장`으로 재세요.
5. 근본 해결은 **5k ∪ val2017 을 제외하고 재학습**하는 것입니다 — 그러면 5k 전수로
   편향 없이 비교됩니다. `example/build_clean_split.py` 가 그 목록을 만듭니다.

## 어쩌다 놓쳤는가

*"COCO 2014 minival = 2017 val"* 이라는 통설을 검증 없이 받아들였습니다. 널리 반복되는
서술이지만 **pjreddie 의 5k 에는 해당하지 않습니다.**

그리고 독립 구현 세 개(ONNX Zoo 그래프 · ultralytics PyTorch · darknet C 바이너리)가
val2017 에서 40.4 / 40.73 / 43.57 로 일치한 것을 검증으로 받아들였습니다.
**구현 독립성은 데이터 오염을 잡지 못합니다** — 셋이 같은 오염된 평가셋을 공유했습니다.

경고 신호는 있었습니다. *"YOLOv3 에 40 대를 적은 논문을 본 적이 없다"* 는 지적이 정확했고,
문헌 전체가 33 을 쓰는데 우리만 40 이 나오면 우리를 의심했어야 합니다.
