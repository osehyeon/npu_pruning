# 산출물 위치와 재생성 절차

2026-08-12 기준. **어떤 산출물이 어디에 있고, 잃었을 때 어떻게 되살리는가**를 적습니다.
용량이 커서 저장소에 못 넣는 것들이 대상입니다.

> 2026-09-16 · 아래 1 절의 위치는 그 뒤 바뀌었습니다. `npu_pruning_lfs.zip` 은 풀어 검증한 뒤 지웠고, 모델과
> 바이너리와 스케줄 기록은 전부 이 체크아웃 안에 있습니다. 현재 위치는 `../../REMEASURE.md` 1 절이 정본입니다.
> GPU 서버에만 있는 재학습 가중치와 패치 코드에 대한 2 절과 3 절은 그대로 유효합니다.

---

## 1. 위치 요약

| 산출물 | 위치 | 크기 | 재생성 비용 |
|---|---|---|---|
| 코드 · 문서 · 측정 결과 | GitHub `osehyeon/npu_pruning` 브랜치 `npu-alignment` | 7.4 MB | — |
| ONNX 21종 · NPU 바이너리 17종 | Mac `~/Desktop/npu_pruning_lfs.zip` | 2.3 GB | 컴파일 3시간 |
| 재학습 가중치 · 체크포인트 | **GPU 서버에만** | 234 MB | **재학습 2.6시간** |
| 패치한 학습 코드 | **GPU 서버에만** | 수 KB | 아래 §3 참조 |
| COCO2014 이미지 · 라벨 | GPU 서버, 로컬은 삭제됨 | 19 GB | 다운로드 40분 |
| 재양자화 uint8 34종 | 로컬 `model/u8/` | 1.1 GB | 양자화 5분 |
| 벤더 재현패키지 | 로컬에만 | 3.0 GB | 복구 불가 |

**한 곳에만 있는 것이 셋입니다.** 재학습 가중치, 패치 코드, 벤더 재현패키지입니다.

---

## 2. GPU 서버 (blackwell, 10.254.70.158)

```
~/yolov3_prune/
  weights/
    yolov3.weights          237 MB   pjreddie 원본
    yolov3-p40.weights       78 MB   프루닝 직후, 재학습 전
    p40_ep10.weights         78 MB   재학습 10 에포크  ← 재생성 비용 최대
    p40_ep10.onnx            78 MB   위를 ONNX 로 내보낸 것
    5k.part / trainvalno5k.part      평가·학습 분할
    prune_masks.json                 복원한 채널 마스크
  code/ul_archive/
    weights/last.pt          78 MB   최종 체크포인트
    weights/best.pt         156 MB   최고 성능 체크포인트
    utils/utils.py, test.py, train.py, utils/datasets.py   ← 패치본
  data/                      19 GB   COCO2014 이미지·라벨·어노테이션
  .venv/                             torch 2.11.0+cu128, numpy 2.4.4, ort 1.28.0
```

---

## 3. 학습 코드 패치 내역

`ultralytics/yolov3` archive 브랜치는 PyTorch 1.x · numpy 1.x 시대 코드입니다.
PyTorch 2.11 · numpy 2.4 에서 돌리려면 아래 수정이 필요합니다. 로직은 바꾸지 않았습니다.

### 저장소 전체

```
np.int → int,  np.float → float,  np.bool → bool,  np.object → object
np.trapz → np.trapezoid,  np.NaN → np.nan,  np.Inf → np.inf
```

### utils/utils.py — build_targets

CPU 와 GPU 텐서를 교차 인덱싱할 수 없게 바뀌었습니다.

```python
at = torch.arange(na, device=targets.device).view(na, 1).repeat(1, nt)
gain[2:] = torch.tensor(p[i].shape, device=targets.device)[[3, 2, 3, 2]]
```

long 텐서에 float 경계로 in-place clamp 를 걸 수 없습니다.

```python
indices.append((b, a, gj.clamp_(0, int(gain[3]) - 1), gi.clamp_(0, int(gain[2]) - 1)))
```

### utils/utils.py — 시각화

matplotlib 색상 형식이 바뀌어 `hex2rgb` 가 실패합니다. 문자열이 아니면 기본색을 쓰도록
가드를 넣습니다. `output_to_target` 은 GPU 텐서를 numpy 로 바로 변환할 수 없으므로
스칼라로 풀어 씁니다.

```python
return np.array([[float(v) for v in t] for t in targets], dtype=float) if len(targets) else np.zeros((0, 7))
```

### test.py

크기 1 배열을 스칼라 자리에 대입할 수 없습니다.

```python
maps[c] = float(np.ravel(ap[i])[0])
print(pf % (names[c], seen, nt[c], float(np.ravel(p[i])[0]), float(np.ravel(r[i])[0]),
            float(np.ravel(ap[i])[0]), float(np.ravel(f1[i])[0])))
```

### utils/datasets.py

길이가 다른 배열 목록을 `np.save` 로 저장할 수 없습니다. 라벨 캐시 저장을 예외 처리로
감쌉니다.

```python
try:
    np.save(np_labels_path, np.array(self.labels, dtype=object), allow_pickle=True)
except Exception:
    pass
```

### train.py

데이터로더 워커가 8개로 제한돼 있습니다. 코어가 많으면 올립니다.

```python
nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 20])
```

---

## 4. 재학습 재현

```bash
python build_trainvalno5k.py --root <coco2014> --fivek 5k.part --out-dir <dir>
python prune_darknet.py --cfg yolov3.cfg --weights yolov3.weights \
                        --masks prune_masks.json --tag 40 \
                        --out-cfg yolov3-p40.cfg --out-weights yolov3-p40.weights
python train.py --cfg yolov3-p40.cfg --data <data> --weights yolov3-p40.weights \
                --epochs 10 --batch-size 32 --img-size 608 --device 0 --notest
```

RTX 5090 기준 에포크당 약 15.5분, 10 에포크 2.6시간입니다. 배치 32 에서 GPU 메모리를
31.2 GB 사용합니다.

---

## 5. ONNX 내보내기 주의

`export_darknet_onnx.py` 로 내보낸 그래프에는 헤드마다 `Reshape` 가 두 겹 붙습니다.
`YOLOLayer` 의 축 변환과 그것을 되돌리는 변환이 남은 것으로 수치는 같지만, `nbc` 가
이 패턴에서 죽습니다.

```
F ModelBuilder.cpp:187] Check failed: offset >= 0 (-1 vs. 0)
```

`strip_head_reshape.py` 로 제거하면 연산자 종류가 배포본과 같은 7종이 되고 컴파일이
통과합니다.

---

## 6. 백업이 필요한 것

우선순위 순입니다.

**재학습 가중치** `p40_ep10.weights` 와 체크포인트. 234 MB 이고 재생성에 2.6시간이
걸립니다. 지금 GPU 서버에만 있습니다.

**패치한 학습 코드.** 용량은 작지만 다시 만들려면 오류를 하나씩 다시 만나야 합니다.
내용은 위 §3 에 기록해 두었으므로 문서만으로도 복원 가능합니다.

**벤더 재현패키지** `YOLOv3_Antara_재현패키지/`. 3.0 GB 이고 사내 메일과 협업툴 기록이
포함되어 저장소에 올릴 수 없습니다. 이 컴퓨터에만 있으며 복구 경로가 없습니다.

COCO2014 는 공개 데이터라 백업 대상이 아닙니다. 재양자화 uint8 모델들도 원본 fp32 와
`requantize.py` 가 있으면 5분이면 다시 만듭니다.
