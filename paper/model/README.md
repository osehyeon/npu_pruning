# 논문용 모델 세트

프루닝율 × staircase 격자와 그 파생 모델 18종. 모든 항목의 해시·구조·연산량·NPU 실측은
`manifest.json` 에 고정돼 있습니다.

## 구성

| 그룹 | 개수 | 내용 |
|---|---:|---|
| `parent/` | 1 | 프루닝 부모. ONNX Model Zoo 의 `yolov3-12.onnx` 를 <https://github.com/onnx/models/blob/main/validated/vision/object_detection_segmentation/yolov3/README.md> 에서 받은 그대로. pjreddie 가중치와 Conv 75개 비트 동일. 파일명 `yolov3-12_bn-unfolded.onnx` 로 BN 을 접지 않았음을 표시. 접은 판본은 `../../model/yolov3_original_fp32.onnx` |
| `sweep/` | 12 | 2025-03 스윕 — 10/30/50 % × sc 8/16/32/64. **재학습 전** |
| `comp250514/` | 3 | 2025-05-14 — 30/40/45 % × sc64. **재학습 전** |
| `final/` | 2 | Nota 제출본 40 % / 50 %. **재학습 후** |
| `ours/` | 0 | 본 연구 재현본 1 종이 있었음. 논문에 쓰지 않아 2026-09-16 에 `fgcs/archive/neubla/models-not-in-paper/paper-model-ours/` 로 옮김. `manifest.json` 의 항목은 기록으로 남김 |

## 매니페스트 필드

```
name · group · ratio_nominal · staircase        식별
sha256 · size_bytes                             무결성
params · channels_total · n_conv                구조
n_conv_aligned64                                64 배수 conv 수 (하드웨어 정렬 지표)
macs_raw · macs_padded                          연산량 (608 입력)
macs_raw_ratio · macs_padded_ratio              부모 대비
pad_waste_pct                                   ⌈C/64⌉·64 패딩으로 인한 낭비
channels_reduction_pct                          실효 채널 감축률
npu_latency_s                                   2025-03 Antara 실측 (연계 문서 E02)
```

`macs_padded = Σ ⌈Cout/64⌉·64 · ⌈Cin/64⌉·64 · k² · H · W`

## 핵심 관찰

**staircase 는 연산량이 아니라 하드웨어 정렬을 정합니다.**

| ratio | sc | raw MACs | padded MACs | 64배수 conv | 패딩 낭비 | NPU (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 8 | **38.0 G** | 61.7 G | 18/75 | 62.4 % | 0.0462 |
| 50 | 16 | 37.9 G | 59.9 G | 32/75 | 58.0 % | 0.0463 |
| 50 | 32 | 38.7 G | 58.9 G | 52/75 | 52.3 % | 0.0463 |
| **50** | **64** | **41.8 G** | **58.7 G** | **70/75** | **40.5 %** | **0.0413** |

sc64 는 raw MACs 가 **가장 많은데** NPU 에서 **가장 빠릅니다**. 상관계수도 padded 쪽이
높습니다 (npu_latency_s 를 가진 15종 기준 raw r = +0.891, padded r = +0.946.
sweep 12종만 보면 +0.878 과 +0.932). 연산량 지표만 보면 정반대 결론이 나옵니다.

## 명목 대 실효

`ratio_nominal` 은 도구에 지정한 값이고 실제 감축은 다릅니다.

```
명목 40 % → 실효 37.12 %      (채널 27,069 → 17,021)
명목 50 % → 실효 45.40 %      (채널 27,069 → 14,781)
```

64 배수 정렬과 고정 채널 레이어(검출 헤드 255ch, 앞단 26개 conv 무손상) 때문입니다.

## 주의

- **`sweep/` 과 `comp250514/` 는 재학습 전** 모델입니다. AP 가 0 근처입니다
  (실측: `comp250514_r40_sc64` = AP 0.03). 정확도 비교에 그대로 쓰면 안 됩니다.
- `final/` 은 Nota 가 **COCO train2017** 로 재학습했습니다. `5k` 평가셋의 87 %가
  그 학습 데이터에 포함되므로 **5k 로 평가하면 안 됩니다** →
  `docs/coco-split-contamination.md`
- archive 로 옮긴 `ours` 는 `trainvalno5k` 로 재학습했으므로 `5k` 가 held-out 입니다.
- `npu_latency_s` 는 2025-03 측정으로, 당시 컴파일 옵션·드라이버 기준입니다.
  현재 조건과 다를 수 있습니다.

## 재현

```bash
python example/build_paper_models.py          # 이 디렉토리 재생성
python example/analyze_staircase_hw.py        # 정렬 가설 검증
python example/analyze_sweep_complexity.py --latency 12
```
