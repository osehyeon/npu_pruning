# 원고가 가리키지 않는 결과 파일

FGCS 원고의 값은 이 폴더의 파일에서 나오지 않습니다. 원고가 쓰는 것은 `../grid_latency_paired.tsv`, `../map_npu_R_40.json`, `../r50.log`,
`../map_u8_final_pruned40.json`, `../map_u8_final_pruned50.json`, `../map_npu_{orig,40,50}_{lw,egs}.json`, `../logs/`, `../nbc_artifacts/` 이며
출처 대응은 `fgcs/neubla/PROVENANCE.md` 에 있습니다. 여기 있는 것은 이전 세션, 다른 평가 집합, 논문에 안 쓴 모델, 분석 스크립트의 산출물입니다.
2026-09-23 에 한 폴더에 섞여 있던 것을 옮겼습니다. 이름만으로 뜻을 적었으며 열어 확인하지 않은 것은 그렇다고 적었습니다.

| 파일 | 뜻 |
|---|---|
| `grid_latency.tsv`, `grid_latency_prev.tsv`, `grid_latency_hirep.tsv`, `grid_latency_msi.tsv` | 몰아서 잰 이전 세션의 지연. 교차 측정 이전 |
| `map_zoo_*.json` | ONNX Model Zoo 원본의 평가. 416 과 608 입력, 표준 NMS 변형 |
| `map_split_5k_orig.json`, `map5k_original.json`, `map5k_p40ep10.json` | 5k 분할 평가. 원본은 이 분할이 학습 자료와 겹침. `docs/coco-split-contamination.md` |
| `map_npu_p40_*.json`, `map_p40ep10_quant.json` | 재학습 재현본 p40ep10. 논문에 쓰지 않기로 함. `fgcs/archive/neubla/models-not-in-paper/` |
| `map_npu_N_orig.json`, `map_npu_R_orig.json` | 원본의 NPU 평가. N 과 R 은 세션 표식이며 원고는 원본 AP 를 비교하지 않음 |
| `map_npu_sw50sc64_egs.json`, `map_npu_sw50sc64_lw.json` | 격자 모델 sw50sc64 의 NPU 평가. 원고의 정확도 표는 final 계보의 pruned40, pruned50 |
| `map_results.json`, `map_results_u8.json` | `docs/pruning-map-comparison.md` 의 비교 결과 |
| `map_u8_final_*_lb.json` | letterbox 전처리 변형. 원고는 `_lb` 가 없는 값 |
| `map_u8_final_neubla_q*.json` | 벤더 양자화본 평가 |
| `map_u8_final_original.json`, `map_u8_final_fp32_*_lb.json` | 원본과 fp32 변형. 원본 AP 는 val2017 이 학습 자료와 겹쳐 원고에서 비교하지 않음 |
| `map_full_fp32.json`, `map_full_u8.json` | 다른 계보의 전체 평가. 원고의 28.55 와 24.68 은 `map_u8_final_pruned40.json`, `pruned50.json` 의 값 |
| `calib_sweep.json`, `calib_sweep_ma.json` | 캘리브레이션 탐색 |
| `staircase_hw.json`, `sweep_complexity.json` | `analyze_staircase_hw.py`, `analyze_sweep_complexity.py` 의 산출물 |
| `eval_map_npu.py.bak` | 스크립트 백업 |
| `grid_latency_paper.tsv`, `grid_latency_rerun.tsv` | 몰아서 잰 이전 두 세션. 세 세션 재현 확인에만 쓰며 사본이 `fgcs/neubla/paper/data/_verify/` 에 있다 |
| `map_split_5k.json`, `map_split_643.json` | 다른 평가 집합. `docs/coco-split-contamination.md` 의 계보 분석에만 쓴다 |
| `paired_grid.json`, `paired_summary.tsv` | 교차 측정의 중간 산출물. 평균은 `grid_latency_paired.tsv` 와 같은 값이다 |
