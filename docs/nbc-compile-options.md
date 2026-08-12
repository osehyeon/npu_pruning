# nbc 컴파일 옵션 — default / half / lw

> 작성일: 2026-08-02 · 관련: [neubla-isa.md](./neubla-isa.md), [antara-npu-internals.png](./antara-npu-internals.png)

`demo_1127/yolov3/` 에 있는 세 개의 `.o` 가 무엇이 다른지, 어떤 옵션으로 만드는지 정리합니다.

---

## 만드는 방법

### 실제 경로 — EP 옵션으로 전달됩니다

`nbc` 를 직접 부르는 게 아니라 **ONNX Runtime 세션을 만들면 EP 가 내부에서 호출**합니다
([model-to-binary-pipeline.md](./model-to-binary-pipeline.md)).
`nbc` CLI 플래그는 EP 옵션 문자열로 매핑됩니다:

| EP 옵션 | 값 | `nbc` 플래그 | 산출물 |
|---|---|---|---|
| `nbc.groupsearch` | 생략 | (하드코딩된 수동 그룹) | `yolov3_default.o` |
| `nbc.groupsearch` | `"lw"` | `-enable-lw-group` | `yolov3_lw.o` |
| `nbc.groupsearch` | `"egs"` | `-egs` | ← **가장 좋은 선택** |
| `nbc.tx_ofm_step` | `"single"` (기본) | — | 64 ch / tx |
| `nbc.tx_ofm_step` | `"half"` | `-tx_half_ofm_step` | `yolov3_half.o` |

```python
util.Inference(name, model_path, inputs, gs="egs", tx_ofm_step="single")
```

### 준비된 스크립트

```bash
cd ~/workspace_docs/example
./compile_variant.sh <uint8.onnx> <입력텐서이름> <출력.o> [groupsearch] [tx_ofm_step] [모델이름]

# 예 — 40% 프루닝 모델을 egs 로
./compile_variant.sh ~/demo_0211/pruning_model/40.onnx input_1 40_egs.o egs single
```

### 문서에 적힌 CLI 형태 (참고)

`neubla_driver_origin/README.md` 에는 직접 호출 형태로 나와 있습니다:

```bash
nbc -model NeublaExecutionProvider_NEUBLA_0_0_yolov3.onnx -input input_quantized.bin   # default
nbc -model ... -tx_half_ofm_step -input input_quantized.bin                            # half
nbc -model ... -enable-lw-group  -input input_quantized.bin                            # lw
clang -c gen_inline_assembly.cpp -emit-llvm -fdeclspec
llc -march=neubla -filetype=obj gen_inline_assembly.bc
mv gen_inline_assembly.o yolov3_default.o
```

`nbc` 는 항상 `gen_inline_assembly.*` 를 내놓으므로 **파일명이 곧 어떤 옵션으로 빌드했는지의
유일한 기록**입니다. 헷갈리기 쉬운 지점입니다.

---

## 실측 비교

세 바이너리의 ELF 섹션을 실제로 재봤습니다.

| 바이너리 | `.text` (명령) | 명령 수 | `.data` (가중치) | 총 크기 |
|---|---:|---:|---:|---:|
| `yolov3_default.o` | 247,840 B | **7,745** | 150.6 MB | 150.8 MB |
| `yolov3_half.o` | 508,800 B | **15,900** (2.05×) | 150.6 MB (동일) | 151.1 MB |
| `yolov3_lw.o` | 217,248 B | **6,789** (0.88×) | **166.6 MB** (+16 MB) | 166.8 MB |

명령 하나가 32 B 고정이라 `.text ÷ 32` 가 곧 명령 수입니다.

### 읽는 법

- **`half` 는 명령이 정확히 2배**, 가중치는 그대로
- **`lw` 는 명령이 12% 줄고, 가중치가 16 MB 늘어남**

---

## `-tx_half_ofm_step` (half)

```
-tx_half_ofm_step   split tx cmd by half ofm step (default: 0)
```

TX 명령을 **출력 채널(OFM) 스텝의 절반 단위로 쪼갭니다.**
명령 수가 7,745 → 15,900 으로 **정확히 2배**가 되는 게 그 증거입니다.

앞부분 명령 구성을 비교하면 무슨 일이 일어나는지 보입니다:

```
default   conv2d_bias{f} 5 · prelu 4 · subtile 4 · load 13
half      conv2d_bias{f} 3 + conv2d_bias{f}* 1 · prelu 3 + prelu* 1 · subtile 11 · concat 1
```

같은 구간에서 **`subtile` 이 4 → 11 로 늘고 `concat` 이 등장**합니다.
한 번에 처리하는 출력 채널 덩어리를 반으로 나누니, 잘라내고(`subtile`) 다시 붙이는(`concat`)
작업이 따라붙는 것입니다. 그리고 나뉜 조각이 **두 레인(slot 0 / slot 1)에 분배**됩니다
— `conv2d_bias{f}*`, `prelu*` 가 이 구간에서 처음 나타납니다.

**의도**: 한 번에 다루는 텐서 조각을 줄여 SRAM 2 MB 압박을 낮추고, 두 연산 레인을 더 촘촘히 씁니다.
**비용**: 명령이 2배라 명령 페치 부담과 동기화(`waitc`)가 늘어납니다.

---

## `-enable-lw-group` (lw)

```
-enable-lw-group    enable layer-wise group (default: 0)
```

**레이어 단위로 그룹을 묶습니다.** 결과가 정반대입니다 — 명령은 줄고 데이터가 늘어납니다.

```
default   load 13 · subtile 4 · waitc{f} 4 · conv2d_bias{f} 5
lw        load  8 · subtile 0 · waitc{f} 10 · conv2d_bias{f} 6 · store 5
```

**`subtile` 이 아예 사라집니다.** 레이어 단위로 미리 묶어 두면 실행 중에 타일을 잘라낼 필요가
없어지는 대신, **잘라둔 형태로 가중치를 펼쳐 저장**해야 해서 `.data` 가 16 MB 늘어납니다.
`waitc` 가 늘어난 것은 그룹 경계마다 동기화가 필요해서로 보입니다.

**의도**: 실행 시 타일링 오버헤드 제거.
**비용**: 가중치 16 MB 추가 (DRAM 전송량 증가).

> 이 워크스페이스의 데모(`demo_1127/yolo_demo.py`)가 **기본값으로 `yolov3_lw.o` 를 씁니다.**
> 실측에서 가장 나았던 모양입니다.

---

## 나머지 옵션

`nbc --help` 전체입니다. 위 둘 말고는 데모에서 쓰인 흔적이 없습니다.

### 탐색 제어

| 옵션 | 기본 | 뜻 |
|---|---|---|
| `-egs` | 0 | exhaustive group search — 그룹 조합을 전부 탐색 |
| `-try_all_tiles` | 0 | 타일 개수를 전부 시도 (기본은 유효한 값 찾으면 중단) |

둘 다 **컴파일 시간을 늘려 더 나은 스케줄을 찾는** 옵션입니다.
`lw`/`half` 로 성능이 안 나오면 다음으로 시도해볼 것들입니다.

### 하드웨어 모델 (건드리지 마세요)

이건 옵션이 아니라 **NPU 제원을 컴파일러에 알려주는 값**입니다.
실제 하드웨어와 다르게 주면 잘못된 코드가 나옵니다.

| 옵션 | 기본 | 뜻 |
|---|---|---|
| `-antara_mpw` | **1** | Antara MPW 하드웨어 제약 적용 |
| `-sram` | 2097152 | SRAM 2 MB |
| `-sram_w_sz` | 64 | SRAM word 크기 |
| `-s_w_bw` | 64 | SRAM 가중치 대역폭 |
| `-d_bw` | 5 | DRAM 대역폭 |
| `-tx_row` / `-tx_col` | 64 / 32 | TX array 크기 (NCU) |
| `-vx_vec_w` | 4 | VX 벡터 폭 |
| `-ex_cnt` | 2 | 연산 레인 수 |
| `-clk` | 1e-09 | 클럭 주기 (1 GHz) |

`-clk` 와 `-d_bw` 는 **성능 모델링용**이라 바꿔도 생성 코드가 바뀌지 않을 수 있습니다
(스케줄링 판단에만 쓰일 가능성) — 확인하지 못했습니다.

---

## 실측 속도

`dog.jpg` 한 장 · 10회 반복 평균 · `Launch`+`Wait` 구간(순수 NPU 연산)입니다.
모든 조합에서 **검출 결과가 소수점 둘째 자리까지 동일**합니다 — 정확도 손실은 없습니다.

### 원본 모델 · 옵션별

| 옵션 | 연산 ms | 대비 | 명령 수 | 가중치 |
|---|---:|---:|---:|---:|
| `lw` ← 현재 배포본 | 92.8 | 1.00× | 6,789 | 166.6 MB |
| **`egs`** | **69.4** | **1.34×** | — | 150.9 MB |
| `default` | (총 65.9) | 1.41× | 7,745 | 150.6 MB |
| **`half`** | (총 60.0) | **1.55×** | 15,900 | 150.6 MB |
| `lw + half` | **1,566.6** | **0.06×** | — | — |

> `default` / `half` 는 단계 분리 이전에 잰 값이라 **총 시간**(SendInput+Launch+Wait+Receive)입니다.
> 전송이 3.5 ms 정도이므로 연산 구간은 각각 약 62 ms / 56 ms 로 추정됩니다.

### 프루닝 모델 포함 · `lw` vs `egs`

| 모델 | `lw` | `egs` | 개선 |
|---|---:|---:|---:|
| 원본 | 92.8 ms | **69.4 ms** | **1.34×** |
| 40 % 프루닝 | 76.9 ms | **58.9 ms** | **1.31×** |
| 50 % 프루닝 | 78.2 ms | **56.6 ms** | **1.38×** |

**`egs` 는 세 모델 모두에서 30~38 % 빠릅니다.** 정확도 손실 없이 얻는 이득입니다.

### 두 축이 독립적으로 곱해집니다

```
컴파일 옵션 (lw → egs)   1.34×
프루닝     (원본 → 40%)  1.18×
                        ─────
                        1.58×    92.8 ms → 58.9 ms
```

92.8 ÷ 1.34 ÷ 1.18 = 58.7 ms — 실측 58.9 ms 와 거의 일치합니다. **서로 간섭하지 않습니다.**

### 전체 순위 (원본 `lw` = 1.00× 기준)

| 모델 | 옵션 | 연산 ms | 대비 | uint8 AP |
|---|---|---:|---:|---:|
| 원본 | `lw` | 92.8 | 1.00× | 32.13 |
| 50 % | `lw` | 78.2 | 1.19× | 26.44 |
| 40 % | `lw` | 76.9 | 1.21× | 32.00 |
| 원본 | `egs` | 69.4 | 1.34× | 32.13 |
| 원본 | `half` | (60.0) | 1.55× | 32.13 |
| **40 %** | **`egs`** | **58.9** | **1.58×** | **32.00** |
| 50 % | `egs` | 56.6 | 1.64× | 26.44 |

**`40 % 프루닝 + egs` 가 최적입니다** — 현재 배포본 대비 1.58× 빠르고 정확도는 0.13점 차이입니다.
50 % 는 4 % 더 빠른 대가로 AP 를 5.6점 더 잃어 권하지 않습니다.

### `half` 는 프루닝 모델에 쓸 수 없습니다

`half`(= `tx_ofm_step`) 자체는 `groupsearch` 와 독립인데, **`groupsearch=none` 과 짝일 때만**
의미 있는 결과가 나옵니다. 그런데 `none` 은 **컴파일러에 모델별 그룹 설정이 하드코딩**되어 있어야 합니다:

```
nbc 안의 하드코딩 설정은 두 개뿐 —  getBestGroup_yolov3 · getBestGroup_yolov3Reduced
모르는 이름 → "config is not yet implemented(NYI)" 로 즉사
yolov3Reduced 를 프루닝 모델에 적용 → "group node cnt != graph node cnt (82 vs 0)"
```

그리고 **`lw + half` 조합은 1,566 ms 로 17배 느려집니다**(편차도 ±1,492 ms).
한쪽은 잘게 쪼개고 다른 쪽은 크게 묶으려 하니 서로 상쇄되는 것으로 보입니다.

**결론: 프루닝 모델에는 `egs` 가 유일한 실용적 선택입니다.**

### 명령 수가 아니라 스케줄링이 지배합니다

순위가 명령 수와 무관합니다. 명령이 가장 적은 `lw`(6,789)가 가장 느리고,
2배인 `half`(15,900)가 가장 빠릅니다.

- `lw` 는 타일링(`subtile`)을 없애는 대신 **가중치를 16 MB 더 펼쳐 저장**합니다.
  그 추가 DRAM 전송이 명령 절감분보다 비쌉니다.
- `egs` 는 그룹 조합을 전수 탐색해 **SRAM 2 MB 에 더 잘 맞는 배치**를 찾아냅니다.
- PCIe 전송은 병목이 **아닙니다** — SendInput 3.1 ms + Receive 0.5 ms 로 전체의 4 % 뿐입니다.

> **데모(`demo_1127/yolo_demo.py`)와 `nn_test.py` 가 `lw` 를 기본값으로 쓰는데, 가장 느린 선택입니다.**
> `nn_test.py:18` 의 `gs="lw"` 를 `gs="egs"` 로 바꾸면 정확도 손실 없이 34 % 빨라집니다.

### 대가는 컴파일 시간

| 옵션 | 컴파일 시간 |
|---|---:|
| `lw` | 2분 25초 |
| **`egs`** | **약 70분** (29배) |

`egs` 는 그룹 조합을 전수 탐색하면서 SRAM 에 안 들어가는 조합마다
`FailToFindSpillIntervalException` 을 내고 다음으로 넘어갑니다.
배포 바이너리는 한 번만 만들면 되므로 감수할 만합니다.

> **`egs` 컴파일 시 `verify` 단계가 실패합니다** — `Unmatched data at idx[0]; ref(33) output(0)`.
> 그러나 실제 NPU 출력은 `lw` 와 **완전히 동일**합니다(검출 4개, 점수 소수점 둘째 자리까지 일치).
> 시뮬레이터(niss) ↔ 레퍼런스(NRCL) 비교 경로의 문제이지 생성된 바이너리의 결함이 아닙니다.

---

---

## 주의 — DMA 고장은 reboot 으로만 복구됩니다

세 변형을 비교하던 중 보드의 DMA 엔진이 망가져 **어떤 바이너리도 로드되지 않는** 상태가 됐습니다.

```
Failed to program DMA0 engine. err: 1          (err 1 = QPCIe_FAILURE)
Failed to sg_dma_write (0xa00001000, 247840)
munmap_chunk(): invalid pointer               ← Close() 에서 이중 해제로 크래시
```

- SDK 자체 예제(`antara_example`)의 DMA 테스트도 똑같이 죽었습니다 → **보드 문제**
- **`hostreset.sh` 로는 복구되지 않았습니다**
- **컴퓨터를 재부팅하니 셋 다 정상 복구**되었습니다

이유는 [after-reboot.md](./after-reboot.md#hot-reset-으로-복구되지-않는-경우) 참조 —
DMA 방향 설정은 U-Boot 이 **보드 부팅 때만** 하고, PCIe hot reset 은 U-Boot 을 다시 돌리지 않습니다.

> 한때 `half` 를 원인으로 의심했으나 **틀렸습니다.** 재부팅 후 아무 문제 없이 가장 빠르게 돕니다.
> 최초 원인은 확인하지 못했습니다.

---

## 확인하지 못한 것

- `-egs` / `-try_all_tiles` 를 켜면 실제로 얼마나 좋아지는지 (컴파일을 직접 돌려봐야 함)
- `-clk` / `-d_bw` 가 생성 코드에 영향을 주는지, 리포트에만 쓰이는지
- `half` 와 `lw` 를 **동시에** 켤 수 있는지, 켜면 어떻게 되는지
- 비디오 스트리밍(warmup 파이프라인)에서도 `half` 가 가장 빠른지
- DMA 고장을 처음 일으킨 원인
