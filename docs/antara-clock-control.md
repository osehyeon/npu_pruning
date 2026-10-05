# Antara NE 클럭 제어 — 런타임 주파수 변경

> 작성일 2026-08-24 · 근거 `~/workdir/antara-software` 부트로더와 애플리케이션 펌웨어
> 공식 문서는 없고 소스에서 복원했습니다. 관련 [antara-npu-internals.png](./antara-npu-internals.png) · [neubla-isa.md](./neubla-isa.md)

## 요약

**NE 의 동작 주파수를 UART 콘솔 명령으로 바꿀 수 있습니다.** 보드 재설계나 재빌드가
필요 없습니다. 이미 `dbgprint uart off` 를 넣을 때 쓰는 그 콘솔입니다.

연산 병목과 메모리 병목을 실측으로 분리할 수 있는 유일한 수단입니다.

---

## 1. 클럭 구성

### 기준 발진기와 시스템 PLL

`bootloader/fsbl/prci/prci_init.c` 에 있습니다.

```
XIN        32 MHz
Core PLL    1 GHz      m=250 · p=4 · s=1 · RISC-V 코어
Peri PLL   1200 MHz    m=300 · p=4 · s=1 · 주변장치
```

분주기 기본값은 `prci_set_clock_div_default()` 입니다.

```
corepll_div         DIV_N(2)
main_corepll_div    DIV_N(2)
main_peripll_div0   DIV_N(12)
main_peripll_div1   DIV_N(6)
main_spi_clock_div  DIV_N(6)
```

### NE 전용 PLL

`application/software/app_fw/ne/include/clock_ne.h` 입니다.

```
NE_PLL_COUNT   2       PLL0 · PLL1
NE_CLK_COUNT   3       BUS · SYS · VX

NE_CLK_BUS_ID  0
NE_CLK_SYS_ID  1
NE_CLK_VX_ID   2
```

`antara-npu-internals.png` 의 `PLL0 → bus · sys`, `PLL1 → VX 독립` 과 일치합니다.

### 주파수 계산식

```
PLL_CALC_RATE(P, M, S) = XIN_RATE × M / (P << S)
                       = 32 MHz × M / (P << S)

P   XIN/12MHz ~ XIN/4MHz  =  2 ~ 8
M   0x040 ~ 0x3FF         =  64 ~ 1023
S   0 ~ 6

PLL_MIN_RATE    25 MHz
PLL_MAX_RATE   3.2 GHz
PLL 락 타임아웃  500 us
```

---

## 2. 콘솔 명령

`application/software/app_fw/ne/cmd/ne_clock.c` 가 두 명령을 등록합니다.

```
ne_pll   [id] [rate]     PLL 주파수 설정 · id 는 0 또는 1
ne_clock [id] [rate]     클럭 도메인 설정 · id 는 0(BUS) 1(SYS) 2(VX)
```

`ne_pll` 은 설정 전 이전 값을 출력합니다.

```c
LOG_INFO("pll rate prev : %llu\r\n", clk_pll_get_rate(pll_id));
```

**현재 주파수를 읽는 수단이 이것뿐입니다.** 조회 전용 명령은 없습니다.

### 사용 절차

```bash
minicom -D /dev/ttyUSB0

# 현재 값 확인 겸 설정
ne_pll 0 800000000        # PLL0 을 800 MHz 로

# 도메인별 설정
ne_clock 1 400000000      # SYS 클럭 400 MHz
```

값은 Hz 단위 정수입니다.

---

## 3. 계획한 실험 — 클럭 스윕

### 목적

컴파일러 스케줄 트레이스로 **메모리 이동이 전체 시간의 65 %** 라고 추정했습니다.
클럭을 바꿔 지연을 재면 이를 실측으로 검증할 수 있습니다.

```
지연이 클럭에 반비례       연산 병목
지연이 클럭에 둔감          메모리 병목
```

프루닝 전후를 각각 쓸어보면 **압축이 병목을 어느 쪽으로 옮기는지** 드러납니다.
압축할수록 메모리 지배가 심해질 것으로 예상합니다.

### 설계

```
클럭    기본값 포함 3~4 단계
모델    원본 1종 + 프루닝 대표 2~3종
측정    bench_npu.py 로 모델당 반복 측정
소요    모델당 5분 안팎
```

### 주의

**재부팅 후에만 가능합니다.** 현재 보드는 DMA 고장 상태이고
[npu-measurement-pitfalls.md](./npu-measurement-pitfalls.md) 의 절차가 필요합니다.

콘솔에 접속하는 김에 `dbgprint uart off` 를 반드시 함께 넣어야 합니다.
안 끄면 추론마다 UART 로그가 쌓여 측정이 왜곡됩니다.

클럭을 올릴 때는 **락 실패와 발열**을 확인해야 합니다. `PLL_MAX_RATE` 가 3.2 GHz 로
잡혀 있지만 이는 PLL 의 이론 상한이지 NE 의 안전 동작 범위가 아닙니다.
기본값보다 크게 올리는 것은 권하지 않습니다.

---

## 4. 확인하지 못한 것

```
NE 의 기본 동작 주파수     ne_pll 로 조회해야 알 수 있음
전압 제어 경로            릴리즈 노트에 "NE voltage 0.9V by default" 만 있고 코드 없음
LPDDR 대역폭              nbc -d_bw 5 라는 파라미터만 보임
공칭 TOPS                 벤더 자료 없음
소비 전력                 측정 수단 없음
```

전압과 전력을 못 다루므로 **에너지 효율은 논문 범위에서 제외**해야 합니다.

---

## 5. 벤더 사양과 복원값의 불일치 — 조사 결과

선행 논문 Table 1 과 우리 복원값이 어긋나 원인을 조사했습니다.

| 항목 | 논문 Table 1 | 우리 복원 |
|---|---|---|
| MAC array | 32 × 48 | tx_row 64 · tx_col 32 |
| 유닛당 MAC | 1,536 | 2,048 |
| 총 MAC | 3 K · 1.5 K × 2 | 4,096 |
| SRAM | 1.5 MB | 2 MiB |
| 성능 | 6 TOPS | — |

### 근거 1 — nbc 기본 파라미터

`nbc --help` 를 직접 실행해 확인했습니다.

```
-tx_row       64          tx row size
-tx_col       32          tx column size
-ex_cnt        2          ex engine cnt
-sram    2097152          정확히 2 MiB
-sram_w_sz    64          Sram word size
-s_w_bw       64          Sram Weight BW
-clk       1e-09          1 GHz
-d_bw          5          Dram BW
-antara_mpw    1          enable Antara MPW H/W constraints
```

### 근거 2 — TOPS 검산은 논문 쪽이 맞습니다

```
논문   32 × 48 = 1,536 × 2유닛 = 3,072 MAC
       3,072 × 2 ops × 1 GHz = 6.14 TOPS   →  표의 6 TOPS 와 일치

nbc    64 × 32 = 2,048 × 2 = 4,096 MAC
       4,096 × 2 ops × 1 GHz = 8.19 TOPS   →  6 TOPS 와 불일치
```

### 근거 3 — 정렬 실측은 64 를 지지합니다

스윕 12종의 padded MACs 를 여러 정렬 단위로 계산해 NPU 실측과 대조했습니다.

| 정렬 A | 피어슨 r | 근거 |
|---:|---:|---|
| 1 | +0.8787 | 정렬 없음 |
| 16 | +0.8856 | |
| 24 | +0.8788 | |
| 32 | +0.9017 | tx_col |
| 48 | +0.8696 | 논문의 48 |
| **64** | **+0.9315** | tx_row · sram_w_sz · s_w_bw |
| 128 | +0.9304 | 64의 배수 |

**48 은 정렬 없음보다도 낮습니다.** 물리 배열이 48 이라면 나올 수 없는 결과입니다.

### 근거 4 — 시뮬레이터 헤더

`neubla_dev_env/tmp/neubla_mlp-build/include/niss/` 에 사이클 시뮬레이터 헤더가 있습니다.

`NissCommon.h`

```c
constexpr uint32_t WORD_SIZE{64U};          // TODO: make configurable
constexpr uint32_t INST_SIZE_BYTE{256 / 8}; // 32 바이트
constexpr uint32_t MAX_FUSED_INSTR{5U};
constexpr uint32_t TILE_ATTR_SIZE{128U};
constexpr uint32_t MAX_TENSOR_DIMS{4U};
```

**64 는 `WORD_SIZE` 로 정의되어 있습니다.** MAC 배열 치수가 아니라 메모리 워드 크기입니다.

`NissNpuCore.h`

```c
PARAMETER(uint64_t, sram_size_mb, 2, "SRAM size in MB")
PARAMETER(uint64_t, sram_private_region_kb, 2048, "The size of the private SRAM region in KiB")
```

SRAM 2 MB 이고 그중 2,048 KiB 가 private 영역입니다.

`NissDispatcher.h` — 실행 유닛 구성

```c
enum class UnitType { TX, VX, CCU, Reshaper, DDMA_LOAD, DDMA_STORE, Broadcaster, SCU };

PARAMETER(uint32_t, tx_units, 2, "The number of TX units in the NC")
PARAMETER(uint32_t, vx_units, 2, "The number of VX units in the NC")
PARAMETER(uint32_t, reshaper_units, 1, ...)
PARAMETER(uint32_t, ddma_load_units, 1, ...)
PARAMETER(uint32_t, ddma_store_units, 1, ...)
PARAMETER(uint32_t, ccu_units, 1, ...)
PARAMETER(uint32_t, instr_queue_size, 128, ...)
PARAMETER(std::string, scheduling_type, "compiler-hint", "round-robin or compiler-hint")
PARAMETER(bool, fuse, true, "Enable instruction fusing in the dispatcher")
```

**TX 2개 · VX 2개**는 `antara-npu-internals.png` 의 NCU0/NCU1 · VX0/VX1 과 일치합니다.
그룹 JSON 트레이스의 `TX0` `TX1` `VX0` `Reshaper0` 도 이 유닛 이름입니다.

`MemoryType` 에 `SramPgasShared` · `SramPgasPrivate` · `SramCorePrivate` · `Dram` 이 있어
PGAS 기반 다중 코어 SRAM 공유를 상정한 설계임을 알 수 있습니다.
`NissNPU.h` 주석에 `default must be kept as core_id = 0 for Antara compatibility` 가 있어
**Antara 는 단일 코어**임을 시사합니다.

### 결론

**모순이 아니라 층위가 다릅니다.**

```
물리 MAC 배열    32 × 48 · 유닛당 1,536 · TX 2개로 3,072 · 6 TOPS   ← 논문
채널 타일 단위    64 · SRAM WORD_SIZE 및 가중치 대역폭과 동일        ← nbc · 시뮬레이터
```

우리가 관측한 `64 채널 정렬` 은 물리 배열이 64여서가 아니라
**메모리 워드와 타일 단위가 64여서** 생깁니다.

선행 논문 본문도 이를 뒷받침합니다.

> the Single option allows a single TX Engine to compute the output Feature Map
> in 64-channel units

`nbc` 의 `tx_row 64` 는 물리 행 수가 아니라 **OFM 채널 타일 폭**으로 읽어야 합니다.

### 남은 불일치

SRAM 이 논문 1.5 MB, 시뮬레이터와 nbc 는 2 MB 입니다.
`-antara_mpw` 플래그가 기본 1 로 켜져 있는데 MPW 는 시제품 웨이퍼 런을 뜻합니다.
리비전 차이일 가능성이 있으나 확증하지 못했습니다.
`sram_private_region_kb 2048` 이 전체와 같아 공유 영역 배분으로도 설명되지 않습니다.

### 논문 표기 권고

정렬 근거를 물리 배열이 아니라 메모리 타일로 서술해야 합니다.

> Antara 컴파일러는 출력 특징맵을 64 채널 단위로 타일링하며, SRAM 워드 크기와
> 가중치 대역폭도 64 이다. 따라서 잔여 채널이 64 의 배수가 아니면 패딩이 발생한다.

이러면 논문 Table 1 의 `32 × 48` 과 충돌하지 않고 실측 상관 0.9315 와도 맞습니다.

`antara-npu-internals.png` 의 `TX array 64 × 32` 표기는 오해를 부르므로
`TX tile 64ch × 32` 로 고치는 것이 정확합니다.
