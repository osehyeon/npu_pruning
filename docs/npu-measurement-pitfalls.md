# NPU 측정 함정 — 재는 조건이 결과를 바꿉니다

2026-08-11 확인. **여기 적힌 두 가지를 모르면 NPU 수치가 통째로 틀립니다.**

---

## 1. 보드 펌웨어의 UART 디버그 출력 (`dbgprint`)

### 증상

지연이 **모델과 무관하게 약 31 ms** 부풀고, 그 결과 **가속 배수가 희석**됩니다.

| 모델 (`maD20` · `lw`) | dbgprint **ON** | dbgprint **OFF** | 차이 |
|---|---:|---:|---:|
| 원본 | 90.5 ms | **58.3 ms** | −35.6 % |
| 40 % 프루닝 | 76.4 ms | **44.8 ms** | −41.4 % |
| 50 % 프루닝 | 74.8 ms | **43.1 ms** | −42.4 % |
| **가속 배수** | 1.18× / 1.21× | **1.30× / 1.35×** | |

오버헤드가 분모·분자에 함께 들어가므로 **프루닝 이득이 실제보다 작게 보입니다.**

### 원인

`app_fw/terminal_service/terminal_task.c` 의 기본값입니다.

```c
static uint8_t debug_print_uart              = APP_ON;             // 기본 켜짐
static debug_log_level_t dbg_log_threshold_uart = LOG_LEVEL_VERBOSE;  // 최대 상세
```

추론 경로(`app_fw/ne/`)에 로그 호출이 **151개** 있습니다 — `LOG_INFO` 46 · `LOG_ERROR` 40 ·
`LOG_DEBUG` 35 · `printf` 32. 115200 bps 에서 한 줄에 수 ms 이므로 추론마다 누적됩니다.

### 끄는 법

```bash
minicom -D /dev/ttyUSB0
```

```
dbgprint                 # 인자 없으면 현재 상태만 출력
dbgprint uart off        # 끄기
```

종료는 `Ctrl-A` → `X`.

**보드가 재부팅되면 기본값(ON)으로 돌아갑니다.** 측정 전마다 확인하세요.

### 호스트 쪽 `o_dbg` 와 다릅니다

| | 위치 | 영향 | 확인 |
|---|---|---|---|
| `o_dbg` | 호스트 PLDA 드라이버 | 커널 로그로 **디스크**를 채움 | `cat /sys/module/plda/parameters/o_dbg` |
| `dbgprint` | **보드 펌웨어** | UART 출력으로 **지연** 증가 | `minicom` → `dbgprint` |

둘 다 꺼야 합니다. `load_driver.sh` 는 `o_dbg=0` 이 기본이지만 `dbgprint` 는 손대지 않습니다.

---

## 2. DMA 고장 — 재부팅만이 답입니다

### 증상

```
LoadModel: 1                        (실패)
munmap_chunk(): invalid pointer     (그 직후 Close() 에서 크래시)
```

`Init()` 은 0(성공)인데 `LoadModel()` 이 실패합니다. 어떤 `.o` 를 써도 같습니다.

> `munmap_chunk()` 는 라이브러리의 이중 해제 버그일 뿐입니다.
> **진짜 원인은 그 위 출력**(`LoadModel` 반환값)을 봐야 합니다.

### 듣지 않는 조치

- `load_driver.sh -r` (드라이버 재로드) — ✗
- `sudo hostreset.sh` (PCIe hot reset) — ✗

PCIe hot reset 은 엔드포인트 레지스터만 리셋하고 **Antara SoC 는 계속 돌아** U-Boot 이
재실행되지 않습니다. DMA 방향 설정은 U-Boot 이 **보드 부팅 때 한 번만** 합니다.

### 듣는 조치

**컴퓨터 재부팅.** 슬롯에 PERST# 가 걸려 보드가 재부팅되고 U-Boot 이 DMA 를 다시 설정합니다.

재부팅 후:

```bash
minicom -D /dev/ttyUSB0     # dbgprint uart off  ← 잊지 말 것
~/workdir/load_driver.sh
```

→ 상세: [after-reboot.md](./after-reboot.md)

---

## 3. 실측 — staircase 와 하드웨어 정렬

`dbgprint` OFF · `maD20` 양자화 · `lw` 컴파일 · 50 % 프루닝 4종.

| sc | raw MACs (G) | padded MACs (G) | 64배수 conv | 바이너리 | **NPU (ms)** |
|---:|---:|---:|---:|---:|---:|
| 8 | **38.2** | 61.9 | 18/75 | 112 MB | 48.5 |
| 16 | 38.1 | 60.2 | 32/75 | 111 MB | 48.5 |
| 32 | 39.0 | 59.2 | 52/75 | 109 MB | 48.4 |
| **64** | **42.0** | **58.9** | **70/75** | **105 MB** | **43.5** |

**sc64 는 raw MACs 가 가장 많은데 가장 빠릅니다.** 연산량 지표만 보면 정반대 결론이 나옵니다.

`padded MACs = Σ ⌈Cout/64⌉·64 · ⌈Cin/64⌉·64 · k² · H · W`

### 2025-03 원 측정과의 대조

```
2025-03   sc8/16/32 ≈ 0.0462~0.0463 s → sc64 0.0413 s    개선 10.6 %
2026-08   sc8/16/32 ≈ 48.4~48.5 ms   → sc64 43.5 ms      개선 10.2 %
```

**1년 5개월 간격의 독립 측정이 개선폭까지 일치**합니다. 절대값은 당시 `dbgprint` 상태를
모르므로 비교하지 않습니다.

---

## 4. 컴파일 옵션 `egs`

50 %/sc64 기준:

```
lw    43.5 ms   105 MB
egs   22.3 ms    92 MB      1.88×  (연산 구간만 1.98×)
```

**주의**: 비정렬 모델(sc8/16/32)에서는 `egs` 컴파일이 `Unmatched data` 로 실패했습니다.
다만 해당 4종은 **재학습 전 상태로 유효한 출력을 내지 못하므로**(AP 0.03, 검출 0개)
정렬이 원인인지 검증 단계 산물인지 **분리하지 못했습니다.** 일반화하지 마세요.

---

## 측정 체크리스트

```
□ minicom → dbgprint uart off          ← 재부팅 때마다
□ ~/workdir/load_driver.sh  (o_dbg=0 확인)
□ LoadModel 실패 시 → 컴퓨터 재부팅 (hostreset 아님)
□ 모델마다 프로세스를 새로 띄울 것          ← run_bench.sh 가 처리
□ 워밍업 1회 제외, 10회 평균                ← bench_npu.py 기본값
```

한 프로세스에서 두 번째 모델을 로드하면 DMA 오류가 납니다. `Close()` 를 불러도 같습니다.
