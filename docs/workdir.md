# `./workdir` 분석

> 작성일: 2026-08-02 · 대상: `/home/osehn/workdir` (4.4GB)
> 상위 문서: [README.md](./README.md)

---

## 한 줄 요약

**Antara NPU 보드의 "밑단" — 펌웨어(RISC-V)와 호스트 PCIe 드라이버를 빌드/구동하는 공간**입니다.

홈 디렉토리의 다른 폴더들(`demo_*`, `nota*`, `yolo*`)이 **ONNX/추론 = 소프트웨어 상위 계층**이라면,
`workdir`은 그 아래에 깔린 **보드 부팅 · 펌웨어 · PCIe 통신 계층**입니다. 계층이 완전히 다릅니다.

```
┌──────────────────────────────────────┐
│ demo_1127 / nota / yolo_val …        │  ONNX·추론·양자화  (Python)
├──────────────────────────────────────┤
│ NeublaDriver.so / nbc 컴파일러        │  런타임·컴파일러
├──────────────────────────────────────┤
│ workdir  ← 여기                       │  펌웨어(RISC-V) + PCIe 드라이버(C)
└──────────────────────────────────────┘
              Antara EVB 하드웨어
```

---

## 구성

```
workdir/
├── Dockerfile              # 펌웨어 빌드 환경 (ubuntu:22.04 + pyenv 3.7 + RISC-V 빌드 의존성)
├── create_docker_img.sh    # → neubla/antara 이미지 빌드
├── docker_run.sh           # 컨테이너 실행 (RISCV_PATH / TOOL_PATH 주입, --privileged)
├── entrypoint.sh           # 컨테이너 안에서 호스트와 같은 UID/GID 유저 생성 (developers gid=1006)
├── antara_example.sh       # 호스트에서 PCIe 예제 바로 실행하는 단축 스크립트
├── hostreset.sh            # 보드 핫리셋 단축 스크립트
└── antara-software/        # 실체 (git@github.com:NeublaCorp/antara-software.git)
```

루트의 4개 셸 스크립트는 전부 **껍데기/편의 스크립트**이고, 내용물은 `antara-software/` 하나입니다.

---

## `antara-software/` — 보드 펌웨어 SDK

Neubla가 **SEMIFIVE**를 통해 받은 RISC-V SoC 소프트웨어 스택입니다.
(라이선스에 `LICENSE.SEMIFIVE`, 문서가 `gitlab.semifive.com`의 `pioneer-e-sdk-antara`를 가리킴 →
SiFive Freedom E SDK 계열을 SEMIFIVE가 Antara용으로 포팅한 것)

| 하위 폴더 | 크기 | 역할 |
|---|---|---|
| `toolchains/` | **3.8G** | RISC-V GNU 툴체인 10.2.0 (ubuntu14 / centos6 두 벌) — **폴더 용량의 86%** |
| `bootloader/` | 270M | `fsbl`(1단계: PLL 기동) + `u-boot`(2단계: DDR·PCIe 기동) |
| `antara-pcie-sdk/` | 232M | **호스트 쪽** PCIe 드라이버 + 라이브러리 + 예제 |
| `application/` | 161M | 보드에서 도는 앱 펌웨어. freedom-metal, FreeRTOS-metal, `app_fw` + 예제 45종 |
| `bin/` | 3.8M | 빌드 산출물 (2025-02-18 빌드) |
| `tools/` | 88K | GPT 이미지 생성 스크립트 + TRACE32(`.cmm`) 플래시/디버그 스크립트 |
| `doc/` | 60K | 부팅 과정 문서 + 다이어그램 |

### 빌드 흐름 (`Makefile`)

```
make [BSP=antara-evb] [PROGRAM=app_fw] [CONFIGURATION=debug]

  fsbl        → bin/fsbl.bin          (1단계 부트로더)
  u-boot-spl  → bin/u-boot-spl.bin    (2단계 부트로더)
  application → bin/app.bin           (앱 펌웨어)
        ↓  tools/mkgptimg.sh
  bin/image.gpt      (플래시용 GPT 이미지)
  bin/image_sd.gpt   (SD카드용)
```

`bin/`에 `image.gpt`, `app.itb`, `ILM.bin`/`DLM.bin`까지 이미 들어 있어
**2025-02-18에 실제로 빌드가 끝난 상태**입니다.

### 부팅 3단계

`README.md` 기준: **FSBL**(PLL) → **U-Boot SPL**(DDR + PCIe) → **App FW**(FreeRTOS 기반 응용 펌웨어).
`doc/boot_process.md`와 `the-flow-of-boot-process.png`에 흐름도가 있습니다.

---

## `antara-pcie-sdk/` — 호스트 ↔ 보드 통신

여기가 실제로 **NPU를 PC에 물려 쓰는 부분**입니다. 기반은 **PLDA** PCIe IP.

```
PCIeDriver/plda.ko          커널 모듈 (PLDA PCIe 드라이버)
quickPCIelib/qpcie_api/     유저 스페이스 API 라이브러리
quickPCIe_examples/
  ├── antara_example/       기본 예제 (legacy INT / msi / msix 모드)
  ├── antara_ep_example/    EP(Endpoint) 모드 예제 ← 홈의 "ST-Antara EP mode build guide.pdf"와 짝
  ├── NE_TC__#001~#004/     NE(Neural Engine) 테스트 케이스 — ILM/DLM/instruction/input/ref 바이너리
  ├── hotreset.sh           lspci로 PLDA 장치 찾아 핫리셋 + 드라이버 재로드
  └── qpcie / vdma / xr3 예제
quickPCIe_tests/            테스트 스위트
sanity_test/, pcie_dram_harsh_test/   안정성·DRAM 스트레스 테스트
documentation/              PLDA 공식 PDF 5종 (API 유저가이드, 드라이버 설치가이드 등)
neubla_driver_origin/       ← 홈의 동명 폴더와 같은 것 (여기에도 사본이 들어와 있음)
```

실행 절차 (`run.sh`):

```bash
sudo rmmod plda.ko
sudo insmod ./PCIeDriver/plda.ko o_dbg=8 max_irq=32   # max_irq=32 → MSI 32개
export LD_LIBRARY_PATH=$PWD/quickPCIelib/qpcie_api/release:$LD_LIBRARY_PATH
sudo ./quickPCIe_examples/antara_example/build/antara_example msi
```

---

## 정리하며 눈에 띈 점

1. **`toolchains/` 3.8GB가 전체의 86%.** 그것도 `ubuntu14`/`centos6` 두 벌인데, `docker_run.sh`는 **ubuntu14 쪽만** `RISCV_PATH`로 잡습니다. centos6 툴체인은 이 환경에서 안 쓰입니다.

2. **`neubla_driver_origin/`이 두 군데 있습니다.** `~/neubla_driver_origin`(131M)과 `antara-pcie-sdk/neubla_driver_origin`. SDK 안쪽이 원래 위치로 보이고, 홈 루트 쪽이 꺼내둔 사본으로 보입니다.

3. **날짜로 본 작업 흐름**
   - 2024-03/04: Docker 환경 + `antara-software` 원본 배포 (v0.2, 2024-01-02 릴리스)
   - 2025-01-06: PCIe SDK 쪽 손댐 (예제 빌드, 드라이버 빌드, 단축 스크립트 3개 추가)
   - 2025-02-18: 펌웨어 전체 빌드 (`bin/` 산출물)
   - 2025-04-22: 마지막 수정

4. **`workdir`은 사실상 읽기 전용 SDK에 가깝습니다.** 직접 작성한 코드는 루트의 셸 스크립트 4개 정도이고, 나머지는 벤더(SEMIFIVE/PLDA/SiFive)에서 받은 배포본입니다. 백업/정리 시 우선순위가 낮은 대신, **보드가 안 잡히거나 리셋이 필요할 때 반드시 돌아와야 하는 폴더**입니다.

5. **문제 생겼을 때 볼 파일**
   - 보드 인식 안 됨 → `antara-pcie-sdk/run.sh`, `quickPCIe_examples/hotreset.sh`
   - 부팅 단계 이해 → `antara-software/doc/boot_process.md`
   - 펌웨어 재빌드 → `antara-software/README.md` + 루트 `docker_run.sh`
   - NE(NPU 코어) 자체 검증 → `quickPCIe_examples/NE_TC__#001~004`, `sorting_ne_sanity.sh`
   - `doc/DeveloperGuide.md`의 링크는 전부 사내 `gitlab.semifive.com:8888`이라 **외부에서 접근 불가**
