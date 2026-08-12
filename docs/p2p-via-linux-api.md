# Linux P2P API로 이 기능을 추가할 수 있는가

> 작성일: 2026-08-02 · 선행: [p2p-support.md](./p2p-support.md), [p2p-where-to-implement.md](./p2p-where-to-implement.md)

---

## 먼저 짚을 것 — 펌웨어에는 Linux가 없습니다

Antara 보드에서 도는 건 **FreeRTOS + freedom-metal (베어메탈 RISC-V)** 입니다.

```
application/
├── freedom-metal/       ← SiFive 베어메탈 HAL
├── FreeRTOS-metal/      ← RTOS
└── software/app_fw/     ← 응용 펌웨어
```

Linux 커널이 없으므로 **펌웨어에 Linux P2P API를 넣는 것은 불가능**합니다.
Linux P2PDMA API(`pci_p2pdma_*`)는 **호스트 쪽 커널 모듈 — 즉 `plda.ko` — 에 들어갑니다.**

질문을 다시 쓰면: **"호스트 드라이버 `plda.ko`에 Linux P2PDMA API를 넣어 P2P를 구현할 수 있는가?"**
→ **방향에 따라 다릅니다. 한 방향은 됩니다(펌웨어 수정 불필요). 다른 방향은 Linux API가 해결해 주지 않습니다.**

---

## 전제 조건은 전부 충족되어 있다

| 항목 | 상태 |
|---|---|
| `CONFIG_PCI_P2PDMA` | **`=y`** |
| `CONFIG_ZONE_DEVICE` | **`=y`** |
| P2PDMA 심볼 모듈 export | **전부 `EXPORT_SYMBOL_GPL`** |
| `plda.ko` 라이선스 | **`GPL v2`** → GPL-only 심볼 링크 **가능** |
| Antara BAR | BAR2 256M / BAR4 512M, **전부 prefetchable** |

`Module.symvers`에서 확인한 사용 가능 심볼:

```
pci_p2pdma_add_resource      pci_p2pmem_alloc_sgl     pci_alloc_p2pmem
pci_p2pdma_distance_many     pci_p2pmem_free_sgl      pci_free_p2pmem
pci_p2pmem_find_many         pci_p2pmem_publish       pci_p2pmem_virt_to_bus
pci_has_p2pmem               dma_pci_p2pdma_supported pci_p2pdma_enable_store
```

**라이선스 벽이 없다는 게 큰 행운입니다.** PLDA 드라이버가 GPL v2라서 GPL-only 심볼을 그대로 쓸 수 있습니다.
(현재 `/sys/bus/pci/devices/0000:01:00.0/`에 `p2pmem`이 없는 건, 아직 아무도 등록을 안 했다는 뜻일 뿐입니다.)

---

## 방향 A — Antara를 P2P "제공자"로: **Linux API가 정확히 이 용도. 펌웨어 무관**

**다른 장치가 Antara BAR로 직접 DMA를 넣는** 방향입니다.
DMA를 수행하는 주체가 상대 장치이므로 **Antara 펌웨어도, U-Boot의 DMA 방향 설정도 건드릴 필요가 없습니다.**

`plda.ko`의 probe에 추가할 것:

```c
/* BAR4(512M) 중 일부를 P2P 메모리로 등록 */
err = pci_p2pdma_add_resource(pDev, BAR4, p2p_size, p2p_offset);
if (err) { ... }

pci_p2pmem_publish(pDev, true);   /* 다른 드라이버가 찾을 수 있게 공개 */
```

이러면 커널이 그 BAR 영역에 `struct page` 백킹(`MEMORY_DEVICE_PCI_P2PDMA`)을 만들고,
NVMe 등 클라이언트 드라이버가 `pci_p2pmem_find_many()` → `pci_p2pmem_alloc_sgl()` 로 할당해
**호스트 RAM을 거치지 않고** Antara BAR에 직접 DMA 할 수 있게 됩니다.

### 주의: 현재 드라이버와 충돌합니다

`plda_pcie.c:557~570`이 **모든 BAR을 `pci_request_regions` + `pci_iomap`** 합니다:

```c
if ( (err = pci_request_regions(pDev, DRIVER_NAME)) ) ...
bars[bar].kernel_address = pci_iomap(pDev, bar, bars[bar].length);
```

`pci_p2pdma_add_resource()`는 해당 영역을 `devm_memremap_pages()`로 잡으므로 겹치면 안 됩니다.
→ **P2P로 내줄 영역은 `pci_iomap` 대상에서 빼거나**, `offset`/`size` 인자로 BAR의 일부만 떼어내야 합니다.
BAR4를 통째로 내주면 NE(NPU 코어) 접근 창이 줄어들 수 있으니 **부분 할당이 안전**합니다.

---

## 방향 B — Antara가 P2P "개시자"로: **Linux API가 해결해 주지 않습니다**

**Antara의 DMA 엔진이 상대 장치 BAR을 읽고 쓰는** 방향입니다.
이쪽은 Linux P2PDMA API가 낄 자리가 거의 없습니다. 이유:

**Antara의 DMA 엔진은 커널 DMA API를 지나가지 않습니다.**
유저스페이스(`libpldaqpcie`)가 BAR0의 레지스터(`REG_DMA*_SRCADDR/DESTADDR/CONTROL`)를 직접 써서 프로그래밍합니다.
커널은 `/dev/plda0` mmap을 열어주는 통로일 뿐, 전송 경로에 관여하지 않습니다.
따라서 `pci_p2pdma_map_sg()` 같은 매핑 API를 끼워 넣을 지점 자체가 없습니다.

Linux API에서 이 방향에 쓸 수 있는 건 **보조 역할** 둘뿐입니다:

| API | 용도 |
|---|---|
| `pci_p2pdma_distance_many()` | "이 두 장치가 P2P 가능한 토폴로지인가" 판정 |
| `pci_p2pmem_virt_to_bus()` | 상대 BAR의 **버스 주소** 획득 → `T_P2PTarget.TargetAddr`에 넣을 값 |

**실제 전송을 뚫는 문제는 그대로 남습니다** — [p2p-where-to-implement.md](./p2p-where-to-implement.md)에 적은
U-Boot `pcie_ctl_dma_default_setting()`의 채널 방향 고정, 그리고 호스트 API의 `E_PEER_TO_PEER` 거부.

---

## 진짜 벽 — 호스트 브리지가 화이트리스트에 없습니다

```
$ lspci -nn -s 00:00.0
00:00.0 Host bridge [0600]: Intel Corporation Device [8086:4668] (rev 02)
```

`8086:4668` = AlderLake-S.

커널 `drivers/pci/p2pdma.c`의 `p2pdma_whitelist[]`에는 루트 컴플렉스를 통과하는 P2P가 검증된
호스트 브리지 ID만 들어 있습니다 — Intel Xeon E5/E7, SkyLake-E 계열(0x3c00, 0x3c01, 0x2f00, 0x2f01,
0x2020, 0x2030~0x2033, 0x09a2 등)과 AMD Zen. **`0x4668`은 없습니다.**

> 이 목록은 커널 소스 기준이며, 이 장비에는 헤더만 있고 소스가 없어 로컬에서 직접 확인하지는 못했습니다.
> 다만 AlderLake-S 같은 소비자용 칩이 화이트리스트에 있을 가능성은 사실상 없습니다.

화이트리스트에 없으면 커널은 **두 장치가 공통 PCIe 스위치 아래 있을 때만** P2P를 허용합니다.
현재 토폴로지는 정반대입니다:

```
-[0000:00]-+-01.0-[01]----00.0  PLDA XpressRich-AXI   ← 루트포트 A
           +-06.0-[02]----00.0  Samsung NVMe SSD      ← 루트포트 B (다른 포트!)
```

→ `pci_p2pdma_distance_many()`가 **`PCI_P2PDMA_MAP_NOT_SUPPORTED`를 반환하고 커널이 거부**합니다.

### 해결책

**PCIe 스위치를 하나 넣고 Antara와 상대 장치를 그 아래로 모으면** 됩니다.
공통 upstream bridge를 경유하는 경로가 되어 호스트 브리지 화이트리스트와 무관하게 통과합니다
(`PCI_P2PDMA_MAP_BUS_ADDR`). 단 그 스위치의 **ACS redirect는 꺼야** 합니다.

---

## NVIDIA GPU를 붙일 계획이라면 — 또 다른 길

```
$ ls /usr/src | grep nvidia
nvidia-550.144.03          ← 드라이버 소스는 설치되어 있음
$ lspci | grep -i "VGA\|3D"
00:02.0 VGA compatible controller: Intel AlderLake-S GT1   ← 내장만, dGPU 없음
```

NVIDIA 드라이버 소스는 깔려 있는데 정작 dGPU가 없습니다.

만약 NVIDIA dGPU를 추가한다면 **일반 Linux P2PDMA 경로가 아닙니다.**
NVIDIA 드라이버는 GPU BAR을 일반 P2PDMA provider로 등록하지 않으므로,
`pci_p2pmem_find_many()`로는 GPU를 찾을 수 없습니다.
GPU↔NPU는 **`nvidia-peermem` / GPUDirect RDMA** 경로를 써야 하고,
`plda.ko`에서 `nvidia_p2p_get_pages()` 계열 API를 호출하는 **별도 구현**이 필요합니다.

---

## 정리

| 질문 | 답 |
|---|---|
| 펌웨어에 Linux P2P API를 넣을 수 있나 | **불가.** 펌웨어는 FreeRTOS 베어메탈, Linux 없음 |
| 호스트 드라이버(`plda.ko`)에 넣을 수 있나 | **가능.** GPL v2 + 심볼 export + config 전부 충족 |
| 그러면 P2P가 되나 | **방향 A(Antara가 타깃)는 됨. 펌웨어 수정 불필요** |
| 방향 B(Antara가 개시자)는? | Linux API로는 안 됨. DMA 엔진이 커널을 안 지남 → U-Boot/레지스터 문제로 회귀 |
| 지금 장비에서 바로 되나 | **아니오.** 호스트 브리지 미화이트리스트 + 서로 다른 루트 포트 → 커널이 거부 |
| 무엇이 필요한가 | **PCIe 스위치 아래로 두 장치 배치** (+ ACS redirect off), 그리고 **P2P 상대 장치 자체** |

### 실행 순서를 잡는다면

1. **먼저 방향 A만 구현** — `plda.ko` probe에 `pci_p2pdma_add_resource(pDev, BAR4, size, offset)` + `pci_p2pmem_publish()`. BAR iomap 충돌만 피하면 30줄 안쪽. 펌웨어 무관하므로 리스크가 낮습니다.
2. **`pci_has_p2pmem()` / sysfs `p2pmem/`로 등록 확인** — 여기까지는 상대 장치 없이도 검증 가능합니다.
3. **토폴로지 정리** — PCIe 스위치 도입, 상대 장치 확보
4. **`pci_p2pdma_distance_many()`로 판정** — 여기서 통과하면 실제 전송 시도
5. 방향 B가 꼭 필요하면 그때 U-Boot 수정([p2p-where-to-implement.md](./p2p-where-to-implement.md) 참조)

**1~2단계는 지금 장비로도 당장 가능하고, 나머지는 하드웨어 구성이 선행되어야 합니다.**
