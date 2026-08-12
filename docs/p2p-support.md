# Antara SDK에 P2P(PCIe peer-to-peer) 기능이 있는가

> 작성일: 2026-08-02 · 관련: [workdir.md](./workdir.md)
> 모든 주장은 소스 라인·바이너리 역어셈블·실장비 조회로 확인한 것입니다.

---

## 결론

**API에는 있습니다. 단, 아무도 쓰지 않고 있고 현재 장비 구성으로는 시도 자체가 어렵습니다.**

| 계층 | P2P 존재 | 비고 |
|---|:---:|---|
| `plda_qpcie_api.h` 공개 API | **○** | `T_P2PTarget` + `QPCIe_ProgramDmaEngine` 전용 오버로드 |
| 그 오버로드의 `E_PEER_TO_PEER` 방향 | **×** | 코드가 **명시적으로 거부** |
| vDMA 계층 (`E_PEER_TO_PEER` 채널) | **○** | PLDA 범용 IP. **Antara는 vDMA를 안 씀** |
| Antara 예제·테스트 실사용 | **×** | 호출 0건 |
| `NeublaDriver.so` (ML 런타임) 실사용 | **×** | 심볼은 있으나 **호출부 0건** |
| SDK 공식 문서 (PDF 5종) | **×** | 언급 0건 |
| `plda.ko` 커널 P2PDMA 사용 | **×** | `pci_p2pdma_*` 미사용 |

---

## 1. API에는 분명히 있다

`quickPCIelib/qpcie_api/include/plda_qpcie_api.h:145`

```c
typedef struct S_P2PTarget {
    uint64_t TargetAddr;   // 타깃 PCIe 주소 (= 다른 장치의 BAR 주소를 넣을 수 있음)
    uint32_t TargetSize;
} T_P2PTarget;
```

같은 헤더 `:886` — `QPCIe_ProgramDmaEngine`의 **전용 오버로드**:

```c
uint32_t QPCIe_ProgramDmaEngine(
    const uint32_t                  device_index,
    const uint32_t                  dma_index,
    const T_DmaDirection            dma_direction,
    const std::vector<T_P2PTarget>& p2p_target_list,   // ← P2P 타깃 목록
    const T_DmaInterface            axi4_interface );
```

구현(`plda_qpcie_api.cpp:1983~2160`)은 이 주소들을 **SG-DMA 디스크립터에 직접 박아 넣습니다**:

```c
descInfo[i].SrcAddrL = (uint32_t) p2p_target_list[i].TargetAddr & 0xFFFFFFFF;
descInfo[i].SrcAddrH = (uint32_t)(p2p_target_list[i].TargetAddr >> 32);
```

즉 **호스트가 할당한 버퍼가 아니라 임의의 PCIe 물리주소**를 DMA 소스/목적지로 지정할 수 있습니다.
그 주소가 다른 장치의 BAR이면 그게 곧 P2P입니다.

---

## 2. 그런데 "방향으로서의 P2P"는 이 함수가 거부한다

`plda_qpcie_api.cpp:2059~2080`

```c
switch ( dma_direction ) {
    case E_HOST_TO_CARD:
        if ( src != PCIe || (dest < AXI4M0 || dest > AXI4S3) ) return QPCIe_INVALID_PARAMETER;
        break;
    case E_CARD_TO_HOST:
        if ( dest != PCIe || (src < AXI4M0 || src > AXI4S3) ) return QPCIe_INVALID_PARAMETER;
        break;
    case E_PEER_TO_PEER:          // ←←←
    case E_HOST_TO_HOST:
    case E_MIX_DIRECTIONS:
    case E_DUPLEX_DIRECTIONS:
        PLDA_WRITE_LOG( ERR, "E_DUPLEX_DIRECTIONS, E_MIX_DIRECTIONS, E_PEER_TO_PEER and "
                             "E_HOST_TO_HOST could not be used with this function\n" );
        return QPCIe_INVALID_PARAMETER;
}
```

**정리하면 이 API의 "P2P"는 반쪽짜리입니다.**

| | 소스 | 목적지 |
|---|---|---|
| `E_HOST_TO_CARD` + P2P 목록 | **임의 PCIe 주소(= peer BAR)** | 카드 AXI (DDR/NE) |
| `E_CARD_TO_HOST` + P2P 목록 | 카드 AXI | **임의 PCIe 주소(= peer BAR)** |
| `E_PEER_TO_PEER` (양쪽 다 PCIe) | ✗ 거부 | ✗ 거부 |

한쪽 끝이 항상 Antara 자신이어야 합니다.
**"NPU DMA 엔진이 GPU BAR을 직접 읽어 온다"** 같은 건 가능하지만,
**"카드 A → 카드 B, Antara는 관여 안 함"** 은 이 함수로 안 됩니다.

---

## 3. 완전한 P2P는 vDMA 계층에만 있고, Antara는 vDMA를 안 쓴다

`quickPCIe_tests/vdma/` 아래에는 `E_PEER_TO_PEER`가 **채널 방향**으로 제대로 구현돼 있습니다 —
latency 테스트, throughput 테스트(1/32 채널 × 80/512/4096 바이트), 디스크립터 레지스터 매핑까지:

```cpp
// vdma_axi_channel.cpp:115
case E_PEER_TO_PEER:
    channel_reg.emplace_back(this->src_channel_regs, addr);
    channel_reg.emplace_back(this->dst_channel_regs,
                             addr + vDMA_AXI_Descriptor::getDestinationDescriptorOffset());
```

**하지만 이건 PLDA XpressRich 레퍼런스 디자인용 범용 테스트 스위트입니다.**
Antara 쪽 코드는 vDMA API(`QPCIe_OpenVDmaChannel`)를 **단 한 번도 호출하지 않습니다.**
`antara_example`이 쓰는 API 전체:

```
QPCIe_Init / Uninit / AllocateBuffer / ReleaseBuffer
QPCIe_ProgramDmaEngine / StartDmaTransfer / AbortDmaTransfer
QPCIe_ReadMemory / WriteMemory / ReadDeviceRegister
QPCIe_SetupInterrupts / WaitForInterrupt / ClearInterrupts / GetIrqDirection
QPCIe_ReadBarUserAddr / ReadDeviceInfo / GetAPIVersion
```
→ vDMA 계열 함수 **0개**. 평범한 SG-DMA 엔진만 씁니다.

---

## 4. 실사용은 전무하다 — 바이너리로 확인

SDK 전체의 `QPCIe_ProgramDmaEngine` 호출 **20군데가 전부** 버퍼 오버로드 + `E_HOST_TO_CARD`/`E_CARD_TO_HOST`입니다.
`quickPCIe_examples`에서 `T_P2PTarget`/`E_PEER_TO_PEER` 검색 결과 **0건**.

ML 런타임 쪽도 확인했습니다. `NeublaDriver.so`에 P2P 오버로드 **심볼은 있습니다**:

```
$ nm -DC NeublaDriver.cpython-38-x86_64-linux-gnu.so | grep ProgramDmaEngine
0000000000023f20 T QPCIe_ProgramDmaEngine(uint, uint, T_DmaDirection, void const*, ...)
0000000000024410 T QPCIe_ProgramDmaEngine(uint, uint, T_DmaDirection,
                                          std::vector<S_P2PTarget,...> const&, T_DmaInterface)
```

`T`(정의됨)인 이유는 `ldd`에 `libpldaqpcie`가 없는 데서 알 수 있듯 **qpcie 라이브러리를 통째로 정적 링크**했기 때문입니다.
**심볼이 있다 ≠ 호출한다** 이므로 역어셈블로 호출부를 직접 셌습니다:

```
$ objdump -dC NeublaDriver...so | grep "call.*ProgramDmaEngine"
18bdf:  call 9840 <QPCIe_ProgramDmaEngine(uint, uint, T_DmaDirection, void const*, ...)@plt>
18ee2:  call 9840 <QPCIe_ProgramDmaEngine(uint, uint, T_DmaDirection, void const*, ...)@plt>
```

**버퍼 오버로드만 2번 호출, P2P 오버로드는 호출부 0건.**
정적 링크로 딸려 들어왔을 뿐 실제로는 안 씁니다.

### 문서에도 없다

```
0  documentation/linux_driver_revision_history_v286.pdf
0  documentation/linux_sdk_revision_history_v325.pdf
0  documentation/qpcie_api_revision_history_v255.pdf
0  documentation/quickpcie_api_user_guide_v255.pdf
0  documentation/software_driver_installation_guide_v128.pdf
```
(`peer|p2p` 대소문자 무시 검색 결과. API User Guide는 v2.5.5 / 2020-02)

→ P2P 오버로드는 **문서보다 나중에 추가된 미문서화 기능**입니다.

### 드라이버도 커널 P2PDMA를 안 쓴다

`plda.ko`의 strings에 `MEMORY_DEVICE_PCI_P2PDMA`, `pci_p2pdma`, `FOLL_PCI_P2PDMA`가 보이지만,
소스(`PCIeDriver/src/`)에는 해당 API 호출이 **하나도 없습니다.** 그 문자열들은 BTF 타입 정보일 뿐입니다.
즉 이 SDK의 P2P는 커널 도움 없이 **유저스페이스가 물리주소를 디스크립터에 직접 박는** 방식입니다.
(주소 검증도 IOMMU 매핑도 없다는 뜻 — 잘못 넣으면 그대로 엉뚱한 곳에 씁니다.)

---

## 5. 현재 장비로는 시도 자체가 어렵다

```
$ lspci -tv
-[0000:00]-+-01.0-[01]----00.0  PLDA XpressRich-AXI Ref Design   ← NPU
           +-02.0              Intel AlderLake-S GT1            ← 내장 그래픽
           +-06.0-[02]----00.0  Samsung NVMe SSD                 ← 유일한 P2P 후보
           +-1c.2-[04]----00.0  Realtek RTL8125 2.5GbE
```

세 가지 장벽:

1. **dGPU가 없습니다.** P2P의 주 용도인 GPU↔NPU 직접 전송을 할 상대가 아예 없습니다. 내장 GT1은 시스템 메모리를 쓰므로 P2P 대상이 아닙니다.
2. **NPU와 NVMe가 서로 다른 루트 포트 아래**(`00:01.0` vs `00:06.0`)입니다. P2P TLP가 CPU 루트 컴플렉스를 관통해야 하는데, 소비자용 Intel(AlderLake-S)은 루트포트 간 P2P 지원이 부실하고 성능도 나쁩니다. PCIe 스위치 한 개 아래로 모으는 게 정석입니다.
3. **검증 경로가 없습니다.** 예제도 테스트도 문서도 없어서, 동작하지 않을 때 SDK 문제인지 하드웨어 문제인지 구분할 기준이 없습니다.

참고로 환경 자체는 나쁘지 않습니다:
- 호스트 커널 `CONFIG_PCI_P2PDMA=y` (다만 이 드라이버는 그 경로를 안 씀)
- IOMMU 비활성 (`/proc/cmdline`에 옵션 없음, `/sys/class/iommu/` 비어있음) → raw 물리주소 방식엔 오히려 유리

---

## 6. 반대로 — Antara를 P2P **대상**으로 쓰는 건 조건이 좋다

```
$ lspci -vv -s 01:00.0
Region 0: Memory at 90000000 (64-bit, prefetchable) [size=16K]
Region 2: Memory at 80000000 (64-bit, prefetchable) [size=256M]
Region 4: Memory at 60000000 (64-bit, prefetchable) [size=512M]
```

BAR가 크고(256M + 512M) **전부 prefetchable**입니다.
P2P에서 타깃이 되려면 BAR가 prefetchable해야 하는데 그 조건을 이미 만족합니다.
즉 **"다른 장치가 Antara BAR로 직접 밀어 넣는" 방향이 오히려 더 현실적**입니다.

---

## 7. 그래도 시도한다면

```cpp
std::vector<T_P2PTarget> targets;
targets.push_back({ peer_bar_phys_addr, transfer_size });   // 상대 장치 BAR 물리주소

QPCIe_ProgramDmaEngine(board, dma_idx,
                       E_CARD_TO_HOST,      // ← E_PEER_TO_PEER 아님! 거부당함
                       targets,
                       E_AXI4_Master0);
QPCIe_StartDmaTransfer(...);
```

준비 사항:
1. 상대 장치 BAR의 **물리주소** 확보 — `/sys/bus/pci/devices/<BDF>/resource` 또는 `lspci -vv`
2. 상대 BAR가 **prefetchable + 64-bit**인지 확인
3. 가능하면 두 장치를 **같은 PCIe 스위치 아래**로. 안 되면 ACS 완화(`pcie_acs_override=downstream` 등) 검토
4. `sanity_test/sanity_test.cpp`를 본떠 **작은 크기부터** 검증 — 주소 검증이 전혀 없으므로 잘못된 주소는 시스템을 죽일 수 있음
5. 로그는 `insmod plda.ko o_dbg=8`로 올려 dmesg 확인

**요약: 기능은 API에 잠재해 있지만 미문서화·미검증·미사용이고, 현재 하드웨어에는 P2P 상대가 없습니다.
필요하다면 dGPU 추가 + 동일 스위치 배치부터 선행되어야 합니다.**
