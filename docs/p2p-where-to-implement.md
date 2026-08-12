# P2P를 하려면 펌웨어부터 만들어야 하는가?

> 작성일: 2026-08-02 · 선행: [p2p-support.md](./p2p-support.md)
> 모든 주장은 소스 라인으로 확인한 것입니다.

---

## 결론

**펌웨어부터 새로 만들 필요는 없습니다.**
하드웨어는 P2P가 가능한 구조로 열려 있고, **부팅 때 펌웨어가 Host↔Card로 못박아 놓은 것**입니다.
그 못은 런타임에 뽑을 수 있는 종류(그냥 레지스터 4비트)입니다.

| 계층 | 상태 | 근거 |
|---|---|---|
| **하드웨어 (PLDA XpressRich-AXI IP)** | **열려 있음** | src/dest를 4비트 ID로 지정. `DMA_IF_PCIE=0x0` 존재, 양쪽 PCIe를 막는 구조 없음. 양방향 SG 모드(`DMA_SG_TYPE_SRCDST_SG=0x3`)까지 정의됨 |
| **펌웨어 (U-Boot SPL)** | **막고 있음** | `pcie_ctl_dma_default_setting()`이 4채널 전부 Host↔Card로 고정 |
| **호스트 API** | **막고 있음** | `E_PEER_TO_PEER` 방향 명시적 거부 |

즉 **"구현되어 있는데 안 쓰는 것"에 가깝습니다.** 다만 "쓰기만 하면 되는" 상태는 아니고,
위 두 층의 못을 뽑아야 합니다.

---

## 1. 펌웨어가 실제로 하는 일 — 여기가 진짜 장벽

`bootloader/u-boot/arch/riscv/cpu/sesame/pcie_ep_drv/device/pcie_ep_ctl.c:241`

```c
void pcie_ctl_dma_default_setting(unsigned int base)
{
    dma_srcid_set(base, 0, DMA_IF_PCIE);       dma_dstid_set(base, 0, DMA_IF_AXI4_MST0);
    dma_sg_set   (base, 0, DMA_SG_TYPE_SRC_SG, DMA_SG_ID_PCIE);

    dma_srcid_set(base, 1, DMA_IF_AXI4_MST0);  dma_dstid_set(base, 1, DMA_IF_PCIE);
    dma_sg_set   (base, 1, DMA_SG_TYPE_DST_SG, DMA_SG_ID_PCIE);

    dma_srcid_set(base, 2, DMA_IF_PCIE);       dma_dstid_set(base, 2, DMA_IF_AXI4_MST0);
    dma_sg_set   (base, 2, DMA_SG_TYPE_SRC_SG, DMA_SG_ID_PCIE);

    dma_srcid_set(base, 3, DMA_IF_AXI4_MST0);  dma_dstid_set(base, 3, DMA_IF_PCIE);
    dma_sg_set   (base, 3, DMA_SG_TYPE_DST_SG, DMA_SG_ID_PCIE);
}
```

| DMA 채널 | SRC | DST | 방향 |
|:---:|---|---|---|
| 0 | PCIe | AXI4_MST0 | Host → Card |
| 1 | AXI4_MST0 | PCIe | Card → Host |
| 2 | PCIe | AXI4_MST0 | Host → Card |
| 3 | AXI4_MST0 | PCIe | Card → Host |

**PCIe→PCIe 채널이 하나도 없습니다.** 이게 P2P가 안 되는 실질적 이유입니다.

호출 경로: `pcie_ep_driver.c:225` → **U-Boot SPL에 포함**
(`spl/u-boot-spl.map:6597`에 `0x080146ce pcie_ctl_dma_default_setting` — 빌드된 바이너리에 실제로 들어있음)

그래서 호스트 API가 SRCPARAM/DESTPARAM을 **읽어서 검증만** 하는 것입니다
(`plda_qpcie_api.cpp:2044~2078`) — "이 채널은 이미 이렇게 배선돼 있다"를 전제하고 어긋나면 거부합니다.

---

## 2. 그런데 그 "배선"은 그냥 레지스터 4비트다

```c
void dma_srcid_set(unsigned int base, unsigned int idx, unsigned int id)
{
    src_param = readl(DMA_ENGINE(base, idx) + DMA_ENGINE_DMA_SRCPARAM);
    src_param = (src_param & ~0xF) | (0xF & id);      // 하위 4비트만 교체
    writel(src_param, DMA_ENGINE(base, idx) + DMA_ENGINE_DMA_SRCPARAM);
}
```

RTL에 굳어 있는 게 아니라 **런타임에 바꿀 수 있는 레지스터**입니다.
그리고 IP는 PCIe를 다른 인터페이스와 동등한 하나의 ID로 취급합니다:

```c
#define DMA_IF_PCIE        0x0     // ← src에도 dst에도 넣을 수 있음
#define DMA_IF_AXI4_MST0   0x4
...
#define DMA_SG_TYPE_SRCDST_SG  0x3 // ← 소스·목적지 양쪽 SG. P2P SG 전송에 필요한 모드
```

`DMA_SG_TYPE_SRCDST_SG`(양쪽 동시 scatter-gather)가 정의되어 있다는 건
**IP가 "양쪽 다 PCIe"를 상정하고 설계됐다**는 강한 방증입니다.

---

## 3. 그 레지스터는 호스트에서도 접근 가능하다

펌웨어 쪽 주소:
```c
#define DMA_ENGINE_BASE        0x400
#define DMA_ENGINE_SIZE        0x40
#define DMA_ENGINE(base, i)    ((base) + 0x400 + (i) * 0x40)
#define DMA_ENGINE_DMA_SRCPARAM  0x00
```

호스트 쪽 주소 (`plda_qpcie_registers.h:149`):
```c
REG_DMA0_SRCPARAM = 256,
```

접근 코드 (`plda_qpcie_api.cpp`, ReadDeviceRegister):
```c
(uint32_t*)( ( (uint32_t*)pBoard->resources.bars[0].user_address ) + register_offset )
//            ↑ BAR0                                                 ↑ uint32_t* 산술 = ×4
```

**256 × 4 = 1024 = 0x400.** 펌웨어의 `DMA_ENGINE_BASE`와 **정확히 같은 레지스터**입니다.
BAR0 = `Region 0: Memory at 90000000 [size=16K]` — 16K 안에 DMA(0x400~) + ATU 테이블(0x600~0xB00)이 다 들어갑니다.

그리고 `QPCIe_WriteDeviceRegister`에는 **화이트리스트가 없습니다.** 범위 검사뿐:

```c
if ( ( nb_registers + register_offset ) > (uint32_t)( REG_ATR_AXI4_SLV3_T7_TRSL_MASK + 1 ) ) {
    return QPCIe_INVALID_PARAMETER;   // 이게 전부
}
```

`REG_DMA0_SRCPARAM = 256`은 당연히 범위 안입니다.
**→ 소프트웨어적으로는 호스트에서 런타임에 덮어쓸 수 있습니다. 펌웨어 재빌드/플래시 없이.**

---

## 4. 선택지 — 난이도 순

### (A) 호스트에서 런타임 재설정 — 가장 쉽고, 펌웨어 무관

```c
uint32_t pcie_if = 0x0;   // DMA_IF_PCIE
QPCIe_WriteDeviceRegister(dev, REG_DMA3_SRCPARAM,  &pcie_if, 1);
QPCIe_WriteDeviceRegister(dev, REG_DMA3_DESTPARAM, &pcie_if, 1);   // 양쪽 다 PCIe

// 그 다음엔 QPCIe_ProgramDmaEngine 을 못 씀 (방향 검증에서 걸림)
// → SRCADDR_L/H, DESTADDR_L/H, LENGTH, CONTROL 을 직접 write
QPCIe_WriteDeviceRegister(dev, REG_DMA3_SRCADDR_L,  &src_lo, 1);   // peer A BAR 물리주소
QPCIe_WriteDeviceRegister(dev, REG_DMA3_DESTADDR_L, &dst_lo, 1);   // peer B BAR 물리주소
...
```

- 장점: 플래시 불필요, 되돌리기 쉬움(재부팅하면 원상복구)
- 단점: API의 방향 검증을 우회해야 하므로 `ProgramDmaEngine`/`StartDmaTransfer`를 안 쓰고 레지스터를 직접 다뤄야 함. 실질적으로 **드라이버 계층을 새로 조금 쓰는 것**

### (B) U-Boot 수정 — 영구적, 깔끔

`pcie_ctl_dma_default_setting()`에 P2P 채널을 하나 추가:

```c
    dma_srcid_set(base, 3, DMA_IF_PCIE);
    dma_dstid_set(base, 3, DMA_IF_PCIE);
    dma_sg_set   (base, 3, DMA_SG_TYPE_SRCDST_SG, DMA_SG_ID_PCIE);
```

그 다음 `make` → `bin/image.gpt` → 플래시.
호스트 API도 `E_PEER_TO_PEER` 거부를 풀어야 하므로 `plda_qpcie_api.cpp`도 같이 손봐야 합니다.

- 장점: 구조적으로 올바른 위치. 채널 하나를 P2P 전용으로 확보
- 단점: 부트 이미지 재빌드·플래시 절차 필요. 실패 시 보드 복구 리스크

### (C) RTL / 실리콘 — **거의 확실히 불필요**

IP에 `DMA_IF_PCIE`와 `DMA_SG_TYPE_SRCDST_SG`가 정의돼 있고 인터페이스 선택이 런타임 레지스터이므로,
합성 단계에서 P2P가 잘려나갔을 가능성은 낮습니다.

---

## 5. 아직 확인 못 한 두 가지

정직하게 말하면 위 결론에는 검증 안 된 고리가 둘 있습니다.

1. **SRCPARAM/DESTPARAM이 호스트 BAR 쪽에서 정말 쓰기 가능한가**
   소프트웨어 제약은 없음을 확인했지만, IP가 이 필드를 **EP 측(펌웨어)에서만 쓰기 허용**하도록
   합성됐을 수 있습니다. `DMA_ENGINE_DMA_SHARE_ACCESS`(오프셋 0x28) 레지스터가 존재하는 걸 보면
   접근 제어 개념 자체는 IP 안에 있습니다.

2. **ATU outbound 창 설정이 추가로 필요한가**
   목적지가 다른 장치의 BAR이면 그 주소가 PCIe로 나가야 합니다.
   `pcie_ctl_windows_table_set()` / `PCIE_CTL_ATR_AXI4_SLV0~3`(0x800~0xB00)이 그 주소 변환 창인데,
   P2P 경로에 맞는 창을 따로 열어줘야 할 수 있습니다.

### 1분이면 판별 가능합니다

보드가 지금 켜져 있으므로(`01:00.0`, `plda` 로드됨, `/dev/plda0` 존재)
BAR0의 아래 오프셋을 읽으면 현재 4채널 배선이 그대로 나옵니다:

| 채널 | SRCPARAM | DESTPARAM |
|:---:|---|---|
| 0 | `0x400` | `0x404` |
| 1 | `0x440` | `0x444` |
| 2 | `0x480` | `0x484` |
| 3 | `0x4C0` | `0x4C4` |

기대값: 채널0 = `0x0`/`0x4`, 채널1 = `0x4`/`0x0`, 채널2 = `0x0`/`0x4`, 채널3 = `0x4`/`0x0`

그 다음 채널3의 DESTPARAM에 `0x0`을 써 보고 되읽어 반영되면 **(A) 경로가 열려 있다**는 확답이 됩니다.
(재부팅하면 U-Boot가 원상복구하므로 되돌리기 쉬움)

---

## 요약

- **"구현되어 있는데 안 쓰는 것"** 쪽이 맞습니다 — 단 "설정만 안 된" 게 아니라 **부팅 때 반대로 못박혀 있는** 상태
- 못은 **U-Boot SPL의 `pcie_ctl_dma_default_setting()` 한 함수**
- 그 못은 **런타임 레지스터**라서 호스트에서도 뽑을 수 있을 가능성이 높음
- **RTL/실리콘 수정은 불필요**
- 다만 [p2p-support.md](./p2p-support.md)에 적었듯 **현재 장비엔 P2P 상대(dGPU)가 없습니다.**
  기능을 뚫는 것과 쓸모가 있는 것은 별개입니다.
