# 재부팅 후 해야 하는 것

> 작성일: 2026-08-02 · 관련: [why-hostreset-flow.md](./why-hostreset-flow.md), [../example/README.md](../example/README.md)

---

## 한 줄 요약

**드라이버만 다시 올리면 됩니다. hotreset 은 필요 없습니다.**

```bash
~/workdir/load_driver.sh
```

이어서 실행까지 하면 두 줄입니다.

```bash
~/workdir/load_driver.sh
cd ~/workspace_docs/example && ./run_npu.sh both
```

---

## 무엇이 남고 무엇이 사라지나

| 항목 | 재부팅 후 | 이유 |
|---|:---:|---|
| **udev 규칙** (`/etc/udev/rules.d/51-plda-*.rules`) | ✅ 유지 | 디스크에 있는 설정 파일 |
| **보드 펌웨어** (FSBL → U-Boot → app_fw) | ✅ 자동 | 플래시에서 부팅 |
| **`plda` 커널 모듈** | ❌ **사라짐** | out-of-tree 모듈, 자동 로드 설정 없음 |
| **`/dev/plda0`** | ❌ 사라짐 | 모듈이 만드는 노드 |

확인해 본 결과 자동 로드 장치가 하나도 없습니다:

```
modinfo plda            → ERROR: Module plda not found      (시스템 모듈 경로에 없음)
/etc/modules-load.d/    → plda 설정 없음
systemd 서비스           → 없음
```

`plda.ko` 는 `~/workdir/antara-software/antara-pcie-sdk/PCIeDriver/` 에만 있습니다.

---

## 절차

### 1. 드라이버 로드

```bash
~/workdir/load_driver.sh
```

**멱등입니다** — 이미 로드돼 있으면 건드리지 않고 상태만 출력합니다.
재부팅 직후든 아니든 그냥 치면 됩니다.

```
  lspci  : 01:00.0 Communication controller: PLDA XpressRich-AXI Ref Design (rev 02)
  lsmod  : plda                   81920  0
  /dev/plda0 : crw-rw-rw- root:root
           → sudo 없이 접근 가능
```

| 옵션 | 동작 |
|---|---|
| (없음) | 로드. 이미 있으면 상태만 출력 |
| `-r` | 강제 재로드 — **hotreset 후 `max_irq=32` 복구용**, `o_dbg` 끄기용 |
| `-d` | 디버그 로그 켜고 로드 (`o_dbg=8`) — [주의](#o_dbg8-이-디스크를-채웁니다) |
| `-u` | 내리기 |
| `-h` | 도움말 |

스크립트가 하는 일:
- `o_dbg=0 max_irq=32` 로 `insmod` (`max_irq` 는 `run.sh` 와 동일)
- `udevadm settle` 로 권한 적용까지 대기
- `lspci` 에 보드가 없으면 `hostreset.sh` 를 안내하고 중단
- `/dev/plda0` 을 읽을 수 있는지까지 확인
- 이미 로드돼 있는데 `o_dbg` 가 켜져 있으면 경고

### `o_dbg=8` 이 디스크를 채웁니다

**2026-08-04 에 이것 때문에 측정이 통째로 죽었습니다.** 원인을 찾는 데 시간이 걸렸으니 기록해 둡니다.

COCO 5,000장 × 4타깃 NPU 측정을 돌리는 동안:

```
/var/log/kern.log   7.7 GB
/var/log/syslog     7.7 GB     ← 같은 내용 중복 기록
                   ────────
루트 파티션 100 % → 배경 작업이 출력을 못 써서 사망
```

최근 20만 줄을 세어보니 **199,851 줄이 `plda:`** 였습니다. 추론 1회마다 드라이버가
커널 로그를 남기므로, 장시간 실행에서는 로그가 추론 결과보다 훨씬 빨리 쌓입니다.

증상이 헷갈립니다 — 디스크가 차면 셸 명령의 출력조차 저장되지 않아
**`df` 같은 진단 명령마저 실행되지 않습니다.** 그때는 다른 터미널에서 직접 확인하세요.

그래서 `load_driver.sh` 의 기본값을 `o_dbg=0` 으로 내렸습니다.
드라이버를 실제로 디버깅할 때만 `-d` 로 켜고, **끝나면 `-r` 로 되돌리세요.**

이미 커진 로그는 이렇게 비웁니다 (지우지 말고 truncate — syslogd 가 파일을 열어두고 있습니다):

```bash
sudo truncate -s 0 /var/log/kern.log /var/log/syslog
sudo journalctl --vacuum-size=200M
```

> `sudo ./run.sh` 를 써도 드라이버는 올라가지만, 그건 로드 **후 예제까지 실행**합니다.
> 드라이버만 올리려면 `load_driver.sh` 가 맞습니다.

### 2. 실행

```bash
cd ~/workspace_docs/example && ./run_npu.sh both
```

`/dev/plda0` 이 `crw-------` 로 나오면 udev 규칙이 설치되지 않은 것입니다
([설치 방법](#부록--udev-규칙-설치)).

---

## hotreset 은 왜 필요 없나

전원을 껐다 켜면 **보드 자체가 전원 리셋**되고 FSBL → U-Boot → app_fw 가 플래시에서 새로 부팅합니다.
`hotreset.sh` 가 하는 PCIe secondary bus reset + 재열거는 **이미 일어난 것과 같은 효과**라 중복입니다.

### hot reset 으로 복구되지 않는 경우

**DMA 고장은 `hostreset.sh` 로 못 고칩니다. 재부팅해야 합니다.** 실제로 겪었습니다.

증상:

```
Failed to program DMA0 engine. err: 1              (err 1 = QPCIe_FAILURE)
Failed to sg_dma_write (0xa00001000, 247840)
Failed to copy antara object file [ret: -5]
munmap_chunk(): invalid pointer                   ← 그 직후 Close() 에서 크래시
```

- 어떤 `.o` 를 써도 `LoadModel` 이 실패합니다
- **SDK 자체 예제(`antara_example`)의 DMA 테스트도 똑같이 죽습니다** → 보드 문제 확정
- `sudo ./hostreset.sh` + 드라이버 재로드로도 **복구되지 않았습니다**
- **컴퓨터 재부팅으로 복구되었습니다**

**왜 재부팅만 듣는가**

DMA 엔진의 방향 설정은 U-Boot 이 **보드 부팅 때 딱 한 번** 합니다
(`bootloader/u-boot/arch/riscv/cpu/sesame/pcie_ep_drv/device/pcie_ep_ctl.c:241`):

```c
void pcie_ctl_dma_default_setting(unsigned int base)
{
    dma_srcid_set(base, 0, DMA_IF_PCIE);       dma_dstid_set(base, 0, DMA_IF_AXI4_MST0);
    dma_srcid_set(base, 1, DMA_IF_AXI4_MST0);  dma_dstid_set(base, 1, DMA_IF_PCIE);
    ...
}
```

| | DMA 설정 |
|---|---|
| **PCIe hot reset** (`hostreset.sh`) | 엔드포인트 레지스터는 리셋되지만 **Antara SoC 는 계속 돌아** U-Boot 이 다시 실행되지 않음 → 복구 안 됨 |
| **컴퓨터 재부팅** | 슬롯에 PERST# 가 걸려 **보드가 재부팅** → FSBL → U-Boot → DMA 설정 복구 |

> 라이브러리 버그 하나가 진단을 어렵게 합니다. `QPCIe_ProgramDmaEngine` 의 실패 경로에
> `delete pBoard->interrupt_lock;` 만 있고 포인터를 비우지 않아, 이후 `Close()` 가
> 이중 해제로 죽습니다. **진짜 에러 메시지가 크래시에 묻히므로**
> `munmap_chunk(): invalid pointer` 가 보이면 그 위쪽 출력을 먼저 확인하세요.

### hotreset 이 도움이 되는 경우

평상시 워크플로가 아니라 **복구용**입니다.

| 증상 | 조치 |
|---|---|
| `NeublaDriver.Init()` 실패 | `sudo ~/workdir/hostreset.sh` |
| `Launch()` 가 멈추거나 타임아웃 | 〃 |
| `lspci` 에 장치가 안 보임 | 〃 (그래도 안 보이면 전원 재투입) |
| 이전 실행이 비정상 종료됨 | 〃 |

```bash
cd ~/workdir && sudo ./hostreset.sh
cd ~/workdir/antara-software/antara-pcie-sdk && sudo ./run.sh
```

**반드시 호스트에서** 실행해야 합니다 — 컨테이너 안에서 하면 결과가 보이지 않습니다
([why-hostreset-flow.md](./why-hostreset-flow.md)).

주의: `hotreset_drv_reload.py` 가 드라이버를 `max_irq=1` 로 하드코딩해 다시 올립니다.
그래서 리셋 후에는 `~/workdir/load_driver.sh -r` 로 `max_irq=32` 를 복구해야 합니다
(`-r` 은 `o_dbg` 도 기본값 0 으로 되돌립니다).

---

## 부팅 시 자동 로드 (권하지 않음)

```bash
sudo cp ~/workdir/antara-software/antara-pcie-sdk/PCIeDriver/plda.ko \
        /lib/modules/$(uname -r)/extra/
sudo depmod -a
echo plda | sudo tee /etc/modules-load.d/plda.conf
echo "options plda o_dbg=0 max_irq=32" | sudo tee /etc/modprobe.d/plda.conf
```

**커널이 업데이트되면 깨집니다.** `plda.ko` 는 `6.8.0-49-generic` 에 맞춰 빌드되어 있어
(vermagic 확인함) 커널이 바뀌면 자동 로드가 실패하고 부팅 로그에 에러가 남습니다.
그때는 컨테이너에서 모듈을 다시 빌드해야 합니다.

커널을 고정해 쓰는 장비가 아니라면 **`load_driver.sh` 를 그때그때 실행하는 편이 안전합니다.**

---

## 부록 — udev 규칙 설치

한 번만 하면 재부팅 후에도 `/dev/plda0` 이 `0666` 으로 생성됩니다.

```bash
sudo cp ~/workdir/antara-software/antara-pcie-sdk/PCIeDriver/51-plda-pcie-udev.rules \
        /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=plda_chardev
sudo udevadm settle          # ← 이거 없으면 아래 ls 가 옛날 값을 봅니다
ls -la /dev/plda0
```

**이 장비에서는 이미 설치·적용 완료했습니다:**

```
crw-rw-rw- 1 root root 234, 0  8월  2 22:36 /dev/plda0
```

규칙 내용과 장치 속성이 일치하는 것도 확인했습니다:

```
규칙 : SUBSYSTEM=="plda_chardev", MODE="0666"
장치 : E: SUBSYSTEM=plda_chardev
```

### 두 가지 함정

1. **`udevadm trigger` 없이 규칙만 복사하면 적용되지 않습니다.**
   udev 규칙은 노드가 *생성될 때* 적용되므로, 이미 존재하는 `/dev/plda0` 은 그대로 남습니다.
   `trigger` 가 change 이벤트를 보내 규칙을 다시 태웁니다.

2. **`udevadm settle` 을 빼면 확인이 어긋납니다.**
   `trigger` 는 uevent 를 *보내고 바로* 반환하고, 실제 권한 변경은 udevd 가 비동기로 합니다.
   `&& ls` 로 이어 붙이면 아직 안 바뀐 값을 보게 됩니다 — 실제로 겪었습니다.
   `load_driver.sh` 는 이 대기를 자동으로 합니다.

공용 장비라 `0666` 이 부담스러우면 그룹 방식으로:

```bash
sudo groupadd -f plda && sudo usermod -aG plda $USER
echo 'SUBSYSTEM=="plda_chardev", MODE="0660", GROUP="plda"' | \
    sudo tee /etc/udev/rules.d/51-plda-pcie-udev.rules
sudo udevadm control --reload-rules && sudo udevadm trigger --subsystem-match=plda_chardev
```

그룹 추가는 재로그인해야 반영됩니다 (`newgrp plda` 로 현재 셸만 즉시 적용 가능).
