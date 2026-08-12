# 왜 "이미지 빌드 → 컨테이너 진입 → hostreset → 컨테이너 탈출 → 재실행" 이라는 이상한 절차가 필요한가

> 작성일: 2026-08-02 · 관련: [workdir.md](./workdir.md)
> 이 문서의 주장은 전부 실제 소스/실행 결과로 확인한 것입니다.

---

## 결론

**장치를 바꾸는 행위는 호스트 커널 전역인데, 그 결과를 보는 쪽은 컨테이너 네임스페이스 안이라서** 생기는
"쓰기는 전역 / 읽기는 격리"의 비대칭입니다.

`--privileged` 컨테이너 안에서 `rmmod`/`insmod`를 하면 **호스트 커널이 실제로 바뀝니다.**
그런데 그 결과물인 `/dev/plda0` 노드는 **컨테이너 시작 시점에 복사된 사본**이라 갱신되지 않습니다.
그래서 리셋한 컨테이너는 자기가 방금 망가뜨린 노드를 계속 붙잡고 있게 되고,
**컨테이너를 나갔다 다시 들어와야만** 새 사본을 받습니다.

---

## 증거 1 — `/dev/plda0`의 major 번호는 동적 할당이다

`PCIeDriver/src/plda_pcie.c:1220`

```c
if ( 0 > (char_major = plda_pcie_register_chrdev(0, DRIVER_NAME)) )
                                             /* ↑ 0 = 커널에게 알아서 배정해달라 */
```

`plda_pcie.c:639`

```c
device_create(chardev_class, NULL, MKDEV(char_major, pBoard->plda_index),
              NULL, DEVICE_NODE_NAME"%d", pBoard->plda_index);   /* → /dev/plda0 */
```

즉 **`insmod` 할 때마다 major가 달라질 수 있고, 노드는 매번 삭제 후 재생성**됩니다.
현재 호스트 상태:

```
$ ls -la /dev/plda0
crw------- 1 root root 234, 0  7월 30 17:28 /dev/plda0
                       ↑major   ↑마지막 insmod 시각
```

major 234는 고정값이 아니라 그때 배정받은 번호입니다.

---

## 증거 2 — 컨테이너의 `/dev`는 라이브 뷰가 아니라 **시작 시점 스냅샷**이다

이게 핵심입니다. 직접 확인했습니다:

```
$ docker run --rm --privileged --entrypoint /bin/bash neubla/antara -c \
    'findmnt -no FSTYPE,SOURCE /dev; ls -la /dev/plda*'

tmpfs  tmpfs                                    ← devtmpfs 아님, 호스트 /dev 바인드마운트도 아님
crw------- 1 root root 234, 0 Aug  2 11:51 /dev/plda0
                              ↑ 컨테이너 시작 시각 (호스트 노드의 7월 30이 아님)
```

runc는 컨테이너의 `/dev`를 **tmpfs로 새로 만들고**, 시작 시점의 호스트 장치 목록을 보고
`mknod`로 하나씩 **복사해 넣습니다.** 호스트의 devtmpfs를 그대로 보여주는 게 아닙니다.

결과:

| 호스트에서 일어난 일 | 컨테이너 안에서 보이는 것 |
|---|---|
| 컨테이너 시작 후 노드가 새로 생김 | **안 보임** |
| 노드가 지워졌다 다른 major로 재생성됨 | **옛 major를 가리키는 죽은 노드가 그대로 남음** |

두 번째 경우 `open("/dev/plda0")` 은 `ENODEV`/`ENXIO`로 실패합니다.
**이것이 "컨테이너를 나갔다 다시 들어와야 하는" 직접적 원인입니다.**

---

## 증거 3 — hot reset은 PCIe 장치를 버스에서 뽑았다 다시 꽂는 수준의 조작이다

`quickPCIe_examples/hotreset_drv_reload.py`가 하는 일:

```python
rmmod("plda")                                          # 1. 드라이버 내림  → /dev/plda0 삭제
setpci -s <parent> BRIDGE_CONTROL.B=0x52               # 2. Secondary Bus Reset 비트 set
setpci -s <parent> BRIDGE_CONTROL.B=0x12               #    해제
time.sleep(4)
echo 1 > /sys/devices/.../<dev>/remove                 # 3. PCI 트리에서 제거
echo 1 > /sys/devices/.../rescan                       #    재열거
time.sleep(2)
insmod(plda.ko, "o_dbg=5 max_irq=1")                   # 4. 드라이버 재로드 → /dev/plda0 재생성
```

장치가 PCI 트리에서 **사라졌다가 다시 나타납니다.** 이건 본질적으로 호스트 전역 동작이고,
컨테이너가 격리해 줄 수 있는 종류의 일이 아닙니다.
그런데 `--privileged` 때문에 컨테이너 안에서 **실행은 됩니다** — "동작은 하는데 결과가 안 보이는" 상황이 만들어집니다.

---

## 증거 4 — 파라미터가 불일치해서 재실행이 한 번 더 강제된다

같은 드라이버를 두 곳에서 서로 다른 옵션으로 올립니다:

| 파일 | insmod 옵션 |
|---|---|
| `antara-pcie-sdk/run.sh` | `o_dbg=8 **max_irq=32**`  (MSI 32개) |
| `quickPCIe_examples/hotreset_drv_reload.py` | `o_dbg=5 **max_irq=1**`  ← **하드코딩** |

`hotreset.sh`를 돌리고 나면 드라이버가 **`max_irq=1`** 로 올라와 있습니다.
`run.sh`가 기대하는 인터럽트 구성이 아닙니다.
그래서 리셋 후에 `run.sh`를 다시 돌려 rmmod/insmod를 해줘야 정상 설정으로 돌아옵니다.
**"다시 실행해야 하는" 이유의 나머지 절반이 이것입니다.** (스냅샷 문제와 별개의 독립 원인)

---

## 구조적 원인 — 성격이 다른 두 도구가 한 폴더에 얹혀 있다

|  | 목적 | 원래 실행 위치 | 근거 |
|---|---|---|---|
| `neubla/antara` 이미지 | **펌웨어 크로스 빌드** | 컨테이너 | Dockerfile: pyenv 3.7, `jinja2`/`pydevicetree`/`pyparsing`(device tree), `u-boot-tools`, `bison`/`flex`, `docker_run.sh`가 `RISCV_PATH`/`TOOL_PATH` 주입 |
| `antara-pcie-sdk` | **호스트 드라이버·보드 제어** | 호스트 | `run.sh`의 rmmod/insmod, `hotreset.sh`의 `Please run as root`, `lspci`/`setpci`/sysfs 직접 조작 |

PCIe SDK는 **전용 호스트 머신**을 전제로 만들어진 bare-metal 도구입니다.
`.quickpciesdk`의 `"the SDK environment shall be sourced only once"` 가드, root 강제, sysfs 직접 write —
전부 컨테이너를 전혀 고려하지 않은 설계입니다.

Docker는 나중에 **펌웨어 빌드 재현성** 때문에 붙은 것이고, 둘을 통합한 사람이 없었습니다.
Dockerfile에 `pciutils`, `kmod`, `linux-headers-$(uname -r)`가 슬쩍 추가돼 있는 게 그 흔적입니다 —
빌드용 이미지에 드라이버 작업이 얹힌 것입니다.

**이름 자체가 증거입니다.** `workdir/hostreset.sh` — 말 그대로 **host**reset,
호스트에서 돌리라고 만든 래퍼입니다. `antara_example.sh`도 마찬가지로 호스트에서
`sudo ./...build/antara_example`를 직접 부릅니다. **둘 다 컨테이너용이 아닙니다.**

---

## 덤으로 발견한 부작용

1. **udev 규칙이 설치돼 있지 않습니다.**
   `PCIeDriver/51-plda-pcie-udev.rules` (`SUBSYSTEM=="plda_chardev", MODE="0666"`)가
   `/etc/udev/rules.d/`에 없습니다. 그래서 노드가 `crw------- root root` →
   모든 스크립트에 `sudo`가 붙습니다. 컨테이너 안에는 udevd 자체가 없어서
   설치해도 컨테이너가 만든 노드에는 적용되지 않습니다.

2. **커널 헤더가 이미지에 구워집니다.**
   Dockerfile의 `linux-headers-$(uname -r)`는 **이미지 빌드 시점**의 호스트 커널을 박아 넣습니다.
   현재는 모듈 vermagic `6.8.0-49-generic`이 호스트와 일치하지만,
   호스트 커널이 업데이트되면 그 이미지에서 빌드한 `plda.ko`는 로드에 실패합니다.

3. **ML 데모도 똑같은 노드에 의존합니다.**
   `NeublaDriver.cpython-38-x86_64-linux-gnu.so` 안에 `/dev/%s%d` + `plda` 문자열이 있습니다.
   즉 `demo_1127` 컨테이너도 `/dev/plda0`을 엽니다.
   → **리셋은 반드시 호스트에서 먼저, 데모 컨테이너는 그 다음에 시작**해야 합니다.
   순서가 바뀌면 동일한 stale 노드 문제를 겪습니다.
   (데모는 `neubla/demo_1022` 이미지로, `neubla/antara`와는 별개입니다.)

---

## 권장 절차

컨테이너를 들락거릴 필요가 없어집니다:

```bash
# 1) 보드 리셋 — 호스트에서
cd ~/workdir
sudo ./hostreset.sh

# 2) 원하는 인터럽트 설정으로 드라이버 재로드 — 호스트에서
cd ~/workdir/antara-software/antara-pcie-sdk
sudo ./run.sh                 # max_irq=32 로 다시 올림 + 예제 실행
#   또는
cd ~/workdir && sudo ./antara_example.sh

# 3) 그 다음에 데모 컨테이너 시작
cd ~/demo_1127 && ./docker_run.sh
```

- `neubla/antara` 컨테이너는 **펌웨어를 빌드할 때만** 들어가면 됩니다. 보드 제어와는 무관합니다.
- `hotreset_drv_reload.py`의 `insmod ... o_dbg=5 max_irq=1` 하드코딩을 인자로 빼면
  2번 단계의 재로드를 생략할 수 있습니다.

### 굳이 컨테이너 안에서 장치를 써야 한다면

`docker_run.sh`에 `-v /dev:/dev`를 추가하면 호스트 devtmpfs가 **라이브로** 보여서
재생성된 노드가 그대로 반영됩니다 (tmpfs 사본 위에 덮어씌우는 방식).

```bash
DOCKER_ENV_CMD+=(-v "/dev:/dev")
```

다만 컨테이너의 `/dev/console`, `/dev/shm`, `/dev/pts`까지 호스트 것으로 바뀌므로
부작용이 있을 수 있습니다. 근본 해결이 아니라 회피책이고,
**리셋은 호스트에서 하고 컨테이너는 나중에 띄우는 쪽이 더 안전합니다.**
