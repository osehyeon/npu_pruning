# 왜 데모는 도커 컨테이너 안에서만 돌리는가

> 작성일: 2026-08-02 · 관련: [why-hostreset-flow.md](./why-hostreset-flow.md), [../onnxruntime-neubla/README.md](../onnxruntime-neubla/README.md)

---

## 결론

**도커가 기술적으로 필수인 건 아닙니다.** 네 가지 불편이 겹쳐 있을 뿐이고,
그 중 셋은 호스트에서도 해결 가능합니다.

| 이유 | 실제 강제력 | 호스트에서 해결 가능? |
|---|---|---|
| ① **Python 3.8 ABI 고정** | 강함 | ✅ conda 3.8 환경으로 가능 (실증함) |
| ② **패키지 스택 부재** | 중간 | ✅ pip 설치하면 됨 |
| ③ **`/dev/plda0` 이 root 전용** | 강함 | ⚠️ sudo 필요, 또는 udev 규칙 설치 |
| ④ **X11 GUI 포워딩** | 약함 | ✅ 호스트는 원래 DISPLAY 가 있음 |

핵심은 **① + ③** 이고, 도커는 이 둘을 한 번에 덮어버리는 가장 게으른 해법이었습니다.

---

## ① Python 3.8 ABI 고정 — 가장 직접적인 이유

드라이버 모듈 파일명이 답을 말합니다:

```
NeublaDriver.cpython-38-x86_64-linux-gnu.so
                    ^^
```

**Python 3.8 전용 확장 모듈**입니다. 호스트 기본 파이썬은 3.10이라 그대로는 import되지 않습니다.
그래서 Dockerfile이 3.8을 못박습니다:

```dockerfile
RUN pyenv install 3.8.18 && pyenv global 3.8.18
```

### 그런데 호스트에서도 됩니다 — 확인했습니다

conda `test` 환경이 Python 3.8.20 이라 그대로 로드됩니다:

```
$ PYTHONPATH=~/demo_1127 ~/miniforge3/envs/test/bin/python -c "import NeublaDriver; print(dir(NeublaDriver))"
import OK: /home/osehn/demo_1127/NeublaDriver.cpython-38-x86_64-linux-gnu.so
내용: ['NeublaDriver']
```

**네이티브 의존성도 전부 호스트에서 해결됩니다:**

```
libstdc++.so.6   libbfd-2.38-system.so   libgcc_s.so.1   libc.so.6   libm.so.6   libz.so.1
```

빠진 것 없이 다 잡힙니다. 요구 심볼 버전도 최대 `GLIBC_2.34` 라 Ubuntu 22.04 호스트에서 여유 있습니다.

> 다만 **`libbfd-2.38-system.so` 는 binutils 2.38 = Ubuntu 22.04 정확히 그 버전**입니다.
> 호스트 배포판이 다르면 이것 하나 때문에 깨집니다. 도커가 이 취약점을 가려주고 있었습니다.

---

## ② 패키지 스택이 호스트 3.8 환경에 없음

데모가 요구하는 것:

```python
import cv2, numpy, onnx, onnxruntime
from NeublaDriver import NeublaDriver
from ultralytics.utils import YAML
```

Dockerfile이 설치하는 것:

```dockerfile
RUN pip install --no-cache-dir jinja2 pydevicetree pyparsing \
                               pybind11 ultralytics onnx onnxruntime tabulate
```

호스트 conda `test`(3.8.20) 실태:

```
cv2            없음
numpy          1.24.4
onnx           없음
onnxruntime    없음
ultralytics    없음
tabulate       없음
```

**numpy 말고는 전부 없습니다.** 그냥 설치하면 되는 문제지만,
Python 3.8은 이미 EOL이라 최신 ultralytics/onnxruntime 휠이 3.8을 안 받는 경우가 생깁니다.
이미지가 만들어진 시점의 버전 조합이 그대로 얼어 있는 게 도커의 실질적 가치입니다.

---

## ③ `/dev/plda0` 이 root 전용 — 두 번째 핵심 이유

```
$ ls -la /dev/plda0
crw------- 1 root root 234, 0  7월 30 17:28 /dev/plda0

$ python3 -c "open('/dev/plda0','rb')"
PermissionError: [Errno 13] Permission denied: '/dev/plda0'
```

일반 사용자 `osehn` 으로는 **열 수 없습니다.**

원인은 [why-hostreset-flow.md](./why-hostreset-flow.md) 에 적은 대로
**udev 규칙이 설치되지 않아서** 입니다:

```
PCIeDriver/51-plda-pcie-udev.rules :  SUBSYSTEM=="plda_chardev", MODE="0666"
→ /etc/udev/rules.d/ 에 없음
```

컨테이너는 이걸 두 겹으로 우회합니다:

```bash
docker run --rm -it --privileged ...          # device cgroup 제한 해제
```
```bash
# entrypoint.sh
usermod -aG sudo "$user"
echo "$user ALL=(ALL:ALL) NOPASSWD: ALL" > /etc/sudoers.d/$user
```

즉 **컨테이너 안에서는 무조건 root가 될 수 있게** 만들어 놨습니다.
호스트에서 하려면 데모를 `sudo` 로 돌리거나 udev 규칙을 설치해야 합니다.

**udev 규칙 설치가 정공법입니다:**

```bash
sudo cp ~/workdir/antara-software/antara-pcie-sdk/PCIeDriver/51-plda-pcie-udev.rules \
        /etc/udev/rules.d/
sudo udevadm control --reload-rules
# 드라이버 재로드 후 /dev/plda0 이 0666 으로 생성됨
```

---

## ④ X11 GUI — 사실 컨테이너 때문에 생긴 문제

데모가 `cv2.imshow` 로 결과 영상을 띄우기 때문에 디스플레이가 필요합니다.
그래서 `docker_run.sh` 가 이런 짓을 합니다:

```bash
--net host  -e DISPLAY=${DISPLAY:-:0}  -v /tmp/.X11-unix/:/tmp/.X11-unix
-v $HOME/.Xauthority:/tmp/.Xauthority  --env XAUTHORITY=/tmp/.Xauthority
```

**호스트에서 돌리면 이 설정이 전부 불필요합니다.** DISPLAY가 이미 있으니까요.
이건 도커를 쓰는 이유가 아니라 **도커를 써서 생긴 비용**입니다.

---

## 종합 — 무엇이 진짜 강제인가

```
Python 3.8 ABI   →  conda 3.8 환경이면 해결       (실증됨)
패키지 스택       →  pip 설치로 해결               (단, 3.8 EOL 이라 버전 고정 필요)
/dev/plda0 권한   →  udev 규칙 설치 또는 sudo      ← 남는 진짜 장벽
X11              →  호스트에선 애초에 문제 아님
binutils 2.38    →  호스트가 Ubuntu 22.04 라 우연히 맞음  ← 숨은 취약점
```

**도커는 "Python 3.8 + 패키지 + root 권한 + 배포판 고정" 을 한 방에 주는 편의 장치**이지,
아키텍처적 필연은 아닙니다.

역설적으로 도커 때문에 **손해**도 있습니다 —
`--privileged` 컨테이너의 `/dev` 가 시작 시점 스냅샷이라
드라이버를 재로드하면 컨테이너를 나갔다 다시 들어와야 합니다
([why-hostreset-flow.md](./why-hostreset-flow.md)).

---

## 호스트에서 돌려보려면

```bash
# 1) 장치 권한 (한 번만)
sudo cp ~/workdir/antara-software/antara-pcie-sdk/PCIeDriver/51-plda-pcie-udev.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules

# 2) 보드 리셋 + 드라이버 로드 (호스트에서 먼저!)
cd ~/workdir && sudo ./hostreset.sh
cd ~/workdir/antara-software/antara-pcie-sdk && sudo ./run.sh
ls -la /dev/plda0            # 0666 으로 바뀌었는지 확인

# 3) Python 3.8 환경에 패키지 설치
~/miniforge3/envs/test/bin/pip install opencv-python onnx onnxruntime ultralytics tabulate

# 4) 실행
cd ~/demo_1127
PYTHONPATH=. ~/miniforge3/envs/test/bin/python yolo_demo.py
```

**아직 실제로 돌려보지는 않았습니다** — 3번에서 Python 3.8용 휠이 안 잡힐 가능성이 있고,
2번은 하드웨어를 건드리므로 확인이 필요합니다.

기존 도커 경로가 잘 돌고 있다면 굳이 옮길 이유는 없습니다.
다만 **왜 컨테이너여야 하는지 몰라서 못 벗어나는 상황은 아니게 되었습니다.**

---

## 확인하지 못한 것

- Python 3.8 에 `opencv-python` / 최신 `ultralytics` 휠이 아직 배포되는지
- `NeublaDriver` 를 실제로 인스턴스화했을 때(호스트, sudo 없이) 동작하는지 — import 까지만 확인함
- 컨테이너 이미지(`neubla/demo_1022`, 7 GB)에 들어있는 정확한 패키지 버전 목록
