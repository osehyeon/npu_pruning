# 재부팅 절차 · 2026-08-24

DMA 고장으로 보드 측정이 막혀 있음. 재부팅만이 복구 수단임.
컴파일이 끝난 뒤에 진행할 것.

---

## 1. 컴파일이 끝났는지 확인

```bash
tail -3 ~/workspace_docs/example/compile_half.log
```

`════ 전체 완료` 가 보이면 끝난 것임. 아니면 기다릴 것.

진행 중인지 바로 보려면 이럼.

```bash
pgrep -f _skipconcat_run.py && echo "컴파일 중" || echo "없음"
```

---

## 2. 재부팅

```bash
sudo reboot
```

PCIe 슬롯에 PERST# 가 걸려 보드가 재부팅되고 U-Boot 이 DMA 를 다시 설정함.
`load_driver.sh -r` 이나 `hostreset.sh` 는 듣지 않음.

---

## 3. 부팅 후 · 순서 중요

**드라이버를 먼저 올리고 그다음 UART 를 끕니다.** 순서가 반대면
드라이버 적재가 `dbgprint` 설정을 초기화합니다.

### 3-1. 드라이버 적재

```bash
~/workdir/load_driver.sh
```

기본이 `o_dbg=0` 임. 디버깅할 때만 `-d` 를 붙일 것.
`o_dbg=8` 로 올리면 커널 로그가 디스크를 채움.

스크립트 내부 `sudo` 가 tty 를 요구해 실패하면 직접 넣습니다.

```bash
cd ~/workdir
sudo insmod ./antara-software/antara-pcie-sdk/PCIeDriver/plda.ko o_dbg=0 max_irq=32
```

### 3-2. UART 로그 끄기

```bash
minicom -D /dev/ttyUSB0
```

프롬프트 `SEMIFIVE$` 에서 이렇게 입력함.

```
dbgprint uart off
dbgprint save off
```

**명령은 `dbgprint` 입니다.** `set_dbgprint` 는 펌웨어의 C 함수 이름이고
콘솔 명령이 아닙니다. 보드에서 `help` 를 치면 확인됩니다.

**안 끄면 모델과 무관하게 약 31 ms 가 얹혀 가속 배수가 희석됩니다.**
minicom 종료는 `Ctrl-A` 누르고 `X` 임.

minicom 없이 스크립트로 넣으려면 이렇게 합니다.

```bash
sudo stty -F /dev/ttyUSB0 115200 raw -echo
sudo bash -c "printf 'dbgprint uart off\r' > /dev/ttyUSB0"
```

명령 끝은 `\r` 하나입니다. `\r\n` 을 보내면 남는 `\n` 이 `CMD ERR` 를 만듭니다.

### 3-3. 확인

```bash
lsmod | grep plda        # plda 모듈이 보여야 함
ls -la /dev/plda0        # 장치 노드가 있어야 함
df -h /                  # 여유 공간 확인
```

## 4. Claude 세션 복구

```bash
cd ~/workspace_docs
claude --continue
```

`--continue` 는 이 디렉터리의 가장 최근 세션을 그대로 이어줌.
목록에서 고르려면 `claude --resume` 임.

대화 기록은 파일로 남아 있음.

```
~/.claude/projects/-home-osehn-workspace-docs/25e738c4-b595-4b95-82f8-b8e06c6e2340.jsonl
```

---

## 5. 재부팅 후 할 일

우선순위 순임.

### 5-1. egs 바이너리 수치 검증

egs 는 벤더 검증을 우회해 회수한 것이라 **결과가 맞는지 확인되지 않았음.**

```bash
cd ~/workspace_docs/example
LIMIT=200 ./run_map_npu.sh sw50sc64_lw sw50sc64_egs
```

두 mAP 가 같으면 egs 바이너리가 정상임.

### 5-2. 격자 실측

```
13 모델 × 4 조합 = 52회
lw·single 은 이미 측정됨 · 나머지 39회
```

`bench_npu.py` 의 `TARGETS` 에 항목을 추가해야 함.

### 5-3. 클럭 스윕

`docs/antara-clock-control.md` 참조. minicom 에서 이럼.

```
ne_pll   [id] [rate]      id 0=PLL0 · 1=PLL1
ne_clock [id] [rate]      id 0=BUS · 1=SYS · 2=VX
```

메모리 병목 65 % 추정을 실측으로 검증하는 수단임.
기본값보다 크게 올리지 말 것. 발열과 락 실패 위험이 있음.

### 5-4. 단계별 시간 재측정

선행 논문의 분해를 우리 장비에서 재현함. 같은 i7-12700 임.

---

## 6. 참고 문서

```
docs/npu-measurement-pitfalls.md      DMA 고장 · dbgprint
docs/after-reboot.md                  재부팅 상세
docs/antara-clock-control.md          클럭 제어
docs/compiler-grouping-analysis.md    lw 대 egs 트레이스
paper/related-work-positioning.md     기여 위치
paper/prior-paper-ko.md               선행 논문 정리
```
