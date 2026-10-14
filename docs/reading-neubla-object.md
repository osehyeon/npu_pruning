# 컴파일된 `.o` 를 직접 읽는 방법

> 작성일: 2026-09-16 · 관련: [neubla-isa.md](./neubla-isa.md), [model-to-binary-pipeline.md](./model-to-binary-pipeline.md)
> 근거 파일: `example/asm/`, 원고 폴더의 `paper/data/_verify/decode_o.py` 와 `decode_o.txt` (이 저장소 바깥, `../paper/data/_verify/`)

`nbc` 가 만든 `.o` 안에서 합성곱 명령이 실제로 어떤 채널 수와 타일 크기로 발행되었는지를 읽는 절차입니다.
"패딩된 MACs 가 바이너리에 있는가" 를 확인하려고 만들었고, 답은 없다입니다. 명령이 지시한 연산량은 13 종 모델
전부에서 ONNX 원형 MACs 와 같았습니다.

---

## 1. `.o` 에 무엇이 있는가

```
$ file example/sw50sc8_lw.o
ELF 32-bit MSB relocatable, ChipON KungFu32 (e_machine 769), not stripped

섹션      크기(sw50sc8_lw)  내용
.text     137,056 B        명령. 32 바이트 고정 길이, 4,283 개
.data     ~104 MB          가중치 wgt__*, 편향 bias__*, DRAM 활성화 버퍼 out__*, 입력 in__*
.symtab   235 개            위 데이터의 이름과 주소와 크기
```

명령 하나가 32 바이트이고 `.text` 크기를 32 로 나누면 명령 수가 나옵니다. ELF 헤더는 빅엔디언이고
명령 안의 필드는 리틀엔디언입니다.

## 2. 도구 셋 · 무엇이 되고 무엇이 안 되는가

| 도구 | 위치 | 결과 |
|---|---|---|
| macOS `nm -S`, `objdump -h` | 로컬 | 됩니다. 심볼 표와 섹션 헤더는 아키텍처와 무관 |
| 벤더 `llvm-objdump -d --triple=neubla` | 워크스테이션 `~/neubla_dev_env/tmp/neubla_mlp-install/bin` | 13 개 명령 뒤 `printDataLayout` assertion 으로 죽습니다. 2026-08-02 문서의 역어셈블은 다른 바이너리였음 |
| 벤더 `llvm-mc --triple=neubla -show-encoding` | 같은 곳 | **됩니다.** 텍스트 명령 한 줄을 32 바이트 인코딩으로 바꿔 줍니다. 이 문서의 핵심 |
| 컴파일러가 남긴 `gen_inline_assembly.cpp` | 워크스테이션 `~/demo_0211/nbc/neubla/` | 마지막 컴파일의 명령 텍스트. `.o` 는 이것을 clang 과 어셈블러가 인코딩한 것 |

라이브러리 경로를 잡아야 도구가 뜹니다.

```bash
cd ~
L=$(find neubla_dev_env -name "*.so*" -path "*lib*" | xargs -n1 dirname | sort -u | tr '\n' ':')
export LD_LIBRARY_PATH=$L
B=~/neubla_dev_env/tmp/neubla_mlp-install/bin
```

## 3. 절차

### 3.1 심볼 표에서 저장 크기를 읽는다 · 로컬

```bash
nm -S example/sw50sc8_lw.o | grep 'wgt__n_conv2d_10_0'
# 03414a80 0003b100 D wgt__n_conv2d_10_0_weight_quantized     0x3b100 = 241,920 = 224 × 120 × 9
```

- `wgt__` 가중치는 출력 채널을 32 단위로 올린 크기입니다. 328 채널은 352 로 저장됩니다. 입력 채널은 원형 그대로.
- `out__` DRAM 활성화 버퍼는 채널을 64 단위로 올린 크기입니다. 120 채널 152² 은 128 × 152² 바이트.
  예외는 모델 입력 `in__input_1_quantized` 가 3 채널 그대로인 것과 Concat 출력의 NCHW 사본 하나입니다.
- 스크립트 `object_padding.py` 가 12 종을 한 번에 봅니다. 원고 폴더의 `paper/data/_archive/` 에 있습니다.

### 3.2 명령 텍스트를 얻는다 · 워크스테이션

컴파일러는 호스트에서 돌고 보드가 필요 없습니다. 컴파일하면 `gen_inline_assembly.cpp` 가 덮어써지므로
모델별로 컴파일 직후에 뽑아야 합니다.

```bash
cd ~/workspace_docs/example
./compile_variant.sh ~/workspace_docs/model/u8/sw50sc8_u8_maD20.onnx input.1 /tmp/probe.o lw single
md5sum /tmp/probe.o ~/workspace_docs/example/sw50sc8_lw.o        # 같아야 합니다. 2026-09-16 에 같았음
grep -v "__declspec" ~/demo_0211/nbc/neubla/gen_inline_assembly.cpp | grep -n 'asm(\|^//' > sw50sc8_lw_asm.txt
```

`__declspec` 줄은 가중치 데이터 배열이라 뺍니다. 남는 것이 명령 4,283 줄과 주석입니다. 이 파일이
`example/asm/sw50sc8_lw_asm.txt` 입니다.

```
asm("	load	%t1{12,p,u8,mt(3,3,3,32),st(3,3,3),MCRSm,0}	,{dram(1108992,3,3,3)}");
asm("	conv2d_bias{f}	%t5{1,p,u8,mt(608,51,32,1),st(608,51,64),NCHWc,112384}	,%t0	,%t1	,%t2	,%t3	,{m(OP),p(1,0,1,1),s(1,1),d(1,1)}");
```

텐서 레지스터 `%tN{개수, 클래스, dtype, mt(...), st(...), 레이아웃, SRAM 주소}` 에서 `mt` 가 타일 형상입니다.
가중치는 `MCRS*` 레이아웃이고 `mt(R, S, Cin, M)`, 활성화는 `mt(W, H, C, N)` 입니다. `st` 는 스트라이드.

### 3.3 텍스트가 `.o` 와 같은지 확인한다 · 워크스테이션

```bash
grep -v "__declspec" gen_inline_assembly.cpp | grep "asm(" \
  | sed -E 's/^asm\("(.*)"\);$/\1/' | sed 's/\\t/\t/g' > /tmp/all.s
$B/llvm-mc --triple=neubla -show-encoding /tmp/all.s | grep encoding: \
  | sed -E 's/.*encoding: \[(.*)\]/\1/' | tr -d ' ' > /tmp/all.enc
```

`/tmp/all.enc` 의 i 번째 줄이 `.text` 의 i 번째 32 바이트와 같으면 텍스트 한 줄이 명령 하나입니다.
sw50sc8 lw 에서 4,283 개 전부 같았습니다. 이 단계가 있어야 다음 단계의 필드 맵을 `.o` 에 적용할 수 있습니다.

### 3.4 필드 위치를 정한다 · 차등 인코딩

같은 명령에서 값 하나만 바꿔 어셈블하고 달라진 바이트를 봅니다.

```bash
enc(){ printf "%s\n" "$1" > /tmp/v.s; $B/llvm-mc --triple=neubla -show-encoding /tmp/v.s | grep -o "encoding: .*"; }
wl='	load	%t3{19,p,u8,mt(3,3,32,64),st(3,3,32),MCRScm,896}	,{dram(1110016,3,3,32)}'
enc "$wl"
enc "$(echo "$wl" | sed 's/mt(3,3,32,64)/mt(3,3,120,64)/')"     # 바이트 11,12 가 0x00,0x02 → 0x80,0x07
```

32 → `0x0200`, 120 → `0x0780`, 128 → `0x0800`. 리틀엔디언 16 비트를 4 만큼 오른쪽으로 밀면 채널 수입니다.
이렇게 정한 필드가 아래 표입니다.

| 바이트 | 필드 | 식 | 확인한 값 |
|---|---|---|---|
| 31 | opcode | `conv2d_bias` 0xe8. `load` 는 0x80 미만이고 하위 7 비트가 목적지 레지스터. 그 외 op 는 0x80 이상 | op 별 집합이 겹치지 않음 |
| 4 | 레이아웃 | 하위 3 비트. NCHW 1 · NHWC 2 · NCHWc 5 · MCRSm 6 · MCRScm 7 | 상위 비트는 레지스터 개수 등 |
| 0..2 | SRAM 주소 | LE24 | 773376 = `00 cd 0b` |
| 5 | 출력 스트라이드 채널 | `b5 << 4` | 64 → 0x04, 256 → 0x10 |
| 6, 8 | 커널 R, S | `b6 >> 2`, `b8` | 3 → 0x0c, 0x03 |
| 9,10 | 가중치 M (출력 채널) | `LE16 >> 6` | 64 → 0x1000, 56 → 0x0e00, 63 → 0x0fc0 |
| 11,12 | 채널 C (모든 텐서 기술자) | `LE16 >> 4` | 위 |
| 13,14 | 타일 H | `LE16 >> 2` | 34 → 0x88, 76 → 0x0130 |
| 15,16 | 타일 W | `LE16 & 0xfff` | 608 → 0x260, 304 → 0x130 |
| 19..21 | DRAM 주소 | `LE24 × 64` | 1110016 → 0x43c0 |
| 27,28 | conv 의 가중치 레지스터 | `(LE16 >> 6) & 0x7f` | t1 → b27 0x40 |
| 28,29 | conv 의 입력 레지스터 | `(LE16 >> 5) & 0x7f` | t2 → b28 0x40 |
| 29,30 | conv 의 목적지 레지스터 | `(b29 >> 4) \| ((b30 & 7) << 4)` | t70 → 0x60, 0x0c |

레지스터 번호는 텍스트에서도 0 에서 127 사이라 7 비트로 충분합니다. 128 을 넘는 번호는 `llvm-mc` 가 거부했습니다.

### 3.5 `.o` 만 읽는 해독기 · 로컬

원고 폴더의 `paper/data/_verify/decode_o.py` 가 위 표로 `.text` 를 훑습니다.

1. `load` 이면 목적지 레지스터에 레이아웃, R, S, C, M, DRAM 주소를 기록한다.
2. `conv2d_bias` 이면 가중치 레지스터의 기록을 꺼내 R, S, C 를 얻고, 명령 자신의 M, H, W 를 읽는다.
3. `M × C × R × S × W × H` 를 그 합성곱에 더한다. 합성곱 이름은 가중치의 DRAM 주소를 `nm` 의 `wgt__` 심볼 범위에 대응해 얻는다.

```bash
python3 paper/data/decode_o.py            # 13 종 요약 → decode_o.txt
python3 paper/data/decode_o.py sw50sc8    # 층별 표      → decode_o_sw50sc8.txt
```

## 4. 검증

- sw50sc8 lw 에서 텍스트를 해석한 `asm_conv_tiles.py` 와 `.o` 만 해독한 `decode_o.py` 의 층별 명령 수, M 분포, C 분포, MACs 비가 75 층 전부 같았습니다.
- 13 종 `.o` 에서 합성곱 명령의 입력 채널이 원형과 다른 층은 0 개, 명령 MACs 합과 ONNX 원형의 비는 전부 1.0000 입니다.

## 5. 읽어서 알게 된 것

- 합성곱 명령의 입력 채널은 원형 그대로입니다. 120 은 120.
- 출력 채널은 64 이거나 남은 만큼입니다. sw50sc8 에는 8, 16, 24, 32, 40, 48, 56, 63, 64 가 나옵니다.
- 64 단위는 출력 스트라이드 `st`, DRAM 버퍼 크기, 타일 수 셋에만 있습니다.
- 그러므로 `ceil(C/64)·64` 로 계산한 padded MACs 는 바이너리 어디에도 없는 모형값입니다. 하드웨어가 채널 120 짜리 명령을 내부에서 128 로 처리하는지는 이 자료로 판정할 수 없습니다. 컴파일러의 시간 모형이 120 과 128 을 같게 매기는 것이 유일한 간접 증거입니다.

## 6. 하지 않은 것과 주의

- `conv2d_bias` 와 `load` 외의 명령은 필드를 정하지 않았습니다. `subtile`, `concat`, `add`, `prelu`, `store` 의 텐서 기술자도 같은 자리에 있을 가능성이 크지만 확인하지 않았습니다.
- 32 바이트 가운데 위 표에 없는 바이트의 뜻은 모릅니다. 슬롯과 퓨즈 플래그는 [neubla-isa.md](./neubla-isa.md) 참조.
- `half` 옵션 기록의 ofmStep 번호는 0, 1, 3 처럼 건너뛰므로 스케줄 기록의 타일 수 세기는 lw 와 egs 에서만 씁니다. `.o` 해독은 옵션과 무관합니다.
- 인라인 어셈블리는 마지막 컴파일의 것만 남습니다. 지금 `example/asm/` 에는 sw50sc8 lw 하나만 있고, 다른 모델은 `.o` 만 해독했습니다. 텍스트가 필요하면 3.2 를 모델마다 반복합니다.
