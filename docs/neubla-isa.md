# Neubla NPU 명령어 세트 (Antara NE)

> 작성일: 2026-08-02 · 관련: [antara-npu-internals.png](./antara-npu-internals.png)
> LLVM 백엔드와 실제 컴파일 산출물에서 추출했습니다. 공식 문서는 없습니다.

---

## 추출 방법

툴체인이 `neubla_dev_env/tmp/neubla_mlp-install/` 에 있습니다. 라이브러리 경로만 잡아주면 동작합니다.

```bash
cd ~
L=$(find neubla_dev_env -name "*.so*" -path "*lib*" | xargs -n1 dirname | sort -u | tr '\n' ':')
export LD_LIBRARY_PATH=$L
B=neubla_dev_env/tmp/neubla_mlp-install/bin

# 등록된 타깃은 neubla 하나뿐 (LLVM 15 포크)
$B/llc --version

# 컴파일된 모델 역어셈블
$B/llvm-objdump -d --triple=neubla --section=.text demo_1127/yolov3/yolov3_lw.o
```

니모닉 표는 `libLLVMNeublaAsmParser.so` 의 AsmMatcher 문자열 테이블에서 나옵니다
(`abs` ~ `waits` 알파벳순 97개, 누락 없음).

디버그 경로가 남아 있어 원본 소스 트리도 보입니다:
`/workspace/dev/neubla_mlp/src/nbtools/llvm/llvm/lib/Target/Neubla/`

---

## 명령어 형식

- **명령 하나가 32바이트 고정** (역어셈블 주소가 0x20 간격)
- 오브젝트는 `elf32-unknown`, 디스어셈블 시 `--triple=neubla` 필요
- YOLOv3 기준 `.text` 217 KB(≈6,800개), `.data` 166 MB(가중치)

모든 명령어는 **슬롯 플래그**와 **퓨즈 플래그**를 갖습니다. AsmMatcher의 변환 함수 이름이 그대로 증거입니다:

```
Convert__Reg1_2__Slot1_1__Fuse1_0__MemTileInfo12_3__Reg1_4__Reg1_5__Reg1_6__AttrConv2d9_7
                 ^^^^^^^^  ^^^^^^^
```

| 플래그 | 표기 | 의미 |
|---|---|---|
| **Slot** | 니모닉 뒤 `*` (slot 1) / 없음 (slot 0) | **연산 레인 선택.** 아래 참조 |
| **Fuse** | `{f}` | 다음 명령과 융합 실행 (SRAM 왕복 없음) |

`printSlotOperand` 구현이 1비트임을 확정합니다 — `imm==1 → "*"`, `imm==0 → ""`, 그 외 `assert(false && "invalid slot")`.

### 텐서 레지스터

```
%t5{0x1, p, u8, mt(0x260,0x33,0x20,0x1), st(0x260,0x33,0x40), NCHWc, 0x1b700}
     │    │   │   └ 메모리 타일 형상        └ 스트라이드        │       └ SRAM 주소
     │    │   └ dtype (u8 / i32)                              └ 레이아웃
     │    └ 클래스 (p = 텐서, r = 양자화 정보 레지스터)
     └ 개수/랭크
```

관측된 레이아웃: `NCHW` · `NCHWc` · `NHWC` · `MCRSm` · `MCRScm`
(`MCRS*` 는 가중치 전용 — M=출력채널, C=입력채널, R/S=커널)

### 오퍼랜드 클래스

AsmParser의 `MCK_*` 목록입니다. 괄호 안은 세부 필드 개수입니다.

```
TileRegs · Imm · Slot(1) · Fuse(1) · MemTileInfo(12)
SingleQuantInfo(9) · DualQuantInfo · MergedQuantInfo(10)
AttrConv2d(9) · AttrPool(8) · AttrPad(7) · AttrUpsample · AttrDRAM · AttrDir(1) · AttrMode(1)
```

---

## 전체 명령어 97개

### 행렬 · 컨볼루션 (12)

```
conv2d          conv2d_bias
dwconv2d        dwconv2d_bias           depthwise
trconv2d        trconv2d_bias           transposed (deconv)
matmul2d        matmul2d_bias
fully_connected fully_connected_bias
maxpool2d       avgpool2d
```

### 활성화 (12)

```
relu  relun  prelu  elu  gelu  swish
sigmoid  sigmoid_inv  hsigmoid  hswish
tanh     tanh_inv
```

`sigmoid_inv` / `tanh_inv` 는 역함수입니다. 초월함수 계열은 VX의 **LUT 512 B** 가 담당하는 것으로 보입니다.

### 원소별 산술 (20)

```
add  sub  rsub  mul  div  neg  abs
max  min  clamp
square  sqrt  rsqrt  recp
pow  pow2  exp  log
ceil  floor
```

`rsub` 는 역뺄셈(b−a), `recp` 는 역수, `pow2` 는 2의 거듭제곱입니다.

### 비트 · 논리 (11)

```
band  bor  bxor  bnot        비트 연산
land  lor  lxor  lnot        논리 연산
asr   lsl  lsr               시프트
```

### 비교 (6)

```
cmpeq  cmpne  cmpgt  cmpge  cmplt  cmple
```

### 리덕션 (10)

```
reduce_sum  reduce_mean  reduce_max  reduce_min  reduce_mul
reduce_and  reduce_or    reduce_xor
argmax      argmin
```

### 형태 · 재배열 (12)

```
concat     subtile     pad        replicate
reshape    transpose   reverse
squeeze    unsqueeze
upsample   space2depth depth2space
```

`subtile` 은 큰 텐서에서 타일을 잘라내는 명령으로, 실제 코드에서 **가장 많이 등장합니다**
(표본 중 107회 — SRAM 2 MB 안에서 타일링하느라 그렇습니다).
`concat` 은 `{dir(Y)}` / `{dir(Z)}` 로 결합 축을 지정합니다.

### 메모리 · 이동 (4)

```
load  load_comp  store  move
```

`load_comp` 는 압축된 데이터 로드로 추정됩니다(미확인).

### 동기화 (3)

```
waitc   카운터 대기 — 2개의 64비트 비트마스크를 받음
waits   
sync
```

### 양자화 (3)

```
set_mq_info   multiply quant   — shift1(...), 스케일
set_dq_info   dequant          — shift2(...), shift1(...)
set_sq_info   single quant
```

### 타일 속성 (4)

```
set_tile_attr  mov_tile_attr  clr_tile_attr  clear
```

---

## 슬롯 = 연산 레인 선택

`*` 가 붙는 명령이 무엇인지가 답을 줍니다. `yolov3_half.o` 표본:

| slot 1 (`*`) 을 쓰는 명령 | 항상 slot 0 인 명령 |
|---|---|
| `conv2d_bias{f}*` · `prelu{f}*` · `prelu*` · `add*` | `subtile` `concat` `load` `store` `waitc` `set_mq_info` `set_dq_info` |

**2개씩 존재하는 연산기(NCU×2, VX×2)를 쓰는 명령만 슬롯을 씁니다.**
DDMA·RESHAPER는 하나뿐이라 고를 대상이 없습니다.

실제 코드의 결정적 패턴:

```
conv2d_bias{f}  %t79{... 0x20채널 ..., 0x1b3e00}, %t45, %t74, %t77, %t56, {m(YR),...}
prelu           %t78{...            , 0x1b3e00}, %t79, 0x19, %t55
conv2d_bias{f}* %t88{... 0x20채널 ..., 0x1b3e20}, %t45, %t84, %t75, %t54, {m(YR),...}
prelu*          %t61{...            , 0x1b3e20}, %t88, 0x19, %t86
concat          %t89{... 0x40채널 ...}, %t78, %t61, {dir(Z)}
```

**같은 입력 `%t45` 에 다른 가중치를 물려 출력 채널을 반씩 두 레인에서 계산하고 채널 축으로 합칩니다.**
그리고 `prelu` 의 슬롯은 항상 앞선 `conv2d_bias` 의 슬롯과 일치합니다 —
즉 슬롯이 고르는 건 개별 유닛이 아니라 **NCU + VX 한 쌍으로 된 레인**입니다.

```
레인 0 (slot 0) :  NCU0 ──융합──▶ VX0
레인 1 (slot 1) :  NCU1 ──융합──▶ VX1
공유            :  CCU · DDMA · RESHAPER · NPU SRAM
```

nbc의 `-ex_cnt 2` 가 가리키는 것이 이 레인 2개입니다.
VLIW 발행 슬롯은 아닙니다 — 그랬다면 메모리·형태 명령도 슬롯을 나눠 썼을 텐데 예외 없이 0입니다.

---

## 실제 코드 — YOLOv3 첫 레이어

```
waitc -0x1, -0x1
load        %t0{0x1,p,u8,mt(0x260,0x34,0x3,0x1),st(0x280,0x34,0x3),NCHW,0x380},
              dram(0x0,0x260,0x260,0x3)                                    ← 입력 608×608×3
load        %t1{0xc,p,u8,mt(0x3,0x3,0x3,0x20),st(0x3,0x3,0x3),MCRSm,0x0},
              dram(0x10ec00,0x3,0x3,0x3)                                   ← 3×3 커널 가중치
load        %t2{0xc,p,i32,...,NHWC,0x1fff00}, dram(0x10ef80,0x20,0x1,0x1)  ← bias (i32)
set_mq_info %t3{0x1,r,mq,0x71,0x0,shift1(0x0,0x16,0x0),0x645a,0x1d}
set_dq_info %t4{0x1,r,dq,0x0,shift2(0x12,0x0,0x0),0x6666,0x0,...,0x1d}
conv2d_bias{f} %t5{...,NCHWc,0x1b700}, %t0, %t1, %t2, %t3,
              {m(OP), p(0x1,0x0,0x1,0x1), s(0x1,0x1), d(0x1,0x1)}          ← pad/stride/dilation
prelu       %t6{...}, %t5, 0x19, %t4                                       ← 융합 활성화
store       %t6, dram(0x10f000,0x260,0x260,0x40)
waitc{f} 0x0, 0x1
```

`conv2d_bias` 의 속성 블록: `m()` = 모드(`OP`, `YR` 관측), `p()` = 패딩 4방향,
`s()` = 스트라이드, `d()` = 딜레이션.

---

## 확인하지 못한 것

- `load_comp` · `waits` · `sync` · `move` · `clear` 계열의 정확한 의미 — 컴파일된 YOLOv3에 나타나지 않습니다
- **명령어에 엔진 종류 필드는 없습니다.** conv2d 계열이 NCU, 원소별/활성화가 VX라는 배분은 하드웨어 제원(64×32 TX array vs vector width 4 + LUT)에서 추론한 것입니다
- `set_sq_info` 는 관측되지 않았습니다 (YOLOv3는 `mq`/`dq` 만 사용)
- 32바이트 명령의 비트 필드 배치 — 인코더(`getSlot1Encoding` 등)를 더 파야 합니다
- `EM_SVX` 라는 ELF 머신 타입 문자열이 백엔드에 있는데 용도 미상입니다
