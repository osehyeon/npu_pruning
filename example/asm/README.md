# 컴파일러 인라인 어셈블리 · 명령 스트림

`nbc` 는 `.o` 를 만들기 전에 `~/demo_0211/nbc/neubla/gen_inline_assembly.cpp` 를 남긴다. 명령 하나가
`asm("...")` 한 줄이고 텐서 레지스터에 타일 형상 `mt(W,H,C,N)` 과 스트라이드 `st(...)` 가 적혀 있다.
가중치는 `MCRS*` 레이아웃이고 `mt(R,S,Cin,M)` 이다. 마지막 컴파일의 것만 남으므로 모델별로 다시
컴파일해서 뽑아야 한다. 컴파일러는 호스트에서 돌고 보드가 필요 없다.

    # 워크스테이션
    cd ~/workspace_docs/example
    ./compile_variant.sh ~/workspace_docs/model/u8/sw50sc8_u8_maD20.onnx input.1 /tmp/x.o lw single
    grep -v "__declspec" ~/demo_0211/nbc/neubla/gen_inline_assembly.cpp | grep -n 'asm(\|^//' > sw50sc8_lw_asm.txt

2026-09-16 에 sw50sc8 을 lw single 로 다시 컴파일한 `.o` 의 md5 가 `example/sw50sc8_lw.o` 와 같았다.
컴파일은 결정적이고, 이 명령 스트림이 그 바이너리의 것이다. 해석은 `paper/data/asm_conv_tiles.py`.

| 파일 | 내용 |
|---|---|
| `sw50sc8_lw_asm.txt` | 4,792 줄. 가중치 데이터 배열은 뺐다 |

## `.o` 를 직접 읽기 · 2026-09-16

절차 전체는 [`docs/reading-neubla-object.md`](../../docs/reading-neubla-object.md) 에 있습니다.

벤더 `llvm-objdump` 는 이 `.o` 들에서 assertion 으로 멈추지만 `llvm-mc --triple=neubla -show-encoding` 은 동작한다.
위 텍스트 4,283 줄을 `llvm-mc` 로 어셈블한 인코딩이 `.o` 의 `.text` 32 바이트 워드 4,283 개와 전부 같았다.
같은 명령을 값 하나씩 바꿔 어셈블해 필드 위치를 정했고, 그 필드 맵으로 텍스트 없이 `.o` 만 읽는 해독기가
`paper/data/decode_o.py` 다. sw50sc8 에서 층별 명령 수, 출력 채널 M, 입력 채널 C, 명령 MACs 가 텍스트 해석과
75 층 전부 같았고, 13 종 `.o` 에서 명령이 지시한 MACs 합이 ONNX 원형 MACs 와 정확히 같았다(`decode_o.txt`).

| 바이트 | 필드 | 식 |
|---|---|---|
| 31 | opcode | conv2d_bias 0xe8 · load 는 0x80 미만이고 하위 7 비트가 목적지 레지스터 |
| 4 | 레이아웃 (하위 3 비트) | MCRSm 6 · MCRScm 7 · NCHW 1 · NHWC 2 · NCHWc 5 |
| 0..2 | SRAM 주소 | LE24 |
| 6, 8 | R, S | b6>>2, b8 |
| 9,10 | 가중치 M | LE16>>6 |
| 11,12 | 채널 C | LE16>>4 |
| 13,14 | 타일 H | LE16>>2 |
| 15,16 | 타일 W | LE16 & 0xfff |
| 5 | 출력 스트라이드 채널 | b5<<4 |
| 19..21 | DRAM 주소 | LE24 × 64 |
| 27,28 · 28,29 · 29,30 | conv 의 가중치 · 입력 · 목적지 레지스터 | (LE16>>6)&0x7f · (LE16>>5)&0x7f · (b29>>4)\|((b30&7)<<4) |
