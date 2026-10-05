# uint8 재양자화 모델

2026-09-16 에 워크스테이션 `~/workspace_docs/model/u8/` 에서 원본 34 개를 그대로 받았고, 그중 우리가 재학습한 `p40ep10_u8_maD20.onnx` 는 논문에 쓰지 않아
`fgcs/archive/neubla/models-not-in-paper/` 로 옮겨 33 개가 남았습니다.
지문은 `MD5SUMS` 에 있습니다.

| 꼬리표 | 뜻 |
|---|---|
| `*_u8_maD20.onnx` | 논문이 쓴 것. MovingAverage 캘리브레이션 20 장. 15 종 |
| `*_u8_ma1` `*_u8_ma20` `*_u8_ma100` | 캘리브레이션 장수 스윕 |
| `*_u8_mm*` | MinMax 캘리브레이션 대조군 |

다시 만들어야 할 때는 `example/requantize.py --sweep-all --method MovingAverage --tag D --calib 20`
이고 ONNX Runtime 1.12.1 이 필요합니다. 원본이 있으므로 재생성보다 원본을 쓰십시오.
