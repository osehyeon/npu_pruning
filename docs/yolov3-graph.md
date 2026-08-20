# YOLOv3 그래프 전개

`model/yolov3_original_fp32.onnx` 를 노드 단위로 그대로 펼친 것임.
onnx shape inference 로 얻은 실제 텐서 형상이고 배치 축은 생략했음.

노드 174개 · Conv 75 · LeakyRelu 72 · Add 23 · Resize 2 · Concat 2.
BN 은 Conv 에 접혀 있어 별도 노드가 없음.

오른쪽 세로선은 건너뛰는 간선임. `-+` 에서 갈라져 나가 `<+` 로 합류함.
Add 는 residual 이고 Concat 은 route 임.

| 기호 | 뜻 |
|---|---|
| `-+` | 이 노드의 출력이 아래쪽 어딘가로 분기함 |
| `\|` | 분기한 값이 지나가는 중 |
| `<+` | 여기서 합류함 |
| `s` | stride |
| `p` | pad |
| `a` | LeakyRelu 기울기 |

weight 열은 Cout x Cin x k x k 임.

```
  #   op          weight / attr           output        
---------------------------------------------------------
      input                               3x608x608
  0   Conv_0      32x3x3x3      s1 p1   32x608x608
  1   LeakyRelu                 a=0.1   32x608x608
  2   Conv_1      64x32x3x3     s2 p1   64x304x304
  3   LeakyRelu                 a=0.1   64x304x304      -+
  4   Conv_2      32x64x1x1     s1 p0   32x304x304       |
  5   LeakyRelu                 a=0.1   32x304x304       |
  6   Conv_3      64x32x3x3     s1 p1   64x304x304       |
  7   LeakyRelu                 a=0.1   64x304x304       |
  8   Add                               64x304x304      <+
  9   Conv_4      128x64x3x3    s2 p1   128x152x152
 10   LeakyRelu                 a=0.1   128x152x152     -+
 11   Conv_5      64x128x1x1    s1 p0   64x152x152       |
 12   LeakyRelu                 a=0.1   64x152x152       |
 13   Conv_6      128x64x3x3    s1 p1   128x152x152      |
 14   LeakyRelu                 a=0.1   128x152x152      |
 15   Add                               128x152x152     <+
 16   Conv_7      64x128x1x1    s1 p0   64x152x152       |
 17   LeakyRelu                 a=0.1   64x152x152       |
 18   Conv_8      128x64x3x3    s1 p1   128x152x152      |
 19   LeakyRelu                 a=0.1   128x152x152      |
 20   Add                               128x152x152     <+
 21   Conv_9      256x128x3x3   s2 p1   256x76x76
 22   LeakyRelu                 a=0.1   256x76x76       -+
 23   Conv_10     128x256x1x1   s1 p0   128x76x76        |
 24   LeakyRelu                 a=0.1   128x76x76        |
 25   Conv_11     256x128x3x3   s1 p1   256x76x76        |
 26   LeakyRelu                 a=0.1   256x76x76        |
 27   Add                               256x76x76       <+
 28   Conv_12     128x256x1x1   s1 p0   128x76x76        |
 29   LeakyRelu                 a=0.1   128x76x76        |
 30   Conv_13     256x128x3x3   s1 p1   256x76x76        |
 31   LeakyRelu                 a=0.1   256x76x76        |
 32   Add                               256x76x76       <+
 33   Conv_14     128x256x1x1   s1 p0   128x76x76        |
 34   LeakyRelu                 a=0.1   128x76x76        |
 35   Conv_15     256x128x3x3   s1 p1   256x76x76        |
 36   LeakyRelu                 a=0.1   256x76x76        |
 37   Add                               256x76x76       <+
 38   Conv_16     128x256x1x1   s1 p0   128x76x76        |
 39   LeakyRelu                 a=0.1   128x76x76        |
 40   Conv_17     256x128x3x3   s1 p1   256x76x76        |
 41   LeakyRelu                 a=0.1   256x76x76        |
 42   Add                               256x76x76       <+
 43   Conv_18     128x256x1x1   s1 p0   128x76x76        |
 44   LeakyRelu                 a=0.1   128x76x76        |
 45   Conv_19     256x128x3x3   s1 p1   256x76x76        |
 46   LeakyRelu                 a=0.1   256x76x76        |
 47   Add                               256x76x76       <+
 48   Conv_20     128x256x1x1   s1 p0   128x76x76        |
 49   LeakyRelu                 a=0.1   128x76x76        |
 50   Conv_21     256x128x3x3   s1 p1   256x76x76        |
 51   LeakyRelu                 a=0.1   256x76x76        |
 52   Add                               256x76x76       <+
 53   Conv_22     128x256x1x1   s1 p0   128x76x76        |
 54   LeakyRelu                 a=0.1   128x76x76        |
 55   Conv_23     256x128x3x3   s1 p1   256x76x76        |
 56   LeakyRelu                 a=0.1   256x76x76        |
 57   Add                               256x76x76       <+
 58   Conv_24     128x256x1x1   s1 p0   128x76x76        |
 59   LeakyRelu                 a=0.1   128x76x76        |
 60   Conv_25     256x128x3x3   s1 p1   256x76x76        |
 61   LeakyRelu                 a=0.1   256x76x76        |
 62   Add                               256x76x76       <+-----+
 63   Conv_26     512x256x3x3   s2 p1   512x38x38              |
 64   LeakyRelu                 a=0.1   512x38x38       -+     |
 65   Conv_27     256x512x1x1   s1 p0   256x38x38        |     |
 66   LeakyRelu                 a=0.1   256x38x38        |     |
 67   Conv_28     512x256x3x3   s1 p1   512x38x38        |     |
 68   LeakyRelu                 a=0.1   512x38x38        |     |
 69   Add                               512x38x38       <+     |
 70   Conv_29     256x512x1x1   s1 p0   256x38x38        |     |
 71   LeakyRelu                 a=0.1   256x38x38        |     |
 72   Conv_30     512x256x3x3   s1 p1   512x38x38        |     |
 73   LeakyRelu                 a=0.1   512x38x38        |     |
 74   Add                               512x38x38       <+     |
 75   Conv_31     256x512x1x1   s1 p0   256x38x38        |     |
 76   LeakyRelu                 a=0.1   256x38x38        |     |
 77   Conv_32     512x256x3x3   s1 p1   512x38x38        |     |
 78   LeakyRelu                 a=0.1   512x38x38        |     |
 79   Add                               512x38x38       <+     |
 80   Conv_33     256x512x1x1   s1 p0   256x38x38        |     |
 81   LeakyRelu                 a=0.1   256x38x38        |     |
 82   Conv_34     512x256x3x3   s1 p1   512x38x38        |     |
 83   LeakyRelu                 a=0.1   512x38x38        |     |
 84   Add                               512x38x38       <+     |
 85   Conv_35     256x512x1x1   s1 p0   256x38x38        |     |
 86   LeakyRelu                 a=0.1   256x38x38        |     |
 87   Conv_36     512x256x3x3   s1 p1   512x38x38        |     |
 88   LeakyRelu                 a=0.1   512x38x38        |     |
 89   Add                               512x38x38       <+     |
 90   Conv_37     256x512x1x1   s1 p0   256x38x38        |     |
 91   LeakyRelu                 a=0.1   256x38x38        |     |
 92   Conv_38     512x256x3x3   s1 p1   512x38x38        |     |
 93   LeakyRelu                 a=0.1   512x38x38        |     |
 94   Add                               512x38x38       <+     |
 95   Conv_39     256x512x1x1   s1 p0   256x38x38        |     |
 96   LeakyRelu                 a=0.1   256x38x38        |     |
 97   Conv_40     512x256x3x3   s1 p1   512x38x38        |     |
 98   LeakyRelu                 a=0.1   512x38x38        |     |
 99   Add                               512x38x38       <+     |
100   Conv_41     256x512x1x1   s1 p0   256x38x38        |     |
101   LeakyRelu                 a=0.1   256x38x38        |     |
102   Conv_42     512x256x3x3   s1 p1   512x38x38        |     |
103   LeakyRelu                 a=0.1   512x38x38        |     |
104   Add                               512x38x38       <+--+  |
105   Conv_43     1024x512x3x3  s2 p1   1024x19x19          |  |
106   LeakyRelu                 a=0.1   1024x19x19      -+  |  |
107   Conv_44     512x1024x1x1  s1 p0   512x19x19        |  |  |
108   LeakyRelu                 a=0.1   512x19x19        |  |  |
109   Conv_45     1024x512x3x3  s1 p1   1024x19x19       |  |  |
110   LeakyRelu                 a=0.1   1024x19x19       |  |  |
111   Add                               1024x19x19      <+  |  |
112   Conv_46     512x1024x1x1  s1 p0   512x19x19        |  |  |
113   LeakyRelu                 a=0.1   512x19x19        |  |  |
114   Conv_47     1024x512x3x3  s1 p1   1024x19x19       |  |  |
115   LeakyRelu                 a=0.1   1024x19x19       |  |  |
116   Add                               1024x19x19      <+  |  |
117   Conv_48     512x1024x1x1  s1 p0   512x19x19        |  |  |
118   LeakyRelu                 a=0.1   512x19x19        |  |  |
119   Conv_49     1024x512x3x3  s1 p1   1024x19x19       |  |  |
120   LeakyRelu                 a=0.1   1024x19x19       |  |  |
121   Add                               1024x19x19      <+  |  |
122   Conv_50     512x1024x1x1  s1 p0   512x19x19        |  |  |
123   LeakyRelu                 a=0.1   512x19x19        |  |  |
124   Conv_51     1024x512x3x3  s1 p1   1024x19x19       |  |  |
125   LeakyRelu                 a=0.1   1024x19x19       |  |  |
126   Add                               1024x19x19      <+  |  |
127   Conv_52     512x1024x1x1  s1 p0   512x19x19           |  |
128   LeakyRelu                 a=0.1   512x19x19           |  |
129   Conv_53     1024x512x3x3  s1 p1   1024x19x19          |  |
130   LeakyRelu                 a=0.1   1024x19x19          |  |
131   Conv_54     512x1024x1x1  s1 p0   512x19x19           |  |
132   LeakyRelu                 a=0.1   512x19x19           |  |
133   Conv_55     1024x512x3x3  s1 p1   1024x19x19          |  |
134   LeakyRelu                 a=0.1   1024x19x19          |  |
135   Conv_56     512x1024x1x1  s1 p0   512x19x19           |  |
136   LeakyRelu                 a=0.1   512x19x19           |  |
137   Conv_57     256x512x1x1   s1 p0   256x19x19           |  |
138   Conv_58     1024x512x3x3  s1 p1   1024x19x19          |  |
139   LeakyRelu                 a=0.1   256x19x19       -+  |  |
140   LeakyRelu                 a=0.1   1024x19x19       |  |  |
141   Resize                    x2      256x38x38       <+  |  |
142   Conv_59     255x1024x1x1  s1 p0   255x19x19        |  |  |
143   Concat                    axis=1  768x38x38       <+--+  |
144   Conv_60     256x768x1x1   s1 p0   256x38x38              |
145   LeakyRelu                 a=0.1   256x38x38              |
146   Conv_61     512x256x3x3   s1 p1   512x38x38              |
147   LeakyRelu                 a=0.1   512x38x38              |
148   Conv_62     256x512x1x1   s1 p0   256x38x38              |
149   LeakyRelu                 a=0.1   256x38x38              |
150   Conv_63     512x256x3x3   s1 p1   512x38x38              |
151   LeakyRelu                 a=0.1   512x38x38              |
152   Conv_64     256x512x1x1   s1 p0   256x38x38              |
153   LeakyRelu                 a=0.1   256x38x38              |
154   Conv_65     128x256x1x1   s1 p0   128x38x38              |
155   Conv_66     512x256x3x3   s1 p1   512x38x38              |
156   LeakyRelu                 a=0.1   128x38x38       -+     |
157   LeakyRelu                 a=0.1   512x38x38        |     |
158   Resize                    x2      128x76x76       <+     |
159   Conv_67     255x512x1x1   s1 p0   255x38x38        |     |
160   Concat                    axis=1  384x76x76       <+-----+
161   Conv_68     128x384x1x1   s1 p0   128x76x76
162   LeakyRelu                 a=0.1   128x76x76
163   Conv_69     256x128x3x3   s1 p1   256x76x76
164   LeakyRelu                 a=0.1   256x76x76
165   Conv_70     128x256x1x1   s1 p0   128x76x76
166   LeakyRelu                 a=0.1   128x76x76
167   Conv_71     256x128x3x3   s1 p1   256x76x76
168   LeakyRelu                 a=0.1   256x76x76
169   Conv_72     128x256x1x1   s1 p0   128x76x76
170   LeakyRelu                 a=0.1   128x76x76
171   Conv_73     256x128x3x3   s1 p1   256x76x76
172   LeakyRelu                 a=0.1   256x76x76
173   Conv_74     255x256x1x1   s1 p0   255x76x76
```

## 그래프 출력

| 출력 | 노드 | 형상 | 격자 | 담당 |
|---|---|---|---|---|
| OUT 1 | Conv_59 | 255x19x19 | 19 x 19 | 큰 객체 |
| OUT 2 | Conv_67 | 255x38x38 | 38 x 38 | 중간 객체 |
| OUT 3 | Conv_74 | 255x76x76 | 76 x 76 | 작은 객체 |

255 는 3앵커 x 85 이고 85 는 박스 4 + objectness 1 + 클래스 80 임.
Conv_59 · Conv_67 · Conv_74 만 LeakyRelu 가 뒤따르지 않음.

## 해상도 전환

stride 2 인 Conv 는 다섯 개임.

| Conv | 전 | 후 |
|---|---|---|
| Conv_1 | 608 | 304 |
| Conv_4 | 304 | 152 |
| Conv_9 | 152 | 76 |
| Conv_26 | 76 | 38 |
| Conv_43 | 38 | 19 |

Resize 는 route 경로에서만 쓰이고 19 에서 38 로, 38 에서 76 으로 키움.

## 건너뛰는 간선 29개

| 종류 | 개수 | 거리 |
|---|---|---|
| residual · Add | 23 | 노드 5칸 |
| route · Concat 로 가는 백본 출력 | 2 | 39칸과 98칸 |
| Resize 앞뒤 | 4 | 2칸 |

residual 은 전부 블록 내부라 거리가 일정함. route 두 개만 그래프를 가로질러
길게 이어짐. Add_62 가 Concat_160 으로, Add_104 가 Concat_143 으로 감.
