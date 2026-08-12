#!/usr/bin/env python3
"""헤드 뒤의 Reshape 왕복을 제거해 nbc 가 받는 형태로 만든다.

`export_darknet_onnx.py` 로 내보낸 그래프는 헤드마다 Reshape 가 두 겹 붙는다:

    Conv(105) → Reshape(module_list.106)  (1,255,H,W)→(1,3,H,W,85)
              → Reshape(/Reshape)         (1,3,H,W,85)→(1,255,H,W)   = 항등

`YOLOLayer` 의 축 변환과 그것을 되돌리는 변환이 그대로 남은 것이다. 수치는 같지만
`nbc` 가 이 패턴에서 죽는다:

    F ModelBuilder.cpp:187] Check failed: offset >= 0 (-1 vs. 0)

Nota 배포본(178 노드, op 7종)에는 Reshape 가 아예 없다. 그래서 마지막 Conv 출력을
그대로 그래프 출력으로 승격시켜 Reshape 계열을 잘라낸다.

    python strip_head_reshape.py --in x.onnx --out y.onnx
"""
from __future__ import annotations

import argparse
from pathlib import Path

import onnx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    args = ap.parse_args()

    m = onnx.load(args.src)
    g = m.graph
    produced = {o: n for n in g.node for o in n.output}
    out_names = [o.name for o in g.output]

    # 각 그래프 출력에서 Reshape/Constant 를 거슬러 올라가 최초의 Conv 를 찾는다
    keep_out, drop = [], set()
    for name in out_names:
        cur = name
        while True:
            n = produced.get(cur)
            if n is None or n.op_type not in ("Reshape", "Transpose", "Squeeze", "Unsqueeze"):
                break
            drop.add(n.name)
            for i in n.input[1:]:                     # shape 상수도 제거 대상
                c = produced.get(i)
                if c is not None and c.op_type == "Constant":
                    drop.add(c.name)
            cur = n.input[0]
        keep_out.append((name, cur, produced.get(cur)))

    print(f"입력 그래프: 노드 {len(g.node)} · 출력 {out_names}")
    for name, src, n in keep_out:
        print(f"  {name} ← {n.op_type if n else '?'} {src[:45]}")

    # Reshape 계열 제거
    new_nodes = [n for n in g.node if n.name not in drop]
    print(f"제거한 노드 {len(g.node) - len(new_nodes)}개")
    del g.node[:]
    g.node.extend(new_nodes)

    # 출력 이름을 Conv 출력으로 교체 (형상은 추론에 맡김)
    old_outputs = list(g.output)
    del g.output[:]
    for (name, src, _), vi in zip(keep_out, old_outputs):
        nv = onnx.ValueInfoProto()
        nv.CopyFrom(vi)
        nv.name = src
        g.output.append(nv)

    m = onnx.shape_inference.infer_shapes(m)
    onnx.checker.check_model(m)
    onnx.save(m, args.dst)
    ops = sorted({n.op_type for n in m.graph.node})
    print(f"\n출력 그래프: 노드 {len(m.graph.node)} · op {len(ops)}종 {ops}")
    print(f"출력 텐서: {[o.name for o in m.graph.output]}")
    print(f"→ {args.dst}  ({Path(args.dst).stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
