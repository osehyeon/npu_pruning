#!/usr/bin/env python3
"""정렬 계단의 원인 — 두 TX 유닛의 부하 분산.

컴파일러 스케줄 기록에서 유닛별 예상 시간을 뽑아
지연을 결정하는 것이 총합이 아니라 max(TX0, TX1) 임을 보인다.

    python3 analyze_tx_balance.py           # 어디서 실행해도 됨. ../../npu_pruning/ 을 상대 경로로 찾음
"""
import json, glob, re, collections, math, sys
from pathlib import Path
import onnx

ROOT = Path(__file__).resolve().parents[2] / "npu_pruning"
ART = ROOT / "example/nbc_artifacts"
MDIR = ROOT / "model/u8"

def channels(tag):
    """conv 이름 → 출력 채널. 2026-09-17 이전에는 그래프 순서의 위치로 반환해, 아래
    per_conv() 가 정규식으로 뽑은 이름의 숫자를 그 위치로 오인해 엉뚱한 채널 수를 붙이는
    버그가 있었다(n_conv2d_12 등 이름의 "_0" 없는 접미사도 Nota 의 내부 번호일 뿐 그래프
    위치가 아님). 이름으로 직접 찾도록 고쳤다."""
    m=onnx.load(str(MDIR/f"{tag}_u8_maD20.onnx"))
    init={t.name:t for t in m.graph.initializer}
    return {n.name.strip("/").split("/")[0]: init[n.input[3]].dims[0]
            for n in m.graph.node if n.op_type=="QLinearConv"}

def units(combo, tag):
    c=collections.Counter(); n=0
    for f in glob.glob(str(ART/combo/tag/"*group_*.json")):
        n+=1
        for e in json.load(open(f))["traceEvents"]:
            t=e.get("tid","")
            if t and t!="NONE" and not t.startswith("<"): c[t]+=float(e["dur"])
    return c, n

RE=re.compile(r"(/n_conv2d_\d+(?:_\d+)?)/Conv_quant__")
def per_conv(combo, tag):
    d=collections.defaultdict(lambda: collections.Counter())
    for f in glob.glob(str(ART/combo/tag/"*group_*.json")):
        for e in json.load(open(f))["traceEvents"]:
            t=e.get("tid","")
            if not t.startswith("TX"): continue
            m=RE.match(e["name"])
            if m: d[m.group(1)][t]+=float(e["dur"])
    return d

LAT={}
for ln in open("grid_latency_paired.tsv").read().splitlines()[1:]:
    f=ln.split("\t"); LAT[(f[0],f[1])]=float(f[2])

out=[]
out.append("# 정렬 계단의 원인 — TX 부하 분산\n")
out.append("## 유닛별 예상 시간 · ratio 50 · lw\n")
out.append(f"{'sc':>4}{'groups':>8}{'MEM_LDA':>10}{'MEM_STA':>10}{'MEM_LDW':>10}"
           f"{'TX0':>9}{'TX1':>9}{'TX합':>9}{'max(TX)':>9}{'실측ms':>9}")
for s in (8,16,32,64):
    tag=f"sw50sc{s}"; c,n=units("lw",tag)
    tx0,tx1=c['TX0'],c['TX1']
    out.append(f"{s:>4}{n:>8}{c['MEM_LDA']:10.2f}{c['MEM_STA']:10.2f}{c['MEM_LDW']:10.2f}"
               f"{tx0:9.2f}{tx1:9.2f}{tx0+tx1:9.2f}{max(tx0,tx1):9.2f}{LAT[(tag,'lw')]:9.2f}")
out.append("\nTX 총합은 sc64 가 가장 크지만 max(TX) 는 가장 작다.")
out.append("지연은 두 유닛의 합이 아니라 더 오래 걸리는 쪽이 결정한다.\n")

out.append("## 배분이 갈리는 Conv · sc32 대 sc64\n")
ch32, ch64 = channels("sw50sc32"), channels("sw50sc64")
d32, d64 = per_conv("lw","sw50sc32"), per_conv("lw","sw50sc64")
out.append(f"{'conv':14}{'Cout32':>8}{'Cout64':>8}"
           f"{'sc32 TX0':>10}{'sc32 TX1':>10}{'sc64 TX0':>10}{'sc64 TX1':>10}")
# 배분이 바뀐 정도(균형의 변화)로 순위를 매긴다. 2026-09-17 이전에는 TX 총합으로 순위를
# 매겼는데, 채널 버그와 맞물려 눈에 띄는 여덞 개를 우연히도 찾아냈을 뿐이었다.
keys=sorted(set(d32)|set(d64),
            key=lambda k: -abs((d32[k]['TX0']-d32[k]['TX1'])-(d64[k]['TX0']-d64[k]['TX1'])))
for k in keys[:12]:
    name=k.strip("/")
    a=ch32.get(name,0); b=ch64.get(name,0)
    out.append(f"{k:14}{a:>8}{b:>8}{d32[k]['TX0']:10.3f}{d32[k]['TX1']:10.3f}"
               f"{d64[k]['TX0']:10.3f}{d64[k]['TX1']:10.3f}")
out.append("\nCout 가 224 에서 256 으로 늘어난 conv 여덜 개가 sc32 에서는 한 유닛이 통째로")
out.append("맡고 sc64 에서는 둘로 갈라진다. 224 는 4 개 타일 중 마지막 하나가 32 채널만")
out.append("채운 것이고 256 은 4 개 타일이 전부 64 채널로 꽉 찬 것이다.\n")

out.append("## 전 조합 요약\n")
out.append(f"{'ratio':>6}{'sc':>4}{'combo':>9}{'TX0':>9}{'TX1':>9}{'max(TX)':>9}{'실측ms':>9}")
for r in (10,30,50):
    for s in (8,16,32,64):
        for combo in ("lw","lwhalf","egs","egshalf"):
            tag=f"sw{r}sc{s}"; c,_=units(combo,tag)
            if not c: continue
            out.append(f"{r:>5}%{s:>4}{combo:>9}{c['TX0']:9.2f}{c['TX1']:9.2f}"
                       f"{max(c['TX0'],c['TX1']):9.2f}{LAT[(tag,combo)]:9.2f}")
txt="\n".join(out)
print(txt)
Path("tx_balance_analysis.txt").write_text(txt)
