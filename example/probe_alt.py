import sys,time,numpy as np,cv2,onnx,onnxruntime as ort
sys.path.insert(0,'.')
from pathlib import Path
from NeublaDriver import NeublaDriver
H=Path.home(); SIZE=608; tag=sys.argv[1]; combo=sys.argv[2]
so=ort.SessionOptions(); so.log_severity_level=3
img=cv2.imread(str(H/"datasets/coco/images/val2017/000000000139.jpg"))
x=(cv2.resize(cv2.cvtColor(img,cv2.COLOR_BGR2RGB),(SIZE,SIZE)).astype(np.float32)/255).transpose(2,0,1)[None]
mp=H/f"workspace_docs/model/u8/{tag}_u8_maD20.onnx"
m=onnx.load(str(mp)); n=m.graph.input[0].name; q0=m.graph.node[0].output[0]
f=Path(f"/tmp/{tag}.front.onnx")
if not f.exists(): onnx.utils.extract_model(str(mp),str(f),[n],[q0])
q=ort.InferenceSession(str(f),so,providers=["CPUExecutionProvider"]).run(None,{n:x})[0].tobytes()
d=NeublaDriver(); d.Init()
for t in range(8):
    if d.LoadModel(str(H/f"workspace_docs/example/{tag}_{combo}.o"))==0: break
def run(gap=0.0):
    if gap: time.sleep(gap)
    t0=time.perf_counter(); d.SendInput(q,3*SIZE*SIZE); d.Launch(); d.Wait()
    o=d.ReceiveOutputs(); return (time.perf_counter()-t0)*1000, o
r=[run()[0] for _ in range(8)]
print("gap 0.0 :", " ".join(f"{t:.0f}" for t in r))
r=[run(0.2)[0] for _ in range(8)]
print("gap 0.2 :", " ".join(f"{t:.0f}" for t in r))
a,oa=run(); b,ob=run()
na=np.frombuffer(oa[0],dtype=np.uint8); nb=np.frombuffer(ob[0],dtype=np.uint8)
print(f"빠름 {a:.0f} ms · 느림 {b:.0f} ms · 출력 동일 {np.array_equal(na,nb)}")
