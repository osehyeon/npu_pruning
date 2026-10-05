import sys,time,numpy as np,cv2,onnx,onnxruntime as ort
sys.path.insert(0,'.')
from pathlib import Path
from NeublaDriver import NeublaDriver
H=Path.home(); SIZE=608
so=ort.SessionOptions(); so.log_severity_level=3
mp=H/"workspace_docs/model/u8/p40ep10_u8_maD20.onnx"
m=onnx.load(str(mp)); n=m.graph.input[0].name; q0=m.graph.node[0].output[0]
f=Path("/tmp/p40.front.onnx")
if not f.exists(): onnx.utils.extract_model(str(mp),str(f),[n],[q0])
fr=ort.InferenceSession(str(f),so,providers=["CPUExecutionProvider"])
def prep(p):
    x=(cv2.resize(cv2.cvtColor(cv2.imread(p),cv2.COLOR_BGR2RGB),(SIZE,SIZE)
       ).astype(np.float32)/255).transpose(2,0,1)[None]
    return fr.run(None,{n:x})[0].tobytes()
A=prep(str(H/"datasets/coco/images/val2017/000000000139.jpg"))
B=prep(str(H/"datasets/coco/images/val2017/000000000285.jpg"))
d=NeublaDriver(); d.Init()
for t in range(8):
    if d.LoadModel(str(H/"workspace_docs/example/p40ep10_lwhalf.o"))==0: break
def run(q):
    t0=time.perf_counter(); d.SendInput(q,3*SIZE*SIZE); d.Launch(); d.Wait()
    o=d.ReceiveOutputs(); return (time.perf_counter()-t0)*1000, np.frombuffer(o[0],np.uint8).copy()
prev=None
for i in range(6):
    q,tag = (A,"A") if i%2==0 else (B,"B")
    t,o = run(q)
    same = "이전과 동일" if prev is not None and np.array_equal(o,prev) else "새 출력"
    print(f"  {i} 입력{tag} {t:7.0f} ms  {same}")
    prev=o
