#!/usr/bin/env python3
"""Antara NPU 에서 YOLOv3 이미지 한 장을 추론하는 최소 예제.

demo_1127/yolo_demo.py(480줄, 비디오·카메라·GUI 포함)에서 핵심 경로만 뽑아
스스로 검증 가능하게 다시 쓴 것입니다.

추론 경로는 3단입니다:

    이미지 → [전처리] → front ONNX (CPU)  ─ 양자화 ─→ uint8
                          ↓
                       NPU (.o 바이너리)   ─ 3개 헤드 ─→ uint8
                          ↓
                     [역양자화] → back ONNX (CPU) → dets, labels → 그리기

모드:
    --mode check   하드웨어를 건드리지 않고 환경·자산만 점검
    --mode cpu     NPU 없이 원본 ONNX 를 CPU 로 통째로 실행 (기준값)
    --mode npu     실제 NPU 실행 (기본)
    --mode both    cpu 와 npu 를 모두 실행하고 결과를 비교

자세한 배경: ../docs/why-docker-for-demos.md
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

# ─────────────────────────────────────────────── 모델 규격 (원본 데모와 동일)

INPUT_SIZE = (608, 608)

FRONT_INPUT_NAMES = ["input"]
FRONT_OUTPUT_NAMES = ["input_quantized"]

# NPU 출력 3개 헤드에 대응하는 DequantizeLinear 노드
TARGET_DEQUANT_LAYER_NAMES = [
    "onnx::Transpose_684_DequantizeLinear",
    "onnx::Transpose_688_DequantizeLinear",
    "onnx::Transpose_692_DequantizeLinear",
]
BACK_INPUT_NAMES = ["onnx::Transpose_684", "onnx::Transpose_688", "onnx::Transpose_692"]
BACK_OUTPUT_NAMES = ["dets", "labels"]

# 608x608 입력일 때 YOLOv3 헤드 해상도
YOLO_HEAD_SPECS = ((255, 19, 19), (255, 38, 38), (255, 76, 76))

DEMO = Path.home() / "demo_1127"
DEFAULT_MODEL = DEMO / "yolov3/yolov3_d53_mstrain-608_273e_coco_optim_opset12.neubla_u8_lwq_movingaverage.onnx"
DEFAULT_BINARY = DEMO / "yolov3/yolov3_lw.o"
DEFAULT_IMAGE = DEMO / "yolov3/dog.jpg"


def log(stage: str, msg: str) -> None:
    print(f"[{stage:>7}] {msg}", flush=True)


def die(msg: str, hint: str = "") -> None:
    print(f"\n✗ {msg}", file=sys.stderr)
    if hint:
        print(f"  → {hint}", file=sys.stderr)
    sys.exit(1)


# ─────────────────────────────────────────────── 1. 환경 점검


def check_env(args: argparse.Namespace) -> None:
    """하드웨어를 건드리기 전에 막힐 만한 것을 전부 먼저 확인합니다.

    check 모드에서는 NPU 항목까지 전부 점검하되 실패해도 죽지 않습니다.
    """
    dry = args.mode == "check"          # 점검만 — 실패해도 중단하지 않음
    need_npu = dry or args.mode in ("npu", "both")
    ok = True

    log("check", f"python {sys.version.split()[0]}  ({sys.executable})")
    if sys.version_info[:2] != (3, 8) and need_npu:
        print("         ! NeublaDriver 는 cpython-38 전용입니다. NPU 모드는 3.8 이 필요합니다.")
        print("           예: ~/miniforge3/envs/test/bin/python (3.8.20)")
        ok = False

    for mod in ("cv2", "numpy", "onnx", "onnxruntime"):
        try:
            m = __import__(mod)
            log("check", f"{mod} {getattr(m, '__version__', '?')}")
        except ImportError:
            print(f"         ! {mod} 없음")
            ok = False

    for label, path in (("모델", args.model), ("바이너리", args.binary), ("이미지", args.image)):
        p = Path(path)
        if p.exists():
            log("check", f"{label}: {p.name} ({p.stat().st_size / 1e6:.1f} MB)")
        else:
            print(f"         ! {label} 없음: {p}")
            ok = False

    if need_npu:
        dev = Path("/dev/plda0")
        if not dev.exists():
            print("         ! /dev/plda0 없음 — 드라이버가 로드되지 않았습니다")
            ok = False
        else:
            try:
                dev.open("rb").close()
                log("check", "/dev/plda0 열기 가능")
            except PermissionError:
                print("         ! /dev/plda0 권한 없음 (crw------- root root)")
                print("           udev 규칙 설치 또는 sudo 필요 — docs/why-docker-for-demos.md 참조")
                ok = False
        try:
            __import__("NeublaDriver")
            log("check", "NeublaDriver import 가능")
        except ImportError as e:
            print(f"         ! NeublaDriver import 실패: {e}")
            print(f"           PYTHONPATH 에 {DEMO} 를 넣어야 합니다")
            ok = False

    print()
    if ok:
        log("check", "이상 없음")
    elif dry:
        log("check", "위 ! 항목이 해결되어야 --mode npu 를 쓸 수 있습니다 (--mode cpu 는 가능할 수 있음)")
    else:
        die("점검에서 걸린 항목이 있습니다", "--mode check 로 전체 점검 후 ! 표시를 해결하세요")


# ─────────────────────────────────────────────── 2. 전처리 / 후처리


def preprocess(bgr: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
    import cv2

    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, INPUT_SIZE)
    image = (resized.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
    return image, (w, h)


def draw(img: np.ndarray, dets: np.ndarray, labels: np.ndarray,
         img_wh: Tuple[int, int], classes: Sequence[str], conf: float) -> Tuple[np.ndarray, int]:
    import cv2

    img_w, img_h = img_wh
    sx, sy = img_w / INPUT_SIZE[0], img_h / INPUT_SIZE[1]
    rng = np.random.default_rng(0xC0C0)
    palette = rng.uniform(0, 255, size=(len(classes), 3)).astype(np.uint8)

    out = img.copy()
    n = 0
    for det, cls in zip(dets[0], labels[0]):
        x1, y1, x2, y2, score = det
        if score < conf:
            continue
        n += 1
        x1, y1, x2, y2 = int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)
        color = tuple(int(c) for c in palette[int(cls)])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        text = f"{classes[int(cls)]}: {score:.2f}"
        cv2.putText(out, text, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return out, n


COCO80 = (
    "person bicycle car motorcycle airplane bus train truck boat traffic_light "
    "fire_hydrant stop_sign parking_meter bench bird cat dog horse sheep cow "
    "elephant bear zebra giraffe backpack umbrella handbag tie suitcase frisbee "
    "skis snowboard sports_ball kite baseball_bat baseball_glove skateboard "
    "surfboard tennis_racket bottle wine_glass cup fork knife spoon bowl banana "
    "apple sandwich orange broccoli carrot hot_dog pizza donut cake chair couch "
    "potted_plant bed dining_table toilet tv laptop mouse remote keyboard "
    "cell_phone microwave oven toaster sink refrigerator book clock vase "
    "scissors teddy_bear hair_drier toothbrush"
).split()


def load_classes() -> List[str]:
    """ultralytics 가 있으면 그쪽 이름을, 없으면 내장 COCO-80 을 씁니다.

    ultralytics 는 데모의 원래 의존성이지만 이 예제에는 필수가 아닙니다.
    """
    try:
        from ultralytics.utils import YAML
        from ultralytics.utils.checks import check_yaml

        return YAML.load(check_yaml("coco128.yaml"))["names"]
    except Exception:
        return list(COCO80)


# ─────────────────────────────────────────────── 3. CPU 기준 경로


def run_cpu(args: argparse.Namespace, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """원본 양자화 ONNX 를 CPU 로 통째로 실행합니다. NPU 불필요."""
    import onnxruntime as ort

    log("cpu", "세션 생성 중... (62MB 모델이라 시간이 걸립니다)")
    sess = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    t0 = time.perf_counter()
    dets, labels = sess.run(BACK_OUTPUT_NAMES, {FRONT_INPUT_NAMES[0]: image})
    dt = time.perf_counter() - t0
    log("cpu", f"추론 완료 {dt * 1000:.0f} ms")
    return dets, labels, dt


# ─────────────────────────────────────────────── 4. NPU 경로


def split_model(model_path: Path) -> Tuple[Path, Path]:
    """front/back 서브그래프를 만듭니다 (이미 있으면 재사용)."""
    import onnx

    front, back = Path(str(model_path) + ".front"), Path(str(model_path) + ".back")
    if front.exists() and back.exists():
        log("split", "기존 front/back 서브그래프 재사용")
        return front, back

    log("split", "front/back 서브그래프 생성 중...")
    onnx.utils.extract_model(str(model_path), str(front), FRONT_INPUT_NAMES, FRONT_OUTPUT_NAMES)
    onnx.utils.extract_model(str(model_path), str(back), BACK_INPUT_NAMES, BACK_OUTPUT_NAMES)
    return front, back


def load_dequant_params(model_path: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """NPU 출력을 float 로 되돌릴 scale / zero_point 를 모읍니다."""
    import onnx

    m = onnx.load(str(model_path))
    inits = {i.name: i for i in m.graph.initializer}
    scale: Dict[str, np.ndarray] = {}
    zero: Dict[str, np.ndarray] = {}
    for node in m.graph.node:
        if node.name in TARGET_DEQUANT_LAYER_NAMES:
            s, z = node.input[1], node.input[2]
            if s in inits:
                scale[node.name] = onnx.numpy_helper.to_array(inits[s])
            if z in inits:
                zero[node.name] = onnx.numpy_helper.to_array(inits[z])
    return scale, zero


def run_npu(args: argparse.Namespace, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    import onnxruntime as ort
    from NeublaDriver import NeublaDriver

    front_path, back_path = split_model(Path(args.model))
    log("npu", "front/back 세션 생성 중...")
    front_sess = ort.InferenceSession(str(front_path), providers=["CPUExecutionProvider"])
    back_sess = ort.InferenceSession(str(back_path), providers=["CPUExecutionProvider"])
    scale, zero = load_dequant_params(Path(args.model))

    driver = NeublaDriver()
    if driver.Init() != 0:
        die("NeublaDriver.Init() 실패", "보드 리셋이 필요할 수 있습니다 — docs/why-hostreset-flow.md")
    log("npu", "드라이버 초기화 완료")

    # Close() 를 빠뜨리면 다음 실행의 LoadModel 이 DMA 에러로 실패합니다.
    # 예외가 나든 말든 반드시 불리도록 try/finally 로 감쌉니다.
    try:
        return _run_npu_inner(args, image, driver, front_sess, back_sess, scale, zero)
    finally:
        driver.Close()


def _run_npu_inner(args, image, driver, front_sess, back_sess, scale, zero):
    if driver.LoadModel(str(args.binary)) != 0:
        die("LoadModel 실패",
            "이전 실행이 Close() 없이 끝났을 수 있습니다. 한 번 더 시도해 보고, "
            "계속 실패하면 sudo ~/workdir/hostreset.sh")
    log("npu", f"모델 적재 완료 ({Path(args.binary).name})")

    # ── 1단: front (CPU) — 정규화된 float 를 uint8 로 양자화
    q = front_sess.run(None, {FRONT_INPUT_NAMES[0]: image})[0].tobytes()

    # ── 2단: NPU
    #
    # Launch() 는 비동기입니다. Wait() 로 완료를 기다린 뒤에 ReceiveOutputs() 를
    # 불러야 합니다. 이걸 빼면 이전 실행의(또는 초기화되지 않은) 버퍼가 돌아와
    # 검출 100개가 전부 score 1.00 인 쓰레기값이 나옵니다.
    #
    # 원본 데모의 비디오 루프는 Wait() 없이 도는데, 그건 시작 전에 warmup 으로
    # SendInput+Launch 를 한 번 더 해서 한 프레임 밀린 파이프라인으로 돌리기
    # 때문입니다. 단일 이미지에서는 Wait() 가 맞습니다.
    t0 = time.perf_counter()
    if driver.SendInput(q, 3 * INPUT_SIZE[0] * INPUT_SIZE[1]) != 0:
        die("SendInput 실패")
    if driver.Launch() != 0:
        die("Launch 실패")
    if driver.Wait() != 0:
        die("Wait 실패", "보드가 응답하지 않습니다 — sudo ~/workdir/hostreset.sh")
    raw = driver.ReceiveOutputs()
    npu_dt = time.perf_counter() - t0
    log("npu", f"NPU 실행 {npu_dt * 1000:.1f} ms  (출력 {len(raw)}개 버퍼)")

    # ── 3단: 역양자화 + back (CPU)
    out_f32 = [
        (np.frombuffer(buf, dtype=np.uint8).astype(np.float32) - zero[name]) * scale[name]
        for name, buf in zip(TARGET_DEQUANT_LAYER_NAMES, raw)
    ]
    feeds = {
        BACK_INPUT_NAMES[i]: out_f32[i][: int(np.prod(YOLO_HEAD_SPECS[i]))].reshape(1, *YOLO_HEAD_SPECS[i])
        for i in range(3)
    }
    dets, labels = back_sess.run(BACK_OUTPUT_NAMES, feeds)
    return dets, labels, npu_dt


# ─────────────────────────────────────────────── main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("check", "cpu", "npu", "both"), default="npu")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--binary", default=str(DEFAULT_BINARY))
    ap.add_argument("--image", default=str(DEFAULT_IMAGE))
    ap.add_argument("--out", default="result.jpg")
    ap.add_argument("--conf", type=float, default=0.5)
    args = ap.parse_args()

    check_env(args)
    if args.mode == "check":
        return

    import cv2

    bgr = cv2.imread(args.image)
    if bgr is None:
        die(f"이미지를 읽을 수 없습니다: {args.image}")
    image, img_wh = preprocess(bgr)
    log("input", f"{Path(args.image).name}  {img_wh[0]}x{img_wh[1]} → {INPUT_SIZE[0]}x{INPUT_SIZE[1]}")

    classes = load_classes()
    results = {}

    if args.mode in ("cpu", "both"):
        results["cpu"] = run_cpu(args, image)
    if args.mode in ("npu", "both"):
        results["npu"] = run_npu(args, image)

    print()
    for tag, (dets, labels, dt) in results.items():
        rendered, n = draw(bgr, dets, labels, img_wh, classes, args.conf)
        out = Path(args.out) if len(results) == 1 else Path(args.out).with_suffix(f".{tag}.jpg")
        cv2.imwrite(str(out), rendered)
        log(tag, f"검출 {n}개 (conf ≥ {args.conf}) · {dt * 1000:.1f} ms → {out}")

    if len(results) == 2:
        (cd, cl, ct), (nd, nl, nt) = results["cpu"], results["npu"]
        same = int((cl[0] == nl[0]).sum())
        log("compare", f"라벨 일치 {same}/{cl.shape[1]} · 속도 CPU {ct * 1000:.0f} ms vs NPU {nt * 1000:.1f} ms")


if __name__ == "__main__":
    main()
