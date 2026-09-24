"""Web demo deteksi + klasifikasi fertilitas telur.

Pipeline:
    Gambar -> YOLO11n (deteksi telur) -> crop per box -> Classifier pilihan (fertile/infertile)

Cara pakai:
    pip install -r requirements-web.txt
    python app.py
    # buka http://127.0.0.1:5000

Model dibaca dari directory `model/` (bisa diubah via env MODEL_DIR).
  Detector : file .pt yang dipilih via UI (default: yolo11n.pt)
  Classifier: semua file .pt di folder model/ otomatis terdeteksi sebagai pilihan

Env opsional:
    MODEL_DIR, DETECTOR_FILE,
    DET_IMGSZ (default 640), CLS_IMGSZ (default 320),
    DET_CONF (default 0.25), THR_FERTILE (default 0.5),
    CROP_MARGIN (default 0.10), PORT (default 5000)
"""

import base64
import io
import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageDraw

# ---------------------------------------------------------------- konfigurasi
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.getenv("MODEL_DIR", str(BASE_DIR / "model")))
DETECTOR_FILE = os.getenv("DETECTOR_FILE", "yolo11n.pt")

DET_IMGSZ = int(os.getenv("DET_IMGSZ", "640"))
CLS_IMGSZ = int(os.getenv("CLS_IMGSZ", "320"))
DET_CONF_DEFAULT = float(os.getenv("DET_CONF", "0.25"))
THR_FERTILE_DEFAULT = float(os.getenv("THR_FERTILE", "0.5"))
CROP_MARGIN = float(os.getenv("CROP_MARGIN", "0.10"))

TARGET_CLASSES = ["fertile", "infertile"]
COLORS = {"fertile": (0, 200, 0), "infertile": (220, 30, 30)}
MAX_UPLOAD_MB = 10

# File YOLO yang dikecualikan dari daftar classifier
_YOLO_KEYWORDS = {"yolo", "yolov", "yolo11", "yolov8", "yolov9", "yolov10"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

# Cache detector (dimuat sekali)
_detector = None
_det_error = None

# Cache classifier per nama file: {filename: {"model": ..., "tf": ..., "device": ...}}
_classifiers: dict = {}
_cls_global_device = None


# ------------------------------------------------- helper: scan model folder
def _scan_classifier_files() -> list[str]:
    """Kembalikan list nama file .pt di MODEL_DIR yang bukan YOLO."""
    if not MODEL_DIR.exists():
        return []
    files = []
    for f in sorted(MODEL_DIR.glob("*.pt")):
        name_lower = f.stem.lower()
        is_yolo = any(kw in name_lower for kw in _YOLO_KEYWORDS)
        if not is_yolo:
            files.append(f.name)
    return files


def _default_classifier_file() -> str:
    """Pilih classifier default: mobilenet_egg_best.pt jika ada, atau file .pt pertama."""
    files = _scan_classifier_files()
    for name in files:
        if "mobilenet_egg_best" in name:
            return name
    return files[0] if files else ""


# ------------------------------------------------- loading model
def _ensure_detector():
    """Muat YOLO detector sekali secara lazy."""
    global _detector, _det_error
    if _detector is not None:
        return
    _det_error = None
    try:
        from ultralytics import YOLO
    except ImportError as e:
        _det_error = f"Dependensi belum terinstal ({e}). Jalankan: pip install -r requirements-web.txt"
        raise RuntimeError(_det_error) from e

    det_path = MODEL_DIR / DETECTOR_FILE
    if not det_path.exists():
        _det_error = f"File detector tidak ditemukan: {det_path}"
        raise RuntimeError(_det_error)
    _detector = YOLO(str(det_path))


def _load_classifier(cls_filename: str):
    """Muat classifier berdasarkan nama file. Hasil di-cache per nama file."""
    global _cls_global_device

    if cls_filename in _classifiers:
        return _classifiers[cls_filename]

    try:
        import torch
        import torch.nn as nn
        from torchvision import models, transforms
    except ImportError as e:
        raise RuntimeError(
            f"Dependensi belum terinstal ({e}). Jalankan: pip install -r requirements-web.txt"
        ) from e

    cls_path = MODEL_DIR / cls_filename
    if not cls_path.exists():
        raise RuntimeError(f"File classifier tidak ditemukan: {cls_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _cls_global_device = device

    ckpt = torch.load(str(cls_path), map_location=device, weights_only=False)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    # Bersihkan prefix 'module.' jika disimpan dari DistributedDataParallel
    state_dict = {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }

    num_classes = len(TARGET_CLASSES)
    if isinstance(ckpt, dict) and "classes" in ckpt and isinstance(ckpt["classes"], (list, tuple)):
        num_classes = len(ckpt["classes"])

    clf = models.mobilenet_v2(weights=None)
    in_feat = clf.classifier[1].in_features

    # Deteksi arsitektur head: Sequential(Dropout, Linear) atau Linear langsung
    if "classifier.1.1.weight" in state_dict:
        clf.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_feat, num_classes),
        )
    else:
        clf.classifier[1] = nn.Linear(in_feat, num_classes)

    try:
        clf.load_state_dict(state_dict)
    except Exception:
        # Fallback jika arsitektur head terbalik
        if isinstance(clf.classifier[1], nn.Sequential):
            clf.classifier[1] = nn.Linear(in_feat, num_classes)
        else:
            clf.classifier[1] = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(in_feat, num_classes),
            )
        clf.load_state_dict(state_dict, strict=False)

    clf.eval().to(device)

    # Ambil img_size dari checkpoint bila tersedia, fallback ke CLS_IMGSZ
    img_sz = CLS_IMGSZ
    if isinstance(ckpt, dict) and "img_size" in ckpt:
        try:
            img_sz = int(ckpt["img_size"])
        except (ValueError, TypeError):
            pass

    tf = transforms.Compose([
        transforms.Resize((img_sz, img_sz)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    entry = {"model": clf, "tf": tf, "device": device, "img_size": img_sz, "file": cls_filename}
    _classifiers[cls_filename] = entry
    return entry


# ---------------------------------------------------------------- inferensi
def run_pipeline(
    image: Image.Image,
    det_conf: float,
    thr_fertile: float,
    cls_filename: str,
):
    """Jalankan detector -> crop -> classifier yang dipilih. Kembalikan list deteksi."""
    import torch

    _ensure_detector()
    clf_entry = _load_classifier(cls_filename)

    W, H = image.size

    det_res = _detector.predict(
        source=image, conf=det_conf, iou=0.6, imgsz=DET_IMGSZ, verbose=False
    )[0]
    boxes = (
        det_res.boxes.xyxy.cpu().numpy()
        if det_res.boxes is not None
        else []
    )

    clf_model = clf_entry["model"]
    clf_tf = clf_entry["tf"]
    clf_device = clf_entry["device"]

    detections = []
    with torch.no_grad():
        for box in boxes:
            x0, y0, x1, y1 = (max(0.0, float(v)) for v in box)
            x1, y1 = min(float(W), x1), min(float(H), y1)
            if x1 <= x0 or y1 <= y0:
                continue

            # Crop dengan margin agar sesuai dengan distribusi data saat training
            bw, bh = x1 - x0, y1 - y0
            pad_x = (bw * CROP_MARGIN) / 2.0
            pad_y = (bh * CROP_MARGIN) / 2.0
            cx0 = max(0.0, x0 - pad_x)
            cy0 = max(0.0, y0 - pad_y)
            cx1 = min(float(W), x1 + pad_x)
            cy1 = min(float(H), y1 + pad_y)

            crop = image.crop((cx0, cy0, cx1, cy1))
            x = clf_tf(crop).unsqueeze(0).to(clf_device)
            prob = torch.softmax(clf_model(x), dim=1).cpu().numpy()[0]
            cls_id = 0 if prob[0] >= thr_fertile else 1
            detections.append({
                "box": [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)],
                "cls": TARGET_CLASSES[cls_id],
                "cls_id": cls_id,
                "prob": round(float(prob[cls_id]), 4),
            })
    return detections


def annotate_image(image: Image.Image, detections):
    """Gambar box + label ke salinan gambar. Fungsi murni (bisa dites tanpa model)."""
    out = image.copy()
    draw = ImageDraw.Draw(out)
    for d in detections:
        x0, y0, x1, y1 = d["box"]
        color = COLORS.get(d["cls"], (255, 255, 0))
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        label = f"{d['cls']} {d['prob']:.2f}"
        tx0, ty0 = x0, max(0, y0 - 16)
        tx1, ty1 = x0 + len(label) * 8 + 6, y0
        draw.rectangle([tx0, ty0, tx1, ty1], fill=color)
        draw.text((tx0 + 3, ty0 + 1), label, fill=(255, 255, 255))
    return out


def _img_to_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# ------------------------------------------------------------------- routes
@app.get("/")
def index():
    classifier_files = _scan_classifier_files()
    default_cls = _default_classifier_file()
    return render_template(
        "index.html",
        det_conf=DET_CONF_DEFAULT,
        thr_fertile=THR_FERTILE_DEFAULT,
        classifier_files=classifier_files,
        default_classifier=default_cls,
    )


@app.get("/api/models")
def list_models():
    """Kembalikan daftar classifier yang tersedia dan info detector."""
    classifier_files = _scan_classifier_files()
    return jsonify({
        "detector": DETECTOR_FILE,
        "classifiers": classifier_files,
        "default_classifier": _default_classifier_file(),
    })


@app.get("/api/health")
def health():
    try:
        import torch
        _ensure_detector()
        default_cls = _default_classifier_file()
        clf_entry = _load_classifier(default_cls) if default_cls else None
        ok = True
        info = {
            "device": str(_cls_global_device or ("cuda" if torch.cuda.is_available() else "cpu")),
            "cuda": torch.cuda.is_available(),
            "detector": str(MODEL_DIR / DETECTOR_FILE),
            "classifier_default": default_cls,
            "classifiers_available": _scan_classifier_files(),
            "img_size": clf_entry["img_size"] if clf_entry else CLS_IMGSZ,
        }
    except RuntimeError as e:
        ok = False
        info = {"error": str(e)}
    return jsonify({"ok": ok, "models": info})


@app.post("/api/predict")
def predict():
    if "image" not in request.files:
        return jsonify({"ok": False, "error": "Tidak ada file 'image'."}), 400

    # Ambil parameter
    try:
        det_conf = float(request.form.get("det_conf", DET_CONF_DEFAULT))
        thr_fertile = float(request.form.get("thr_fertile", THR_FERTILE_DEFAULT))
    except ValueError:
        return jsonify({"ok": False, "error": "det_conf / thr_fertile harus angka."}), 400

    det_conf = min(max(det_conf, 0.01), 0.99)
    thr_fertile = min(max(thr_fertile, 0.01), 0.99)

    # Ambil nama classifier yang dipilih user (fallback ke default)
    cls_filename = request.form.get("classifier_file", "").strip()
    available = _scan_classifier_files()
    if not cls_filename or cls_filename not in available:
        cls_filename = _default_classifier_file()
    if not cls_filename:
        return jsonify({"ok": False, "error": "Tidak ada file classifier (.pt) di folder model/."}), 503

    try:
        image = Image.open(request.files["image"].stream).convert("RGB")
    except Exception:
        return jsonify({"ok": False, "error": "File bukan gambar yang valid."}), 400

    try:
        detections = run_pipeline(image, det_conf, thr_fertile, cls_filename)
    except RuntimeError as e:
        return jsonify({"ok": False, "error": str(e)}), 503

    counts = {c: sum(1 for d in detections if d["cls"] == c) for c in TARGET_CLASSES}
    return jsonify({
        "ok": True,
        "classifier_used": cls_filename,
        "detections": detections,
        "counts": counts,
        "annotated": _img_to_data_url(annotate_image(image, detections)),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
