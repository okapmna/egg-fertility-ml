"""Web demo deteksi + klasifikasi fertilitas telur.

Pipeline:
    Gambar -> YOLO11n (deteksi telur) -> crop per box (batch) -> Classifier pilihan (fertile/infertile)

Fitur Utama:
    - Multi-file upload (batch processing seperti Streamlit)
    - Batch inference pada MobileNetV2 (jauh lebih cepat dan responsif)
    - Pemilihan model dinamis dari folder model/
    - Perhitungan metrik lengkap (KPI cards, fertility rate %)
    - Tampilan side-by-side (Original vs Annotated)

Cara pakai:
    source /home/pmna/Documents/APP/egg-fertil-ml/venv/bin/activate
    python app.py
    # buka http://127.0.0.1:5000
"""

import base64
import io
import os
import time
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
CROP_MARGIN_DEFAULT = float(os.getenv("CROP_MARGIN", "0.10"))

TARGET_CLASSES = ["fertile", "infertile"]
COLORS = {"fertile": (34, 197, 94), "infertile": (239, 68, 68)}  # Hijau & Merah modern
MAX_UPLOAD_MB = 25

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
    """Kembalikan list nama file .pt di MODEL_DIR yang merupakan classifier."""
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
        _det_error = f"Dependensi belum terinstal ({e}). Aktifkan venv dan jalankan: pip install -r requirements-web.txt"
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
            f"Dependensi PyTorch belum terinstal ({e}). Pastikan venv aktif."
        ) from e

    cls_path = MODEL_DIR / cls_filename
    if not cls_path.exists():
        raise RuntimeError(f"File classifier tidak ditemukan: {cls_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _cls_global_device = device

    ckpt = torch.load(str(cls_path), map_location=device, weights_only=False)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    # Bersihkan prefix 'module.' jika disimpan dari DDP
    state_dict = {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }

    num_classes = len(TARGET_CLASSES)
    if isinstance(ckpt, dict) and "classes" in ckpt and isinstance(ckpt["classes"], (list, tuple)):
        num_classes = len(ckpt["classes"])

    clf = models.mobilenet_v2(weights=None)
    in_feat = clf.classifier[1].in_features

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
        if isinstance(clf.classifier[1], nn.Sequential):
            clf.classifier[1] = nn.Linear(in_feat, num_classes)
        else:
            clf.classifier[1] = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(in_feat, num_classes),
            )
        clf.load_state_dict(state_dict, strict=False)

    clf.eval().to(device)

    # Ambil img_size dari checkpoint bila tersedia
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


# ---------------------------------------------------------------- inferensi optimal
def run_pipeline(
    image: Image.Image,
    det_conf: float,
    thr_fertile: float,
    cls_filename: str,
    crop_margin: float = CROP_MARGIN_DEFAULT,
):
    """Jalankan detector -> crop -> classifier dengan optimasi Batch Inference."""
    import torch

    t_start = time.time()
    _ensure_detector()
    clf_entry = _load_classifier(cls_filename)

    W, H = image.size

    # 1. Deteksi Telur dengan YOLO
    t_det0 = time.time()
    det_res = _detector.predict(
        source=image, conf=det_conf, iou=0.6, imgsz=DET_IMGSZ, verbose=False
    )[0]
    det_time_ms = round((time.time() - t_det0) * 1000, 1)

    boxes = (
        det_res.boxes.xyxy.cpu().numpy()
        if det_res.boxes is not None
        else []
    )

    clf_model = clf_entry["model"]
    clf_tf = clf_entry["tf"]
    clf_device = clf_entry["device"]

    crop_tensors = []
    valid_boxes = []

    # 2. Persiapan Potongan Gambar (Crop dengan margin)
    for box in boxes:
        x0, y0, x1, y1 = (max(0.0, float(v)) for v in box)
        x1, y1 = min(float(W), x1), min(float(H), y1)
        if x1 <= x0 or y1 <= y0:
            continue

        bw, bh = x1 - x0, y1 - y0
        pad_x = (bw * crop_margin) / 2.0
        pad_y = (bh * crop_margin) / 2.0
        cx0 = max(0.0, x0 - pad_x)
        cy0 = max(0.0, y0 - pad_y)
        cx1 = min(float(W), x1 + pad_x)
        cy1 = min(float(H), y1 + pad_y)

        crop = image.crop((cx0, cy0, cx1, cy1))
        crop_tensors.append(clf_tf(crop))
        valid_boxes.append([round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)])

    detections = []
    cls_time_ms = 0.0

    # 3. Batch Forward Pass ke Classifier (Satu kali eksekusi untuk semua telur)
    if crop_tensors:
        t_cls0 = time.time()
        with torch.no_grad():
            batch_x = torch.stack(crop_tensors).to(clf_device)
            probs = torch.softmax(clf_model(batch_x), dim=1).cpu().numpy()
        cls_time_ms = round((time.time() - t_cls0) * 1000, 1)

        for box, prob in zip(valid_boxes, probs):
            cls_id = 0 if prob[0] >= thr_fertile else 1
            detections.append({
                "box": box,
                "cls": TARGET_CLASSES[cls_id],
                "cls_id": cls_id,
                "prob": round(float(prob[cls_id]), 4),
            })

    total_time_ms = round((time.time() - t_start) * 1000, 1)

    timing = {
        "det_ms": det_time_ms,
        "cls_ms": cls_time_ms,
        "total_ms": total_time_ms,
    }
    return detections, timing


def annotate_image(image: Image.Image, detections):
    """Gambar box + label ke salinan gambar dengan visual modern."""
    out = image.copy()
    draw = ImageDraw.Draw(out)
    for d in detections:
        x0, y0, x1, y1 = d["box"]
        color = COLORS.get(d["cls"], (255, 255, 0))
        # Kotak deteksi
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        # Label badge
        label = f"{d['cls'].upper()} {d['prob']:.2f}"
        tx0, ty0 = x0, max(0.0, y0 - 18)
        tx1, ty1 = x0 + len(label) * 8 + 8, y0
        draw.rectangle([tx0, ty0, tx1, ty1], fill=color)
        draw.text((tx0 + 4, ty0 + 2), label, fill=(255, 255, 255))
    return out


def _img_to_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
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
        crop_margin=round(CROP_MARGIN_DEFAULT * 100),
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
    """Health check: mengembalikan info status server, device, dan model."""
    try:
        import torch
        _ensure_detector()
        default_cls = _default_classifier_file()
        ok = True
        info = {
            "device": "CUDA (GPU)" if torch.cuda.is_available() else "CPU",
            "cuda": torch.cuda.is_available(),
            "detector": DETECTOR_FILE,
            "default_classifier": default_cls,
            "classifiers_available": _scan_classifier_files(),
        }
    except Exception as e:
        ok = False
        info = {"error": str(e)}
    return jsonify({"ok": ok, "models": info})


@app.post("/api/predict")
def predict():
    """Mendukung upload gambar tunggal maupun batch (multiple images) sekaligus."""
    files = request.files.getlist("images")
    if not files or all(f.filename == "" for f in files):
        if "image" in request.files and request.files["image"].filename != "":
            files = [request.files["image"]]
        else:
            return jsonify({"ok": False, "error": "Silakan pilih atau unggah gambar telur."}), 400

    try:
        det_conf = float(request.form.get("det_conf", DET_CONF_DEFAULT))
        thr_fertile = float(request.form.get("thr_fertile", THR_FERTILE_DEFAULT))
        margin_pct = float(request.form.get("crop_margin", CROP_MARGIN_DEFAULT * 100))
        crop_margin = margin_pct / 100.0
    except ValueError:
        return jsonify({"ok": False, "error": "Parameter numerik tidak valid."}), 400

    det_conf = min(max(det_conf, 0.01), 0.99)
    thr_fertile = min(max(thr_fertile, 0.01), 0.99)
    crop_margin = min(max(crop_margin, 0.0), 0.50)

    cls_filename = request.form.get("classifier_file", "").strip()
    available = _scan_classifier_files()
    if not cls_filename or cls_filename not in available:
        cls_filename = _default_classifier_file()
    if not cls_filename:
        return jsonify({"ok": False, "error": "Tidak ada file classifier (.pt) di folder model/."}), 503

    results = []
    agg_fertile = 0
    agg_infertile = 0
    agg_total = 0

    for file_obj in files:
        if not file_obj or file_obj.filename == "":
            continue
        try:
            image = Image.open(file_obj.stream).convert("RGB")
        except Exception:
            continue

        try:
            detections, timing = run_pipeline(
                image, det_conf, thr_fertile, cls_filename, crop_margin
            )
        except Exception as e:
            return jsonify({"ok": False, "error": f"Error saat inferensi: {e}"}), 503

        counts = {c: sum(1 for d in detections if d["cls"] == c) for c in TARGET_CLASSES}
        agg_fertile += counts["fertile"]
        agg_infertile += counts["infertile"]
        agg_total += len(detections)

        annotated_img = annotate_image(image, detections)

        results.append({
            "filename": file_obj.filename,
            "width": image.width,
            "height": image.height,
            "detections": detections,
            "counts": counts,
            "timing": timing,
            "original": _img_to_data_url(image),
            "annotated": _img_to_data_url(annotated_img),
        })

    if not results:
        return jsonify({"ok": False, "error": "Tidak ada file gambar valid yang dapat diproses."}), 400

    fertility_rate = round((agg_fertile / max(agg_total, 1)) * 100, 1)

    # Mengembalikan format batch yang kaya, tetap kompatibel jika cuma 1 gambar
    first = results[0]
    return jsonify({
        "ok": True,
        "classifier_used": cls_filename,
        "summary": {
            "total_images": len(results),
            "total_detected": agg_total,
            "fertile": agg_fertile,
            "infertile": agg_infertile,
            "fertility_rate_pct": fertility_rate,
        },
        "results": results,
        # Field backward-compatible untuk 1 gambar
        "detections": first["detections"],
        "counts": first["counts"],
        "annotated": first["annotated"],
        "timing": first["timing"],
    })


# --------------------------------------------------------- global error handler
@app.errorhandler(Exception)
def handle_exception(e):
    """Tangkap semua error dan kembalikan JSON bukan HTML error 500."""
    import traceback
    return jsonify({
        "ok": False,
        "error": f"{type(e).__name__}: {e}",
        "detail": traceback.format_exc(),
    }), 500


@app.errorhandler(404)
def handle_404(e):
    return jsonify({"ok": False, "error": "Endpoint tidak ditemukan."}), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
