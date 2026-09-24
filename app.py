"""Web demo deteksi + klasifikasi fertilitas telur.

Pipeline (sama seperti notebook v3):
    Gambar -> YOLO11n (deteksi telur) -> crop per box -> MobileNetV2 (fertile/infertile)

Cara pakai:
    pip install -r requirements-web.txt
    python app.py
    # buka http://127.0.0.1:5000

Model dibaca dari directory `model/` (bisa diubah via env MODEL_DIR):
    model/yolo11n.pt            -> detector YOLO (ultralytics)
    model/mobilenet_egg_best.pt -> classifier MobileNetV2 (state_dict, input 320x320)

Env opsional:
    MODEL_DIR, DETECTOR_FILE, CLASSIFIER_FILE,
    DET_IMGSZ (default 640), CLS_IMGSZ (default 320),
    DET_CONF (default 0.25), THR_FERTILE (default 0.5), PORT (default 5000)
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
CLASSIFIER_FILE = os.getenv("CLASSIFIER_FILE", "mobilenet_egg_best.pt")

DET_IMGSZ = int(os.getenv("DET_IMGSZ", "640"))
CLS_IMGSZ = int(os.getenv("CLS_IMGSZ", "320"))
DET_CONF_DEFAULT = float(os.getenv("DET_CONF", "0.25"))
THR_FERTILE_DEFAULT = float(os.getenv("THR_FERTILE", "0.5"))

TARGET_CLASSES = ["fertile", "infertile"]
COLORS = {"fertile": (0, 200, 0), "infertile": (220, 30, 30)}
MAX_UPLOAD_MB = 10

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

# Cache model setelah pertama kali dimuat.
_detector = None
_classifier = None
_cls_device = None
_model_error = None


# ------------------------------------------------------------ loading model
def _load_models():
    """Load detector + classifier secara lazy. Import torch/ultralytics di sini
    agar halaman error-nya jelas bila dependensi / file model belum ada."""
    global _detector, _classifier, _cls_device, _model_error
    if _detector is not None and _classifier is not None:
        return
    _model_error = None
    try:
        import torch
        from ultralytics import YOLO
        from torchvision import models, transforms
    except ImportError as e:
        _model_error = (
            f"Dependensi belum terinstal ({e}). "
            "Jalankan: pip install -r requirements-web.txt"
        )
        raise RuntimeError(_model_error) from e

    det_path = MODEL_DIR / DETECTOR_FILE
    cls_path = MODEL_DIR / CLASSIFIER_FILE
    missing = [str(p) for p in (det_path, cls_path) if not p.exists()]
    if missing:
        _model_error = f"File model tidak ditemukan: {', '.join(missing)}"
        raise RuntimeError(_model_error)

    _cls_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _detector = YOLO(str(det_path))

    ckpt = torch.load(str(cls_path), map_location=_cls_device, weights_only=False)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    # Bersihkan prefix 'module.' jika disimpan dari DistributedDataParallel / DataParallel
    state_dict = {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }

    import torch.nn as nn

    clf = models.mobilenet_v2(weights=None)
    in_feat = clf.classifier[1].in_features
    num_classes = len(TARGET_CLASSES)
    if isinstance(ckpt, dict) and "classes" in ckpt and isinstance(ckpt["classes"], (list, tuple)):
        num_classes = len(ckpt["classes"])

    # Model head bisa berupa:
    # 1) Sequential(Dropout, Linear) -> state_dict memiliki key 'classifier.1.1.weight' (dari train_mobilenet_classifier.ipynb)
    # 2) Linear langsung -> state_dict memiliki key 'classifier.1.weight' (dari v3 simple pipeline)
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

    clf.eval().to(_cls_device)
    _classifier = clf

    # Gunakan img_size dari checkpoint bila tersedia, fallback ke CLS_IMGSZ
    img_sz = CLS_IMGSZ
    if isinstance(ckpt, dict) and "img_size" in ckpt:
        try:
            img_sz = int(ckpt["img_size"])
        except (ValueError, TypeError):
            pass

    _tf = transforms.Compose(
        [
            transforms.Resize((img_sz, img_sz)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    _load_models.tf = _tf  # type: ignore[attr-defined]


# ---------------------------------------------------------------- inferensi
def run_pipeline(image: Image.Image, det_conf: float, thr_fertile: float):
    """Jalankan detector -> crop -> classifier. Kembalikan list deteksi."""
    _load_models()  # raise RuntimeError (pesan jelas) bila deps/model belum siap
    import torch

    W, H = image.size

    det_res = _detector.predict(
        source=image, conf=det_conf, iou=0.6, imgsz=DET_IMGSZ, verbose=False
    )[0]
    boxes = (
        det_res.boxes.xyxy.cpu().numpy()
        if det_res.boxes is not None
        else []
    )

    detections = []
    with torch.no_grad():
        for box in boxes:
            x0, y0, x1, y1 = (max(0.0, float(v)) for v in box)
            x1, y1 = min(float(W), x1), min(float(H), y1)
            if x1 <= x0 or y1 <= y0:
                continue
            crop = image.crop((x0, y0, x1, y1))
            x = _load_models.tf(crop).unsqueeze(0).to(_cls_device)  # type: ignore[attr-defined]
            prob = torch.softmax(_classifier(x), dim=1).cpu().numpy()[0]
            cls_id = 0 if prob[0] >= thr_fertile else 1
            detections.append(
                {
                    "box": [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)],
                    "cls": TARGET_CLASSES[cls_id],
                    "cls_id": cls_id,
                    "prob": round(float(prob[cls_id]), 4),
                }
            )
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
    return render_template(
        "index.html",
        det_conf=DET_CONF_DEFAULT,
        thr_fertile=THR_FERTILE_DEFAULT,
    )


@app.get("/api/health")
def health():
    try:
        _load_models()
        import torch

        ok = True
        info = {
            "device": str(_cls_device),
            "cuda": torch.cuda.is_available(),
            "detector": str(MODEL_DIR / DETECTOR_FILE),
            "classifier": str(MODEL_DIR / CLASSIFIER_FILE),
        }
    except RuntimeError as e:
        ok = False
        info = {"error": str(e)}
    return jsonify({"ok": ok, "models": info})


@app.post("/api/predict")
def predict():
    if "image" not in request.files:
        return jsonify({"ok": False, "error": "Tidak ada file 'image'."}), 400
    try:
        det_conf = float(request.form.get("det_conf", DET_CONF_DEFAULT))
        thr_fertile = float(request.form.get("thr_fertile", THR_FERTILE_DEFAULT))
    except ValueError:
        return jsonify({"ok": False, "error": "det_conf / thr_fertile harus angka."}), 400
    det_conf = min(max(det_conf, 0.01), 0.99)
    thr_fertile = min(max(thr_fertile, 0.01), 0.99)

    try:
        image = Image.open(request.files["image"].stream).convert("RGB")
    except Exception:
        return jsonify({"ok": False, "error": "File bukan gambar yang valid."}), 400

    try:
        detections = run_pipeline(image, det_conf, thr_fertile)
    except RuntimeError as e:
        return jsonify({"ok": False, "error": str(e)}), 503

    counts = {c: sum(1 for d in detections if d["cls"] == c) for c in TARGET_CLASSES}
    return jsonify(
        {
            "ok": True,
            "detections": detections,
            "counts": counts,
            "annotated": _img_to_data_url(annotate_image(image, detections)),
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
