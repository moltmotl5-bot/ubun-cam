"""YOLO11n detection. Ultralytics is imported only when a model is actually loaded."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from ubuncam.pipeline import Detection, keep_classes


def ensure_onnx(source_weights: Path, onnx_path: Path, imgsz: int) -> Path:
    """Return the ONNX file, exporting YOLO11n from the .pt weights if needed."""
    if onnx_path.is_file():
        return onnx_path
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    source_weights.parent.mkdir(parents=True, exist_ok=True)
    from ultralytics import YOLO

    model = YOLO(str(source_weights))
    exported = Path(model.export(format="onnx", imgsz=imgsz, device="cpu"))
    exported = exported.resolve()
    target = onnx_path.resolve()
    if exported != target:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(exported), str(target))
    return target


class Detector:
    def __init__(self, weights: Path, classes: tuple[str, ...], confidence: float, iou: float, imgsz: int):
        from ultralytics import YOLO

        self.classes = classes
        self.confidence = confidence
        self.iou = iou
        self.imgsz = imgsz
        self.model = YOLO(str(weights), task="detect")
        names = getattr(self.model, "names", None) or {}
        if not names:
            names = {2: "car", 5: "bus", 7: "truck"}
        by_name = {str(name): int(class_id) for class_id, name in dict(names).items()}
        missing = [name for name in classes if name not in by_name]
        if missing:
            raise RuntimeError(
                f"Model {weights} has no class(es) {missing}. "
                "Use the COCO YOLO11n weights (car, bus, truck)."
            )
        self._ids = [by_name[name] for name in classes]
        self._names = {int(class_id): str(name) for class_id, name in dict(names).items()}

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self.model.predict(
            frame,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.imgsz,
            classes=self._ids,
            device="cpu",
            verbose=False,
        )
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = _numpy(boxes.xyxy)
        confs = _numpy(boxes.conf)
        class_ids = _numpy(boxes.cls).astype(int)
        found = []
        for box, conf, class_id in zip(xyxy, confs, class_ids):
            name = self._names.get(int(class_id))
            if name is None:
                continue
            found.append(
                Detection(
                    xyxy=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                    class_name=name,
                    confidence=float(conf),
                )
            )
        return keep_classes(found, self.classes)


def _numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)
