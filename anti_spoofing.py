import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence, Union

import cv2
import numpy as np


@dataclass(frozen=True)
class AntiSpoofResult:
    ready: bool
    is_real: bool
    is_spoof: bool
    spoof_score: float
    threshold: float
    label: str
    error: Optional[str] = None


class AntiSpoofClassifier:
    def __init__(
        self,
        model_dir: Union[os.PathLike, str],
        threshold: Optional[float] = None,
        interpreter_factory: Optional[Callable[[str], object]] = None,
    ):
        self.model_dir = Path(model_dir)
        self.model_path = self.model_dir / "best.tflite"
        self.metadata_path = self.model_dir / "metadata.json"
        self.ready = False
        self.error: Optional[str] = None
        self.threshold = 0.5
        self.input_size = (224, 224)
        self.crop_margin = 0.25
        self._interpreter = None
        self._input_index = None
        self._output_index = None
        self._input_dtype = np.float32

        try:
            self._load_metadata()
            self.threshold = self._resolve_threshold(threshold)
            if not self.model_path.exists():
                raise FileNotFoundError(f"Anti-spoof model not found: {self.model_path}")

            factory = interpreter_factory or self._default_interpreter_factory()
            self._interpreter = factory(str(self.model_path))
            self._interpreter.allocate_tensors()
            input_details = self._interpreter.get_input_details()[0]
            output_details = self._interpreter.get_output_details()[0]
            self._input_index = input_details["index"]
            self._output_index = output_details["index"]
            self._input_dtype = np.dtype(input_details.get("dtype", np.float32))
            self.ready = True
        except Exception as exc:
            self.error = str(exc)
            self.ready = False

    def _load_metadata(self) -> None:
        if not self.metadata_path.exists():
            return
        with self.metadata_path.open("r", encoding="utf-8") as fh:
            metadata = json.load(fh)

        value = metadata.get("recommended_threshold", metadata.get("default_threshold"))
        if value is not None:
            self.threshold = float(value)

        input_size = metadata.get("input_size")
        if isinstance(input_size, Sequence) and len(input_size) >= 2:
            self.input_size = (int(input_size[0]), int(input_size[1]))

        preprocessing = metadata.get("preprocessing_options", {})
        margin = preprocessing.get("face_crop_margin")
        if margin is not None:
            self.crop_margin = float(margin)

    def _resolve_threshold(self, threshold: Optional[float]) -> float:
        env_threshold = os.getenv("ANTI_SPOOF_THRESHOLD")
        if threshold is not None:
            return float(threshold)
        if env_threshold:
            return float(env_threshold)
        return float(self.threshold)

    @staticmethod
    def _default_interpreter_factory() -> Callable[[str], object]:
        try:
            import tensorflow as tf

            Interpreter = tf.lite.Interpreter
        except ImportError:
            from tflite_runtime.interpreter import Interpreter

        return lambda model_path: Interpreter(model_path=model_path)

    def status(self) -> dict:
        return {
            "ready": self.ready,
            "threshold": self.threshold,
            "model_dir": str(self.model_dir),
            "error": self.error,
        }

    def unavailable_result(self) -> AntiSpoofResult:
        return AntiSpoofResult(
            ready=False,
            is_real=False,
            is_spoof=True,
            spoof_score=1.0,
            threshold=self.threshold,
            label="spoof",
            error=self.error or "Anti-spoof classifier is unavailable",
        )

    def predict(self, frame: np.ndarray, bbox: Sequence[float]) -> AntiSpoofResult:
        if not self.ready or self._interpreter is None:
            return self.unavailable_result()

        try:
            crop = self._crop_face(frame, bbox)
            if crop is None or crop.size == 0:
                raise ValueError("Invalid face crop")

            tensor = self._preprocess(crop)
            self._interpreter.set_tensor(self._input_index, tensor)
            self._interpreter.invoke()
            output = self._interpreter.get_tensor(self._output_index)
            spoof_score = float(np.asarray(output, dtype=np.float32).reshape(-1)[0])
            is_spoof = spoof_score >= self.threshold
            return AntiSpoofResult(
                ready=True,
                is_real=not is_spoof,
                is_spoof=is_spoof,
                spoof_score=spoof_score,
                threshold=self.threshold,
                label="spoof" if is_spoof else "real",
            )
        except Exception as exc:
            return AntiSpoofResult(
                ready=True,
                is_real=False,
                is_spoof=True,
                spoof_score=1.0,
                threshold=self.threshold,
                label="spoof",
                error=str(exc),
            )

    def _crop_face(self, frame: np.ndarray, bbox: Sequence[float]) -> Optional[np.ndarray]:
        if frame is None or frame.ndim != 3 or len(bbox) < 4:
            return None

        height, width = frame.shape[:2]
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        box_w = x2 - x1
        box_h = y2 - y1
        if box_w <= 0 or box_h <= 0:
            return None

        pad_x = box_w * self.crop_margin
        pad_y = box_h * self.crop_margin
        left = max(0, int(np.floor(x1 - pad_x)))
        top = max(0, int(np.floor(y1 - pad_y)))
        right = min(width, int(np.ceil(x2 + pad_x)))
        bottom = min(height, int(np.ceil(y2 + pad_y)))
        if right <= left or bottom <= top:
            return None
        return frame[top:bottom, left:right]

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        target_w, target_h = self.input_size[0], self.input_size[1]
        resized = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = np.expand_dims(tensor, axis=0)
        return tensor.astype(self._input_dtype, copy=False)
