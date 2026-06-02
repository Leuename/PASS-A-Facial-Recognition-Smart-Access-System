from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_NAME = "best.tflite"
DEFAULT_MODEL_DIR = BASE_DIR / "anti_spoof_model"
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / DEFAULT_MODEL_NAME
DEFAULT_THRESHOLD = 0.35
DEFAULT_NUM_THREADS = max(1, int(os.getenv("ARCLIGHT_ANTI_SPOOF_THREADS", "2")))


class AntiSpoofUnavailable(RuntimeError):
    pass


def _load_cv2():
    try:
        import cv2

        return cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing Python dependency 'cv2'. Install OpenCV before using anti-spoofing.") from exc


def _load_numpy():
    try:
        import numpy

        return numpy
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing Python dependency 'numpy'. Install NumPy before using anti-spoofing.") from exc


@dataclass(frozen=True)
class AntiSpoofResult:
    spoof_score: float
    threshold: float

    @property
    def is_real(self) -> bool:
        return self.spoof_score < self.threshold

    @property
    def label(self) -> str:
        return "real" if self.is_real else "spoof"


@dataclass(frozen=True)
class LivenessDecision:
    is_live: bool
    label: str
    reason: str
    spoof_score: Optional[float] = None


def parse_bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def resolve_model_path(model_path: Optional[str | os.PathLike[str]] = None) -> Path:
    if model_path:
        return Path(model_path)

    env_path = os.getenv("ARCLIGHT_ANTI_SPOOF_MODEL")
    if env_path:
        return Path(env_path)

    return DEFAULT_MODEL_PATH


def load_interpreter_class():
    try:
        from tflite_runtime.interpreter import Interpreter

        return Interpreter
    except Exception as tflite_error:
        try:
            import tensorflow as tf

            return tf.lite.Interpreter
        except Exception as tensorflow_error:
            raise AntiSpoofUnavailable(
                "Anti-spoofing needs tflite-runtime on Raspberry Pi or TensorFlow Lite. "
                f"tflite_runtime error: {tflite_error}; tensorflow error: {tensorflow_error}"
            ) from tensorflow_error


def create_interpreter(interpreter_class, model_path: str, num_threads: int):
    try:
        return interpreter_class(model_path=model_path, num_threads=max(1, int(num_threads)))
    except TypeError:
        return interpreter_class(model_path)


def preprocess_face_crop(
    frame,
    bbox: Sequence[float],
    input_size: tuple[int, int] = (224, 224),
    margin: float = 0.2,
):
    cv2 = _load_cv2()
    np = _load_numpy()

    if frame is None or frame.size == 0:
        raise ValueError("frame is empty")

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    box_w = max(0.0, x2 - x1)
    box_h = max(0.0, y2 - y1)
    if box_w <= 0 or box_h <= 0:
        raise ValueError(f"invalid face bounding box: {bbox}")

    pad_x = box_w * margin
    pad_y = box_h * margin
    x1 = max(0, int(round(x1 - pad_x)))
    y1 = max(0, int(round(y1 - pad_y)))
    x2 = min(w, int(round(x2 + pad_x)))
    y2 = min(h, int(round(y2 + pad_y)))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"face bounding box is outside the frame: {bbox}")

    crop = frame[y1:y2, x1:x2]
    resized = cv2.resize(crop, input_size, interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    return np.expand_dims(rgb, axis=0).astype(np.float32)


def spoof_score_from_output(output) -> float:
    np = _load_numpy()

    values = np.asarray(output, dtype=np.float32).reshape(-1)
    if values.size == 0:
        raise ValueError("anti-spoofing model returned an empty output")

    if values.size == 1:
        score = float(values[0])
    else:
        score = float(values[1])

    return float(np.clip(score, 0.0, 1.0))


class AntiSpoofClassifier:
    def __init__(
        self,
        model_path: Optional[str | os.PathLike[str]] = None,
        threshold: float = DEFAULT_THRESHOLD,
        interpreter_factory: Optional[Callable[[str], object]] = None,
        debug_crop_dir: Optional[str | os.PathLike[str]] = None,
        debug_crop_limit: int = 20,
        num_threads: int = DEFAULT_NUM_THREADS,
    ):
        self.model_path = resolve_model_path(model_path)
        self.threshold = float(threshold)
        self.debug_crop_dir = Path(debug_crop_dir) if debug_crop_dir else None
        self.debug_crop_limit = max(0, int(debug_crop_limit))
        self.debug_crop_count = 0
        self.num_threads = max(1, int(num_threads))
        if self.debug_crop_dir is not None:
            self.debug_crop_dir.mkdir(parents=True, exist_ok=True)

        if interpreter_factory is None:
            if not self.model_path.exists():
                raise FileNotFoundError(f"anti-spoofing model not found: {self.model_path}")
            interpreter_class = load_interpreter_class()
            self.interpreter = create_interpreter(interpreter_class, str(self.model_path), self.num_threads)
        else:
            self.interpreter = interpreter_factory(str(self.model_path))

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_index = self.input_details[0]["index"]
        self.output_index = self.output_details[0]["index"]
        self.input_size = self._read_input_size(self.input_details[0])
        np = _load_numpy()
        self.input_dtype = self.input_details[0].get("dtype", np.float32)

    def predict(self, frame, bbox: Sequence[float]) -> AntiSpoofResult:
        np = _load_numpy()
        tensor = preprocess_face_crop(frame, bbox, input_size=self.input_size)
        self._maybe_save_debug_crop(tensor)
        if self.input_dtype != np.float32:
            tensor = tensor.astype(self.input_dtype)

        self.interpreter.set_tensor(self.input_index, tensor)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_index)
        return AntiSpoofResult(
            spoof_score=spoof_score_from_output(output),
            threshold=self.threshold,
        )

    def _maybe_save_debug_crop(self, tensor) -> None:
        if self.debug_crop_dir is None or self.debug_crop_count >= self.debug_crop_limit:
            return

        cv2 = _load_cv2()
        np = _load_numpy()

        self.debug_crop_count += 1
        rgb = np.clip(tensor[0], 0, 255).astype(np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        output_path = self.debug_crop_dir / f"anti_spoof_crop_{self.debug_crop_count:04d}.png"
        if not cv2.imwrite(str(output_path), bgr):
            raise RuntimeError(f"failed to write anti-spoof debug crop: {output_path}")

    @staticmethod
    def _read_input_size(input_detail: dict) -> tuple[int, int]:
        shape = input_detail.get("shape")
        if shape is None or len(shape) < 4:
            return (224, 224)

        height = int(shape[1])
        width = int(shape[2])
        if height <= 0 or width <= 0:
            return (224, 224)
        return (width, height)


def evaluate_liveness(
    frame,
    bbox: Sequence[float],
    anti_spoof_classifier: Optional[AntiSpoofClassifier],
    enabled: bool = True,
    fail_closed: bool = True,
    unavailable_reason: Optional[str] = None,
    show_score: bool = False,
) -> LivenessDecision:
    if not enabled:
        return LivenessDecision(True, "Anti-spoof disabled", "anti_spoof_disabled")

    if anti_spoof_classifier is None:
        if fail_closed:
            reason = unavailable_reason or "classifier not loaded"
            return LivenessDecision(
                False,
                f"Anti-spoof unavailable: {reason}",
                "anti_spoof_unavailable",
            )
        return LivenessDecision(True, "Anti-spoof unavailable", "anti_spoof_unavailable")

    try:
        result = anti_spoof_classifier.predict(frame, bbox)
    except Exception as exc:
        return LivenessDecision(False, f"Spoof error: {exc}", "anti_spoof_error")

    if result.is_real:
        label = f"Live {result.spoof_score:.2f}" if show_score else "Live"
        return LivenessDecision(True, label, "real", result.spoof_score)

    return LivenessDecision(
        False,
        f"Spoof {result.spoof_score:.2f}",
        "spoof",
        result.spoof_score,
    )
