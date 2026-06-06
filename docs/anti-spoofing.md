# Anti-Spoofing MobileNetV2 Code Walkthrough

Presentation version. Short, code-focused, and limited to the anti-spoofing model.

## 1. What This Model Does

The anti-spoofing model checks whether a detected face looks real or fake before the system performs face recognition.

In plain terms:

- Real face: continue to recognition.
- Fake/spoof face: reject immediately.
- Model broken or unavailable: reject for safety.

The MobileNetV2 model itself is stored as a TensorFlow Lite file:

```text
models/anti_spoof_model/best.tflite
```

The Python code does not rebuild MobileNetV2. It loads the finished `.tflite` model and runs it.

## 2. Model Settings

Source: `models/anti_spoof_model/metadata.json`

```json
{
  "task": "face_anti_spoofing_binary_classifier",
  "sigmoid_output": "spoof_score = probability that image is spoof",
  "decision_rule": {
    "spoof_score >= threshold": "reject_as_spoof",
    "spoof_score < threshold": "continue_to_face_recognition"
  },
  "recommended_threshold": 0.5,
  "input_size": [224, 224, 3],
  "preprocessing": "rescale 1./255"
}
```

What it means:

- The model gives one score called `spoof_score`.
- A higher score means the face looks more fake.
- If the score is `0.5` or higher, reject it.
- The model expects a `224 x 224` color image.
- Pixel values must be scaled from `0..255` down to `0..1`.

## 3. Result Format

Source: `anti_spoofing.py:11-19`

```python
@dataclass(frozen=True)
class AntiSpoofResult:
    ready: bool
    is_real: bool
    is_spoof: bool
    spoof_score: float
    threshold: float
    label: str
    error: Optional[str] = None
```

What it does:

This is the small answer object returned by the anti-spoofing check.

Important fields:

- `is_real`: the face passed.
- `is_spoof`: the face failed.
- `spoof_score`: the model's fake-face score.
- `label`: simple output, either `real` or `spoof`.
- `error`: what went wrong, if prediction failed.

## 4. Loading the Model

Source: `anti_spoofing.py:29-56`

```python
self.model_dir = Path(model_dir)
self.model_path = self.model_dir / "best.tflite"
self.metadata_path = self.model_dir / "metadata.json"
self.threshold = 0.5
self.input_size = (224, 224)
self.crop_margin = 0.25

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
self.ready = True
```

What it does:

The classifier prepares everything needed to run the model:

1. Finds `best.tflite`.
2. Reads `metadata.json`.
3. Sets the threshold and image size.
4. Opens the TensorFlow Lite model.
5. Stores the model input and output positions.
6. Marks itself as ready.

If loading fails, `ready` stays `False`.

## 5. Loading Model Metadata

Source: `anti_spoofing.py:61-78`

```python
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
```

What it does:

The model file comes with an instruction sheet. This code reads that sheet and applies:

- the spoof threshold,
- the expected image size,
- the face crop margin.

This avoids hardcoding every model rule in Python.

## 6. Threshold Override

Source: `anti_spoofing.py:80-86`

```python
env_threshold = os.getenv("ANTI_SPOOF_THRESHOLD")
if threshold is not None:
    return float(threshold)
if env_threshold:
    return float(env_threshold)
return float(self.threshold)
```

What it does:

The threshold can be changed without editing the model:

1. Direct code value wins first.
2. `ANTI_SPOOF_THRESHOLD` environment variable wins second.
3. Metadata/default value is used last.

For a stricter door lock, the team can tune this value.

## 7. Cropping the Face

Source: `anti_spoofing.py:152-171`

```python
height, width = frame.shape[:2]
x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
box_w = x2 - x1
box_h = y2 - y1

pad_x = box_w * self.crop_margin
pad_y = box_h * self.crop_margin

left = max(0, int(np.floor(x1 - pad_x)))
top = max(0, int(np.floor(y1 - pad_y)))
right = min(width, int(np.ceil(x2 + pad_x)))
bottom = min(height, int(np.ceil(y2 + pad_y)))

return frame[top:bottom, left:right]
```

What it does:

The face detector gives a box around the face. This code:

- takes that box,
- adds extra space around it,
- keeps it inside the camera frame,
- cuts out only that face area.

The model checks the face crop, not the entire camera image.

## 8. Preparing the Image for MobileNetV2

Source: `anti_spoofing.py:173-179`

```python
target_w, target_h = self.input_size[0], self.input_size[1]
resized = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_AREA)
rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
tensor = rgb.astype(np.float32) / 255.0
tensor = np.expand_dims(tensor, axis=0)
return tensor.astype(self._input_dtype, copy=False)
```

What it does:

The crop is converted into the exact format the model expects:

- resize to `224 x 224`,
- convert OpenCV color order from BGR to RGB,
- scale pixels from `0..255` to `0..1`,
- add a batch dimension so the model receives one image.

Without this step, the model may read the image incorrectly.

## 9. Running Prediction

Source: `anti_spoofing.py:118-140`

```python
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
```

What it does:

This is the core decision:

1. Crop the face.
2. Prepare the crop.
3. Send it to the MobileNetV2 TFLite model.
4. Read the model's score.
5. Compare score to threshold.
6. Return `real` or `spoof`.

Decision rule:

```text
spoof_score >= threshold  => reject
spoof_score < threshold   => continue
```

## 10. Failing Safely

Source: `anti_spoofing.py:107-116`, `anti_spoofing.py:141-150`

```python
return AntiSpoofResult(
    ready=False,
    is_real=False,
    is_spoof=True,
    spoof_score=1.0,
    threshold=self.threshold,
    label="spoof",
    error=self.error or "Anti-spoof classifier is unavailable",
)
```

```python
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
```

What it does:

If the model is unavailable or prediction crashes, the system treats the face as spoof.

For a door lock, this is safer than accidentally trusting a bad or incomplete prediction.

## 11. Server Startup

Source: `arclight_server.py:115-123`

```python
print("Loading anti-spoof model...")
anti_spoof = AntiSpoofClassifier(ANTI_SPOOF_DIR)
if anti_spoof.ready:
    print(f"  Anti-spoof ready. threshold={anti_spoof.threshold:.2f}")
else:
    print(f"  Anti-spoof unavailable: {anti_spoof.error}")
```

What it does:

When the app starts, it loads the anti-spoofing model first and reports whether the safety check is available.

## 12. Camera and Enrollment Require the Model

Source: `arclight_server.py:153-157`, `arclight_server.py:377-380`, `arclight_server.py:419-422`

```python
def require_anti_spoof_ready():
    status = get_anti_spoof_status()
    if not status["ready"]:
        detail = status.get("error") or "Anti-spoof classifier is unavailable"
        raise HTTPException(status_code=503, detail=f"Anti-spoof unavailable: {detail}")
```

```python
@app.post("/api/camera/start")
async def start_camera():
    require_anti_spoof_ready()
```

```python
@app.post("/api/faces/enroll/start")
async def enroll_start(req: EnrollRequest):
    require_anti_spoof_ready()
```

What it does:

The camera and enrollment flows are blocked if anti-spoofing is not ready.

In plain terms:

The system should not run sensitive face workflows without the fake-face checker.

## 13. Live Camera Rejection

Source: `arclight_server.py:273-283`

```python
liveness = anti_spoof.predict(frame, (x1, y1, x2, y2)) if anti_spoof else None
if liveness is None or not liveness.is_real:
    color = (0, 0, 255)
    spoof_score = liveness.spoof_score if liveness else 1.0
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, f"Spoof  {spoof_score:.2f}",
                (x1, max(y1 - 8, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    continue

name, score = recognize(face.embedding)
```

What it does:

The live camera checks liveness before recognition.

If the face is spoof:

- draw a red box,
- show the spoof score,
- skip recognition,
- do not unlock from that face.

Only real-looking faces reach `recognize(...)`.

## 14. Enrollment Rejection

Source: `arclight_server.py:325-329`

```python
liveness = anti_spoof.predict(frame, (x1, y1, x2, y2)) if anti_spoof else None
if liveness is None or not liveness.is_real:
    spoof_score = liveness.spoof_score if liveness else 1.0
    enroll_state["message"] = f"Spoof rejected ({spoof_score:.2f}). Use a real face."
    return
```

What it does:

Enrollment also uses anti-spoofing.

If the face looks fake, the system does not save it as a trusted enrolled face.

This prevents someone from enrolling a photo or screen image.

## Summary

It works like this:

1. Detect a face.
2. Crop and prepare the face image.
3. Run the TFLite MobileNetV2 anti-spoof model.
4. Get a spoof score.
5. Reject if the score is too high.
6. Continue to recognition only if the face looks real.

The most important security behavior:

```text
If anti-spoofing fails, the system rejects instead of trusting the face.
```
