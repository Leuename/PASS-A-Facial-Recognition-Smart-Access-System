# Arclight Raspberry Pi Runtime

This folder contains the clean runtime set for Raspberry Pi 4B deployment.

## Included Files

- `arclight_server.py` - FastAPI server, camera stream, GPIO lock/button control.
- `arclight_ui.html` - browser UI served at `/`.
- `arclight_face_recognition.py` - `face_recognition` detection, encoding, matching, and pickle helpers.
- `encodings.pickle` - current known-face database copied from the existing `face_recognition` project.
- `anti_spoof.py` and `anti_spoof_model/` - TFLite anti-spoofing gate.
- `requirements.txt` - Raspberry Pi setup notes and pip dependencies.
- `Faces4Arclight/` - enrollment folder placeholder.
- `savedlogs/` - exported attendance log folder.
- `run.sh` - server startup script.

## Run

```bash
chmod +x run.sh
./run.sh
```

Open:

```text
http://<raspberry-pi-ip>:8000
```

## Runtime Notes

- The active recognition path uses `face_recognition`, not InsightFace or YOLO.
- `ARCLIGHT_FACE_DETECTION_MODEL=hog` is the default and is the practical CPU choice on Raspberry Pi 4B.
- `ARCLIGHT_FACE_ENCODING_MODEL=large` is the default because `encodings.pickle` was trained with the large model.
- Anti-spoofing is enabled and fail-closed by default. If the TFLite runtime or model cannot load, known faces will not unlock/log.

Useful environment variables:

```bash
export ARCLIGHT_FACE_CV_SCALER=4
export ARCLIGHT_RECOGNITION_TOLERANCE=0.6
export ARCLIGHT_FACE_DETECTION_MODEL=hog
export ARCLIGHT_FACE_ENCODING_MODEL=large
export ARCLIGHT_ANTI_SPOOF_THRESHOLD=0.35
export ARCLIGHT_ANTI_SPOOF_THREADS=2
export ARCLIGHT_ANTI_SPOOF_BBOX_EXPANSION=0.35
export ARCLIGHT_ANTI_SPOOF_CROP_MARGIN=0.2
```

Anti-spoof bounding-box tuning:

- `ARCLIGHT_ANTI_SPOOF_BBOX_EXPANSION` expands the detected face box before it is sent to anti-spoofing, matching the Windows testing behavior. Set `0` to disable this outer expansion.
- `ARCLIGHT_ANTI_SPOOF_CROP_MARGIN` controls the inner crop margin used by the anti-spoof model preprocessor. Set `0` to crop exactly the incoming bbox.
