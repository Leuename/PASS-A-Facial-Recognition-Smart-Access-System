from __future__ import annotations

import pickle
from pathlib import Path
from typing import Callable, Sequence

import numpy as np


DEFAULT_TOLERANCE = 0.35
DEFAULT_CV_SCALER = 4
DEFAULT_DETECTION_MODEL = "hog"
DEFAULT_ENCODING_MODEL = "large"


def load_encodings(path: str | Path, missing_ok: bool = False):
    path = Path(path)
    if not path.exists():
        if missing_ok:
            return [], []
        raise RuntimeError(f"Encodings file not found: {path}")

    with path.open("rb") as file:
        data = pickle.loads(file.read())

    encodings = list(data.get("encodings", []))
    names = list(data.get("names", []))
    if len(encodings) != len(names):
        raise RuntimeError(f"Encodings file has mismatched encodings and names: {path}")

    return encodings, names


def save_encodings(path: str | Path, encodings: Sequence, names: Sequence[str], metadata: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "encodings": [np.asarray(encoding, dtype=np.float64) for encoding in encodings],
        "names": list(names),
        "metadata": metadata or {},
    }
    with path.open("wb") as file:
        file.write(pickle.dumps(payload))


def match_face(
    face_encoding,
    known_face_encodings: Sequence,
    known_face_names: Sequence[str],
    face_distance: Callable | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
):
    if not known_face_encodings or not known_face_names:
        return "Unknown", 0.0

    if face_distance is None:
        try:
            import face_recognition
        except ModuleNotFoundError as exc:
            raise RuntimeError("Missing Python dependency 'face_recognition'.") from exc
        face_distance = face_recognition.face_distance

    distances = np.asarray(face_distance(known_face_encodings, face_encoding), dtype=np.float64)
    if distances.size == 0:
        return "Unknown", 0.0

    best_index = int(np.argmin(distances))
    best_distance = float(distances[best_index])
    confidence = max(0.0, min(1.0, 1.0 - best_distance))
    if best_distance <= float(tolerance):
        return str(known_face_names[best_index]), confidence
    return "Unknown", confidence


def match_faces(
    face_encodings: Sequence,
    known_face_encodings: Sequence,
    known_face_names: Sequence[str],
    face_distance: Callable | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
):
    return [
        match_face(
            face_encoding,
            known_face_encodings,
            known_face_names,
            face_distance=face_distance,
            tolerance=tolerance,
        )
        for face_encoding in face_encodings
    ]


def upsert_person(
    known_face_encodings: Sequence,
    known_face_names: Sequence[str],
    name: str,
    new_encodings: Sequence,
):
    kept = [
        (encoding, person_name)
        for encoding, person_name in zip(known_face_encodings, known_face_names)
        if person_name != name
    ]
    encodings = [encoding for encoding, _ in kept]
    names = [person_name for _, person_name in kept]
    encodings.extend(new_encodings)
    names.extend([name] * len(new_encodings))
    return encodings, names


def delete_person(known_face_encodings: Sequence, known_face_names: Sequence[str], name: str):
    kept = [
        (encoding, person_name)
        for encoding, person_name in zip(known_face_encodings, known_face_names)
        if person_name != name
    ]
    removed = len(kept) != len(known_face_names)
    return [encoding for encoding, _ in kept], [person_name for _, person_name in kept], removed


def list_people(known_face_encodings: Sequence, known_face_names: Sequence[str]):
    people = {}
    for encoding, name in zip(known_face_encodings, known_face_names):
        dims = int(np.asarray(encoding).shape[0])
        if name not in people:
            people[name] = {"name": name, "dims": dims, "samples": 0}
        people[name]["samples"] += 1
    return [people[name] for name in sorted(people)]


def scaled_face_location_to_bbox(face_location, cv_scaler: int = DEFAULT_CV_SCALER):
    top, right, bottom, left = [int(value) for value in face_location]
    return [
        int(left * cv_scaler),
        int(top * cv_scaler),
        int(right * cv_scaler),
        int(bottom * cv_scaler),
    ]


def detect_and_encode_faces(
    frame,
    cv2,
    face_recognition,
    cv_scaler: int = DEFAULT_CV_SCALER,
    detection_model: str = DEFAULT_DETECTION_MODEL,
    encoding_model: str = DEFAULT_ENCODING_MODEL,
):
    resized_frame = cv2.resize(frame, (0, 0), fx=(1 / cv_scaler), fy=(1 / cv_scaler))
    rgb_resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_resized_frame, model=detection_model)
    face_encodings = face_recognition.face_encodings(
        rgb_resized_frame,
        face_locations,
        model=encoding_model,
    )
    return face_locations, face_encodings
