"""Video integrity checks and reviewed PMC label loading."""
import csv
from pathlib import Path
import cv2
import numpy as np


def verify_video(path, expected, size):
    capture = cv2.VideoCapture(str(path))
    count = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame.shape[:2] != (size[1], size[0]):
            raise RuntimeError(f'Wrong decoded dimensions: {path}')
        count += 1
    capture.release()
    if count != expected:
        raise RuntimeError(f'{path}: decoded {count}, expected {expected}')
    return count


def load_labels(data, sequence):
    """Reviewed zero masks remain valid; missing masks are never assumed empty."""
    with (data / 'meta_PMC9883282_1.csv').open(newline='') as file:
        rows = [r for r in csv.DictReader(file) if r['sequence_id'] == sequence]
    if not rows:
        raise ValueError(f'No metadata for {sequence}')
    labels = {}
    for row in rows:
        if row['mask_status'].lower() != 'true':
            continue
        path = data / 'masks' / Path(row['mask_path']).name
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(path)
        if not set(np.unique(mask)).issubset({0, 128, 255}):
            raise ValueError(f'Unexpected annotation IDs: {path}')
        labels[int(row['frame_index'])] = np.where(mask == 255, 1, np.where(mask == 128, 2, 0)).astype('uint8')
    return rows, labels
