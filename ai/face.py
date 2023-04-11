import dlib
import numpy as np


def detect_faces(detector, img):
    w, h = img.size
    return [{
        'x': box.left() / w,
        'y': box.top() / h,
        'w': (box.right() - box.left()) / w,
        'h': (box.bottom() - box.top()) / h,
    } for box in detector(np.asarray(img), 1)]


def predict_landmarks(predictor, img, box):
    w, h = img.size
    box = dlib.rectangle(
        left=int(box['x'] * w),
        right=int((box['x'] + box['w']) * w),
        top=int(box['y'] * h),
        bottom=int((box['y'] + box['h']) * h),
    )
    lms = predictor(np.asarray(img), box)
    return [[p.x, p.y] for p in lms.parts()]


def landmarks_to_alignment_coords(lm):
    lm = np.array(lm)
    lm_chin = lm[0:17]  # left-right
    lm_eyebrow_left = lm[17:22]  # left-right
    lm_eyebrow_right = lm[22:27]  # left-right
    lm_nose = lm[27:31]  # top-down
    lm_nostrils = lm[31:36]  # top-down
    lm_eye_left = lm[36:42]  # left-clockwise
    lm_eye_right = lm[42:48]  # left-clockwise
    lm_mouth_outer = lm[48:60]  # left-clockwise
    lm_mouth_inner = lm[60:68]  # left-clockwise

    # calculate auxiliary vectors
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # choose oriented crop rectangle
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    return (quad + 0.5).flatten()
