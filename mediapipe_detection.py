import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import urllib.request
import time
import numpy as np
import threading
from queue import Queue

print("Programme : 2 MAINS + VISAGE (simplifié) + DÉTECTION D'OBJETS (Lite2) + DÉTECTION POSE - Version Debug")
print("-" * 90)

# ────────────────────────────────────────────────
# Options de debug
# ────────────────────────────────────────────────
USE_THREADING = True  # Mettez False si threading cause des problèmes (ex. blocage)
DEBUG_PRINT = True    # Prints supplémentaires pour debug

# Téléchargement auto des modèles
def download_if_missing(filename, url, expected_hash=None):
    if not os.path.exists(filename):
        print(f"Téléchargement {filename} depuis {url} ...")
        urllib.request.urlretrieve(url, filename)
        print(f"{filename} téléchargé.")

hand_model_file = "hand_landmarker.task"
hand_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

face_model_file = "face_landmarker.task"
face_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"

obj_model_file = "efficientdet_lite2.tflite"
obj_url = "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite2/float16/latest/efficientdet_lite2.tflite"

pose_model_file = "pose_landmarker.task"
pose_url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"

download_if_missing(hand_model_file, hand_url)
download_if_missing(face_model_file, face_url)
download_if_missing(obj_model_file, obj_url)
download_if_missing(pose_model_file, pose_url)

# Configuration MediaPipe Tasks
BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
ObjectDetector = vision.ObjectDetector
ObjectDetectorOptions = vision.ObjectDetectorOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
RunningMode = vision.RunningMode

CONFIG = {
    "num_hands": 2,
    "min_hand_conf": 0.4,
    "num_faces": 1,
    "min_face_conf": 0.4,
    "obj_max_results": 8,
    "obj_score_threshold": 0.4,
    "frame_resize_factor": 0.8,
    "gesture_recognition": True,
    "face_blendshapes": True,
    "num_poses": 1,
    "min_pose_conf": 0.4,
}

hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=hand_model_file),
    running_mode=RunningMode.VIDEO,
    num_hands=CONFIG["num_hands"],
    min_hand_detection_confidence=CONFIG["min_hand_conf"],
    min_hand_presence_confidence=CONFIG["min_hand_conf"],
    min_tracking_confidence=CONFIG["min_hand_conf"]
)

face_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=face_model_file),
    running_mode=RunningMode.VIDEO,
    num_faces=CONFIG["num_faces"],
    min_face_detection_confidence=CONFIG["min_face_conf"],
    min_face_presence_confidence=CONFIG["min_face_conf"],
    min_tracking_confidence=CONFIG["min_face_conf"],
    output_face_blendshapes=CONFIG["face_blendshapes"],
    output_facial_transformation_matrixes=False
)

obj_options = ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path=obj_model_file),
    running_mode=RunningMode.VIDEO,
    max_results=CONFIG["obj_max_results"],
    score_threshold=CONFIG["obj_score_threshold"]
)

pose_options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=pose_model_file),
    running_mode=RunningMode.VIDEO,
    num_poses=CONFIG["num_poses"],
    min_pose_detection_confidence=CONFIG["min_pose_conf"],
    min_pose_presence_confidence=CONFIG["min_pose_conf"],
    min_tracking_confidence=CONFIG["min_pose_conf"]
)

hand_landmarker = HandLandmarker.create_from_options(hand_options)
face_landmarker = FaceLandmarker.create_from_options(face_options)
object_detector = ObjectDetector.create_from_options(obj_options)
pose_landmarker = PoseLandmarker.create_from_options(pose_options)

# Connexions
HAND_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4),
    (0,5), (5,6), (6,7), (7,8),
    (0,9), (9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9), (9,13), (13,17)
]

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32)
]

IRON_HUD_FACE_LINES = [
    (10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389),
    (389, 356), (356, 454), (454, 323), (323, 361), (361, 288), (288, 397),
    (397, 365), (365, 379), (379, 378), (378, 400), (400, 377), (377, 152),
    (152, 148), (148, 176), (176, 149), (149, 150), (150, 136), (136, 172),
    (172, 58), (58, 132), (132, 93), (93, 234), (234, 127), (127, 162),
    (162, 21), (21, 54), (54, 103), (103, 67), (67, 109), (109, 10),
    (10, 152), (33, 263), (234, 454), (61, 291), (17, 0),
    (33, 133), (133, 160), (160, 159), (159, 158), (158, 157), (157, 33),
    (263, 362), (362, 385), (385, 386), (386, 387), (387, 388), (388, 263),
    (61, 291), (291, 17), (17, 314), (314, 405), (405, 321), (321, 375), (375, 61),
    (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9),
    (151, 337), (337, 296), (296, 333), (333, 285), (285, 417)
]

# Variables
fps = 0.0
start_time = time.time()
frame_count = 0

# Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Impossible d'ouvrir la webcam")
    exit()

original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
new_width = int(original_width * CONFIG["frame_resize_factor"])
new_height = int(original_height * CONFIG["frame_resize_factor"])
cap.set(cv2.CAP_PROP_FRAME_WIDTH, new_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, new_height)

print(f"Webcam ouverte avec résolution {new_width}x{new_height} – Détection simplifiée activée !")
print("q ou Q pour quitter")
print("-" * 90)

detection_queue = Queue()

def detection_worker(mp_image, timestamp_ms):
    if DEBUG_PRINT:
        print("Début des détections...")
    hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
    face_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)
    obj_result = object_detector.detect_for_video(mp_image, timestamp_ms)
    pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
    detection_queue.put((hand_result, face_result, obj_result, pose_result))
    if DEBUG_PRINT:
        print("Détections terminées.")

# Fonctions gestes et émotions
def is_finger_extended(landmarks, tip_idx, pip_idx, mcp_idx):
    tip = landmarks[tip_idx]
    pip = landmarks[pip_idx]
    mcp = landmarks[mcp_idx]
    dist_tip_mcp = np.linalg.norm([tip.x - mcp.x, tip.y - mcp.y, tip.z - mcp.z])
    dist_pip_mcp = np.linalg.norm([pip.x - mcp.x, pip.y - mcp.y, pip.z - mcp.z])
    dist_tip_pip = np.linalg.norm([tip.x - pip.x, tip.y - pip.y, tip.z - pip.z])
    if dist_tip_mcp > 0.95 * (dist_tip_pip + dist_pip_mcp):
        return True
    return False

def detect_gesture(hand_landmarks):
    if not hand_landmarks:
        return "Aucun"
    finger_indices = {
        'thumb': {'tip': 4, 'pip': 3, 'mcp': 2},
        'index': {'tip': 8, 'pip': 6, 'mcp': 5},
        'middle': {'tip': 12, 'pip': 10, 'mcp': 9},
        'ring': {'tip': 16, 'pip': 14, 'mcp': 13},
        'pinky': {'tip': 20, 'pip': 18, 'mcp': 17}
    }
    extended_fingers = {finger: is_finger_extended(hand_landmarks, idx['tip'], idx['pip'], idx['mcp']) for finger, idx in finger_indices.items()}
    num_extended = sum(extended_fingers.values())
    thumb_tip = hand_landmarks[4]
    index_tip = hand_landmarks[8]
    dist_thumb_index = np.linalg.norm([thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y, thumb_tip.z - index_tip.z])
    if dist_thumb_index < 0.04 and extended_fingers['middle'] and extended_fingers['ring'] and extended_fingers['pinky']:
        return "OK"
    if num_extended == 0:
        return "Poing"
    elif num_extended == 5:
        return "Ouvert"
    elif num_extended == 1 and extended_fingers['thumb']:
        return "Pouce en l'air"
    elif num_extended == 1 and extended_fingers['index']:
        return "Pointer"
    elif num_extended == 2 and extended_fingers['index'] and extended_fingers['middle']:
        return "Peace"
    elif num_extended == 3 and extended_fingers['thumb'] and extended_fingers['index'] and extended_fingers['pinky']:
        return "Rock"
    elif num_extended == 4 and not extended_fingers['pinky']:
        return "Quatre"
    else:
        return f"{num_extended} doigts"

def detect_emotion(blendshapes):
    if not blendshapes:
        return "Neutre"
    smile_score = max((shape.score for shape in blendshapes if "smile" in shape.category_name.lower()), default=0)
    if smile_score > 0.5:
        return "Sourire"
    return "Neutre"

frame_timestamp_ms = 0
loop_start_time = time.time()

# Créer une fenêtre redimensionnable
cv2.namedWindow('Simplified Detection - Debug', cv2.WINDOW_NORMAL)

while True:
    loop_time = time.time()
    success, image = cap.read()
    if not success:
        print("Erreur lecture caméra")
        break
    if DEBUG_PRINT:
        print("Frame lu avec succès.")

    image = cv2.flip(image, 1)
    if image.shape[1] != new_width:
        image = cv2.resize(image, (new_width, new_height))

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    frame_timestamp_ms = int((time.time() - start_time) * 1000)

    if USE_THREADING:
        detection_thread = threading.Thread(target=detection_worker, args=(mp_image, frame_timestamp_ms))
        detection_thread.start()
        annotated_image = image.copy()
        detection_thread.join()
        hand_result, face_result, obj_result, pose_result = detection_queue.get()
    else:
        # Sans threading pour debug
        annotated_image = image.copy()
        hand_result = hand_landmarker.detect_for_video(mp_image, frame_timestamp_ms)
        face_result = face_landmarker.detect_for_video(mp_image, frame_timestamp_ms)
        obj_result = object_detector.detect_for_video(mp_image, frame_timestamp_ms)
        pose_result = pose_landmarker.detect_for_video(mp_image, frame_timestamp_ms)

    # ─── MAINS ───────────────────────────────────────────────
    hand_count = 0
    gestures = []
    if hand_result.hand_landmarks:
        for hand_landmarks_list in hand_result.hand_landmarks:
            hand_count += 1
            gesture = detect_gesture(hand_landmarks_list) if CONFIG["gesture_recognition"] else "N/A"
            gestures.append(gesture)
            landmarks_np = np.array([(lm.x * annotated_image.shape[1], lm.y * annotated_image.shape[0]) for lm in hand_landmarks_list], dtype=int)
            for lm in landmarks_np:
                cv2.circle(annotated_image, tuple(lm), 3, (0, 255, 0), -1)
            for start_idx, end_idx in HAND_CONNECTIONS:
                cv2.line(annotated_image, tuple(landmarks_np[start_idx]), tuple(landmarks_np[end_idx]), (255, 0, 0), 1)

    if hand_count > 0:
        gesture_text = " | ".join(gestures)
        cv2.putText(annotated_image, f"{hand_count} main{'s' if hand_count>1 else ''} - Gestes: {gesture_text}",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # ─── VISAGE ───────────────────────────────────────────────
    if face_result.face_landmarks:
        blendshapes_lists = face_result.face_blendshapes if face_result.face_blendshapes else []
        for i, face_landmarks_list in enumerate(face_result.face_landmarks):
            blendshapes_list = blendshapes_lists[i] if i < len(blendshapes_lists) else []
            h, w, _ = annotated_image.shape
            landmarks_np = np.array([(lm.x * w, lm.y * h) for lm in face_landmarks_list], dtype=int)
            for start_idx, end_idx in IRON_HUD_FACE_LINES:
                if start_idx >= len(landmarks_np) or end_idx >= len(landmarks_np):
                    continue
                pt1, pt2 = tuple(landmarks_np[start_idx]), tuple(landmarks_np[end_idx])
                cv2.line(annotated_image, pt1, pt2, (0, 255, 255), 1)

    # ─── POSE ───────────────────────────────────────────────
    pose_count = 0
    if pose_result.pose_landmarks:  # Commentez ce bloc pour tester sans pose si lent
        for pose_landmarks_list in pose_result.pose_landmarks:
            pose_count += 1
            landmarks_np = np.array([(lm.x * annotated_image.shape[1], lm.y * annotated_image.shape[0]) for lm in pose_landmarks_list], dtype=int)
            for lm in landmarks_np:
                cv2.circle(annotated_image, tuple(lm), 3, (255, 0, 255), -1)
            for start_idx, end_idx in POSE_CONNECTIONS:
                if start_idx < len(landmarks_np) and end_idx < len(landmarks_np):
                    cv2.line(annotated_image, tuple(landmarks_np[start_idx]), tuple(landmarks_np[end_idx]), (0, 255, 255), 1)

    if pose_count > 0:
        cv2.putText(annotated_image, f"{pose_count} pose{'s' if pose_count > 1 else ''} détectée{'s' if pose_count > 1 else ''}",
                    (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

    # ─── OBJETS ─────────────────────────────────────────────────
    if obj_result.detections:
        h, w, _ = annotated_image.shape
        for detection in obj_result.detections:
            bbox = detection.bounding_box
            category = detection.categories[0]
            label = category.category_name or f"Obj {category.index}"
            score = category.score
            x1 = int(bbox.origin_x)
            y1 = int(bbox.origin_y)
            x2 = int(bbox.origin_x + bbox.width)
            y2 = int(bbox.origin_y + bbox.height)
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 140, 255), 1)
            text = f"{label} {score:.2f}"
            cv2.putText(annotated_image, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)

    # Rien détecté
    if hand_count == 0 and not face_result.face_landmarks and not obj_result.detections and pose_count == 0:
        cv2.putText(annotated_image, "Rien détecté", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # FPS
    frame_count += 1
    elapsed = time.time() - loop_start_time
    if elapsed > 1.0:
        fps = frame_count / elapsed
        frame_count = 0
        loop_start_time = time.time()
        if DEBUG_PRINT:
            print(f"FPS calculé : {fps:.1f}")
    cv2.putText(annotated_image, f"FPS: {fps:.1f}", (annotated_image.shape[1] - 100, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('Simplified Detection - Debug', annotated_image)
    if DEBUG_PRINT:
        print("Frame affiché.")

    if cv2.waitKey(30) & 0xFF in (ord('q'), ord('Q')):  # Augmenté à 30 pour mieux rafraîchir
        break

# Nettoyage
cap.release()
cv2.destroyAllWindows()
hand_landmarker.close()
face_landmarker.close()
object_detector.close()
pose_landmarker.close()

print("\nProgramme terminé")
