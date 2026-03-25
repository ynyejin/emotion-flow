import cv2
import os
import mediapipe as mp

frames_dir = "frames"

files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)

face_counts = []

for file_name in files:
    file_path = os.path.join(frames_dir, file_name)
    frame = cv2.imread(file_path)

    if frame is None:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb)

    if results.detections:
        count = len(results.detections)
    else:
        count = 0

    face_counts.append(count)

    print(f"{file_name} | faces: {count}")

print("\n얼굴 검출 완료")
print(f"총 프레임 수: {len(face_counts)}")