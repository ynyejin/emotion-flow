import cv2
import os
import mediapipe as mp

frames_dir = "frames"
result_dir = "face_results"
crop_dir = "face_crops"

os.makedirs(result_dir, exist_ok=True)
os.makedirs(crop_dir, exist_ok=True)

files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)

for file_name in files:
    file_path = os.path.join(frames_dir, file_name)
    frame = cv2.imread(file_path)

    if frame is None:
        continue

    h_img, w_img, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb)

    face_count = 0

    if results.detections:
        for i, detection in enumerate(results.detections):
            bbox = detection.location_data.relative_bounding_box

            x = int(bbox.xmin * w_img)
            y = int(bbox.ymin * h_img)
            w = int(bbox.width * w_img)
            h = int(bbox.height * h_img)

            # 이미지 밖으로 나가지 않게 보정
            x = max(0, x)
            y = max(0, y)
            w = min(w, w_img - x)
            h = min(h, h_img - y)

            # 얼굴 박스 그리기
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 얼굴 crop 저장
            face_crop = frame[y:y+h, x:x+w]
            crop_path = os.path.join(crop_dir, f"{file_name[:-4]}_face{i}.jpg")
            cv2.imwrite(crop_path, face_crop)

            face_count += 1

    # 박스 그려진 결과 이미지 저장
    result_path = os.path.join(result_dir, file_name)
    cv2.imwrite(result_path, frame)

    print(f"{file_name} | faces detected: {face_count}")

print("\n얼굴 박스 결과 저장 완료")
print(f"박스 결과 폴더: {result_dir}")
print(f"크롭 결과 폴더: {crop_dir}")