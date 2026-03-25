import cv2
import os
import time

video_path = "sample.mp4"
output_dir = "frames"

os.makedirs(output_dir, exist_ok=True)

start_time = time.time()  # 시작 시간

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("영상 열기 실패")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    print("FPS 정보를 가져올 수 없습니다.")
    cap.release()
    exit()

frame_interval = int(fps)

frame_idx = 0
saved_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % frame_interval == 0:
        resized = cv2.resize(frame, (640, 360))
        output_path = os.path.join(output_dir, f"frame_{saved_idx:03d}.jpg")
        cv2.imwrite(output_path, resized)
        saved_idx += 1

    frame_idx += 1

cap.release()

end_time = time.time()  # 끝 시간

print("프레임 추출 완료")
print(f"총 저장된 프레임 수: {saved_idx}")
print(f"총 처리 시간: {end_time - start_time:.2f}초")