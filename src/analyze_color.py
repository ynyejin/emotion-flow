import cv2
import os
import csv
import matplotlib.pyplot as plt

frames_dir = "frames"
output_csv = "color_features.csv"

# 프레임 파일 정렬
files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])

results = []

brightness_values = []
saturation_values = []
red_ratios = []
blue_ratios = []

# 프레임 분석
for file_name in files:
    file_path = os.path.join(frames_dir, file_name)
    frame = cv2.imread(file_path)

    if frame is None:
        continue

    # HSV 변환
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # 평균 값 계산
    avg_brightness = v.mean()
    avg_saturation = s.mean()

    # 빨간색 비율
    red_mask1 = ((h >= 0) & (h <= 10))
    red_mask2 = ((h >= 170) & (h <= 179))
    red_mask = red_mask1 | red_mask2
    red_ratio = red_mask.mean()

    # 파란색 비율
    blue_mask = ((h >= 100) & (h <= 130))
    blue_ratio = blue_mask.mean()

    # 결과 저장 (딕셔너리)
    results.append({
        "frame": file_name,
        "brightness": avg_brightness,
        "saturation": avg_saturation,
        "red_ratio": red_ratio,
        "blue_ratio": blue_ratio
    })

    # 그래프용 리스트
    brightness_values.append(avg_brightness)
    saturation_values.append(avg_saturation)
    red_ratios.append(red_ratio)
    blue_ratios.append(blue_ratio)

    # 터미널 출력
    print(
        f"{file_name} | "
        f"brightness={avg_brightness:.2f}, "
        f"saturation={avg_saturation:.2f}, "
        f"red_ratio={red_ratio:.3f}, "
        f"blue_ratio={blue_ratio:.3f}"
    )

print("\n분석 완료")
print(f"총 프레임 수: {len(results)}")

# =========================
# CSV 저장
# =========================
with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    # 헤더
    writer.writerow(["frame", "brightness", "saturation", "red_ratio", "blue_ratio"])

    # 데이터
    for row in results:
        writer.writerow([
            row["frame"],
            round(float(row["brightness"]), 2),
            round(float(row["saturation"]), 2),
            round(float(row["red_ratio"]), 4),
            round(float(row["blue_ratio"]), 4)
        ])

print(f"CSV 저장 완료: {output_csv}")

# =========================
# 그래프 생성
# =========================
x = list(range(len(results)))

plt.figure(figsize=(12, 6))
plt.plot(x, brightness_values, marker='o', label='Brightness')
plt.plot(x, saturation_values, marker='o', label='Saturation')
plt.plot(x, red_ratios, marker='o', label='Red Ratio')
plt.plot(x, blue_ratios, marker='o', label='Blue Ratio')

plt.title("Color Feature Flow Over Time")
plt.xlabel("Frame Index (1 frame per second)")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("color_feature_flow.png")
plt.show()