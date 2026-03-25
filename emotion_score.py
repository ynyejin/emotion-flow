import csv
import numpy as np
import matplotlib.pyplot as plt

input_csv = "color_features.csv"

frames = []
brightness_values = []
saturation_values = []
red_ratios = []
blue_ratios = []

# =========================
# CSV 읽기
# =========================
with open(input_csv, mode="r", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        frames.append(row["frame"])
        brightness_values.append(float(row["brightness"]))
        saturation_values.append(float(row["saturation"]))
        red_ratios.append(float(row["red_ratio"]))
        blue_ratios.append(float(row["blue_ratio"]))

print(f"CSV 로드 완료: {input_csv}")
print(f"총 프레임 수: {len(frames)}")

# =========================
# 정규화 함수
# =========================
def normalize(values):
    arr = np.array(values, dtype=float)
    min_val = arr.min()
    max_val = arr.max()

    if max_val - min_val == 0:
        return np.zeros_like(arr)

    return (arr - min_val) / (max_val - min_val)

# 정규화
brightness_n = normalize(brightness_values)
saturation_n = normalize(saturation_values)
red_n = normalize(red_ratios)
blue_n = normalize(blue_ratios)

# =========================
# 감정 점수 계산
# =========================
tension_scores = (
    (1 - brightness_n) * 0.4 +
    saturation_n * 0.4 +
    red_n * 0.2
)

calmness_scores = (
    brightness_n * 0.6 +
    (1 - saturation_n) * 0.4
)

# 출력 확인
for i in range(len(frames)):
    print(
        f"{frames[i]} | "
        f"Tension={tension_scores[i]:.3f}, "
        f"Calmness={calmness_scores[i]:.3f}"
    )

# =========================
# 그래프 생성
# =========================
x = list(range(len(frames)))

plt.figure(figsize=(12, 6))
plt.plot(x, tension_scores, marker='o', label='Tension')
plt.plot(x, calmness_scores, marker='o', label='Calmness')

plt.title("Emotion Flow (Color-based)")
plt.xlabel("Frame Index (1 frame per second)")
plt.ylabel("Normalized Score")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("emotion_flow.png")
plt.show()