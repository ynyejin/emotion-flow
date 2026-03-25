import cv2
import os
import matplotlib.pyplot as plt

frames_dir = "frames"

files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])

results = []

brightness_values = []
saturation_values = []
red_ratios = []
blue_ratios = []

for file_name in files:
    file_path = os.path.join(frames_dir, file_name)
    frame = cv2.imread(file_path)

    if frame is None:
        continue

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    avg_brightness = v.mean()
    avg_saturation = s.mean()

    red_mask1 = ((h >= 0) & (h <= 10))
    red_mask2 = ((h >= 170) & (h <= 179))
    red_mask = red_mask1 | red_mask2
    red_ratio = red_mask.mean()

    blue_mask = ((h >= 100) & (h <= 130))
    blue_ratio = blue_mask.mean()

    results.append({
        "frame": file_name,
        "brightness": avg_brightness,
        "saturation": avg_saturation,
        "red_ratio": red_ratio,
        "blue_ratio": blue_ratio
    })

    brightness_values.append(avg_brightness)
    saturation_values.append(avg_saturation)
    red_ratios.append(red_ratio)
    blue_ratios.append(blue_ratio)

    print(
        f"{file_name} | "
        f"brightness={avg_brightness:.2f}, "
        f"saturation={avg_saturation:.2f}, "
        f"red_ratio={red_ratio:.3f}, "
        f"blue_ratio={blue_ratio:.3f}"
    )

print("\n분석 완료")
print(f"총 프레임 수: {len(results)}")

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