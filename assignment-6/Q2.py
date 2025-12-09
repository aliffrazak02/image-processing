import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Setup the same Ramp Image
width = 150
height = 100
img = np.zeros((height, width), dtype=np.float64)

for col in range(width):
    if col < 50:
        img[:, col] = 0
    elif col >= 100:
        img[:, col] = 255
    else:
        img[:, col] = (col - 50) * (255 / 50)

# 2. Compute Blurred Version
blurred = cv2.GaussianBlur(img, (9, 9), 2.0)

# 3. Compute Laplacian of the BLURRED image
laplacian = cv2.Laplacian(blurred, cv2.CV_64F)

# 4. Extract Center Row Profiles
mid_row = height // 2
profile_orig = img[mid_row, :]
profile_blur = blurred[mid_row, :]
profile_lap = laplacian[mid_row, :]

# Scale Laplacian for better visualization
profile_lap_scaled = profile_lap * 5 

plt.figure(figsize=(10, 6))

# Main Plot: Overlaid Profiles
plt.plot(profile_orig, label='Original Intensity', color='blue', linewidth=1.5)
plt.plot(profile_blur, label='Blurred Intensity', color='orange', linewidth=1.5)
plt.plot(profile_lap_scaled, label='Laplacian (Scaled)', color='green', linewidth=1.5)

# Highlight Zero Crossing
zero_cross_idx = np.where(np.diff(np.sign(profile_lap)))[0]
valid_crossings = [x for x in zero_cross_idx if 50 < x < 100]

if valid_crossings:
    zc = valid_crossings[0]
    plt.axvline(x=zc, color='red', linestyle='--', label='Zero Crossing')

plt.title("Center-row Profiles: Intensity vs Laplacian")
plt.xlabel("Pixel Position (x)")
plt.ylabel("Intensity / Value")
plt.grid(True, alpha=0.5)
plt.legend()

# Zoom Inset
plt.axes([0.6, 0.2, 0.25, 0.25])
plt.plot(profile_lap_scaled[40:110], color='green')
plt.title("Laplacian Zoom")
plt.grid(True)
plt.xticks([])
plt.yticks([])

plt.show()