import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Generate Test Image (Ramp Edge)
width = 150
height = 100
# FIX: Use float64 here as well
img = np.zeros((height, width), dtype=np.float64)

# Create a ramp from column 50 to 100
for col in range(width):
    if col < 50:
        img[:, col] = 0
    elif col >= 100:
        img[:, col] = 255
    else:
        img[:, col] = (col - 50) * (255 / 50)

# Normalize to 0-1 range if desired, though logic holds without it for derivatives

# 2. First Derivative (Sobel)
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobel_abs = np.abs(sobel_x)

# 3. Gaussian Blur (Pre-processing for Second Derivative)
blurred = cv2.GaussianBlur(img, (5, 5), 1.0)

# 4. Second Derivative (Laplacian of Gaussian)
laplacian = cv2.Laplacian(blurred, cv2.CV_64F)

# 5. Zero Crossing Detection
zero_crossings = np.zeros_like(laplacian)
rows, cols = laplacian.shape
for r in range(rows):
    for c in range(1, cols):
        if laplacian[r, c] * laplacian[r, c-1] < 0:
            zero_crossings[r, c] = 255

plt.figure(figsize=(12, 10))

# Row 1: Intensity Profile
plt.subplot(3, 1, 1)
mid_row = height // 2
plt.plot(img[mid_row, :], label='Original Intensity')
plt.title("Intensity Profile (Center Row)")
plt.grid(True)
plt.legend()

# Row 2: Visualizing Images
plt.subplot(3, 2, 3)
plt.imshow(img, cmap='gray')
plt.title("Original Ramp Edge")
plt.axis('off')

plt.subplot(3, 2, 4)
plt.imshow(blurred, cmap='gray')
plt.title("Blurred Ramp (Gaussian)")
plt.axis('off')

plt.subplot(3, 2, 5)
plt.imshow(sobel_abs, cmap='gray')
plt.title("First Derivative (Sobel)")
plt.axis('off')

plt.subplot(3, 2, 6)
plt.imshow(zero_crossings, cmap='gray')
plt.title("Detected Zero-Crossings (LoG)")
plt.axis('off')

plt.tight_layout()
plt.show()