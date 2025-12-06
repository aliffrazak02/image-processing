import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label

def compute_center_of_gravity(component_mask):
    """Compute center of gravity for a binary mask."""
    ys, xs = np.nonzero(component_mask)
    if len(xs) == 0:
        return None
    cx = int(np.mean(xs))
    cy = int(np.mean(ys))
    return (cx, cy)

binary_img = cv2.imread("assignment-5\\bwimage.png", cv2.IMREAD_GRAYSCALE)

_, binary_img = cv2.threshold(binary_img, 127, 255, cv2.THRESH_BINARY)

labeled, num_components = label(binary_img)

print(f"Detected components: {num_components}")

label_norm = (labeled.astype(np.float32) / labeled.max() * 255).astype(np.uint8)
color_map = cv2.applyColorMap(label_norm, cv2.COLORMAP_JET)

output = color_map.copy()

for comp_label in range(1, num_components + 1):
    component_mask = (labeled == comp_label)

    # Compute center of gravity
    center = compute_center_of_gravity(component_mask)
    if center is None:
        continue

    cx, cy = center

    # Draw a cross (+)
    size = 8
    color = (255, 255, 255)  # white cross
    thickness = 2

    cv2.line(output, (cx - size, cy), (cx + size, cy), color, thickness)
    cv2.line(output, (cx, cy - size), (cx, cy + size), color, thickness)

plt.figure(figsize=(10, 5))
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title("Connected Components with Center Cross Marks")
plt.axis("off")
plt.show()
