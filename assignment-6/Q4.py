import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load and Normalize Image
I = cv2.imread("assignment-6\coins.png", cv2.IMREAD_GRAYSCALE)
if I is None:
    raise FileNotFoundError("coins.png not found")

Id = I.astype(np.float32) / 255.0

# 2. Perform Iterative Thresholding (Same as Q3)
T = 0.5 * (Id.min() + Id.max())
deltaT = 0.01
done = False

while not done:
    g = Id >= T
    # Handle empty classes to avoid division by zero
    mean_fg = Id[g].mean() if g.any() else 0
    mean_bg = Id[~g].mean() if (~g).any() else 0
    
    Tnext = 0.5 * (mean_fg + mean_bg)
    done = abs(T - Tnext) < deltaT
    T = Tnext

# 3. Create Binary Mask and Remove Background
# Create binary mask (1 for object, 0 for background)
binary_mask = (Id >= T).astype(np.float32)

# Apply mask to original image (Element-wise multiplication)
# Pixels where mask is 0 become 0 (black), others retain original intensity
background_removed = Id * binary_mask

# 4. Display Results
plt.figure(figsize=(15, 5))

# Original
plt.subplot(1, 3, 1)
plt.imshow(Id, cmap="gray")
plt.title("Original")
plt.axis("off")

# Binary Mask
plt.subplot(1, 3, 2)
plt.imshow(binary_mask, cmap="gray")
plt.title(f"Binary (T={T:.3f})")
plt.axis("off")

# Background Removed
plt.subplot(1, 3, 3)
plt.imshow(background_removed, cmap="gray")
plt.title("Background Removed")
plt.axis("off")

plt.tight_layout()
plt.show()