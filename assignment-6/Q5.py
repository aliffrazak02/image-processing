import cv2
import numpy as np
import matplotlib.pyplot as plt

# Setup 
I = cv2.imread("assignment-6\coins.png", cv2.IMREAD_GRAYSCALE)
if I is None:
    raise FileNotFoundError("coins.png not found")

Id = I.astype(np.float32) / 255.0

# Iterative Thresholding
T = 0.5 * (Id.min() + Id.max())
deltaT = 0.01
done = False
while not done:
    g = Id >= T
    mean_fg = Id[g].mean() if g.any() else 0
    mean_bg = Id[~g].mean() if (~g).any() else 0
    Tnext = 0.5 * (mean_fg + mean_bg)
    done = abs(T - Tnext) < deltaT
    T = Tnext

# Create initial binary image 
binary_img = (Id >= T).astype(np.uint8) * 255

# Create a copy to modify
filled_mask = binary_img.copy()

# Mouse callback function
def fill_holes(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.floodFill(filled_mask, None, (x, y), 255)
        cv2.circle(display_img, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow("Click to select seed points, then press any key", display_img)

# Prepare display image (convert to BGR so we can see red click markers)
display_img = cv2.cvtColor(filled_mask, cv2.COLOR_GRAY2BGR)

cv2.namedWindow("Click to select seed points, then press any key")
cv2.setMouseCallback("Click to select seed points, then press any key", fill_holes)
cv2.imshow("Click to select seed points, then press any key", display_img)

print("Instruction: Click on the black holes inside the coins to fill them.")
print("Press any key in the image window when finished...")
cv2.waitKey(0)
cv2.destroyAllWindows()

# 3. Final Background Removal
# Normalize the filled mask 
final_mask = filled_mask.astype(np.float32) / 255.0

# Apply new mask to original image
final_result = Id * final_mask

# 4. Display Final Comparison
plt.figure(figsize=(15, 5))

# Original
plt.subplot(1, 3, 1)
plt.imshow(Id, cmap="gray")
plt.title("Original")
plt.axis("off")

# Binary (Filled)
plt.subplot(1, 3, 2)
plt.imshow(final_mask, cmap="gray")
plt.title("Binary (Filled)")
plt.axis("off")

# Background Removed (Improved)
plt.subplot(1, 3, 3)
plt.imshow(final_result, cmap="gray")
plt.title("Background Removed")
plt.axis("off")

plt.tight_layout()
plt.show()