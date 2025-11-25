import cv2
import numpy as np
import matplotlib.pyplot as plt

def complement(img):
    """Return binary complement: 0↔255."""
    return 255 - img

def reflect_se(struct_elem):
    """Reflect structuring element B (flip horizontally & vertically)."""
    return np.flipud(np.fliplr(struct_elem))

A = cv2.imread("assignment-5\\blob.png", cv2.IMREAD_GRAYSCALE)
_, A = cv2.threshold(A, 127, 255, cv2.THRESH_BINARY)

# Structuring element B (3×3 of ones)
B = np.ones((3, 3), dtype=np.uint8)
B_hat = reflect_se(B)

# Identity (1):  (A ⊖ B)^c  =  A^c ⊕ B̂
erosion_A = cv2.erode(A, B, iterations=1)
lhs1 = complement(erosion_A)

rhs1 = cv2.dilate(complement(A), B_hat, iterations=1)

# Identity (2):  A ⊕ B  =  (A^c ⊖ B̂)^c
dilation_A = cv2.dilate(A, B, iterations=1)
lhs2 = dilation_A

rhs2 = complement(cv2.erode(complement(A), B_hat, iterations=1))


# Display Results

plt.figure(figsize=(12, 10))

# Identity (1) 
plt.subplot(2, 2, 1)
plt.imshow(lhs1, cmap="gray")
plt.title("Complement of Eroded")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(rhs1, cmap="gray")
plt.title("Dilation of Complement")
plt.axis("off")

# Identity (2) 
plt.subplot(2, 2, 3)
plt.imshow(lhs2, cmap="gray")
plt.title("Dilation of Original")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(rhs2, cmap="gray")
plt.title("Complement of Eroded Complement")
plt.axis("off")

plt.tight_layout()
plt.show()
