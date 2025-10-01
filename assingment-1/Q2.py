import numpy as np
import cv2
import matplotlib.pyplot as plt


def Mask4e(M: int, N: int, rUL: int, cUL: int, rLR: int, cLR: int):
    # Initialize mask with 0s
    mask = np.zeros((M, N), dtype=np.uint8)

    # Check if rectangle is within bounds
    if (rUL < 0 or cUL < 0 or rLR >= M or cLR >= N):
        raise ValueError("Rectangle exceeds mask dimensions")

    # Fill rectangle region with 1s
    mask[rUL:rLR+1, cUL:cLR+1] = 1

    return mask

if __name__ == "__main__":
    # Read grayscale image
    img = cv2.imread("BaboonGray.png", cv2.IMREAD_GRAYSCALE)
    M, N = img.shape  # rows (height), cols (width)

    # Side length of square = half of image size (both directions)
    side = min(M, N) // 2  

    # Compute top-left and bottom-right corners (centered square)
    rUL = (M - side) // 2
    cUL = (N - side) // 2
    rLR = rUL + side - 1
    cLR = cUL + side - 1

    # Generate mask
    mask = Mask4e(M, N, rUL, cUL, rLR, cLR)
    
    num_ones = np.sum(mask)   # since mask only has 0/1
    expected = side * side

    print("Number of 1s in mask:", num_ones)
    print("Expected number of 1s:", expected)
    
    # Apply mask (multiply element-wise)
    masked_img = img * mask

    # Display
    plt.imshow(masked_img, cmap='gray')
    plt.axis("off")

    plt.show()
