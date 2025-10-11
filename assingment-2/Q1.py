import cv2
import matplotlib.pyplot as plt
import numpy as np

def imArithmetic4e(f1, f2, op):

    # Convert to float for arithmetic precision
    f1 = f1.astype(np.float32)
    f2 = f2.astype(np.float32)

    if op == 'add':
        g = cv2.add(f1, f2)
    elif op == 'sub':
        g = cv2.subtract(f1, f2)
    elif op == 'mul':
        g = cv2.multiply(f1, f2)
    elif op == 'div':
        g = cv2.divide(f1, f2)
    else:
        raise ValueError("op must be 'add', 'sub', 'mul', or 'div'")

    return g


def Mask4e(M, N, rUL, cUL, rLR, cLR):
    mask = np.zeros((M, N), dtype=np.uint8)
    mask[rUL:rLR, cUL:cLR] = 255
    return mask


if __name__ == "__main__":
    # Read grayscale image
    img = cv2.imread('BaboonGray.png', cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("BaboonGray.png not found")

    M, N = img.shape

    # Create rectangular ROI mask
    rUL, cUL, rLR, cLR = 50, 50, 450, 450  # Example coordinates
    mask = Mask4e(M, N, rUL, cUL, rLR, cLR)

    # Apply mask using multiplication
    result = imArithmetic4e(img, mask, 'mul')
    
    # Display results
    plt.figure()
    plt.imshow(result, cmap='gray')
    plt.title('Baboon with ROI Mask Applied')
    plt.axis("off")

    plt.show()
