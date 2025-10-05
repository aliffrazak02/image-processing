import cv2
import matplotlib.pyplot as plt
import numpy as np

def imArtihmetic4e(f1, f2, op):
    if op == 'add':
        return cv2.add(f1, f2)
    elif op == 'sub':
        return cv2.subtract(f1, f2)
    elif op == 'mul':
        return cv2.multiply(f1, f2)
    elif op == 'div':
        return cv2.divide(f1, f2)
    else:
        raise ValueError("op must be 'add', 'sub', 'mul', or 'div'")    

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
    # Read the image
    img = cv2.imread('BaboonGray.png', cv2.IMREAD_GRAYSCALE)

    # Process the image to create a mask

    
    
    # Display the grayscale image
    plt.figure()
    plt.imshow(plt.gray, cmap='gray')
    plt.title('Grayscale Image')
    plt.axis('off')
    plt.show()