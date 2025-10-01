import cv2
import matplotlib.pyplot as plt
import numpy as np

def scanLine4e(f, l, loc):
    if loc == 'row':
        return f[l, :]
    elif loc == 'col':
        return f[:, l]
    else:
        raise ValueError("loc must be 'row' or 'col'")
    return 0

if __name__ == "__main__":
    # Read the image
    img = cv2.imread('Baboon.png')

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Get the middle row
    mid_row = gray.shape[0] // 2
    scan_line = scanLine4e(gray, mid_row, 'row')

    # Plot scan line
    plt.figure()
    plt.plot(scan_line)
    plt.xlabel('Pixel Index')
    plt.ylabel('Intensity')
    plt.title('Scan Line Plot')
    plt. show()
