import cv2
import matplotlib.pyplot as plt
import numpy as np

def insert(f1, f2, x, y):
    result = f1.copy()

    # Get the height and width of the smaller image (logo)
    h, w = f2.shape

    # Define the region of interest (ROI) in the large image
    roi = result[y:y+h, x:x+w]

    # Add the logo to region of interest
    inserted = cv2.add(roi, f2)

    # Replace that region with the inserted image
    result[y:y+h, x:x+w] = inserted

    return result

if __name__ == "__main__":
    img1 = cv2.imread('cameraman.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('small_ubc_logo.jpg', cv2.IMREAD_GRAYSCALE)

    # Insert the image 
    inserted_img = insert(img1, img2, 5, 5)

    # Display both images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img1, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(inserted_img, cmap='gray')
    plt.title('Modified Image')
    plt.axis('off')

    plt.show()
