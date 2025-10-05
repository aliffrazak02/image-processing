import cv2
import matplotlib.pyplot as plt
import numpy as np

def brightnessCorr(f,percent,op):
    if op == 'brighten':
        result = f.astype(np.float32) * (1 + percent)  # increase brightness
    elif op == 'darken':
        result = f.astype(np.float32) * (1 - percent)  # decrease brightness
    else:
        raise ValueError("op must be 'brighten' or 'darken'")
    
    # Clip values to stay in [0, 255]
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result

def insert(f1,f2,x,y):
    return 0
    
if __name__ == "__main__":
    # Read the image
    img1 = cv2.imread('cameraman.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('small_ubc_logo.jpg', cv2.IMREAD_GRAYSCALE)

    # Brighten the image by 30%
    brightened_img = brightnessCorr(img1, 0.3, 'brighten')

    # Display the original, brightened, and darkened images

    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(img1, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(brightened_img, cmap='gray')
    plt.title('Brightened Image')
    plt.axis('off')
    plt.show()
    
    # Insert img2 into img1 at position (10, 10)
    inserted_img = insert(img1, img2, 10, 10)
    plt.figure()
    plt.imshow(inserted_img, cmap='gray')
    plt.title('Image with Inserted Logo')
    plt.axis('off')
    plt.show()
    
    