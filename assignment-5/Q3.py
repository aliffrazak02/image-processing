import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage # Only allowed for label function 

def question_3():
    # Load Image (Binary expected, similar to blobs or text)
    img = cv2.imread('assignment-5/bwinmage.png', 0) # Using blobs as example source
    
    if img is None:
        # Dummy data: Two separate rectangles
        img = np.zeros((300, 300), dtype=np.uint8)
        cv2.rectangle(img, (30, 30), (100, 100), 255, -1)
        cv2.rectangle(img, (150, 150), (280, 200), 255, -1)

    # Ensure binary
    _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # 1. Extract connected components using scipy.ndimage.label [cite: 49]
    labeled_array, num_features = ndimage.label(binary_img)
    
    print(f"Number of components found: {num_features}")

    # 2. Display results using different colors [cite: 48]
    # Normalize labels to 0-255 range to apply a colormap
    # Note: This works well for visualization.
    if num_features > 0:
        label_norm = (labeled_array * (255 / num_features)).astype(np.uint8)
    else:
        label_norm = labeled_array.astype(np.uint8)
        
    # Apply 'Jet' or 'Rainbow' colormap. 
    # Background (label 0) needs to remain distinct (usually handled by masking or black background)
    color_labeled = cv2.applyColorMap(label_norm, cv2.COLORMAP_JET)
    
    # Set background (label 0) to black explicitly in the color image
    color_labeled[labeled_array == 0] = [0, 0, 0]

    # Convert BGR to RGB for Matplotlib
    color_labeled_rgb = cv2.cvtColor(color_labeled, cv2.COLOR_BGR2RGB)

    # 3. Overlay cross-shaped symbol on centroid [cite: 48]
    # Since scipy is only for 'label', we calculate centroids using numpy/opencv
    
    # Create a copy for annotation
    annotated_img = color_labeled_rgb.copy()

    for i in range(1, num_features + 1):
        # Create mask for current component
        component_mask = (labeled_array == i).astype(np.uint8) * 255
        
        # Calculate Moments
        M = cv2.moments(component_mask)
        
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
            # Draw Cross 'x' or '+' on the center
            # Using red color (255, 0, 0)
            size = 5
            # Horizontal line of cross
            cv2.line(annotated_img, (cX - size, cY), (cX + size, cY), (255, 0, 0), 2)
            # Vertical line of cross
            cv2.line(annotated_img, (cX, cY - size), (cX, cY + size), (255, 0, 0), 2)

    # --- Displaying Results ---
    plt.figure(figsize=(12, 6))

    # Original
    plt.subplot(1, 3, 1)
    plt.imshow(binary_img, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    # Color Labeled
    plt.subplot(1, 3, 2)
    plt.imshow(color_labeled_rgb)
    plt.title("Color Labeled Image")
    plt.axis('off')

    # Centroid Annotation
    plt.subplot(1, 3, 3)
    plt.imshow(annotated_img)
    plt.title("Centroid Annotation")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    question_3()