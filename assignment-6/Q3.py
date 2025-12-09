import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the image
I = cv2.imread("assignment-6\coins.png", cv2.IMREAD_GRAYSCALE)

# Handle case where image is not found
if I is None:
    print("Error: 'coins.png' not found. Please ensure the image is in the correct path.")
else:
    # Normalize image to range [0, 1]
    Id = I.astype(np.float32) / 255.0

    # 2. Initialize Thresholding Variables
    # Initial estimate
    T = 0.5 * (Id.min() + Id.max())
    
    deltaT = 0.01
    done = False
    
    # track iterations and history
    iteration_count = 0
    T_history = [T]  # Store initial T

    # 3. Iterative Threshold Selection Loop
    while not done:
        iteration_count += 1
        
        # Segmentation: g is mask for object
        g = Id >= T
        
        # Calculate new threshold 
        mean_fg = Id[g].mean() if g.any() else 0
        mean_bg = Id[~g].mean() if (~g).any() else 0
        
        Tnext = 0.5 * (mean_fg + mean_bg)
        
        # Check for convergence
        done = abs(T - Tnext) < deltaT
        
        # Update T
        T = Tnext
        
        # NEW: Store the new T in history
        T_history.append(T)

    # 4. Generate Final Binary Image
    binary = (Id >= T).astype(np.uint8) * 255

    # Print required statistics to console (matching assignment screenshot style)
    print(f"Converged in {iteration_count} iterations, final T = {T:.4f}")
    print(f"Threshold history: {T_history}")

    # 5. Display Results in a Single Figure
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(I, cmap="gray")
    plt.title("Original")
    plt.axis("off")
    
    # Subplot 2: Thresholded Image
    plt.subplot(1, 2, 2)
    plt.imshow(binary, cmap="gray")
    plt.title(f"Iterative Thresholding (T={T:.3f})")
    plt.axis("off")
    
    # Add a suptitle to mimic the output description found in the PDF example
    plt.suptitle(f"Converged in {iteration_count} iterations. Final T={T:.3f}", fontsize=12)
    
    plt.tight_layout()
    plt.show()