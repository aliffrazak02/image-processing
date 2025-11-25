import cv2
import numpy as np
import matplotlib.pyplot as plt

def freqz2(h, H, W):
    """
    Computes the frequency response of a 2D filter h.
    Mimics the behavior described for the lecture implementation.
    """
    # Pad the filter to size H x W
    h_padded = np.zeros((H, W))
    h_h, h_w = h.shape
    
    # Center the kernel in the padded array to avoid phase shifts
    # or place at top-left depending on specific lecture implementation.
    # Usually, for magnitude visualization, standard padding is sufficient.
    h_padded[:h_h, :h_w] = h
    
    # Compute 2D FFT
    H_fft = np.fft.fft2(h_padded)
    
    # Shift zero frequency to center
    H_shifted = np.fft.fftshift(H_fft)
    
    return np.abs(H_shifted)

def question_1():
    # Load image (Greyscale)
    # Note: Replace 'cameraman.tif' with your actual image path
    img = cv2.imread('assignment-5/cameraman.tif', 0) 
    if img is None:
        # Create a dummy image if file not found for demonstration
        img = np.zeros((256, 256), dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (200, 200), 255, -1)
    
    rows, cols = img.shape

    # --- (a) Design Spatial Averaging Filter ---
    # Average of 4 neighbors (N, S, W, E) in a 3x3 neighborhood [cite: 6]
    kernel = np.array([
        [0, 1/4, 0],
        [1/4, 0, 1/4],
        [0, 1/4, 0]
    ])

    # --- (b) Frequency Domain Response ---
    # Use freqz2 equivalent to plot filter function [cite: 7]
    H_mag = freqz2(kernel, 64, 64) # Using smaller grid for smoother wireframe plot
    
    # Plotting the Frequency Response (3D Surface)
    fig = plt.figure(figsize=(10, 8))
    
    # 1. Original Image
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(img, cmap='gray')
    ax1.set_title("Original Image")
    ax1.axis('off')

    # 2. Spatial Filtering
    img_spatial = cv2.filter2D(img, -1, kernel)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(img_spatial, cmap='gray')
    ax2.set_title("Filtering in Spatial Domain")
    ax2.axis('off')

    # 3. Filter Frequency Response (3D Plot)
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    X = np.linspace(-0.5, 0.5, 64)
    Y = np.linspace(-0.5, 0.5, 64)
    X, Y = np.meshgrid(X, Y)
    ax3.plot_surface(X, Y, H_mag, cmap='viridis')
    ax3.set_title("Filter Frequency Response")
    ax3.set_xlabel("Fx")
    ax3.set_ylabel("Fy")
    ax3.set_zlabel("Magnitude")

    # --- (c) Frequency Domain Filtering ---
    # Hint: convolution in spatial is multiplication in frequency [cite: 9]
    
    # 1. FFT of Image
    f_img = np.fft.fft2(img)
    
    # 2. FFT of Kernel (Must be padded to image size)
    k_padded = np.zeros((rows, cols))
    kh, kw = kernel.shape
    # Center pad or top-left pad. Standard convolution requires proper anchor handling.
    # Placing kernel center at (0,0) implies wrapping. 
    # We place top-left of kernel at top-left of pad, then roll to center alignment if needed.
    k_padded[:kh, :kw] = kernel
    
    f_kernel = np.fft.fft2(k_padded)
    
    # 3. Multiplication
    f_result = f_img * f_kernel
    
    # 4. Inverse FFT
    img_freq_filtered = np.fft.ifft2(f_result)
    img_freq_filtered = np.abs(img_freq_filtered)
    
    # Crop/Shift to match spatial (spatial filter2D handles boundaries, FFT wraps)
    # Simple shift to align for visualization
    img_freq_filtered = np.roll(img_freq_filtered, -1, axis=0)
    img_freq_filtered = np.roll(img_freq_filtered, -1, axis=1)

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.imshow(img_freq_filtered, cmap='gray')
    ax4.set_title("Filtering in Frequency Domain")
    ax4.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    question_1()