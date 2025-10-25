import cv2
import matplotlib.pyplot as plt
import numpy as np

# f: gray scale image, 
# cx: positive scaling factors along the 𝑥 direction, 
# cy: positive scaling factors along the 𝑦 direction

#(a)
def imageScaling4e(f,cx,cy):
    h,w = f.shape
    new_h = int(h * cy)
    new_w = int(w * cx)
    scaled = np.zeros((new_h, new_w), dtype=np.uint8)
    
    for y_new in range(new_h):
        for x_new in range(new_w):
            # inverse mapping
            x_old = int(round(x_new / cx))
            y_old = int(round(y_new / cy))
            
            # boundary check
            if 0 <= x_old < w and 0 <= y_old < h:
                scaled[y_new, x_new] = f[y_old, x_old]

    return scaled

#(b)
def imageRotate4e(f,theta,mode='full'):
        h, w = f.shape
        rad = np.deg2rad(theta)
        cos_t, sin_t = np.cos(rad), np.sin(rad)

        if mode == 'full':
            new_w = int(abs(w * cos_t) + abs(h * sin_t))
            new_h = int(abs(h * cos_t) + abs(w * sin_t))
        else:
            new_w, new_h = w, h

        rotated = np.zeros((new_h, new_w), dtype=np.uint8)

        # centers
        cx, cy = w / 2, h / 2
        cx_new, cy_new = new_w / 2, new_h / 2

        for y_new in range(new_h):
            for x_new in range(new_w):
                # shift to center
                x_shifted = x_new - cx_new
                y_shifted = y_new - cy_new

                # rotation
                x_old =  cos_t * x_shifted - sin_t * y_shifted + cx
                y_old =  sin_t * x_shifted + cos_t * y_shifted + cy

                # nearest neighbor interpolation
                x_old = int(round(x_old))
                y_old = int(round(y_old))

                if 0 <= x_old < w and 0 <= y_old < h:
                    rotated[y_new, x_new] = f[y_old, x_old]

        return rotated

if __name__ == "__main__":
    # (a)
    f = cv2.imread('BaboonGray.png', cv2.IMREAD_GRAYSCALE)
    scaled = imageScaling4e(f, 2.0, 0.5)
    plt.imshow(scaled, cmap='gray')
    plt.title('Scaled Image')
    plt.axis('off')
    plt.show()
    
    # (b)
    f = cv2.imread('BaboonGray.png', cv2.IMREAD_GRAYSCALE)
    rot_crop = imageRotate4e(f, 45, 'crop')
    rot_full = imageRotate4e(f, 45, 'full')

    plt.subplot(1,2,1)
    plt.imshow(rot_crop, cmap='gray')
    plt.title('Rotated Image (Crop Mode)')
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(rot_full, cmap='gray')
    plt.title('Rotated Image (Full Mode)')   
    plt.axis('off')
    plt.show()
