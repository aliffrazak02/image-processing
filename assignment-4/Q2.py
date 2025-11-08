import numpy as np

def imageConv(f, h, mode='zero'):
    # Flip the kernel both horizontally and vertically
    h_flipped = np.flipud(np.fliplr(h))
    
    # Get dimensions
    f_h, f_w = f.shape
    h_h, h_w = h.shape
    pad_h = h_h // 2
    pad_w = h_w // 2

    # Padding
    if mode == 'zero':
        f_padded = np.pad(f, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    elif mode == 'replicate':
        f_padded = np.pad(f, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    else:
        raise ValueError("mode must be 'zero' or 'replicate'")

    # Output matrix
    result = np.zeros_like(f, dtype=float)

    # Perform convolution
    for i in range(f_h):
        for j in range(f_w):
            region = f_padded[i:i+h_h, j:j+h_w]
            result[i, j] = np.sum(region * h_flipped)
    
    return result

A = np.array([
    [5, 8, 3, 4, 6, 2, 3, 7],
    [3, 2, 1, 9, 5, 1, 0, 1],
    [9, 0, 5, 9, 5, 0, 4, 8],
    [4, 7, 2, 7, 9, 0, 6, 9],
    [9, 7, 9, 1, 8, 4, 1, 9],
    [5, 2, 1, 8, 4, 1, 0, 9],
    [8, 1, 9, 5, 4, 9, 3, 8],
    [3, 7, 1, 2, 4, 3, 8, 3]
])

B = np.array([
    [2, 1, 0],
    [1, 1, -1],
    [0, -1, -2]
])

C = imageConv(A, B, mode='zero')
print(np.round(C).astype(int))
