import cv2
import numpy as np
import matplotlib.pyplot as plt

def question_2():
    # Load image
    # Requirement: use 'blobs.png' [cite: 37]
    img = cv2.imread('assignment-5/blob.png', 0)
    
    if img is None:
        print("blobs.png not found. Generating dummy binary image.")
        img = np.zeros((300, 300), dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (150, 150), 255, -1)
        cv2.circle(img, (200, 200), 40, 255, -1)

    # Ensure binary
    _, A = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Define Structuring Element B: 3x3 with all coefficients equal to 1 [cite: 37]
    B = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    
    # Since B is symmetric (rectangle of ones), B_hat (reflection) is equal to B.
    
    # --- Equation (1): (A dilate B)^c = A^c erode B_hat ---
    # LHS: Complement of Dilation
    dilated = cv2.dilate(A, B)
    lhs_1 = cv2.bitwise_not(dilated)
    
    # RHS: Erosion of Complement
    A_comp = cv2.bitwise_not(A)
    rhs_1 = cv2.erode(A_comp, B) # B is B_hat here

    # Verify Equality
    diff_1 = cv2.absdiff(lhs_1, rhs_1)
    print(f"Equation 1 Max Difference: {np.max(diff_1)} (Should be 0)")

    # --- Equation (2): (A erode B)^c = A^c dilate B_hat ---
    # LHS: Complement of Erosion
    eroded = cv2.erode(A, B)
    lhs_2 = cv2.bitwise_not(eroded)
    
    # RHS: Dilation of Complement
    rhs_2 = cv2.dilate(A_comp, B) # B is B_hat here

    # Verify Equality
    diff_2 = cv2.absdiff(lhs_2, rhs_2)
    print(f"Equation 2 Max Difference: {np.max(diff_2)} (Should be 0)")

    # --- Plotting Results to match Figure in PDF [cite: 40-44] ---
    plt.figure(figsize=(12, 10))
    
    # Equation 1 Plots
    plt.subplot(2, 2, 1)
    plt.imshow(lhs_1, cmap='gray')
    plt.title("Eq 1 LHS: Complement of Dilated")
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(rhs_1, cmap='gray')
    plt.title("Eq 1 RHS: Erosion of Complement")
    plt.axis('off')

    # Equation 2 Plots
    plt.subplot(2, 2, 3)
    plt.imshow(lhs_2, cmap='gray')
    plt.title("Eq 2 LHS: Complement of Eroded")
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(rhs_2, cmap='gray')
    plt.title("Eq 2 RHS: Dilation of Complement")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    question_2()