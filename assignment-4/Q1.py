import argparse
import cv2
import numpy as np


def equalize_roi(image, roi):
    """Apply histogram equalization to a selected region."""
    x, y, w, h = roi
    result = image.copy()

    # Extract region
    region = image[y : y + h, x : x + w]

    # Convert to YCrCb and equalize Y channel only
    ycrcb = cv2.cvtColor(region, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])  # Equalize Y channel
    equalized_region = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    # Put back into result
    result[y : y + h, x : x + w] = equalized_region
    return result


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="ROI histogram equalization")
    parser.add_argument("--image", "-i", required=True, help="Input image path")
    args = parser.parse_args()

    # Load image
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Cannot read {args.image}")
        return

    print("Select ROI with mouse, press ENTER to confirm, ESC to cancel")

    while True:
        # Select ROI
        roi = cv2.selectROI("Select Region", image, showCrosshair=True)
        cv2.destroyWindow("Select Region")

        if roi[2] == 0 or roi[3] == 0:  # No selection
            break

        # Apply equalization
        result = equalize_roi(image, roi)

        # Show comparison
        comparison = np.hstack([image, result])
        cv2.imshow(
            "Original | Equalized - Press 's' to save, 'r' to reselect, 'q' to quit",
            comparison,
        )

        # Handle user input
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord("s"):
                output_path = args.image.replace(".", "_equalized.")
                cv2.imwrite(output_path, result)
                print(f"Saved: {output_path}")
            elif key == ord("r"):
                cv2.destroyAllWindows()
                break
            elif key == ord("q") or key == 27:
                cv2.destroyAllWindows()
                return


if __name__ == "__main__":
    main()