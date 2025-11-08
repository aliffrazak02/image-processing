#!/usr/bin/env python3
"""
region_hist_eq.py

Interactive region-based histogram equalization.

Usage:
    python region_hist_eq.py --image path/to/image.jpg

How it works:
- The script loads the image and opens a window.
- Use the mouse to select a rectangular ROI (drag, then release).
  * OpenCV's cv2.selectROI is used for a simple interactive selection.
- After selection the histogram equalization is applied only to the ROI.
  * For grayscale images: cv2.equalizeHist is applied directly.
  * For color images: the ROI is converted to YCrCb, equalizeHist is applied to Y channel only, then converted back.
- The equalized result is shown side-by-side with the original.
- Press:
    - 's' to save the result (appends "_roi_eq" to the input filename),
    - 'r' to re-select ROI and re-apply,
    - 'q' or ESC to quit.

Dependencies:
    pip install opencv-python numpy
"""
import argparse
import os
import sys

import cv2
import numpy as np


def equalize_roi_color(img, roi):
    """
    Apply histogram equalization to ROI of a color image using YCrCb.

    img: input color image (BGR)
    roi: tuple (x, y, w, h)
    """
    x, y, w, h = roi
    patch = img[y : y + h, x : x + w]

    # Convert patch to YCrCb and equalize Y channel only
    ycrcb = cv2.cvtColor(patch, cv2.COLOR_BGR2YCrCb)
    y_channel, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y_channel)
    ycrcb_eq = cv2.merge((y_eq, cr, cb))
    patch_eq = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)

    out = img.copy()
    out[y : y + h, x : x + w] = patch_eq
    return out


def equalize_roi_gray(img_gray, roi):
    """
    Apply histogram equalization to ROI of a grayscale image.

    img_gray: single-channel image
    roi: (x,y,w,h)
    """
    x, y, w, h = roi
    patch = img_gray[y : y + h, x : x + w]
    patch_eq = cv2.equalizeHist(patch)
    out = img_gray.copy()
    out[y : y + h, x : x + w] = patch_eq
    return out


def draw_border(img, roi, color=(0, 255, 0), thickness=2):
    x, y, w, h = roi
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)


def parse_args():
    p = argparse.ArgumentParser(description="Region-based histogram equalization (interactive ROI).")
    p.add_argument("--image", "-i", required=True, help="Path to input image")
    return p.parse_args()


def main():
    args = parse_args()
    image_path = args.image

    if not os.path.isfile(image_path):
        print(f"ERROR: File not found: {image_path}")
        sys.exit(1)

    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        print(f"ERROR: Cannot read image: {image_path}")
        sys.exit(1)

    # Keep original copy
    orig = img_bgr.copy()

    # Determine if grayscale or color (we'll treat read color as color;
    # also provide a grayscale mode for single-channel)
    img_gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)

    window_name = "Select ROI - press ENTER/SPACE when done, ESC to cancel"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, orig)
    cv2.waitKey(1)

    while True:
        print("\n-- Instructions --")
        print(" * Select ROI with the mouse (drag).")
        print(" * Press ENTER or SPACE to confirm selection.")
        print(" * Press ESC or 'c' to cancel selection and quit.")
        roi = cv2.selectROI(window_name, orig, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow(window_name)

        x, y, w, h = roi
        if w == 0 or h == 0:
            print("No ROI selected (w or h is zero). Exiting.")
            break

        print(f"Selected ROI: x={x}, y={y}, w={w}, h={h}")

        # Decide whether to treat image as color or grayscale (we keep color if read as BGR)
        is_color = len(orig.shape) == 3 and orig.shape[2] == 3

        if is_color:
            out = equalize_roi_color(orig, roi)
        else:
            # Not expected since we read color, but keep for completeness
            out_gray = equalize_roi_gray(img_gray, roi)
            out = cv2.cvtColor(out_gray, cv2.COLOR_GRAY2BGR)

        # Draw ROI border on copies to show selection
        vis_orig = orig.copy()
        draw_border(vis_orig, roi, color=(0, 255, 0), thickness=2)
        vis_out = out.copy()
        draw_border(vis_out, roi, color=(0, 255, 0), thickness=2)

        # Stack side-by-side for comparison
        h_max = max(vis_orig.shape[0], vis_out.shape[0])
        w_sum = vis_orig.shape[1] + vis_out.shape[1]
        comparison = np.zeros((h_max, w_sum, 3), dtype=np.uint8)
        comparison[: vis_orig.shape[0], : vis_orig.shape[1]] = vis_orig
        comparison[: vis_out.shape[0], vis_orig.shape[1] : vis_orig.shape[1] + vis_out.shape[1]] = vis_out

        win = "Original (left)  |  ROI-equalized (right)  -- press 's' to save, 'r' to reselect, 'q' to quit"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.imshow(win, comparison)

        while True:
            k = cv2.waitKey(0) & 0xFF
            if k == ord("s"):
                # Save result
                base, ext = os.path.splitext(image_path)
                out_path = base + "_roi_eq" + ext
                cv2.imwrite(out_path, out)
                print(f"Saved equalized image: {out_path}")
            elif k == ord("r"):
                # reselect ROI: close window and break to outer loop
                cv2.destroyWindow(win)
                break
            elif k == ord("q") or k == 27:
                # quit
                cv2.destroyAllWindows()
                print("Quitting.")
                return
            else:
                print("Keys: 's' save, 'r' reselect ROI, 'q' or ESC quit")

        # outer loop continues to reselect ROI

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()