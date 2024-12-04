"""
IntraOp_Image_Registration_Automatic.py

Author: David Brang, djbrang@umich.edu
Copyright 2024. University of Michigan. All rights reserved.
License: CC BY-NC 4.0

Purpose:
This script performs automatic image registration of two input intraoperative images using pre-defined control points. The goal is to align the images for enhanced visualization and analysis in medical applications.

Dependencies:
- Python 3.7+
- Required libraries: 
  - OpenCV (Install with: `pip install opencv-python`)
  - NumPy (Install with: `pip install numpy`)
  - Matplotlib (optional, for visualization; Install with: `pip install matplotlib`)
  - imageio (for GIF creation): Install with pip install imageio

Inputs: Photo w/o grid, then photo w grid
1. Reference Image: The base image to which the target image will be aligned (e.g., `reference_image.png`). Usually without grid.
2. Target Image: The image to be aligned to the reference image (e.g., `target_image.png`). Usually with grid.

Outputs: Registration files from two methods (SIFT + FLANN and ORB + FLANN)
- Registered Images: The aligned target image saved as a file (e.g., `aligned_sift_flann.jpg`).
- Control point matches (e.g. 'matches_sift_flann.jpg')
- Transition between target and registered (e.g., `transition_orb_flann.gif`)

Usage:
1. Install the required libraries.
2. Place the input images in the working directory.
3. Run the script: `python IntraOp_Image_Registration_Automatic.py reference_image.png target_image.png`.
"""

import cv2
import numpy as np
import os
import argparse
import imageio

def preprocess_image(img):
    # Apply histogram equalization
    img_eq = cv2.equalizeHist(img)
    # Apply Gaussian Blur to reduce noise
    img_blur = cv2.GaussianBlur(img_eq, (5,5), 0)
    return img_blur

def align_images_feature_based(img1, img2, method, matcher):
    # Preprocess images
    img1_prep = preprocess_image(img1)
    img2_prep = preprocess_image(img2)
    
    # Detect keypoints and descriptors
    keypoints1, descriptors1 = method.detectAndCompute(img1_prep, None)
    keypoints2, descriptors2 = method.detectAndCompute(img2_prep, None)
    
    # Check if descriptors are None
    if descriptors1 is None or descriptors2 is None:
        print("Could not find descriptors in one of the images.")
        return None, None, None, None, None

    # Ensure descriptors are in the correct data type
    if descriptors1.dtype != np.float32:
        descriptors1 = descriptors1.astype(np.float32)
    if descriptors2.dtype != np.float32:
        descriptors2 = descriptors2.astype(np.float32)
    
    # Match descriptors
    if matcher == "FLANN":
        # Adjust parameters based on descriptor type
        if descriptors1.dtype.type == np.float32:
            # SIFT descriptors are float32
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            # ORB, BRISK, AKAZE descriptors are uint8
            FLANN_INDEX_LSH = 6
            index_params= dict(algorithm=FLANN_INDEX_LSH,
                               table_number=6,
                               key_size=12,
                               multi_probe_level=1)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)
        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
    elif matcher == "BF":
        normType = method.defaultNorm()
        bf = cv2.BFMatcher(normType, crossCheck=False)
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)
        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
    
    # Extract location of good matches
    if len(good_matches) > 10:
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Find homography
        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        
        height, width = img1.shape
        aligned_img = cv2.warpPerspective(img2_color, H, (width, height))
        
        return aligned_img, H, good_matches, keypoints1, keypoints2
    else:
        print(f"Not enough matches are found - {len(good_matches)}/10")
        return None, None, None, None, None

def drawMatches_vertical(img1, kp1, img2, kp2, matches, matchesMask=None, flags=2):
    # Stack images vertically
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]
    max_cols = max([cols1, cols2])
    out = np.zeros((rows1 + rows2, max_cols, 3), dtype='uint8')
    out[:rows1, :cols1] = img1
    out[rows1:rows1 + rows2, :cols2] = img2

    # Draw the matches
    for i, match in enumerate(matches):
        # Get the matching keypoints for each of the images
        img1_idx = match.queryIdx
        img2_idx = match.trainIdx

        # x - columns
        # y - rows
        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt

        # Adjust y2 coordinate because img2 is below img1
        y2 += rows1

        # Draw a small circle at both co-ordinates
        cv2.circle(out, (int(x1), int(y1)), 4, (0, 255, 0), 1)
        cv2.circle(out, (int(x2), int(y2)), 4, (0, 255, 0), 1)

        # Draw a line in between the two points
        if matchesMask is None or matchesMask[i]:
            cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)

    return out

def create_transition_gif(img1, img2, output_path, num_frames=10):
    frames = []
    alpha_values = np.linspace(0, 1, num_frames)
    for alpha in alpha_values:
        blended = cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)
        frames.append(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for imageio
    imageio.mimsave(output_path, frames, duration=0.1)

if __name__ == "__main__":
    import sys

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Align two images using feature matching.')
    parser.add_argument('image1', type=str, help='Path to the first image.')
    parser.add_argument('image2', type=str, help='Path to the second image.')
    args = parser.parse_args()

    image1_path = args.image1
    image2_path = args.image2

    # Load the two images in color and grayscale
    img1_color = cv2.imread(image1_path)
    img2_color = cv2.imread(image2_path)
    if img1_color is None:
        print(f"Error loading image1 from {image1_path}")
        sys.exit(1)
    if img2_color is None:
        print(f"Error loading image2 from {image2_path}")
        sys.exit(1)
    img1_gray = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)

    # Output folder to store results
    common_dir = os.path.commonpath([image1_path, image2_path])
    output_dir = os.path.join(common_dir, 'Output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save input image paths for ease of rerunning
    with open(os.path.join(output_dir, 'input_images.txt'), 'w') as f:
        f.write(f"Image 1: {image1_path}\n")
        f.write(f"Image 2: {image2_path}\n")

    # **Save the two original images into the 'Output' folder**
    cv2.imwrite(os.path.join(output_dir, 'NoGrid.jpg'), img1_color)
    cv2.imwrite(os.path.join(output_dir, 'Grid.jpg'), img2_color)

    # 1. SIFT + FLANN
    print("Running SIFT + FLANN...")
    sift = cv2.SIFT_create()
    aligned_sift_flann, H_sift_flann, matches_sift_flann, kp1_sift_flann, kp2_sift_flann = align_images_feature_based(img1_gray, img2_gray, sift, "FLANN")
    if aligned_sift_flann is not None:
        cv2.imwrite(os.path.join(output_dir, 'aligned_sift_flann.jpg'), aligned_sift_flann)
        # Draw matches vertically
        img_matches = drawMatches_vertical(img1_color, kp1_sift_flann, img2_color, kp2_sift_flann, matches_sift_flann)
        cv2.imwrite(os.path.join(output_dir, 'matches_sift_flann.jpg'), img_matches)
        # Create a GIF transitioning between img1_color and aligned_sift_flann
        create_transition_gif(img1_color, aligned_sift_flann, os.path.join(output_dir, 'transition_sift_flann.gif'))
    else:
        print("SIFT + FLANN alignment failed.")

    # 2. ORB + FLANN
    print("Running ORB + FLANN...")
    orb = cv2.ORB_create(nfeatures=5000)
    aligned_orb_flann, H_orb_flann, matches_orb_flann, kp1_orb_flann, kp2_orb_flann = align_images_feature_based(img1_gray, img2_gray, orb, "FLANN")
    if aligned_orb_flann is not None:
        cv2.imwrite(os.path.join(output_dir, 'aligned_orb_flann.jpg'), aligned_orb_flann)
        img_matches = drawMatches_vertical(img1_color, kp1_orb_flann, img2_color, kp2_orb_flann, matches_orb_flann)
        cv2.imwrite(os.path.join(output_dir, 'matches_orb_flann.jpg'), img_matches)
        # Create a GIF transitioning between img1_color and aligned_orb_flann
        create_transition_gif(img1_color, aligned_orb_flann, os.path.join(output_dir, 'transition_orb_flann.gif'))
    else:
        print("ORB + FLANN alignment failed.")

    print(f"Results saved to: {output_dir}")
