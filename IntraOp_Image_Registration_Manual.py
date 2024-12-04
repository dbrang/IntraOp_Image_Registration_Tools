"""
IntraOp_Image_Registration_Manual.py

Author: David Brang, djbrang@umich.edu
Copyright 2024. University of Michigan. All rights reserved.
License: CC BY-NC 4.0

Purpose:
This script facilitates manual image registration of intraoperative photos through interactive control point selection. Users can define corresponding points on two images to achieve precise alignment.

Dependencies:
- Python 3.7+
- Required libraries: 
  - OpenCV (Install with: `pip install opencv-python`)
  - NumPy (Install with: `pip install numpy`)
  - Matplotlib (for interactive point selection; Install with: `pip install matplotlib`)
  - imageio (for GIF creation): Install with pip install imageio

Inputs: Photo w/o grid, then photo w grid
1. Reference Image: The base image to which the target image will be aligned (e.g., `reference_image.png`). Usually without grid.
2. Target Image: The image to be aligned to the reference image (e.g., `target_image.png`). Usually with grid.

Processes:
Control points selected interactively through the GUI.

Outputs:
- Registered Image: The aligned target image saved as a file (e.g., `registered_image.png`).
- (Optional) Visualization of alignment and control points.

Usage:
1. Install the required libraries.
2. Place the input images in the working directory.
3. Run the script: `python IntraOp_Image_Registration_Manual.py reference_image.png target_image.png`.
"""

import matplotlib.pyplot as plt
import argparse
import cv2
import numpy as np
import imageio
import os
from matplotlib.widgets import Cursor  # Importing Cursor widget
from matplotlib.widgets import Button  # Importing Button widget

# Global variables to store clicked points and line references for both images
points_img1 = []
points_img2 = []
lines_img1 = []
lines_img2 = []
point_history = []  # To keep track of points added
fig = None

# Function to handle mouse clicks on the first image
def onclick_img1(event):
    if event.inaxes == ax1:
        x, y = int(event.xdata), int(event.ydata)
        points_img1.append((x, y))
        point_history.append(('img1', (x, y)))
        # Plot the point and store the Line2D object
        line, = ax1.plot(x, y, 'bo', markersize=4, markerfacecolor='none')  # Note the comma to unpack
        lines_img1.append(line)
        fig.canvas.draw()

# Function to handle mouse clicks on the second image
def onclick_img2(event):
    if event.inaxes == ax2:
        x, y = int(event.xdata), int(event.ydata)
        points_img2.append((x, y))
        point_history.append(('img2', (x, y)))
        # Plot the point and store the Line2D object
        line, = ax2.plot(x, y, 'bo', markersize=4, markerfacecolor='none')
        lines_img2.append(line)
        fig.canvas.draw()

# Function to display images, capture clicks, and provide instructions
def manual_point_selection(img1, img2):
    global fig, ax1, ax2

    # Get the dimensions of the images
    height1, width1 = img1.shape[:2]
    height2, width2 = img2.shape[:2]
    max_height = max(height1, height2)
    total_width = width1 + width2

    # Decide the figure size based on image dimensions to make images as large as possible
    dpi = 100  # Dots per inch
    fig_width = total_width / dpi
    fig_height = max_height / dpi

    fig = plt.figure(figsize=(fig_width, fig_height + 50 / dpi), dpi=dpi)  # Add extra space for the button and title

    # Maximize the figure window
    manager = plt.get_current_fig_manager()
    try:
        manager.window.showMaximized()
    except AttributeError:
        try:
            manager.frame.Maximize(True)
        except AttributeError:
            pass  # Could not maximize the window

    # Set the height fractions for button and images
    button_height = 50 / (fig_height * dpi)  # Convert pixels to fraction
    image_height = 1 - button_height - 0.05  # 0.05 for title

    # Add instructions back in above the images
    fig.suptitle('Click corresponding points on both images. Close the window when done.', fontsize=14, color='blue')

    # Create axes for the two images without any margins
    ax1 = fig.add_axes([0, button_height, width1 / total_width, image_height])
    ax2 = fig.add_axes([width1 / total_width, button_height, width2 / total_width, image_height])

    # Display images
    ax1.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    ax1.axis('off')  # Remove axes
    ax2.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    ax2.axis('off')  # Remove axes

    # Add crosshair cursor to both images
    cursor1 = Cursor(ax1, useblit=True, color='blue', linewidth=1)
    cursor2 = Cursor(ax2, useblit=True, color='blue', linewidth=1)

    # Connect the click events
    fig.canvas.mpl_connect('button_press_event', onclick_img1)
    fig.canvas.mpl_connect('button_press_event', onclick_img2)

    # Add Undo Button
    # Define the undo function
    def undo_last_point(event):
        """
        Undo the last point clicked on either image.
        Removes the last point from the appropriate list,
        and updates the plot accordingly.
        """
        if point_history:
            last_image, last_point = point_history.pop()
            if last_image == 'img1':
                points_img1.pop()
                last_line = lines_img1.pop()
                last_line.remove()
            elif last_image == 'img2':
                points_img2.pop()
                last_line = lines_img2.pop()
                last_line.remove()
            fig.canvas.draw()
        else:
            print("No points to undo")

    # Add the Undo button to the figure below the images
    button_ax = fig.add_axes([0.45, 0, 0.1, button_height])
    undo_button = Button(button_ax, 'Undo Last Point')
    undo_button.on_clicked(undo_last_point)

    plt.show()

    return points_img1, points_img2

# Function to apply homography transformation and return the registered image
def register_images(img1, img2, points_img1, points_img2):
    # Ensure the points are in the right format
    pts1 = np.float32(points_img1)
    pts2 = np.float32(points_img2)
    
    # Find the homography matrix based on manual points
    h_matrix, status = cv2.findHomography(pts1, pts2, 0)
    
    # Warp the first image to align with the second one using the homography matrix
    height, width, channels = img2.shape
    registered_img = cv2.warpPerspective(img1, h_matrix, (width, height))
    
    return registered_img

# Function to draw matches vertically using manual points
def drawMatches_vertical(img1, img2, pts1, pts2):
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]
    max_cols = max([cols1, cols2])
    
    out = np.zeros((rows1 + rows2, max_cols, 3), dtype='uint8')
    out[:rows1, :cols1] = img1
    out[rows1:rows1 + rows2, :cols2] = img2

    for pt1, pt2 in zip(pts1, pts2):
        x1, y1 = pt1
        x2, y2 = pt2

        # Adjust y2 coordinate because img2 is below img1
        y2 += rows1

        # Draw circles and lines between matched points
        cv2.circle(out, (int(x1), int(y1)), 4, (0, 255, 0), 1)
        cv2.circle(out, (int(x2), int(y2)), 4, (0, 255, 0), 1)
        cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)

    return out

# Function to create transition GIF between two images
def create_transition_gif(img1, img2, output_path, num_frames=10):
    frames = []
    alpha_values = np.linspace(0, 1, num_frames)
    for alpha in alpha_values:
        blended = cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)
        frames.append(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for imageio
    imageio.mimsave(output_path, frames, duration=0.1)

if __name__ == "__main__":
    # Parsing command-line arguments for image paths
    parser = argparse.ArgumentParser(description="Manual landmark selection for image registration.")
    parser.add_argument("image1", help="Path to the first image.")
    parser.add_argument("image2", help="Path to the second image.")
    
    args = parser.parse_args()

    # Load the images
    img1 = cv2.imread(args.image1)
    img2 = cv2.imread(args.image2)

    # Check if images are loaded properly
    if img1 is None:
        print(f"Error loading image 1 from path: {args.image1}")
        exit(1)
    if img2 is None:
        print(f"Error loading image 2 from path: {args.image2}")
        exit(1)

    # Call the manual point selection function
    points_img1, points_img2 = manual_point_selection(img1, img2)
    
    # Register the images using the manually selected points
    if len(points_img1) >= 4 and len(points_img2) >= 4:
        registered_img = register_images(img1, img2, points_img1, points_img2)

        # Draw the matches between the images
        matches_img = drawMatches_vertical(img1, img2, points_img1, points_img2)
        
        # Save the registered image and matches image
        common_dir = os.path.commonpath([args.image1, args.image2])
        output_dir = os.path.join(common_dir, 'Output')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save input image paths for ease of rerunning
        with open(os.path.join(output_dir, 'input_images.txt'), 'w') as f:
            f.write(f"Image 1: {args.image1}\n")
            f.write(f"Image 2: {args.image2}\n")
        
        # Save input images and output images
        cv2.imwrite(os.path.join(output_dir, 'NoGrid.jpg'), img1)
        cv2.imwrite(os.path.join(output_dir, 'Grid.jpg'), img2)
        cv2.imwrite(os.path.join(output_dir, 'RegisteredImage.jpg'), registered_img)
        cv2.imwrite(os.path.join(output_dir, 'Matches.jpg'), matches_img)

        # Create and save the transition GIF using img2 and registered_img
        gif_output_path = os.path.join(output_dir, 'transition.gif')
        create_transition_gif(img2, registered_img, gif_output_path)

        print(f"Output saved in {output_dir}")
    else:
        print("You need at least 4 points in each image for registration.")
