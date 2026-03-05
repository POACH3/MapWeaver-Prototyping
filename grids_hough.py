"""
grids_hough.py

NOTES:
    - import image
    - image processing (grayscale, gaussian blur?, edge detect, threshold high or low)
    - hough transform
    - fit lines
    - find grid spacing
    - fit grid

    cv2.imread() for image import.
    cv2.cvtColor() for color space conversion (e.g., RGB to grayscale).
    cv2.GaussianBlur() for noise reduction.
    cv2.Canny() for edge detection.

    cv2.HoughLines() for detecting lines in an image (standard Hough Transform).
    cv2.HoughLinesP() for probabilistic Hough Transform (can help with shorter lines and noise).

    cv2.threshold() for simple binary thresholding.
    cv2.adaptiveThreshold() for adaptive thresholding when grid lines have varying brightness.

"""

import cv2
import numpy as np
import matplotlib.pyplot as plt



def line_angle(line):
    """
    Calculates the angle of a line.

    Args:
        line (list): A list of line endpoints represented as [x1, y1, x2, y2].

    Returns:
        (float): The angle in degrees.
    """
    x1, y1, x2, y2 = line[0]
    return np.degrees(np.arctan2(y2 - y1, x2 - x1))


def filter_horiz_vert(lines):
    """
    Filters lines, keeping only horizontal and vertical lines.

    Args:
        lines (list): A list of line endpoints represented as [x1, y1, x2, y2].

    Returns:
        vertical_lines (list): A list of vertical line endpoints represented as [x1, y1, x2, y2].
        horizontal_lines (list): A list of horizontal line endpoints represented as [x1, yx, x2, y2].
    """
    # Separate lines into vertical and horizontal based on their angle
    vertical_lines = []
    horizontal_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = line_angle(line)

        # Vertical lines (within 5 degrees of 90° or 270°)
        if (abs(angle - 90) < 5 or abs(angle - 270) < 5):
            vertical_lines.append((x1, y1, x2, y2))

        # Horizontal lines (within 5 degrees of 0° or 180°)
        elif (abs(angle) < 5 or abs(angle - 180) < 5):
            horizontal_lines.append((x1, y1, x2, y2))

    return vertical_lines, horizontal_lines


def draw_hough(vertical_lines, horizontal_lines, image):
    """
    Draws lines on an image.

    Args:
        vertical_lines (list): A list of vertical line endpoints represented as [x1, y1, x2, y2].
        horizontal_lines (list): A list of horizontal line endpoints represented as [x1, y1, x2, y2].
        image (numpy.ndarray): The image to draw on.
    """

    if vertical_lines or horizontal_lines is not None:
        for line in vertical_lines:
        #for line in filtered_vertical_lines:
            #cv2.line(image_color, (line[0], 0), (line[0], image.shape[0]), (0, 255, 0), 2)  # Vertical lines in green

            x1, y1, x2, y2 = line
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        for line in horizontal_lines:
        #for line in filtered_horizontal_lines:
            #cv2.line(image_color, (0, line[0]), (image.shape[1], line[0]), (0, 255, 0), 2)  # Horizontal lines in green
            x1, y1, x2, y2 = line
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            """
        # draw detected lines
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image_color, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green lines for visibility
            """
    else:
        print("No lines detected")


"""
# Function to find the regular gap between lines
def find_regular_interval(lines):
    # Sort lines based on their position
    positions = []
    for line in lines:
        positions.append(line[0])  # For vertical lines, use x1 or x2 (they're approximately the same)

    positions.sort()

    # Calculate the gaps between consecutive lines
    gaps = []
    for i in range(1, len(positions)):
        gap = positions[i] - positions[i - 1]
        gaps.append(gap)

    # Find the most common gap (using a histogram)
    gap_histogram = {}
    for gap in gaps:
        gap_histogram[gap] = gap_histogram.get(gap, 0) + 1

    # Sort gaps by frequency and return the most common gap
    sorted_gaps = sorted(gap_histogram.items(), key=lambda x: x[1], reverse=True)
    regular_gap = sorted_gaps[0][0] if sorted_gaps else None

    return regular_gap


# Find the regular intervals for vertical and horizontal lines
vertical_gap = find_regular_interval(vertical_lines)
horizontal_gap = find_regular_interval(horizontal_lines)

# If gaps are detected, filter lines by regular interval
filtered_vertical_lines = []
filtered_horizontal_lines = []

if vertical_gap:
    # Filter vertical lines that match the detected gap
    positions = [line[0] for line in vertical_lines]
    positions.sort()
    for i in range(1, len(positions)):
        if positions[i] - positions[i - 1] == vertical_gap:
            filtered_vertical_lines.append((positions[i - 1], positions[i]))

if horizontal_gap:
    # Filter horizontal lines that match the detected gap
    positions = [line[0] for line in horizontal_lines]
    positions.sort()
    for i in range(1, len(positions)):
        if positions[i] - positions[i - 1] == horizontal_gap:
            filtered_horizontal_lines.append((positions[i - 1], positions[i]))

"""

def draw_grid(intersection, interval, image):
    """
    Draws a grid on the image.

    Args:
        intersection (tuple): The (x, y) coordinate of any grid intersection.
        interval (int): The interval length of the grid.
        image (numpy.ndarray): The image to draw on.
    """
    x, y = intersection[0], intersection[1] # first vertical and first horizontal lines of the grid
    height, width = image.shape[:2]

    # find how many lines to draw
    num_vert_left = (x // interval) + 1
    num_vert_right = ((width - x) // interval) + 1
    num_horiz_top = (y // interval) + 1
    num_horiz_bottom = ((height - y) // interval) + 1

    # cv2.line(image, (x, 0), (x, height), (0, 0, 255), 2)
    # cv2.line(image, (0, y), (width, y), (0, 0, 255), 2)

    # iterate through and draw
    for i in range(num_vert_left):
        cv2.line(image, (x, 0), (x, height), (0, 255, 0), 4)
        x -= interval

    x = intersection[0] + interval
    for i in range(num_vert_right):
        cv2.line(image, (x, 0), (x, height), (0, 0, 255), 2)
        x += interval

    for i in range(num_horiz_top):
        cv2.line(image, (0, y), (width, y), (0, 255, 0), 4)
        y -= interval

    y = intersection[1] + interval
    for i in range(num_horiz_bottom):
        cv2.line(image, (0, y), (width, y), (0, 0, 255), 2)
        y += interval


# load image (grayscale)
image = cv2.imread('map1.jpg', cv2.IMREAD_GRAYSCALE)

# gaussian blur
image = cv2.GaussianBlur(image, (5, 5), 0)
# canny seems to do better without gaussian blur, sobel better with


# CANNY EDGE DETECT
edges_canny = cv2.Canny(image, 50, 250)
cv2.imwrite('edges_canny.jpg', edges_canny)



# SOBEL EDGE DETECT
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3) # vertical edges (x-direction gradient)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3) # horizontal edges (y-direction gradient)

# convert gradients to absolute values and normalize
sobel_x = cv2.convertScaleAbs(sobel_x)
sobel_y = cv2.convertScaleAbs(sobel_y)

# combine sobel x and y edges
edges_combined = cv2.bitwise_or(sobel_x, sobel_y)

# threshold the edges to get a binary edge map
_, sobel_edges_thresholded = cv2.threshold(edges_combined, 50, 255, cv2.THRESH_BINARY)
_, x_edges_thresholded = cv2.threshold(sobel_x, 50, 255, cv2.THRESH_BINARY)
_, y_edges_thresholded = cv2.threshold(sobel_y, 50, 255, cv2.THRESH_BINARY)

# save edge image
cv2.imwrite('edges_sobel.jpg', sobel_edges_thresholded)
cv2.imwrite('edges_sobel_x.jpg', x_edges_thresholded)
cv2.imwrite('edges_sobel_y.jpg', y_edges_thresholded)



# hough fit
lines_canny = cv2.HoughLinesP(edges_canny, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=3)
lines_sobel = cv2.HoughLinesP(sobel_edges_thresholded, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=3)
lines_sobel_x = cv2.HoughLinesP(x_edges_thresholded, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=3)
lines_sobel_y = cv2.HoughLinesP(y_edges_thresholded, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=3)


# convert to color to draw lines for matplotlib
image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

canny_lines = image_color.copy()
vertical_lines, horizontal_lines = filter_horiz_vert(lines_canny)
draw_hough(vertical_lines, horizontal_lines, canny_lines)

sobel_lines = image_color.copy()
vertical_lines, horizontal_lines = filter_horiz_vert(lines_sobel)
draw_hough(vertical_lines, horizontal_lines, sobel_lines)

sobel_x_lines = image_color.copy()
vertical_lines, horizontal_lines = filter_horiz_vert(lines_sobel_x)
draw_hough(vertical_lines, horizontal_lines, sobel_x_lines)

sobel_y_lines = image_color.copy()
vertical_lines, horizontal_lines = filter_horiz_vert(lines_sobel_y)
draw_hough(vertical_lines, horizontal_lines, sobel_y_lines)


color_canny = cv2.cvtColor(edges_canny.copy(), cv2.COLOR_GRAY2BGR)
draw_grid((800,700), 100, color_canny)
cv2.imwrite('grid_test.jpg', color_canny)



# display image
# plt.figure(figsize=(10, 10))
# plt.imshow(cv2.cvtColor(canny_lines, cv2.COLOR_BGR2RGB))
# plt.axis('off')
# plt.show()
#
# plt.figure(figsize=(10, 10))
# plt.imshow(cv2.cvtColor(sobel_lines, cv2.COLOR_BGR2RGB))
# plt.axis('off')
# plt.show()
#
# plt.figure(figsize=(10, 10))
# plt.imshow(cv2.cvtColor(sobel_x_lines, cv2.COLOR_BGR2RGB))
# plt.axis('off')
# plt.show()
#
# plt.figure(figsize=(10, 10))
# plt.imshow(cv2.cvtColor(sobel_y_lines, cv2.COLOR_BGR2RGB))
# plt.axis('off')
# plt.show()

# save line fit image
cv2.imwrite('lines_canny.jpg', canny_lines)
cv2.imwrite('lines_sobel.jpg', sobel_lines)
cv2.imwrite('lines_sobel_x.jpg', sobel_x_lines)
cv2.imwrite('lines_sobel_y.jpg', sobel_y_lines)