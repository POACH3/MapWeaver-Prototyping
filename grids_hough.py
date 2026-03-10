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
from sklearn.cluster import DBSCAN



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
        if (abs(angle - 90) < 2 or abs(angle - 270) < 2):
            vertical_lines.append((x1, y1, x2, y2))

        # Horizontal lines (within 5 degrees of 0° or 180°)
        elif (abs(angle) < 2 or abs(angle - 180) < 2):
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



def filter_length(lines, threshold):
    """
    Returns only lines whose length is greater than the given threshold.

    DELETE AND USE HOUGH THRESHOLD

    Args:
        lines (list): A list of line endpoints represented as [x1, y1, x2, y2].
        threshold (int): The threshold length.

    Returns:
        thresholded_lines (list): A list of line endpoints represented as [x1, y1, x2, y2].
    """
    filtered_lines = []
    for line in lines:
        x1, y1, x2, y2 = line
        l1_length = abs(x1 - x2) + abs(y1 - y2) # L1 distance costs less
        if l1_length > threshold:
            filtered_lines.append(line)

    return filtered_lines

def average_clusters(vertical_lines, horizontal_lines):
    """
    Uses DBSCAN to find Hough line clusters, then
    creates a line which is the average of each cluster.

    Args:
        vertical_lines (list): list of vertical lines
        horizontal_lines (list): list of horizontal lines

    Returns:
        x_averages (list): list of the average vertical lines
        y_averages (list): list of the average horizontal lines
    """
    # get points for each hough line
    # vertical_lines, horizontal_lines = filter_horiz_vert(lines_sobel) # repeat just to get the separate lines lists

    x_points = []
    y_points = []

    for line in vertical_lines:
        x1, y1, x2, y2 = line[0]
        x_points.append(x1)
        x_points.append(x2)

    for line in horizontal_lines:
        x1, y1, x2, y2 = line[0]
        y_points.append(y1)
        y_points.append(y2)

    x_points_2d = np.array(x_points).reshape(-1, 1)
    y_points_2d = np.array(y_points).reshape(-1, 1)

    # pass to DBSCAN
    dbscan = DBSCAN(eps=10, min_samples=2)
    x_labels = dbscan.fit_predict(x_points_2d)
    y_labels = dbscan.fit_predict(y_points_2d)

    print("x_labels: ", x_labels)
    print("y_labels: ", y_labels)

    # take average of clusters
    x_clusters = {}
    num_x_clusters = {}
    for i in range(len(x_labels)):
        if x_labels[i] != -1:
            if x_labels[i] not in x_clusters:
                x_clusters[x_labels[i]] = 0
            if x_labels[i] not in num_x_clusters:
                num_x_clusters[x_labels[i]] = 0

                x_clusters[x_labels[i]] += x_points[i]
                num_x_clusters[x_labels[i]] += 1

    x_averages = []
    for i in range(len(x_clusters)):
        x_averages.append(x_clusters[i] / num_x_clusters[i])

    y_clusters = {}
    num_y_clusters = {}
    for i in range(len(y_labels)):
        if y_labels[i] != -1:
            if y_labels[i] not in y_clusters:
                y_clusters[y_labels[i]] = 0
            if y_labels[i] not in num_y_clusters:
                num_y_clusters[y_labels[i]] = 0

                y_clusters[y_labels[i]] += y_points[i]
                num_y_clusters[y_labels[i]] += 1

    y_averages = []
    for i in range(len(y_clusters)):
        y_averages.append(y_clusters[i] / num_y_clusters[i])

    return x_averages, y_averages


def merge_inline(lines, epsilon):
    """
    Merge nearby parallel (ish) lines within epsilon distance.

    Args:
        lines (list): list of lines in the format (x1, y1, x2, y2)
        epsilon (int): the distance threshold for merging lines

    Returns:
        merged_lines (list): list of merged lines
    """
    merged_lines = []

    if abs(lines[0][0] - lines[0][2]) < abs(lines[0][1] - lines[0][3]):  # x values are closer than y values
        sorted_lines = sorted(lines, key=lambda line: min(line[0], line[2])) # sort by smalled x-coord

        l = 0
        r = 1
        y_vals = []

        while r < len(sorted_lines):
            l_x1, l_y1, l_x2, l_y2 = sorted_lines[l]
            y_vals = [l_y1, l_y2] # reset

            if abs(sorted_lines[r][0] - sorted_lines[l][0]) < epsilon:
                r += 1
            else:
                r_x1, r_y1, r_x2, r_y2 = sorted_lines[r]
                y_vals.extend([r_y1, r_y2])

                ave_x = sum([sorted_lines[i][0] for i in range(l, r + 1)]) / (r - l + 1)
                min_y = min(y_vals)
                max_y = max(y_vals)

                merged_lines.append((ave_x, min_y, ave_x, max_y))

                l = r+1 # move l past r to not double count the same line in the merged lines
                r += 2

    else:
        # same logic by for the y values
        sorted_lines = sorted(lines, key=lambda line: min(line[1], line[3]))

        l = 0
        r = 1
        x_vals = []

        while r < len(sorted_lines):
            l_x1, l_y1, l_x2, l_y2 = sorted_lines[l]
            x_vals = [l_x1, l_x2]

            if abs(sorted_lines[r][1] - sorted_lines[l][1]) < epsilon:
                r += 1
            else:
                r_x1, r_y1, r_x2, r_y2 = sorted_lines[r]
                x_vals.extend([r_x1, r_x2])

                ave_y = sum([sorted_lines[i][1] for i in range(l, r + 1)]) / (r - l + 1)
                min_x = min(x_vals)
                max_x = max(x_vals)

                merged_lines.append((min_x, ave_y, max_x, ave_y))

                l = r+1 # move l past r to not double count the same line in the merged lines
                r += 2

    return merged_lines

def estimate_interval():
    """
    vote on intervals (lines vote or just do the mode of gaps between adjacent lines and check that the SD is low).
    do second round and check if the intervals are multiples (within tolerance) of good candidate intervals.

    :return:
    """
    pass

def confident_intersection():
    """
    select vertical and horizontal line with high confidence (long, has lots of interval multiples, _____)
    in order to return the most confident intersection

    """
    pass

def match_grid():
    """
    pattern matching using least squares with a tentative grid.
    tentative grid informed by interval estimate and confident intersection.

    """
    pass

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
image = cv2.GaussianBlur(image, (5, 5), 1)
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
sobel_combined = cv2.bitwise_or(sobel_x, sobel_y)

# threshold the edges to get a binary edge map
_, sobel_edges_thresholded = cv2.threshold(sobel_combined, 50, 255, cv2.THRESH_BINARY)
_, x_edges_thresholded = cv2.threshold(sobel_x, 50, 255, cv2.THRESH_BINARY)
_, y_edges_thresholded = cv2.threshold(sobel_y, 50, 255, cv2.THRESH_BINARY)

# morpho ops
kernel = np.ones((2, 2), np.uint8)
morpho_sobel_x = cv2.dilate(sobel_x, kernel, iterations=1)
#morpho_sobel_x = cv2.erode(morpho_sobel_x, kernel, iterations=1)
_, x_edges_thresholded = cv2.threshold(morpho_sobel_x, 50, 255, cv2.THRESH_BINARY)
#morphoblur_sobel_x = cv2.GaussianBlur(x_edges_thresholded, (5, 5), 1)


# save edge image
cv2.imwrite('edges_sobel.jpg', sobel_edges_thresholded)
cv2.imwrite('edges_sobel_x.jpg', x_edges_thresholded)
cv2.imwrite('edges_sobel_y.jpg', y_edges_thresholded)

edges_or = cv2.bitwise_or(sobel_combined, edges_canny)
_, edges_or_thresholded = cv2.threshold(edges_or, 50, 255, cv2.THRESH_BINARY)
dilated_or = cv2.dilate(edges_or_thresholded, kernel, iterations=1)
eroded_or = cv2.erode(dilated_or, kernel, iterations=2)

edges_and = cv2.bitwise_and(sobel_combined, edges_canny)
_, edges_and_thresholded = cv2.threshold(edges_and, 50, 255, cv2.THRESH_BINARY)

cv2.imwrite('edges_or.jpg', eroded_or)
cv2.imwrite('edges_and.jpg', edges_and_thresholded)


# hough fit
lines_canny = cv2.HoughLinesP(edges_canny, 1, np.pi / 180, threshold=10, minLineLength=50, maxLineGap=5)
lines_sobel = cv2.HoughLinesP(sobel_edges_thresholded, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=5)
lines_sobel_x = cv2.HoughLinesP(x_edges_thresholded, 1, np.pi / 180, threshold=10, minLineLength=50, maxLineGap=5)
lines_sobel_y = cv2.HoughLinesP(y_edges_thresholded, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=5)
lines_edges_or = cv2.HoughLinesP(eroded_or, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=5)
lines_edges_and = cv2.HoughLinesP(edges_and_thresholded, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=5)


# convert to color to draw lines for matplotlib
image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

length_threshold = 100

canny_lines = image_color.copy()
vertical_lines, horizontal_lines = filter_horiz_vert(lines_canny)
vertical_lines = filter_length(vertical_lines, length_threshold)
horizontal_lines = filter_length(horizontal_lines, length_threshold)
draw_hough(vertical_lines, horizontal_lines, canny_lines)

sobel_lines = image_color.copy()
vertical_lines, horizontal_lines = filter_horiz_vert(lines_sobel)
vertical_lines = filter_length(vertical_lines, length_threshold)
horizontal_lines = filter_length(horizontal_lines, length_threshold)
draw_hough(vertical_lines, horizontal_lines, sobel_lines)

sobel_x_lines = image_color.copy()
vertical_lines, horizontal_lines = filter_horiz_vert(lines_sobel_x)
vertical_lines = filter_length(vertical_lines, length_threshold)
horizontal_lines = filter_length(horizontal_lines, length_threshold)
draw_hough(vertical_lines, horizontal_lines, sobel_x_lines)

sobel_y_lines = image_color.copy()
vertical_lines, horizontal_lines = filter_horiz_vert(lines_sobel_y)
vertical_lines = filter_length(vertical_lines, length_threshold)
horizontal_lines = filter_length(horizontal_lines, length_threshold)
draw_hough(vertical_lines, horizontal_lines, sobel_y_lines)

edges_or_lines = image_color.copy()
vertical_lines, horizontal_lines = filter_horiz_vert(lines_edges_or)
vertical_lines = filter_length(vertical_lines, length_threshold)
horizontal_lines = filter_length(horizontal_lines, length_threshold)
draw_hough(vertical_lines, horizontal_lines, edges_or_lines)

edges_and_lines = image_color.copy()
vertical_lines, horizontal_lines = filter_horiz_vert(lines_edges_and)
vertical_lines = filter_length(vertical_lines, length_threshold)
horizontal_lines = filter_length(horizontal_lines, length_threshold)
draw_hough(vertical_lines, horizontal_lines, edges_and_lines)

sobel_indiv_lines = image_color.copy()
_, horizontal_lines = filter_horiz_vert(lines_sobel_y)
vertical_lines, _ = filter_horiz_vert(lines_sobel_x)
vertical_lines = filter_length(vertical_lines, length_threshold)
horizontal_lines = filter_length(horizontal_lines, length_threshold)
draw_hough(vertical_lines, horizontal_lines, sobel_indiv_lines)



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
cv2.imwrite('sobel_indiv_lines.jpg', sobel_indiv_lines)
cv2.imwrite('lines_edges_or.jpg', edges_or_lines)
cv2.imwrite('lines_edges_and.jpg', edges_and_lines)




x_averages, y_averages = average_clusters(vertical_lines, horizontal_lines)

dbscan_image = sobel_indiv_lines.copy()
height, width = dbscan_image.shape[:2]

for i in range(len(x_averages)):
    cv2.line(dbscan_image, (int(x_averages[i]), 0), (int(x_averages[i]), height), (255, 255, 0), 2)

for i in range(len(y_averages)):
    cv2.line(dbscan_image, (0, int(y_averages[i])), (width, int(y_averages[i])), (255, 255, 0), 2)

cv2.imwrite('dbscan_sobel.jpg', dbscan_image)





# reject dense, wide clusters of hough lines? (caused by a wall)
# extend hough lines that are nearly perfectly in line and then reject short lines?



# draw estimated grid