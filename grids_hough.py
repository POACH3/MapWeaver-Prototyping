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
    x1, y1, x2, y2 = line
    return np.degrees(np.arctan2(y2 - y1, x2 - x1))


def filter_vert_horiz(lines):
    """
    Filters and sorts lines, keeping only horizontal and vertical lines.

    Args:
        lines (list): A list of line endpoints represented as [x1, y1, x2, y2].

    Returns:
        vertical_lines (list): A list of vertical line endpoints represented as [x1, y1, x2, y2].
        horizontal_lines (list): A list of horizontal line endpoints represented as [x1, yx, x2, y2].
    """
    vertical_lines = []
    horizontal_lines = []

    if lines is not None:
        for line in lines:
            #x1, y1, x2, y2 = line[0] # for 3d array
            x1, y1, x2, y2 = line # for 2d array
            #angle = line_angle(line[0])
            angle = line_angle(line)

            # vertical lines (within 2 degrees of 90° or 270°)
            if (abs(angle - 90) < 2 or abs(angle - 270) < 2):
                vertical_lines.append((x1, y1, x2, y2))

            # horizontal lines (within 2 degrees of 0° or 180°)
            elif (abs(angle) < 2 or abs(angle - 180) < 2):
                horizontal_lines.append((x1, y1, x2, y2))

    return vertical_lines, horizontal_lines


def draw_hough(lines, image):
    """
    Draws lines on an image.

    Args:
        vertical_lines (list): A list of vertical line endpoints represented as [x1, y1, x2, y2].
        horizontal_lines (list): A list of horizontal line endpoints represented as [x1, y1, x2, y2].
        image (numpy.ndarray): The image to draw on.
    """
    vertical_lines, horizontal_lines = filter_vert_horiz(lines)

    if vertical_lines is not None:
        for line in vertical_lines:
            #cv2.line(image_color, (line[0], 0), (line[0], image.shape[0]), (0, 255, 0), 2)  # Vertical lines in green
            x1, y1, x2, y2 = line
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    if horizontal_lines is not None:
        for line in horizontal_lines:
            #cv2.line(image_color, (0, line[0]), (image.shape[1], line[0]), (0, 255, 0), 2)  # Horizontal lines in green
            x1, y1, x2, y2 = line
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if vertical_lines is None and horizontal_lines is None:
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

def average_clusters(lines, epsilon):
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
    vertical_lines, horizontal_lines = filter_vert_horiz(lines) # repeat just to get the separate lines lists
    x_averages = []
    y_averages = []

    if vertical_lines is not None:
        x_points = []

        # this adds the end points. should it instead be the midpoint?
        for line in vertical_lines:
            # x1, y1, x2, y2 = line[0]
            x1, y1, x2, y2 = line
            x_points.append(x1)
            x_points.append(x2)

            x_points_2d = np.array(x_points).reshape(-1, 1)

            dbscan = DBSCAN(eps=10, min_samples=2)
            x_labels = dbscan.fit_predict(x_points_2d)
            #print("x_labels: ", x_labels)

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

            for i in range(len(x_clusters)):  # add averaging (clustered) lines to list
                x_averages.append(x_clusters[i] / num_x_clusters[i])

            for i in range(len(x_labels)):  # add "noise" (non-cluster) lines to list
                if x_labels[i] == -1:
                    x_averages.append(x_points[i])

    if horizontal_lines is not None:
        y_points = []

        for line in horizontal_lines:
            #x1, y1, x2, y2 = line[0]
            x1, y1, x2, y2 = line
            y_points.append(y1)
            y_points.append(y2)

        y_points_2d = np.array(y_points).reshape(-1, 1)

        # pass to DBSCAN
        dbscan = DBSCAN(eps=10, min_samples=2)
        y_labels = dbscan.fit_predict(y_points_2d)
        #print("y_labels: ", y_labels)


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

        for i in range(len(y_clusters)):
            y_averages.append(y_clusters[i] / num_y_clusters[i])

        for i in range(len(y_labels)): # add "noise" (non-cluster) lines to list
            if y_labels[i] == -1:
                y_averages.append(y_points[i])

    return x_averages, y_averages

# x_averages, y_averages = average_clusters(vertical_lines, horizontal_lines)
#
# dbscan_image = sobel_indiv_lines.copy()
# height, width = dbscan_image.shape[:2]
#
# for i in range(len(x_averages)):
#     cv2.line(dbscan_image, (int(x_averages[i]), 0), (int(x_averages[i]), height), (255, 255, 0), 2)
#
# for i in range(len(y_averages)):
#     cv2.line(dbscan_image, (0, int(y_averages[i])), (width, int(y_averages[i])), (255, 255, 0), 2)
#
# cv2.imwrite('dbscan_sobel.jpg', dbscan_image)


def merge_inline(lines, epsilon):
    """
    Merge nearby parallel (ish) lines within epsilon distance.

    Notes:
        seems to favor top or left endpoint (for vertical lines)
        sort by average instead of min?
        average greedily in both directions to avoid bias?

    Args:
        lines (list): list of lines in the format (x1, y1, x2, y2)
        epsilon (int): the distance threshold for merging lines

    Returns:
        merged_lines (list): list of merged lines
    """
    merged_lines = []

    x1, y1, x2, y2 = lines[0]
    if abs(x2 - x1) < abs(y2 - y1):                                # x values are closer than y values
        #sorted_lines = sorted(lines, key=lambda line: min(x1, x2)) # sort by smalled x-coord
        sorted_lines = sorted(lines, key=lambda line: min(line[0], line[2]))  # sort by smalled x-coord

        l = 0
        r = 1
        while r < len(sorted_lines):
            l_x1, l_y1, l_x2, l_y2 = sorted_lines[l]
            r_x1, r_y1, r_x2, r_y2 = sorted_lines[r]

            if abs(r_x1 - l_x1) < epsilon:
                r += 1
            else:
                r -= 1 # go back to last r
                y_vals = []

                #ave_x = int(sum([sorted_lines[i][0] for i in range(l, r + 1)]) / (r - l + 1))
                #ave_x = sum([sorted_lines[i][0] for i in range(l, r + 1)]) // (r - l + 1)
                ave_x = 0
                for i in range(l, r+1):
                    ave_x += sorted_lines[i][0]
                    y_vals.append(sorted_lines[i][1])
                    y_vals.append(sorted_lines[i][3])
                #ave_x = ave_x // ((r - l) + 1)
                ave_x = int(round(ave_x / ((r - l) + 1))) # avoid downward (left/up) bias
                min_y = min(y_vals)
                max_y = max(y_vals)

                merged_lines.append((ave_x, min_y, ave_x, max_y))

                l = r+1 # move l past r to not double count the same line in the merged lines
                r += 2

        # Handle the last group of lines if any (after the loop ends)
        if r > l:  # Check if there are remaining lines between l and r-1
            ave_x = sum([sorted_lines[i][0] for i in range(l, r)]) // (r - l)
            y_vals = []
            for i in range(l, r):
                y_vals.append(sorted_lines[i][1])
                y_vals.append(sorted_lines[i][3])
            min_y = min(y_vals)
            max_y = max(y_vals)

            merged_lines.append((ave_x, min_y, ave_x, max_y))

    else:
        # same logic by for the y values
        sorted_lines = sorted(lines, key=lambda line: min(y1, y2))

        l = 0
        r = 1
        while r < len(sorted_lines):
            l_x1, l_y1, l_x2, l_y2 = sorted_lines[l]
            r_x1, r_y1, r_x2, r_y2 = sorted_lines[r]

            if abs(r_y1 - l_y1) < epsilon:
                r += 1
            else:
                r -= 1  # go back to last r
                x_vals = []

                # ave_x = int(sum([sorted_lines[i][0] for i in range(l, r + 1)]) / (r - l + 1))
                # ave_x = sum([sorted_lines[i][0] for i in range(l, r + 1)]) // (r - l + 1)
                ave_y = 0
                for i in range(l, r + 1):
                    ave_y += sorted_lines[i][1]
                    x_vals.append(sorted_lines[i][0])
                    x_vals.append(sorted_lines[i][2])
                #ave_y = ave_y // ((r - l) + 1)
                ave_y = int(round(ave_y / ((r - l) + 1)))
                min_x = min(x_vals)
                max_x = max(x_vals)

                merged_lines.append((min_x, ave_y, max_x, ave_y))

                l = r + 1  # move l past r to not double count the same line in the merged lines
                r += 2

        # Handle the last group of lines if any (after the loop ends)
        if r > l:  # Check if there are remaining lines between l and r-1
            ave_y = sum([sorted_lines[i][1] for i in range(l, r)]) // (r - l)
            x_vals = []
            for i in range(l, r):
                x_vals.append(sorted_lines[i][0])
                x_vals.append(sorted_lines[i][2])
            min_x = min(x_vals)
            max_x = max(x_vals)

            merged_lines.append((min_x, ave_y, max_x, ave_y))

    return merged_lines

def estimate_interval(lines, epsilon):
    """
    vote on intervals (lines vote or just do the mode of gaps between adjacent lines and check that the SD is low).
    do second round and check if the intervals are multiples (within tolerance) of good candidate intervals.

    Args:
        lines (list): list of lines in the format (x1, y1, x2, y2)
        epsilon (int): the allowed error for a vote to still count for an interval

    Returns:
        likely_interval (int): the voted interval
    """
    # lines = average_clusters(horizontal_lines, epsilon)
    #
    # _, y_lines = lines
    # print(y_lines)
    # x1, y1, x2, y2 = y_lines
    # if abs(x2 - x1) < abs(y2 - y1): # case lines are vertical
    #     line_orientation = 0
    # else:
    #     line_orientation = 1

    adj_intervals = []
    votes = {}

    x1, y1, x2, y2 = lines[0]
    if abs(x2 - x1) < abs(y2 - y1): # case lines are vertical
        line_orientation = 0
    else:
        line_orientation = 1

    # record all adjacent intervals
    for i in range(len(lines) - 1):
        adj_intervals.append(int(abs(lines[i+1][line_orientation] - lines[i][line_orientation])))

    # average intervals within epsilon and round
    grouped_intervals = []
    for interval in adj_intervals:
        added = False
        for group in grouped_intervals:
            # calculate the current middle of the group (average of existing intervals)
            group_middle = sum(group) / len(group)

            # interval within epsilon of the group's middle?
            if abs(group_middle - interval) <= epsilon:
                group.append(interval)
                added = True
                break

        if not added: # start new group
            grouped_intervals.append([interval])

    for group in grouped_intervals:
        avg_interval = sum(group) / len(group)
        votes[avg_interval] = len(group)


    candidate_intervals = list(votes.keys())

    # go through all other (non-adjacent) intervals and check for multiples of highest voted interval

    for i in range(len(lines) - 2):
        for j in range(i + 2, len(lines)):  # j starts from i+2 to avoid adjacent intervals
            nonadj_interval = int(abs(lines[j][line_orientation] - lines[i][line_orientation]))

            # # this allows each interval to vote ones– the closest candidate interval
            # closest_candidate = min(candidate_intervals, key=lambda x: abs(nonadj_interval - x))
            # votes[closest_candidate] += 1

            # this allows one interval to vote multiple times
            for k in range(len(candidate_intervals)):
                # if nonadj_interval % candidate_intervals[k] < epsilon: # doesn't allow for more error for higher multiple
                #     votes[candidate_intervals[k]] += 1

                candidate_interval = candidate_intervals[k]
                relative_epsilon = epsilon * candidate_interval

                # check if the interval is a multiple of the candidate interval (with relative tolerance)
                if abs(nonadj_interval % candidate_interval) < relative_epsilon:
                    votes[candidate_interval] += 1

    # for i in range(len(lines) - 2):
    #     for j in range(i + 2, len(lines)):  # j starts from i+2 to avoid adjacent intervals
    #         nonadj_interval = int(abs(lines[j][line_orientation] - lines[i][line_orientation]))





    # check the standard deviation of each interval group– should be low, indicating high consensus for whatever candidate they vote for
    # sd = np.std(candidate_intervals)
    # print(sd)

    #variation / mean_spacing # to convert variability to relative scale

    # Calculate standard deviation for voting confidence
    vote_values = list(votes.values())
    sd = np.std(vote_values)

    # Calculate mean interval spacing to normalize variability
    mean_spacing = np.mean(adj_intervals)
    variation = sd / mean_spacing

    # Confidence measure: Inversely proportional to the standard deviation
    confidence = 1 / (1 + sd)  # This can be adjusted as needed

    # If multiple intervals have close votes, add a fallback to break ties
    if len(set(vote_values)) == 1:
        confidence *= 0.5  # Lower confidence if votes are very close

    likely_interval = int(max(votes, key=votes.get))

    print(f"Likely Interval: {likely_interval}")
    print(f"Votes: {votes}")
    print(f"SD: {sd}, Variation: {variation}")
    print(f"Confidence: {confidence}")

    # need to record all the lines (or at least intervals) voting for each interval, then average the interval value they vote for

    return likely_interval # return confidence?


def estimate_intersection(vertical_lines, horizontal_lines, interval, shape):
    """
    select vertical and horizontal line with high confidence (long, has lots of interval multiples, _____)
    in order to return the most confident intersection

    Args:
        vertical_lines (list): list of lines in the format (x1, y1, x2, y2)
        horizontal_lines (list): list of lines in the format (x1, y1, x2, y2)
        shape (tuple): pixel dimensions of the image
        interval (int): the grid spacing interval

    Returns:
        intersection (tuple): coordinate represented in the format (x, y)
    """
    height, width, _ = shape

    # extend lines
    extended_vertical = []
    extended_horizontal = []

    for line in vertical_lines:
        x1, _, x2, _ = line
        extended_vertical.append((x1, 0, x2, height))

    for line in horizontal_lines:
        _, y1, _, y2 = line
        extended_horizontal.append((0, y1, width, y2))

    # get intersections
    intersections = []
    for vert_line in extended_vertical:
        v_x1, _, _, _ = vert_line

        for horiz_line in extended_horizontal:
            _, h_y1, _, _ = horiz_line
            coord = (v_x1, h_y1)
            intersections.append(coord)

    # score intersections based on error between every line and interval multiple
    scores_intersection = {}
    # scores_x = {}
    # scores_y = {}
    for intersection in intersections:
        x, y = intersection
        score_x = 0
        score_y = 0

        for line in extended_vertical:
            x1, _, _, _ = line
            #error_x = abs(x - x1) % interval
            error_x = abs(x - round(x1 / interval) * interval)
            score_x += error_x

        for line in extended_horizontal:
            _, y1, _, _ = line
            #error_y = abs(y - y1) % interval
            error_y = abs(y - round(y1 / interval) * interval)
            score_y += error_y

        #score_combined = (score_x / len(vertical_lines)) + (score_y / len(horizontal_lines)) # weight equally or skew to greater number of lines?
        score_combined = score_x + score_y
        scores_intersection[intersection] = score_combined
        # scores_x[x] = score_x
        # scores_y[y] = score_y
    intersection = min(scores_intersection, key=scores_intersection.get)
    return intersection

def match_grid():
    """
    pattern matching using least squares with a tentative grid.
    tentative grid informed by interval estimate and confident intersection.

    """
    pass

def edges(image):
    image = image.copy()

    # gaussian blur
    image = cv2.GaussianBlur(image, (5, 5), 1)
    # canny seems to do better without gaussian blur, sobel better with


    # CANNY EDGE DETECT
    edges_canny = cv2.Canny(image, 50, 250)


    # SOBEL EDGE DETECT
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # vertical edges (x-direction gradient)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # horizontal edges (y-direction gradient)

    # convert gradients to absolute values and normalize
    sobel_x = cv2.convertScaleAbs(sobel_x)
    sobel_y = cv2.convertScaleAbs(sobel_y)

    # combine sobel x and y edges
    sobel_combined = cv2.bitwise_or(sobel_x, sobel_y)

    # threshold the edges to get a binary edge map
    _, sobel_edges_thresholded = cv2.threshold(sobel_combined, 50, 255, cv2.THRESH_BINARY)
    _, edges_sobel_x = cv2.threshold(sobel_x, 50, 255, cv2.THRESH_BINARY)
    _, edges_sobel_y = cv2.threshold(sobel_y, 50, 255, cv2.THRESH_BINARY)

    # save edge images
    # cv2.imwrite('edges_canny.jpg', edges_canny)
    # #cv2.imwrite('edges_sobel.jpg', sobel_edges_thresholded)
    # cv2.imwrite('edges_sobel_x.jpg', edges_sobel_x)
    # cv2.imwrite('edges_sobel_y.jpg', edges_sobel_y)

    # morpho ops
    # kernel = np.ones((2, 2), np.uint8)
    # morpho_sobel_x = cv2.dilate(sobel_x, kernel, iterations=1)
    # # morpho_sobel_x = cv2.erode(morpho_sobel_x, kernel, iterations=1)
    # _, x_edges_thresholded = cv2.threshold(morpho_sobel_x, 50, 255, cv2.THRESH_BINARY)
    # # morphoblur_sobel_x = cv2.GaussianBlur(x_edges_thresholded, (5, 5), 1)
    #
    # edges_or = cv2.bitwise_or(sobel_combined, edges_canny)
    # _, edges_or_thresholded = cv2.threshold(edges_or, 50, 255, cv2.THRESH_BINARY)
    # dilated_or = cv2.dilate(edges_or_thresholded, kernel, iterations=1)
    # eroded_or = cv2.erode(dilated_or, kernel, iterations=2)
    #
    # edges_and = cv2.bitwise_and(sobel_combined, edges_canny)
    # _, edges_and_thresholded = cv2.threshold(edges_and, 50, 255, cv2.THRESH_BINARY)

    # cv2.imwrite('edges_or.jpg', eroded_or)
    # cv2.imwrite('edges_and.jpg', edges_and_thresholded)

    return edges_canny, edges_sobel_x, edges_sobel_y

def detect_grid(image):
    """
    Estimate intervals and intersection.

    """

    # length_threshold = 100
    length_threshold = max(image.shape) * .05

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges_canny, edges_sobel_x, edges_sobel_y = edges(image_gray)

    # hough fit
    canny_lines = cv2.HoughLinesP(edges_canny, 1, np.pi / 180, threshold=10, minLineLength=50, maxLineGap=5)
    #sobel_lines = cv2.HoughLinesP(edges_sobel, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=5)
    sobel_x_lines = cv2.HoughLinesP(edges_sobel_x, 1, np.pi / 180, threshold=10, minLineLength=50, maxLineGap=5)
    sobel_y_lines = cv2.HoughLinesP(edges_sobel_y, 1, np.pi / 180, threshold=10, minLineLength=50, maxLineGap=5)
    # edges_or_lines = cv2.HoughLinesP(eroded_or, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=5)
    # edges_and_lines = cv2.HoughLinesP(edges_and_thresholded, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=5)

    # convert to color for matplotlib and line draw
    color_edges_canny = cv2.cvtColor(edges_canny, cv2.COLOR_GRAY2BGR)
    color_edges_sobel_x = cv2.cvtColor(edges_sobel_x, cv2.COLOR_GRAY2BGR)
    color_edges_sobel_y = cv2.cvtColor(edges_sobel_y, cv2.COLOR_GRAY2BGR)


    lines_sobel_x = color_edges_sobel_x.copy()
    sobel_x_lines = np.squeeze(sobel_x_lines) # houghlinep returns a 3d array
    vertical_lines, _ = filter_vert_horiz(sobel_x_lines)
    # print('num vert: ' + str(len(vertical_lines)))
    # print('num hor: ' + str(len(horizontal_lines)))
    vertical_lines = merge_inline(vertical_lines, 5)
    #horizontal_lines = merge_inline(horizontal_lines, 10)
    vertical_lines = filter_length(vertical_lines, length_threshold)
    #horizontal_lines = filter_length(horizontal_lines, length_threshold)
    draw_hough(vertical_lines, lines_sobel_x)

    lines_sobel_y = color_edges_sobel_y.copy()
    sobel_y_lines = np.squeeze(sobel_y_lines) # houghlinep returns a 3d array
    _, horizontal_lines = filter_vert_horiz(sobel_y_lines)
    #vertical_lines = merge_inline(vertical_lines, 10)
    horizontal_lines = merge_inline(horizontal_lines, 5)
    #vertical_lines = filter_length(vertical_lines, length_threshold)
    horizontal_lines = filter_length(horizontal_lines, length_threshold)
    draw_hough(horizontal_lines, lines_sobel_y)

    # lines_sobel = image_color.copy() # pre-hough sobel combine (anded)
    # vertical_lines, horizontal_lines = filter_horiz_vert(lines_sobel)
    # vertical_lines = filter_length(vertical_lines, length_threshold)
    # horizontal_lines = filter_length(horizontal_lines, length_threshold)
    # draw_hough(vertical_lines, horizontal_lines, sobel_lines)

    lines_sobel_xy = cv2.cvtColor(cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR) # post-hough sobel combine
    _, horizontal_lines = filter_vert_horiz(sobel_y_lines)
    vertical_lines, _ = filter_vert_horiz(sobel_x_lines)
    vertical_lines = merge_inline(vertical_lines, 5)
    horizontal_lines = merge_inline(horizontal_lines, 5)
    # vertical_lines = filter_length(vertical_lines, length_threshold)
    # horizontal_lines = filter_length(horizontal_lines, length_threshold)
    lines = vertical_lines + horizontal_lines
    draw_hough(lines, lines_sobel_xy)
    lines = vertical_lines + horizontal_lines
    interval_sobel = estimate_interval(lines, 1)  # correct for map2 is 72
    intersection_sobel = estimate_intersection(vertical_lines, horizontal_lines, interval_sobel, lines_sobel_xy.shape)
    # draw_grid((198, 286), 72, sobel_indiv_lines) # 214, 1656, 1728
    #draw_grid(intersection, interval, lines_sobel_xy)

    # edges_or_lines = image_color.copy()
    # vertical_lines, horizontal_lines = filter_horiz_vert(lines_edges_or)
    # vertical_lines = filter_length(vertical_lines, length_threshold)
    # horizontal_lines = filter_length(horizontal_lines, length_threshold)
    # draw_hough(vertical_lines, horizontal_lines, edges_or_lines)
    #
    # edges_and_lines = image_color.copy()
    # vertical_lines, horizontal_lines = filter_horiz_vert(lines_edges_and)
    # vertical_lines = filter_length(vertical_lines, length_threshold)
    # horizontal_lines = filter_length(horizontal_lines, length_threshold)
    # draw_hough(vertical_lines, horizontal_lines, edges_and_lines)


    lines_canny = color_edges_canny.copy() # copy?
    canny_lines = np.squeeze(canny_lines) # houghlinep returns a 3d array
    vertical_lines, horizontal_lines = filter_vert_horiz(canny_lines)
    vertical_lines = merge_inline(vertical_lines, 4)
    horizontal_lines = merge_inline(horizontal_lines, 4)
    vertical_lines = filter_length(vertical_lines, length_threshold)
    horizontal_lines = filter_length(horizontal_lines, length_threshold)
    lines = vertical_lines + horizontal_lines
    draw_hough(lines, lines_canny)
    interval_canny = estimate_interval(horizontal_lines, 1)  # correct for map2 is 72
    # need to estimate for vertical
    intersection_canny = estimate_intersection(vertical_lines, horizontal_lines, interval_canny, lines_canny.shape)
    #draw_grid(intersection, interval, canny_grid_lines)

    # save line image
    cv2.imwrite('lines_canny.jpg', lines_canny)
    # cv2.imwrite('lines_sobel.jpg', sobel_lines)
    cv2.imwrite('lines_sobel_x.jpg', lines_sobel_x)
    cv2.imwrite('lines_sobel_y.jpg', lines_sobel_y)
    cv2.imwrite('lines_sobel.jpg', lines_sobel_xy)
    # cv2.imwrite('lines_edges_or.jpg', edges_or_lines)
    # cv2.imwrite('lines_edges_and.jpg', edges_and_lines)

    return intersection_sobel, interval_sobel, intersection_canny, interval_canny

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

    # iterate through and draw
    for i in range(num_vert_left):
        cv2.line(image, (x, 0), (x, height), (255, 255, 0), 1)
        x -= interval

    x = intersection[0] + interval
    for i in range(num_vert_right):
        cv2.line(image, (x, 0), (x, height), (255, 255, 0), 1)
        x += interval

    for i in range(num_horiz_top):
        cv2.line(image, (0, y), (width, y), (255, 255, 0), 1)
        y -= interval

    y = intersection[1] + interval
    for i in range(num_horiz_bottom):
        cv2.line(image, (0, y), (width, y), (255, 255, 0), 1)
        y += interval

    x, y = intersection[0], intersection[1]
    cv2.line(image, (x, 0), (x, height), (0, 255, 255), 2)
    cv2.line(image, (0, y), (width, y), (0, 255, 255), 2)





# load image
image = cv2.imread('map2.jpg')

intersection_sobel, interval_sobel, intersection_canny, interval_canny = detect_grid(image.copy())

grid_sobel = cv2.cvtColor(cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
draw_grid(intersection_sobel, interval_sobel, grid_sobel)
cv2.imwrite('grid_sobel.jpg', grid_sobel)

grid_canny = grid_sobel = cv2.cvtColor(cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
draw_grid(intersection_canny, interval_canny, grid_canny)
cv2.imwrite('grid_canny.jpg', grid_canny)

# color_canny = cv2.cvtColor(edges_canny.copy(), cv2.COLOR_GRAY2BGR)
# draw_grid((800,700), 100, color_canny)
# cv2.imwrite('grid_test.jpg', color_canny)

# get hough lines
# sort/filter lines
# cluster lines
# merge lines
# interval vote



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






# reject dense, wide clusters of hough lines? (caused by a wall)
# extend hough lines that are nearly perfectly in line and then reject short lines?



# draw estimated grid