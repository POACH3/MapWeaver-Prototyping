"""
grids_generate.py

NOTES:
    - synthetic generation

"""

import cv2
import numpy as np
import random
import os

def draw_grid(intersection, interval, color, thickness, image):
    """
    Draws a grid on the image.

    Args:
        intersection (tuple): The (x, y) coordinate of any grid intersection.
        interval (int): The interval length of the grid.
        image (numpy.ndarray): The image to draw on.

    Returns:
        drawn_lines (list): A list of lines drawn on the image.
    """
    drawn_lines = []

    x, y = intersection[0], intersection[1] # first vertical and first horizontal lines of the grid
    height, width = image.shape[:2]

    # find how many lines to draw
    num_vert_left = (x // interval) + 1
    num_vert_right = ((width - x) // interval) + 1
    num_horiz_top = (y // interval) + 1
    num_horiz_bottom = ((height - y) // interval) + 1

    # iterate through and draw
    for i in range(num_vert_left):
        cv2.line(image, (x, 0), (x, height), color, thickness)
        drawn_lines.append((x, 0, x, height))
        x -= interval

    x = intersection[0] + interval
    for i in range(num_vert_right):
        cv2.line(image, (x, 0), (x, height), color, thickness)
        drawn_lines.append((x, 0, x, height))
        x += interval

    for i in range(num_horiz_top):
        cv2.line(image, (0, y), (width, y), color, thickness)
        drawn_lines.append((0, y, width, y))
        y -= interval

    y = intersection[1] + interval
    for i in range(num_horiz_bottom):
        cv2.line(image, (0, y), (width, y), color, thickness)
        drawn_lines.append((0, y, width, y))
        y += interval

    return drawn_lines


def add_gaussian_noise(image, mean=0, sigma=25):
    """
    Adds Gaussian noise to the image.

    Args:
        image: Input image (numpy array)
        mean: Mean of the Gaussian noise (default is 0)
        sigma: Standard deviation of the Gaussian noise (default is 25)

    Returns:
        lines (list): A list of lines drawn on the image.
    """
    gauss = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy = np.add(image.astype(np.float32), gauss)
    noisy = np.clip(noisy, 0, 255)

    return noisy.astype(np.uint8)


def generate_image(img_num, color=False):
    # image dimensions (width, height)
    image_sizes = [256, 512, 1024, 2048]
    img_size = random.choice(image_sizes)

    if color:
        # background
        bg_color = random.randint(0, 255)
        image = np.full((img_size, img_size, 3), bg_color, dtype=np.uint8)
        variance = random.randint(25, 75)
        noisy_image = add_gaussian_noise(image, mean=0, sigma=variance)
        image = noisy_image

        # grid
        interval = random.randint(30,int(img_size / 4))
        intersection = (random.randint(0, img_size), random.randint(0, img_size))
        line_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        line_thickness = random.randint(1,3)

        lines = draw_grid(intersection, interval, line_color, line_thickness, image)

        # occlusion
        points = []
        num_points = random.randint(3,10)
        for i in range(num_points):
            points.append([random.randint(0, img_size), random.randint(0, img_size)])
        points = np.array([points], dtype=np.int32)

        poly_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.fillPoly(image, points, color=poly_color)

        center = (random.randint(0,img_size), random.randint(0,img_size))
        radius = random.randint(10, int(img_size / 2))
        circle_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.circle(image, center, radius, circle_color, -1)

    else:
        # background
        bg_brightness = random.randint(0, 255)
        image = np.full((img_size, img_size), bg_brightness, dtype=np.uint8)
        noisy_image = add_gaussian_noise(image, mean=0, sigma=50)
        image = noisy_image

        # grid
        interval = random.randint(30, int(img_size / 4))
        intersection = (random.randint(0, img_size), random.randint(0, img_size))
        line_brightness = random.randint(0, 255)
        line_thickness = random.randint(1, 3)

        lines = draw_grid(intersection, interval, line_brightness, line_thickness, image)

        # occlusion
        points = []
        num_points = random.randint(3, 10)
        for i in range(num_points):
            points.append([random.randint(0, img_size), random.randint(0, img_size)])
        points = np.array([points], dtype=np.int32)

        poly_brightness = random.randint(0, 255)
        cv2.fillPoly(image, points, color=poly_brightness)

        center = (random.randint(0,img_size), random.randint(0,img_size))
        radius = random.randint(10, int(img_size / 2))
        circle_brightness = random.randint(0, 255)
        cv2.circle(image, center, radius, circle_brightness, -1)

    # Save the image as a .jpg file
    file_path = os.path.join('synthetic_dataset', f'image{img_num}.jpg')
    cv2.imwrite(file_path, image)

    return lines



size_dataset = 10
labels = []
for i in range(1, size_dataset + 1):
    lines = generate_image(i)
    labels.append([i, lines])

labels_array = np.array(labels, dtype=object)
file_path = os.path.join('synthetic_dataset', f'labels.npz')
np.savez(file_path, labels=labels_array)