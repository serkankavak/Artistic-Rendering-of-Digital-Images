# Created by Serkan Kavak and Adnan D. Tahir for EEM 463 Introduction to Image Processing Project
# Eskisehir Technical University
# Project : Artistic Rendering of Digital Images

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def pdist(A, B):
    sum_ = 0
    for a, b in zip(A, B):
        sum_ += (a - b) ** 2
    return np.sqrt(sum_)


source_img = mpimg.imread('Input_Images/Scream.png')  # Texture to be matched
target_img = mpimg.imread('Target_Images/Benedict_Cumberbatch.png')  # Image to be transformed

neighborhood = 5  # Considers an nxn neighborhood (e.g., 5x5)
p = 0.2  # Probability of considering a random pixel
m = 1  # Weighting on intesity matching between images

# Step 1: Initalize output image
source_h, source_w, source_d = source_img.shape  # height, width, depth of source image
target_h, target_w, target_d = target_img.shape  # height, width, depth of target image
n_2 = neighborhood // 2


# Grayscale values. They will be needed to calculate distances
gray_source = rgb2gray(source_img)
gray_target = rgb2gray(target_img)

# Create variables for output image and locations used in input image
out_img = np.zeros((target_h + n_2, target_w + n_2 * 2, source_d))
used_heights = np.zeros((target_h + n_2, target_w + n_2 * 2))
used_widths = np.zeros((target_h + n_2, target_w + n_2 * 2))

# Randomly assign "used" pixel locations to borders

# Top border
used_heights[0:n_2, :] = np.rint(np.random.random((n_2, target_w + n_2 * 2)) * (source_h - 1))
used_widths[0:n_2, :] = np.rint(np.random.random((n_2, target_w + n_2 * 2)) * (source_w - 1))

# Left most border
used_heights[:, 0:n_2] = np.rint(np.random.random((target_h + n_2, n_2)) * (source_h - 1))
used_widths[:, 0:n_2] = np.rint(np.random.random((target_h + n_2, n_2)) * (source_w - 1))

# Right most border
used_heights[:, n_2 + target_w:n_2 * 2 + target_w] = np.rint(np.random.random((target_h + n_2, n_2)) * (source_h - 1))
used_widths[:, n_2 + target_w:n_2 * 2 + target_w] = np.rint(np.random.random((target_h + n_2, n_2)) * (source_w - 1))

# Fill output with appropriate color values
# Fill top border
for w in range(0, target_w + n_2 * 2):
    for h in range(0, n_2):
        out_img[h, w, :] = source_img[int(used_heights[h, w]), int(used_widths[h, w]), :]

# Fill left and right border
W = [w for w in range(0, n_2)]  # left most border cordinates
W.extend([w for w in range(n_2 + target_w, n_2 * 2 + target_w)])  # right most border cordinates

for w in W:
    for h in range(n_2 - 1, target_h + n_2):
        out_img[h, w, :] = source_img[int(used_heights[h, w]), int(used_widths[h, w]), :]

# start to the biggest part of the processing. Calculation for each pixel
for h in range(n_2, target_h + n_2):
    for w in range(n_2, target_w + n_2):
        candidate_locations = []
        candidate_pixels = []

        search_height = [i for i in range(0, n_2 + 1)]
        search_width = [i for i in range(0, neighborhood)]

        # Generate candidate pixels
        for c_h in search_height:
            for c_w in search_width:
                c_w_adj = c_w - n_2
                if ((((c_h == 0 and c_w_adj < 0) or c_h > 0)) and h - c_h <= target_h + n_2):
                    new_height = used_heights[h - c_h, w + c_w_adj] + c_h;
                    new_width = used_widths[h - c_h, w + c_w_adj] - c_w_adj;

                    # If we reach the edge of the image, choose a new pixel
                    while (new_height < neighborhood or new_height > source_h - neighborhood or
                           new_width < neighborhood or new_width > source_w - neighborhood):
                        new_height = round(np.random.rand() * (source_h - 1))
                        new_width = round(np.random.rand() * (source_w - 1))

                    candidate_locations.append((new_height, new_width))
                    candidate_pixels.append(source_img[int(new_height), int(new_width), :])

        # Add random pixel with probability p
        if np.random.rand() < p:
            new_height = round(np.random.rand() * (source_h - 1))
            new_width = round(np.random.rand() * (source_w - 1))
            # If we reach the edge of the image, choose a new pixel
            while (new_height < neighborhood or new_height > source_h - neighborhood or
                   new_width < neighborhood or new_width > source_w - neighborhood):
                new_height = round(np.random.rand() * (source_h - 1))
                new_width = round(np.random.rand() * (source_w - 1))

            candidate_locations.append((new_height, new_width))
            candidate_pixels.append(source_img[int(new_height), int(new_width), :])

        candidate_locations.sort()
        best_dist = 10000
        for c_h, c_w in candidate_locations:
            # Distance between target and the result
            c_h = int(c_h)
            c_w = int(c_w)
            input_values = []
            result_values = []

            # Distance between target and the result
            # input values NOTE : better conversion might exists!
            for ch in range(3):
                for w_ in range(c_w - n_2, c_w + n_2 + 1):
                    for h_ in range(c_h - n_2, c_h):
                        input_values.append(source_img[h_, w_, ch])

                for w_ in range(c_w - n_2, c_w):
                    input_values.append(source_img[c_h, w_, ch])

            # result values
                for w_ in range(w - n_2, w + n_2 + 1):
                    for h_ in range(h - n_2, h):
                        result_values.append(out_img[h_, w_, ch])

                for w_ in range(w - n_2, w):
                    result_values.append(out_img[h, w_, ch])

            input_result_distance = pdist(input_values, result_values)
            n = neighborhood * n_2 + n_2

            # Distance between the intensity of the neighborhood
            # NOTE : this implemented pretty poorly. This is the best we found.
            height = [i for i in range(max(-n_2, 1 - (h - n_2) - 1), min(n_2, target_h - (h - n_2)) + 1)]
            width = [i for i in range(max(-n_2, 1 - (w - n_2) - 1), min(n_2, target_w - (w - n_2)) + 1)]
            height = np.asarray(height)
            width = np.asarray(width)

            # get input vals
            height_in = c_h + height
            width_in = c_w + width
            input_vals = gray_source[height_in[0]:height_in[-1] + 1, width_in[0]:width_in[-1] + 1]

            # get target vals
            height_tar = (h - n_2) + height
            width_tar = (w - n_2) + width
            target_vals = gray_target[height_tar[0]:height_tar[-1] + 1, width_tar[0]:width_tar[-1] + 1]

            input_target_distance = (np.mean(input_vals) - np.mean(target_vals)) ** 2
            distance = m * input_target_distance + (1 / n ** 2) * input_result_distance

            # get the best distance
            if distance < best_dist:
                best_pixel = source_img[c_h, c_w, :]
                best_location = (c_h, c_w)
                best_dist = distance

        # print(h, w)

        # add new pixel
        out_img[h, w, :] = best_pixel
        used_heights[h, w] = best_location[0]
        used_widths[h, w] = best_location[1]

new_out = out_img[n_2:target_h, n_2:target_w + n_2, :]
plt.imshow(new_out)
plt.show()
plt.imsave('output.png', new_out)
