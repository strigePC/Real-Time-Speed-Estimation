import numpy as np

from numba import jit


# Transforming an input image into another using homography
@jit
def transform_image(input_image, output_image, hom):
    # Getting the sizes of input and output images
    in_height = len(input_image)
    in_width = len(input_image[0])
    out_height = len(output_image)
    out_width = len(output_image[0])

    # border_pixels = [
    #     transform_pixel([0, 0], hom),
    #     transform_pixel([in_width - 1, 0], hom),
    #     transform_pixel([0, in_height - 1], hom),
    #     transform_pixel([in_width - 1, in_height - 1], hom)
    # ]
    #
    # y_start = max(0, min(border_pixels[0][1], border_pixels[1][1]))
    # y_end = min(in_height, max(border_pixels[2][1], border_pixels[3][1]))
    #
    # x_start = max(0, min(border_pixels[0][0], border_pixels[2][0]))
    # x_end = min(in_width, max(border_pixels[1][0], border_pixels[3][0]))
    #
    # print(y_start)
    # print(y_end)
    # print(x_start)
    # print(x_end)

    for y in range(out_height):
        for x in range(out_width):
            denominator = hom[2, 0] * x + hom[2, 1] * y + hom[2, 2]  # Optimizing the calculation
            x_in = (hom[0, 0] * x + hom[0, 1] * y + hom[0, 2]) / denominator
            y_in = (hom[1, 0] * x + hom[1, 1] * y + hom[1, 2]) / denominator

            if x_in < 0 or x_in >= in_width or y_in < 0 or y >= in_height:
                continue

            # Assigning an intensity value to the output
            output_image[int(round(y)), int(round(x))] = input_image[int(round(y_in)), int(round(x_in))]

    return output_image


@jit
def calculate_homography(src, dst):
    x = src[:, 0]
    y = src[:, 1]
    x_bar = dst[:, 0]
    y_bar = dst[:, 1]
    a = np.zeros((8, 8))
    b = np.zeros((8, 1))

    for i in range(4):
        a[2 * i] = [x[i], y[i], 1, 0, 0, 0, -x[i] * x_bar[i], -y[i] * x_bar[i]]
        a[2 * i + 1] = [0, 0, 0, x[i], y[i], 1, -x[i] * y_bar[i], -y[i] * y_bar[i]]
        b[2 * i] = x_bar[i]
        b[2 * i + 1] = y_bar[i]

    h_vect = np.linalg.solve(a.T.dot(a), a.T.dot(b))

    h = np.zeros((3, 3))
    h[0] = [h_vect[0], h_vect[1], h_vect[2]]
    h[1] = [h_vect[3], h_vect[4], h_vect[5]]
    h[2] = [h_vect[6], h_vect[7], 1]

    return h


@jit
def opening(image, mask):
    return dilation(erosion(image, mask), mask)


@jit
def closing(image, mask):
    return erosion(dilation(image, mask), mask)


@jit
def erosion(image, mask):
    height_image = len(image)
    width_image = len(image[0])

    result = np.zeros_like(image)

    for y in range(height_image):
        for x in range(width_image):
            if fits(image, mask, (x, y)):
                result[y, x] = 255

    return result


@jit
def dilation(image, mask):
    height_image = len(image)
    width_image = len(image[0])

    result = np.zeros_like(image)

    for y in range(height_image):
        for x in range(width_image):
            if hits(image, mask, (x, y)):
                result[y, x] = 255

    return result


@jit
def fits(image, mask, center):
    height_mask = len(mask)
    width_mask = len(mask[0])
    height_image = len(image)
    width_image = len(image[0])

    for row in range(height_mask):
        for col in range(width_mask):
            row_offset = row - height_mask + round(height_mask / 2)
            col_offset = col - width_mask + round(width_mask / 2)
            target = (center[0] + col_offset, center[1] + row_offset)

            if 0 > target[0] or target[0] >= width_image or 0 > target[1] or target[1] >= height_image:
                continue

            if image[target[1], target[0]] != mask[row, col]:
                return False

    return True


@jit
def hits(image, mask, center):
    height_mask = len(mask)
    width_mask = len(mask[0])
    height_image = len(image)
    width_image = len(image[0])

    for row in range(height_mask):
        for col in range(width_mask):
            row_offset = row - height_mask + round(height_mask / 2)
            col_offset = col - width_mask + round(width_mask / 2)
            target = (center[0] + col_offset, center[1] + row_offset)
            if 0 > target[0] or target[0] >= width_image or 0 > target[1] or target[1] >= height_image:
                continue

            if image[target[1], target[0]] == mask[row, col]:
                return True

    return False


@jit
def convert_mask(mask):
    height_mask = len(mask)
    width_mask = len(mask[0])

    for y in range(height_mask):
        for x in range(width_mask):
            if mask[y, x] == 1:
                mask[y, x] = 255

    return mask
