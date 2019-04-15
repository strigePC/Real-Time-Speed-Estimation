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
