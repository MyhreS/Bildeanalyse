import imageio
import numpy as np
from cv2 import GaussianBlur

def laplacian_Filter(image):
    new_image = np.zeros(image.shape, dtype=np.uint8)  # Creating an emty array (that will be the filtered image).
    # Creating variables for the length and width index
    max_width_index = image.shape[0] - 1
    max_length_index = image.shape[1] - 1
    # Looping through every pixel that will be applied mask on.
    for x in range(1, max_width_index):
        for y in range(1, max_length_index):
            # The formula for calculating the second derivative.
            calculation = abs((-4) * image[x, y] + image[x, y + 1] + image[x, y - 1] + image[x + 1, y] + image[x - 1, y])
            # Setting the pixel value to be the result of the calculation.
            new_image[x, y] = calculation
    return new_image


image = imageio.imread("Lena-Gray.png")  # Reading the image.
image_with = GaussianBlur(image, (5,5), 5) # Apply Gaussain filter.
imageio.imwrite("output-Lena-Gray-only_gaussian.png", image_with) # Writing an image that has only Guassian smoothening applied to it.
image_with = laplacian_Filter(image_with) # Apply Laplacian filter
image_without = laplacian_Filter(image) # Apply Laplacian filter

imageio.imwrite("output-Lena-Gray-with_gaussian.png", image_with) # Writing an image that has Guassian and Laplacian filter applied to it.
imageio.imwrite("output-Lena-Gray-without_gaussian.png", image_without) # Writing an image that has only Laplacian filter applied to it.

