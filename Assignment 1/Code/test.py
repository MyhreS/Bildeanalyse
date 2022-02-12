import imageio
import numpy as np
from scipy.ndimage import gaussian_filter


def Laplacian_Filter(image):
    new_image = np.zeros(image.shape, dtype=np.uint8) # Creating an emty array (that will be the filtered image).

    max_width_index = image.shape[0] - 1
    max_length_index = image.shape[1] - 1
    print("Laplacian filtering")
    for i in range(1, max_width_index):
        for j in range(1, max_length_index):
            dx = (image[i - 1, j + 1] + 2 * image[i, j + 1] + image[i + 1, j + 1]) - (image[i - 1, j - 1] + 2 * image[i, j - 1]
                                                                                      + image[i + 1, j - 1])
            dy = (image[i + 1, j - 1] + 2 * image[i + 1, j] + image[i + 1, j + 1]) - (image[i - 1, j - 1] + 2 * image[i - 1, j]
                                                                                      + image[i - 1, j + 1])

            new_image[i, j] = abs(dx) + abs(dy)
    return new_image


def main():
    image = imageio.imread("Lena-Gray-3.png") # Reading the image.
    #image = gaussian_filter(image, sigma=5) # Using gaussain filter.

    image = Laplacian_Filter(image)
    print(image)
    # Save the transformed image to the filename in the 2nd command line argument.
    imageio.imwrite("output.png", image)

main()