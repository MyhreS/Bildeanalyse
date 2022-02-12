import imageio
import cv2 as cv# install opencv-python
from scipy.ndimage import gaussian_filter

def edge(image, index_x, index_y, distance_from_center):
    max_width_index = len(image[0])-1
    max_lenght_index = len(image)-1
    if index_x - distance_from_center < 0 or index_y - distance_from_center < 0:
        return True
    elif index_x + distance_from_center > max_lenght_index or index_y + distance_from_center > max_width_index:
        return True
    return False

def apply_mask(mask, image, index_x, index_y, distance_from_center):
    if not edge(image, index_x, index_y, distance_from_center):
        # Do calculations
        sum = 0
        for x in range(-distance_from_center, distance_from_center+1):
            for y in range(-distance_from_center, distance_from_center+1):
                sum += image[index_x+x, index_y+y] * mask[distance_from_center+x][distance_from_center+y]
        image[index_x, index_y] = sum
    return image

def laplacian_filter(image):
    mask = [
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ]
    distance_from_center = int((len(mask) - 1) / 2)

    # Apply mask
    for x in range(len(image)):
        for y in range(len(image)):
            image = apply_mask(mask, image, x, y, distance_from_center)

    # Set border to 255
    for x in range(len(image)):
        for y in range(len(image)):
            if edge(image, x, y,distance_from_center):
                image[x, y] = 255
    return image



# Read
image = imageio.imread("Lena-Gray-3.png")


image = gaussian_filter(image, sigma=5)



image = laplacian_filter(image)
print(image)

# Write
imageio.imwrite("output.png", image)


# https://www.youtube.com/watch?v=QIrwQ3N2WcM guide
# LoG = Laplacian of Gausian
# Gaussian filter removes noise. Noise is what makes edges not appering perfect edges in images. So that has to be dealt with. Noise is rappid changes. Laplacian finds rapid changes..

