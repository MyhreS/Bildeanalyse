import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd
import imgaug.augmenters as iaa
from sklearn.model_selection import train_test_split
import random
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, Flatten
from keras.models import Sequential, load_model
from keras.applications.vgg16 import VGG16
import shutil


"""
Preparing the testing data
"""
"""
Reading the custom created testing images
"""

# Read paths
paths_to_testing_images = glob.glob("3_testing_words/test_words/*jpg")
print(paths_to_testing_images)
print()

# Read images from path
testing_images = []
for path_to_testing_image in paths_to_testing_images:
  testing_images.append(cv2.imread(path_to_testing_image))

# Print shape
testing_images = np.array(testing_images)
print(testing_images.shape)
print()


"""
Visualizing the custom testing images
"""

# Plot the 3 images
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.imshow(testing_images[i])
    plt.axis('off')
plt.show()



# """
# Detecting on the custom testing images using the YOLOv5 model with MY weights
# """
# """
# Detect characters in the custom test images using the weight I trained
# """
#
# # Delete all folders in yolov5/runs/detect if run before
# paths_in_detect = glob.glob("yolov5/runs/detect/*")
# for path_in_detect in paths_in_detect:
#     shutil.rmtree(path_in_detect)
#
# # Run detect
# os.system("python yolov5/detect.py --save-txt --weights yolo_trained_weights/best.pt --img 608 --conf 0.4 --source 3_testing_words/test_words")
#
#
#
# """
# Visualize the detected words
# """
#
# # Read images
# detected_images = []
# for path_detected_image in glob.glob("yolov5/runs/detect/exp/*jpg"):
#    detected_images.append(cv2.imread(path_detected_image))
#
# # Plot the 3 images
# for i in range(3):
#     plt.subplot(2, 2, i+1)
#     plt.imshow(detected_images[i])
#     plt.axis("off")
# plt.show()


"""
Cropping the images to get the characters. These will be classified by the CNN later
"""
"""
Getting labels (bounding boxes) and images
"""

# Label paths
labels_paths = glob.glob("yolov5\\runs\\detect\\exp\\labels\\*")

names = []
label_dict = {}
for label_path in labels_paths:
    # Read label file
    label = pd.read_csv(label_path, sep=" ", header=None)
    label = label.drop(0, axis=1)  # Dropping the object detection class name
    label = label.to_numpy()

    # Get name of label file
    label_name = label_path.split("\\")[-1]
    label_name = label_name.split(".txt")[0]
    names.append(label_name) # Add to names list

    # Add to dictionary
    label_dict[label_name] = label


image_dict = {}
for image_path in paths_to_testing_images:
    # Read image file
    image = cv2.imread(image_path, 0)

    # Get name of image file
    image_name = image_path.split("\\")[-1]
    image_name = image_name.split(".jpg")[0]

    # Add to dictionary
    image_dict[image_name] = image


"""
Cropping bounding boxes out of images
"""

# Function that formats coordinates from annotation
def format_coordinates(bounding_box, img_width, img_height):
    x = bounding_box[0]
    y = bounding_box[1]
    w = bounding_box[2]
    h = bounding_box[3]

    x_min = ((2 * x * img_width) - (w * img_width)) / 2
    x_max = ((2 * x * img_width) + (w * img_width)) / 2
    y_min = ((2 * y * img_height) - (h * img_height)) / 2
    y_max = ((2 * y * img_height) + (h * img_height)) / 2

    return x_min, x_max, y_min, y_max

# Crop images from the bounding box labels
new_images = []
for name in names:
    image = image_dict[name]
    label = label_dict[name]
    i = 0
    for bounding_box in label:
        i += 1
        # Coordinates formated
        x_min, x_max, y_min, y_max = format_coordinates(bounding_box, image.shape[1], image.shape[0])
        # Crop bounding box from image
        new_image = image[int(y_min):int(y_max), int(x_min):int(x_max)]
        # Add to new images list
        new_images.append(new_image)

print("Number of character images:",len(new_images))


# Plot first 9 images from new_images
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(new_images[i], cmap="gray")
    plt.axis("off")
plt.show()


"""
Resize the images to be the size that the CNN model expects
"""

# Function that resizes images to 128x128
def resize(images):
  images_128 = []
  for image in images:
      new_image = cv2.resize(image, (128, 128))
      images_128.append(new_image)
  return np.array(images_128)

# Resize images
images_v1 = resize(new_images)
print(images_v1.shape)

# Plot first 9 images from images_v1
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(images_v1[i], cmap="gray")
    plt.axis("off")
plt.show()


"""
Erode images to make them thinner so they are more similar the training data. Custom made
"""


# Custom made erotion function
def erotion(image, iterations): # Must recieve a 1 channel image
    # Create kernel
    kernel = [[0,0,0],[0,1,1],[1,1,1]]
    kernel_size = len(kernel)

    # Add padding
    padding = 1
    padded_image = np.zeros((image.shape[0]+(padding*2), image.shape[1]+(padding*2)), dtype=int)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            padded_image[i+padding, j+padding] = image[i,j]

    # Erotion
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # If point is 1, check if surrounding are not 0 where the kernel is 1. If so, change the point to be 0
            if padded_image[i+padding, j+padding] != 0:
                # Cut out area of image that will be checked
                area_of_interest = padded_image[i:i+kernel_size, j:j+kernel_size]
                apply = False
                # If any index in kernel is 1 and at the same index 1 in area of interest, apply erotion
                for row_area, row_kernel in zip(area_of_interest, kernel):
                    for value_area, value_kernel in zip(row_area, row_kernel):
                        if value_kernel == 1 and value_area == 0:
                            apply = True
                if apply: # Apply erotion if true
                    padded_image[i+padding, j+padding] = 0

    # Remove padding
    image = padded_image[padding:padded_image.shape[0]-padding, padding:padded_image.shape[1]-padding]

    if iterations > 1:
      image = erotion(image, iterations-1)
    return image

# Extra function to keep the depth of the image
def erode(image, iteration):
  image_eroded = erotion(image, iteration)
  for i in range(image.shape[0]):
    for j in range(image.shape[1]):
      image[i, j] = image_eroded[i, j]
  return image


images_v2 = []
for image in images_v1:
  images_v2.append(erode(image, 4))
images_v2 = np.array(images_v2)


# Plot first 9 images from images_v2
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(images_v2[i], cmap="gray")
    plt.axis("off")
plt.show()



"""
The characters are taking up the entire picture because of the detection. Zoom out to be more similar to training data
"""

# Zoom out
images_v3 = []
for image in images_v2:
  image_aug = iaa.Affine(scale=0.7)(image=image) # Zoom out
  images_v3.append(image_aug)

# Plot first 9 images from images_v3
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(images_v3[i], cmap="gray")
    plt.axis("off")
plt.show()
