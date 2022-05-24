import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import pandas as pd
import os


"""
==============================================================================================================
This dataset was created to test the combined YOLO and CNN models on handwritten words. 
There were no such data, so I created one many manually by writing on my Ipad. This is saved in
3_testing_words/pages.
This .py files crops and processes the words out of the pages and saves them in 3_testing_words/testing_words.
It crops the words using the annotations created by me.
This 'testing_words' dataset is used to test the combined YOLO and CNN models in 5_testing.py
==============================================================================================================
"""



# Function that plots a single image
def plot_image(image):
    plt.imshow(image)
    plt.axis('off')
    plt.show()


"""
# Loads the images and bounding box labels
"""

# Read all images in pages directory
paths = glob.glob('3_testing_words/pages/*.jpg')
images_v1 = []
for path in paths:
    image = cv2.imread(path)
    images_v1.append(image)

# Read the labeled bounding boxes per image
labels_v1 = []
for path in paths:
    # Remove jpg from path. Add txt
    path = path[:-4] + '.txt'
    label = pd.read_csv(path, sep=" ", header=None, names=['label','x', 'y', 'w', 'h'])
    label = label.drop(columns=['label'])
    label = label.to_numpy()
    labels_v1.append(label)

plot_image(images_v1[0])
print(labels_v1[0])
print()


"""
Formats the coordinates in labels from YOLO to xmin, ymin, xmax, ymax
"""

# Function that formats bounding box coordinates
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

labels_v2 = []
for label in labels_v1:
    new_label = []
    for bounding_box in label:
        x_min, x_max, y_min, y_max = format_coordinates(bounding_box, images_v1[0].shape[1], images_v1[0].shape[0])
        bounding_box = [int(x_min), int(x_max), int(y_min), int(y_max)]
        new_label.append(bounding_box)
    labels_v2.append(new_label)

print(labels_v2[0])



"""
Crops the bounding boxes from the images.
"""

# Crop images from the bounding box labels
images_v2 = []
for image, label in zip(images_v1, labels_v2):
    for bounding_box in label:
        x_min = bounding_box[0]
        x_max = bounding_box[1]
        y_min = bounding_box[2]
        y_max = bounding_box[3]
        # Crop bounding box from image
        new_image = image[int(y_min):int(y_max), int(x_min):int(x_max)]
        # Add to new images list
        images_v2.append(new_image)


# Plot 3 of the cropped images
for image in images_v2[:3]:
    plot_image(image)


"""
Resizes the images to fit the yolo model format
"""

# Resize images
images_v3 = []
for image in images_v2:
    new_image = cv2.resize(image, (608, 608))
    images_v3.append(new_image)

# Plot 3 of the resized images
for image in images_v3[:3]:
    plot_image(image)


# Shape of images
images_v3 = np.array(images_v3)
print(images_v3.shape)
print()


"""
Thresholds the images to remove noise
"""

# Threshold images
images_v4 = []
for image in images_v3:
    new_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)[1]
    images_v4.append(new_image)

# Plot 2 of the thresholded images
for image in images_v4[:3]:
    plot_image(image)


"""
Saves to directory
"""
# If 1_documents/words include xml or jpg files, print error message
if len(os.listdir('3_testing_words/test_words')) > 0:
    print('3_testing_words/test_words directory is not empty. Please remove images from this directory before running this script.')
    print()

# Write images
else:
    # Write images to file
    for i, image in enumerate(images_v4):
        cv2.imwrite("3_testing_words/test_words/" + str(i) + ".jpg", image)