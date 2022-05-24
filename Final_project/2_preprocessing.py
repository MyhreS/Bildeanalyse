import cv2
import imgaug.augmenters as iaa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

"""
======================================================================================================================
There was no propper dataset of words, so I created a dataset from combining the 2_char74k_&_characters/characters and
2_char74k_&_characters/Fnt datasets.
The 'characters' dataset was a small dataset, so this .py file processes and augments them to a larger dataset.
Lastly this .py file combines the augmented dataset with the 'Fnt' dataset. The resulting dataset is saved in
2_char74k_&_characters/augmented.
It also saves the labels that lates is used to know what label each image is.
This 'augmented' dataset is used to train the CNN model in 4_training.py
======================================================================================================================
"""


"""
Preprocessing 'characters'

Handling the labels
"""
# Reading labels
labels = pd.read_csv('2_char74k_&_characters/characters/english.csv')
print(labels.head())
print()

# Unique labels
unique = labels.label.unique()
print(unique)
print()

# Create a dictonary with the unique labels
label_dict = {k:v for v,k in enumerate(unique)}
print(label_dict)
print()

# Rename the labels
labels['label'] = labels['label'].apply(lambda x: label_dict[x])
print(labels.head())
print()

# Unique labels
unique = labels.label.unique()
print(unique)
print()



"""
Handling the images
"""

# Read images
images = []
for i in range(len(labels)):
    img = cv2.imread('2_char74k_&_characters/characters/' + labels.iloc[i]['image'])
    images.append(img)
images = np.array(images)
print(images.shape)


# Function that plots a single image
def plot_image(image):
    plt.imshow(image)
    plt.axis('off')
    plt.show()

plot_image(images[0])


# Resize the images to 128x128
images_resized = []
for image in images:
    image_resized = cv2.resize(image, (128, 128))
    images_resized.append(image_resized)
images_resized = np.array(images_resized)
print(images_resized.shape)

plot_image(images_resized[0])



# Threshold images
images_thresholded = []
for image in images_resized:
    new_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)[1]
    images_thresholded.append(new_image)

plot_image(images_thresholded[0])



to_be_augmented = images_thresholded





"""
Aaugmenting images
"""
# Function to write image to dictionary
def write_image(writing_image, index):
    # writing_image = writing_image / 255.0 # Normalizing
    name = '2_char74k_&_characters/augmented/images/' + str(index) + '.jpg'
    cv2.imwrite(name, writing_image)
    print(name)

# Variables
labels_v1 = labels["label"].values
labels_v2 = []
count = 0

# 2_char74k_&_characters/augmented/images includes files, print error message
if len(os.listdir('2_char74k_&_characters/augmented/images')) > 0:
    print('2_char74k_&_characters/augmented/images directory is not empty. Please remove images from this directory before running this script.')
    print()

# 2_char74k_&_characters/augmented/labels.csv exists, print error message
elif os.path.exists('2_char74k_&_characters/augmented/labels.csv'):
    print('2_char74k_&_characters/augmented/labels.csv file already exists. Please remove this file before running this script.')
    print()

# Augment images
else:
    for image, label in zip(to_be_augmented, labels_v1):
        # Save original image
        write_image(image, count)
        labels_v2.append(label)
        count += 1

        # Apply gaussian blur using imgaug
        image_aug = iaa.GaussianBlur(sigma=(2.0, 6.0))(image=image)
        write_image(image_aug, count)
        labels_v2.append(label)
        count += 1

        # Apply gaussian noise using imgaug
        image_aug = iaa.AdditiveGaussianNoise(scale=(0, 0.2*255), per_channel=True)(image=image)
        write_image(image_aug, count)
        labels_v2.append(label)
        count += 1

        # Apply rotation using imgaug
        image_aug = iaa.Affine(rotate=(-70, -20))(image=image)
        write_image(image_aug, count)
        labels_v2.append(label)
        count += 1

         # Apply rotation using imgaug
        image_aug = iaa.Affine(rotate=(20, 70))(image=image)
        write_image(image_aug, count)
        labels_v2.append(label)
        count += 1

        # Apply white cutout using imgaug
        image_aug = iaa.Cutout(fill_mode="constant", size=(0.02, 0.2), cval=255)(image=image)
        write_image(image_aug, count)
        labels_v2.append(label)
        count += 1


        # Apply scale using imgaug
        image_aug = iaa.Affine(scale=(1.3, 2.0))(image=image)
        write_image(image_aug, count)
        labels_v2.append(label)
        count += 1

        # Apply padding using imgaug
        image_aug = iaa.Pad(px=(10, 30))(image=image)
        write_image(image_aug, count)
        labels_v2.append(label)
        count += 1

        # Apply scale using imgaug
        image_aug = iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)})(image=image)
        write_image(image_aug, count)
        labels_v2.append(label)
        count += 1

        # Apply rotation and shear using imgaug
        image_aug = iaa.Sequential([
            iaa.Affine(rotate=(-40, 40)),
            iaa.Affine(shear=(-20, 20))
        ])(image=image)
        write_image(image_aug, count)
        labels_v2.append(label)
        count += 1

        # Apply white cutout and rotation using imgaug
        image_aug = iaa.Sequential([
            iaa.Cutout(fill_mode="constant", size=(0.02, 0.2), cval=255),
            iaa.Affine(rotate=(-40, 40))
        ])(image=image)
        write_image(image_aug, count)
        labels_v2.append(label)
        count += 1


        # Apply gaussian noise and rotation using imgaug
        image_aug = iaa.Sequential([
            iaa.Affine(rotate=(-40, 40)),
            iaa.AdditiveGaussianNoise(scale=(0, 0.2*255), per_channel=True),
        ])(image=image)
        write_image(image_aug, count)
        labels_v2.append(label)
        count += 1


        # Apply gaussian noise and shear using imgaug
        image_aug = iaa.Sequential([
            iaa.Affine(shear=(-40, 40)),
            iaa.AdditiveGaussianNoise(scale=(0, 0.2*255), per_channel=True),
        ])(image=image)
        write_image(image_aug, count)
        labels_v2.append(label)
        count += 1

        # Apply scale and shear using imgaug
        image_aug = iaa.Sequential([
            iaa.Affine(scale=(1, 2.0)),
            iaa.Affine(shear=(-40, 40))
        ])(image=image)
        write_image(image_aug, count)
        labels_v2.append(label)
        count += 1


        # Apply scale, rotation, and gauusian blur using imgaug
        image_aug = iaa.Sequential([
            iaa.Affine(scale=(1, 2.0)),
            iaa.Affine(rotate=(-40, 40)),
            iaa.GaussianBlur(sigma=(2.0, 6.0)),
        ])(image=image)
        write_image(image_aug, count)
        labels_v2.append(label)
        count += 1

        # Apply scale, shear, rotation, and gauusian blur using imgaug
        image_aug = iaa.Sequential([
            iaa.Affine(scale=(1, 2.0)),
            iaa.Affine(shear=(-40, 40)),
            iaa.Affine(rotate=(-40, 40)),
            iaa.GaussianBlur(sigma=(2.0, 6.0)),
        ])(image=image)
        write_image(image_aug, count)
        labels_v2.append(label)
        count += 1

        # Apply scale, shear, rotation blur using imgaug
        image_aug = iaa.Sequential([
            iaa.Affine(scale=(1, 2.0)),
            iaa.Affine(shear=(-40, 40)),
            iaa.GaussianBlur(sigma=(2.0, 6.0)),
        ])(image=image)
        write_image(image_aug, count)
        labels_v2.append(label)
        count += 1

        # Apply gaussian noise and scale using imgaug
        image_aug = iaa.Sequential([
            iaa.Affine(scale=(1, 2.0)),
            iaa.AdditiveGaussianNoise(scale=(0, 0.2*255), per_channel=True),
        ])(image=image)
        write_image(image_aug, count)
        labels_v2.append(label)
        count += 1

        # Apply scale, rotation, and gaussian noise using imgaug
        image_aug = iaa.Sequential([
            iaa.Affine(scale=(1, 2.0)),
            iaa.Affine(rotate=(-40, 40)),
            iaa.AdditiveGaussianNoise(scale=(0, 0.2*255), per_channel=True),
        ])(image=image)
        write_image(image_aug, count)
        labels_v2.append(label)
        count += 1

        # Apply scale, shear and gaussian noise using imgaug
        image_aug = iaa.Sequential([
            iaa.Affine(scale=(1, 2.0)),
            iaa.Affine(shear=(-40, 40)),
            iaa.AdditiveGaussianNoise(scale=(0, 0.2*255), per_channel=True),
        ])(image=image)
        write_image(image_aug, count)
        labels_v2.append(label)
        count += 1

        # Apply scale, shear, rotation, and gaussian noise using imgaug
        image_aug = iaa.Sequential([
            iaa.Affine(scale=(1, 2.0)),
            iaa.Affine(shear=(-40, 40)),
            iaa.Affine(rotate=(-40, 40)),
            iaa.AdditiveGaussianNoise(scale=(0, 0.2*255), per_channel=True),
        ])(image=image)
        write_image(image_aug, count)
        labels_v2.append(label)
        count += 1

        # Apply scale and padding using imgaug
        image_aug = iaa.Sequential([
            iaa.Affine(scale=(1, 2.0)),
            iaa.Pad(px=(10, 30))
        ])(image=image)
        write_image(image_aug, count)
        labels_v2.append(label)
        count += 1

        # Apply scale and rotation using imgaug
        image_aug = iaa.Sequential([
            iaa.Affine(scale=(1, 2.0)),
            iaa.Affine(rotate=(-40, 40))
        ])(image=image)
        write_image(image_aug, count)
        labels_v2.append(label)
        count += 1



"""
Preprocessing 'Fnt'
"""
# Read one image from Fnt and display it
test_image = cv2.imread("2_char74k_&_characters/Fnt/Sample001/img001-00001.png")
plot_image(test_image)
print(test_image.shape)
print()

# Read directory paths in 'Fnt'
dir_paths = glob.glob("2_char74k_&_characters/Fnt/*")

# Make sure its sorted
dir_paths.sort(reverse=False)

for label, dir_path in enumerate(dir_paths):
    # Read image paths in directory
    image_paths = glob.glob(dir_path + "/*.png")

    for image_path in image_paths:
        # Read image
        image = cv2.imread(image_path)

        # Resize image
        #image = cv2.resize(image, (128, 128)) # They are already 128x128

        # Threshold image
        image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)[1]

        # Save image
        write_image(image, count)
        labels_v2.append(label)
        count += 1


# Write labels to csv
labels_v3 = pd.DataFrame({"label": labels_v2})
labels_v3.to_csv("2_char74k_&_characters/augmented/labels.csv", index=False)