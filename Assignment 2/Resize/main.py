from PIL import Image
import os

# Resizing training images
training_images_names = os.listdir("data_from_assignment/train")
for name in training_images_names:
    path = "data_from_assignment/train/" + name
    image = Image.open(path)
    new_image = image.resize((640, 640))
    new_image.save('../RCNN/models/research/object_detection/images/train/' + name)


# Resizing testing images
testing_images_names = os.listdir("data_from_assignment/test")
for name in testing_images_names:
    path = "data_from_assignment/test/" + name
    image = Image.open(path)
    new_image = image.resize((640, 640))
    new_image.save('../RCNN/models/research/object_detection/images/test/' + name)
