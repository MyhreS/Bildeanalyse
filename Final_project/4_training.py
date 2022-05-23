import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

"""
Loading the training data

Read 9 images to visualize train images
"""

# Read path to first 9 images
path_words = glob.glob("1_documents/words_v2/train/images/*.jpg")[:9]

# Read images
images_words = []
for path_word in path_words:
  images_words.append(cv2.imread(path_word))

# Print shape
images_words = np.array(images_words)
print(images_words.shape)

# Plot 9 images
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(images_words[i])
    plt.axis("off")
plt.show()


"""
Training the yolov5 model
"""
os.system("python yolov5/train.py --img 608 --batch 16 --epochs 10 --data 1_documents/words_v2/data.yaml --weights yolov5s.pt --cache")


"""
Testing the yolov5 model on MY weights
"""