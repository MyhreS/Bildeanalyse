import glob
import shutil
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import random
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, Flatten
from keras.models import Sequential, load_model
from keras.applications.vgg16 import VGG16



"""
Training a Yolov5 model
"""
"""
Loading the training data
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
Training the model
"""
os.system("python yolov5/train.py --img 608 --batch 16 --epochs 1 --data 1_documents/words_v2/data.yaml --weights yolov5s.pt --cache")


"""
Testing the yolov5 model on MY weights
"""

# Delete all folders in yolov5/runs/detect if run before
paths_in_detect = glob.glob("yolov5/runs/detect/*")
for path_in_detect in paths_in_detect:
    shutil.rmtree(path_in_detect)

# Run detect
os.system("python yolov5/detect.py --save-txt --weights yolo_trained_weights/best.pt --img 608 --conf 0.4 --source 1_documents/words_v2/test/images")



"""
Visualize yolov5 results
"""
# Reading images
detected_images = []
for path_detected_image in glob.glob("yolov5/runs/detect/exp/*jpg"):
   detected_images.append(cv2.imread(path_detected_image))

# Plot first 4 images
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.imshow(detected_images[i])
    plt.axis("off")
plt.show()






"""
Train a CNN model
"""
"""
Loading the training data
"""

# Path to images
number_of_images = len(glob.glob("2_char74k_&_characters/augmented/images/*.jpg"))

print("Reading images.. This might take a while")
# Read images
images = []
for i in range(number_of_images):
    name = "2_char74k_&_characters/augmented/images/"+str(i)+".jpg"
    # Read image using opencv
    image = cv2.imread(name)
    # To numpy array
    image = np.array(image)
    # Add to list
    images.append(image)
print("Done")
# To numpy array
images = np.array(images)
# Print shape
print(images.shape)



"""
Read labels. They are in order: 0, 1, 2 ..
"""

# Read labels
labels = pd.read_csv('2_char74k_&_characters/augmented/labels.csv')
# Print shape
print(labels.shape)
print()

# Print first lines
print(labels.head())
print()


"""
Get classes and number of classes
"""
# Get number of classes
number_of_classes = len(labels["label"].sort_values().unique())
# Get names of classes
labels = np.array(labels["label"])
# Print number of classes
print(number_of_classes)
print()


"""
Split the images and labels into train, val and test. 
"""

# Split in to train and val data
x_train, x_val, y_train, y_val = train_test_split(images,
                                                  labels,
                                                  train_size=0.8,
                                                  test_size=0.2,
                                                  )
# Split training to get test data
x_train, x_test, y_train, y_test = train_test_split(x_train,
                                                  y_train,
                                                  train_size=0.9,
                                                  test_size=0.1,
                                                  )


# Print shape of images and shape
print('training set: ', x_train.shape, y_train.shape)
print('validation set: ', x_val.shape, y_val.shape)
print('testing set: ', x_test.shape, y_test.shape)


"""
Visualize 9 images from the train data. Visualize with their respective label id
"""

# Dictionary of what each label id means
label_id = {"0":"0", "1":"1", "2":"2", "3":"3", "4":"4", "5":"5", "6":"6", "7":"7", "8":"8", "9":"9", "10":"A",
            "11":"B", "12":"C", "13":"D", "14":"E", "15":"F", "16":"G", "17":"H", "18":"I", "19":"J", "20":"K",
            "21":"L", "22":"M", "23":"N", "24":"O", "25":"P", "26":"Q", "27":"R", "28":"S", "29":"T", "30":"U",
            "31":"V", "32":"W", "33":"X", "34":"Y", "35":"Z", "36":"a", "37":"b", "38":"c", "39":"d", "40":"e",
            "41":"f", "42":"g", "43":"h", "44":"i", "45":"j", "46":"k", "47":"l", "48":"m", "49":"n", "50":"o",
            "51":"p", "52":"q", "53":"r", "54":"s", "55":"t", "56":"u", "57":"v", "58":"w", "59":"x", "60":"y",
            "61":"z" }

# Select random indexes from the train data
indexes = random.sample(range(0, len(x_train)), 9)

# Pick the samples from the train data
plotting_images = []
plotting_labels = []
for index in indexes:
  image = x_train[index]
  plotting_images.append(image)
  label = y_train[index]
  plotting_labels.append(label_id[str(label)])

# Visualize the samples
plt.figure(figsize=(8, 8))
plt.suptitle('Training')
for index in range(9):
    plt.subplot(3, 3, index+1)
    plt.imshow(plotting_images[index])
    plt.title(plotting_labels[index])
    plt.axis('off')
plt.show()



"""
Create a CNN model
"""

# Building model function
def build_model():
    # Importing a VGG16 model. Freze layers
    feauture_extraction_model = VGG16(weights="imagenet", include_top=False, input_shape=(128, 128, 3))
    feauture_extraction_model.trainable = False  # Removes the classification layer and freeze the model

    # Define model
    model = Sequential([
        feauture_extraction_model,  # VGG model that is freezed
        Flatten(),
        Dense(80, activation='relu'),  # Dense layers
        Dense(50, activation='relu'),
        Dense(number_of_classes, activation='softmax'),  # Output
    ])

    # Compiling
    adam = Adam(learning_rate=0.0001)
    model.compile(optimizer=adam, loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


"""
Build model
"""
# Build model
model = build_model()
# Print summary
model.summary()


"""
Fit train and val data to train the model
"""

# Specify early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

# Specify adaptive learning rate
adaptive_learning_rate = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                           mode="min", patience=3, min_lr=0.000001)

# Fit and train the model
history = model.fit(x_train, y_train, epochs=1,
                    validation_data=(x_val, y_val),
                    batch_size=16, verbose=1,
                    callbacks=[early_stopping, adaptive_learning_rate]
                    )


"""
Loading the CNN model with the weights I trained
"""

model = load_model("cnn_trained_model/CNN_trained_model.h5")


"""
Evaluating the model to get
"""
results = model.evaluate(x_test, y_test)
print("test loss, test acc:", results)


"""
Visualizing some predictions
"""
first_images = x_test[:9]
predictions = model.predict(first_images)

first_predictions = []
for prediction in predictions:
  first_predictions.append(label_id[str(np.argmax(prediction))])

plt.figure(figsize=(9, 9))
for index in range(9):
    plt.subplot(3, 3, index+1)
    plt.imshow(first_images[index])
    plt.title("Predicted: "+ first_predictions[index])
    plt.axis('off')
plt.show()
