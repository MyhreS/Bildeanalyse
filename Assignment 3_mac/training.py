# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
# https://stackabuse.com/implementing-pca-in-python-with-scikit-learn/
import os
from sklearn.decomposition import PCA
from scipy.io import loadmat
from sklearn import svm
import glob
import numpy as np

# Delete results.txt and results_pca if it already exists
try:
    os.remove('results.txt')
except OSError:
    pass
try:
    os.remove('results_pca.txt')
except OSError:
    pass

# Loads a mat file.
def load_mat(path_to_math_file):
    """Load a mat file. """
    return loadmat(path_to_math_file)['fourier_responses']

# Loads multiple mat files.
def load_mats(path_to_folder):
    # Use glob to find all folders in digit_testing
    label_folders = glob.glob(path_to_folder+'*')


    # Finds all mat files in each folder
    labels = []
    mats = []
    for label_folder in label_folders:
        label = label_folder.split('/')[-1]
        mat = load_mat(label_folder + '/fourier_responses.mat')
        for vector in mat:
            mats.append(vector)
            labels.append(label)

    return np.array(mats), labels

def apply_pca(mats, n_components):
    # Apply PCA to the data
    pca = PCA(n_components=n_components)
    return pca.fit_transform(mats)


"""Load training and testing data with labels. Then train and test multiple SVM classifiers."""
# Loading mats and labels in the digit_training folder.
mats_train, labels_train = load_mats('Digit_training_fourier_response_images/')
# Load the testing data
mats_test, labels_test = load_mats('Digit_testing_fourier_response_images/')

#Checking the shape of the data
print("Shape of training data:", mats_train.shape)
print("Length of labels:", len(labels_train))
# First mat
first_mat = mats_train[0]
print("Shape of first vector:", first_mat.shape)
#print(first_mat)

degrees = [1, 3, 5]
# Create an SVM classifier
for degree in degrees:
    clf = svm.SVC(kernel='poly', C=1, degree=degree)

    # Fit the classifier on the training data
    clf.fit(mats_train, labels_train)

    # Predict the labels of the test data
    accuracy = clf.score(mats_test, labels_test)

    # Save the accuracy to a results.txt file
    with open('results.txt', 'a') as f:
        f.write('Accuracy for degree ' + str(degree) + ': ' + str(accuracy) + '\n')



"""Apply PCA on each vector in the mats_train and mats_test to be 196 features and train multiple new SVM classifiers."""
# Apply pca on the training data
pca_mats_train = apply_pca(mats_train, 196)
# Apply pca on the testing data
pca_mats_test = apply_pca(mats_test, 196)
print("Shape of training data after pca: ", pca_mats_train.shape)
print("Shape of first vector after pca: ", pca_mats_test[0].shape)

degrees = [1, 3, 5]
# Create an SVM classifier
for degree in degrees:
    clf = svm.SVC(kernel='poly', C=1, degree=degree)

    # Fit the classifier on the training data
    clf.fit(pca_mats_train, labels_train)

    # Load the testing data
    mats_test, labels_test = load_mats('Digit_testing_fourier_response_images/')

    # Predict the labels of the test data
    accuracy = clf.score(pca_mats_test, labels_test)

    # Save the accuracy to a results_pca.txt file
    with open('results_pca.txt', 'a') as f:
        f.write('Accuracy for degree ' + str(degree) + ': ' + str(accuracy) + '\n')











