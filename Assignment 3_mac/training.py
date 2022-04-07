# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
# https://stackabuse.com/implementing-pca-in-python-with-scikit-learn/
from scipy.io import loadmat
from sklearn import svm
import glob
import numpy as np

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


# Loading mats and labels in the digit_training folder.
mats_train, labels_train = load_mats('Digit_training_fourier_response_images/')
print(mats_train.shape)
print(len(labels_train))
# First mat
first_mat = mats_train[0]
print(first_mat.shape)
#print(first_mat)

degrees = [1, 3, 5]
# Create an SVM classifier
for degree in degrees:
    clf = svm.SVC(kernel='poly', C=1, degree=degree)

    # Fit the classifier on the training data
    clf.fit(mats_train, labels_train)

    # Load the testing data
    mats_test, labels_test = load_mats('Digit_testing_fourier_response_images/')

    # Predict the labels of the test data
    accuracy = clf.score(mats_test, labels_test)

    # Save the accuracy to a results.txt file
    with open('results.txt', 'a') as f:
        f.write('Accuracy for degree ' + str(degree) + ': ' + str(accuracy) + '\n')

"""Apply PCA on each vector in the mats_train and mats_test to be 196 features."""
from sklearn.decomposition import PCA













