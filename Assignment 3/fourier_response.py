import os
import shutil

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, fftshift
from scipy.io import savemat
from scipy.io import loadmat

def flatten(fourier_response):
    """Flatten the fourier response. """
    return fourier_response.flatten()

# Function to load mat file
def load_mat(path_to_math_file):
    """Load a mat file. """
    return loadmat(path_to_math_file)['fourier_responses']


def extract_fourier_response_for_one_folder(data_folder):
    """Extract Fourier response from all images in the data folder. """

    #Create output folder.
    output_folder = data_folder
    split = output_folder.split('/')
    last_folder = split[-2]
    # Join everything before the last two slashes
    output_folder = '/'.join(split[:-2]) + '_fourier_response_images/' + last_folder + '/'
    # + '_fourier_response_images/'
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    # Load images
    images = []
    for filename in os.listdir(data_folder):
        if filename.endswith(".png"):
            images.append(plt.imread(data_folder + filename))
    images = np.array(images)

    # Extract Fourier response
    fourier_responses = []
    for image in images:
        fourier_response = fftshift(fft2(image))
        fourier_responses.append(fourier_response)

    # To np array
    fourier_responses = np.array(fourier_responses)

    # Flatten fourier responses and make to real values
    flattened_fourier_responses = np.zeros(shape=(len(fourier_responses), 784))
    for i, fourier_response in enumerate(fourier_responses):
        flattened = flatten(fourier_response)
        flattened = flattened.real
        for a, value in enumerate(flattened):
            flattened_fourier_responses[i][a] = flattened[a]

    # Save the flattened fourier responses
    savemat(output_folder + 'fourier_responses.mat', {'fourier_responses': flattened_fourier_responses})


# For each folder in data/Digit_testing/ extract fourier response for images
def extract_fourier_response_for_all_folders_in_folder(data_folder):
    """Extract Fourier response from all images in the data folder. """
    # Load paths to all folders.
    for folder in os.listdir(data_folder):
        print(folder)
        # For each folder create output folder and extract fourier response to it.
        if os.path.isdir(data_folder + folder):
            extract_fourier_response_for_one_folder(data_folder + folder + '/')

# If main run this file.
if __name__ == '__main__':
    # Creating the matrices for the training and testing data
    extract_fourier_response_for_all_folders_in_folder('Digit_training/')
    extract_fourier_response_for_all_folders_in_folder('Digit_testing/')





