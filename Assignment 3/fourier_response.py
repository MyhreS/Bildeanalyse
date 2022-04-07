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


def extract_fourier_response_for_one_folder(data_folder, output_images=False):
    """Extract Fourier response from all images in the data folder. """

    output_folder = data_folder
    if output_images is True:
        """Create output folder if it does not exist. """
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

    # Makes the flattened fourier responses in range of 0 to 1
    for i in range(len(flattened_fourier_responses)):
        maximum = np.max(flattened_fourier_responses[i])
        for a in range(len(flattened_fourier_responses[i])):
            flattened_fourier_responses[i][a] = flattened_fourier_responses[i][a] / maximum

    # Save the flattened fourier responses
    savemat(output_folder + 'fourier_responses.mat', {'fourier_responses': flattened_fourier_responses})

    if output_images is True:
        # Plot/Save Fourier responses
        for i in range(len(fourier_responses)):
            plt.figure(figsize=(8, 8))
            plt.imshow(np.abs(fourier_responses[i]), cmap='gray')
            plt.savefig(output_folder + 'fourier_response_' + str(i) + '.png')
            plt.close()


# For each folder in data/Digit_testing/ extract fourier response for images
def extract_fourier_response_for_all_folders_in_folder(data_folder, output_folder=False):
    """Extract Fourier response from all images in the data folder. """
    # Load images
    for folder in os.listdir(data_folder):
        print(folder)
        if os.path.isdir(data_folder + folder):
            extract_fourier_response_for_one_folder(data_folder + folder + '/', output_folder)

#extract_fourier_response_for_all_folders_in_folder('data/Digit_testing/', False)
#extract_fourier_response_for_all_folders_in_folder('data/Digit_testing/', True)
extract_fourier_response_for_all_folders_in_folder('data/Digit_training/', False)
extract_fourier_response_for_all_folders_in_folder('data/Digit_training/', True)



