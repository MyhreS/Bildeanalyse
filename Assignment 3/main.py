
"""Extract Fourier response from all images in the data folder. """

# TODO: Make directory class and make it iterate down the directory tree
def extract_fourier_response(data_folder, output_folder, scipy=None):
    """Extract Fourier response from all images in the data folder. """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.fftpack import fft2, fftshift
    from scipy.io import savemat

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

    # Save Fourier responses
    fourier_responses = np.array(fourier_responses)
    savemat(output_folder + 'fourier_responses.mat', {'fourier_responses': fourier_responses})
    print(fourier_responses[0])

    # Flattern Fourier responses
    flattened_fourier_responses = []
    for fourier_response in fourier_responses:
        flattened_fourier_responses.append(fourier_response.flatten())
        flattened_fourier_responses = np.array(flattened_fourier_responses)

    """
    # Makes the flattened fourier responses in range of 0 to 1
    for flattened_fourier_response in flattened_fourier_responses:
        maximum = np.max(flattened_fourier_response)
        print(maximum)
        for index in range(len(flattened_fourier_response)):
            flattened_fourier_response[index] = flattened_fourier_response[index] / maximum
    """




    # Plot/Save Fourier responses
    for i in range(len(fourier_responses)):
        plt.figure(figsize=(8, 8))
        plt.imshow(np.abs(fourier_responses[i]), cmap='gray')
        plt.savefig(output_folder + 'fourier_response_' + str(i) + '.png')
        plt.close()

    return None

extract_fourier_response('data/test/', 'data_output/')

