

"""Extract Fourier response from all images in the data folder. """

def extract_fourier_response(data_folder, output_folder, scipy=None):
    """Extract Fourier response from all images in the data folder. """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.fftpack import fft2, ifft2, fftshift
    from scipy.ndimage import gaussian_filter
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

    # Plot Fourier responses
    for i in range(len(fourier_responses)):
        plt.figure(figsize=(8, 8))
        plt.imshow(np.abs(fourier_responses[i]), cmap='gray')
        plt.savefig(output_folder + 'fourier_response_' + str(i) + '.png')
        plt.close()

    return None

extract_fourier_response('data/', 'data_output/')
