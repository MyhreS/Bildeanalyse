# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
# https://stackabuse.com/implementing-pca-in-python-with-scikit-learn/
from scipy.io import loadmat

# Function to load mat file
def load_mat(path_to_math_file):
    """Load a mat file. """
    return loadmat(path_to_math_file)['fourier_responses']