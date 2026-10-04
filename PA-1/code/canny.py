import cv2
import numpy as np

IMG_PATHS = ["images/78004.jpg"]

def read_image():
    # This function will read in an image from the BSDS300 dataset and return it.
    imgs = []
    for img_path in IMG_PATHS:
        imgs.append(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE))
    return imgs

def get_1d_gaussian_kernel(sigma):
    # This funciton will produce a one-dimensional gaussian kernel based on the 
    # formula presented during lectures and will determine the kernels size using the 
    # 65-95-99 rule.

    # Establish base case
    if sigma <= 0:
        raise ValueError("Sigma < 0=")
    # Caculate the radius of the kernel
    kradius = np.ceil(3*sigma)
    # Use np.arrange to create a 1d array based on the radius
    x = np.arange(-kradius, kradius+1, dtype=float)
    # Implement the gaussian function G(x) = exp(-x^2/2sigma^2) / sqrt(2pi)sigma
    g = np.exp((-x**2) / (2*(sigma**2))) / (np.sqrt(2*np.pi)*sigma)
    # Normalize the new kernel and return it
    return (g / np.sum(g))

def get_1d_gaussian_kernel_derivative():
    return

def main():
    print(get_1d_gaussian_kernel(1.0))

main()