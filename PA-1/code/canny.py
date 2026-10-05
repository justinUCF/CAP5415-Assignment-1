import cv2
import numpy as np

IMG_PATHS = ["images/chess.png"]

def read_image():
    # This function will read in an image from the BSDS300 dataset and return it
    # as a greyscale matrix.
    imgs = []
    for img_path in IMG_PATHS:
        imgs.append(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE))
    return imgs

def convolution_1d(img, kernel, axis='horizontal'):
    # This function will produce a matrix which has been convolved with the provided 1d kernel
    # axis = horizontal: apply the kernel horizontally across the image.
    # axis = vertical: apply the kernel vertical across the image.
    # Uses zero padding on edges
    klen = len(kernel)
    kradius = klen // 2
    rows, cols = img.shape
    # Create new empty image for convolution output
    new_img = np.zeros((rows, cols), dtype=float)

    if axis == 'horizontal':
        # Create padded empty canvas for convolution
        canvas = np.zeros((rows, cols+(kradius*2)), dtype=float)
        # Fill center of canvas with original image
        canvas[:, kradius:-kradius] = img
        # Slide the kernel horizontally 
        for i in range(klen):
            new_img += canvas[:, i:i+cols] * kernel[i]

    elif axis == 'vertical':
        canvas = np.zeros((rows+(kradius*2), cols), dtype=float)
        # Fill center of canvas with original image
        canvas[kradius:-kradius,:] = img
        # Slide the kernel vertically 
        for i in range(klen):
            new_img += canvas[i:i+rows, :] * kernel[i]

    else:
        raise ValueError("axis doesnt match options")

    return new_img

def get_1d_gaussian_kernel(sigma):
    # This funciton will produce a one-dimensional gaussian kernel G(x) based on the 
    # formula presented during lectures and will determine the kernels size using the 
    # 65-95-99 rule.
    # Establish base case
    if sigma <= 0:
        raise ValueError("Sigma <= 0")
    # Caculate the radius of the kernel
    kradius = np.ceil(3*sigma)
    # Use np.arrange to create a 1d array based on the radius
    x = np.arange(-kradius, kradius+1, dtype=float)
    # Implement the gaussian function G(x) = exp(-x^2/2sigma^2) / sqrt(2pi)sigma
    g = np.exp((-x**2) / (2*(sigma**2))) / (np.sqrt(2*np.pi)*sigma)
    # Normalize the new kernel and return it
    return (g / np.sum(g))

def get_1d_gaussian_kernel_derivative(sigma):
    # This function produces a one-dimensional gaussian derivative kernel G'(x) based 
    # on the formula presented in class and applies the same 65-95-99 rule as G(x)
    # Establish base case
    if sigma <= 0:
        raise ValueError("Sigma <= 0")
    # Caculate the radius of the kernel
    kradius = int(np.ceil(3 * sigma))
    # Use np.arrange to create a 1d array based on the radius
    x = np.arange(-kradius, kradius + 1, dtype=float)
    # Get standard Gaussian G(x)
    g = get_1d_gaussian_kernel(sigma)
    # Apply derivative formula: G'(x) = (-x / sigma^2) * G(x)
    g_prime = (-x / (sigma**2)) * g
    
    return g_prime

def main():
    img = read_image()[0]
    sigma = 1.5  # Realistic sigma value for clear edges

    # Generate 1D Kernels
    g = get_1d_gaussian_kernel(sigma)
    g_prime = get_1d_gaussian_kernel_derivative(sigma)

    # 1. (a) Ix * Gx
    IxGx = convolution_1d(img, g, axis='horizontal')
    
    # 2. (b) Iy * Gy
    IyGy = convolution_1d(img, g, axis='vertical')
    
    # 3. (c) Ix'
    Ix_prime = convolution_1d(IyGy, g_prime, axis='horizontal')
    
    # 4. (d) Iy'
    Iy_prime = convolution_1d(IxGx, g_prime, axis='vertical')
    
    # 5. (e) Gradient Magnitude
    magnitude = np.sqrt(Ix_prime**2 + Iy_prime**2)

    # Normalize intermediate float outputs to 0-255 uint8 for visual display
    disp_a = cv2.normalize(IxGx, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    disp_b = cv2.normalize(IyGy, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    disp_c = cv2.normalize(Ix_prime, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U) # Neutral gray zero-point
    disp_d = cv2.normalize(Iy_prime, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U) # Neutral gray zero-point
    disp_e = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    blank_f = np.zeros_like(img)  # Placeholder for Step 6 NMS

    # Stack to replicate 2x3 Grid from Figure 0.1 in PDF
    top_row = np.hstack((disp_a, disp_b, disp_c))
    bottom_row = np.hstack((disp_d, disp_e, blank_f))
    grid = np.vstack((top_row, bottom_row))

    cv2.namedWindow("Figure 0.1: Canny Pipeline Steps", cv2.WINDOW_NORMAL)
    cv2.imshow("Figure 0.1: Canny Pipeline Steps", grid)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()