import cv2
import numpy as np
import os

IMG_PATHS = ["images/chess.png", "images/33039.jpg", "images/33066.jpg", "images/78004.jpg"]

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

def non_max_suppression(magnitude, Ix_prime, Iy_prime):
    # This function will remove pixels that are not the maximum along the
    # gradient direction so the edges become only one pixel thick.
    rows, cols = magnitude.shape
    nms_img = np.zeros((rows, cols), dtype=float)

    # Gradient direction in degrees, shifted so everything is between 0 and 180
    angle = np.arctan2(Iy_prime, Ix_prime) * (180.0 / np.pi)
    angle[angle < 0] += 180.0

    # Go through every pixel except the border and check its two neighbors
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            # Pick the two neighbors that go across the edge. I snap the angle
            # to the nearest of the 4 main directions.
            if angle[i, j] < 22.5 or angle[i, j] >= 157.5:
                n1 = magnitude[i, j - 1]
                n2 = magnitude[i, j + 1]
            elif angle[i, j] < 67.5:
                n1 = magnitude[i + 1, j - 1]
                n2 = magnitude[i - 1, j + 1]
            elif angle[i, j] < 112.5:
                n1 = magnitude[i + 1, j]
                n2 = magnitude[i - 1, j]
            else:
                n1 = magnitude[i + 1, j + 1]
                n2 = magnitude[i - 1, j - 1]

            # Keep the pixel only if it is bigger than both of its neighbors
            if magnitude[i, j] >= n1 and magnitude[i, j] >= n2:
                nms_img[i, j] = magnitude[i, j]

    return nms_img

def hysteresis(img):
    # This function will label pixels as strong or weak using two thresholds and
    # then keep the weak pixels that are connected to a strong one.
    high_threshold = img.max() * 0.15
    low_threshold = high_threshold * 0.05

    # Masks for the strong pixels and for everything that is at least weak
    strong = (img >= high_threshold)
    candidates = (img >= low_threshold)

    # Label the connected components of all candidate pixels
    _, labels = cv2.connectedComponents(candidates.astype(np.uint8), connectivity=8)

    # Find the component labels that contain at least one strong pixel
    strong_labels = np.unique(labels[strong])
    strong_labels = strong_labels[strong_labels != 0]

    # Keep every pixel that belongs to one of those components
    final_edges = np.zeros(img.shape, dtype=np.uint8)
    final_edges[np.isin(labels, strong_labels)] = 255

    return final_edges

def canny_pipeline(img, sigma):
    # This function will run the whole canny pipeline for a single image and sigma
    # and return each intermediate result plus the final edge map in order.
    # Generate the 1d gaussian and gaussian derivative kernels
    g = get_1d_gaussian_kernel(sigma)
    g_prime = get_1d_gaussian_kernel_derivative(sigma)

    # (a) convolve with the gaussian in x
    IxGx = convolution_1d(img, g, axis='horizontal')

    # (b) convolve with the gaussian in y
    IyGy = convolution_1d(img, g, axis='vertical')

    # (c) convolve with the derivative of the gaussian x
    Ix_prime = convolution_1d(IyGy, g_prime, axis='horizontal')

    # (d) convolve with the derivative of the gaussian y
    Iy_prime = convolution_1d(IxGx, g_prime, axis='vertical')

    # (e) magnitude 
    magnitude = np.sqrt(Ix_prime**2 + Iy_prime**2)

    # (f) NMS
    nms_out = non_max_suppression(magnitude, Ix_prime, Iy_prime)

    # hysteresis thresholding
    final_edges = hysteresis(nms_out)

    return IxGx, IyGy, Ix_prime, Iy_prime, magnitude, nms_out, final_edges

def add_label(tile, text):
    # This function will draw a short label on the top left of a panel
    bgr = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
    cv2.putText(bgr, text, (8, 26),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return bgr

def show_pipeline_figure(img, sigma, title=None, save_path=None):
    # This function will run the pipeline for one image and plot the input, 
    # intermediate, and final results in a 2x4 grid.
    IxGx, IyGy, Ix_prime, Iy_prime, magnitude, nms_out, final_edges = canny_pipeline(img, sigma)

    # Normalize the float outputs so they can be shown as images
    disp_a = cv2.normalize(IxGx, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    disp_b = cv2.normalize(IyGy, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    disp_c = cv2.normalize(Ix_prime, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    disp_d = cv2.normalize(Iy_prime, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    disp_e = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    disp_f = cv2.normalize(nms_out, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    final = final_edges

    # Label every panel so each step of the pipeline is easy to identify
    panels = [
        add_label(img, "input"),
        add_label(disp_a, "(a) Ix*Gx"),
        add_label(disp_b, "(b) Iy*Gy"),
        add_label(disp_c, "(c) Ix'"),
        add_label(disp_d, "(d) Iy'"),
        add_label(disp_e, "(e) magnitude"),
        add_label(disp_f, "(f) non-max suppression"),
        add_label(final, "final edges"),
    ]
    top_row = np.hstack(panels[:4])
    bottom_row = np.hstack(panels[4:])
    grid = np.vstack((top_row, bottom_row))

    cv2.imwrite(save_path, grid)
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_sigma_comparison(img, sigmas, title=None, save_path=None):
    # This function will run the pipeline with several sigma values and stack the
    # final edge maps side by side so the effect of sigma can be compared.
    # This is done seperately since the pipeline figure only computes 1 sigma value.
    edge_maps = []
    for sigma in sigmas:
        _, _, _, _, _, _, final_edges = canny_pipeline(img, sigma)
        # Label each column with the sigma value it was produced with
        edge_maps.append(add_label(final_edges, "sigma=" + str(sigma)))

    grid = np.hstack(edge_maps)

    cv2.imwrite(save_path, grid)
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return grid

def main():
    # Make a folder next to this file for the images used in the report
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fig_dir = os.path.join(script_dir, "outputs")
    os.makedirs(fig_dir, exist_ok=True)

    imgs = read_image()
    sigmas = [0.5, 1.5, 3.0]
    # After comparing the three values, sigma = 1.5 works best for these images
    # because it removes the noise while still keeping the important edges. The only image 
    # this is debatable on is 33039 due to the sheer amount of edges in the original image.
    best_sigma = 1.5

    # Run the full pipeline on every image and save/show the intermediate results
    for path, img in zip(IMG_PATHS, imgs):
        name = os.path.splitext(os.path.basename(path))[0]
        pipeline_title = name + " (sigma=" + str(best_sigma) + ")"
        comparison_title = name + " sigma comparison"

        show_pipeline_figure(img, best_sigma, title=pipeline_title, save_path=os.path.join(fig_dir, name + "_pipeline.png"))
        show_sigma_comparison(img, sigmas, title=comparison_title, save_path=os.path.join(fig_dir, name + "_sigma_comparison.png"))

if __name__ == "__main__":
    main()
