import scipy
import scipy.signal
from scipy import misc, ndimage
import numpy as np
from skimage import img_as_float
from skimage.color import rgb2gray
import os
import matplotlib.pyplot as plt


def read_image(filename, representation):
    """
    This function reads an image from path
    :param filename: path of the file
    :param representation: 1 if GrayScale, 0 if RGB
    :return: float array of the image
    """
    image = scipy.misc.imread(filename)
    if int(representation) == 1:
        image = rgb2gray(image)
    return img_as_float(image)


def gaus_kernel_calc(kernel_size):
    """
    This is a helper method to calculate the correct approximation of the gaussian kernel according to its size.
    Using convolution and the binomial coefficients.
    :param kernel_size: int number, odd number
    :return: np array of (kernel_size X kernel_size) of the gaussian matrix, normalized
    """
    base_gaus_binom = np.array([[1], [1]])
    kernel_arr = base_gaus_binom
    if kernel_size == 1:
    # If the kernel size is 1 we need a 2d array that keeps the image the same.
        return np.array([[1]]).transpose()
    for i in range(kernel_size - 2):
        kernel_arr = scipy.signal.convolve2d(kernel_arr, base_gaus_binom)
    if kernel_arr.sum() == 0:
        return kernel_arr.transpose()
    return (kernel_arr/kernel_arr.sum()).transpose()


def blur_spatial(im, kernel):
    """
    This function creates blur filter with gaussian matrix, using convolution.
    :param im: the image to blur, np array of 2d floats
    :param kernel: the filter kernel
    :return: the blurred image, np array of 2d floats
    """
    curr = scipy.ndimage.filters.convolve(im, kernel.transpose())
    return scipy.ndimage.filters.convolve(curr, kernel)


def subsample(im):
    """
    Returns all the even cells in the image, both in rows and columns.
    :param im: array of greyscale pixels
    :type im: 2d array
    :return: array of even position from original image
    :rtype: 2d array
    """
    return im[1::2, 1::2]


def up_sample(im):
    """
    Returns the image padded with zeros in the odd positions.
    :param im: array of greyscale pixels
    :type im: 2d array
    :return: array of even position from original image
    :rtype: 2d array
    """
    up_sampled = np.zeros((im.shape[0]*2, im.shape[1]*2))
    up_sampled[1::2, 1::2] = im
    return up_sampled


def expand(im, doubled_kernel):
    """
    This function expands an image that was "pyramided" for the laplcian process.
    :param im: 2d array of greyscale image
    :type im: 2d array
    :param doubled_kernel: 2d array of the kernel to blur the image with after up sampling.
    :type doubled_kernel: 2d array
    :return: the expanded image
    :rtype: 2d array
    """
    up_sampled = up_sample(im)
    expanded = blur_spatial(up_sampled, doubled_kernel)
    return expanded


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    Constructs a Gaussian pyramid of a given image
    :param im: A greyscale image with float values in [0, 1]
    :type im: float values in [0, 1]
    :param max_levels: The maximal number of levels in the resulting pyramid.
    :type max_levels: int
    :param filter_size: The size of the Gaussian filter
    :type filter_size: int, an odd scalar that represents a squared filter
    :return:
        pyr: resulting pyramid pyr as a standard python array
            with maximum length of max_levels, where each element of the array is a greyscale
            image.
        filter_vec: 1D-row of size filter_size used for the pyramid construction.
    :rtype:
        pyr: standard python array
        filter_vec: 1D-row of size filter_size
    """
    pyr = [None]*max_levels
    pyr[0] = im
    amount = 1
    kernel_arr = gaus_kernel_calc(filter_size)
    curr_im = im
    while amount <= (max_levels - 1) and curr_im.size > 16 and kernel_arr.size <= curr_im.size:
        blurred = blur_spatial(curr_im, kernel_arr)
        curr_im = subsample(blurred)
        pyr[amount] = curr_im
        amount += 1

    return pyr[:amount], kernel_arr


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    Constructs a Laplacian pyramid of a given image
    :param im: A greyscale image with double values in [0, 1]
    :type im: double values in [0, 1]
    :param max_levels: The maximal number of levels in the resulting pyramid.
    :type max_levels: int
    :param filter_size: The size of the Gaussian filter
    :type filter_size: int, an odd scalar that represents a squared filter
    :return:
        pyr: resulting pyramid pyr as a standard python array
            with maximum length of max_levels, where each element of the array is a greyscale
            image.
        filter_vec: 1D-row of size filter_size used for the pyramid construction.
    :rtype:
        pyr: standard python array
        filter_vec: 1D-row of size filter_size
    """
    gaussian_pyr, kernel_arr = build_gaussian_pyramid(im, max_levels, filter_size)
    kernel_arr = gaus_kernel_calc(filter_size)
    double_ker = 2*kernel_arr
    pyr = [None]*max_levels
    amount = 0
    curr_im = im
    while amount <= (len(gaussian_pyr)-2) and curr_im.size > 16 and kernel_arr.size <= curr_im.size:
        curr_im = gaussian_pyr[amount] - expand(gaussian_pyr[amount+1], double_ker)
        pyr[amount] = curr_im
        amount += 1

    pyr[amount] = gaussian_pyr[amount]
    return pyr[:amount + 1], kernel_arr

def laplacian_to_image(lpyr, filter_vec, coeff):
    """
    This function is responsible for creating the image back from the laplcian pyramids, while implementing the coeff
    :param lpyr: pyramid as a standard python array where each element of the array is a greyscale image.
    :type lpyr: standard python array
    :param filter_vec: 1D-row used for the pyramid construction.
    :type filter_vec: 1D-row array
    :param coeff: vector. The vector size is the same as the number of levels in the pyramid lpyr.
                Before reconstructing the image img we multiply each level i of the laplacian pyramid by its
                corresponding coefficient coeff[i].
    :type coeff: 1d array
    :return: img, greyscale image
    :rtype: 2d array
    """
    amount = len(lpyr) - 1
    lap_pyr = (np.array(lpyr))*coeff
    orig_img = lap_pyr[amount]
    for i in range(amount, 0, -1):
        orig_img = expand(orig_img, 2 * filter_vec) + lap_pyr[i - 1]
    return orig_img


def render_pyramid(pyr, levels):
    """
    This method streces the pyramids values and puts them together in one image one next to each other.
    :param pyr:  pyramid pyr as a standard python array
            with maximum length of max_levels, where each element of the array is a greyscale
            image.
    :type pyr: standard python array
    :param levels: amount of levels to render
    :type levels: int positive
    :return: the render image
    :rtype: 2d array
    """
    org_size = pyr[0].shape
    rend = (pyr[0] - np.min(pyr[0]))/np.max(pyr[0])
    for py in pyr[1:levels]:
        zer = np.zeros((org_size[0], py.shape[1]))
        min_v = np.min(py)
        max_v = np.max(py)
        zer[:py.shape[0]:] = (py - min_v)/max_v
        rend = np.hstack([rend, zer])
    return rend


def display_pyramid(pyr, levels):
    """
    Displays a pyramid process
    This method streces the pyramids values and puts them together in one image one next to each other.
    :param pyr:  pyramid pyr as a standard python array
            with maximum length of max_levels, where each element of the array is a greyscale
            image.
    :type pyr: standard python array
    :param levels: amount of levels to render
    :type levels: int positive
    """
    rend = render_pyramid(pyr, levels)
    plt.figure()
    plt.imshow(rend, cmap='gray')


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """
    This method is creating a blended image of img1 and img2 according to the given mask and filters.
    :param im1: An image
    :type im1: 2d array
    :param im2: An image
    :type im2: 2d array
    :param mask: An image of bool - true or false (1,0), for which part is which image.
    :type mask: 2d array
    :param max_levels: the maximum levels for the pyramids and process
    :type max_levels: int
    :param filter_size_im: odd number for the filter size
    :type filter_size_im: int
    :param filter_size_mask: odd number for the filter size
    :type filter_size_mask: int
    :return: image, the blended new image
    :rtype: 2d array
    """
    l1, ker_arr = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    l2, ker_arr = build_laplacian_pyramid(im2, max_levels, filter_size_im)
    gm, ker_arr_mask = build_gaussian_pyramid(mask.astype(np.float64), max_levels, filter_size_mask)
    l_out = [None]*len(l2)
    for k in range(len(l_out)):
        l_out[k] = gm[k]*l1[k] + (1 - gm[k])*l2[k]
    blended = laplacian_to_image(l_out, ker_arr, [1]*len(l_out))
    return blended.clip(0, 1)


def relpath(filename):
    """
    This method returns the relative path
    :param filename: the filename
    :type filename: string
    :return: relative path of filename
    :rtype: string
    """
    return os.path.join(os.path.dirname(__file__), filename)

def blending_example1():
    """
    This is the first example, pugs and sunflowers
    :return: im_1, im_2, mask, blended
    :rtype: as described in the pdf
    """
    factor = 1
    im_1 = read_image(relpath('pug_sized.jpg'), 0)
    im_2 = read_image(relpath('sunf_size.jpg'), 0)
    mask = read_image(relpath('maskforsunflower2.jpg'), 1).astype(bool)
    blended = np.zeros(im_1.shape)
    for i in range(3):
        chanel = pyramid_blending(factor*im_1[:, :, i], im_2[:, :, i], 1 - mask, 5, 19, 19)
        blended[:, :, i] = chanel
    plt.figure(1)
    plt.subplot(221)
    plt.imshow(im_1)
    plt.subplot(222)
    plt.imshow(im_2)
    plt.subplot(223)
    plt.imshow(mask, cmap='gray')
    plt.subplot(224)
    plt.imshow(blended)
    plt.show()
    return im_1, im_2, mask, blended


def blending_example2():
    """
    This is the first example, pugs and sunflowers
    :return: im_1, im_2, mask, blended
    :rtype: as described in the pdf
    """
    factor = 1
    im_1 = read_image(relpath('minSuf_size.jpg'), 0)
    im_2 = read_image(relpath('chris_tree_size.jpg'), 0)
    mask = read_image(relpath('minsuf_mask_size.jpg'), 1).astype(bool)
    blended = np.zeros(im_1.shape)
    for i in range(3):
        chanel = pyramid_blending(factor*im_1[:, :, i], im_2[:, :, i], 1 - mask, 4, 19, 19)
        blended[:, :, i] = chanel
    plt.figure(2)
    plt.subplot(221)
    plt.imshow(im_1)
    plt.subplot(222)
    plt.imshow(im_2)
    plt.subplot(223)
    plt.imshow(mask, cmap='gray')
    plt.subplot(224)
    plt.imshow(blended)
    plt.show()
    return im_1, im_2, mask, blended


