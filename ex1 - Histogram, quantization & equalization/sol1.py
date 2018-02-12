import sys
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
# from skimage.color import yiq2rgb
from skimage import img_as_float

COLOR_SIZE = 255
BIN_AMOUNT = 256
RGB_SHAPE_LEN = 3
RGB2YIQMatrix = np.array([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]])
YIQ2RGBMatrix = np.array([[1, 0.956, 0.621], [1, -0.272, -0.647], [1, -1.106, 1.703]])


def read_image(filename, representation):
    """
    This function is responsible on reading an image from a path
    :param filename: The path of the image
    :type filename: String
    :param representation: for RGB = 2, for GrayScale = 1
    :type representation: int number
    :return: The image by pixels
    :rtype: np array, float 64 (3d if RGB, for grayScale 2d)
    """
    image = scipy.misc.imread(filename, False, 'RGB')
    if int(representation) == 1:
        image = rgb2gray(image)
    return img_as_float(image)


def imdisplay(filename, representation):
    """
    Displays a given image by path
    :param filename: path of the image
    :type filename: String
    :param representation: for RGB = 2, for GrayScale = 1
    :type representation: int number
    :return: None
    :rtype: None
    """
    image = read_image(filename, representation)
    plt.imshow(image)
    plt.show()
    return


def rgb2yiq(imRGB):
    """
    This function is responsible for transforming a color space of an image, from RGB to YIQ
    :param imRGB: Original image to transform
    :type imRGB: np 3d array
    :return: The image in YIQ color space
    :rtype: np 3d array
    """
    # Reshaping the image vectors so matrix multiplication would be possible. Therefore also transposing, so we'll have
    # 2d array of the matrix of RGB space vectors (to understand, first vector is in [:,0])
    mid_calc = RGB2YIQMatrix.dot(np.reshape(imRGB.transpose(2, 0, 1), (RGB_SHAPE_LEN, -1)))
    return np.reshape(mid_calc.transpose(), imRGB.shape)


def yiq2rgb(imYIQ):
    """
    This function is responsible for transforming a color space of an image, from YIQ to RGB
    :param imYIQ: Original image to transform
    :type imYIQ: np 3d array
    :return: The image in RGB color space
    :rtype: np 3d array
    """
    # Reshaping the image vectors so matrix multiplication would be possible. Therefore also transposing, so we'll have
    # 2d array of the matrix of RGB space vectors (to understand, first vector is in [:,0])
    mid_calc = YIQ2RGBMatrix.dot(np.reshape(imYIQ.transpose(2, 0, 1), (RGB_SHAPE_LEN, -1)))
    return np.reshape(mid_calc.transpose(), imYIQ.shape)


def normalize_n_stretch_cum_histogram(histogram, size):
    """
    Normalizes and stretches the histogram. If no stretch is needed than it wouldn't affect
    :param histogram: npArray of the cumulative histogram to normalize
    :type histogram: npArray
    :param size: the size of the image
    :type size: int
    :return: normed and stretched cumulative histogram, rounded
    :rtype: npArray
    """
    # Normalizing
    normed = histogram / size
    normed = (normed * COLOR_SIZE).astype(int)
    first_non_zero = np.argmax(normed > 0)
    max_normed = normed.max()
    # Checking if stretch is needed
    if first_non_zero != 0:
        stretched = normed - normed[first_non_zero]
        stretched = stretched / (max_normed - first_non_zero)
        stretched = stretched * COLOR_SIZE
    else:
        stretched = normed
    return stretched


def histogram_equalize_help(image):
    """
    Helpper method. Receives the only channel to be changed (Y channel in yiq for RGB, or simply GrayScale)
    Recieves it in already between [0,255] (2 dimensioned)
    :param image: npArray, the channel to be changed
    :type image: np array
    :return: channel equalized
    :rtype: npArray
    """
    imageSize = (image.shape[0]*image.shape[1])

    orig_histogram, bins = np.histogram(image, BIN_AMOUNT, (0, COLOR_SIZE))
    orig_cum_histogram = np.cumsum(orig_histogram)

    orig_normedNstreched_cum_histogram = normalize_n_stretch_cum_histogram(orig_cum_histogram, imageSize)
    # Using the normed and stretched histogram as lookup table
    return [orig_normedNstreched_cum_histogram.astype(int)[image.astype(int)]]


def histogram_equalize(im_orig):
    """
    This function is responsible for histogram equalization process
    :param im_orig: image in rgb or grayscale, which it's values are between [0,1]
    :return: equalized version of the image, image numpy array
    """
    im_eq = filter_factory(im_orig, histogram_equalize_help)[0].clip(0, 1)
    origin_histog = np.histogram(rgb2yiq(im_orig * COLOR_SIZE)[:,:,0], BIN_AMOUNT, (0, COLOR_SIZE))[0]
    equalized_histog = np.histogram(im_eq, BIN_AMOUNT, (0, COLOR_SIZE))[0]
    return im_eq, origin_histog, equalized_histog


def filter_factory(im_orig, filterFunction):
    """
    Receives an image in (0,1) color values, and a filter function to apply and returns the unclipped result.
    The function should work on an image of  already between [0,255] (2 dimensioned) values
    :param im_orig: The image to apply the filter on
    :type im_orig: np array of 2d
    :param filterFunction: Function which returns 2d np array of the filtered image,
                            and works on the Y chanel or GrayScale, inside an array of results, if other results needed.
                            The filtered image itself is in position 0.
    :type filterFunction: Function
    :return: The image with the filter on, not clipped, but values that would match (0,1) image if clipped
    :rtype: 2d for GrayScale, 3d for RGB, np array fo the image
    """
    image_dimension = len(im_orig.shape)
    image_255_bins = im_orig * COLOR_SIZE

    # If RGB, than calculate for Y channel in YIQ
    if image_dimension == RGB_SHAPE_LEN:
        yiqImage = rgb2yiq(image_255_bins)
        yChannel = yiqImage[:, :, 0]
    else:
        # Else=GrayScale, it is for the whole image
        yChannel = image_255_bins
    filtered_channel = filterFunction(yChannel)

    # If RGB, than put the Y channel back in place, and convert back to RGB
    if image_dimension == RGB_SHAPE_LEN:
        yiq_eq = yiqImage
        yiq_eq[:, :, 0] = filtered_channel[0]
        im_filtered = yiq2rgb(yiq_eq)
    else:
        # Else=GrayScale, the whole image
        im_filtered = filtered_channel[0]
    np.float64(im_filtered)
    filtered_channel[0] = (im_filtered / COLOR_SIZE)
    return filtered_channel


def quantize_first_partition(image, n_quant):
    """
    Given an Image-channel, which is already with values of [0,255] this function calculates the first partition for the
    quantize process
    :param image: Image-channel with values between [0,255]
    :type image: np array
    :param n_quant: The amount of parts required
    :type n_quant: int
    :return: z = the partition, np array of size (n_quant + 1), when the first is 0, and the last is COLOR_SIZE
    :rtype: np array 1d
    """
    histogram, bins = np.histogram(image, BIN_AMOUNT, (0, COLOR_SIZE))
    cum_histogram = np.cumsum(histogram)
    amount = cum_histogram.max() / n_quant
    z = np.zeros((n_quant + 1), 'int')
    z[-1] = COLOR_SIZE

    for i in range(n_quant):
        z[i] = np.argmax(cum_histogram >= (i * amount))
    return z.round()


def quantize_by_channel_factory(n_quant, n_iter, z):
    """
    This is a factory gor "quantize_by_channel" function.
    Receives the base parameters for the quantize process.
    :param n_iter: The amount of iterations to perform in case there is no convergence
    :type n_iter: int
    :param n_quant: The amount of color, color quantity wanted
    :type n_quant: int
    :param z: The first partition of the histogram of the image, required for the quantize process, calculated at
                "quantize_first_partition", it is for the Y channel or GrayScale, with values between [0,255]
    :type z: np array 1d
    :return: quantize_by_channel(channel) function, used by the filter factory
    :rtype: function which recieves a channel\GrayScale image and performs the quantize process according to the variables
            received in this function ("quantize_by_channel_factory")
    """
    def quantize_by_channel(channel):
        """
        This function receives a channel\greyscale image and calculates the quantization according to predefined, given
        values/parameters. It receives an image/channel with values already between [0,255]
        :param channel: The image/channel to quantize, values already between [0,255]
        :type channel: np 2d array
        :return: The quantized channel in values between [0,255]
        :rtype: np 2d array
        """
        iterations, quantity = n_iter, n_quant
        levels = np.zeros(quantity, np.float64)
        errors = np.zeros(iterations, np.float64)
        error_index = 0
        isConverged = False
        histogram, bins = np.histogram(channel, BIN_AMOUNT, (0, COLOR_SIZE))
        while iterations and not isConverged:
            old_z = z.copy()
            for i in range(quantity):
                levels[i] = np.average(np.arange(z[i], z[i+1]), weights=histogram[z[i]:z[i+1]])
                part_error, sum_wh = np.average((np.square(levels[i] - np.arange(z[i], z[i+1]))), weights=histogram[z[i]:z[i+1]], returned=True)
                errors[error_index] += part_error * sum_wh
                if i != 0:
                    z[i] = (levels[i-1] + levels[i])/2
            # Checking if Convergence according to the given definition
            isConverged = np.array_equal(old_z, z)
            error_index += 1
            iterations -= 1
        for index in range(quantity):
            histogram[z[index]:z[index + 1]] = levels[index]
        errors = np.trim_zeros(errors)
        return [histogram[channel.astype(int)], errors]
    return quantize_by_channel


def quantize(im_orig, n_quant, n_iter):
    """
    Responsible for the quantize process of an image.
    :param im_orig: The image to quantize
    :type im_orig: np array of 2d if GrayScale or 3d if RGB
    :param n_quant: The amount of color shades wanted
    :type n_quant: int
    :param n_iter: Maximum amount of iterations in case there is no convergence in the process
    :type n_iter: int
    :return: the quantized image
    :rtype: np array of 2d if GrayScale or 3d if RGB
    """
    initial_z = quantize_first_partition((im_orig * COLOR_SIZE), n_quant)
    quantized_image = filter_factory(im_orig, quantize_by_channel_factory(n_quant, n_iter, initial_z))
    return quantized_image

