import numpy as np
import scipy.signal
from skimage import img_as_float
from skimage.color import rgb2gray


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


def DFT(signal):
    """
    This function calculates the Fourier Transform of a specific signal
    :param signal: np array of the signal (float64)
    :return: np array of the Fourier Transform of the signal, complex numbers
    """
    n = signal.shape[0]
    omega = np.exp(((((-2) * np.pi)*1j) / n))

    e_items = np.vander(omega**np.arange(n), n, True)
    fourier_signal = np.dot(e_items, signal)

    return fourier_signal.astype(np.complex128)


def IDFT(fourier_signal):
    """
    This function calculates the inverse Fourier Transform.
    :param fourier_signal: np array of complex of the Fourier signal
    :return: np array of float64 of the original signal
    """
    n = fourier_signal.shape[0]
    omega = np.exp((((2 * np.pi)*1j) / n))

    e_items = np.vander(omega**np.arange(n), n, True)
    org_signal = np.dot(e_items, fourier_signal)/n

    return org_signal


def DFT2(image):
    """
    This function calculates the Fourier Transform of a specific 2dimensional signals
    :param signal: np array of the signal (float64)
    :return: np array of the Fourier Transform of the signals, complex numbers
    """
    full_dft2 = DFT(DFT(image.transpose()).transpose())
    return full_dft2.astype(np.complex128)


def IDFT2(fourier_image):
    """
    This function calculates the inverse Fourier Transform of 2dimensional signal.
    :param fourier_signal: np array of complex of the Fourier signal
    :return: np array of float64 of the original signal
    """
    return IDFT(IDFT(fourier_image).transpose()).transpose()


def conv_der(im):
    """
    This function calculates the magnitude of derivative of an image using convolution
    :param im: The image to calculate on, 2d np array of floats
    :return: 2d np array that represents the magnitude of the derivative at each point
    """
    derevitive_conv = np.array([[1], [-1]])
    dx = scipy.signal.convolve2d(im, derevitive_conv, 'same')
    dy = scipy.signal.convolve2d(im, derevitive_conv.transpose(), 'same')
    magnitude = np.sqrt(np.abs(dx)**2 + np.abs(dy)**2)

    return magnitude.real.astype(np.float64)


def fourier_der(im):
    """
    This function calculates the magnitude of derivative of an image using Fourier transform
    :param im: The image to calculate on, 2d np array of floats
    :return: 2d np array that represents the magnitude of the derivative at each point
    """
    ft_img = DFT2(im)
    ft_img = np.fft.fftshift(ft_img)

    n_x = im.shape[1]
    coeff_x = (2 * np.pi * 1j)/n_x
    u_freq = np.array([n if n < int(n_x/2) else (n-n_x) for n in range(n_x)]) * 1j
    u_freq = np.array([np.fft.fftshift(u_freq)]*im.shape[0]).transpose()
    dx_ft = coeff_x * IDFT2(np.fft.ifftshift(u_freq.transpose() * ft_img))

    m_y = im.shape[0]
    coeff_y = (2 * np.pi * 1j)/m_y
    v_freq = np.array([m if m < int(m_y/2) else (m-m_y) for m in range(m_y)]) * 1j
    v_freq = np.array([np.fft.fftshift(v_freq)] * im.shape[1]).transpose()
    tr =  IDFT2(np.fft.ifftshift(v_freq * ft_img))
    dy_ft = coeff_y * tr

    magnitude = np.sqrt(np.abs(dx_ft)**2 + np.abs(dy_ft)**2)
    return magnitude.real.astype(np.float64)


def gaus_kernel_calc(kernel_size):
    """
    This is a helper method to calculate the correct approximation of the gaussian kernel according to its size.
    Using convolution and the binomial coefficients.
    :param kernel_size: int number, odd number
    :return: np array of (kernel_size X kernel_size) of the gaussian matrix, normalized
    """
    base_gaus_binom = np.array([[1], [1]])
    kernel = base_gaus_binom

    if kernel_size == 1:
    # If the kernel size is 1 we need a 2d array that keeps the image the same.
        kernel = np.array([[1]])
        kernel = scipy.signal.convolve2d(kernel, kernel.transpose())
        return kernel

    for i in range(kernel_size - 2):
        kernel = scipy.signal.convolve2d(kernel, base_gaus_binom)

    kernel = scipy.signal.convolve2d(kernel, kernel.transpose())
    return kernel/kernel.sum()


def blur_spatial(im, kernel_size):
    """
    This function creates blur filter with gaussian matrix, using convolution.
    :param im: the image to blur, np array of 2d floats
    :param kernel_size: int, the size of the blur
    :return: the blurred image, np array of 2d floats
    """
    kernel = gaus_kernel_calc(kernel_size)

    return scipy.signal.convolve2d(im, kernel, 'same').astype(np.float64)


def blur_fourier(im, kernel_size):
    """
    This function creates blur filter with gaussian matrix, using Fourier Transform.
    :param im: the image to blur, np array of 2d floats
    :param kernel_size: int, the size of the blur
    :return: the blurred image, np array of 2d floats
    """
    kernel = gaus_kernel_calc(kernel_size)

    zeros = np.zeros(im.shape)
    x_mid = np.math.floor(im.shape[1] / 2)
    y_mid = np.math.floor(im.shape[0] / 2)
    distance = np.math.floor(kernel_size / 2)
    zeros[x_mid - distance: x_mid + distance + 1,  y_mid - distance: y_mid + distance + 1] = kernel

    fourier_kernel = DFT2(np.fft.ifftshift(zeros))
    fourier_img = DFT2(im)
    fourier_blured = fourier_kernel * fourier_img

    return IDFT2(fourier_blured).real.astype(np.float64)
