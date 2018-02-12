import scipy
from scipy import misc, ndimage
from skimage import img_as_float
from skimage.color import rgb2gray
import numpy as np

from keras.layers import Input, Dense, Convolution2D, Activation, merge
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import History

import matplotlib.pyplot as plt

import sol5_utils



CONV_KER = 3
X = 0  # X here is rows
Y = 1  # Y here is columns
GREY = 1
DEFAULT_CHANNEL_AM = 1
TRAINING_PRECENT = 0.8
SUB_VAL = 0.5

IMG_COLORES_DIST = 1/255

PATCH_SIZE_NOISE = 24
PATCH_SIZE_BLUR = 16

CHANNELS_DENOISINING = 48
BATCH_SIZE = 100
SAMPLES_PER_EPOCH = 10000
SAMPLES_VALIDATION = 1000

QUICK_BATCH_SIZE = 10
QUICK_SAMPLES_PER_EPOCH = 30
QUICK_NUM_EPOCHS = 2
QUICK_VALIDATION_SAMPLES = 30

COLOR = {0: 'g', 1: 'b', 2: 'r', 3: 'm', 4: 'y'}


def read_image(filename, representation):
    """
    This function reads an image from path
    :param filename: path of the file
    :param representation: 1 if GrayScale, 0 if RGB
    :return: float array of the image
    """
    image = scipy.misc.imread(filename)
    if int(representation) == GREY:
        image = rgb2gray(image)
    return img_as_float(image)


def load_dataset(filenames, batch_size, corruption_func, crop_size):
    """
    This function creates a generator for random couple of patches of 'damaged' image and the same patch in
    the original image.
    :param filenames: A list of filenames of clean images
    :type filenames: list
    :param batch_size: The size of the batch of images for each iteration of Stochastic Gradient Descent
    :type batch_size: int
    :param corruption_func: A function receiving a numpy's array representing an image, and returns a randomly
    corrupted version of the input image
    # :type corruption_func: function
    :param crop_size: A tuple (height, width) specifying the crop size of the patches to extract
    :type crop_size: tuple (int, int)
    :return: Generator object which outputs random tuples of the form (source_batch, target_batch)
    :rtype: Generator object which returns tuple of two np arrays from the shape (batch_size, 1, height, width)
    """
    org_imgs = {}
    N = len(filenames)
    file_names = np.array(filenames)
    CROP_X = crop_size[X]
    CROP_Y = crop_size[Y]
    ORIGINAL = 0
    CORRUPTED = 1
    while True:
        indicis = np.random.randint(N, size=batch_size)
        curr_batch_imgs = file_names[indicis]
        source_batch = []
        target_batch = []
        for file_name in curr_batch_imgs:
            if file_name not in org_imgs.keys():
                cur_img = read_image(file_name, GREY)
                corrupted = corruption_func(cur_img)
                org_imgs[file_name] = [cur_img, corrupted]
            else:
                cur_img = org_imgs[file_name][ORIGINAL]
                corrupted = org_imgs[file_name][CORRUPTED]
            row_pos = np.random.randint(0, cur_img.shape[X]//CROP_X)
            col_pos = np.random.randint(0, cur_img.shape[Y]//CROP_Y)
            source_batch.append([corrupted[row_pos: (row_pos + CROP_X), col_pos: (col_pos + CROP_Y)] - SUB_VAL])
            target_batch.append([cur_img[row_pos: (row_pos + CROP_X), col_pos: (col_pos + CROP_Y)] - SUB_VAL])
        yield (np.array(source_batch), np.array(target_batch))


def resblock(input_tensor, num_channels):
    """
    This function receives a symbolic input tensor and the number of channels for each of its convolution layers
    and returns its symbolic output tensor.
    :param input_tensor: symbolic input tensor
    :type input_tensor:
    :param num_channels: int
    :type num_channels:
    :return:
    :rtype:
    """
    first_convolved = Convolution2D(num_channels, CONV_KER, CONV_KER, border_mode='same')(input_tensor)
    first_relu = Activation('relu')(first_convolved)
    second_convolved = Convolution2D(num_channels, CONV_KER, CONV_KER, border_mode='same')(first_relu)
    addition = merge([input_tensor, second_convolved], mode='sum')
    return addition


def build_nn_model(height, width, num_channels, num_res_blocks):
    """
    This function builds a full neural network model with input dimensions of (1, height, width)
    :param height: Height of the input (X)
    :type height: inr
    :param width: Width of the input (Y)
    :type width: int
    :param num_channels: Amount of channels within the network, the final output will be one channel
    :type num_channels: int
    :param num_res_blocks: Number of residual blockes in the netowrk
    :type num_res_blocks: int
    :return: nn model
    :rtype: Keras model
    """
    if num_res_blocks == 0:
        return
    input_layer = Input(shape=(1, height, width))
    first_conv = Convolution2D(num_channels, CONV_KER, CONV_KER, border_mode='same')(input_layer)
    first_relu = Activation('relu')(first_conv)
    resed_block = resblock(first_relu, num_channels)
    for _ in range(num_res_blocks - 1):
        resed_block = resblock(resed_block, num_channels)

    addition = merge([first_relu, resed_block], mode='sum')
    last_conv = Convolution2D(1, CONV_KER, CONV_KER, border_mode='same')(addition)

    return Model(input=input_layer, output=last_conv)


def train_model(model, images, corruption_func, batch_size, sampels_per_epoch, num_epochs, num_valid_samples):
    """
    This function trains a model.
    :param model: general neural network model for image restoration
    :type model: Model (keras)
    :param images: a list of file paths pointing to image files.
    :type images: list
    :param corruption_func: A function receiving a numpy's array representing an image, and returns a randomly
    corrupted version of the input image
    :type corruption_func:
    :param batch_size: the size of the batch of examples for each SGD iteraration
    :type batch_size: int
    :param sampels_per_epoch: The number of samples in each epoch
    :type sampels_per_epoch: int
    :param num_epochs: The number of epoch for which the optimization will run
    :type num_epochs: int
    :param num_valid_samples: The number of samples in the validation set to test on after every epoch
    :type num_valid_samples: int
    """
    N = len(images)
    images = np.array(images)
    crop_size = model.input_shape[2:]
    training_set = load_dataset(images[: int(np.floor(N*TRAINING_PRECENT))], batch_size, corruption_func, crop_size)
    validation_set = load_dataset(images[int(np.floor(N*TRAINING_PRECENT)):], batch_size, corruption_func, crop_size)
    model.compile(loss='mean_squared_error', optimizer=Adam(beta_2=0.9))
    model.fit_generator(training_set, samples_per_epoch=sampels_per_epoch, nb_epoch=num_epochs,
                        validation_data=validation_set, nb_val_samples=num_valid_samples)


def restore_image(corrupted_image, base_model):
    """
    This function returns a "restored" image according to the model given.
    :param corrupted_image: a greyscale image of shape (height, width) and
                        with values in the [0, 1] range of type float64
    :type corrupted_image: np array of 2 dimensions
    :param base_model: a neural network trained to restore small patches
                    The input and output of the network are images with values in the [−0.5, 0.5] range
    :type base_model:
    :return: Restored image
    :rtype: 2d array greyscale image
    """
    height, width = corrupted_image.shape
    img_input = Input(shape=(1, height, width))
    img_output = base_model(img_input)
    new_model = Model(input=img_input, output=img_output)
    curr_img = np.array([[corrupted_image - SUB_VAL]])
    constructed = np.array((new_model.predict(curr_img)[0] + SUB_VAL)[0]).astype(np.float64)

    return constructed.clip(0, 1)


def add_gaussian_noise(image, min_sigma, max_sigma):
    """
    This function adds a random gaussian noise to an image.
    :param image: a greyscale image with values in the [0, 1] range of type float64.
    :type image:
    :param min_sigma:  a non-negative scalar value representing the minimal variance of the gaussian distribution.
    :type min_sigma:
    :param max_sigma: a non-negative scalar value larger than or equal to min_sigma, representing the maximal
                    variance of the gaussian distribution.
    :type max_sigma:
    :return:
    :rtype:
    """
    sigma = np.random.uniform(min_sigma, max_sigma)
    corrupted = image + np.random.normal(0, sigma, size=image.shape)
    corrupted = IMG_COLORES_DIST * np.round(corrupted * 255)

    return corrupted


def learn_denoising_model(num_res_blocks=5, quick_mode=False):
    """
    This function is creating and learning a denoising model. For validity check use quick mode.
    :param num_res_blocks: The amount of residual blocks in the nn network wanted. default is 5.
    :param quick_mode: Boolean, set to True, then the process wil be quick (but not necessarily create a good model)
    :return: Model, Keras neural network trained model.
    """
    file_names = sol5_utils.images_for_denoising()
    min_sig = 0.
    max_sig = 0.2
    corruption_func = lambda img: add_gaussian_noise(img, min_sigma=min_sig, max_sigma=max_sig)
    patch_size = PATCH_SIZE_NOISE
    channels = CHANNELS_DENOISINING
    batch_size = BATCH_SIZE
    samples_per_epoch = SAMPLES_PER_EPOCH
    num_epochs = 5
    num_validation = SAMPLES_VALIDATION

    if quick_mode:
        batch_size = QUICK_BATCH_SIZE
        samples_per_epoch = QUICK_SAMPLES_PER_EPOCH
        num_epochs = QUICK_NUM_EPOCHS
        num_validation = QUICK_VALIDATION_SAMPLES

    model = build_nn_model(patch_size, patch_size, channels, num_res_blocks)
    train_model(model, file_names, corruption_func=corruption_func, batch_size=batch_size,
                sampels_per_epoch=samples_per_epoch, num_epochs=num_epochs, num_valid_samples=num_validation)
    return model


def add_motion_blur(image, kernel_size, angle):
    """
    Adds a motion blur to an image according to angle and kernel size.
    :param image: a grayscale image with values in the [0, 1] range of type float64
    :type image: np array
    :param kernel_size: an odd integer specifying the size of the kernel (even integers are ill-defined)
    :type kernel_size: int
    :param angle: an angle in radians in the range [0, π).
    :type angle: radians
    """
    kernel = sol5_utils.motion_blur_kernel(kernel_size, angle)

    return scipy.ndimage.filters.convolve(image, kernel)


def random_motion_blur(image, list_of_kernel_sizes):
    """
    Randomly blurring an image with motion blur.
    :param image:  a grayscale image with values in the [0, 1] range of type float64
    :type image: np array
    :param list_of_kernel_sizes: a list of odd integers.
    :type list_of_kernel_sizes: list
    """
    N = len(list_of_kernel_sizes)
    rand_kernel_size = list_of_kernel_sizes[np.random.randint(N)]
    rand_angle = np.random.uniform() * np.pi

    return add_motion_blur(image, rand_kernel_size, rand_angle)


def learn_deblurring_model(num_res_blocks=5, quick_mode=False):
    """
    This function is creating and learning a deblurring model. For validity check use quick mode.
    :param num_res_blocks: The amount of residual blocks in the nn network wanted. default is 5.
    :param quick_mode: Boolean, set to True, then the process wil be quick (but not necessarily create a good model)
    :return: Model, Keras neural network trained model.
    """

    file_names = sol5_utils.images_for_deblurring()
    kernel_size_list = [7]  # Size of kernel in this model is 7, angle is random
    corruption_func = lambda img: random_motion_blur(img, kernel_size_list)
    patch_size = PATCH_SIZE_BLUR
    channels = CHANNELS_DENOISINING
    batch_size = BATCH_SIZE
    samples_per_epoch = SAMPLES_PER_EPOCH
    num_epochs = 10
    num_validation = SAMPLES_VALIDATION

    if quick_mode:
        batch_size = QUICK_BATCH_SIZE
        samples_per_epoch = QUICK_SAMPLES_PER_EPOCH
        num_epochs = QUICK_NUM_EPOCHS
        num_validation = QUICK_VALIDATION_SAMPLES

    model = build_nn_model(patch_size, patch_size, channels, num_res_blocks)
    train_model(model, file_names, corruption_func=corruption_func, batch_size=batch_size,
                sampels_per_epoch=samples_per_epoch, num_epochs=num_epochs, num_valid_samples=num_validation)
    return model


def depth_effect(path, learning_func, corruption_type):
    """
     This method both trains 5 times the model with different residual block numbers,
    saves the models, and presents a plot of it's errors
    :param path: String, the path to keep the model in
    :param learning_func: Function to learn
    :param corruption_type: The name of the corruption in order to save it correctly.
    """
    for i in range(5):
        file_name = path + "model_" + corruption_type + "_res_blocks" + str(i + 1)
        curr_model = learning_func(num_res_blocks=(i + 1))
        curr_model.save(file_name)
        plot_loss_res_blocks(curr_model, i)
        plt.ylabel('loss')
        plt.xlabel('epoch')

    plt.legend(['train1', 'test1', 'train2', 'test2', 'train3', 'test3', 'train4', 'test4', 'train5', 'test5'],
               loc='upper left')
    plt.title(corruption_type + 'models loss')
    plt.show()
    return


def plot_loss_res_blocks(model, num_res_blocks):
    """
    This function plots the i'th lost graph fot a specific model
    :param model: The model to plot the graph for
    :param num_res_blocks: the amount of current res blocks = i'th model
    """
    plt.plot(model.history.history['loss'], COLOR[num_res_blocks], label=('model' + str(num_res_blocks) + 'loss'))
    plt.plot(model.history.history['val_loss'], (COLOR[num_res_blocks] + "--"),
             label=('model' + str(num_res_blocks) + 'val loss'))

