import shutil

import numpy as np
import os
import matplotlib.pyplot as plt
import scipy

from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage import label, center_of_mass, map_coordinates

import sol4_utils

X = 0
Y = 1
HOMOGRAPHY_RAD = 3
DESC_RAD = 3
PATCH_AM = 7
DIST_FROM_EDGE = 7
ORG_IMG = 0
DESCRIPTOR_LEVEL = 2

def estimate_rigid_transform(points1, points2, translation_only=False):
    """
  Computes rigid transforming points1 towards points2, using least squares method.
  points1[i,:] corresponds to poins2[i,:]. In every point, the first coordinate is *x*.
  :param points1: array with shape (N,2). Holds coordinates of corresponding points from image 1.
  :param points2: array with shape (N,2). Holds coordinates of corresponding points from image 2.
  :param translation_only: whether to compute translation only. False (default) to compute rotation as well.
  :return: A 3x3 array with the computed homography.
  """
    centroid1 = points1.mean(axis=0)
    centroid2 = points2.mean(axis=0)

    if translation_only:
        rotation = np.eye(2)
        translation = centroid2 - centroid1

    else:
        centered_points1 = points1 - centroid1
        centered_points2 = points2 - centroid2

        sigma = centered_points2.T @ centered_points1
        U, _, Vt = np.linalg.svd(sigma)

        rotation = U @ Vt
        translation = -rotation @ centroid1 + centroid2

    H = np.eye(3)
    H[:2, :2] = rotation
    H[:2, 2] = translation
    return H


def non_maximum_suppression(image):
    """
  Finds local maximas of an image.
  :param image: A 2D array representing an image.
  :return: A boolean array with the same shape as the input image, where True indicates local maximum.
  """
    # Find local maximas.
    neighborhood = generate_binary_structure(2, 2)
    local_max = maximum_filter(image, footprint=neighborhood) == image
    local_max[image < (image.max() * 0.1)] = False

    # Erode areas to single points.
    lbs, num = label(local_max)
    centers = center_of_mass(local_max, lbs, np.arange(num) + 1)
    centers = np.stack(centers).round().astype(np.int)
    ret = np.zeros_like(image, dtype=np.bool)
    ret[centers[:, 0], centers[:, 1]] = True

    return ret


def spread_out_corners(im, m, n, radius):
    """
  Splits the image im to m by n rectangles and uses harris_corner_detector on each.
  :param im: A 2D array representing an image.
  :param m: Vertical number of rectangles.
  :param n: Horizontal number of rectangles.
  :param radius: Minimal distance of corner points from the boundary of the image.
  :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
  """
    corners = [np.empty((0, 2), dtype=np.int)]
    x_bound = np.linspace(0, im.shape[1], n + 1, dtype=np.int)
    y_bound = np.linspace(0, im.shape[0], m + 1, dtype=np.int)
    for i in range(n):
        for j in range(m):
            # Use Harris detector on every sub image.
            sub_im = im[y_bound[j]:y_bound[j + 1], x_bound[i]:x_bound[i + 1]]
            sub_corners = harris_corner_detector(sub_im)
            sub_corners += np.array([x_bound[i], y_bound[j]])[np.newaxis, :]
            corners.append(sub_corners)
    corners = np.vstack(corners)
    legit = ((corners[:, 0] > radius) & (corners[:, 0] < im.shape[1] - radius) &
             (corners[:, 1] > radius) & (corners[:, 1] < im.shape[0] - radius))
    ret = corners[legit, :]
    return ret


def harris_corner_detector(im):
    """
  Detects harris corners.
  :param im: A 2D array representing an image.
  :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
  """
    der_filter = np.array([[1], [0], [-1]])
    k = 0.04
    der_x = scipy.ndimage.filters.convolve(im, der_filter.transpose())
    der_y = scipy.ndimage.filters.convolve(im, der_filter)

    m_mat = np.zeros((im.shape[0], im.shape[1], 2, 2))
    m_mat[:, :, 0, 0] = sol4_utils.blur_spatial((der_x * der_x), 3)
    m_mat[:, :, 0, 1] = sol4_utils.blur_spatial((der_y * der_x), 3)
    m_mat[:, :, 1, 0] = sol4_utils.blur_spatial((der_y * der_x), 3)
    m_mat[:, :, 1, 1] = sol4_utils.blur_spatial((der_y * der_y), 3)
    response = np.linalg.det(m_mat) - k * np.square(np.trace(m_mat, axis1=2, axis2=3))
    max_response = non_maximum_suppression(response)
    max_indices = np.array(np.nonzero(max_response))

    return np.array([max_indices[1], max_indices[0]]).transpose()


def pyr_point_translator(x, y, org_l, dest_l):
    """
    This function translates a point position from one pyramid level to another
    :param x: x position
    :type x: int can be np array
    :param y: y position
    :type y: int can be np array
    :param dest_l: the pyramid level we want to find out the position of x,y
    :type dest_l: int
    :param org_l: the pyramid level of x,y
    :type org_l: int
    :return: (x',y') the found position in dest level, in double perhaps
    :rtype: np Array
    """
    dest_x = (2.0 ** (org_l - dest_l)) * x
    dest_y = (2.0 ** (org_l - dest_l)) * y
    return np.array([dest_x, dest_y]).transpose()


def sample_descriptor(im, pos, desc_rad):
    """
  Samples descriptors at the given corners.
  :param im: A 2D array representing an image.
  :param pos: An array with shape (N,2), where pos[i,:] are the [x,y] coordinates of the ith corner point.
  :param desc_rad: "Radius" of descriptors to compute.
  :return: A 3D array with shape (N,K,K) containing the ith descriptor at desc[i,:,:].
  """
    k = 1 + 2 * desc_rad
    n = len(pos)
    top_left_cor = np.array([np.add(pos[:, X], - desc_rad), np.add(pos[:, Y], - desc_rad)])
    patch_base_x, patch_base_y = np.array(np.meshgrid(range(k), range(k)))
    descriptors = np.zeros((n, k, k))

    for i in range(n):
        x_cor = top_left_cor[X][i]
        y_cor = top_left_cor[Y][i]
        # The image coordinates match (y, x) and not (x,y)
        patch_base = np.array([np.add(y_cor, patch_base_y), np.add(patch_base_x, x_cor)])
        desc = map_coordinates(im, patch_base, order=1, prefilter=False)
        d_mean = np.mean(desc)
        nor = np.linalg.norm(desc - d_mean)
        if nor != 0:
            descriptors[i] = np.divide((desc - d_mean), (np.linalg.norm(desc - d_mean)))
        else:
            descriptors[i] = np.zeros(desc.shape)
    return descriptors


def match_features(desc1, desc2, min_score):
    """
  Return indices of matching descriptors.
  :param desc1: A feature descriptor array with shape (N1,K,K).
  :param desc2: A feature descriptor array with shape (N2,K,K).
  :param min_score: Minimal match score.
  :return: A list containing:
              1) An array with shape (M,) and dtype int of matching indices in desc1.
              2) An array with shape (M,) and dtype int of matching indices in desc2.
  """
    descriptor_flat_1 = desc1.reshape(desc1.shape[0], desc1.shape[1] ** 2)
    descriptor_flat_2 = desc2.reshape(desc2.shape[0], desc2.shape[1] ** 2)
    dot_prod = np.dot(descriptor_flat_1, descriptor_flat_2.transpose())
    min_requirment = dot_prod > min_score

    maximum_features = np.array(np.zeros((desc1.shape[0], desc2.shape[0])))
    for row in range(desc1.shape[0]):
        two_max = np.argpartition(dot_prod[row, :], -2)[-2:]
        maximum_features[row, two_max] += 1
    for col in range(desc2.shape[0]):
        two_max = np.argpartition(dot_prod[:, col], -2)[-2:]
        maximum_features[two_max, col] += 1

    maximum_features = maximum_features > 1
    maximum_features = maximum_features & min_requirment

    return np.nonzero(maximum_features)


def find_features(pyr):
    """
  Detects and extracts feature points from a pyramid.
  :param pyr: Gaussian pyramid of a grayscale image having 3 levels.
  :return: A list containing:
              1) An array with shape (N,2) of [x,y] feature location per row found in the image.
                 These coordinates are provided at the pyramid level pyr[0].
              2) A feature descriptor array with shape (N,K,K)
  """
    descriptor_radius = DESC_RAD
    patch_amount = PATCH_AM
    min_distance_from_edge = DIST_FROM_EDGE
    corners = spread_out_corners(pyr[ORG_IMG], patch_amount, patch_amount, min_distance_from_edge)
    level_3_pos = np.array(pyr_point_translator(corners[:, X], corners[:, Y], ORG_IMG, DESCRIPTOR_LEVEL))
    descriptors = sample_descriptor(pyr[DESCRIPTOR_LEVEL], level_3_pos, descriptor_radius)
    return [corners, descriptors]


def apply_homography(pos1, H12):
    """
  Apply homography to inhomogenous points.
  :param pos1: An array with shape (N,2) of [x,y] point coordinates.
  :param H12: A 3x3 homography matrix.
  :return: An array with the same shape as pos1 with [x,y] point coordinates obtained from transforming pos1 using H12.
  """
    ones = np.array(np.ones((len(pos1), 1)))
    homo_coords = np.hstack((pos1, ones))
    x_mid, y_mid, z_mid = np.vsplit((np.matmul(H12, homo_coords.transpose())), 3)
    pos2_coords = np.array([x_mid.flatten() / z_mid.flatten(), y_mid.flatten() / z_mid.flatten()]).transpose()

    return pos2_coords


def ransac_homography(pos1, pos2, num_iter, inlier_tol, translation_only=False):
    """
  Computes homography between two sets of points using RANSAC.
  :param pos1: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 1.
  :param pos2: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 2.
  :param num_iter: Number of RANSAC iterations to perform.
  :param inlier_tol: inlier tolerance threshold.
  :param translation_only: see estimate rigid transform
  :return: A list containing:
              1) A 3x3 normalized homography matrix.
              2) An Array with shape (S,) where S is the number of inliers,
                  containing the indices in pos1/pos2 of the maximal set of inlier matches found.
  """
    N = len(pos1)
    max_inliers = 0
    if num_iter == 0:
        return [[], []]

    max_homograph = np.array(np.zeros((3, 3)))
    indices_of_max = np.array([])
    if N == 0:
        return [max_homograph, []]
    for i in range(num_iter):
        rand_p1, rand_p2 = np.random.choice(N, size=2)
        curr_homo = estimate_rigid_transform(np.array([pos1[rand_p1], pos1[rand_p2]]),
                                             np.array([pos2[rand_p1], pos2[rand_p2]]), translation_only)
        pos1_homographied = apply_homography(pos1, curr_homo)
        euclidean_dist = np.array(np.square(np.linalg.norm(pos1_homographied - pos2, axis=1)))
        amount_inlires = np.count_nonzero(euclidean_dist < inlier_tol)

        if amount_inlires > max_inliers:
            max_homograph = curr_homo
            indices_of_max = np.array(np.nonzero(euclidean_dist < inlier_tol))[0]
            max_inliers = amount_inlires

    return [max_homograph, indices_of_max]


def display_matches(im1, im2, pos1, pos2, inliers):
    """
  Dispalay matching points.
  :param im1: A grayscale image.
  :param im2: A grayscale image.
  :parma pos1: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im1.
  :param pos2: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im2.
  :param inliers: An array with shape (S,) of inlier matches.
  """
    im = np.hstack((im1, im2))
    pos2_x = pos2[:, 0] + len(im1[0])
    pos2_y = pos2[:, 1]

    plt.imshow(im, cmap='gray')
    N = len(pos1)
    for i in range(N):
        if i in inliers:
            xs = [pos1[i][0], pos2_x[i]]
            ys = [pos1[i][1], pos2_y[i]]
            plt.plot(xs, ys, c='y', mfc='r', lw=.4, ms=3, marker='o', linestyle='dashed')
        else:
            xs = [pos1[i][0], pos2_x[i]]
            ys = [pos1[i][1], pos2_y[i]]
            plt.plot(xs, ys, c='b', mfc='r', lw=.4, ms=3, marker='o', linestyle='dashed')
    plt.show()


def accumulate_homographies(H_succesive, m):
    """
  Convert a list of succesive homographies to a
  list of homographies to a common reference frame.
  :param H_successive: A list of M-1 3x3 homography
    matrices where H_successive[i] is a homography which transforms points
    from coordinate system i to coordinate system i+1.
  :param m: Index of the coordinate system towards which we would like to
    accumulate the given homographies.
  :return: A list of M 3x3 homography matrices,
    where H2m[i] transforms points from coordinate system i to coordinate system m
  """
    if len(H_succesive) == 0:
        return H_succesive

    H2m = [np.eye(HOMOGRAPHY_RAD)]
    for i in range(m, 0, -1):
        temp_H = H2m[0].dot(H_succesive[i - 1])
        H2m.insert(0, temp_H/temp_H[2, 2])
    for i in range(m, len(H_succesive)):
        temp_H = H2m[i].dot(np.linalg.inv(H_succesive[i]))
        H2m.append(temp_H/temp_H[2, 2])

    return H2m


def compute_bounding_box(homography, w, h):
    """
  computes bounding box of warped image under homography, without actually warping the image
  :param homography: homography
  :param w: width of the image
  :param h: height of the image
  :return: 2x2 array, where the first row is [x,y] of the top left corner,
   and the second row is the [x,y] of the bottom right corner
  """
    top_left_x, top_left_y = apply_homography(np.array([[0, 0]]), homography).transpose()
    top_right_x, top_right_y = apply_homography(np.array([[w, 0]]), homography).transpose()
    bottem_left_x, bottem_left_y = apply_homography(np.array([[0, h]]), homography).transpose()
    bottem_right_x, bottem_right_y = apply_homography(np.array([[w, h]]), homography).transpose()

    min_x = min([top_left_x, top_right_x, bottem_left_x, bottem_right_x])[0]
    max_x = max([top_left_x, top_right_x, bottem_left_x, bottem_right_x])[0]
    min_y = min([top_left_y, top_right_y, bottem_left_y, bottem_right_y])[0]
    max_y = max([top_left_y, top_right_y, bottem_left_y, bottem_right_y])[0]
    return np.array([[min_x, min_y], [max_x, max_y]]).astype(int)


def warp_channel(image, homography):
    """
  Warps a 2D image with a given homography.
  :param image: a 2D image.
  :param homography: homograhpy.
  :return: A 2d warped image.
  """
    top_left, bottem_right = compute_bounding_box(homography, image.shape[Y], image.shape[X])

    x_coords_bw = np.arange(top_left[X], bottem_right[X])
    y_coords_bw = np.arange(top_left[Y], bottem_right[Y])
    x_coords, y_coords = np.array(np.meshgrid(x_coords_bw, y_coords_bw))

    inv_homg = np.linalg.inv(homography)
    coords = np.array([x_coords, y_coords]).transpose()

    org_shape = coords.shape
    org_coords = apply_homography(coords.reshape(-1, 2), inv_homg).reshape(org_shape)
    return map_coordinates(image, [org_coords[:, :, Y].T, org_coords[:, :, X].T], order=1, prefilter=False)


def warp_image(image, homography):
    """
  Warps an RGB image with a given homography.
  :param image: an RGB image.
  :param homography: homograhpy.
  :return: A warped image.
  """
    return np.dstack([warp_channel(image[..., channel], homography) for channel in range(3)])


def filter_homographies_with_translation(homographies, minimum_right_translation):
    """
  Filters rigid transformations encoded as homographies by the amount of translation from left to right.
  :param homographies: homograhpies to filter.
  :param minimum_right_translation: amount of translation below which the transformation is discarded.
  :return: filtered homographies..
  """
    translation_over_thresh = [0]
    last = homographies[0][0, -1]
    for i in range(1, len(homographies)):
        if homographies[i][0, -1] - last > minimum_right_translation:
            translation_over_thresh.append(i)
            last = homographies[i][0, -1]
    return np.array(translation_over_thresh).astype(np.int)


class PanoramicVideoGenerator:
    """
  Generates panorama from a set of images.
  """

    def __init__(self, data_dir, file_prefix, num_images):
        """
    The naming convention for a sequence of images is file_prefixN.jpg,
    where N is a running number 001, 002, 003...
    :param data_dir: path to input images.
    :param file_prefix: see above.
    :param num_images: number of images to produce the panoramas with.
    """
        self.file_prefix = file_prefix
        self.files = [os.path.join(data_dir, '%s%03d.jpg' % (file_prefix, i + 1)) for i in range(num_images)]
        self.files = list(filter(os.path.exists, self.files))
        self.panoramas = None
        self.homographies = None
        print('found %d images' % len(self.files))

    def align_images(self, translation_only=False):
        """
    compute homographies between all images to a common coordinate system
    :param translation_only: see estimte_rigid_transform
    """
        # Extract feature point locations and descriptors.
        points_and_descriptors = []
        for file in self.files:
            image = sol4_utils.read_image(file, 1)
            self.h, self.w = image.shape
            pyramid, _ = sol4_utils.build_gaussian_pyramid(image, 3, 7)
            points_and_descriptors.append(find_features(pyramid))

        # Compute homographies between successive pairs of images.
        Hs = []
        for i in range(len(points_and_descriptors) - 1):
            points1, points2 = points_and_descriptors[i][0], points_and_descriptors[i + 1][0]
            desc1, desc2 = points_and_descriptors[i][1], points_and_descriptors[i + 1][1]

            # Find matching feature points.
            ind1, ind2 = match_features(desc1, desc2, .7)
            points1, points2 = points1[ind1, :], points2[ind2, :]

            # Compute homography using RANSAC.
            H12, inliers = ransac_homography(points1, points2, 100, 6, translation_only)

            # Uncomment for debugging: display inliers and outliers among matching points.
            # In the submitted code this function should be commented out!
            # display_matches(self.images[i], self.images[i+1], points1 , points2, inliers)

            Hs.append(H12)

        # Compute composite homographies from the central coordinate system.
        accumulated_homographies = accumulate_homographies(Hs, (len(Hs) - 1) // 2)
        self.homographies = np.stack(accumulated_homographies)
        self.frames_for_panoramas = filter_homographies_with_translation(self.homographies, minimum_right_translation=5)
        self.homographies = self.homographies[self.frames_for_panoramas]

    def generate_panoramic_images(self, number_of_panoramas):
        """
    combine slices from input images to panoramas.
    :param number_of_panoramas: how many different slices to take from each input image
    """
        assert self.homographies is not None

        # compute bounding boxes of all warped input images in the coordinate system of the middle image (as given by the homographies)
        self.bounding_boxes = np.zeros((self.frames_for_panoramas.size, 2, 2))
        for i in range(self.frames_for_panoramas.size):
            self.bounding_boxes[i] = compute_bounding_box(self.homographies[i], self.w, self.h)

        # change our reference coordinate system to the panoramas
        # all panoramas share the same coordinate system
        global_offset = np.min(self.bounding_boxes, axis=(0, 1))
        self.bounding_boxes -= global_offset

        slice_centers = np.linspace(0, self.w, number_of_panoramas + 2, endpoint=True, dtype=np.int)[1:-1]
        warped_slice_centers = np.zeros((number_of_panoramas, self.frames_for_panoramas.size))
        # every slice is a different panorama, it indicates the slices of the input images from which the panorama
        # will be concatenated
        for i in range(slice_centers.size):
            slice_center_2d = np.array([slice_centers[i], self.h // 2])[None, :]
            # homography warps the slice center to the coordinate system of the middle image
            warped_centers = [apply_homography(slice_center_2d, h) for h in self.homographies]
            # we are actually only interested in the x coordinate of each slice center in the panoramas' coordinate system
            warped_slice_centers[i] = np.array(warped_centers)[:, :, 0].squeeze() - global_offset[0]

        panorama_size = np.max(self.bounding_boxes, axis=(0, 1)).astype(np.int) + 1

        # boundary between input images in the panorama
        x_strip_boundary = ((warped_slice_centers[:, :-1] + warped_slice_centers[:, 1:]) / 2)
        x_strip_boundary = np.hstack([np.zeros((number_of_panoramas, 1)),
                                      x_strip_boundary,
                                      np.ones((number_of_panoramas, 1)) * panorama_size[0]])
        x_strip_boundary = x_strip_boundary.round().astype(np.int)

        self.panoramas = np.zeros((number_of_panoramas, panorama_size[1], panorama_size[0], 3), dtype=np.float64)
        for i, frame_index in enumerate(self.frames_for_panoramas):
            # warp every input image once, and populate all panoramas
            image = sol4_utils.read_image(self.files[frame_index], 2)
            warped_image = warp_image(image, self.homographies[i])
            x_offset, y_offset = self.bounding_boxes[i][0].astype(np.int)
            y_bottom = y_offset + warped_image.shape[0]

            for panorama_index in range(number_of_panoramas):
                # take strip of warped image and paste to current panorama
                boundaries = x_strip_boundary[panorama_index, i:i + 2]
                image_strip = warped_image[:, boundaries[0] - x_offset: boundaries[1] - x_offset]
                x_end = boundaries[0] + image_strip.shape[1]
                self.panoramas[panorama_index, y_offset:y_bottom, boundaries[0]:x_end] = image_strip

        # crop out areas not recorded from enough angles
        # assert will fail if there is overlap in field of view between the left most image and the right most image
        crop_left = int(self.bounding_boxes[0][1, 0])
        crop_right = int(self.bounding_boxes[-1][0, 0])
        assert crop_left < crop_right, 'for testing your code with a few images do not crop.'
        print(crop_left, crop_right)
        self.panoramas = self.panoramas[:, :, crop_left:crop_right, :]

    def save_panoramas_to_video(self):
        assert self.panoramas is not None
        out_folder = 'tmp_folder_for_panoramic_frames/%s' % self.file_prefix
        try:
            shutil.rmtree(out_folder)
        except:
            print('could not remove folder')
            pass
        os.makedirs(out_folder)
        # save individual panorama images to 'tmp_folder_for_panoramic_frames'
        for i, panorama in enumerate(self.panoramas):
            plt.imsave('%s/panorama%02d.png' % (out_folder, i + 1), panorama)
        if os.path.exists('%s.mp4' % self.file_prefix):
            os.remove('%s.mp4' % self.file_prefix)
        # write output video to current folder
        os.system('ffmpeg -framerate 3 -i %s/panorama%%02d.png %s.mp4' %
                  (out_folder, self.file_prefix))

    def show_panorama(self, panorama_index, figsize=(20, 20)):
        assert self.panoramas is not None
        plt.figure(figsize=figsize)
        plt.imshow(self.panoramas[panorama_index].clip(0, 1))
        plt.show()

