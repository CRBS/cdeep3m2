"""
Secondary augmentation operations for CDeep3M

NCMIR, UCSD -- CDeep3M

        Jul 2019 @jihyeonje
Update: Feb 2020 @mhaberl
"""

import random
from random import uniform
import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageEnhance
from skimage.util import random_noise
from skimage.restoration import (
    denoise_tv_chambolle,
    denoise_tv_bregman,
    estimate_sigma)
from scipy.ndimage.interpolation import map_coordinates
from scipy.interpolate import interp1d
from check_img_dims import check_img_dims


def HighContrast(images, factor):
    augmented_images = []
    for i in range(len(images)):
        imagem = Image.fromarray(images[i][:, :, 0], 'L')
        contrast_modifier = ImageEnhance.Contrast(imagem)
        image = np.array(contrast_modifier.enhance(factor))
        augmented_images.append(np.expand_dims(image, axis=2))
        del image, imagem
    return np.asarray(augmented_images)


def LowContrast(images, factor):
    augmented_images = []
    for i in range(len(images)):
        imagem = Image.fromarray(images[i][:, :, 0], 'L')
        contrast_modifier = ImageEnhance.Contrast(imagem)
        image = np.array(contrast_modifier.enhance(factor))
        augmented_images.append(np.expand_dims(image, axis=2))
        del image, imagem
    return np.asarray(augmented_images)


def Blur(images, factor):
    augmented_images = []
    for i in range(len(images)):
        blurred_image = gaussian_filter(images[i], factor)
        augmented_images.append(blurred_image)
        del blurred_image
    return np.asarray(augmented_images)


def Sharpen(images, factor):
    augmented_images = []
    for i in range(len(images)):
        imagem = Image.fromarray(images[i][:, :, 0], 'L')
        enhancer = ImageEnhance.Sharpness(imagem)
        image = np.array(enhancer.enhance(factor))
        augmented_images.append(np.expand_dims(image, axis=2))
        del imagem, image
    return np.asarray(augmented_images)


def UniformNoise(images, factor):
    augmented_images = []
    for i in range(len(images)):
        noise_img = random_noise(images[i], mode='gaussian', var=factor**2)
        noise_img = (255 * noise_img).astype(np.uint8)
        augmented_images.append(noise_img)
        del noise_img
    return np.asarray(augmented_images)


"""
def SaltAndPepper(images, factor):
    augmented_images = []
    ih=images[0].shape[0]
    iw=images[0].shape[1]
    k=0
    salt=True

    noisypixels=(ih*iw*factor)
    for image in images:
        for i in range(ih*iw):
            if k<noisypixels:  #keep track of noise level
                if salt==True:
                    image[r.randrange(0,ih)][r.randrange(0,iw)]=255
                    salt=False
                else:
                        image[r.randrange(0,ih)][r.randrange(0,iw)]=0
                        salt=True
                        k+=1
            else:
                break
    augmented_images.append(np.expand_dims(image,axis=2))
    return np.asarray(augmented_images)
"""


def TV_Chambolle(images, factor):
    augmented_images = []
    for i in range(len(images)):
        sigma_est = estimate_sigma(
            images[i],
            multichannel=True,
            average_sigmas=True) / 100
        tv_denoised = denoise_tv_chambolle(images[i], sigma_est * factor)
        tv_denoised = (255 * tv_denoised).astype(np.uint8)
        augmented_images.append(tv_denoised)
        del tv_denoised
    return np.asarray(augmented_images)


def TV_Bregman(images, factor):
    augmented_images = []
    for i in range(len(images)):
        sigma_est = estimate_sigma(
            images[i],
            multichannel=True,
            average_sigmas=True) / 100
        tv_denoised = denoise_tv_bregman(images[i], sigma_est * factor)
        tv_denoised = (255 * tv_denoised).astype(np.uint8)
        augmented_images.append(np.expand_dims(tv_denoised, axis=2))
        del tv_denoised
    return np.asarray(augmented_images)


def HistogramEqualization(images, factor):
    augmented_images = []
    for i in range(len(images)):
        image_histogram, bins = np.histogram(
            images[i].flatten(), factor, density=True)
        cdf = image_histogram.cumsum()  # cumulative distribution function
        cdf = (factor - 1) * cdf / cdf[-1]  # normalize
        image_equalized = np.interp(images[i].flatten(
        ), bins[:-1], cdf).reshape(images[i].shape).astype(np.uint8)
        augmented_images.append(image_equalized)
        del image_equalized
    return np.asarray(augmented_images)


def Skew(images, labels, factor):
    w, h = images[0].shape[0], images[0].shape[1]
    x1 = 0
    x2 = h
    y1 = 0
    y2 = w
    original_plane = [(y1, x1), (y2, x1), (y2, x2), (y1, x2)]
    max_skew_amount = max(w, h)
    max_skew_amount = int(max_skew_amount * 3 * factor)
    skew_amount = int((1 + max_skew_amount) / 60)

    skew_direction = random.randint(0, 3)

    if skew_direction == 0:
        # Skew possibility 0
        new_plane = [(y1 - skew_amount, x1), (y2, x1), (y2, x2), (y1, x2)]
    elif skew_direction == 1:
        # Skew possibility 1
        new_plane = [(y1, x1), (y2, x1 - skew_amount), (y2, x2), (y1, x2)]
    elif skew_direction == 2:
        # Skew possibility 2
        new_plane = [(y1, x1), (y2, x1), (y2 + skew_amount, x2), (y1, x2)]
    elif skew_direction == 3:
        # Skew possibility 3
        new_plane = [(y1, x1), (y2, x1), (y2, x2), (y1, x2 + skew_amount)]

    matrix = []

    for p1, p2 in zip(new_plane, original_plane):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -
                       p2[0] * p1[0], -p2[0] * p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -
                       p2[1] * p1[0], -p2[1] * p1[1]])

    A = np.matrix(matrix, dtype=np.float)
    B = np.array(original_plane).reshape(8)
    perspective_skew_coefficients_matrix = np.dot(np.linalg.pinv(A), B)
    perspective_skew_coefficients_matrix = np.array(
        perspective_skew_coefficients_matrix).reshape(8)

    augmented_images = []
    augmented_labels = []
    for i in range(len(images)):
        imagem = Image.fromarray(images[i][:, :, 0], 'L')
        image_mod = np.array(
            imagem.transform(
                imagem.size,
                Image.PERSPECTIVE,
                perspective_skew_coefficients_matrix,
                resample=Image.BICUBIC))
        augmented_images.append(np.expand_dims(image_mod, axis=2))
        labelm = Image.fromarray(labels[i][:, :, 0], 'L')
        label_mod = np.array(
            labelm.transform(
                labelm.size,
                Image.PERSPECTIVE,
                perspective_skew_coefficients_matrix,
                resample=Image.BICUBIC))
        augmented_labels.append(np.expand_dims(label_mod, axis=2))
    return np.asarray(augmented_images), binarize(augmented_labels)


def ElasticDistortion(images, labels, sigma):

    random_state = np.random.RandomState(None)
    shape = images[0][:, :, 0].shape

    alpha = 550

    augmented_images = []
    augmented_labels = []
    dx = gaussian_filter(
        (random_state.rand(
            *shape) * 2 - 1),
        sigma,
        mode="constant",
        cval=0) * alpha
    dy = gaussian_filter(
        (random_state.rand(
            *shape) * 2 - 1),
        sigma,
        mode="constant",
        cval=0) * alpha
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    dx = gaussian_filter(
        (random_state.rand(
            *shape) * 2 - 1),
        sigma,
        mode="constant",
        cval=0) * alpha
    dy = gaussian_filter(
        (random_state.rand(
            *shape) * 2 - 1),
        sigma,
        mode="constant",
        cval=0) * alpha
    last_indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    linfit_0 = interp1d([1, len(images)], np.vstack(
        [indices[0][:, 0], last_indices[0][:, 0]]), axis=0)
    linfit_1 = interp1d([1, len(images)], np.vstack(
        [indices[1][:, 0], last_indices[1][:, 0]]), axis=0)

    i = 1
    for j in range(len(images)):
        indices_n = (
            np.expand_dims(
                linfit_0(i), axis=1), np.expand_dims(
                    linfit_1(i), axis=1))
        distorted_image = map_coordinates(
            images[j][:, :, 0], indices_n, order=1, mode='reflect')
        reshaped_img = distorted_image.reshape(images[j][:, :, 0].shape)
        augmented_images.append(np.expand_dims(reshaped_img, axis=2))
        distorted_lbl = map_coordinates(
            labels[i - 1][:, :, 0], indices_n, order=1, mode='reflect')
        reshaped_lbl = distorted_lbl.reshape(labels[i - 1][:, :, 0].shape)
        augmented_labels.append(np.expand_dims(reshaped_lbl, axis=2))
        i += 1

    return np.asarray(augmented_images), binarize(augmented_labels)


def Resize(images, labels, scale, dir):
    min = [0.95, 1, 1.05]
    max = [0.25, 1, 4]
    factors = interp1d([1, 10], np.vstack([min, max]), axis=0)
    newscale = factors(scale)
    augmented_images = []
    augmented_labels = []
    i = 0

    # downscale
    if dir == 0:
        mins = uniform(newscale[0], newscale[1])
        min_x = int(images[0].shape[0] * mins)
        min_y = int(images[0].shape[1] * mins)
        for i in range(len(images)):
            imagem = Image.fromarray(images[i][:, :, 0], 'L')
            new_img = np.array(imagem.resize((min_y, min_x), Image.ANTIALIAS))
            augmented_images.append(np.expand_dims(new_img, axis=2))
            labelm = Image.fromarray(labels[i][:, :, 0], 'L')
            new_lbl = np.array(labelm.resize((min_y, min_x), Image.ANTIALIAS))
            augmented_labels.append(np.expand_dims(new_lbl, axis=2))
        if (min_x <= 325 or min_y <= 325):
            augmented_images, augmented_labels = check_img_dims(
                np.asarray(augmented_images), np.asarray(augmented_labels), 325)

    # upscale
    if dir == 1:
        maxs = uniform(newscale[1], newscale[2])
        max_x = int(images[0].shape[0] * maxs)
        max_y = int(images[0].shape[1] * maxs)
        for image in images:
            imagem = Image.fromarray(image[:, :, 0], 'L')
            new_img = np.array(imagem.resize((max_y, max_x), Image.ANTIALIAS))
            augmented_images.append(np.expand_dims(new_img, axis=2))
            labelm = Image.fromarray(labels[i][:, :, 0], 'L')
            new_lbl = np.array(labelm.resize((max_y, max_x), Image.ANTIALIAS))
            augmented_labels.append(np.expand_dims(new_lbl, axis=2))
            i += 1

    return np.asarray(augmented_images), binarize(augmented_labels)


def binarize(labels):
    binarized_labels = np.asarray(labels)
    if np.max(binarized_labels) > 1:
        binarized_labels[binarized_labels > 128] = 255
        binarized_labels[binarized_labels <= 128] = 0
    elif np.max(binarized_labels) == 1:
        binarized_labels[binarized_labels > 0.5] = 1
        binarized_labels[binarized_labels <= 0.5] = 0
    return binarized_labels
