import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from PIL import Image
from scipy import ndimage
from skimage.feature import graycomatrix, graycoprops

def load_image(infilename):
    """
    Load an image and normalize its values to be between 0 and 1
    infilename: the path to the image to load
    """
    data = mpimg.imread(infilename)
    return data


def img_float_to_uint8(img):
    """
    Convert a float image to a uint8 image 
    img: the image to convert
    """
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg


# Concatenate an image and its groundtruth
def concatenate_images(img, gt_img):
    """
    Concatenate an image and its groundtruth
    img: the image to concatenate
    gt_img: the groundtruth image to concatenate
    """
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3: # RGB image
        cimg = np.concatenate((img, gt_img), axis=1)
    else: # grayscale image
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8) # create a 3 channel image with zeros
        gt_img8 = img_float_to_uint8(gt_img) # convert the grayscale image to uint8
        gt_img_3c[:, :, 0] = gt_img8 # fill the first channel with the grayscale image
        gt_img_3c[:, :, 1] = gt_img8 # fill the second channel with the grayscale image
        gt_img_3c[:, :, 2] = gt_img8 # fill the third channel with the grayscale image
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1) # concatenate the original image and the 3 channel image
    return cimg


def img_crop(im, w, h):
    """
    Crop an image into smaller patches
    im: the image to crop
    w: the width of the patches
    h: the height of the patches
    """
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if is_2d:
                im_patch = im[j : j + w, i : i + h] # crop the image
            else:
                im_patch = im[j : j + w, i : i + h, :] # crop the image
            list_patches.append(im_patch)
    return list_patches


# Extract features consisting of average gray color as well as variance
# This function extracts four features from each image/patch
## Mean: average pixel intensity (useful for brightness)
## Variance: how much the pixel intensities deviate from the mean (useful for texture)
## Edge mean: average gradient magnitude (useful for edges)
## Laplace mean: average laplacian value (useful for edges detection in all directions)
# These features help characterize the texture and intensity patterns in each image/patch
def extract_features(patch):
    # Convert 3D patch to 2D if needed
    if len(patch.shape) == 3:
        patch = np.mean(patch, axis=2)  # Convert to grayscale
        
    # Original features
    mean = np.mean(patch)
    var = np.var(patch)
    
    # Edge features using Sobel
    sobel_x = ndimage.sobel(patch, axis=0)
    sobel_y = ndimage.sobel(patch, axis=1)
    edge_mean = np.mean(np.sqrt(sobel_x**2 + sobel_y**2))
    
    # Laplacian edge features
    laplace = ndimage.laplace(patch)
    laplace_mean = np.mean(np.abs(laplace))

    # feature transformation

    
    return [mean, var, edge_mean, laplace_mean]


def value_to_class(v, foreground_threshold):
    """
    Convert a value to a class
    v: the value to convert
    foreground_threshold: the threshold to convert the value to a class
    """
    mean_value = np.mean(v) # compute the mean of the pixels values in the patch
    if mean_value > foreground_threshold:
        return 1 # if the mean of the pixels values is greater than the threshold, the patch is considered as "foreground"
    else:
        return 0 # otherwise, the patch is considered as "background"


# Convert array of labels to an image
def label_to_img(imgwidth, imgheight, w, h, labels):
    """
    Convert array of labels to an image
    imgwidth: the width of the image
    imgheight: the height of the image
    w: the width of the patches
    h: the height of the patches
    labels: the array of labels
    """
    im = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            im[j : j + w, i : i + h] = labels[idx]
            idx = idx + 1
    return im


def make_img_overlay(img, predicted_img):
    """
    Make an image overlay
    img: the image to overlay
    predicted_img: the predicted image
    """
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:, :, 0] = predicted_img * 255

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, "RGB").convert("RGBA")
    overlay = Image.fromarray(color_mask, "RGB").convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img


# Extract features for a given image
def extract_img_features(filename, patch_size):
    img = load_image(filename)
    img_patches = img_crop(img, patch_size, patch_size)
    X = np.asarray(
        [extract_features(img_patches[i]) for i in range(len(img_patches))]
    )
    return X



