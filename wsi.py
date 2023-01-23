
#Required Libs
import os
dirname = os.path.dirname(__file__)
OPENSLIDE_PATH = os.path.join(dirname, 'env/openslide-win64-20221217/bin')
if hasattr(os, 'add_dll_directory'):
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide


import seaborn as sns
import cv2
from openslide.deepzoom import DeepZoomGenerator
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from openslide import open_slide
import stainNorm_Macenko as snm
import stainNorm_Vahadane as snv
import stainNorm_Reinhard as snr


# Required dir
dir_slide_1 = "whole_slide_images/Normal_Lymphnode.svs"
dir_slide_2 = "whole_slide_images/Reactive_hyperplasia.svs"
tile_1_dir = "tile_images/slide1/raw_tiles"
tile_2_dir = "tile_images/slide2/raw_tiles"
nor_tile_1_dir = "tile_images/slide1/color_norm_tile"
nor_tile_2_dir = "tile_images/slide2/color_norm_tile"

# Function to show a tile 
def show_tiles(single_tile):
    single_tile_RGB = single_tile.convert("RGB")
    single_tile_RGB.show()

# Function to save all the tiles of last level 
def save_tiles(tiles, dir):
    level = tiles.level_count - 1
    rows, cols = tiles.level_tiles[level]
    for row in range(0, rows):
        for col in range(0, cols):
            tile_name = os.path.join(dir, "%d_%d" % (row, col))
            print(tile_name)
            temp_tile = tiles.get_tile(level, (row, col))

            temp_tile_RGB = temp_tile.convert('RGB')
            temp_tile_np = np.array(temp_tile_RGB)
            plt.imsave(tile_name+".png", temp_tile_np)

# Function for binarize the image using otsu or triangle  
def thresholding(img, method='otsu'):
    # convert to grayscale complement image
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_c = 255 - grayscale_img
    thres, thres_img = 0, img_c.copy()
    if method == 'otsu':
        thres, thres_img = cv2.threshold(
            img_c, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == 'triangle':
        thres, thres_img = cv2.threshold(
            img_c, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_TRIANGLE)
    return thres, thres_img, img_c

#plot the histograms of img along with their binarized form resp.
def histogram(img, thres_img, img_c, thres):
    """
    style: ['color', 'grayscale']
    """
    plt.figure(figsize=(15, 15))

    plt.subplot(3, 2, 1)
    plt.imshow(img)
    plt.title('Scaled-down image')

    plt.subplot(3, 2, 2)
    sns.histplot(img.ravel(), bins=np.arange(
        0, 256), color='orange', alpha=0.5)
    sns.histplot(img[:, :, 0].ravel(), bins=np.arange(
        0, 256), color='red', alpha=0.5)
    sns.histplot(img[:, :, 1].ravel(), bins=np.arange(
        0, 256), color='Green', alpha=0.5)
    sns.histplot(img[:, :, 2].ravel(), bins=np.arange(
        0, 256), color='Blue', alpha=0.5)
    plt.legend(['Total', 'Red_Channel', 'Green_Channel', 'Blue_Channel'])
    plt.ylim(0, 0.05e6)
    plt.xlabel('Intensity value')
    plt.title('Color histogram')

    plt.subplot(3, 2, 3)
    plt.imshow(img_c, cmap='gist_gray')
    plt.title('Complement grayscale image')

    plt.subplot(3, 2, 4)
    sns.histplot(img_c.ravel(), bins=np.arange(0, 256))
    plt.axvline(thres, c='red', linestyle="--")
    plt.ylim(0, 0.05e6)
    plt.xlabel('Intensity value')
    plt.title('Grayscale complement histogram')

    plt.subplot(3, 2, 5)
    plt.imshow(thres_img, cmap='gist_gray')
    plt.title('Thresholded image')

    plt.subplot(3, 2, 6)
    sns.histplot(thres_img.ravel(), bins=np.arange(0, 256))
    plt.axvline(thres, c='red', linestyle="--")
    plt.ylim(0, 0.05e6)
    plt.xlabel('Intensity value')
    plt.title('Thresholded histogram')

    plt.tight_layout()
    plt.show()

# Function used to color normalize the h&e stains in the images 
def color_normalize_tiles(ref_img, nor_dir, raw_tile_dir):
    n = snm.Normalizer() # could try snr or snv
    n.fit(ref_img)
    for file_name in os.listdir(raw_tile_dir):
        req_path = os.path.join(raw_tile_dir, file_name)
        img = Image.open(req_path).convert("RGB")
        np_img = np.array(img)
        # print(np_img.shape)
        try:
            nor_img = n.transform(np_img)
        except:
            continue
        nor_dir += '/'
        plt.imsave(nor_dir+file_name, nor_img)


# opening the slides 
slide1 = open_slide(dir_slide_1)
slide2 = open_slide(dir_slide_2)

# Generating the tiles using DeepZoomGenerator
tiles_slide_1 = DeepZoomGenerator(
    slide1, tile_size=1024, overlap=0, limit_bounds=False)
tiles_slide_2 = DeepZoomGenerator(
    slide2, tile_size=1024, overlap=0, limit_bounds=False)

# print(tiles_slide_1.level_count)
# print(tiles_slide_2.level_count)


# Creating a reference tile for color normalization
single_tile_2 = tiles_slide_2.get_tile(15, (15, 19))
single_tile_2_RGB = single_tile_2.convert("RGB")
single_tile_2_np = np.array(single_tile_2_RGB)


#Saving the raw tiles of slide one
save_tiles(tiles_slide_2, tile_2_dir)

# Saving the raw tiles of slide two 
save_tiles(tiles_slide_1, tile_1_dir)

# Performing color normalization on tiles of slide two
color_normalize_tiles(single_tile_2_np, nor_tile_2_dir, tile_2_dir)

#Performing color normalization on tiles of slide one 
color_normalize_tiles(single_tile_2_np, nor_tile_1_dir, tile_1_dir)





# thres_otsu, thres_img, img_c = thresholding(single_tile_1_np, method='otsu')
# histogram(single_tile_1_np, thres_img, img_c, thres_otsu)

# thres_triangle, thres_img, img_c = thresholding(
#     single_tile_1_np, method='triangle')
# histogram(single_tile_1_np, thres_img, img_c, thres_triangle)

# single_tile_2 = tiles_slide_2.get_tile(15, (50, 45))

# show_tiles(single_tile_2)
# show_tiles(single_tile_2)

