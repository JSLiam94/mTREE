import os
import numpy as np
import re
import torch
from PIL import Image
import openslide


root = '/data2/Patho_VLM/data/TCGA_KIRC_patch/'          # root patch for original WSI
svs_root = '/data2/Patho_VLM/data/WSI_origin/TCGA-KIRC/'

def coor_mapping(atten_coor, attention_map, WSI, ratio=0.1):
    """
    Map the coor in attention map to the location in original WSI

    """
    WSI_list = os.listdir(svs_root)
    WSI_path = svs_root + [fname for fname in WSI_list if fname.startswith(WSI)][0]
    simg = openslide.open_slide(WSI_path)
    max_h = int(simg.properties['openslide.level[0].height'])
    max_w = int(simg.properties['openslide.level[0].width'])
    original_size = [max_h, max_w]
    crop_size = (5000.0, 5000.0)
    scaled_cropped_img_size = (500, 500)

    atten_coor = atten_coor.cpu().numpy()
    # align the pooling operation
    atten_coor = (atten_coor * 4).astype(int)

    # 1. attention map map to scaled cropped image
    attention_map_size = attention_map.shape
    # scaled_cropped_img_size = crop_size * ratio     # pooling size
    crop_offset_y = (scaled_cropped_img_size[0] - attention_map_size[0]) // 2
    crop_offset_x = (scaled_cropped_img_size[1] - attention_map_size[1]) // 2
    offsets = np.array([crop_offset_y, crop_offset_x])
    coord_scaled_cropped_img = atten_coor + offsets

    # 2. unscale the coord
    coord_cropped_img = (coord_scaled_cropped_img / ratio).astype(int)

    # 3. Adjust for the center cropping done on A
    crop_offset_y = (original_size[0] - crop_size[0]) // 2
    crop_offset_x = (original_size[1] - crop_size[1]) // 2

    offsets = np.array([crop_offset_y, crop_offset_x])
    coord_origin_img = coord_cropped_img + offsets

    return coord_origin_img

def stack_images(WSI_patch_root, indices, image_paths):
    """
    Read images by the indices of the image_paths list and stack them into a tensor array.

    indices: List of indices to select images.
    image_paths: List of paths to the images.

    Returns: A PyTorch tensor containing the stacked images.
    """
    selected_images = [Image.open(os.path.join(WSI_patch_root,image_paths[i])) for i in indices]
    tensors = [torch.tensor(np.array(img)).permute(2, 0, 1) for img in selected_images]
    stacked_tensor = torch.stack(tensors)
    return stacked_tensor

def extract_coordinates(filenames):
    # Regular expression pattern to extract x and y values
    pattern = r"TCGA-.+_(\d+\.\d+)_(\d+\.\d+).png"

    coordinates = []
    for fname in filenames:
        match = re.search(pattern, fname)
        if match:
            x, y = map(float, match.groups())
            coordinates.append((x, y))

    return np.array(coordinates)

def nearest_points(coor_i, WSI_coor):
    """
    For each point in coor_i, find the nearest point in WSI_coor.

    coor_i: numpy array of shape (2, n), where n is the number of points.
    WSI_coor: numpy array of shape (2, m), where m is the number of points in the high-resolution image.

    Returns a numpy array of shape (2, n) representing the nearest points from WSI_coor for each point in coor_i.
    """
    nearest_coords = np.empty(coor_i.shape[0])
    for i in range(coor_i.shape[0]):
        diffs = WSI_coor - coor_i[i:i+1,:]
        dists = np.sum(diffs**2, axis=1)
        # nearest_coords[:, i] = WSI_coor[:, np.argmin(dists)]  # return index
        nearest_coords[i] = np.argmin(dists)
    return nearest_coords



def WSI_sampling(sample_coor, WSI_name, attention_map):

    batch_size = sample_coor.shape[0]

    for i in range(batch_size):
        atten_coor = sample_coor[i]
        WSI_n = WSI_name[i]

        atten_coor_mapped = coor_mapping(atten_coor, attention_map, WSI_n)

        WSI_patch_root = root + WSI_n
        WSI_patch_list = os.listdir(WSI_patch_root)
        all_patch_coor = extract_coordinates(WSI_patch_list)

        select_coor_list = nearest_points(atten_coor_mapped, all_patch_coor)

        samples = stack_images(WSI_patch_root, select_coor_list.astype(int), WSI_patch_list)

        if i == 0:
            sample_batch = samples.unsqueeze(0)
        else:
            sample_batch = torch.cat((sample_batch, samples.unsqueeze(0)), dim=0)

    return sample_batch
