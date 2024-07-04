import openslide
import numpy as np
import pandas as pd
from PIL import Image
import os
import cv2
import random
import csv
import shutil

data_root = '/data2/Patho_VLM/data/WSI_origin/TCGA-KIRC/'
save_root = '/data2/Patho_VLM/data/TCGA_KIRC_patch/'
file_list = os.listdir(data_root)
file_number = 0

max_h_list = []
max_w_list = []

for file in file_list:
    # file = 'TCGA-3N-A9WD-01Z-00-DX1.3B836595-3D67-4985-9D3B-39A7AE38B550.svs'
    WSI_head = file.split('.')[0]
    # os.makedirs(save_root + WSI_head)
    WSI_save_root = save_root + WSI_head
    file_number += 1
    print(file_number, file)
    scn_file = data_root + file
    simg = openslide.open_slide(scn_file)
    max_h = int(simg.properties['openslide.level[0].height'])
    max_w = int(simg.properties['openslide.level[0].width'])
    max_h_list.append(max_h)
    max_w_list.append(max_w)

    # print('CROP: ', crop_x_total, crop_y_total)

print(1)
