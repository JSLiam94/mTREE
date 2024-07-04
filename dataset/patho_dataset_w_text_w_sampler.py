import os
import torch
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
import openslide
import torch.nn.functional as F
import torchvision.transforms as transforms
import clip

def prompt_engineering(text=''):
    prompt = 'An H&E image patch in [].'.replace('[]', text)
    return prompt


class Patho_Dataset_w_text(Dataset):
    def __init__(self, annotations_file, low_img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file, sep='\t')
        self.WSI_list_in_tsv = list(self.img_labels['Patient ID'])
        self.low_img_dir = low_img_dir
        # self.high_img_dir = high_img_dir
        self.old_img_list = os.listdir(self.low_img_dir)
        self.img_list = []
        # _, self.preprocess = clip.load("ViT-B/32", device="cuda")
        for file in self.old_img_list:
            if file[:12] in self.WSI_list_in_tsv:
                self.img_list.append(file)

        self.transform1 = transforms.CenterCrop(5000)

        self.stage_list = ['STAGE I', 'STAGE II', 'STAGE III', 'STAGE IV' ]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        high_image_name = self.img_list[idx]
        high_img_path = os.path.join(self.low_img_dir, high_image_name)
        x_high = read_image(high_img_path)

        # label = self.img_labels['Months of disease-specific survival'][self.WSI_list_in_tsv.index(high_image_name[:12])]
        stage = self.img_labels['Neoplasm Disease Stage American Joint Committee on Cancer Code'][self.WSI_list_in_tsv.index(high_image_name[:12])]
        label = self.stage_list.index(stage)

        x_high = self.transform1(x_high)

        x_low = F.interpolate(x_high[None, ...], scale_factor=0.1, mode='bilinear')[0]

        # if self.transform:
        #     image = self.transform(image)
        text = prompt_engineering(stage)
        # text = clip.tokenize(text)

        return x_low, x_high, label, text, high_image_name.split('.')[0]

if __name__ == '__main__':
    from torchvision import transforms
    transform = transforms.Compose([transforms.ToTensor()])
    anno_file = '/data2/Patho_VLM/data/TCGA_KIRC/kirc_tcga_pan_can_atlas_2018_clinical_data.tsv'
    img_dir = '/data2/Patho_VLM/data/TCGA_KIRC/WSI_small_2'
    dataset = Patho_Dataset_w_text(annotations_file=anno_file, low_img_dir=img_dir, transform=transform)
    x_low, x_high, label = dataset[0]  # get the first sample
    # /data3/WSI/KIRC
    print(1)