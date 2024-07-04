import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import time
import os

from models.attention_model import AttentionModelTrafficSigns
from models.feature_model import FeatureModelTrafficSigns
from models.classifier import ClassificationHead
from models.text_projection_head import ProjectionHead

from ats.core.ats_layer import ATSModel
from ats.utils.regularizers import MultinomialEntropy
from ats.utils.logging import AttentionSaverTrafficSigns
import clip

from dataset.patho_dataset_w_text_w_sampler import Patho_Dataset_w_text
from train import train, evaluate
from torchvision import transforms

torch.autograd.set_detect_anomaly(True)

def main(opts):
    anno_file = '/data2/Patho_VLM/data/TCGA_KIRC/kirc_tcga_pan_can_atlas_2018_clinical_data.tsv'
    img_dir = '/data2/Patho_VLM/data/TCGA_KIRC/WSI_small_2'

    transform = transforms.Compose([transforms.ToTensor()])

    full_dataset = Patho_Dataset_w_text(annotations_file=anno_file, low_img_dir=img_dir, transform=transform)

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    # train_dataset = Patho_Dataset(annotations_file=anno_file, low_img_dir=img_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers)

    # test_dataset = Patho_Dataset(annotations_file=anno_file, low_img_dir=img_dir, transform=transform)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=opts.batch_size, num_workers=opts.num_workers)

    attention_model = AttentionModelTrafficSigns(squeeze_channels=True, softmax_smoothing=1e-4)
    feature_model = FeatureModelTrafficSigns(in_channels=3, strides=[1, 2, 2, 2], filters=[32, 32, 32, 32])
    classification_head = ClassificationHead(in_channels=32, num_classes=4)
    text_encoder, _ = clip.load("ViT-B/32")
    text_encoder.eval()
    # for param in text_encoder.parameters():
    #     param.data = param.data.to(torch.float32)
    text2atten_model = ProjectionHead(512, hidden_dim=1024, H=123, W=123)

    ats_model = ATSModel(attention_model, feature_model, classification_head, text_encoder, text2atten_model, n_patches=opts.n_patches, patch_size=opts.patch_size)
    ats_model = ats_model.to(opts.device)
    optimizer = optim.Adam([{'params': ats_model.attention_model.part1.parameters(), 'weight_decay': 1e-5},
                            {'params': ats_model.attention_model.part2.parameters()},
                            {'params': ats_model.feature_model.parameters()},
                            {'params': ats_model.classifier.parameters()},
                            # {'params': ats_model.text_encoder.parameters()},
                            # {'params': ats_model.sampler.parameters()},
                            {'params': ats_model.t2a_model.parameters()}
                            ], lr=opts.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.decrease_lr_at, gamma=0.1)

    # logger = AttentionSaverTrafficSigns(opts.output_dir, ats_model, test_dataset, opts)
    # class_weights = train_dataset.class_frequencies
    # class_weights = torch.from_numpy((1. / len(class_weights)) / class_weights).to(opts.device)

    criterion = nn.CrossEntropyLoss()
    entropy_loss_func = MultinomialEntropy(opts.regularizer_strength)
    attention_loss_function = nn.MSELoss()

    for epoch in range(opts.epochs):
        train_loss, train_metrics = train(ats_model, optimizer, train_loader,
                                          criterion, entropy_loss_func, attention_loss_function, opts)
        print('{}: train loss = {}, train acc = {}'.format(epoch, train_loss, train_metrics))
        with torch.no_grad():
            test_loss, test_metrics = evaluate(ats_model, test_loader, criterion,
                                               entropy_loss_func, attention_loss_function, opts)
        print('{}: test loss = {}, test acc = {}'.format(epoch, test_loss, test_metrics))
        # logger(epoch, (train_loss, test_loss), (train_metrics, test_metrics))
        scheduler.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--regularizer_strength", type=float, default=0.05,
                        help="How strong should the regularization be for the attention")
    parser.add_argument("--softmax_smoothing", type=float, default=1e-4,
                        help="Smoothing for calculating the attention map")
    parser.add_argument("--lr", type=float, default=0.001, help="Set the optimizer's learning rate")
    parser.add_argument("--n_patches", type=int, default=5, help="How many patches to sample")
    parser.add_argument("--patch_size", type=int, default=256, help="Patch size of a square patch")
    parser.add_argument("--batch_size", type=int, default=1, help="Choose the batch size for SGD")
    parser.add_argument("--epochs", type=int, default=500, help="How many epochs to train for")
    parser.add_argument("--decrease_lr_at", type=float, default=200, help="Decrease the learning rate in this epoch")
    parser.add_argument("--clipnorm", type=float, default=1, help="Clip the norm of the gradients")
    parser.add_argument("--output_dir", type=str, help="An output directory", default='output/traffic')
    parser.add_argument('--run_name', type=str, default='run')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of workers to use for data loading')

    opts = parser.parse_args()
    opts.run_name = f"{opts.run_name}_{time.strftime('%Y%m%dT%H%M%S')}"
    opts.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    main(opts)
