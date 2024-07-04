import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import pdb

from utils import calc_cls_measures, move_to
import clip

def train(model, optimizer, train_loader, criterion, entropy_loss_func, attn_loss_func, opts):
    """ Train for a single epoch """

    y_probs = np.zeros((0, 4), float)
    y_trues = np.zeros((0), int)
    losses = []

    # Put model in training mode
    model.train()

    for i, (x_low, x_high, label, text, WSI_name) in enumerate(tqdm(train_loader)):
        x_low, x_high, label = move_to([x_low, x_high, label], opts.device)

        optimizer.zero_grad()
        y, attention_map_img, patches, x_low, attention_map_text = model(x_low, x_high, text, WSI_name)

        # entropy_loss = entropy_loss_func(attention_map)
        # define a new loss for the two attention map
        attn_loss = attn_loss_func(attention_map_img, attention_map_text)

        loss = criterion(y, label) + attn_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), opts.clipnorm)
        optimizer.step()

        loss_value = loss.item()
        losses.append(loss_value)

        y_prob = F.softmax(y, dim=1)
        y_probs = np.concatenate([y_probs, y_prob.detach().cpu().numpy()])
        y_trues = np.concatenate([y_trues, label.cpu().numpy()])

    train_loss_epoch = np.round(np.mean(losses), 4)
    metrics = calc_cls_measures(y_probs, y_trues)

    return train_loss_epoch, metrics


def evaluate(model, test_loader, criterion, entropy_loss_func, attention_loss_function, opts):
    """ Evaluate a single epoch """

    y_probs = np.zeros((0, 4), float)
    y_trues = np.zeros((0), int)
    losses = []

    # Put model in eval mode
    model.eval()

    for i, (x_low, x_high, label, text, WSI_name) in enumerate(tqdm(test_loader)):

        x_low, x_high, label = move_to([x_low, x_high, label], opts.device)

        y, attention_map_img, patches, x_low, attention_map_text = model(x_low, x_high, text, WSI_name)

        # entropy_loss = entropy_loss_func(attention_map)
        # define a new loss for the two attention map
        attn_loss = attention_loss_function(attention_map_img, attention_map_text)

        loss = criterion(y, label) + attn_loss

        loss_value = loss.item()
        losses.append(loss_value)

        y_prob = F.softmax(y, dim=1)
        y_probs = np.concatenate([y_probs, y_prob.detach().cpu().numpy()])
        y_trues = np.concatenate([y_trues, label.cpu().numpy()])

    test_loss_epoch = np.round(np.mean(losses), 4)
    metrics = calc_cls_measures(y_probs, y_trues)
    return test_loss_epoch, metrics
