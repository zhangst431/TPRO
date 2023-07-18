import math
import torch
import torchvision
import numpy as np
import pandas as pd
import seaborn as sns
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2 as cv
import wandb

from icecream import ic

ic.configureOutput(includeContext=True)


def encode_cmap(label, is_luad=False):
    cmap = luad_colormap() if is_luad else colormap()
    # ic(cmap[255])
    return cmap[label.astype(np.int16), :]


def luad_colormap(N=256):
    cmap = np.zeros((N, 3), dtype=np.uint8)
    cmap[0] = np.array([205, 51, 51], dtype=np.uint8)
    cmap[1] = np.array([0, 255, 0], dtype=np.uint8)
    cmap[2] = np.array([65, 105, 225], dtype=np.uint8)
    cmap[3] = np.array([255, 165, 0], dtype=np.uint8)
    cmap[4] = np.array([255, 255, 255], dtype=np.uint8)

    # ic(cmap)
    return cmap


def colormap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap


def denormalize_img(imgs=None, mean=(0.78171339, 0.50528533, 0.78436638), std=(0.11812672, 0.21593271, 0.15903414)):
    _imgs = torch.zeros_like(imgs)
    _imgs[:, 0, :, :] = imgs[:, 0, :, :] * std[0] + mean[0]
    _imgs[:, 1, :, :] = imgs[:, 1, :, :] * std[1] + mean[1]
    _imgs[:, 2, :, :] = imgs[:, 2, :, :] * std[2] + mean[2]
    _imgs *= 255
    _imgs = _imgs.type(torch.uint8)

    return _imgs


def denormalize_img2(imgs=None, mean=None, std=None):
    # _imgs = torch.zeros_like(imgs)
    imgs = denormalize_img(imgs, mean, std)

    return imgs


def tensorboard_image(imgs=None, cam=None, CLASSES=("background", "benign", "malignant"),
                      mean=[0.78171339, 0.50528533, 0.78436638], std=[0.11812672, 0.21593271, 0.15903414]):
    ## images
    _imgs = denormalize_img(imgs=imgs, mean=mean, std=std)
    grid_imgs = torchvision.utils.make_grid(tensor=_imgs, nrow=2)

    cam_dict = {}
    if cam is not None:
        cam = cam.cpu()
        for i, cls in enumerate(CLASSES):
            cam_heatmap = plt.get_cmap('jet')(cam[:, i, :, :].numpy())[:, :, :, 0:3] * 255
            cam_cmap = torch.from_numpy(cam_heatmap).permute([0, 3, 1, 2])
            cam_img = cam_cmap * 0.5 + _imgs.cpu() * 0.5
            grid_cam = torchvision.utils.make_grid(tensor=cam_img.type(torch.uint8), nrow=2)
            cam_dict[cls] = grid_cam

    return grid_imgs, cam_dict


def tensorboard_edge(edge=None, n_row=2):
    ##
    edge = F.interpolate(edge, size=[224, 224], mode='bilinear', align_corners=False)[:, 0, ...]
    edge = edge.cpu()
    edge_heatmap = plt.get_cmap('viridis')(edge.numpy())[:, :, :, 0:3] * 255
    edge_cmap = torch.from_numpy(edge_heatmap).permute([0, 3, 1, 2])

    grid_edge = torchvision.utils.make_grid(tensor=edge_cmap.type(torch.uint8), nrow=n_row)

    return grid_edge


def tensorboard_attn(attns=None, size=[224, 224], n_pix=0, n_row=4):
    n = len(attns)
    imgs = []
    for idx, attn in enumerate(attns):
        b, hw, _ = attn.shape
        h = w = int(np.sqrt(hw))

        attn_ = attn.clone()  # - attn.min()
        # attn_ = attn_ / attn_.max()
        _n_pix = int(h * n_pix) * (w + 1)
        attn_ = attn_[:, _n_pix, :].reshape(b, 1, h, w)

        attn_ = F.interpolate(attn_, size=size, mode='bilinear', align_corners=True)

        attn_ = attn_.cpu()[:, 0, :, :]

        def minmax_norm(x):
            for i in range(x.shape[0]):
                x[i, ...] = x[i, ...] - x[i, ...].min()
                x[i, ...] = x[i, ...] / x[i, ...].max()
            return x

        attn_ = minmax_norm(attn_)

        attn_heatmap = plt.get_cmap('viridis')(attn_.detach().numpy())[:, :, :, 0:3] * 255
        attn_heatmap = torch.from_numpy(attn_heatmap).permute([0, 3, 2, 1])
        imgs.append(attn_heatmap)
    attn_img = torch.cat(imgs, dim=0)

    grid_attn = torchvision.utils.make_grid(tensor=attn_img.type(torch.uint8), nrow=n_row).permute(0, 2, 1)

    return grid_attn


def tensorboard_attn2(attns=None, size=[224, 224], n_pixs=[0.0, 0.3, 0.6, 0.9], n_row=4, with_attn_pred=False):
    n = len(attns)
    attns_top_layers = []
    attns_last_layer = []
    grid_attns = []
    if with_attn_pred:
        _attns_top_layers = attns[:-3]
        _attns_last_layer = attns[-3:-1]
    else:
        _attns_top_layers = attns[:-2]
        _attns_last_layer = attns[-2:]
    # ic(len(_attns_top_layers), len(_attns_last_layer))

    attns_top_layers = [_attns_top_layers[i][:, 0, ...] for i in range(len(_attns_top_layers))]  # 只取第0个head的attn map
    # ic(len(attns_top_layers))
    if with_attn_pred:
        attns_top_layers.append(attns[-1])
    grid_attn_top_case0 = tensorboard_attn(attns_top_layers, n_pix=n_pixs[0], n_row=n_row)
    grid_attn_top_case1 = tensorboard_attn(attns_top_layers, n_pix=n_pixs[1], n_row=n_row)
    grid_attn_top_case2 = tensorboard_attn(attns_top_layers, n_pix=n_pixs[2], n_row=n_row)
    grid_attn_top_case3 = tensorboard_attn(attns_top_layers, n_pix=n_pixs[3], n_row=n_row)
    grid_attns.append(grid_attn_top_case0)
    grid_attns.append(grid_attn_top_case1)
    grid_attns.append(grid_attn_top_case2)
    grid_attns.append(grid_attn_top_case3)

    for attn in _attns_last_layer:
        for i in range(attn.shape[1]):
            attns_last_layer.append(attn[:, i, :, :])
    grid_attn_last_case0 = tensorboard_attn(attns_last_layer, n_pix=n_pixs[0], n_row=2 * n_row)
    grid_attn_last_case1 = tensorboard_attn(attns_last_layer, n_pix=n_pixs[1], n_row=2 * n_row)
    grid_attn_last_case2 = tensorboard_attn(attns_last_layer, n_pix=n_pixs[2], n_row=2 * n_row)
    grid_attn_last_case3 = tensorboard_attn(attns_last_layer, n_pix=n_pixs[3], n_row=2 * n_row)
    grid_attns.append(grid_attn_last_case0)
    grid_attns.append(grid_attn_last_case1)
    grid_attns.append(grid_attn_last_case2)
    grid_attns.append(grid_attn_last_case3)

    return grid_attns


def tensorboard_label(labels=None):
    ## labels
    labels_cmap = encode_cmap(np.squeeze(labels, axis=1))
    # ic(type(labels_cmap), labels_cmap.shape)
    labels_cmap = torch.from_numpy(labels_cmap).permute([0, 3, 1, 2])
    grid_labels = torchvision.utils.make_grid(tensor=labels_cmap, nrow=2)

    return grid_labels


def tensorboard_test_batch1(imgs=None, cam=None, pseudo_label=None, gt_label=None, pred=None, full_pred=None,
                            CLASSES=("background", "benign", "malignant"), is_luad=False,
                            mean=[0.78171339, 0.50528533, 0.78436638],
                            std=[0.11812672, 0.21593271, 0.15903414]):
    all_imgs = []
    # ic(imgs)
    _imgs = denormalize_img(imgs=imgs, mean=mean, std=std)

    all_imgs.append(_imgs.squeeze())

    if gt_label is not None:
        gt_label = encode_cmap(np.squeeze(gt_label, axis=1), is_luad=is_luad)
        gt_label = torch.from_numpy(gt_label).permute([0, 3, 1, 2])
        all_imgs.append(gt_label.squeeze())

    if cam is not None:
        cam = cam.cpu()
        for i, cls in enumerate(CLASSES):
            cam_heatmap = plt.get_cmap('jet')(cam[:, i, :, :].numpy())[:, :, :, 0:3] * 255
            cam_cmap = torch.from_numpy(cam_heatmap).permute([0, 3, 1, 2])
            # cam_img = cam_cmap
            cam_img = cam_cmap * 0.5 + _imgs.cpu() * 0.5
            cam_img = cam_img.type(torch.uint8)
            all_imgs.append(cam_img.squeeze())

    if pseudo_label is not None:
        pseudo_label = encode_cmap(np.squeeze(pseudo_label, axis=1), is_luad=is_luad)
        pseudo_label = torch.from_numpy(pseudo_label).permute([0, 3, 1, 2])
        all_imgs.append(pseudo_label.squeeze())

    if pred is not None:
        pred = encode_cmap(np.squeeze(pred, axis=1), is_luad=is_luad)
        pred = torch.from_numpy(pred).permute([0, 3, 1, 2])
        all_imgs.append(pred.squeeze())

    if full_pred is not None:
        full_pred = encode_cmap(np.squeeze(full_pred, axis=1), is_luad=is_luad)
        full_pred = torch.from_numpy(full_pred).permute([0, 3, 1, 2])
        all_imgs.append(full_pred.squeeze())



    # all_imgs = torch.cat(all_imgs, dim=0)
    all_imgs = torchvision.utils.make_grid(all_imgs, nrow=len(all_imgs))

    return all_imgs


def tensorboard_test_batch1_deep_sup(imgs=None, cam=None, pseudo_label=None, gt_label=None, pred=None, full_pred=None,
                            CLASSES=("background", "benign", "malignant"), is_luad=False,
                            mean=[0.78171339, 0.50528533, 0.78436638],
                            std=[0.11812672, 0.21593271, 0.15903414]):
    all_imgs = []
    # ic(imgs)
    _imgs = denormalize_img(imgs=imgs, mean=mean, std=std)

    for layer in range(len(cam)):
        all_imgs.append(_imgs.squeeze())

        if gt_label is not None:
            gt_label = encode_cmap(np.squeeze(gt_label, axis=1), is_luad=is_luad)
            gt_label = torch.from_numpy(gt_label).permute([0, 3, 1, 2])
            all_imgs.append(gt_label.squeeze())

        if cam is not None:
            cam[layer] = cam[layer].cpu()
            for i, cls in enumerate(CLASSES):
                cam_heatmap = plt.get_cmap('jet')(cam[layer][:, i, :, :].numpy())[:, :, :, 0:3] * 255
                cam_cmap = torch.from_numpy(cam_heatmap).permute([0, 3, 1, 2])
                # cam_img = cam_cmap
                cam_img = cam_cmap * 0.5 + _imgs.cpu() * 0.5
                cam_img = cam_img.type(torch.uint8)
                all_imgs.append(cam_img.squeeze())

        if pseudo_label is not None:
            pseudo_label[layer] = encode_cmap(np.squeeze(pseudo_label[layer], axis=1), is_luad=is_luad)
            pseudo_label[layer] = torch.from_numpy(pseudo_label[layer]).permute([0, 3, 1, 2])
            all_imgs.append(pseudo_label[layer].squeeze())

    if pred is not None:
        pred = encode_cmap(np.squeeze(pred, axis=1), is_luad=is_luad)
        pred = torch.from_numpy(pred).permute([0, 3, 1, 2])
        all_imgs.append(pred.squeeze())

    if full_pred is not None:
        full_pred = encode_cmap(np.squeeze(full_pred, axis=1), is_luad=is_luad)
        full_pred = torch.from_numpy(full_pred).permute([0, 3, 1, 2])
        all_imgs.append(full_pred.squeeze())



    # all_imgs = torch.cat(all_imgs, dim=0)
    all_imgs = torchvision.utils.make_grid(all_imgs, nrow=len(all_imgs))

    return all_imgs


def tensorboard_one_token_attn(attn=None, n_pix=0, size=[224, 224]):
    b, num_heads, nq, hw = attn.shape
    h, w = int(math.sqrt(hw)), int(math.sqrt(hw))
    attn_copy = attn.clone().detach()
    n_pix = int(n_pix * nq)
    _attn = attn_copy[:, :, n_pix, :].reshape(b, num_heads, h, w)
    _attn = F.interpolate(_attn, size=size, mode="bilinear", align_corners=True)
    _attn = _attn.cpu().data.numpy()

    channel_min = np.min(_attn, axis=(2, 3), keepdims=True)
    channel_max = np.max(_attn, axis=(2, 3), keepdims=True)
    _attn = (_attn - channel_min) / (channel_max - channel_min)
    _attn = _attn.reshape(b * num_heads, size[0], size[1])
    attn_heatmap = plt.get_cmap('viridis')(_attn)[:, :, :, 0:3] * 255
    attn_heatmap = attn_heatmap.astype(np.uint8)
    attn_heatmap = torch.from_numpy(attn_heatmap).permute([0, 3, 1, 2])
    # ic(attn_heatmap.dtype)

    return attn_heatmap


def tensorboard_one_token_attn_list(attn_list=None, n_pix=[0.0, 0.3, 0.6, 0.9], size=[224, 224], nrow=8):
    heatmap_list = []

    for n in n_pix:
        tmp_heatmap_list = []
        for attn in attn_list:
            tmp_heatmap_list.append(tensorboard_one_token_attn(attn))
        heatmap_list.append(torch.cat(tmp_heatmap_list, dim=0))

    all_heatmap = torch.cat(heatmap_list, dim=0)
    all_heatmap = torchvision.utils.make_grid(all_heatmap, nrow=nrow)

    return all_heatmap



def tensorboard_mean_attn(attn=None, size=[224, 224]):
    b, num_heads, nq, hw = attn.shape
    h, w = int(math.sqrt(hw)), int(math.sqrt(hw))
    attn_copy = attn.clone().detach()
    attn_copy = attn_copy.mean(dim=2).reshape(b, num_heads, h, w)
    attn_copy = F.interpolate(attn_copy, size=size, mode="bilinear", align_corners=False)
    attn_copy = attn_copy.cpu().data.numpy()

    channel_min = np.min(attn_copy, axis=(2, 3), keepdims=True)
    channel_max = np.max(attn_copy, axis=(2, 3), keepdims=True)
    attn_copy = (attn_copy - channel_min) / (channel_max - channel_min)
    attn_copy = attn_copy.reshape(b * num_heads, size[0], size[1])
    attn_heatmap = plt.get_cmap('viridis')(attn_copy)[:, :, :, 0:3] * 255
    attn_heatmap = attn_heatmap.astype(np.uint8)
    attn_heatmap = torch.from_numpy(attn_heatmap).permute([0, 3, 1, 2])

    return attn_heatmap


def tensorboard_mean_heads_attn(attn=None, size=[224, 224]):
    b, num_heads, nq, hw = attn.shape
    h, w = int(math.sqrt(hw)), int(math.sqrt(hw))
    attn_copy = attn.clone().detach()
    attn_copy = attn_copy.mean(dim=2).reshape(b, num_heads, h, w)
    attn_copy = attn_copy.mean(dim=1).reshape(b, 1, h, w)
    attn_copy = F.interpolate(attn_copy, size=size, mode="bilinear", align_corners=False)
    attn_copy = attn_copy.cpu().data.numpy()

    channel_min = np.min(attn_copy, axis=(2, 3), keepdims=True)
    channel_max = np.max(attn_copy, axis=(2, 3), keepdims=True)
    attn_copy = (attn_copy - channel_min) / (channel_max - channel_min)
    attn_copy = attn_copy.reshape(b, size[0], size[1])
    attn_heatmap = plt.get_cmap('viridis')(attn_copy)[:, :, :, 0:3] * 255
    attn_heatmap = attn_heatmap.astype(np.uint8)
    attn_heatmap = torch.from_numpy(attn_heatmap).permute([0, 3, 1, 2])

    return attn_heatmap


def tensorboard_mean_attn_list(attn_list=None, size=[224, 224], nrow=8):
    heatmap_list = []

    for attn in attn_list:
        tmp_heatmap = tensorboard_mean_attn(attn)
        heatmap_list.append(tmp_heatmap)

    all_heatmap = torch.cat(heatmap_list, dim=0)
    all_heatmap = torchvision.utils.make_grid(all_heatmap, nrow=nrow)

    return all_heatmap


def tensorboard_attn_batch_mean(imgs=None, cam=None, pseudo_label=None, gt_label=None, pred=None, full_pred=None,
                                 attn_list=None, CLASSES=("background", "benign", "malignant"), is_luad=False,
                                 mean=[0.78171339, 0.50528533, 0.78436638],
                                 std=[0.11812672, 0.21593271, 0.15903414]):
    all_imgs = []
    # ic(imgs)
    _imgs = denormalize_img(imgs=imgs, mean=mean, std=std)

    all_imgs.append(_imgs.squeeze())

    if gt_label is not None:
        gt_label = encode_cmap(np.squeeze(gt_label, axis=1), is_luad=is_luad)
        gt_label = torch.from_numpy(gt_label).permute([0, 3, 1, 2])
        all_imgs.append(gt_label.squeeze())

    if cam is not None:
        cam = cam.cpu()
        for i, cls in enumerate(CLASSES):
            cam_heatmap = plt.get_cmap('jet')(cam[:, i, :, :].numpy())[:, :, :, 0:3] * 255
            cam_cmap = torch.from_numpy(cam_heatmap).permute([0, 3, 1, 2])
            # cam_img = cam_cmap
            cam_img = cam_cmap * 0.5 + _imgs.cpu() * 0.5
            cam_img = cam_img.type(torch.uint8)
            all_imgs.append(cam_img.squeeze())

    if pseudo_label is not None:
        pseudo_label = encode_cmap(np.squeeze(pseudo_label, axis=1), is_luad=is_luad)
        pseudo_label = torch.from_numpy(pseudo_label).permute([0, 3, 1, 2])
        all_imgs.append(pseudo_label.squeeze())

    if pred is not None:
        pred = encode_cmap(np.squeeze(pred, axis=1), is_luad=is_luad)
        pred = torch.from_numpy(pred).permute([0, 3, 1, 2])
        all_imgs.append(pred.squeeze())

    if full_pred is not None:
        full_pred = encode_cmap(np.squeeze(full_pred, axis=1), is_luad=is_luad)
        full_pred = torch.from_numpy(full_pred).permute([0, 3, 1, 2])
        all_imgs.append(full_pred.squeeze())

    c, h, w = gt_label.squeeze().shape
    for i in range(8 - len(all_imgs)):
        all_imgs.append(torch.zeros((c, h, w), dtype=torch.uint8))
    # all_imgs = torch.cat(all_imgs, dim=0)
    for attn in attn_list:
        tmp_heatmap = tensorboard_mean_attn(attn, size=(h, w))
        for i in range(tmp_heatmap.shape[0]):
            all_imgs.append(tmp_heatmap[i, :, :, :])

    all_imgs = torchvision.utils.make_grid(all_imgs, nrow=8)

    return all_imgs


def tensorboard_attn_batch_mean_deep_sup(imgs=None, cam=None, pseudo_label=None, gt_label=None, pred=None, full_pred=None,
                                 attn_list=None, CLASSES=("background", "benign", "malignant"), is_luad=False,
                                 mean=[0.78171339, 0.50528533, 0.78436638],
                                 std=[0.11812672, 0.21593271, 0.15903414]):
    all_imgs = []
    # ic(imgs)
    _imgs = denormalize_img(imgs=imgs, mean=mean, std=std)

    for layer in range(len(cam)):
        all_imgs.append(_imgs.squeeze())

        if gt_label is not None:
            _gt_label = encode_cmap(np.squeeze(gt_label, axis=1), is_luad=is_luad)
            _gt_label = torch.from_numpy(_gt_label).permute([0, 3, 1, 2])
            all_imgs.append(_gt_label.squeeze())

        if cam is not None:
            cam[layer] = cam[layer].cpu()
            for i, cls in enumerate(CLASSES):
                cam_heatmap = plt.get_cmap('jet')(cam[layer][:, i, :, :].numpy())[:, :, :, 0:3] * 255
                cam_cmap = torch.from_numpy(cam_heatmap).permute([0, 3, 1, 2])
                # cam_img = cam_cmap
                cam_img = cam_cmap * 0.5 + _imgs.cpu() * 0.5
                cam_img = cam_img.type(torch.uint8)
                all_imgs.append(cam_img.squeeze())

        if pseudo_label is not None:
            pseudo_label[layer] = encode_cmap(np.squeeze(pseudo_label[layer], axis=1), is_luad=is_luad)
            pseudo_label[layer] = torch.from_numpy(pseudo_label[layer]).permute([0, 3, 1, 2])
            all_imgs.append(pseudo_label[layer].squeeze())

    if pred is not None:
        pred = encode_cmap(np.squeeze(pred, axis=1), is_luad=is_luad)
        pred = torch.from_numpy(pred).permute([0, 3, 1, 2])
        all_imgs.append(pred.squeeze())

    if full_pred is not None:
        full_pred = encode_cmap(np.squeeze(full_pred, axis=1), is_luad=is_luad)
        full_pred = torch.from_numpy(full_pred).permute([0, 3, 1, 2])
        all_imgs.append(full_pred.squeeze())

    c, h, w = _gt_label.squeeze().shape
    for i in range(8 - len(all_imgs)):
        all_imgs.append(torch.zeros((c, h, w), dtype=torch.uint8))
    # all_imgs = torch.cat(all_imgs, dim=0)
    
    for attn in attn_list:
        tmp_heatmap = tensorboard_mean_attn(attn, size=(h, w))
        for i in range(tmp_heatmap.shape[0]):
            all_imgs.append(tmp_heatmap[i, :, :, :])

    all_imgs = torchvision.utils.make_grid(all_imgs, nrow=8)

    return all_imgs


def tensorboard_attn_batch_mean_heads(imgs=None, cam=None, pseudo_label=None, gt_label=None, pred=None, full_pred=None,
                                      attn_list=None, CLASSES=("background", "benign", "malignant"), is_luad=False,
                                      mean=[0.78171339, 0.50528533, 0.78436638],
                                      std=[0.11812672, 0.21593271, 0.15903414]):
    all_imgs = []
    # ic(imgs)
    _imgs = denormalize_img(imgs=imgs, mean=mean, std=std)

    all_imgs.append(_imgs.squeeze())

    gt_label = encode_cmap(np.squeeze(gt_label, axis=1), is_luad=is_luad)
    gt_label = torch.from_numpy(gt_label).permute([0, 3, 1, 2])
    all_imgs.append(gt_label.squeeze())

    if cam is not None:
        cam = cam.cpu()
        for i, cls in enumerate(CLASSES):
            cam_heatmap = plt.get_cmap('jet')(cam[:, i, :, :].numpy())[:, :, :, 0:3] * 255
            cam_cmap = torch.from_numpy(cam_heatmap).permute([0, 3, 1, 2])
            # cam_img = cam_cmap
            cam_img = cam_cmap * 0.5 + _imgs.cpu() * 0.5
            cam_img = cam_img.type(torch.uint8)
            all_imgs.append(cam_img.squeeze())

    if pseudo_label is not None:
        pseudo_label = encode_cmap(np.squeeze(pseudo_label, axis=1), is_luad=is_luad)
        pseudo_label = torch.from_numpy(pseudo_label).permute([0, 3, 1, 2])
        all_imgs.append(pseudo_label.squeeze())

    if pred is not None:
        pred = encode_cmap(np.squeeze(pred, axis=1), is_luad=is_luad)
        pred = torch.from_numpy(pred).permute([0, 3, 1, 2])
        all_imgs.append(pred.squeeze())

    if full_pred is not None:
        full_pred = encode_cmap(np.squeeze(full_pred, axis=1), is_luad=is_luad)
        full_pred = torch.from_numpy(full_pred).permute([0, 3, 1, 2])
        all_imgs.append(full_pred.squeeze())

    c, h, w = gt_label.squeeze().shape
    for i in range(8 - len(all_imgs)):
        all_imgs.append(torch.zeros((c, h, w), dtype=torch.uint8))
    # all_imgs = torch.cat(all_imgs, dim=0)

    for attn in attn_list:
        tmp_heatmap = tensorboard_mean_heads_attn(attn, size=(h, w))
        for i in range(tmp_heatmap.shape[0]):
            all_imgs.append(tmp_heatmap[i, :, :, :])

    all_imgs = torchvision.utils.make_grid(all_imgs, nrow=8)

    return all_imgs


def tensorboard_attn_batch_one_token(imgs=None, cam=None, pseudo_label=None, gt_label=None, pred=None, full_pred=None,
                                     attn_list=None, CLASSES=("background", "benign", "malignant"), is_luad=True,
                                     mean=[0.78171339, 0.50528533, 0.78436638],
                                     std=[0.11812672, 0.21593271, 0.15903414]):
    all_imgs = []
    # ic(imgs)
    _imgs = denormalize_img(imgs=imgs, mean=mean, std=std)

    all_imgs.append(_imgs.squeeze())

    gt_label = encode_cmap(np.squeeze(gt_label, axis=1), is_luad=True)
    gt_label = torch.from_numpy(gt_label).permute([0, 3, 1, 2])
    all_imgs.append(gt_label.squeeze())

    if cam is not None:
        cam = cam.cpu()
        for i, cls in enumerate(CLASSES):
            cam_heatmap = plt.get_cmap('jet')(cam[:, i, :, :].numpy())[:, :, :, 0:3] * 255
            cam_cmap = torch.from_numpy(cam_heatmap).permute([0, 3, 1, 2])
            # cam_img = cam_cmap
            cam_img = cam_cmap * 0.5 + _imgs.cpu() * 0.5
            cam_img = cam_img.type(torch.uint8)
            all_imgs.append(cam_img.squeeze())

    if pseudo_label is not None:
        pseudo_label = encode_cmap(np.squeeze(pseudo_label, axis=1), is_luad=True)
        pseudo_label = torch.from_numpy(pseudo_label).permute([0, 3, 1, 2])
        all_imgs.append(pseudo_label.squeeze())

    if pred is not None:
        pred = encode_cmap(np.squeeze(pred, axis=1), is_luad=True)
        pred = torch.from_numpy(pred).permute([0, 3, 1, 2])
        all_imgs.append(pred.squeeze())

    if full_pred is not None:
        full_pred = encode_cmap(np.squeeze(full_pred, axis=1), is_luad=True)
        full_pred = torch.from_numpy(full_pred).permute([0, 3, 1, 2])
        all_imgs.append(full_pred.squeeze())

    c, h, w = gt_label.squeeze().shape
    for i in range(8 - len(all_imgs)):
        all_imgs.append(torch.zeros((c, h, w), dtype=torch.uint8))
    # all_imgs = torch.cat(all_imgs, dim=0)

    _n_pix = [0, 0.3, 0.6, 0.9]
    for n_pix in _n_pix:
        for attn in attn_list:
            tmp_heatmap = tensorboard_one_token_attn(attn, size=(h, w), n_pix=n_pix)
            for i in range(tmp_heatmap.shape[0]):
                all_imgs.append(tmp_heatmap[i, :, :, :])

    all_imgs = torchvision.utils.make_grid(all_imgs, nrow=8)

    return all_imgs


def tensorboard_attn_matrix(attn=None):
    b, num_heads, nq, nkv = attn.shape
    attn = attn.cpu().data.numpy()

    channel_min = np.min(attn, axis=(2, 3), keepdims=True)
    channel_max = np.max(attn, axis=(2, 3), keepdims=True)

    attn = (attn - channel_min) / (channel_max - channel_min)
    attn = torch.from_numpy((attn * 255).astype(np.uint8))

    all_imgs = []
    for i in range(num_heads):
        attn_copy = attn[:, i, :, :]
        all_imgs.append(attn_copy)

    return all_imgs


def tensorboard_attn_matrix_list(attn_list=None, ):
    bottom_imgs = []
    top_imgs = []

    for attn in attn_list[:-2]:
        tmp_imgs = tensorboard_attn_matrix(attn)
        bottom_imgs.extend(tmp_imgs)

    for attn in attn_list[-2:]:
        tmp_imgs = tensorboard_attn_matrix(attn)
        top_imgs.extend(tmp_imgs)

    bottom_imgs = torchvision.utils.make_grid(bottom_imgs, nrow=8)
    top_imgs = torchvision.utils.make_grid(top_imgs, nrow=8)

    return bottom_imgs, top_imgs


def tensorboard_norm_map(norm_map=None, size=[224, 224]):
    b, c, h, w = norm_map.shape
    channel_max = norm_map.max()
    channel_min = norm_map.min()

    norm_map = (norm_map - channel_min) / (channel_max - channel_min)
    norm_map = F.interpolate(norm_map, (224, 224), mode="bilinear", align_corners=False)

    norm_map = norm_map.cpu().data.numpy()
    norm_map = norm_map.reshape(b, size[0], size[1])
    norm_heatmap = plt.get_cmap("viridis")(norm_map)[:, :, :, 0:3] * 255
    norm_heatmap = norm_heatmap.astype(np.uint8)

    norm_heatmap = torch.from_numpy(norm_heatmap).permute(0, 3, 1, 2).squeeze()

    return norm_heatmap


def tensorboard_val_batch(imgs=None, cam=None, pseudo_label=None, gt_label=None, pred=None, full_pred=None,
                          norm_map=None, similarity_map=None,
                          CLASSES=("background", "benign", "malignant"), is_luad=False,
                          mean=[0.78171339, 0.50528533, 0.78436638],
                          std=[0.11812672, 0.21593271, 0.15903414]):
    all_imgs = []
    # ic(imgs)
    _imgs = denormalize_img(imgs=imgs, mean=mean, std=std)

    all_imgs.append(_imgs.squeeze())

    if gt_label is not None:
        gt_label = encode_cmap(np.squeeze(gt_label, axis=1), is_luad=is_luad)
        gt_label = torch.from_numpy(gt_label).permute([0, 3, 1, 2])
        all_imgs.append(gt_label.squeeze())

    if cam is not None:
        cam = cam.cpu()
        for i, cls in enumerate(CLASSES):
            cam_heatmap = plt.get_cmap('jet')(cam[:, i, :, :].numpy())[:, :, :, 0:3] * 255
            cam_cmap = torch.from_numpy(cam_heatmap).permute([0, 3, 1, 2])
            # cam_img = cam_cmap
            cam_img = cam_cmap * 0.5 + _imgs.cpu() * 0.5
            cam_img = cam_img.type(torch.uint8)
            all_imgs.append(cam_img.squeeze())

    if pseudo_label is not None:
        pseudo_label = encode_cmap(np.squeeze(pseudo_label, axis=1), is_luad=is_luad)
        pseudo_label = torch.from_numpy(pseudo_label).permute([0, 3, 1, 2])
        all_imgs.append(pseudo_label.squeeze())

    if pred is not None:
        pred = encode_cmap(np.squeeze(pred, axis=1), is_luad=is_luad)
        pred = torch.from_numpy(pred).permute([0, 3, 1, 2])
        all_imgs.append(pred.squeeze())

    if full_pred is not None:
        full_pred = encode_cmap(np.squeeze(full_pred, axis=1), is_luad=is_luad)
        full_pred = torch.from_numpy(full_pred).permute([0, 3, 1, 2])
        all_imgs.append(full_pred.squeeze())

    if norm_map is not None:
        norm_map = tensorboard_norm_map(norm_map)
        all_imgs.append(norm_map)

    all_imgs.append(torch.zeros_like(norm_map))

    if similarity_map is not None:
        for i in range(similarity_map.shape[1]):
            term_map = tensorboard_norm_map(similarity_map[:, i, :, :].unsqueeze(dim=1))
            all_imgs.append(term_map)

    # all_imgs = torch.cat(all_imgs, dim=0)
    all_imgs = torchvision.utils.make_grid(all_imgs, nrow=8)

    return all_imgs


def vis_matrix(matrix, save_path=None):
    index = ["blk1", "blk2"]
    columns = ["head{}".format(_) for _ in range(8)]
    df = pd.DataFrame(matrix, index=index, columns=columns)

    plt.figure(figsize=(7.5, 2))
    ax = sns.heatmap(df, xticklabels=df.columns, yticklabels=df.index, cmap="viridis", linewidths=6, annot=True)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

