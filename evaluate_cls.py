from datetime import datetime
import os
import re
import time
import glob
import logging
import ttach as tta
from tqdm import tqdm

import argparse
import torch
from PIL import Image
from skimage import morphology
import numpy as np
import cv2 as cv
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F

from cls_network.model import ClsNetwork
from utils.pyutils import str2bool, set_seed, setup_logger
from utils import evaluate
from utils import trainutils

from icecream import ic
ic.configureOutput(includeContext=True)

start_time = datetime.now()

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="luad")
    parser.add_argument("--cls_num_classes", type=int, default=4)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--img_root", type=str, default="../data/LUAD-HistoSeg")
    parser.add_argument("--palette_path", type=str, default="./datasets/luad_palette.npy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backbone", type=str, default="mit_b1")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--label_feature_path", type=str, default=None)
    parser.add_argument("--knowledge_feature_path", type=str, default=None)
    parser.add_argument("--n_ratio", type=float, default=0.5)
    parser.add_argument("--l1", type=float, default=0.3)
    parser.add_argument("--l2", type=float, default=0.3)
    parser.add_argument("--l3", type=float, default=0.4)
    parser.add_argument("--save_dir", type=str, default=None)

    args = parser.parse_args()

    return args


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def get_seg_label(cams, inputs, label, cfg, attn_weights):
    with torch.no_grad():
        b, c, h, w = inputs.shape
        label = label.view(b, -1, 1, 1).cpu().data.numpy()

        cams = cams.cpu().data.numpy()
        cams = np.maximum(0, cams)
        channel_max = np.max(cams, axis=(2, 3), keepdims=True)
        channel_min = np.min(cams, axis=(2, 3), keepdims=True)
        cams = (cams - channel_min) / (channel_max - channel_min + 1e-6)
        cams = cams * label
        cams = torch.from_numpy(cams)
        cams = F.interpolate(cams, (h, w), mode="bilinear", align_corners=False)

        cam_max = torch.max(cams, dim=1, keepdim=True)[0]
        bg_cam = (1 - cam_max) ** 10
        cam_all = torch.cat([cams, bg_cam], dim=1)

    return cam_all


def main():
    args = get_args()
    PALETTE = list(np.load(args.palette_path))
    set_seed(args.seed)

    args.dataset = args.dataset.lower()
    if args.dataset == "luad":
        args.seg_num_classes = 5
        args.cls_num_classes = 4
        args.cls_gate = 0.15
        args.l1 = 0.3 
        args.l2 = 0.3
        args.l3 = 0.4
        args.img_root = "./data/LUAD-HistoSeg/{}/img/*.png".format(args.split)
        args.mask_root = "./data/LUAD-HistoSeg/{}/mask/*.png".format(args.split)
        CLASSES = ["TE", "NEC", "LYM", "TAS", "BACK"]
    elif args.dataset == "bcss":
        args.seg_num_classes = 5
        args.cls_num_classes = 4
        args.cls_gate = 0.4
        args.l1 = 0.1 
        args.l2 = 0.1 
        args.l3 = 0.8
        args.img_root = "./data/BCSS-WSSS/{}/img/*.png".format(args.split)
        args.mask_root = "./data/BCSS-WSSS/{}/mask/*.png".format(args.split)
        CLASSES = ["TUM", "STR", "LYM", "NEC", "BACK"]
    img_paths, mask_paths = glob.glob(args.img_root), glob.glob(args.mask_root)
    if len(mask_paths) == 0:
        mask_paths = [None] * len(img_paths)
    
    model_dir = re.split("/checkpoints/", args.model_path)[0]
    test_dir = os.path.join(model_dir, "test_log")
    os.makedirs(test_dir, exist_ok=True)
    setup_logger(os.path.join(test_dir, f"eval-{args.split}.log"))
    logging.info(args)

    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(args.gpu)
    logging.info("using {} device.".format(device))

    model = ClsNetwork(
        backbone=args.backbone,
        stride=[4, 2, 2, 1],
        cls_num_classes=args.cls_num_classes,
        pretrained=True,
        n_ratio=args.n_ratio,
        l_fea_path=args.label_feature_path,
        k_fea_path=args.knowledge_feature_path)

    tta_transform = tta.Compose([
        tta.HorizontalFlip(),
        tta.Multiply(factors=[0.9, 1.0, 1.1])
    ])

    logging.info("=================================================================================")
    # weights_dict = torch.load(model_path, map_location='cpu')
    weights_dict = torch.load(args.model_path, map_location='cpu')
    weights_dict = weights_dict["model"]
    weights_dict = {k: v for k, v in weights_dict.items() if k in model.state_dict().keys()}
    logging.info('loading from checkpoint: {}'.format(os.path.basename(args.model_path)))

    # load weights
    model.load_state_dict(weights_dict, strict=True)
    model.to(device)
    model.eval()

    MEAN, STD = trainutils.get_mean_std(args.dataset)
    transform = A.Compose([
        A.Normalize(MEAN, STD),
        ToTensorV2(transpose_mask=True)
    ])

    fuse234_matrix = evaluate.ConfusionMatrixAllClass(num_classes=args.seg_num_classes)
    fuse234_matrix.reset()

    with torch.no_grad():
        all_cls_pred4 = []
        all_cls_labels = []
        for index, img_path in tqdm(enumerate(img_paths),
                                    total=len(img_paths), ncols=100, ascii=" >="):
            mask_path = mask_paths[index]
            img_name = os.path.basename(img_path)[:-4]

            img = cv.imread(img_path, cv.IMREAD_UNCHANGED)
            mask = np.array(Image.open(mask_path))[:, :, None].astype(np.uint8)  \
                   if mask_path is not None \
                   else np.zeros([img.shape[0], img.shape[1], 1], dtype=np.int32)

            transdormed = transform(image=img, mask=mask)
            inputs = transdormed["image"].float().unsqueeze(dim=0).cuda().float()
            gt_mask = transdormed["mask"].float().unsqueeze(dim=0).cuda()
            if args.split == "train":
                if args.dataset == "luad":
                    term_split = re.split("-\[|\].", img_path)
                    cls_label = np.array(list(map(int, term_split[1].split(" ")))).reshape((1, -1))
                else:
                    term_split = re.split("\[|\]", img_path)
                    cls_label = np.array([int(x) for x in term_split[1]]).reshape((1, -1))
            else:
                cls_label = np.zeros((1, args.cls_num_classes))
                x = np.unique(mask) if np.unique(mask)[-1] != 4 else np.unique(mask)[:-1]
                cls_label[:, x] = 1

            cls1, cam1, cls2, cam2, cls3, cam3, cls4, cam4, attns = model(inputs)

            cls_pred4 = (torch.sigmoid(cls4) > 0.5).float().cpu().data.numpy()
            all_cls_pred4.append(cls_pred4)
            all_cls_labels.append(cls_label)

            # aug smooth
            segs1 = []
            segs2 = []
            segs3 = []
            segs4 = []
            for tta_tran in tta_transform:
                augmented_tensor = tta_tran.augment_image(inputs)
                cls1, cam1, cls2, cam2, cls3, cam3, cls4, cam4, attns = model(augmented_tensor)
                cam1 = get_seg_label(cam1, augmented_tensor, torch.from_numpy(cls_label).cuda() if args.split == "train" else torch.sigmoid(cls4) > args.cls_gate, args, attns).cuda()
                cam1 = tta_tran.deaugment_mask(cam1)
                segs1.append(cam1)
                cam2 = get_seg_label(cam2, augmented_tensor, torch.from_numpy(cls_label).cuda() if args.split == "train" else torch.sigmoid(cls4) > args.cls_gate, args, attns).cuda()
                cam2 = tta_tran.deaugment_mask(cam2)
                segs2.append(cam2)
                cam3 = get_seg_label(cam3, augmented_tensor, torch.from_numpy(cls_label).cuda() if args.split == "train" else torch.sigmoid(cls4) > args.cls_gate, args, attns).cuda()
                cam3 = tta_tran.deaugment_mask(cam3)
                segs3.append(cam3)
                cam4 = get_seg_label(cam4, augmented_tensor, torch.from_numpy(cls_label).cuda() if args.split == "train" else torch.sigmoid(cls4) > args.cls_gate, args, attns).cuda()
                cam4 = tta_tran.deaugment_mask(cam4)
                segs4.append(cam4)
            segs1 = torch.cat(segs1, dim=0).mean(dim=0, keepdim=True)
            segs2 = torch.cat(segs2, dim=0).mean(dim=0, keepdim=True)
            segs3 = torch.cat(segs3, dim=0).mean(dim=0, keepdim=True)
            segs4 = torch.cat(segs4, dim=0).mean(dim=0, keepdim=True)

            if args.dataset == "luad":
                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                ret, binary = cv.threshold(gray, 200, 255, cv.THRESH_BINARY)
                binary = np.uint8(binary)
                dst = morphology.remove_small_objects(binary == 255, min_size=80, connectivity=1).astype(np.uint8)
                priori_bg_mask = (1 - dst).reshape(1, 1, img.shape[0], img.shape[1])
                priori_bg_mask = torch.from_numpy(priori_bg_mask).cuda()
                segs1[:, :-1, :, :] *= priori_bg_mask
                segs2[:, :-1, :, :] *= priori_bg_mask
                segs3[:, :-1, :, :] *= priori_bg_mask
                segs4[:, :-1, :, :] *= priori_bg_mask

            fuse234 = args.l1 * segs2 + args.l2 * segs3 + args.l3 * segs4
            output_fuse234 = torch.argmax(fuse234, dim=1, keepdim=True).long()

            if args.save_dir is not None:
                # save mask
                pred_mask = Image.fromarray(output_fuse234.cpu().clone().squeeze().numpy().astype(np.uint8)).convert('P')
                pred_mask.putpalette(PALETTE)
                pred_mask.save(os.path.join(args.save_dir, img_name + ".png"))

            fuse234_matrix.update(gt_mask.clone(), output_fuse234.clone())

        all_cls_labels = np.concatenate(all_cls_labels, axis=0)
        all_cls_pred4 = np.concatenate(all_cls_pred4, axis=0)
        acc4 = (all_cls_pred4 == all_cls_labels).all(axis=1).sum() / all_cls_pred4.shape[0] * 100
        per_cls_acc4 = (all_cls_pred4 == all_cls_labels).sum(axis=0) / all_cls_pred4.shape[0] * 100
        fuse234_IOU = fuse234_matrix.compute()[2] * 100
        logging.info(
            "=============================================================================================================")
        logging.info("fuse234 IOU: {}, mean: {}".format(list(fuse234_IOU.cpu().data.numpy()),fuse234_IOU.cpu().data.numpy()[:-1].mean()))
        logging.info("acc4: {:.2f}".format(acc4))
        logging.info("per_cls_acc4: TE: {:.2f}, NEC: {:.2f}, LYM: {:.2f}, TAS: {:.2f}, mean: {:.2f}".format(per_cls_acc4[0],
                                                                                                            per_cls_acc4[1],
                                                                                                            per_cls_acc4[2],
                                                                                                            per_cls_acc4[3],
                                                                                                            per_cls_acc4.mean()))
        end_time = datetime.now()
        logging.info("infference finished, cost time: {}".format((end_time - start_time).seconds // 60))


if __name__ == '__main__':
    main()
