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

from seg_network.model import SegNetwork
from utils.pyutils import str2bool, set_seed, setup_logger
from utils import evaluate
from utils.tta_wrapper import SegmentationTTAWrapper


start_time = datetime.now()

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="luad")
    parser.add_argument("--cls_num_classes", type=int, default=4)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--palette_path", type=str, default="./datasets/luad_palette.npy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backbone", type=str, default="mit_b3")
    parser.add_argument("--split", type=str, default="test")

    args = parser.parse_args()

    return args


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    args = get_args()
    PALETTE = list(np.load(args.palette_path))
    set_seed(args.seed)

    args.dataset = args.dataset.lower()
    if args.dataset == "luad":
        args.seg_num_classes = 5
        args.cls_num_classes = 4
        args.img_root = "./data/LUAD-HistoSeg/{}/img/*.png".format(args.split)
        args.mask_root = "./data/LUAD-HistoSeg/{}/mask/*.png".format(args.split)
        CLASSES = ["TE", "NEC", "LYM", "TAS", "BACK"]
    elif args.dataset == "bcss":
        args.seg_num_classes = 5
        args.cls_num_classes = 4
        args.img_root = "./data/BCSS-WSSS/{}/img/*.png".format(args.split)
        args.mask_root = "./data/BCSS-WSSS/{}/mask/*.png".format(args.split)
        CLASSES = ["TUM", "STR", "LYM", "NEC", "BACK"]
    img_paths, mask_paths = glob.glob(args.img_root), glob.glob(args.mask_root)
    if len(mask_paths) == 0:
        mask_paths = [None] * len(img_paths)

    model_dir = re.split("/checkpoints/", args.model_path)[0]
    os.makedirs(os.path.join(model_dir, "test_log"), exist_ok=True)
    setup_logger(os.path.join(model_dir, "test_log", "eval_{}".format(args.split) + ".log"))
    logging.info(args)

    # get devices
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(args.gpu)
    logging.info("using {} device.".format(device))

    # create model
    model = SegNetwork(backbone=args.backbone,
                       stride=[4, 2, 2, 1],
                       seg_num_classes=args.seg_num_classes,
                       embedding_dim=256,
                       pretrained=True)

    logging.info("=================================================================================")
    weights_dict = torch.load(args.model_path, map_location='cpu')
    model_iter = weights_dict["iter"]
    weights_dict = weights_dict["model"]
    weights_dict = {k: v for k, v in weights_dict.items() if k in model.state_dict().keys()}
    logging.info('loading from checkpoint: {}'.format(os.path.basename(args.model_path)))

    # load weights
    model.load_state_dict(weights_dict, strict=True)
    model.to(device)
    model.eval()

    if args.dataset == "luad":
        MEAN, STD = [0.69164956, 0.50502165, 0.73429191], [0.15521939, 0.22774934, 0.18846453]
    elif args.dataset == "bcss":
        MEAN, STD = [0.66791496, 0.47791372, 0.70623304], [0.1736589, 0.22564577, 0.19820057]


    transform = A.Compose([
        A.Normalize(MEAN, STD),
        ToTensorV2(transpose_mask=True)
    ])

    tta_model = SegmentationTTAWrapper(model, tta.aliases.d4_transform(), merge_mode='mean')
    seg_matrix = evaluate.ConfusionMatrixAllClass(num_classes=args.seg_num_classes)
    seg_matrix.reset()

    with torch.no_grad():
        for index, img_path in enumerate(tqdm(img_paths, ncols=100, ascii=' >=')):
            mask_path = mask_paths[index]
            img_name = os.path.basename(img_path)[:-4]

            img = cv.imread(img_path, cv.IMREAD_UNCHANGED)
            mask = np.array(Image.open(mask_path))[:, :, None].astype(np.uint8) if mask_path is not None else np.zeros((img.shape[0], img.shape[1], 1))

            transdormed = transform(image=img, mask=mask)
            inputs = transdormed["image"].float().unsqueeze(dim=0).cuda()
            gt_mask = transdormed["mask"].float().unsqueeze(dim=0)
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

            segs = tta_model(inputs)

            # priori mask, but no priori_mask is better
            if args.dataset == "luad":
                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                ret, binary = cv.threshold(gray, 200, 255, cv.THRESH_BINARY)
                binary = np.uint8(binary)
                dst = morphology.remove_small_objects(binary == 255, min_size=80, connectivity=1).astype(np.uint8)
                priori_bg_mask = dst.reshape(1, img.shape[0], img.shape[1])
                priori_bg_mask = torch.from_numpy(priori_bg_mask).cuda()
                segs[:, -1, :, :] *= priori_bg_mask

            output = torch.argmax(segs, dim=1, keepdim=True).long().cpu()
            seg_matrix.update(gt_mask.clone(), output.clone())

        logging.info("=============================================================================================================")
        logging.info(str(seg_matrix))
        end_time = datetime.now()
        logging.info("infference finished, cost time: {}".format((end_time - start_time).seconds // 60))


if __name__ == '__main__':
    main()
