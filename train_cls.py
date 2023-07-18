import argparse
import datetime
import logging
import os
import wandb
import numpy as np
import cv2 as cv
from omegaconf import OmegaConf
from tqdm import tqdm
import ttach as tta
from skimage import morphology

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from utils.trainutils import get_cls_dataset, all_reduced
from utils.optimizer import PolyWarmupAdamW
from utils.pyutils import str2bool, set_seed, setup_logger, AverageMeter
from utils.evaluate import ConfusionMatrixAllClass
from cls_network.model import ClsNetwork


start_time = datetime.datetime.now()

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default=None)
parser.add_argument("--local_rank", default=0, type=int, help="local_rank")
parser.add_argument('--backend', default='nccl')
parser.add_argument("--wandb_log", type=str2bool, default=False)
args = parser.parse_args()


def get_seg_label(cams, inputs, label):
    with torch.no_grad():
        b, c, h, w = inputs.shape
        label = label.view(b, -1, 1, 1).cpu().data.numpy()
        cams = cams.cpu().data.numpy()
        cams = np.maximum(cams, 0)
        channel_max = np.max(cams, axis=(2, 3), keepdims=True)
        channel_min = np.min(cams, axis=(2, 3), keepdims=True)
        cams = (cams - channel_min) / (channel_max - channel_min + 1e-6)
        cams = cams * label
        cams = torch.from_numpy(cams)
        cams = F.interpolate(cams, size=(h, w), mode="bilinear", align_corners=False)
        cam_max = torch.max(cams, dim=1, keepdim=True)[0]
        bg_cam = (1 - cam_max) ** 10
        cam_all = torch.cat([cams, bg_cam], dim=1)

    return cam_all


def cal_eta(time0, cur_iter, total_iter):
    time_now = datetime.datetime.now()
    time_now = time_now.replace(microsecond=0)
    # time_now = datetime.datetime.strptime(time_now.strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')

    scale = (total_iter - cur_iter) / float(cur_iter)
    delta = (time_now - time0)
    eta = (delta * scale)
    time_fin = time_now + eta
    eta = time_fin.replace(microsecond=0) - time_now
    return str(delta), str(eta)


def validate(model=None, data_loader=None, cfg=None, cls_loss_func=None):
    model.eval()
    avg_meter = AverageMeter()
    fuse234_matrix = ConfusionMatrixAllClass(num_classes=cfg.dataset.cls_num_classes + 1)
    tta_transform = tta.Compose([
        tta.HorizontalFlip(),
        tta.Multiply(factors=[0.9, 1.0, 1.1])
    ])
    with torch.no_grad():
        for data in tqdm(data_loader,
                         total=len(data_loader), ncols=100, ascii=" >="):
            name, inputs, cls_label, labels = data

            inputs = inputs.cuda().float()
            b, c, h, w = inputs.shape
            labels = labels.cuda()
            cls_label = cls_label.cuda().float()

            cls1, cam1, cls2, cam2, cls3, cam3, cls4, cam4, attns = model(inputs, )

            cls_loss1 = cls_loss_func(cls1, cls_label)
            cls_loss2 = cls_loss_func(cls2, cls_label)
            cls_loss3 = cls_loss_func(cls3, cls_label)
            cls_loss4 = cls_loss_func(cls4, cls_label)
            cls_loss = cfg.train.l1 * cls_loss1 + cfg.train.l2 * cls_loss2 + cfg.train.l3 * cls_loss3 + cfg.train.l4 * cls_loss4

            cls4 = (torch.sigmoid(cls4) > 0.5).float()
            all_cls_acc4 = (cls4 == cls_label).all(dim=1).float().sum() / cls4.shape[0] * 100
            avg_cls_acc4 = ((cls4 == cls_label).sum(dim=0) / cls4.shape[0]).mean() * 100
            avg_meter.add({"all_cls_acc4": all_cls_acc4, "avg_cls_acc4": avg_cls_acc4, "cls_loss": cls_loss})
            # get eval cams
            cams1 = []
            cams2 = []
            cams3 = []
            cams4 = []
            for tta_trans in tta_transform:
                augmented_tensor = tta_trans.augment_image(inputs)
                cls1, cam1, cls2, cam2, cls3, cam3, cls4, cam4, attns = model(augmented_tensor)
                cam1 = get_seg_label(cam1, augmented_tensor, torch.sigmoid(cls4) > 0.15).cuda()
                cam1 = tta_trans.deaugment_mask(cam1).unsqueeze(dim=0)
                cams1.append(cam1)
                cam2 = get_seg_label(cam2, augmented_tensor, torch.sigmoid(cls4) > 0.15).cuda()
                cam2 = tta_trans.deaugment_mask(cam2).unsqueeze(dim=0)
                cams2.append(cam2)
                cam3 = get_seg_label(cam3, augmented_tensor, torch.sigmoid(cls4) > 0.15).cuda()
                cam3 = tta_trans.deaugment_mask(cam3).unsqueeze(dim=0)
                cams3.append(cam3)
                cam4 = get_seg_label(cam4, augmented_tensor, torch.sigmoid(cls4) > 0.15).cuda()
                cam4 = tta_trans.deaugment_mask(cam4).unsqueeze(dim=0)
                cams4.append(cam4)
            cams1 = torch.cat(cams1, dim=0).mean(dim=0)
            cams2 = torch.cat(cams2, dim=0).mean(dim=0)
            cams3 = torch.cat(cams3, dim=0).mean(dim=0)
            cams4 = torch.cat(cams4, dim=0).mean(dim=0)

            # priori mask
            if cfg.dataset.name == "luad":
                img = cv.imread(os.path.join(cfg.dataset.val_root, 'valid', 'img', name[0]), cv.IMREAD_UNCHANGED)
                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                ret, binary = cv.threshold(gray, 200, 255, cv.THRESH_BINARY)
                binary = np.uint8(binary)
                dst = morphology.remove_small_objects(binary == 255, min_size=80, connectivity=1).astype(np.uint8)
                priori_bg_mask = (1 - dst).reshape(1, 1, img.shape[0], img.shape[1])
                priori_bg_mask = torch.from_numpy(priori_bg_mask).cuda()
                cams1[:, :-1, :, :] *= priori_bg_mask
                cams2[:, :-1, :, :] *= priori_bg_mask
                cams3[:, :-1, :, :] *= priori_bg_mask
                cams4[:, :-1, :, :] *= priori_bg_mask

            fuse234 = 0.3 * cams2 + 0.3 * cams3 + 0.4 * cams4
            fuse_label234 = torch.argmax(fuse234, dim=1).cuda()

            fuse234_matrix.update(labels.detach().clone(), fuse_label234.clone())

    all_cls_acc4, avg_cls_acc4, cls_loss = avg_meter.pop('all_cls_acc4'), avg_meter.pop("avg_cls_acc4"), avg_meter.pop("cls_loss")
    fuse234_score = fuse234_matrix.compute()[2]
    model.train()
    return all_cls_acc4, avg_cls_acc4, fuse234_score, cls_loss


def train(cfg):
    num_workers = 10

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend=args.backend)
    gpu_ids = [int(x) for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
    n_gpus = len(gpu_ids)

    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)

    # ============ prepare data ============
    # data augmentation
    train_dataset, val_dataset = get_cls_dataset(cfg)
    logging.info("use {} images for training, {} images for validation".format(len(train_dataset), len(val_dataset)))
    logging.info(f"use {n_gpus} GPUs: {gpu_ids}")

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.train.samples_per_gpu,
                              num_workers=num_workers,
                              pin_memory=True,
                              drop_last=False,
                              shuffle=False,
                              sampler=train_sampler,
                              prefetch_factor=4)
    iters_per_epoch = len(train_loader)
    cfg.train.max_iters = cfg.train.epoch * iters_per_epoch
    cfg.train.eval_iters = iters_per_epoch
    cfg.scheduler.warmup_iter = cfg.scheduler.warmup_iter * iters_per_epoch

    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=False,
                            drop_last=False)

    device = torch.device(args.local_rank)

    # ============ prepare model ============
    wetr = ClsNetwork(backbone=cfg.model.backbone.config,
                      stride=cfg.model.backbone.stride,
                      cls_num_classes=cfg.dataset.cls_num_classes,
                      n_ratio=cfg.model.n_ratio,
                      pretrained=cfg.train.pretrained,
                      k_fea_path=cfg.model.knowledge_feature_path,
                      l_fea_path=cfg.model.label_feature_path)

    logging.info('\nNetwork config: \n%s' % (wetr))
    param_groups = wetr.get_param_groups()
    wetr.to(device)
    wetr.train()

    optimizer = PolyWarmupAdamW(
        params=param_groups,
        lr=cfg.optimizer.learning_rate,
        weight_decay=cfg.optimizer.weight_decay,
        betas=cfg.optimizer.betas,
        warmup_iter=cfg.scheduler.warmup_iter,
        max_iter=cfg.train.max_iters,
        warmup_ratio=cfg.scheduler.warmup_ratio,
        power=cfg.scheduler.power
    )
    logging.info('\nOptimizer: \n%s' % optimizer)
    train_loader_iter = iter(train_loader)
    train_sampler.set_epoch(np.random.randint(cfg.train.max_iters))
    wetr = DistributedDataParallel(wetr, device_ids=[args.local_rank], find_unused_parameters=True)

    avg_meter = AverageMeter()

    loss_function = nn.BCEWithLogitsLoss()

    best_fuse234_dice = 0.0

    for n_iter in range(cfg.train.max_iters):
        wetr.train()
        try:
            img_name, inputs, cls_labels, gt_label = next(train_loader_iter)
        except:
            train_sampler.set_epoch(int((n_iter + 1) / iters_per_epoch))
            train_loader_iter = iter(train_loader)
            img_name, inputs, cls_labels, gt_label = next(train_loader_iter)

        inputs = inputs.to(device).float()
        cls_labels = cls_labels.to(device).float()

        cls1, cam1, cls2, cam2, cls3, cam3, cls4, cam4, attns = wetr(inputs)

        loss1 = loss_function(cls1, cls_labels)
        loss2 = loss_function(cls2, cls_labels)
        loss3 = loss_function(cls3, cls_labels)
        loss4 = loss_function(cls4, cls_labels)
        loss = cfg.train.l1 * loss1 + cfg.train.l2 * loss2 + cfg.train.l3 * loss3 + cfg.train.l4 * loss4

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cls_pred4 = (torch.sigmoid(cls4) > 0.5).float()
        all_cls_acc4 = (cls_pred4 == cls_labels).all(dim=1).float().sum() / cls_labels.shape[0] * 100
        avg_cls_acc4 = ((cls_pred4 == cls_labels).sum(dim=0) / cls_labels.shape[0]).mean() * 100

        all_reduced(loss, n_gpus)
        all_reduced(all_cls_acc4, n_gpus)
        all_reduced(avg_cls_acc4, n_gpus)

        avg_meter.add({'cls_loss': loss,"all_cls_acc4": all_cls_acc4, "avg_cls_acc4": avg_cls_acc4})

        if args.local_rank == 0:
            if (n_iter + 1) % 100 == 0:
                delta, eta = cal_eta(time0, n_iter + 1, cfg.train.max_iters)
                cur_lr = optimizer.param_groups[0]['lr']
                logging.info(
                    "Iter: %d / %d; Elasped: %s; ETA: %s; LR: %.3e; cls_loss: %.4f; all_acc4: %.2f; avg_acc4: %.2f" % (
                        n_iter + 1, cfg.train.max_iters, delta, eta, cur_lr, loss,
                        all_cls_acc4, avg_cls_acc4))

            if args.wandb_log:
                iter_wandb_log = {"iter_log/lr{}".format(i): x["lr"] for i, x in enumerate(optimizer.param_groups)}
                iter_wandb_log.update({"iter_log/iter_train_loss": loss.item()})
                wandb.log(iter_wandb_log, step=n_iter)

            if (n_iter + 1) % cfg.train.eval_iters == 0 or (n_iter + 1) == cfg.train.max_iters:
                cls_loss, all_cls_acc4, avg_cls_acc4 = avg_meter.pop('cls_loss'), avg_meter.pop('all_cls_acc4'), avg_meter.pop("avg_cls_acc4")

                wandb_log = {
                    "loss/train_loss": cls_loss,
                    "acc/all_cls_acc4": all_cls_acc4,
                    "acc/avg_cls_acc4": avg_cls_acc4,
                }

                val_all_acc4, val_avg_acc4, fuse234_score, val_cls_loss = validate(model=wetr,
                                                                                   data_loader=val_loader,
                                                                                   cfg=cfg,
                                                                                   cls_loss_func=loss_function)
                logging.info("val all acc4: %.6f" % (val_all_acc4))
                logging.info("val avg acc4: %.6f" % (val_avg_acc4))
                logging.info("fuse234 score: {}, mIOU: {:.4f}".format(fuse234_score, fuse234_score[:-1].mean()))
                wandb_log.update({
                    "acc/val_all_acc4": val_all_acc4,
                    "acc/val_avg_acc4": val_avg_acc4,
                    "loss/val_loss": val_cls_loss,
                    "miou/val_mIOU_fuse234": fuse234_score[:-1].mean()
                })
                if args.wandb_log:
                    wandb.log(wandb_log, step=n_iter)
                state_dict = {
                    "cfg": cfg,
                    "iter": n_iter,
                    "optimizer": optimizer.state_dict(),
                    "model": wetr.module.state_dict()
                }
                if fuse234_score[:-1].mean() > best_fuse234_dice:
                    best_fuse234_dice = fuse234_score[:-1].mean()
                    torch.save(state_dict, os.path.join(cfg.work_dir.ckpt_dir, "best_cam.pth"))

    if args.local_rank == 0:
        torch.cuda.empty_cache()
        logging.info("start test seg......")
        logging.info(
            "python evaluate_cls.py --model_path '{}' --gpu 0 --backbone {} "
            "--dataset {} --label_feature_path '{}' --knowledge_feature_path '{}' --n_ratio {}".format(
                os.path.join(cfg.work_dir.ckpt_dir, "best_cam.pth"),
                cfg.model.backbone.config,
                cfg.dataset.name, cfg.model.label_feature_path,
                cfg.model.knowledge_feature_path, cfg.model.n_ratio))
        os.system(
            "python evaluate_cls.py --model_path '{}' --gpu 0 --backbone {} "
            "--dataset {} --label_feature_path '{}' --knowledge_feature_path '{}' --n_ratio {}".format(
                os.path.join(cfg.work_dir.ckpt_dir, "best_cam.pth"),
                cfg.model.backbone.config,
                cfg.dataset.name, cfg.model.label_feature_path,
                cfg.model.knowledge_feature_path, cfg.model.n_ratio))
        logging.info("test seg finished.......")
        logging.info("start val seg......")
        logging.info(
            "python evaluate_cls.py --model_path '{}' --gpu 0 --backbone {} "
            " --split valid --dataset {} --label_feature_path '{}' --knowledge_feature_path '{}' --n_ratio {}".format(
                os.path.join(cfg.work_dir.ckpt_dir, "best_cam.pth"),
                cfg.model.backbone.config, cfg.dataset.name,
                cfg.model.label_feature_path, cfg.model.knowledge_feature_path,
                cfg.model.n_ratio))
        os.system(
            "python evaluate_cls.py --model_path '{}' --gpu 0 --backbone {} "
            "--split valid --dataset {} --label_feature_path '{}' --knowledge_feature_path '{}' --n_ratio {}".format(
                os.path.join(cfg.work_dir.ckpt_dir, "best_cam.pth"),
                cfg.model.backbone.config, cfg.dataset.name,
                cfg.model.label_feature_path, cfg.model.knowledge_feature_path,
                cfg.model.n_ratio))
        logging.info("val seg finished.......")
    
    dist.barrier()

    end_time = datetime.datetime.now()
    logging.info(f'cost {end_time- start_time}')


if __name__ == "__main__":
    cfg = OmegaConf.load(args.config)

    cfg.work_dir.dir = os.path.dirname(args.config)

    timestamp = "{0:%Y-%m-%d-%H-%M}".format(datetime.datetime.now())

    cfg.work_dir.ckpt_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.ckpt_dir, timestamp)
    cfg.work_dir.pred_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.pred_dir)
    cfg.work_dir.train_log_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.train_log_dir)

    os.makedirs(cfg.work_dir.dir, exist_ok=True)
    os.makedirs(cfg.work_dir.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.pred_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.train_log_dir, exist_ok=True)

    if args.local_rank == 0:
        if args.wandb_log:
            wandb.init(project=f'TPRO-{cfg.dataset.name}-cls')
        setup_logger(filename=os.path.join(cfg.work_dir.train_log_dir, timestamp + '.log'))
        logging.info('\nargs: %s' % args)
        logging.info('\nconfigs: %s' % cfg)

    # fix random seed
    set_seed(0)
    train(cfg=cfg)
