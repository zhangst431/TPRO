import argparse
import datetime
import logging
import os
import wandb
import numpy as np
import torch
from tqdm import tqdm
import random
from omegaconf import OmegaConf

import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from utils.optimizer import PolyWarmupAdamW
from utils.pyutils import str2bool, set_seed, setup_logger, AverageMeter
from utils.evaluate import ConfusionMatrixAllClass
from utils.trainutils import get_wsss_dataset
from seg_network.model import SegNetwork

start_time = datetime.datetime.now()

parser = argparse.ArgumentParser()
parser.add_argument("--config", default=None, type=str)
parser.add_argument("--local_rank", default=0, type=int, help="local_rank")
parser.add_argument('--backend', default='nccl')
parser.add_argument("--wandb_log", type=str2bool, default=False)
args = parser.parse_args()


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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


def validate(model=None, data_loader=None, cfg=None, loss_func=None):
    model.eval()
    avg_meter = AverageMeter()
    seg_matrix = ConfusionMatrixAllClass(num_classes=cfg.dataset.seg_num_classes)
    with torch.no_grad():
        for _, data in tqdm(enumerate(data_loader),
                            total=len(data_loader), ncols=100, ascii=" >="):
            name, inputs, cls_label, labels = data

            inputs = inputs.cuda()
            b, c, h, w = inputs.shape
            labels = labels.cuda().long()

            _, segs, attns = model(inputs, )

            loss = loss_func(segs, labels)

            avg_meter.add({"loss": loss})

            seg_matrix.update(labels, torch.argmax(segs, dim=1))

            # break

    loss = avg_meter.pop('loss')
    seg_score = seg_matrix.compute()[2]
    model.train()
    return seg_score, loss


def train(cfg):
    num_workers = 10
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend=args.backend, )

    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)

    train_dataset, val_dataset = get_wsss_dataset(cfg)
    logging.info("use {} images for training, {} images for validation".format(len(train_dataset), len(val_dataset)))

    g = torch.Generator()
    g.manual_seed(0)
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.train.samples_per_gpu,
                              num_workers=num_workers,
                              pin_memory=True,
                              drop_last=False,
                              shuffle=False,
                              sampler=train_sampler,
                              worker_init_fn=seed_worker,
                              generator=g)
    iters_per_epoch = len(train_loader)
    cfg.train.max_iters = cfg.train.epoch * iters_per_epoch
    cfg.train.eval_iters = iters_per_epoch

    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=False,
                            drop_last=False)

    device = torch.device(args.local_rank)

    wetr = SegNetwork(backbone=cfg.model.backbone.config,
                      stride=cfg.model.backbone.stride,
                      seg_num_classes=cfg.dataset.seg_num_classes,
                      embedding_dim=256,
                      pretrained=True)
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

    # for n_iter in tqdm(range(cfg.train.max_iters), total=cfg.train.max_iters, dynamic_ncols=True):
    avg_meter = AverageMeter()

    loss_func = nn.CrossEntropyLoss()

    best_seg_IOU = 0.0

    for n_iter in range(cfg.train.max_iters):
        wetr.train()
        try:
            img_name, inputs, cls_labels, gt_label = next(train_loader_iter)
        except:
            train_sampler.set_epoch(int((n_iter + 1) / iters_per_epoch))
            train_loader_iter = iter(train_loader)
            img_name, inputs, cls_labels, gt_label = next(train_loader_iter)

        inputs = inputs.to(device)
        gt_label = gt_label.to(device).long()

        _, segs, attns = wetr(inputs)

        loss = loss_func(segs, gt_label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        dist.all_reduce(loss)
        loss = loss / 2
        avg_meter.add({'loss': loss, })

        if args.local_rank == 0:
            if (n_iter + 1) % 100 == 0:
                delta, eta = cal_eta(time0, n_iter + 1, cfg.train.max_iters)
                cur_lr = optimizer.param_groups[0]['lr']
                logging.info(
                    "Iter: %d/%d; Elasped: %s; ETA: %s; LR: %.3e; loss: %.4f" % (
                    n_iter + 1, cfg.train.max_iters, delta, eta, cur_lr, loss))

            if args.wandb_log:
                iter_wandb_log = {"iter_log/lr{}".format(i): x["lr"] for i, x in enumerate(optimizer.param_groups)}
                iter_wandb_log.update({"iter_log/iter_train_loss": loss.item()})
                wandb.log(iter_wandb_log, step=n_iter)

            if (n_iter + 1) % cfg.train.eval_iters == 0:
                loss = avg_meter.pop('loss')

                wandb_log = {
                    "loss/train_loss": loss,
                }

                val_IOU, val_loss = validate(model=wetr, data_loader=val_loader, cfg=cfg, loss_func=loss_func)
                logging.info("seg score: {}, mIoU: {:.4f}".format(val_IOU, val_IOU[:-1].mean()))
                wandb_log.update({
                    "loss/val_loss": val_loss,
                    "miou/val_mIOU": val_IOU[:-1].mean()
                })
                if args.wandb_log:
                    wandb.log(wandb_log, step=n_iter)
                state_dict = {
                    "cfg": cfg,
                    "iter": n_iter,
                    "optimizer": optimizer.state_dict(),
                    "model": wetr.module.state_dict()
                }
                if val_IOU[:-1].mean() > best_seg_IOU:
                    best_seg_IOU = val_IOU[:-1].mean()
                    torch.save(state_dict, os.path.join(cfg.work_dir.ckpt_dir, "best_seg.pth"))

    if args.local_rank == 0:
        torch.cuda.empty_cache()
        logging.info("start test seg......")
        logging.info(
            "python evaluate_seg.py --model_path '{}' --gpu 0 --backbone {} --dataset {} ".format(
                os.path.join(cfg.work_dir.ckpt_dir, "best_seg.pth"),
                cfg.model.backbone.config, cfg.dataset.name))
        os.system(
            "python evaluate_seg.py --model_path '{}' --gpu 0 --backbone {} --dataset {} ".format(
                os.path.join(cfg.work_dir.ckpt_dir, "best_seg.pth"),
                cfg.model.backbone.config, cfg.dataset.name))
        logging.info("test seg finished.......")
        logging.info("start val seg......")
        logging.info("python evaluate_seg.py --model_path '{}' --gpu 0 --backbone {} --dataset {} --split valid".format(
            os.path.join(cfg.work_dir.ckpt_dir, "best_seg.pth"),
            cfg.model.backbone.config, cfg.dataset.name))
        os.system("python evaluate_seg.py --model_path '{}' --gpu 0 --backbone {} --dataset {} --split valid".format(
            os.path.join(cfg.work_dir.ckpt_dir, "best_seg.pth"),
            cfg.model.backbone.config, cfg.dataset.name))
        logging.info("val seg finished.......")
    dist.barrier()
    torch.cuda.empty_cache()

    end_time = datetime.datetime.now()
    logging.info(f'cost time: {end_time - start_time}')


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
            wandb.init(project="TPRO-{}-wsss".format(cfg.dataset.name))
        setup_logger(filename=os.path.join(cfg.work_dir.train_log_dir, timestamp + '.log'))
        logging.info('\nargs: %s' % args)
        logging.info('\nconfigs: %s' % cfg)

    # fix random seed
    set_seed(0)
    train(cfg=cfg)
