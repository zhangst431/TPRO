## TPRO: Text-prompting-based Weakly Supervised Histopathology Tissue Segmentation
This is the official pytorch implementation of our MICCAI 2023 paper "TPRO: Text-prompting-based Weakly Supervised Histopathology Tissue Segmentation".

![frame_work](./figures/framework.png)

### Preparation
Download [LUAD-HistoSeg](https://drive.google.com/drive/folders/1E3Yei3Or3xJXukHIybZAgochxfn6FJpr) and [BCSS-WSSS](https://drive.google.com/drive/folders/1iS2Z0DsbACqGp7m6VDJbAcgzeXNEFr77) datasets and orgnize the directory sctructure in the following format:

```
data/
|--LUAD-HistoSeg
   |--train
      |--img
   |--test
      |--img
      |--mask
   |--valid
      |--img
      |--mask
|--BCSS-WSSS
   |--train
      |--img
   |--test
      |--img
      |--mask
   |--valid
      |--img
      |--mask
```
The ImageNet-1k pre-trained weights of vision encoder can be download from the  official [SegFormer](https://github.com/NVlabs/SegFormer#training) implementation.

### Train a CLassification Network
```bash
# LUAD-HistoSeg
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 16732 train_cls.py --config ./work_dirs/luad/classification/config.yaml
# BCSS-WSSS
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 16372 train_cls.py --config ./work_dirs/bcss/classification/config.yaml
```

### Extract Pseudo Labels
```bash
# LUAD-HistoSeg
python evaluate_cls.py --dataset luad --model_path path/to/classification/model --save_dir ./work_dirs/luad/classification/predictions --split train
# BCSS-WSSS
python evaluate_cls.py --dataset bcss --model_path path/to/classification/model --save_dir ./work_dirs/bcss/classification/predictions --split train
```

### Train a Segmentation Network
```bash
# LUAD-HistoSeg
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 16372 train_seg.py --config ./work_dirs/luad/segmentation/config.yaml
# BCSS-WSSS
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 16372 train_seg.py --config ./work_dirs/bcss/segmentation/config.yaml
```
