

## Requirements
Pytorch 1.7.0,
timm 0.3.2,
torchprofile 0.0.4,
apex

- Evaluate example:
```
python train.py /path/to/imagenet/ --model multiPatternGCN -b 128 --pretrain_path /path/to/pretrained/model/ --evaluate
```

## Training

- Training ViG on 8 GPUs:
```
python -m torch.distributed.launch --nproc_per_node=8 train.py /path/to/imagenet/ --model multiPatternGCN --sched cosine --epochs 300 --opt adamw -j 8 --warmup-lr 1e-6 --mixup .8 --cutmix 1.0 --model-ema --model-ema-decay 0.99996 --aa rand-m9-mstd0.5-inc1 --color-jitter 0.4 --warmup-epochs 20 --opt-eps 1e-8 --repeated-aug --remode pixel --reprob 0.25 --amp --lr 2e-3 --weight-decay .05 --drop 0 --drop-path .1 -b 128 --output /path/to/save/models/
```

- Training Pyramid ViG on 8 GPUs:
```
python -m torch.distributed.launch --nproc_per_node=8 train.py /path/to/imagenet/ --model multiPatternGCN --sched cosine --epochs 300 --opt adamw -j 8 --warmup-lr 1e-6 --mixup .8 --cutmix 1.0 --model-ema --model-ema-decay 0.99996 --aa rand-m9-mstd0.5-inc1 --color-jitter 0.4 --warmup-epochs 20 --opt-eps 1e-8 --repeated-aug --remode pixel --reprob 0.25 --amp --lr 2e-3 --weight-decay .05 --drop 0 --drop-path .1 -b 128 --output /path/to/save/models/
```



