#!/bin/bash
#SBATCH --job-name=augmixnopre
#SBATCH --gres gpu:16
#SBATCH --nodes 1
#SBATCH --cpus-per-task=80
#SBATCH --partition=multigpu

# python src/wise_ft.py   \
#     --train-dataset=ImageNet  \
#     --epochs=10  \
#     --lr=0.00003  \
#     --batch-size=512  \
#     --cache-dir=cache  \
#     --model=ViT-B/32  \
#     --eval-datasets=ImageNet,ImageNetV2,ImageNetR,ImageNetA,ImageNetSketch  \
#     --template=openai_imagenet_template  \
#     --results-db=results.jsonl  \
#     --save=models/wiseft/ViTB32  \
#     --data-location=~/data \
#     --alpha 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0

export PYTHONPATH="$PYTHONPATH:$PWD"
NAME="B16-augmix-3-2-1-nopre"
python src/wise_ft.py   \
    --train-dataset=ImageNet  \
    --epochs=10  \
    --lr=0.00003  \
    --batch-size=512  \
    --cache-dir=cache  \
    --model=ViT-B/16  \
    --eval-datasets=ImageNet,ImageNetV2,ImageNetR,ImageNetA,ImageNetSketch \
    --template=openai_imagenet_template  \
    --results-db=Results/${NAME}.jsonl  \
    --save=models/wiseft/${NAME}\\ \
    --data-location=/nfs/users/ext_sanoojan.baliah/Sanoojan/data/ \
    --alpha 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 > Outs/${NAME}.out


# /nfs/users/ext_muzammal.naseer/kanchana_v0/data/imagenet

# CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,8,9,10,11,12,13,14,15