export PYTHONPATH="$PYTHONPATH:$PWD"
python src/wise_ft.py   \
    --eval-datasets=ImageNet,ImageNetV2,ImageNetR,ImageNetA  \
    --load=models/wiseft/B16-distill-soft-5.0-check/zeroshot.pt,models/wiseft/B16-distill-soft-5.0-check/finetuned/checkpoint_10.pt  \
    --results-db=results.jsonl  \
    --save=models/wiseft/check  \
    --data-location=/share/data/drive_1 \
    --alpha 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0

# ,ImageNetV2,ImageNetR,ImageNetA,ImageNetSketch