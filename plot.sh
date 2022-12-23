export PYTHONPATH="$PYTHONPATH:$PWD"
python src/scatter_plot.py  \
    --eval-datasets=ImageNetV2,ImageNetR,ImageNetA,ImageNetSketch  \
    --results-db=Results/results-bl-16.jsonl  \
    --save plots