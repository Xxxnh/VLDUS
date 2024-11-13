#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

CONFIG=$1
CHECKPOINT=$2
GPUS=$3

# $PYTHON -m torch.distributed.launch --master_port 10003 --nproc_per_node=$GPUS \
#     $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4}

# $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS \
#     $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4}

CUDA_VISIBLE_DEVICES=0,1,2,3 $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4}
# $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS \
#     $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4}
