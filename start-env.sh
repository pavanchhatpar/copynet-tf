#!/bin/bash
if [ ! -f "lock.env" ]; then
    echo "run `./setup-env.sh` first"
    exit 1
fi

source "lock.env"

# if [ "$1" == "nb" ]; then
docker run -it --gpus all --rm\
    -u $(id -u):$(id -g)\
    -v "`pwd`:/tf/src"\
    -v "`echo $DATA_DIR`:`echo $DATA_DIR`"\
    -p 8991:8888\
    copynet-tf/$1:latest
# else
#     docker run -it --gpus all --rm\
#         -u $(id -u):$(id -g)\
#         -v "`pwd`:/tf/src"\
#         -v "`echo $DATA_DIR`:`echo $DATA_DIR`"\
#         copynet-tf/$1:latest
# fi