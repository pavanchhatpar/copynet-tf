if [ ! -z $1 ]; then
    if [ $1 = "--no-docker" ]; then
        build_docker=false
    else
        build_docker=true
    fi
else
    build_docker=true
fi

if [ ! -f ".env" ]; then
    echo "Refer to sample.env to make .env file as per your project"
    echo "See README for more details"
    exit 2
fi

if [ -f "lock.env" ]; then
    source "lock.env"
    if [ "$build_docker" = true ]; then
        docker rmi copynet-tf/nb:latest
        docker rmi copynet-tf/bash:latest
    fi
    rm -rf $PYTHON_VENV_PATH
    rm ./data
    rm "lock.env"
fi

source ".env"
if [ -z "$DATA_DIR" ]; then
    echo ".env should set DATA_DIR"
    exit 1
fi
if [ -z "$PYTHON_VENV_PATH" ]; then
    echo ".env should set PYTHON_VENV_PATH"
    exit 1
fi
if [ "$build_docker" = true ]; then
    if [ -z "$JUPYTER_PASSWD" ]; then
        echo ".env should set JUPYTER_PASSWD"
        exit 1
    fi
fi

ln -s $DATA_DIR ./data
if [ "$build_docker" = true ]; then
    docker build --build-arg JUPYTER_PASSWD --target bash -t copynet-tf/bash:latest .
    docker build --build-arg JUPYTER_PASSWD --target nb -t copynet-tf/nb:latest .
fi

python3 -m venv $PYTHON_VENV_PATH
source $PYTHON_VENV_PATH/bin/activate\
    && pip install --upgrade pip setuptools\
    && pip install --no-cache-dir -r requirements.txt\
    && python -m spacy download en_core_web_sm

cp ".env" "lock.env"
echo "***Environment is ready***"
if [ "$build_docker" = true ]; then
    echo ""
    echo "-- Docker images prepared --"
    echo "copynet-tf/bash:latest"
    echo "copynet-tf/nb:latest"
    echo ""
    echo "-- Run commands --"
    echo "./start-env.sh bash"
    echo "./start-env.sh nb"
fi
