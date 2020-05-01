# CopyNet implementation with TensorFlow 2
 - Incorporating Copying Mechanism in Sequence-to-Sequence Learning
 - Uses `TensorFlow 2.0` and above APIs with `tf.keras` too
 - Adapted from AllenNLP's PyTorch implementation, their blog referenced 
 below was very helpful to understand the math from an implementation
 perspective

![Python package](https://github.com/pavanchhatpar/copynet-tf/workflows/Python%20package/badge.svg)
![Upload Python Package](https://github.com/pavanchhatpar/copynet-tf/workflows/Upload%20Python%20Package/badge.svg)

## Environment to run examples
### Setup
- Copy `sample.env` to `.env` and enter appropriate values for the variables
 - A brief description of each is provided as a comment in that file
 - Post that run,
   ```bash
   ./setup-env.sh [--no-docker]
   ```
 - Uses env file to configure project environment
 - Builds required docker images (if you don't wanna use Docker then pass 
   `--no-docker` option to the `setup-env.sh` script)
 - Makes a python environment and installes required packages in it
 - Prepares an `lock.env` file. Do not edit/ delete it

### Rebuilding environment
 - You may change environment config in the process of development
 - This includes adding a new python package to requirements.txt
 - After changing run,
    ```
    ./setup-env.sh [--no-docker]
    ```
 - If you do not want Docker, then pass `--no-docker` option similar to before

### Start environment
 - At the end of setup script you will be shown the commands to start the 
 environments
 - They are,
   ```bash
   ./start-env.sh nb    # For Dockerized jupyter server
   ./start-env.sh bash  # For Dockerized bash
   ```
 - It is not necessary to use the `start-env.sh` script for virtualenv, the
 regular `source` command to activate it is enough

### Note on Dockerized environment
 - The dockerized environment is specifically helpful and recommended when 
 using `GPU`
 - It takes care of many nuances involved in setting up CUDA. Your host machine
 should just have correct NVIDIA drivers and nothing else
 - It is recommended to run the examples in this environment to ensure all
 correct dependencies are met

## Run examples
 - Instructions to run an example are detailed in its own folders respectively

## References
 - Incorporating Copying Mechanism in Sequence-to-Sequence Learning: ([paper](https://arxiv.org/abs/1603.06393))
 - AllenNLP implementation: ([blog](https://medium.com/@epwalsh10/incorporating-a-copy-mechanism-into-sequence-to-sequence-models-40917280b89d)) ([code](https://github.com/epwalsh/nlp-models))
 - BLEU score metric: ([code](https://github.com/tensorflow/nmt/blob/master/nmt/scripts/bleu.py))