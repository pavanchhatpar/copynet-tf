# Examples
## Run the examples
 - It is recommended to run examples using the package
 - Instructions to run an example are detailed in its own folders respectively
 - The example has a Google colab link for end-to-end demo on running with
 package and source as well.
 - Refer next section to setup the local environment only if running via source

## Environment to run examples from source
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