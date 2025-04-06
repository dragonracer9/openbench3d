# Setup Guide for OpenBench3d in WSL Ubuntu 20.04 LTS

## Initial Setup of the Distro

1) Install WSL: run ```wsl --install``` in pwsh or cmd

2) Install the WSL Ubuntu 20.04 LTS distro by running ```wsl --install -d Ubuntu-20.04```

3) Run the WSL machine by running the command ```wsl -d ubuntu-20.04```

4) Set up your Ubuntu   with a username and password

5) Run ```sudo apt update && sudo apt upgrade -y``` to update your system

6) Install the c++ build tools and essential linear algebra tooling through ```sudo apt install build-essential libopenblas-dev```

7) Install CUDA 11.3 by running  the follwing commands, which may be found though this [link](https://developer.nvidia.com/cuda-11.3.0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local).
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda-repo-wsl-ubuntu-11-3-local_11.3.0-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-11-3-local_11.3.0-1_amd64.deb
sudo apt-key add /var/cuda-repo-wsl-ubuntu-11-3-local/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda
```

8) Install Anaconda on WSL by going to [Anaconda's Website](https://www.anaconda.com/download/success) and copying the LINK to the 64-bit Linux installer (currently at the time of writing, the installer link is https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh) and running
```bash
wget <link to conda download>
bash Anaconda[YOUR VERSION].sh # Anaconda3-2024.10-1-Linux-x86_64.sh here
```

9) Complete Anaconda install by following the instructions given by the installer (iirc you should say yes to initialising conda at boot or however they phrase it - it's in the menu at almost the very end or something anyway)

10) Reboot the WSL machine

## Install Requirements

1) Run the setup instructions as described in the [README.md](README.md) until step 3, that is,
```bash
conda create --name=openmask3d python=3.8.5 # create new virtual environment
conda activate openmask3d # activate it
bash install_requirements.sh  # install requirements
```

**IF you run into issues installing  the MinkowskiEngine package, try installing via the method used in [Mask3d](https://github.com/JonasSchult/Mask3D)**
```bash
git clone --recursive "https://github.com/NVIDIA/MinkowskiEngine"
cd MinkowskiEngine
git checkout 02fc608bea4c0549b0a7b00ca1bf15dee4a0b228
python setup.py install --force_cuda --blas=openblas
```
Then, run the rest of the install script.

For some reason, this doesnt import all necessary packages, and so

2) Install the remaining packages `pip` says are required by other packages. **NOTE: DO NOT CHANGE THE VERSIONS OF ANY ALREADY INSTALLED PACKEGES, JUST INSTALL THOSE PACKAGES `pip` SAYS ARE REQUIRED BUT NOT INSTALLED.** Run `pip install <package>` and let its dependency resolver figure out what version is needed. Unfortunately, I can't list the packages here, as I forgot which ones they are. 

**I've exported the conda environment to an [`environment.yml`](environment.yml) file which can be used to create the environment.** I cannot guarantee successful installation if installing via the ènvironemt.yml`file. More info can be found at https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html.

3) Once all the packages are installed, run the last step of the installation process from the [README](README.md)
```bash
pip install -e .  # install current repository in editable mode
```

