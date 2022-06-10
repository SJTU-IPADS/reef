# REEF Installation

## Software Version
* Ubuntu 18.04
* ROCm 4.3.0
* CMake > 3.18
* grpc 1.45
* glog 0.6.0
* googletest 1.11.0 

## Installation Overview

The installation has six major steps:
1. Install ROCm-4.3
2. Install the customized GPU kernel driver (for reset-based preemption), and reboot
3. (Recommended, but Optional) create the ROCm docker container
4. Install the customized GPU runtime (hip, rocclr)
5. Install other software dependencies (e.g., grpc)
6. Build REEF

The customized GPU kernel driver and GPU runtime can be found [here](https://github.com/SJTU-IPADS/reef-artifacts/tree/master/reef-env).

## Install Dependencies

### Install ROCm-4.3
```sh
# Ensure the system is up to date.
$ sudo apt update
$ sudo apt dist-upgrade
$ sudo apt install libnuma-dev
$ sudo reboot

# Add the ROCm apt repository.
$ sudo apt install wget gnupg2
$ wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
$ echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/4.3/ ubuntu main' | sudo tee /etc/apt/sources.list.d/rocm.list
$ sudo apt update

# Install the ROCm package and reboot.
$ sudo apt install rocm-dkms && sudo reboot

# Add ROCm binaries to PATH.
$ echo 'export PATH=$PATH:/opt/rocm/bin:/opt/rocm/rocprofiler/bin:/opt/rocm/opencl/bin' | sudo tee -a /etc/profile.d/rocm.sh
```


### Build & Install the Customized Kernel Driver
```sh
$ git clone https://github.com/SJTU-IPADS/reef-artifacts.git
$ cd reef-artifacts/reef-env/amdgpu-dkms
# Notice: The script will reboot
$ ./update-kern-module.sh
```

### Build & Install rocclr
```sh
# in reef-artifacts/reef-env
$ export REEF_ENV_ROOT=`pwd`
$ cd rocclr
$ mkdir build
$ cd build
$ cmake -DOPENCL_DIR="${REEF_ENV_ROOT}/ROCm-OpenCL-Runtime" -DCMAKE_INSTALL_PREFIX=/opt/rocm/rocclr ..
$ sudo make install
```

### Build & Install hip
```sh
# in reef-artifacts/reef-env
$ export REEF_ENV_ROOT=`pwd`
$ cd hip
$ mkdir build
$ cd build
$ cmake -DCMAKE_PREFIX_PATH="${REEF_ENV_ROOT}/rocclr/build;/opt/rocm/hip" ..
$ sudo make install
```

### Install CMake
```sh
$ wget https://github.com/Kitware/CMake/releases/download/v3.22.4/cmake-3.22.4-linux-x86_64.sh
$ sh cmake-3.22.4-linux-x86_64
# you can also add this cmake version to ~/.bashrc
$ export PATH=~/cmake-3.22.4-linux-x86_64/bin:$PATH 
$ cmake --version
cmake version 3.22.4
```

### Install glog
```sh
$ git clone https://github.com/google/glog
$ cd glog
$ mkdir build; cd build
$ cmake ..
$ sudo make install
```

### Install gtest
```sh
$ git clone -b  https://github.com/google/googletest
$ cd googletest
$ mkdir build; cd build
$ cmake ..
$ sudo make install
```

### Install grpc + protobuf
```sh
$ git clone -b 1.45.0 https://github.com/grpc/grpc
$ cd grpc
$ git submodule update --init
$ mkdir -p cmake/build; cd cmake/build
$ cmake ../..
$ sudo make install
```


## Build REEF

### Build Resource
This step compiles the DNN models' device code.
```sh
$ cd resource
$ make
```

### Build REEF
```sh
$ mkdir build; cd build
$ cmake ..
$ make -j4
```

### Run tests
```sh
# in ./build
$ ./unit_test
```