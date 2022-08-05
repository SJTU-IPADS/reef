# REEF - Real-time GPU-accelerated DNN Inference Scheduling System

REEF is a real-time GPU-accelerated DNN inference scheduling system that supports instant kernel preemption and controlled concurrent execution in GPU scheduling.

## Table of Contents

- [Introduction](#introduction)
- [Paper](#paper)
- [REEF Example](#reef-example)
- [Project Structure](#project-structure)
- [Hardware Requirement](#hardware-requirement)
- [Installation](#installation)
- [Artifact Evaluation](#artifact-evaluation)


## Introduction

REEF is a real-time GPU-accelerated DNN inference scheduling system. 
REEF divides DNN inference tasks into two priorities: *real-time tasks(RT tasks)* and *best-effort tasks(BE tasks)*.
The scheduling goal of REEF is to minimize the latency of RT task and improve the throughput as much as possible.

REEF achieves such goal by providing two key techniques:

* *Reset-based Preemption:* BE tasks can be preempted in a few microseconds once a RT task arrives. The preemption is achieved by just killing
the running BE kernels and clearing the queued BE kernels, which is bases on the *idempotence* of DNN inference kernel.

* *Dynamic Kernel Padding(DKP):* BE tasks can be co-executed with the RT task by only using the CUs leftover from the RT kernels. This approach can improve the throughput and avoid starvation of BE tasks with minimal latency overhead on RT tasks.

## Paper
If you use REEF in your research, please cite our paper:
```bibtex
@inproceedings {osdi2022reef,
  author = {Mingcong Han and Hanze Zhang and Rong Chen and Haibo Chen},
  title = {Microsecond-scale Preemption for Concurrent {GPU-accelerated} {DNN} Inferences},
  booktitle = {16th USENIX Symposium on Operating Systems Design and Implementation (OSDI 22)},
  year = {2022},
  isbn = {978-1-939133-28-1},
  address = {Carlsbad, CA},
  pages = {539--558},
  url = {https://www.usenix.org/conference/osdi22/presentation/han},
  publisher = {USENIX Association},
  month = jul,
}
```

## REEF Example

After [building REEF](INSTALL.md), the example below can show how REEF works when there are concurrent tasks (one RT and multiple BEs).

First, start a REEF server.
```bash
# in ./build
$ ./reef_server
```

Then, start multiple BE clients. 
```bash
# in ./build
$ for i in {1..4}; do ./reef_client_cont ../resource/resnet152 resnet152 0 0 & done
```
You can see 4 BE clients are submitting BE tasks concurrently, the client will echo the inference latency of each task, e.g.:
```
client 3 inference latency: 16.567 ms
client 2 inference latency: 29.347 ms
client 1 inference latency: 32.506 ms
client 0 inference latency: 24.848 ms
```

Then, start a RT client, which submitting requests without pause.
```bash
# in ./build
$ ./reef_client_cont ../resource/resnet152 resnet152 1 0
```

You can see the RT client has the lowest inference latency.
```
...
client 4 inference latency: 12.743 ms
client 4 inference latency: 12.608 ms
client 4 inference latency: 12.944 ms
client 4 inference latency: 12.637 ms
```

While, the BE task can still execute concurrently with RT task without affecting the performance of RT tasks.
```
...
client 2 inference latency: 48.183 ms
client 1 inference latency: 68.599 ms
client 0 inference latency: 34.857 ms
client 3 inference latency: 43.565 ms
```




## Project Structure
```
> tree .
├── cmake                     
├── resource                      # DNN model resources for the evaluations
│   ├── resnet                    # DNN model for ResNet
│   │   ├── resnet.json           # The schedule graph (meta data) of the DNN model
│   │   ├── resnet.cu             # The raw GPU device code (GPU kernels) for the DNN model
│   │   ├── resnet.trans.cu       # The transformed GPU device code which supports dynamic kernel padding
│   │   ├── resnet.be.cu          # The transformed GPU devide code which supports reset-based preemption
│   │   ├── resnet.profile.json   # The profile of the kernel execution time
│   ├── densenet
│   ├── inception
├── script                        # Utility scripts
└── src                           # source code
│   ├── example                   # REEF examples
│   └── reef                      # REEF source code
└── env.sh                        # Environment variables
│
```

## Hardware Requirement

Currently, REEF only supports **AMD Radeon Instinct MI50 GPU**.


## Installation

see [INSTALL](INSTALL.md).


## Artifact Evaluation

For OSDI'22 atrifact evaluation, see [reef-artifacts](https://github.com/SJTU-IPADS/reef-artifacts).
