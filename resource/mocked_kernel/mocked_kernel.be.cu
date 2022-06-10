#include <hip/hip_runtime.h>

#define NUM_BLOCKS 64
#define NUM_TREHAD_PER_BLOCK 128
#define BLOCKDIM_X 4
#define BLOCKDIM_Y 8
#define BLOCKDIM_Z 4

__device__ void multiply_device(float* __restrict__ a, float* __restrict__ b, float* __restrict__ temp){
    __shared__ float buffer[1024 * 32 / 4];
    int blockOffset = blockIdx.x;
    int blockSize = NUM_TREHAD_PER_BLOCK;
    int threadOffset = threadIdx.x + threadIdx.y * BLOCKDIM_X + threadIdx.z * BLOCKDIM_X * BLOCKDIM_Y;
    int arrayOffset = blockOffset * blockSize + threadOffset;

    temp[arrayOffset] = a[threadOffset] * b[arrayOffset];
}

__device__ void add_device(float* __restrict__ a, float* __restrict__ b, float* __restrict__ temp){
    __shared__ float buffer[1024 * 32 / 4];
    int blockOffset = blockIdx.x;
    int blockSize = NUM_TREHAD_PER_BLOCK;
    int threadOffset = threadIdx.x + threadIdx.y * BLOCKDIM_X + threadIdx.z * BLOCKDIM_X * BLOCKDIM_Y;
    int arrayOffset = blockOffset * blockSize + threadOffset;

    temp[arrayOffset] = a[threadOffset] + b[arrayOffset];
}

extern "C" __global__ void multiply(int* preempted, int* task_slot, float* __restrict__ a, float* __restrict__ b, float* __restrict__ temp) {
    if (*preempted) return;
    multiply_device(a, b, temp);
    if (threadIdx.x + threadIdx.y + threadIdx.z == 0)
        atomicAdd(task_slot, 1);
}        

extern "C" __global__ void add(int* preempted, int* task_slot, float* __restrict__ a, float* __restrict__ b, float* __restrict__ temp) {
    if (*preempted) return;
    add_device(a, b, temp);
    if (threadIdx.x + threadIdx.y + threadIdx.z == 0)
        atomicAdd(task_slot, 1);
}        
