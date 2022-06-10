#include <hip/hip_runtime.h>

#define NUM_BLOCKS 64
#define NUM_TREHAD_PER_BLOCK 128
#define BLOCKDIM_X 4
#define BLOCKDIM_Y 8
#define BLOCKDIM_Z 4

#define CU_NUM 60

__device__ __forceinline__ bool is_first_thread() {
  return threadIdx.x == 0;
}

__device__ __forceinline__ unsigned int get_cu_id() {
  return blockIdx.x % CU_NUM;
}

__device__ __forceinline__ dim3 get_3d_idx(int idx, dim3 dim) {
  dim3 result;
  result.x = idx % dim.x;
  result.y = idx / dim.x % dim.y;
  result.z = idx / (dim.x * dim.y);
  return result;
}

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

extern "C" __global__  __attribute__((amdgpu_num_vgpr(25))) __attribute__((amdgpu_num_sgpr(30))) void multiply_device_wrapper(float* __restrict__ a, float* __restrict__ b, float* __restrict__ temp) {
    // Force the compiler to use all the index
    if (threadIdx.x + threadIdx.y * 4 + threadIdx.z * 8 * 4 >= 4 * 8 * 4) return;
    // if (blockIdx.x + blockIdx.y * 64 + blockIdx.z * 1 * 64 >= 64 * 1 * 1) return;
    multiply_device((float* __restrict__)a,(float* __restrict__)b,(float* __restrict__)temp);
    asm volatile(";; end_flag"); // jump back to the caller
}

extern "C" __global__  __attribute__((amdgpu_num_vgpr(25))) __attribute__((amdgpu_num_sgpr(30))) void add_device_wrapper(float* __restrict__ a, float* __restrict__ b, float* __restrict__ temp) {
    // Force the compiler to use all the index
    if (threadIdx.x + threadIdx.y * 4 + threadIdx.z * 8 * 4 >= 4 * 8 * 4) return;
    // if (blockIdx.x + blockIdx.y * 64 + blockIdx.z * 1 * 64 >= 64 * 1 * 1) return;
    add_device((float* __restrict__)a,(float* __restrict__)b,(float* __restrict__)temp);
    asm volatile(";; end_flag"); // jump back to the caller
}

extern "C" __global__  void multiply(
    void* func_l, int layers_l, int task_num_l, int task_offset_l, float** param_l,
    void* func_r, int layers_r, int task_num_r, int task_offset_r, float** param_r,
    int cu_partition) {
    asm volatile(";; caller_flag");
    return;
}

extern "C" __global__  void add(
    void* func_l, int layers_l, int task_num_l, int task_offset_l, float** param_l,
    void* func_r, int layers_r, int task_num_r, int task_offset_r, float** param_r,
    int cu_partition) {
    asm volatile(";; caller_flag");
    return;
}

extern "C" __device__ __noinline__ dim3 get_3d_idx_64_1_1(int idx) {
  dim3 dim(64, 1, 1);
  dim3 result;
  result.x = idx % dim.x;
  result.y = idx / dim.x % dim.y;
  result.z = idx / (dim.x * dim.y);
  return result;
}

extern "C" __device__ __noinline__ dim3 get_3d_idx_4_8_4(int idx) {
  dim3 dim(4, 8, 4);
  dim3 result;
  result.x = idx % dim.x;
  result.y = idx / dim.x % dim.y;
  result.z = idx / (dim.x * dim.y);
  return result;
}

__global__ void get_3d_idx_caller(int* buf) {
    dim3 task_idx;

    task_idx = get_3d_idx_64_1_1(threadIdx.x);
    buf[task_idx.x] = task_idx.x;
    buf[task_idx.y] = task_idx.y;
    buf[task_idx.z] = task_idx.z;

    task_idx = get_3d_idx_4_8_4(threadIdx.x);
    buf[task_idx.x] = task_idx.x;
    buf[task_idx.y] = task_idx.y;
    buf[task_idx.z] = task_idx.z;

}

#define CALL_FRAMEWORK(idx) \
extern "C" __global__ void call_framework_##idx(\
  void* func_l, int layers_l, int task_num_l, int task_offset_l, float** param_l,\
  void* func_r, int layers_r, int task_num_r, int task_offset_r, float** param_r,\
  int cu_partition) \
{\
  asm volatile(\
    "  s_load_dwordx2 s[14:15], s[4:5], 0x0\n"\
    "  s_waitcnt lgkmcnt(0)\n"\
    "  s_setpc_b64 s[14:15]\n"\
    "  s_endpgm\n"\
  );\
}

CALL_FRAMEWORK(1)
CALL_FRAMEWORK(2)
CALL_FRAMEWORK(3)
CALL_FRAMEWORK(4)
CALL_FRAMEWORK(5)
CALL_FRAMEWORK(6)
CALL_FRAMEWORK(7)
CALL_FRAMEWORK(8)
CALL_FRAMEWORK(9)
CALL_FRAMEWORK(10)

#define MERGE_FRAMEWORK(idx) \
extern "C" __global__ void merge_framework_##idx(\
  void* func_l, int layers_l, int task_num_l, int task_offset_l, float** param_l,\
  void* func_r, int layers_r, int task_num_r, int task_offset_r, float** param_r,\
  int cu_partition) \
{\
  asm volatile(\
    "  s_load_dword s10, s[4:5], 0x40\n"\
    "  s_load_dwordx2 s[12:13], s[4:5], 0x0\n"\
    "  s_load_dwordx2 s[14:15], s[4:5], 0x20\n"\
    "  s_mul_hi_u32 s11, s6, 0x88888889\n"\
    "  s_lshr_b32 s11, s11, 5\n"\
    "  s_mul_i32 s11, s11, 60\n"\
    "  s_sub_i32 s11, s6, s11\n"\
    "  s_waitcnt lgkmcnt(0)\n"\
    "  s_cmp_ge_u32 s11, s10\n"\
    "  s_mov_b64 s[10:11], -1\n"\
    "  s_cbranch_scc1 MyBB"#idx"_3\n"\
    "; %bb.1:                                ; %Flow\n"\
    "  s_andn2_b64 vcc, exec, s[10:11]\n"\
    "  s_cbranch_vccz MyBB"#idx"_4\n"\
    "  s_endpgm\n"\
    "MyBB"#idx"_3:\n"\
    "  s_setpc_b64 s[14:15]\n"\
    "  s_endpgm\n"\
    "MyBB"#idx"_4:\n"\
    "  s_setpc_b64 s[12:13]\n"\
    "  s_endpgm\n"\
  );\
}
MERGE_FRAMEWORK(1)
MERGE_FRAMEWORK(2)
MERGE_FRAMEWORK(3)
MERGE_FRAMEWORK(4)
MERGE_FRAMEWORK(5)
MERGE_FRAMEWORK(6)
MERGE_FRAMEWORK(7)
MERGE_FRAMEWORK(8)
MERGE_FRAMEWORK(9)
MERGE_FRAMEWORK(10)
MERGE_FRAMEWORK(nostack_1)
MERGE_FRAMEWORK(nostack_2)
MERGE_FRAMEWORK(nostack_3)
MERGE_FRAMEWORK(nostack_4)
MERGE_FRAMEWORK(nostack_5)
MERGE_FRAMEWORK(nostack_6)
MERGE_FRAMEWORK(nostack_7)
MERGE_FRAMEWORK(nostack_8)
MERGE_FRAMEWORK(nostack_9)
MERGE_FRAMEWORK(nostack_10)
