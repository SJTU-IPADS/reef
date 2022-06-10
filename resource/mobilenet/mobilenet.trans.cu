#include <hip/hip_runtime.h>

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

__device__ void fused_nn_conv2d_add_nn_relu_12_kernel0_device(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2){
  float compute[8];
  __shared__ float pad_temp_shared[512];
  __shared__ float placeholder_shared[1024];
  compute[(0)] = 0.000000e+00f;
  compute[(2)] = 0.000000e+00f;
  compute[(4)] = 0.000000e+00f;
  compute[(6)] = 0.000000e+00f;
  compute[(1)] = 0.000000e+00f;
  compute[(3)] = 0.000000e+00f;
  compute[(5)] = 0.000000e+00f;
  compute[(7)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 8; ++rc_outer) {
    __syncthreads();
    pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)))] = placeholder[(((((((rc_outer * 50176) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)))];
    pad_temp_shared[(((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1))] = placeholder[((((((((rc_outer * 50176) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 1))];
    placeholder_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))] = placeholder1[((((((((int)blockIdx.z) * 8192) + (((int)threadIdx.z) * 512)) + (((int)threadIdx.y) * 128)) + (rc_outer * 16)) + (((int)threadIdx.x) * 4)))];
    placeholder_shared[(((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 1))] = placeholder1[(((((((((int)blockIdx.z) * 8192) + (((int)threadIdx.z) * 512)) + (((int)threadIdx.y) * 128)) + (rc_outer * 16)) + (((int)threadIdx.x) * 4)) + 1))];
    placeholder_shared[(((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 2))] = placeholder1[(((((((((int)blockIdx.z) * 8192) + (((int)threadIdx.z) * 512)) + (((int)threadIdx.y) * 128)) + (rc_outer * 16)) + (((int)threadIdx.x) * 4)) + 2))];
    placeholder_shared[(((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 3))] = placeholder1[(((((((((int)blockIdx.z) * 8192) + (((int)threadIdx.z) * 512)) + (((int)threadIdx.y) * 128)) + (rc_outer * 16)) + (((int)threadIdx.x) * 4)) + 3))];
    __syncthreads();
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)))], placeholder_shared[((((int)threadIdx.z) * 16))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)))], placeholder_shared[(((((int)threadIdx.z) * 16) + 256))], compute[(2)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)))], placeholder_shared[(((((int)threadIdx.z) * 16) + 512))], compute[(4)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)))], placeholder_shared[(((((int)threadIdx.z) * 16) + 768))], compute[(6)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 1))], placeholder_shared[((((int)threadIdx.z) * 16))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 1))], placeholder_shared[(((((int)threadIdx.z) * 16) + 256))], compute[(3)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 1))], placeholder_shared[(((((int)threadIdx.z) * 16) + 512))], compute[(5)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 1))], placeholder_shared[(((((int)threadIdx.z) * 16) + 768))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 32))], placeholder_shared[(((((int)threadIdx.z) * 16) + 1))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 32))], placeholder_shared[(((((int)threadIdx.z) * 16) + 257))], compute[(2)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 32))], placeholder_shared[(((((int)threadIdx.z) * 16) + 513))], compute[(4)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 32))], placeholder_shared[(((((int)threadIdx.z) * 16) + 769))], compute[(6)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 33))], placeholder_shared[(((((int)threadIdx.z) * 16) + 1))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 33))], placeholder_shared[(((((int)threadIdx.z) * 16) + 257))], compute[(3)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 33))], placeholder_shared[(((((int)threadIdx.z) * 16) + 513))], compute[(5)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 33))], placeholder_shared[(((((int)threadIdx.z) * 16) + 769))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 64))], placeholder_shared[(((((int)threadIdx.z) * 16) + 2))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 64))], placeholder_shared[(((((int)threadIdx.z) * 16) + 258))], compute[(2)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 64))], placeholder_shared[(((((int)threadIdx.z) * 16) + 514))], compute[(4)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 64))], placeholder_shared[(((((int)threadIdx.z) * 16) + 770))], compute[(6)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 65))], placeholder_shared[(((((int)threadIdx.z) * 16) + 2))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 65))], placeholder_shared[(((((int)threadIdx.z) * 16) + 258))], compute[(3)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 65))], placeholder_shared[(((((int)threadIdx.z) * 16) + 514))], compute[(5)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 65))], placeholder_shared[(((((int)threadIdx.z) * 16) + 770))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 96))], placeholder_shared[(((((int)threadIdx.z) * 16) + 3))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 96))], placeholder_shared[(((((int)threadIdx.z) * 16) + 259))], compute[(2)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 96))], placeholder_shared[(((((int)threadIdx.z) * 16) + 515))], compute[(4)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 96))], placeholder_shared[(((((int)threadIdx.z) * 16) + 771))], compute[(6)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 97))], placeholder_shared[(((((int)threadIdx.z) * 16) + 3))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 97))], placeholder_shared[(((((int)threadIdx.z) * 16) + 259))], compute[(3)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 97))], placeholder_shared[(((((int)threadIdx.z) * 16) + 515))], compute[(5)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 97))], placeholder_shared[(((((int)threadIdx.z) * 16) + 771))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 128))], placeholder_shared[(((((int)threadIdx.z) * 16) + 4))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 128))], placeholder_shared[(((((int)threadIdx.z) * 16) + 260))], compute[(2)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 128))], placeholder_shared[(((((int)threadIdx.z) * 16) + 516))], compute[(4)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 128))], placeholder_shared[(((((int)threadIdx.z) * 16) + 772))], compute[(6)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 129))], placeholder_shared[(((((int)threadIdx.z) * 16) + 4))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 129))], placeholder_shared[(((((int)threadIdx.z) * 16) + 260))], compute[(3)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 129))], placeholder_shared[(((((int)threadIdx.z) * 16) + 516))], compute[(5)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 129))], placeholder_shared[(((((int)threadIdx.z) * 16) + 772))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 160))], placeholder_shared[(((((int)threadIdx.z) * 16) + 5))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 160))], placeholder_shared[(((((int)threadIdx.z) * 16) + 261))], compute[(2)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 160))], placeholder_shared[(((((int)threadIdx.z) * 16) + 517))], compute[(4)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 160))], placeholder_shared[(((((int)threadIdx.z) * 16) + 773))], compute[(6)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 161))], placeholder_shared[(((((int)threadIdx.z) * 16) + 5))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 161))], placeholder_shared[(((((int)threadIdx.z) * 16) + 261))], compute[(3)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 161))], placeholder_shared[(((((int)threadIdx.z) * 16) + 517))], compute[(5)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 161))], placeholder_shared[(((((int)threadIdx.z) * 16) + 773))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 192))], placeholder_shared[(((((int)threadIdx.z) * 16) + 6))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 192))], placeholder_shared[(((((int)threadIdx.z) * 16) + 262))], compute[(2)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 192))], placeholder_shared[(((((int)threadIdx.z) * 16) + 518))], compute[(4)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 192))], placeholder_shared[(((((int)threadIdx.z) * 16) + 774))], compute[(6)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 193))], placeholder_shared[(((((int)threadIdx.z) * 16) + 6))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 193))], placeholder_shared[(((((int)threadIdx.z) * 16) + 262))], compute[(3)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 193))], placeholder_shared[(((((int)threadIdx.z) * 16) + 518))], compute[(5)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 193))], placeholder_shared[(((((int)threadIdx.z) * 16) + 774))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 224))], placeholder_shared[(((((int)threadIdx.z) * 16) + 7))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 224))], placeholder_shared[(((((int)threadIdx.z) * 16) + 263))], compute[(2)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 224))], placeholder_shared[(((((int)threadIdx.z) * 16) + 519))], compute[(4)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 224))], placeholder_shared[(((((int)threadIdx.z) * 16) + 775))], compute[(6)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 225))], placeholder_shared[(((((int)threadIdx.z) * 16) + 7))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 225))], placeholder_shared[(((((int)threadIdx.z) * 16) + 263))], compute[(3)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 225))], placeholder_shared[(((((int)threadIdx.z) * 16) + 519))], compute[(5)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 225))], placeholder_shared[(((((int)threadIdx.z) * 16) + 775))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 256))], placeholder_shared[(((((int)threadIdx.z) * 16) + 8))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 256))], placeholder_shared[(((((int)threadIdx.z) * 16) + 264))], compute[(2)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 256))], placeholder_shared[(((((int)threadIdx.z) * 16) + 520))], compute[(4)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 256))], placeholder_shared[(((((int)threadIdx.z) * 16) + 776))], compute[(6)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 257))], placeholder_shared[(((((int)threadIdx.z) * 16) + 8))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 257))], placeholder_shared[(((((int)threadIdx.z) * 16) + 264))], compute[(3)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 257))], placeholder_shared[(((((int)threadIdx.z) * 16) + 520))], compute[(5)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 257))], placeholder_shared[(((((int)threadIdx.z) * 16) + 776))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 288))], placeholder_shared[(((((int)threadIdx.z) * 16) + 9))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 288))], placeholder_shared[(((((int)threadIdx.z) * 16) + 265))], compute[(2)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 288))], placeholder_shared[(((((int)threadIdx.z) * 16) + 521))], compute[(4)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 288))], placeholder_shared[(((((int)threadIdx.z) * 16) + 777))], compute[(6)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 289))], placeholder_shared[(((((int)threadIdx.z) * 16) + 9))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 289))], placeholder_shared[(((((int)threadIdx.z) * 16) + 265))], compute[(3)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 289))], placeholder_shared[(((((int)threadIdx.z) * 16) + 521))], compute[(5)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 289))], placeholder_shared[(((((int)threadIdx.z) * 16) + 777))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 320))], placeholder_shared[(((((int)threadIdx.z) * 16) + 10))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 320))], placeholder_shared[(((((int)threadIdx.z) * 16) + 266))], compute[(2)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 320))], placeholder_shared[(((((int)threadIdx.z) * 16) + 522))], compute[(4)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 320))], placeholder_shared[(((((int)threadIdx.z) * 16) + 778))], compute[(6)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 321))], placeholder_shared[(((((int)threadIdx.z) * 16) + 10))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 321))], placeholder_shared[(((((int)threadIdx.z) * 16) + 266))], compute[(3)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 321))], placeholder_shared[(((((int)threadIdx.z) * 16) + 522))], compute[(5)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 321))], placeholder_shared[(((((int)threadIdx.z) * 16) + 778))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 352))], placeholder_shared[(((((int)threadIdx.z) * 16) + 11))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 352))], placeholder_shared[(((((int)threadIdx.z) * 16) + 267))], compute[(2)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 352))], placeholder_shared[(((((int)threadIdx.z) * 16) + 523))], compute[(4)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 352))], placeholder_shared[(((((int)threadIdx.z) * 16) + 779))], compute[(6)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 353))], placeholder_shared[(((((int)threadIdx.z) * 16) + 11))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 353))], placeholder_shared[(((((int)threadIdx.z) * 16) + 267))], compute[(3)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 353))], placeholder_shared[(((((int)threadIdx.z) * 16) + 523))], compute[(5)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 353))], placeholder_shared[(((((int)threadIdx.z) * 16) + 779))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 384))], placeholder_shared[(((((int)threadIdx.z) * 16) + 12))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 384))], placeholder_shared[(((((int)threadIdx.z) * 16) + 268))], compute[(2)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 384))], placeholder_shared[(((((int)threadIdx.z) * 16) + 524))], compute[(4)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 384))], placeholder_shared[(((((int)threadIdx.z) * 16) + 780))], compute[(6)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 385))], placeholder_shared[(((((int)threadIdx.z) * 16) + 12))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 385))], placeholder_shared[(((((int)threadIdx.z) * 16) + 268))], compute[(3)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 385))], placeholder_shared[(((((int)threadIdx.z) * 16) + 524))], compute[(5)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 385))], placeholder_shared[(((((int)threadIdx.z) * 16) + 780))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 416))], placeholder_shared[(((((int)threadIdx.z) * 16) + 13))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 416))], placeholder_shared[(((((int)threadIdx.z) * 16) + 269))], compute[(2)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 416))], placeholder_shared[(((((int)threadIdx.z) * 16) + 525))], compute[(4)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 416))], placeholder_shared[(((((int)threadIdx.z) * 16) + 781))], compute[(6)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 417))], placeholder_shared[(((((int)threadIdx.z) * 16) + 13))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 417))], placeholder_shared[(((((int)threadIdx.z) * 16) + 269))], compute[(3)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 417))], placeholder_shared[(((((int)threadIdx.z) * 16) + 525))], compute[(5)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 417))], placeholder_shared[(((((int)threadIdx.z) * 16) + 781))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 448))], placeholder_shared[(((((int)threadIdx.z) * 16) + 14))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 448))], placeholder_shared[(((((int)threadIdx.z) * 16) + 270))], compute[(2)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 448))], placeholder_shared[(((((int)threadIdx.z) * 16) + 526))], compute[(4)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 448))], placeholder_shared[(((((int)threadIdx.z) * 16) + 782))], compute[(6)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 449))], placeholder_shared[(((((int)threadIdx.z) * 16) + 14))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 449))], placeholder_shared[(((((int)threadIdx.z) * 16) + 270))], compute[(3)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 449))], placeholder_shared[(((((int)threadIdx.z) * 16) + 526))], compute[(5)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 449))], placeholder_shared[(((((int)threadIdx.z) * 16) + 782))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 480))], placeholder_shared[(((((int)threadIdx.z) * 16) + 15))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 480))], placeholder_shared[(((((int)threadIdx.z) * 16) + 271))], compute[(2)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 480))], placeholder_shared[(((((int)threadIdx.z) * 16) + 527))], compute[(4)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 480))], placeholder_shared[(((((int)threadIdx.z) * 16) + 783))], compute[(6)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 481))], placeholder_shared[(((((int)threadIdx.z) * 16) + 15))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 481))], placeholder_shared[(((((int)threadIdx.z) * 16) + 271))], compute[(3)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 481))], placeholder_shared[(((((int)threadIdx.z) * 16) + 527))], compute[(5)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 481))], placeholder_shared[(((((int)threadIdx.z) * 16) + 783))], compute[(7)]);
  }
  T_relu[(((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)))] = max((compute[(0)] + placeholder2[(((((int)blockIdx.z) * 64) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 50176))] = max((compute[(2)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 16))]), 0.000000e+00f);
  T_relu[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 100352))] = max((compute[(4)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 32))]), 0.000000e+00f);
  T_relu[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 150528))] = max((compute[(6)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 48))]), 0.000000e+00f);
  T_relu[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 1))] = max((compute[(1)] + placeholder2[(((((int)blockIdx.z) * 64) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 50177))] = max((compute[(3)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 16))]), 0.000000e+00f);
  T_relu[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 100353))] = max((compute[(5)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 32))]), 0.000000e+00f);
  T_relu[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 150529))] = max((compute[(7)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 48))]), 0.000000e+00f);
}

__device__ void fused_nn_conv2d_add_nn_relu_7_kernel0_device(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2){
  __shared__ float PaddedInput_shared[841];
  __shared__ float placeholder_shared[9];
  float PaddedInput_shared_local[9];
  float placeholder_shared_local[9];
  float DepthwiseConv2d[1];
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) < 841) {
      if (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 14) + ((int)threadIdx.y)) < 61) {
        PaddedInput_shared[((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)))] = (((29 <= (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x))) && (1 <= ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) % 29))) ? placeholder[(((((((int)blockIdx.z) * 784) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) / 29) * 28)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) % 29)) - 29))] : 0.000000e+00f);
      }
    }
  }
  if (((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) < 9) {
    if (((int)threadIdx.y) < 1) {
      placeholder_shared[(((((int)threadIdx.y) * 14) + ((int)threadIdx.x)))] = placeholder1[((((((int)threadIdx.y) * 14) + (((int)blockIdx.z) * 9)) + ((int)threadIdx.x)))];
    }
  }
  __syncthreads();
  for (int ax2 = 0; ax2 < 3; ++ax2) {
    for (int ax3 = 0; ax3 < 3; ++ax3) {
      PaddedInput_shared_local[(((ax2 * 3) + ax3))] = PaddedInput_shared[(((((((int)threadIdx.y) * 58) + (ax2 * 29)) + (((int)threadIdx.x) * 2)) + ax3))];
    }
  }
  for (int ax21 = 0; ax21 < 3; ++ax21) {
    for (int ax31 = 0; ax31 < 3; ++ax31) {
      placeholder_shared_local[(((ax21 * 3) + ax31))] = placeholder_shared[(((ax21 * 3) + ax31))];
    }
  }
  DepthwiseConv2d[(0)] = 0.000000e+00f;
  for (int di = 0; di < 3; ++di) {
    for (int dj = 0; dj < 3; ++dj) {
      DepthwiseConv2d[(0)] = __ocml_fma_f32(PaddedInput_shared_local[(((di * 3) + dj))], placeholder_shared_local[(((di * 3) + dj))], DepthwiseConv2d[(0)]);
    }
  }
  T_relu[((((((int)blockIdx.z) * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)))] = max((DepthwiseConv2d[(0)] + placeholder2[(((int)blockIdx.z))]), 0.000000e+00f);
}

__device__ void fused_nn_conv2d_add_nn_relu_6_kernel0_device(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2){
  float compute[2];
  __shared__ float pad_temp_shared[448];
  __shared__ float placeholder_shared[512];
  compute[(0)] = 0.000000e+00f;
  compute[(1)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 16; ++rc_outer) {
    __syncthreads();
    pad_temp_shared[((((((int)threadIdx.z) * 14) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)))] = placeholder[(((((((rc_outer * 3136) + ((((int)threadIdx.z) >> 1) * 196)) + (((int)blockIdx.y) * 28)) + ((((int)threadIdx.z) & 1) * 14)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)))];
    if (((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) >> 4) + ((int)threadIdx.z)) < 32) {
      if ((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) < 512) {
        if (((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) < 16) {
          if (((int)threadIdx.x) < 4) {
            placeholder_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)))] = placeholder1[((((((((int)blockIdx.z) * 8192) + (((int)threadIdx.z) * 256)) + (rc_outer * 16)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)))];
          }
        }
      }
    }
    if ((((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 1) >> 4) + ((int)threadIdx.z)) < 32) {
      if ((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) < 511) {
        if (((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) < 15) {
          if (((int)threadIdx.x) < 4) {
            placeholder_shared[(((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1))] = placeholder1[(((((((((int)blockIdx.z) * 8192) + (((int)threadIdx.z) * 256)) + (rc_outer * 16)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1))];
          }
        }
      }
    }
    __syncthreads();
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)))], placeholder_shared[((((int)threadIdx.z) * 16))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 1))], placeholder_shared[((((int)threadIdx.z) * 16))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 28))], placeholder_shared[(((((int)threadIdx.z) * 16) + 1))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 29))], placeholder_shared[(((((int)threadIdx.z) * 16) + 1))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 56))], placeholder_shared[(((((int)threadIdx.z) * 16) + 2))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 57))], placeholder_shared[(((((int)threadIdx.z) * 16) + 2))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 84))], placeholder_shared[(((((int)threadIdx.z) * 16) + 3))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 85))], placeholder_shared[(((((int)threadIdx.z) * 16) + 3))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 112))], placeholder_shared[(((((int)threadIdx.z) * 16) + 4))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 113))], placeholder_shared[(((((int)threadIdx.z) * 16) + 4))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 140))], placeholder_shared[(((((int)threadIdx.z) * 16) + 5))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 141))], placeholder_shared[(((((int)threadIdx.z) * 16) + 5))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 168))], placeholder_shared[(((((int)threadIdx.z) * 16) + 6))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 169))], placeholder_shared[(((((int)threadIdx.z) * 16) + 6))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 196))], placeholder_shared[(((((int)threadIdx.z) * 16) + 7))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 197))], placeholder_shared[(((((int)threadIdx.z) * 16) + 7))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 224))], placeholder_shared[(((((int)threadIdx.z) * 16) + 8))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 225))], placeholder_shared[(((((int)threadIdx.z) * 16) + 8))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 252))], placeholder_shared[(((((int)threadIdx.z) * 16) + 9))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 253))], placeholder_shared[(((((int)threadIdx.z) * 16) + 9))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 280))], placeholder_shared[(((((int)threadIdx.z) * 16) + 10))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 281))], placeholder_shared[(((((int)threadIdx.z) * 16) + 10))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 308))], placeholder_shared[(((((int)threadIdx.z) * 16) + 11))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 309))], placeholder_shared[(((((int)threadIdx.z) * 16) + 11))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 336))], placeholder_shared[(((((int)threadIdx.z) * 16) + 12))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 337))], placeholder_shared[(((((int)threadIdx.z) * 16) + 12))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 364))], placeholder_shared[(((((int)threadIdx.z) * 16) + 13))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 365))], placeholder_shared[(((((int)threadIdx.z) * 16) + 13))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 392))], placeholder_shared[(((((int)threadIdx.z) * 16) + 14))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 393))], placeholder_shared[(((((int)threadIdx.z) * 16) + 14))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 420))], placeholder_shared[(((((int)threadIdx.z) * 16) + 15))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 421))], placeholder_shared[(((((int)threadIdx.z) * 16) + 15))], compute[(1)]);
  }
  T_relu[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 28)) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)))] = max((compute[(0)] + placeholder2[(((((int)blockIdx.z) * 32) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[(((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 28)) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)) + 1))] = max((compute[(1)] + placeholder2[(((((int)blockIdx.z) * 32) + ((int)threadIdx.z)))]), 0.000000e+00f);
}

__device__ void fused_nn_conv2d_add_nn_relu_14_kernel0_device(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2){
  float compute[8];
  __shared__ float pad_temp_shared[512];
  __shared__ float placeholder_shared[1024];
  compute[(0)] = 0.000000e+00f;
  compute[(4)] = 0.000000e+00f;
  compute[(2)] = 0.000000e+00f;
  compute[(6)] = 0.000000e+00f;
  compute[(1)] = 0.000000e+00f;
  compute[(5)] = 0.000000e+00f;
  compute[(3)] = 0.000000e+00f;
  compute[(7)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 4; ++rc_outer) {
    __syncthreads();
    pad_temp_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)))] = placeholder[(((((((rc_outer * 50176) + ((((((int)threadIdx.z) * 2) + ((int)threadIdx.y)) >> 2) * 3136)) + (((int)blockIdx.y) * 224)) + ((((((int)threadIdx.z) * 2) + ((int)threadIdx.y)) & 3) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)))];
    pad_temp_shared[(((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1))] = placeholder[((((((((rc_outer * 50176) + ((((((int)threadIdx.z) * 2) + ((int)threadIdx.y)) >> 2) * 3136)) + (((int)blockIdx.y) * 224)) + ((((((int)threadIdx.z) * 2) + ((int)threadIdx.y)) & 3) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 1))];
    placeholder_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))] = placeholder1[((((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 128)) + (((int)threadIdx.y) * 64)) + (rc_outer * 16)) + (((int)threadIdx.x) * 4)))];
    placeholder_shared[(((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 1))] = placeholder1[(((((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 128)) + (((int)threadIdx.y) * 64)) + (rc_outer * 16)) + (((int)threadIdx.x) * 4)) + 1))];
    placeholder_shared[(((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 2))] = placeholder1[(((((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 128)) + (((int)threadIdx.y) * 64)) + (rc_outer * 16)) + (((int)threadIdx.x) * 4)) + 2))];
    placeholder_shared[(((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 3))] = placeholder1[(((((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 128)) + (((int)threadIdx.y) * 64)) + (rc_outer * 16)) + (((int)threadIdx.x) * 4)) + 3))];
    __syncthreads();
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)))], placeholder_shared[((((int)threadIdx.z) * 16))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)))], placeholder_shared[(((((int)threadIdx.z) * 16) + 512))], compute[(4)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 16))], placeholder_shared[((((int)threadIdx.z) * 16))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 16))], placeholder_shared[(((((int)threadIdx.z) * 16) + 512))], compute[(6)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 1))], placeholder_shared[((((int)threadIdx.z) * 16))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 1))], placeholder_shared[(((((int)threadIdx.z) * 16) + 512))], compute[(5)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 17))], placeholder_shared[((((int)threadIdx.z) * 16))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 17))], placeholder_shared[(((((int)threadIdx.z) * 16) + 512))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 32))], placeholder_shared[(((((int)threadIdx.z) * 16) + 1))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 32))], placeholder_shared[(((((int)threadIdx.z) * 16) + 513))], compute[(4)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 48))], placeholder_shared[(((((int)threadIdx.z) * 16) + 1))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 48))], placeholder_shared[(((((int)threadIdx.z) * 16) + 513))], compute[(6)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 33))], placeholder_shared[(((((int)threadIdx.z) * 16) + 1))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 33))], placeholder_shared[(((((int)threadIdx.z) * 16) + 513))], compute[(5)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 49))], placeholder_shared[(((((int)threadIdx.z) * 16) + 1))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 49))], placeholder_shared[(((((int)threadIdx.z) * 16) + 513))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 64))], placeholder_shared[(((((int)threadIdx.z) * 16) + 2))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 64))], placeholder_shared[(((((int)threadIdx.z) * 16) + 514))], compute[(4)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 80))], placeholder_shared[(((((int)threadIdx.z) * 16) + 2))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 80))], placeholder_shared[(((((int)threadIdx.z) * 16) + 514))], compute[(6)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 65))], placeholder_shared[(((((int)threadIdx.z) * 16) + 2))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 65))], placeholder_shared[(((((int)threadIdx.z) * 16) + 514))], compute[(5)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 81))], placeholder_shared[(((((int)threadIdx.z) * 16) + 2))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 81))], placeholder_shared[(((((int)threadIdx.z) * 16) + 514))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 96))], placeholder_shared[(((((int)threadIdx.z) * 16) + 3))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 96))], placeholder_shared[(((((int)threadIdx.z) * 16) + 515))], compute[(4)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 112))], placeholder_shared[(((((int)threadIdx.z) * 16) + 3))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 112))], placeholder_shared[(((((int)threadIdx.z) * 16) + 515))], compute[(6)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 97))], placeholder_shared[(((((int)threadIdx.z) * 16) + 3))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 97))], placeholder_shared[(((((int)threadIdx.z) * 16) + 515))], compute[(5)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 113))], placeholder_shared[(((((int)threadIdx.z) * 16) + 3))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 113))], placeholder_shared[(((((int)threadIdx.z) * 16) + 515))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 128))], placeholder_shared[(((((int)threadIdx.z) * 16) + 4))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 128))], placeholder_shared[(((((int)threadIdx.z) * 16) + 516))], compute[(4)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 144))], placeholder_shared[(((((int)threadIdx.z) * 16) + 4))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 144))], placeholder_shared[(((((int)threadIdx.z) * 16) + 516))], compute[(6)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 129))], placeholder_shared[(((((int)threadIdx.z) * 16) + 4))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 129))], placeholder_shared[(((((int)threadIdx.z) * 16) + 516))], compute[(5)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 145))], placeholder_shared[(((((int)threadIdx.z) * 16) + 4))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 145))], placeholder_shared[(((((int)threadIdx.z) * 16) + 516))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 160))], placeholder_shared[(((((int)threadIdx.z) * 16) + 5))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 160))], placeholder_shared[(((((int)threadIdx.z) * 16) + 517))], compute[(4)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 176))], placeholder_shared[(((((int)threadIdx.z) * 16) + 5))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 176))], placeholder_shared[(((((int)threadIdx.z) * 16) + 517))], compute[(6)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 161))], placeholder_shared[(((((int)threadIdx.z) * 16) + 5))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 161))], placeholder_shared[(((((int)threadIdx.z) * 16) + 517))], compute[(5)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 177))], placeholder_shared[(((((int)threadIdx.z) * 16) + 5))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 177))], placeholder_shared[(((((int)threadIdx.z) * 16) + 517))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 192))], placeholder_shared[(((((int)threadIdx.z) * 16) + 6))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 192))], placeholder_shared[(((((int)threadIdx.z) * 16) + 518))], compute[(4)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 208))], placeholder_shared[(((((int)threadIdx.z) * 16) + 6))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 208))], placeholder_shared[(((((int)threadIdx.z) * 16) + 518))], compute[(6)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 193))], placeholder_shared[(((((int)threadIdx.z) * 16) + 6))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 193))], placeholder_shared[(((((int)threadIdx.z) * 16) + 518))], compute[(5)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 209))], placeholder_shared[(((((int)threadIdx.z) * 16) + 6))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 209))], placeholder_shared[(((((int)threadIdx.z) * 16) + 518))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 224))], placeholder_shared[(((((int)threadIdx.z) * 16) + 7))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 224))], placeholder_shared[(((((int)threadIdx.z) * 16) + 519))], compute[(4)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 240))], placeholder_shared[(((((int)threadIdx.z) * 16) + 7))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 240))], placeholder_shared[(((((int)threadIdx.z) * 16) + 519))], compute[(6)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 225))], placeholder_shared[(((((int)threadIdx.z) * 16) + 7))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 225))], placeholder_shared[(((((int)threadIdx.z) * 16) + 519))], compute[(5)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 241))], placeholder_shared[(((((int)threadIdx.z) * 16) + 7))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 241))], placeholder_shared[(((((int)threadIdx.z) * 16) + 519))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 256))], placeholder_shared[(((((int)threadIdx.z) * 16) + 8))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 256))], placeholder_shared[(((((int)threadIdx.z) * 16) + 520))], compute[(4)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 272))], placeholder_shared[(((((int)threadIdx.z) * 16) + 8))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 272))], placeholder_shared[(((((int)threadIdx.z) * 16) + 520))], compute[(6)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 257))], placeholder_shared[(((((int)threadIdx.z) * 16) + 8))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 257))], placeholder_shared[(((((int)threadIdx.z) * 16) + 520))], compute[(5)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 273))], placeholder_shared[(((((int)threadIdx.z) * 16) + 8))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 273))], placeholder_shared[(((((int)threadIdx.z) * 16) + 520))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 288))], placeholder_shared[(((((int)threadIdx.z) * 16) + 9))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 288))], placeholder_shared[(((((int)threadIdx.z) * 16) + 521))], compute[(4)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 304))], placeholder_shared[(((((int)threadIdx.z) * 16) + 9))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 304))], placeholder_shared[(((((int)threadIdx.z) * 16) + 521))], compute[(6)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 289))], placeholder_shared[(((((int)threadIdx.z) * 16) + 9))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 289))], placeholder_shared[(((((int)threadIdx.z) * 16) + 521))], compute[(5)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 305))], placeholder_shared[(((((int)threadIdx.z) * 16) + 9))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 305))], placeholder_shared[(((((int)threadIdx.z) * 16) + 521))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 320))], placeholder_shared[(((((int)threadIdx.z) * 16) + 10))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 320))], placeholder_shared[(((((int)threadIdx.z) * 16) + 522))], compute[(4)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 336))], placeholder_shared[(((((int)threadIdx.z) * 16) + 10))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 336))], placeholder_shared[(((((int)threadIdx.z) * 16) + 522))], compute[(6)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 321))], placeholder_shared[(((((int)threadIdx.z) * 16) + 10))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 321))], placeholder_shared[(((((int)threadIdx.z) * 16) + 522))], compute[(5)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 337))], placeholder_shared[(((((int)threadIdx.z) * 16) + 10))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 337))], placeholder_shared[(((((int)threadIdx.z) * 16) + 522))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 352))], placeholder_shared[(((((int)threadIdx.z) * 16) + 11))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 352))], placeholder_shared[(((((int)threadIdx.z) * 16) + 523))], compute[(4)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 368))], placeholder_shared[(((((int)threadIdx.z) * 16) + 11))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 368))], placeholder_shared[(((((int)threadIdx.z) * 16) + 523))], compute[(6)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 353))], placeholder_shared[(((((int)threadIdx.z) * 16) + 11))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 353))], placeholder_shared[(((((int)threadIdx.z) * 16) + 523))], compute[(5)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 369))], placeholder_shared[(((((int)threadIdx.z) * 16) + 11))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 369))], placeholder_shared[(((((int)threadIdx.z) * 16) + 523))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 384))], placeholder_shared[(((((int)threadIdx.z) * 16) + 12))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 384))], placeholder_shared[(((((int)threadIdx.z) * 16) + 524))], compute[(4)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 400))], placeholder_shared[(((((int)threadIdx.z) * 16) + 12))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 400))], placeholder_shared[(((((int)threadIdx.z) * 16) + 524))], compute[(6)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 385))], placeholder_shared[(((((int)threadIdx.z) * 16) + 12))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 385))], placeholder_shared[(((((int)threadIdx.z) * 16) + 524))], compute[(5)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 401))], placeholder_shared[(((((int)threadIdx.z) * 16) + 12))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 401))], placeholder_shared[(((((int)threadIdx.z) * 16) + 524))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 416))], placeholder_shared[(((((int)threadIdx.z) * 16) + 13))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 416))], placeholder_shared[(((((int)threadIdx.z) * 16) + 525))], compute[(4)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 432))], placeholder_shared[(((((int)threadIdx.z) * 16) + 13))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 432))], placeholder_shared[(((((int)threadIdx.z) * 16) + 525))], compute[(6)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 417))], placeholder_shared[(((((int)threadIdx.z) * 16) + 13))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 417))], placeholder_shared[(((((int)threadIdx.z) * 16) + 525))], compute[(5)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 433))], placeholder_shared[(((((int)threadIdx.z) * 16) + 13))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 433))], placeholder_shared[(((((int)threadIdx.z) * 16) + 525))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 448))], placeholder_shared[(((((int)threadIdx.z) * 16) + 14))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 448))], placeholder_shared[(((((int)threadIdx.z) * 16) + 526))], compute[(4)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 464))], placeholder_shared[(((((int)threadIdx.z) * 16) + 14))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 464))], placeholder_shared[(((((int)threadIdx.z) * 16) + 526))], compute[(6)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 449))], placeholder_shared[(((((int)threadIdx.z) * 16) + 14))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 449))], placeholder_shared[(((((int)threadIdx.z) * 16) + 526))], compute[(5)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 465))], placeholder_shared[(((((int)threadIdx.z) * 16) + 14))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 465))], placeholder_shared[(((((int)threadIdx.z) * 16) + 526))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 480))], placeholder_shared[(((((int)threadIdx.z) * 16) + 15))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 480))], placeholder_shared[(((((int)threadIdx.z) * 16) + 527))], compute[(4)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 496))], placeholder_shared[(((((int)threadIdx.z) * 16) + 15))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 496))], placeholder_shared[(((((int)threadIdx.z) * 16) + 527))], compute[(6)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 481))], placeholder_shared[(((((int)threadIdx.z) * 16) + 15))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 481))], placeholder_shared[(((((int)threadIdx.z) * 16) + 527))], compute[(5)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 497))], placeholder_shared[(((((int)threadIdx.z) * 16) + 15))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 497))], placeholder_shared[(((((int)threadIdx.z) * 16) + 527))], compute[(7)]);
  }
  T_relu[(((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)))] = max((compute[(0)] + placeholder2[(((((int)blockIdx.z) * 64) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 100352))] = max((compute[(4)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 32))]), 0.000000e+00f);
  T_relu[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 112))] = max((compute[(2)] + placeholder2[(((((int)blockIdx.z) * 64) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 100464))] = max((compute[(6)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 32))]), 0.000000e+00f);
  T_relu[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 1))] = max((compute[(1)] + placeholder2[(((((int)blockIdx.z) * 64) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 100353))] = max((compute[(5)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 32))]), 0.000000e+00f);
  T_relu[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 113))] = max((compute[(3)] + placeholder2[(((((int)blockIdx.z) * 64) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 100465))] = max((compute[(7)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 32))]), 0.000000e+00f);
}

__device__ void fused_nn_conv2d_add_nn_relu_18_kernel0_device(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2){
  float compute[7];
  __shared__ float pad_temp_shared[495];
  __shared__ float placeholder_shared[288];
  for (int yy_init = 0; yy_init < 7; ++yy_init) {
    compute[(yy_init)] = 0.000000e+00f;
  }
  for (int rc_outer = 0; rc_outer < 3; ++rc_outer) {
    __syncthreads();
    if (((((int)threadIdx.z) * 16) + ((int)threadIdx.x)) < 495) {
      pad_temp_shared[(((((int)threadIdx.z) * 16) + ((int)threadIdx.x)))] = (((1 <= ((((int)blockIdx.y) * 14) + (((((int)threadIdx.z) * 16) + ((int)threadIdx.x)) / 33))) && (1 <= ((((int)blockIdx.x) * 32) + (((((int)threadIdx.z) * 16) + ((int)threadIdx.x)) % 33)))) ? placeholder[(((((((rc_outer * 50176) + (((int)blockIdx.y) * 3136)) + ((((((int)threadIdx.z) * 16) + ((int)threadIdx.x)) / 33) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.z) * 16) + ((int)threadIdx.x)) % 33)) - 225))] : 0.000000e+00f);
    }
    if (((((int)threadIdx.x) / 9) + ((int)threadIdx.z)) < 32) {
      if (((((int)threadIdx.z) * 3) + (((int)threadIdx.x) / 3)) < 96) {
        if (((((int)threadIdx.z) * 9) + ((int)threadIdx.x)) < 288) {
          if (((int)threadIdx.x) < 9) {
            placeholder_shared[(((((int)threadIdx.z) * 9) + ((int)threadIdx.x)))] = placeholder1[((((((int)threadIdx.z) * 27) + (rc_outer * 9)) + ((int)threadIdx.x)))];
          }
        }
      }
    }
    __syncthreads();
    for (int ry_inner = 0; ry_inner < 3; ++ry_inner) {
      for (int rx_inner = 0; rx_inner < 3; ++rx_inner) {
        for (int yy = 0; yy < 7; ++yy) {
          compute[(yy)] = __ocml_fma_f32(pad_temp_shared[(((((yy * 66) + (ry_inner * 33)) + (((int)threadIdx.x) * 2)) + rx_inner))], placeholder_shared[((((((int)threadIdx.z) * 9) + (ry_inner * 3)) + rx_inner))], compute[(yy)]);
        }
      }
    }
  }
  for (int ax2_inner_inner_inner = 0; ax2_inner_inner_inner < 7; ++ax2_inner_inner_inner) {
    T_relu[((((((((int)threadIdx.z) * 12544) + (((int)blockIdx.y) * 784)) + (ax2_inner_inner_inner * 112)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)))] = max((compute[(ax2_inner_inner_inner)] + placeholder2[(((int)threadIdx.z))]), 0.000000e+00f);
  }
}

__device__ void fused_nn_conv2d_add_nn_relu_17_kernel0_device(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2){
  __shared__ float PaddedInput_shared[1824];
  __shared__ float placeholder_shared[9];
  float PaddedInput_shared_local[18];
  float placeholder_shared_local[9];
  float DepthwiseConv2d[4];
  PaddedInput_shared[(((((int)threadIdx.y) * 28) + ((int)threadIdx.x)))] = ((((1 <= ((((int)blockIdx.y) * 14) + (((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) / 114))) && (1 <= (((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) % 114))) && ((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) % 114) < 113)) ? placeholder[((((((((int)blockIdx.z) * 12544) + (((int)blockIdx.y) * 1568)) + ((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) / 114) * 112)) + (((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) % 114)) - 113))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 392))] = (((1 <= ((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 50) % 114)) && (((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 50) % 114) < 113)) ? placeholder[((((((((int)blockIdx.z) * 12544) + (((int)blockIdx.y) * 1568)) + (((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 392) / 114) * 112)) + ((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 50) % 114)) - 113))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 784))] = (((1 <= ((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 100) % 114)) && (((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 100) % 114) < 113)) ? placeholder[((((((((int)blockIdx.z) * 12544) + (((int)blockIdx.y) * 1568)) + (((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 784) / 114) * 112)) + ((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 100) % 114)) - 113))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 1176))] = (((1 <= ((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 36) % 114)) && (((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 36) % 114) < 113)) ? placeholder[((((((((int)blockIdx.z) * 12544) + (((int)blockIdx.y) * 1568)) + (((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 1176) / 114) * 112)) + ((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 36) % 114)) - 113))] : 0.000000e+00f);
  if (((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) < 256) {
    if (((int)threadIdx.y) < 10) {
      PaddedInput_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 1568))] = ((((((((int)blockIdx.y) * 14) + ((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 1568) / 114)) < 113) && (1 <= ((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 86) % 114))) && (((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 86) % 114) < 113)) ? placeholder[((((((((int)blockIdx.z) * 12544) + (((int)blockIdx.y) * 1568)) + (((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 1568) / 114) * 112)) + ((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 86) % 114)) - 113))] : 0.000000e+00f);
    }
  }
  if (((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) < 9) {
    if (((int)threadIdx.y) < 1) {
      placeholder_shared[(((((int)threadIdx.y) * 28) + ((int)threadIdx.x)))] = placeholder1[((((((int)threadIdx.y) * 28) + (((int)blockIdx.z) * 9)) + ((int)threadIdx.x)))];
    }
  }
  __syncthreads();
  PaddedInput_shared_local[(0)] = PaddedInput_shared[(((((int)threadIdx.y) * 114) + (((int)threadIdx.x) * 4)))];
  PaddedInput_shared_local[(1)] = PaddedInput_shared[((((((int)threadIdx.y) * 114) + (((int)threadIdx.x) * 4)) + 1))];
  PaddedInput_shared_local[(2)] = PaddedInput_shared[((((((int)threadIdx.y) * 114) + (((int)threadIdx.x) * 4)) + 2))];
  PaddedInput_shared_local[(3)] = PaddedInput_shared[((((((int)threadIdx.y) * 114) + (((int)threadIdx.x) * 4)) + 3))];
  PaddedInput_shared_local[(4)] = PaddedInput_shared[((((((int)threadIdx.y) * 114) + (((int)threadIdx.x) * 4)) + 4))];
  PaddedInput_shared_local[(5)] = PaddedInput_shared[((((((int)threadIdx.y) * 114) + (((int)threadIdx.x) * 4)) + 5))];
  PaddedInput_shared_local[(6)] = PaddedInput_shared[((((((int)threadIdx.y) * 114) + (((int)threadIdx.x) * 4)) + 114))];
  PaddedInput_shared_local[(7)] = PaddedInput_shared[((((((int)threadIdx.y) * 114) + (((int)threadIdx.x) * 4)) + 115))];
  PaddedInput_shared_local[(8)] = PaddedInput_shared[((((((int)threadIdx.y) * 114) + (((int)threadIdx.x) * 4)) + 116))];
  PaddedInput_shared_local[(9)] = PaddedInput_shared[((((((int)threadIdx.y) * 114) + (((int)threadIdx.x) * 4)) + 117))];
  PaddedInput_shared_local[(10)] = PaddedInput_shared[((((((int)threadIdx.y) * 114) + (((int)threadIdx.x) * 4)) + 118))];
  PaddedInput_shared_local[(11)] = PaddedInput_shared[((((((int)threadIdx.y) * 114) + (((int)threadIdx.x) * 4)) + 119))];
  PaddedInput_shared_local[(12)] = PaddedInput_shared[((((((int)threadIdx.y) * 114) + (((int)threadIdx.x) * 4)) + 228))];
  PaddedInput_shared_local[(13)] = PaddedInput_shared[((((((int)threadIdx.y) * 114) + (((int)threadIdx.x) * 4)) + 229))];
  PaddedInput_shared_local[(14)] = PaddedInput_shared[((((((int)threadIdx.y) * 114) + (((int)threadIdx.x) * 4)) + 230))];
  PaddedInput_shared_local[(15)] = PaddedInput_shared[((((((int)threadIdx.y) * 114) + (((int)threadIdx.x) * 4)) + 231))];
  PaddedInput_shared_local[(16)] = PaddedInput_shared[((((((int)threadIdx.y) * 114) + (((int)threadIdx.x) * 4)) + 232))];
  PaddedInput_shared_local[(17)] = PaddedInput_shared[((((((int)threadIdx.y) * 114) + (((int)threadIdx.x) * 4)) + 233))];
  placeholder_shared_local[(0)] = placeholder_shared[(0)];
  placeholder_shared_local[(1)] = placeholder_shared[(1)];
  placeholder_shared_local[(2)] = placeholder_shared[(2)];
  placeholder_shared_local[(3)] = placeholder_shared[(3)];
  placeholder_shared_local[(4)] = placeholder_shared[(4)];
  placeholder_shared_local[(5)] = placeholder_shared[(5)];
  placeholder_shared_local[(6)] = placeholder_shared[(6)];
  placeholder_shared_local[(7)] = placeholder_shared[(7)];
  placeholder_shared_local[(8)] = placeholder_shared[(8)];
  DepthwiseConv2d[(0)] = 0.000000e+00f;
  DepthwiseConv2d[(0)] = __ocml_fma_f32(PaddedInput_shared_local[(0)], placeholder_shared_local[(0)], DepthwiseConv2d[(0)]);
  DepthwiseConv2d[(0)] = __ocml_fma_f32(PaddedInput_shared_local[(1)], placeholder_shared_local[(1)], DepthwiseConv2d[(0)]);
  DepthwiseConv2d[(0)] = __ocml_fma_f32(PaddedInput_shared_local[(2)], placeholder_shared_local[(2)], DepthwiseConv2d[(0)]);
  DepthwiseConv2d[(0)] = __ocml_fma_f32(PaddedInput_shared_local[(6)], placeholder_shared_local[(3)], DepthwiseConv2d[(0)]);
  DepthwiseConv2d[(0)] = __ocml_fma_f32(PaddedInput_shared_local[(7)], placeholder_shared_local[(4)], DepthwiseConv2d[(0)]);
  DepthwiseConv2d[(0)] = __ocml_fma_f32(PaddedInput_shared_local[(8)], placeholder_shared_local[(5)], DepthwiseConv2d[(0)]);
  DepthwiseConv2d[(0)] = __ocml_fma_f32(PaddedInput_shared_local[(12)], placeholder_shared_local[(6)], DepthwiseConv2d[(0)]);
  DepthwiseConv2d[(0)] = __ocml_fma_f32(PaddedInput_shared_local[(13)], placeholder_shared_local[(7)], DepthwiseConv2d[(0)]);
  DepthwiseConv2d[(0)] = __ocml_fma_f32(PaddedInput_shared_local[(14)], placeholder_shared_local[(8)], DepthwiseConv2d[(0)]);
  DepthwiseConv2d[(1)] = 0.000000e+00f;
  DepthwiseConv2d[(1)] = __ocml_fma_f32(PaddedInput_shared_local[(1)], placeholder_shared_local[(0)], DepthwiseConv2d[(1)]);
  DepthwiseConv2d[(1)] = __ocml_fma_f32(PaddedInput_shared_local[(2)], placeholder_shared_local[(1)], DepthwiseConv2d[(1)]);
  DepthwiseConv2d[(1)] = __ocml_fma_f32(PaddedInput_shared_local[(3)], placeholder_shared_local[(2)], DepthwiseConv2d[(1)]);
  DepthwiseConv2d[(1)] = __ocml_fma_f32(PaddedInput_shared_local[(7)], placeholder_shared_local[(3)], DepthwiseConv2d[(1)]);
  DepthwiseConv2d[(1)] = __ocml_fma_f32(PaddedInput_shared_local[(8)], placeholder_shared_local[(4)], DepthwiseConv2d[(1)]);
  DepthwiseConv2d[(1)] = __ocml_fma_f32(PaddedInput_shared_local[(9)], placeholder_shared_local[(5)], DepthwiseConv2d[(1)]);
  DepthwiseConv2d[(1)] = __ocml_fma_f32(PaddedInput_shared_local[(13)], placeholder_shared_local[(6)], DepthwiseConv2d[(1)]);
  DepthwiseConv2d[(1)] = __ocml_fma_f32(PaddedInput_shared_local[(14)], placeholder_shared_local[(7)], DepthwiseConv2d[(1)]);
  DepthwiseConv2d[(1)] = __ocml_fma_f32(PaddedInput_shared_local[(15)], placeholder_shared_local[(8)], DepthwiseConv2d[(1)]);
  DepthwiseConv2d[(2)] = 0.000000e+00f;
  DepthwiseConv2d[(2)] = __ocml_fma_f32(PaddedInput_shared_local[(2)], placeholder_shared_local[(0)], DepthwiseConv2d[(2)]);
  DepthwiseConv2d[(2)] = __ocml_fma_f32(PaddedInput_shared_local[(3)], placeholder_shared_local[(1)], DepthwiseConv2d[(2)]);
  DepthwiseConv2d[(2)] = __ocml_fma_f32(PaddedInput_shared_local[(4)], placeholder_shared_local[(2)], DepthwiseConv2d[(2)]);
  DepthwiseConv2d[(2)] = __ocml_fma_f32(PaddedInput_shared_local[(8)], placeholder_shared_local[(3)], DepthwiseConv2d[(2)]);
  DepthwiseConv2d[(2)] = __ocml_fma_f32(PaddedInput_shared_local[(9)], placeholder_shared_local[(4)], DepthwiseConv2d[(2)]);
  DepthwiseConv2d[(2)] = __ocml_fma_f32(PaddedInput_shared_local[(10)], placeholder_shared_local[(5)], DepthwiseConv2d[(2)]);
  DepthwiseConv2d[(2)] = __ocml_fma_f32(PaddedInput_shared_local[(14)], placeholder_shared_local[(6)], DepthwiseConv2d[(2)]);
  DepthwiseConv2d[(2)] = __ocml_fma_f32(PaddedInput_shared_local[(15)], placeholder_shared_local[(7)], DepthwiseConv2d[(2)]);
  DepthwiseConv2d[(2)] = __ocml_fma_f32(PaddedInput_shared_local[(16)], placeholder_shared_local[(8)], DepthwiseConv2d[(2)]);
  DepthwiseConv2d[(3)] = 0.000000e+00f;
  DepthwiseConv2d[(3)] = __ocml_fma_f32(PaddedInput_shared_local[(3)], placeholder_shared_local[(0)], DepthwiseConv2d[(3)]);
  DepthwiseConv2d[(3)] = __ocml_fma_f32(PaddedInput_shared_local[(4)], placeholder_shared_local[(1)], DepthwiseConv2d[(3)]);
  DepthwiseConv2d[(3)] = __ocml_fma_f32(PaddedInput_shared_local[(5)], placeholder_shared_local[(2)], DepthwiseConv2d[(3)]);
  DepthwiseConv2d[(3)] = __ocml_fma_f32(PaddedInput_shared_local[(9)], placeholder_shared_local[(3)], DepthwiseConv2d[(3)]);
  DepthwiseConv2d[(3)] = __ocml_fma_f32(PaddedInput_shared_local[(10)], placeholder_shared_local[(4)], DepthwiseConv2d[(3)]);
  DepthwiseConv2d[(3)] = __ocml_fma_f32(PaddedInput_shared_local[(11)], placeholder_shared_local[(5)], DepthwiseConv2d[(3)]);
  DepthwiseConv2d[(3)] = __ocml_fma_f32(PaddedInput_shared_local[(15)], placeholder_shared_local[(6)], DepthwiseConv2d[(3)]);
  DepthwiseConv2d[(3)] = __ocml_fma_f32(PaddedInput_shared_local[(16)], placeholder_shared_local[(7)], DepthwiseConv2d[(3)]);
  DepthwiseConv2d[(3)] = __ocml_fma_f32(PaddedInput_shared_local[(17)], placeholder_shared_local[(8)], DepthwiseConv2d[(3)]);
  T_relu[(((((((int)blockIdx.z) * 12544) + (((int)blockIdx.y) * 1568)) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)))] = max((DepthwiseConv2d[(0)] + placeholder2[(((int)blockIdx.z))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 12544) + (((int)blockIdx.y) * 1568)) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)) + 1))] = max((DepthwiseConv2d[(1)] + placeholder2[(((int)blockIdx.z))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 12544) + (((int)blockIdx.y) * 1568)) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)) + 2))] = max((DepthwiseConv2d[(2)] + placeholder2[(((int)blockIdx.z))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 12544) + (((int)blockIdx.y) * 1568)) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)) + 3))] = max((DepthwiseConv2d[(3)] + placeholder2[(((int)blockIdx.z))]), 0.000000e+00f);
}

__device__ void fused_nn_conv2d_add_nn_relu_11_kernel0_device(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2){
  __shared__ float PaddedInput_shared[3249];
  __shared__ float placeholder_shared[9];
  float PaddedInput_shared_local[9];
  float placeholder_shared_local[9];
  float DepthwiseConv2d[1];
  PaddedInput_shared[(((((int)threadIdx.y) * 28) + ((int)threadIdx.x)))] = (((57 <= ((((int)threadIdx.y) * 28) + ((int)threadIdx.x))) && (1 <= (((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) % 57))) ? placeholder[(((((((int)blockIdx.z) * 3136) + ((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) / 57) * 56)) + (((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) % 57)) - 57))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 784))] = ((1 <= ((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 43) % 57)) ? placeholder[(((((((int)blockIdx.z) * 3136) + (((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 784) / 57) * 56)) + ((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 43) % 57)) - 57))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 1568))] = ((1 <= ((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 29) % 57)) ? placeholder[(((((((int)blockIdx.z) * 3136) + (((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 1568) / 57) * 56)) + ((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 29) % 57)) - 57))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 2352))] = ((1 <= ((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 15) % 57)) ? placeholder[(((((((int)blockIdx.z) * 3136) + (((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 2352) / 57) * 56)) + ((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 15) % 57)) - 57))] : 0.000000e+00f);
  if (((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) < 113) {
    if (((int)threadIdx.y) < 5) {
      PaddedInput_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 3136))] = ((1 <= ((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 1) % 57)) ? placeholder[(((((((int)blockIdx.z) * 3136) + (((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 3136) / 57) * 56)) + ((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 1) % 57)) - 57))] : 0.000000e+00f);
    }
  }
  if (((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) < 9) {
    if (((int)threadIdx.y) < 1) {
      placeholder_shared[(((((int)threadIdx.y) * 28) + ((int)threadIdx.x)))] = placeholder1[((((((int)threadIdx.y) * 28) + (((int)blockIdx.z) * 9)) + ((int)threadIdx.x)))];
    }
  }
  __syncthreads();
  PaddedInput_shared_local[(0)] = PaddedInput_shared[(((((int)threadIdx.y) * 114) + (((int)threadIdx.x) * 2)))];
  PaddedInput_shared_local[(1)] = PaddedInput_shared[((((((int)threadIdx.y) * 114) + (((int)threadIdx.x) * 2)) + 1))];
  PaddedInput_shared_local[(2)] = PaddedInput_shared[((((((int)threadIdx.y) * 114) + (((int)threadIdx.x) * 2)) + 2))];
  PaddedInput_shared_local[(3)] = PaddedInput_shared[((((((int)threadIdx.y) * 114) + (((int)threadIdx.x) * 2)) + 57))];
  PaddedInput_shared_local[(4)] = PaddedInput_shared[((((((int)threadIdx.y) * 114) + (((int)threadIdx.x) * 2)) + 58))];
  PaddedInput_shared_local[(5)] = PaddedInput_shared[((((((int)threadIdx.y) * 114) + (((int)threadIdx.x) * 2)) + 59))];
  PaddedInput_shared_local[(6)] = PaddedInput_shared[((((((int)threadIdx.y) * 114) + (((int)threadIdx.x) * 2)) + 114))];
  PaddedInput_shared_local[(7)] = PaddedInput_shared[((((((int)threadIdx.y) * 114) + (((int)threadIdx.x) * 2)) + 115))];
  PaddedInput_shared_local[(8)] = PaddedInput_shared[((((((int)threadIdx.y) * 114) + (((int)threadIdx.x) * 2)) + 116))];
  placeholder_shared_local[(0)] = placeholder_shared[(0)];
  placeholder_shared_local[(1)] = placeholder_shared[(1)];
  placeholder_shared_local[(2)] = placeholder_shared[(2)];
  placeholder_shared_local[(3)] = placeholder_shared[(3)];
  placeholder_shared_local[(4)] = placeholder_shared[(4)];
  placeholder_shared_local[(5)] = placeholder_shared[(5)];
  placeholder_shared_local[(6)] = placeholder_shared[(6)];
  placeholder_shared_local[(7)] = placeholder_shared[(7)];
  placeholder_shared_local[(8)] = placeholder_shared[(8)];
  DepthwiseConv2d[(0)] = 0.000000e+00f;
  DepthwiseConv2d[(0)] = __ocml_fma_f32(PaddedInput_shared_local[(0)], placeholder_shared_local[(0)], DepthwiseConv2d[(0)]);
  DepthwiseConv2d[(0)] = __ocml_fma_f32(PaddedInput_shared_local[(1)], placeholder_shared_local[(1)], DepthwiseConv2d[(0)]);
  DepthwiseConv2d[(0)] = __ocml_fma_f32(PaddedInput_shared_local[(2)], placeholder_shared_local[(2)], DepthwiseConv2d[(0)]);
  DepthwiseConv2d[(0)] = __ocml_fma_f32(PaddedInput_shared_local[(3)], placeholder_shared_local[(3)], DepthwiseConv2d[(0)]);
  DepthwiseConv2d[(0)] = __ocml_fma_f32(PaddedInput_shared_local[(4)], placeholder_shared_local[(4)], DepthwiseConv2d[(0)]);
  DepthwiseConv2d[(0)] = __ocml_fma_f32(PaddedInput_shared_local[(5)], placeholder_shared_local[(5)], DepthwiseConv2d[(0)]);
  DepthwiseConv2d[(0)] = __ocml_fma_f32(PaddedInput_shared_local[(6)], placeholder_shared_local[(6)], DepthwiseConv2d[(0)]);
  DepthwiseConv2d[(0)] = __ocml_fma_f32(PaddedInput_shared_local[(7)], placeholder_shared_local[(7)], DepthwiseConv2d[(0)]);
  DepthwiseConv2d[(0)] = __ocml_fma_f32(PaddedInput_shared_local[(8)], placeholder_shared_local[(8)], DepthwiseConv2d[(0)]);
  T_relu[((((((int)blockIdx.z) * 784) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)))] = max((DepthwiseConv2d[(0)] + placeholder2[(((int)blockIdx.z))]), 0.000000e+00f);
}

__device__ void fused_nn_conv2d_add_nn_relu_16_kernel0_device(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2){
  float compute[16];
  __shared__ float pad_temp_shared[1024];
  __shared__ float placeholder_shared[1024];
  compute[(0)] = 0.000000e+00f;
  compute[(4)] = 0.000000e+00f;
  compute[(8)] = 0.000000e+00f;
  compute[(12)] = 0.000000e+00f;
  compute[(1)] = 0.000000e+00f;
  compute[(5)] = 0.000000e+00f;
  compute[(9)] = 0.000000e+00f;
  compute[(13)] = 0.000000e+00f;
  compute[(2)] = 0.000000e+00f;
  compute[(6)] = 0.000000e+00f;
  compute[(10)] = 0.000000e+00f;
  compute[(14)] = 0.000000e+00f;
  compute[(3)] = 0.000000e+00f;
  compute[(7)] = 0.000000e+00f;
  compute[(11)] = 0.000000e+00f;
  compute[(15)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 2; ++rc_outer) {
    __syncthreads();
    pad_temp_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))] = placeholder[(((((((rc_outer * 200704) + (((int)threadIdx.z) * 12544)) + (((int)blockIdx.y) * 448)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + (((int)threadIdx.x) * 4)))];
    pad_temp_shared[(((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 1))] = placeholder[((((((((rc_outer * 200704) + (((int)threadIdx.z) * 12544)) + (((int)blockIdx.y) * 448)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + (((int)threadIdx.x) * 4)) + 1))];
    pad_temp_shared[(((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 2))] = placeholder[((((((((rc_outer * 200704) + (((int)threadIdx.z) * 12544)) + (((int)blockIdx.y) * 448)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + (((int)threadIdx.x) * 4)) + 2))];
    pad_temp_shared[(((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 3))] = placeholder[((((((((rc_outer * 200704) + (((int)threadIdx.z) * 12544)) + (((int)blockIdx.y) * 448)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + (((int)threadIdx.x) * 4)) + 3))];
    placeholder_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))] = placeholder1[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + (rc_outer * 16)) + (((int)threadIdx.x) * 4)))];
    placeholder_shared[(((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 1))] = placeholder1[((((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + (rc_outer * 16)) + (((int)threadIdx.x) * 4)) + 1))];
    placeholder_shared[(((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 2))] = placeholder1[((((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + (rc_outer * 16)) + (((int)threadIdx.x) * 4)) + 2))];
    placeholder_shared[(((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 3))] = placeholder1[((((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + (rc_outer * 16)) + (((int)threadIdx.x) * 4)) + 3))];
    __syncthreads();
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)))], placeholder_shared[((((int)threadIdx.z) * 16))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)))], placeholder_shared[(((((int)threadIdx.z) * 16) + 256))], compute[(4)]);
    compute[(8)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)))], placeholder_shared[(((((int)threadIdx.z) * 16) + 512))], compute[(8)]);
    compute[(12)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)))], placeholder_shared[(((((int)threadIdx.z) * 16) + 768))], compute[(12)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 1))], placeholder_shared[((((int)threadIdx.z) * 16))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 1))], placeholder_shared[(((((int)threadIdx.z) * 16) + 256))], compute[(5)]);
    compute[(9)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 1))], placeholder_shared[(((((int)threadIdx.z) * 16) + 512))], compute[(9)]);
    compute[(13)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 1))], placeholder_shared[(((((int)threadIdx.z) * 16) + 768))], compute[(13)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 2))], placeholder_shared[((((int)threadIdx.z) * 16))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 2))], placeholder_shared[(((((int)threadIdx.z) * 16) + 256))], compute[(6)]);
    compute[(10)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 2))], placeholder_shared[(((((int)threadIdx.z) * 16) + 512))], compute[(10)]);
    compute[(14)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 2))], placeholder_shared[(((((int)threadIdx.z) * 16) + 768))], compute[(14)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 3))], placeholder_shared[((((int)threadIdx.z) * 16))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 3))], placeholder_shared[(((((int)threadIdx.z) * 16) + 256))], compute[(7)]);
    compute[(11)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 3))], placeholder_shared[(((((int)threadIdx.z) * 16) + 512))], compute[(11)]);
    compute[(15)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 3))], placeholder_shared[(((((int)threadIdx.z) * 16) + 768))], compute[(15)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 64))], placeholder_shared[(((((int)threadIdx.z) * 16) + 1))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 64))], placeholder_shared[(((((int)threadIdx.z) * 16) + 257))], compute[(4)]);
    compute[(8)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 64))], placeholder_shared[(((((int)threadIdx.z) * 16) + 513))], compute[(8)]);
    compute[(12)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 64))], placeholder_shared[(((((int)threadIdx.z) * 16) + 769))], compute[(12)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 65))], placeholder_shared[(((((int)threadIdx.z) * 16) + 1))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 65))], placeholder_shared[(((((int)threadIdx.z) * 16) + 257))], compute[(5)]);
    compute[(9)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 65))], placeholder_shared[(((((int)threadIdx.z) * 16) + 513))], compute[(9)]);
    compute[(13)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 65))], placeholder_shared[(((((int)threadIdx.z) * 16) + 769))], compute[(13)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 66))], placeholder_shared[(((((int)threadIdx.z) * 16) + 1))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 66))], placeholder_shared[(((((int)threadIdx.z) * 16) + 257))], compute[(6)]);
    compute[(10)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 66))], placeholder_shared[(((((int)threadIdx.z) * 16) + 513))], compute[(10)]);
    compute[(14)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 66))], placeholder_shared[(((((int)threadIdx.z) * 16) + 769))], compute[(14)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 67))], placeholder_shared[(((((int)threadIdx.z) * 16) + 1))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 67))], placeholder_shared[(((((int)threadIdx.z) * 16) + 257))], compute[(7)]);
    compute[(11)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 67))], placeholder_shared[(((((int)threadIdx.z) * 16) + 513))], compute[(11)]);
    compute[(15)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 67))], placeholder_shared[(((((int)threadIdx.z) * 16) + 769))], compute[(15)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 128))], placeholder_shared[(((((int)threadIdx.z) * 16) + 2))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 128))], placeholder_shared[(((((int)threadIdx.z) * 16) + 258))], compute[(4)]);
    compute[(8)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 128))], placeholder_shared[(((((int)threadIdx.z) * 16) + 514))], compute[(8)]);
    compute[(12)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 128))], placeholder_shared[(((((int)threadIdx.z) * 16) + 770))], compute[(12)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 129))], placeholder_shared[(((((int)threadIdx.z) * 16) + 2))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 129))], placeholder_shared[(((((int)threadIdx.z) * 16) + 258))], compute[(5)]);
    compute[(9)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 129))], placeholder_shared[(((((int)threadIdx.z) * 16) + 514))], compute[(9)]);
    compute[(13)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 129))], placeholder_shared[(((((int)threadIdx.z) * 16) + 770))], compute[(13)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 130))], placeholder_shared[(((((int)threadIdx.z) * 16) + 2))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 130))], placeholder_shared[(((((int)threadIdx.z) * 16) + 258))], compute[(6)]);
    compute[(10)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 130))], placeholder_shared[(((((int)threadIdx.z) * 16) + 514))], compute[(10)]);
    compute[(14)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 130))], placeholder_shared[(((((int)threadIdx.z) * 16) + 770))], compute[(14)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 131))], placeholder_shared[(((((int)threadIdx.z) * 16) + 2))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 131))], placeholder_shared[(((((int)threadIdx.z) * 16) + 258))], compute[(7)]);
    compute[(11)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 131))], placeholder_shared[(((((int)threadIdx.z) * 16) + 514))], compute[(11)]);
    compute[(15)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 131))], placeholder_shared[(((((int)threadIdx.z) * 16) + 770))], compute[(15)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 192))], placeholder_shared[(((((int)threadIdx.z) * 16) + 3))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 192))], placeholder_shared[(((((int)threadIdx.z) * 16) + 259))], compute[(4)]);
    compute[(8)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 192))], placeholder_shared[(((((int)threadIdx.z) * 16) + 515))], compute[(8)]);
    compute[(12)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 192))], placeholder_shared[(((((int)threadIdx.z) * 16) + 771))], compute[(12)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 193))], placeholder_shared[(((((int)threadIdx.z) * 16) + 3))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 193))], placeholder_shared[(((((int)threadIdx.z) * 16) + 259))], compute[(5)]);
    compute[(9)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 193))], placeholder_shared[(((((int)threadIdx.z) * 16) + 515))], compute[(9)]);
    compute[(13)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 193))], placeholder_shared[(((((int)threadIdx.z) * 16) + 771))], compute[(13)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 194))], placeholder_shared[(((((int)threadIdx.z) * 16) + 3))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 194))], placeholder_shared[(((((int)threadIdx.z) * 16) + 259))], compute[(6)]);
    compute[(10)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 194))], placeholder_shared[(((((int)threadIdx.z) * 16) + 515))], compute[(10)]);
    compute[(14)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 194))], placeholder_shared[(((((int)threadIdx.z) * 16) + 771))], compute[(14)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 195))], placeholder_shared[(((((int)threadIdx.z) * 16) + 3))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 195))], placeholder_shared[(((((int)threadIdx.z) * 16) + 259))], compute[(7)]);
    compute[(11)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 195))], placeholder_shared[(((((int)threadIdx.z) * 16) + 515))], compute[(11)]);
    compute[(15)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 195))], placeholder_shared[(((((int)threadIdx.z) * 16) + 771))], compute[(15)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 256))], placeholder_shared[(((((int)threadIdx.z) * 16) + 4))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 256))], placeholder_shared[(((((int)threadIdx.z) * 16) + 260))], compute[(4)]);
    compute[(8)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 256))], placeholder_shared[(((((int)threadIdx.z) * 16) + 516))], compute[(8)]);
    compute[(12)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 256))], placeholder_shared[(((((int)threadIdx.z) * 16) + 772))], compute[(12)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 257))], placeholder_shared[(((((int)threadIdx.z) * 16) + 4))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 257))], placeholder_shared[(((((int)threadIdx.z) * 16) + 260))], compute[(5)]);
    compute[(9)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 257))], placeholder_shared[(((((int)threadIdx.z) * 16) + 516))], compute[(9)]);
    compute[(13)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 257))], placeholder_shared[(((((int)threadIdx.z) * 16) + 772))], compute[(13)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 258))], placeholder_shared[(((((int)threadIdx.z) * 16) + 4))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 258))], placeholder_shared[(((((int)threadIdx.z) * 16) + 260))], compute[(6)]);
    compute[(10)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 258))], placeholder_shared[(((((int)threadIdx.z) * 16) + 516))], compute[(10)]);
    compute[(14)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 258))], placeholder_shared[(((((int)threadIdx.z) * 16) + 772))], compute[(14)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 259))], placeholder_shared[(((((int)threadIdx.z) * 16) + 4))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 259))], placeholder_shared[(((((int)threadIdx.z) * 16) + 260))], compute[(7)]);
    compute[(11)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 259))], placeholder_shared[(((((int)threadIdx.z) * 16) + 516))], compute[(11)]);
    compute[(15)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 259))], placeholder_shared[(((((int)threadIdx.z) * 16) + 772))], compute[(15)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 320))], placeholder_shared[(((((int)threadIdx.z) * 16) + 5))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 320))], placeholder_shared[(((((int)threadIdx.z) * 16) + 261))], compute[(4)]);
    compute[(8)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 320))], placeholder_shared[(((((int)threadIdx.z) * 16) + 517))], compute[(8)]);
    compute[(12)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 320))], placeholder_shared[(((((int)threadIdx.z) * 16) + 773))], compute[(12)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 321))], placeholder_shared[(((((int)threadIdx.z) * 16) + 5))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 321))], placeholder_shared[(((((int)threadIdx.z) * 16) + 261))], compute[(5)]);
    compute[(9)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 321))], placeholder_shared[(((((int)threadIdx.z) * 16) + 517))], compute[(9)]);
    compute[(13)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 321))], placeholder_shared[(((((int)threadIdx.z) * 16) + 773))], compute[(13)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 322))], placeholder_shared[(((((int)threadIdx.z) * 16) + 5))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 322))], placeholder_shared[(((((int)threadIdx.z) * 16) + 261))], compute[(6)]);
    compute[(10)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 322))], placeholder_shared[(((((int)threadIdx.z) * 16) + 517))], compute[(10)]);
    compute[(14)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 322))], placeholder_shared[(((((int)threadIdx.z) * 16) + 773))], compute[(14)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 323))], placeholder_shared[(((((int)threadIdx.z) * 16) + 5))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 323))], placeholder_shared[(((((int)threadIdx.z) * 16) + 261))], compute[(7)]);
    compute[(11)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 323))], placeholder_shared[(((((int)threadIdx.z) * 16) + 517))], compute[(11)]);
    compute[(15)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 323))], placeholder_shared[(((((int)threadIdx.z) * 16) + 773))], compute[(15)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 384))], placeholder_shared[(((((int)threadIdx.z) * 16) + 6))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 384))], placeholder_shared[(((((int)threadIdx.z) * 16) + 262))], compute[(4)]);
    compute[(8)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 384))], placeholder_shared[(((((int)threadIdx.z) * 16) + 518))], compute[(8)]);
    compute[(12)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 384))], placeholder_shared[(((((int)threadIdx.z) * 16) + 774))], compute[(12)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 385))], placeholder_shared[(((((int)threadIdx.z) * 16) + 6))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 385))], placeholder_shared[(((((int)threadIdx.z) * 16) + 262))], compute[(5)]);
    compute[(9)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 385))], placeholder_shared[(((((int)threadIdx.z) * 16) + 518))], compute[(9)]);
    compute[(13)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 385))], placeholder_shared[(((((int)threadIdx.z) * 16) + 774))], compute[(13)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 386))], placeholder_shared[(((((int)threadIdx.z) * 16) + 6))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 386))], placeholder_shared[(((((int)threadIdx.z) * 16) + 262))], compute[(6)]);
    compute[(10)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 386))], placeholder_shared[(((((int)threadIdx.z) * 16) + 518))], compute[(10)]);
    compute[(14)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 386))], placeholder_shared[(((((int)threadIdx.z) * 16) + 774))], compute[(14)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 387))], placeholder_shared[(((((int)threadIdx.z) * 16) + 6))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 387))], placeholder_shared[(((((int)threadIdx.z) * 16) + 262))], compute[(7)]);
    compute[(11)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 387))], placeholder_shared[(((((int)threadIdx.z) * 16) + 518))], compute[(11)]);
    compute[(15)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 387))], placeholder_shared[(((((int)threadIdx.z) * 16) + 774))], compute[(15)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 448))], placeholder_shared[(((((int)threadIdx.z) * 16) + 7))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 448))], placeholder_shared[(((((int)threadIdx.z) * 16) + 263))], compute[(4)]);
    compute[(8)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 448))], placeholder_shared[(((((int)threadIdx.z) * 16) + 519))], compute[(8)]);
    compute[(12)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 448))], placeholder_shared[(((((int)threadIdx.z) * 16) + 775))], compute[(12)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 449))], placeholder_shared[(((((int)threadIdx.z) * 16) + 7))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 449))], placeholder_shared[(((((int)threadIdx.z) * 16) + 263))], compute[(5)]);
    compute[(9)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 449))], placeholder_shared[(((((int)threadIdx.z) * 16) + 519))], compute[(9)]);
    compute[(13)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 449))], placeholder_shared[(((((int)threadIdx.z) * 16) + 775))], compute[(13)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 450))], placeholder_shared[(((((int)threadIdx.z) * 16) + 7))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 450))], placeholder_shared[(((((int)threadIdx.z) * 16) + 263))], compute[(6)]);
    compute[(10)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 450))], placeholder_shared[(((((int)threadIdx.z) * 16) + 519))], compute[(10)]);
    compute[(14)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 450))], placeholder_shared[(((((int)threadIdx.z) * 16) + 775))], compute[(14)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 451))], placeholder_shared[(((((int)threadIdx.z) * 16) + 7))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 451))], placeholder_shared[(((((int)threadIdx.z) * 16) + 263))], compute[(7)]);
    compute[(11)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 451))], placeholder_shared[(((((int)threadIdx.z) * 16) + 519))], compute[(11)]);
    compute[(15)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 451))], placeholder_shared[(((((int)threadIdx.z) * 16) + 775))], compute[(15)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 512))], placeholder_shared[(((((int)threadIdx.z) * 16) + 8))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 512))], placeholder_shared[(((((int)threadIdx.z) * 16) + 264))], compute[(4)]);
    compute[(8)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 512))], placeholder_shared[(((((int)threadIdx.z) * 16) + 520))], compute[(8)]);
    compute[(12)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 512))], placeholder_shared[(((((int)threadIdx.z) * 16) + 776))], compute[(12)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 513))], placeholder_shared[(((((int)threadIdx.z) * 16) + 8))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 513))], placeholder_shared[(((((int)threadIdx.z) * 16) + 264))], compute[(5)]);
    compute[(9)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 513))], placeholder_shared[(((((int)threadIdx.z) * 16) + 520))], compute[(9)]);
    compute[(13)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 513))], placeholder_shared[(((((int)threadIdx.z) * 16) + 776))], compute[(13)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 514))], placeholder_shared[(((((int)threadIdx.z) * 16) + 8))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 514))], placeholder_shared[(((((int)threadIdx.z) * 16) + 264))], compute[(6)]);
    compute[(10)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 514))], placeholder_shared[(((((int)threadIdx.z) * 16) + 520))], compute[(10)]);
    compute[(14)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 514))], placeholder_shared[(((((int)threadIdx.z) * 16) + 776))], compute[(14)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 515))], placeholder_shared[(((((int)threadIdx.z) * 16) + 8))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 515))], placeholder_shared[(((((int)threadIdx.z) * 16) + 264))], compute[(7)]);
    compute[(11)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 515))], placeholder_shared[(((((int)threadIdx.z) * 16) + 520))], compute[(11)]);
    compute[(15)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 515))], placeholder_shared[(((((int)threadIdx.z) * 16) + 776))], compute[(15)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 576))], placeholder_shared[(((((int)threadIdx.z) * 16) + 9))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 576))], placeholder_shared[(((((int)threadIdx.z) * 16) + 265))], compute[(4)]);
    compute[(8)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 576))], placeholder_shared[(((((int)threadIdx.z) * 16) + 521))], compute[(8)]);
    compute[(12)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 576))], placeholder_shared[(((((int)threadIdx.z) * 16) + 777))], compute[(12)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 577))], placeholder_shared[(((((int)threadIdx.z) * 16) + 9))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 577))], placeholder_shared[(((((int)threadIdx.z) * 16) + 265))], compute[(5)]);
    compute[(9)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 577))], placeholder_shared[(((((int)threadIdx.z) * 16) + 521))], compute[(9)]);
    compute[(13)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 577))], placeholder_shared[(((((int)threadIdx.z) * 16) + 777))], compute[(13)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 578))], placeholder_shared[(((((int)threadIdx.z) * 16) + 9))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 578))], placeholder_shared[(((((int)threadIdx.z) * 16) + 265))], compute[(6)]);
    compute[(10)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 578))], placeholder_shared[(((((int)threadIdx.z) * 16) + 521))], compute[(10)]);
    compute[(14)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 578))], placeholder_shared[(((((int)threadIdx.z) * 16) + 777))], compute[(14)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 579))], placeholder_shared[(((((int)threadIdx.z) * 16) + 9))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 579))], placeholder_shared[(((((int)threadIdx.z) * 16) + 265))], compute[(7)]);
    compute[(11)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 579))], placeholder_shared[(((((int)threadIdx.z) * 16) + 521))], compute[(11)]);
    compute[(15)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 579))], placeholder_shared[(((((int)threadIdx.z) * 16) + 777))], compute[(15)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 640))], placeholder_shared[(((((int)threadIdx.z) * 16) + 10))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 640))], placeholder_shared[(((((int)threadIdx.z) * 16) + 266))], compute[(4)]);
    compute[(8)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 640))], placeholder_shared[(((((int)threadIdx.z) * 16) + 522))], compute[(8)]);
    compute[(12)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 640))], placeholder_shared[(((((int)threadIdx.z) * 16) + 778))], compute[(12)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 641))], placeholder_shared[(((((int)threadIdx.z) * 16) + 10))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 641))], placeholder_shared[(((((int)threadIdx.z) * 16) + 266))], compute[(5)]);
    compute[(9)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 641))], placeholder_shared[(((((int)threadIdx.z) * 16) + 522))], compute[(9)]);
    compute[(13)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 641))], placeholder_shared[(((((int)threadIdx.z) * 16) + 778))], compute[(13)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 642))], placeholder_shared[(((((int)threadIdx.z) * 16) + 10))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 642))], placeholder_shared[(((((int)threadIdx.z) * 16) + 266))], compute[(6)]);
    compute[(10)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 642))], placeholder_shared[(((((int)threadIdx.z) * 16) + 522))], compute[(10)]);
    compute[(14)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 642))], placeholder_shared[(((((int)threadIdx.z) * 16) + 778))], compute[(14)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 643))], placeholder_shared[(((((int)threadIdx.z) * 16) + 10))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 643))], placeholder_shared[(((((int)threadIdx.z) * 16) + 266))], compute[(7)]);
    compute[(11)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 643))], placeholder_shared[(((((int)threadIdx.z) * 16) + 522))], compute[(11)]);
    compute[(15)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 643))], placeholder_shared[(((((int)threadIdx.z) * 16) + 778))], compute[(15)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 704))], placeholder_shared[(((((int)threadIdx.z) * 16) + 11))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 704))], placeholder_shared[(((((int)threadIdx.z) * 16) + 267))], compute[(4)]);
    compute[(8)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 704))], placeholder_shared[(((((int)threadIdx.z) * 16) + 523))], compute[(8)]);
    compute[(12)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 704))], placeholder_shared[(((((int)threadIdx.z) * 16) + 779))], compute[(12)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 705))], placeholder_shared[(((((int)threadIdx.z) * 16) + 11))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 705))], placeholder_shared[(((((int)threadIdx.z) * 16) + 267))], compute[(5)]);
    compute[(9)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 705))], placeholder_shared[(((((int)threadIdx.z) * 16) + 523))], compute[(9)]);
    compute[(13)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 705))], placeholder_shared[(((((int)threadIdx.z) * 16) + 779))], compute[(13)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 706))], placeholder_shared[(((((int)threadIdx.z) * 16) + 11))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 706))], placeholder_shared[(((((int)threadIdx.z) * 16) + 267))], compute[(6)]);
    compute[(10)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 706))], placeholder_shared[(((((int)threadIdx.z) * 16) + 523))], compute[(10)]);
    compute[(14)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 706))], placeholder_shared[(((((int)threadIdx.z) * 16) + 779))], compute[(14)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 707))], placeholder_shared[(((((int)threadIdx.z) * 16) + 11))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 707))], placeholder_shared[(((((int)threadIdx.z) * 16) + 267))], compute[(7)]);
    compute[(11)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 707))], placeholder_shared[(((((int)threadIdx.z) * 16) + 523))], compute[(11)]);
    compute[(15)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 707))], placeholder_shared[(((((int)threadIdx.z) * 16) + 779))], compute[(15)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 768))], placeholder_shared[(((((int)threadIdx.z) * 16) + 12))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 768))], placeholder_shared[(((((int)threadIdx.z) * 16) + 268))], compute[(4)]);
    compute[(8)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 768))], placeholder_shared[(((((int)threadIdx.z) * 16) + 524))], compute[(8)]);
    compute[(12)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 768))], placeholder_shared[(((((int)threadIdx.z) * 16) + 780))], compute[(12)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 769))], placeholder_shared[(((((int)threadIdx.z) * 16) + 12))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 769))], placeholder_shared[(((((int)threadIdx.z) * 16) + 268))], compute[(5)]);
    compute[(9)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 769))], placeholder_shared[(((((int)threadIdx.z) * 16) + 524))], compute[(9)]);
    compute[(13)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 769))], placeholder_shared[(((((int)threadIdx.z) * 16) + 780))], compute[(13)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 770))], placeholder_shared[(((((int)threadIdx.z) * 16) + 12))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 770))], placeholder_shared[(((((int)threadIdx.z) * 16) + 268))], compute[(6)]);
    compute[(10)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 770))], placeholder_shared[(((((int)threadIdx.z) * 16) + 524))], compute[(10)]);
    compute[(14)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 770))], placeholder_shared[(((((int)threadIdx.z) * 16) + 780))], compute[(14)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 771))], placeholder_shared[(((((int)threadIdx.z) * 16) + 12))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 771))], placeholder_shared[(((((int)threadIdx.z) * 16) + 268))], compute[(7)]);
    compute[(11)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 771))], placeholder_shared[(((((int)threadIdx.z) * 16) + 524))], compute[(11)]);
    compute[(15)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 771))], placeholder_shared[(((((int)threadIdx.z) * 16) + 780))], compute[(15)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 832))], placeholder_shared[(((((int)threadIdx.z) * 16) + 13))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 832))], placeholder_shared[(((((int)threadIdx.z) * 16) + 269))], compute[(4)]);
    compute[(8)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 832))], placeholder_shared[(((((int)threadIdx.z) * 16) + 525))], compute[(8)]);
    compute[(12)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 832))], placeholder_shared[(((((int)threadIdx.z) * 16) + 781))], compute[(12)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 833))], placeholder_shared[(((((int)threadIdx.z) * 16) + 13))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 833))], placeholder_shared[(((((int)threadIdx.z) * 16) + 269))], compute[(5)]);
    compute[(9)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 833))], placeholder_shared[(((((int)threadIdx.z) * 16) + 525))], compute[(9)]);
    compute[(13)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 833))], placeholder_shared[(((((int)threadIdx.z) * 16) + 781))], compute[(13)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 834))], placeholder_shared[(((((int)threadIdx.z) * 16) + 13))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 834))], placeholder_shared[(((((int)threadIdx.z) * 16) + 269))], compute[(6)]);
    compute[(10)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 834))], placeholder_shared[(((((int)threadIdx.z) * 16) + 525))], compute[(10)]);
    compute[(14)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 834))], placeholder_shared[(((((int)threadIdx.z) * 16) + 781))], compute[(14)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 835))], placeholder_shared[(((((int)threadIdx.z) * 16) + 13))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 835))], placeholder_shared[(((((int)threadIdx.z) * 16) + 269))], compute[(7)]);
    compute[(11)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 835))], placeholder_shared[(((((int)threadIdx.z) * 16) + 525))], compute[(11)]);
    compute[(15)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 835))], placeholder_shared[(((((int)threadIdx.z) * 16) + 781))], compute[(15)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 896))], placeholder_shared[(((((int)threadIdx.z) * 16) + 14))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 896))], placeholder_shared[(((((int)threadIdx.z) * 16) + 270))], compute[(4)]);
    compute[(8)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 896))], placeholder_shared[(((((int)threadIdx.z) * 16) + 526))], compute[(8)]);
    compute[(12)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 896))], placeholder_shared[(((((int)threadIdx.z) * 16) + 782))], compute[(12)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 897))], placeholder_shared[(((((int)threadIdx.z) * 16) + 14))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 897))], placeholder_shared[(((((int)threadIdx.z) * 16) + 270))], compute[(5)]);
    compute[(9)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 897))], placeholder_shared[(((((int)threadIdx.z) * 16) + 526))], compute[(9)]);
    compute[(13)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 897))], placeholder_shared[(((((int)threadIdx.z) * 16) + 782))], compute[(13)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 898))], placeholder_shared[(((((int)threadIdx.z) * 16) + 14))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 898))], placeholder_shared[(((((int)threadIdx.z) * 16) + 270))], compute[(6)]);
    compute[(10)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 898))], placeholder_shared[(((((int)threadIdx.z) * 16) + 526))], compute[(10)]);
    compute[(14)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 898))], placeholder_shared[(((((int)threadIdx.z) * 16) + 782))], compute[(14)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 899))], placeholder_shared[(((((int)threadIdx.z) * 16) + 14))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 899))], placeholder_shared[(((((int)threadIdx.z) * 16) + 270))], compute[(7)]);
    compute[(11)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 899))], placeholder_shared[(((((int)threadIdx.z) * 16) + 526))], compute[(11)]);
    compute[(15)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 899))], placeholder_shared[(((((int)threadIdx.z) * 16) + 782))], compute[(15)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 960))], placeholder_shared[(((((int)threadIdx.z) * 16) + 15))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 960))], placeholder_shared[(((((int)threadIdx.z) * 16) + 271))], compute[(4)]);
    compute[(8)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 960))], placeholder_shared[(((((int)threadIdx.z) * 16) + 527))], compute[(8)]);
    compute[(12)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 960))], placeholder_shared[(((((int)threadIdx.z) * 16) + 783))], compute[(12)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 961))], placeholder_shared[(((((int)threadIdx.z) * 16) + 15))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 961))], placeholder_shared[(((((int)threadIdx.z) * 16) + 271))], compute[(5)]);
    compute[(9)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 961))], placeholder_shared[(((((int)threadIdx.z) * 16) + 527))], compute[(9)]);
    compute[(13)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 961))], placeholder_shared[(((((int)threadIdx.z) * 16) + 783))], compute[(13)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 962))], placeholder_shared[(((((int)threadIdx.z) * 16) + 15))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 962))], placeholder_shared[(((((int)threadIdx.z) * 16) + 271))], compute[(6)]);
    compute[(10)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 962))], placeholder_shared[(((((int)threadIdx.z) * 16) + 527))], compute[(10)]);
    compute[(14)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 962))], placeholder_shared[(((((int)threadIdx.z) * 16) + 783))], compute[(14)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 963))], placeholder_shared[(((((int)threadIdx.z) * 16) + 15))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 963))], placeholder_shared[(((((int)threadIdx.z) * 16) + 271))], compute[(7)]);
    compute[(11)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 963))], placeholder_shared[(((((int)threadIdx.z) * 16) + 527))], compute[(11)]);
    compute[(15)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 963))], placeholder_shared[(((((int)threadIdx.z) * 16) + 783))], compute[(15)]);
  }
  T_relu[((((((((int)threadIdx.z) * 12544) + (((int)blockIdx.y) * 448)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + (((int)threadIdx.x) * 4)))] = max((compute[(0)] + placeholder2[(((int)threadIdx.z))]), 0.000000e+00f);
  T_relu[(((((((((int)threadIdx.z) * 12544) + (((int)blockIdx.y) * 448)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + (((int)threadIdx.x) * 4)) + 200704))] = max((compute[(4)] + placeholder2[((((int)threadIdx.z) + 16))]), 0.000000e+00f);
  T_relu[(((((((((int)threadIdx.z) * 12544) + (((int)blockIdx.y) * 448)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + (((int)threadIdx.x) * 4)) + 401408))] = max((compute[(8)] + placeholder2[((((int)threadIdx.z) + 32))]), 0.000000e+00f);
  T_relu[(((((((((int)threadIdx.z) * 12544) + (((int)blockIdx.y) * 448)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + (((int)threadIdx.x) * 4)) + 602112))] = max((compute[(12)] + placeholder2[((((int)threadIdx.z) + 48))]), 0.000000e+00f);
  T_relu[(((((((((int)threadIdx.z) * 12544) + (((int)blockIdx.y) * 448)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + (((int)threadIdx.x) * 4)) + 1))] = max((compute[(1)] + placeholder2[(((int)threadIdx.z))]), 0.000000e+00f);
  T_relu[(((((((((int)threadIdx.z) * 12544) + (((int)blockIdx.y) * 448)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + (((int)threadIdx.x) * 4)) + 200705))] = max((compute[(5)] + placeholder2[((((int)threadIdx.z) + 16))]), 0.000000e+00f);
  T_relu[(((((((((int)threadIdx.z) * 12544) + (((int)blockIdx.y) * 448)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + (((int)threadIdx.x) * 4)) + 401409))] = max((compute[(9)] + placeholder2[((((int)threadIdx.z) + 32))]), 0.000000e+00f);
  T_relu[(((((((((int)threadIdx.z) * 12544) + (((int)blockIdx.y) * 448)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + (((int)threadIdx.x) * 4)) + 602113))] = max((compute[(13)] + placeholder2[((((int)threadIdx.z) + 48))]), 0.000000e+00f);
  T_relu[(((((((((int)threadIdx.z) * 12544) + (((int)blockIdx.y) * 448)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + (((int)threadIdx.x) * 4)) + 2))] = max((compute[(2)] + placeholder2[(((int)threadIdx.z))]), 0.000000e+00f);
  T_relu[(((((((((int)threadIdx.z) * 12544) + (((int)blockIdx.y) * 448)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + (((int)threadIdx.x) * 4)) + 200706))] = max((compute[(6)] + placeholder2[((((int)threadIdx.z) + 16))]), 0.000000e+00f);
  T_relu[(((((((((int)threadIdx.z) * 12544) + (((int)blockIdx.y) * 448)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + (((int)threadIdx.x) * 4)) + 401410))] = max((compute[(10)] + placeholder2[((((int)threadIdx.z) + 32))]), 0.000000e+00f);
  T_relu[(((((((((int)threadIdx.z) * 12544) + (((int)blockIdx.y) * 448)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + (((int)threadIdx.x) * 4)) + 602114))] = max((compute[(14)] + placeholder2[((((int)threadIdx.z) + 48))]), 0.000000e+00f);
  T_relu[(((((((((int)threadIdx.z) * 12544) + (((int)blockIdx.y) * 448)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + (((int)threadIdx.x) * 4)) + 3))] = max((compute[(3)] + placeholder2[(((int)threadIdx.z))]), 0.000000e+00f);
  T_relu[(((((((((int)threadIdx.z) * 12544) + (((int)blockIdx.y) * 448)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + (((int)threadIdx.x) * 4)) + 200707))] = max((compute[(7)] + placeholder2[((((int)threadIdx.z) + 16))]), 0.000000e+00f);
  T_relu[(((((((((int)threadIdx.z) * 12544) + (((int)blockIdx.y) * 448)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + (((int)threadIdx.x) * 4)) + 401411))] = max((compute[(11)] + placeholder2[((((int)threadIdx.z) + 32))]), 0.000000e+00f);
  T_relu[(((((((((int)threadIdx.z) * 12544) + (((int)blockIdx.y) * 448)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + (((int)threadIdx.x) * 4)) + 602115))] = max((compute[(15)] + placeholder2[((((int)threadIdx.z) + 48))]), 0.000000e+00f);
}

__device__ void fused_nn_conv2d_add_nn_relu_2_kernel0_device(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2){
  float compute[1];
  __shared__ float pad_temp_shared[1568];
  __shared__ float placeholder_shared[512];
  compute[(0)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 16; ++rc_outer) {
    __syncthreads();
    pad_temp_shared[((((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)))] = placeholder[(((((rc_outer * 1568) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)))];
    pad_temp_shared[(((((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)) + 1))] = placeholder[(((((rc_outer * 1568) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 14)) + ((((int)threadIdx.x) * 2) + 1)))];
    if (((((((int)threadIdx.y) * 5) + ((int)threadIdx.x)) >> 5) + ((int)threadIdx.z)) < 16) {
      if ((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 5)) + ((int)threadIdx.x)) < 512) {
        if (((((int)threadIdx.y) * 5) + ((int)threadIdx.x)) < 32) {
          if (((int)threadIdx.x) < 5) {
            placeholder_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 5)) + ((int)threadIdx.x)))] = placeholder1[((((((((int)blockIdx.z) * 8192) + (((int)threadIdx.z) * 512)) + (rc_outer * 32)) + (((int)threadIdx.y) * 5)) + ((int)threadIdx.x)))];
          }
        }
      }
    }
    __syncthreads();
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 7) + ((int)threadIdx.x)))], placeholder_shared[((((int)threadIdx.z) * 32))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 49))], placeholder_shared[(((((int)threadIdx.z) * 32) + 1))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 98))], placeholder_shared[(((((int)threadIdx.z) * 32) + 2))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 147))], placeholder_shared[(((((int)threadIdx.z) * 32) + 3))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 196))], placeholder_shared[(((((int)threadIdx.z) * 32) + 4))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 245))], placeholder_shared[(((((int)threadIdx.z) * 32) + 5))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 294))], placeholder_shared[(((((int)threadIdx.z) * 32) + 6))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 343))], placeholder_shared[(((((int)threadIdx.z) * 32) + 7))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 392))], placeholder_shared[(((((int)threadIdx.z) * 32) + 8))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 441))], placeholder_shared[(((((int)threadIdx.z) * 32) + 9))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 490))], placeholder_shared[(((((int)threadIdx.z) * 32) + 10))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 539))], placeholder_shared[(((((int)threadIdx.z) * 32) + 11))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 588))], placeholder_shared[(((((int)threadIdx.z) * 32) + 12))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 637))], placeholder_shared[(((((int)threadIdx.z) * 32) + 13))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 686))], placeholder_shared[(((((int)threadIdx.z) * 32) + 14))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 735))], placeholder_shared[(((((int)threadIdx.z) * 32) + 15))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 784))], placeholder_shared[(((((int)threadIdx.z) * 32) + 16))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 833))], placeholder_shared[(((((int)threadIdx.z) * 32) + 17))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 882))], placeholder_shared[(((((int)threadIdx.z) * 32) + 18))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 931))], placeholder_shared[(((((int)threadIdx.z) * 32) + 19))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 980))], placeholder_shared[(((((int)threadIdx.z) * 32) + 20))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1029))], placeholder_shared[(((((int)threadIdx.z) * 32) + 21))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1078))], placeholder_shared[(((((int)threadIdx.z) * 32) + 22))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1127))], placeholder_shared[(((((int)threadIdx.z) * 32) + 23))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1176))], placeholder_shared[(((((int)threadIdx.z) * 32) + 24))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1225))], placeholder_shared[(((((int)threadIdx.z) * 32) + 25))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1274))], placeholder_shared[(((((int)threadIdx.z) * 32) + 26))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1323))], placeholder_shared[(((((int)threadIdx.z) * 32) + 27))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1372))], placeholder_shared[(((((int)threadIdx.z) * 32) + 28))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1421))], placeholder_shared[(((((int)threadIdx.z) * 32) + 29))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1470))], placeholder_shared[(((((int)threadIdx.z) * 32) + 30))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1519))], placeholder_shared[(((((int)threadIdx.z) * 32) + 31))], compute[(0)]);
  }
  T_relu[(((((((int)blockIdx.z) * 784) + (((int)threadIdx.z) * 49)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)))] = max((compute[(0)] + placeholder2[(((((int)blockIdx.z) * 16) + ((int)threadIdx.z)))]), 0.000000e+00f);
}

__device__ void fused_nn_batch_flatten_kernel0_device(float* __restrict__ tensor, float* __restrict__ placeholder){
  tensor[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)))] = placeholder[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)))];
}

__device__ void fused_nn_conv2d_add_nn_relu_13_kernel0_device(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2){
  __shared__ float PaddedInput_shared[3364];
  __shared__ float placeholder_shared[9];
  float PaddedInput_shared_local[16];
  float placeholder_shared_local[9];
  float DepthwiseConv2d[4];
  PaddedInput_shared[(((((int)threadIdx.y) * 28) + ((int)threadIdx.x)))] = ((((58 <= ((((int)threadIdx.y) * 28) + ((int)threadIdx.x))) && (1 <= (((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) % 58))) && ((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) % 58) < 57)) ? placeholder[(((((((int)blockIdx.z) * 3136) + ((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) / 58) * 56)) + (((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) % 58)) - 57))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 784))] = (((1 <= ((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 30) % 58)) && (((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 30) % 58) < 57)) ? placeholder[(((((((int)blockIdx.z) * 3136) + (((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 784) / 58) * 56)) + ((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 30) % 58)) - 57))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 1568))] = (((1 <= ((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 2) % 58)) && (((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 2) % 58) < 57)) ? placeholder[(((((((int)blockIdx.z) * 3136) + (((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 1568) / 58) * 56)) + ((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 2) % 58)) - 57))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 2352))] = (((1 <= ((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 32) % 58)) && (((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 32) % 58) < 57)) ? placeholder[(((((((int)blockIdx.z) * 3136) + (((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 2352) / 58) * 56)) + ((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 32) % 58)) - 57))] : 0.000000e+00f);
  if (((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) < 228) {
    if (((int)threadIdx.y) < 9) {
      PaddedInput_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 3136))] = ((((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) < 170) && (1 <= ((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 4) % 58))) && (((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 4) % 58) < 57)) ? placeholder[(((((((int)blockIdx.z) * 3136) + (((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 3136) / 58) * 56)) + ((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 4) % 58)) - 57))] : 0.000000e+00f);
    }
  }
  if (((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) < 9) {
    if (((int)threadIdx.y) < 1) {
      placeholder_shared[(((((int)threadIdx.y) * 28) + ((int)threadIdx.x)))] = placeholder1[((((((int)threadIdx.y) * 28) + (((int)blockIdx.z) * 9)) + ((int)threadIdx.x)))];
    }
  }
  __syncthreads();
  PaddedInput_shared_local[(0)] = PaddedInput_shared[(((((int)threadIdx.y) * 116) + (((int)threadIdx.x) * 2)))];
  PaddedInput_shared_local[(1)] = PaddedInput_shared[((((((int)threadIdx.y) * 116) + (((int)threadIdx.x) * 2)) + 1))];
  PaddedInput_shared_local[(2)] = PaddedInput_shared[((((((int)threadIdx.y) * 116) + (((int)threadIdx.x) * 2)) + 2))];
  PaddedInput_shared_local[(3)] = PaddedInput_shared[((((((int)threadIdx.y) * 116) + (((int)threadIdx.x) * 2)) + 3))];
  PaddedInput_shared_local[(4)] = PaddedInput_shared[((((((int)threadIdx.y) * 116) + (((int)threadIdx.x) * 2)) + 58))];
  PaddedInput_shared_local[(5)] = PaddedInput_shared[((((((int)threadIdx.y) * 116) + (((int)threadIdx.x) * 2)) + 59))];
  PaddedInput_shared_local[(6)] = PaddedInput_shared[((((((int)threadIdx.y) * 116) + (((int)threadIdx.x) * 2)) + 60))];
  PaddedInput_shared_local[(7)] = PaddedInput_shared[((((((int)threadIdx.y) * 116) + (((int)threadIdx.x) * 2)) + 61))];
  PaddedInput_shared_local[(8)] = PaddedInput_shared[((((((int)threadIdx.y) * 116) + (((int)threadIdx.x) * 2)) + 116))];
  PaddedInput_shared_local[(9)] = PaddedInput_shared[((((((int)threadIdx.y) * 116) + (((int)threadIdx.x) * 2)) + 117))];
  PaddedInput_shared_local[(10)] = PaddedInput_shared[((((((int)threadIdx.y) * 116) + (((int)threadIdx.x) * 2)) + 118))];
  PaddedInput_shared_local[(11)] = PaddedInput_shared[((((((int)threadIdx.y) * 116) + (((int)threadIdx.x) * 2)) + 119))];
  PaddedInput_shared_local[(12)] = PaddedInput_shared[((((((int)threadIdx.y) * 116) + (((int)threadIdx.x) * 2)) + 174))];
  PaddedInput_shared_local[(13)] = PaddedInput_shared[((((((int)threadIdx.y) * 116) + (((int)threadIdx.x) * 2)) + 175))];
  PaddedInput_shared_local[(14)] = PaddedInput_shared[((((((int)threadIdx.y) * 116) + (((int)threadIdx.x) * 2)) + 176))];
  PaddedInput_shared_local[(15)] = PaddedInput_shared[((((((int)threadIdx.y) * 116) + (((int)threadIdx.x) * 2)) + 177))];
  placeholder_shared_local[(0)] = placeholder_shared[(0)];
  placeholder_shared_local[(1)] = placeholder_shared[(1)];
  placeholder_shared_local[(2)] = placeholder_shared[(2)];
  placeholder_shared_local[(3)] = placeholder_shared[(3)];
  placeholder_shared_local[(4)] = placeholder_shared[(4)];
  placeholder_shared_local[(5)] = placeholder_shared[(5)];
  placeholder_shared_local[(6)] = placeholder_shared[(6)];
  placeholder_shared_local[(7)] = placeholder_shared[(7)];
  placeholder_shared_local[(8)] = placeholder_shared[(8)];
  DepthwiseConv2d[(0)] = 0.000000e+00f;
  DepthwiseConv2d[(0)] = __ocml_fma_f32(PaddedInput_shared_local[(0)], placeholder_shared_local[(0)], DepthwiseConv2d[(0)]);
  DepthwiseConv2d[(0)] = __ocml_fma_f32(PaddedInput_shared_local[(1)], placeholder_shared_local[(1)], DepthwiseConv2d[(0)]);
  DepthwiseConv2d[(0)] = __ocml_fma_f32(PaddedInput_shared_local[(2)], placeholder_shared_local[(2)], DepthwiseConv2d[(0)]);
  DepthwiseConv2d[(0)] = __ocml_fma_f32(PaddedInput_shared_local[(4)], placeholder_shared_local[(3)], DepthwiseConv2d[(0)]);
  DepthwiseConv2d[(0)] = __ocml_fma_f32(PaddedInput_shared_local[(5)], placeholder_shared_local[(4)], DepthwiseConv2d[(0)]);
  DepthwiseConv2d[(0)] = __ocml_fma_f32(PaddedInput_shared_local[(6)], placeholder_shared_local[(5)], DepthwiseConv2d[(0)]);
  DepthwiseConv2d[(0)] = __ocml_fma_f32(PaddedInput_shared_local[(8)], placeholder_shared_local[(6)], DepthwiseConv2d[(0)]);
  DepthwiseConv2d[(0)] = __ocml_fma_f32(PaddedInput_shared_local[(9)], placeholder_shared_local[(7)], DepthwiseConv2d[(0)]);
  DepthwiseConv2d[(0)] = __ocml_fma_f32(PaddedInput_shared_local[(10)], placeholder_shared_local[(8)], DepthwiseConv2d[(0)]);
  DepthwiseConv2d[(1)] = 0.000000e+00f;
  DepthwiseConv2d[(1)] = __ocml_fma_f32(PaddedInput_shared_local[(1)], placeholder_shared_local[(0)], DepthwiseConv2d[(1)]);
  DepthwiseConv2d[(1)] = __ocml_fma_f32(PaddedInput_shared_local[(2)], placeholder_shared_local[(1)], DepthwiseConv2d[(1)]);
  DepthwiseConv2d[(1)] = __ocml_fma_f32(PaddedInput_shared_local[(3)], placeholder_shared_local[(2)], DepthwiseConv2d[(1)]);
  DepthwiseConv2d[(1)] = __ocml_fma_f32(PaddedInput_shared_local[(5)], placeholder_shared_local[(3)], DepthwiseConv2d[(1)]);
  DepthwiseConv2d[(1)] = __ocml_fma_f32(PaddedInput_shared_local[(6)], placeholder_shared_local[(4)], DepthwiseConv2d[(1)]);
  DepthwiseConv2d[(1)] = __ocml_fma_f32(PaddedInput_shared_local[(7)], placeholder_shared_local[(5)], DepthwiseConv2d[(1)]);
  DepthwiseConv2d[(1)] = __ocml_fma_f32(PaddedInput_shared_local[(9)], placeholder_shared_local[(6)], DepthwiseConv2d[(1)]);
  DepthwiseConv2d[(1)] = __ocml_fma_f32(PaddedInput_shared_local[(10)], placeholder_shared_local[(7)], DepthwiseConv2d[(1)]);
  DepthwiseConv2d[(1)] = __ocml_fma_f32(PaddedInput_shared_local[(11)], placeholder_shared_local[(8)], DepthwiseConv2d[(1)]);
  DepthwiseConv2d[(2)] = 0.000000e+00f;
  DepthwiseConv2d[(2)] = __ocml_fma_f32(PaddedInput_shared_local[(4)], placeholder_shared_local[(0)], DepthwiseConv2d[(2)]);
  DepthwiseConv2d[(2)] = __ocml_fma_f32(PaddedInput_shared_local[(5)], placeholder_shared_local[(1)], DepthwiseConv2d[(2)]);
  DepthwiseConv2d[(2)] = __ocml_fma_f32(PaddedInput_shared_local[(6)], placeholder_shared_local[(2)], DepthwiseConv2d[(2)]);
  DepthwiseConv2d[(2)] = __ocml_fma_f32(PaddedInput_shared_local[(8)], placeholder_shared_local[(3)], DepthwiseConv2d[(2)]);
  DepthwiseConv2d[(2)] = __ocml_fma_f32(PaddedInput_shared_local[(9)], placeholder_shared_local[(4)], DepthwiseConv2d[(2)]);
  DepthwiseConv2d[(2)] = __ocml_fma_f32(PaddedInput_shared_local[(10)], placeholder_shared_local[(5)], DepthwiseConv2d[(2)]);
  DepthwiseConv2d[(2)] = __ocml_fma_f32(PaddedInput_shared_local[(12)], placeholder_shared_local[(6)], DepthwiseConv2d[(2)]);
  DepthwiseConv2d[(2)] = __ocml_fma_f32(PaddedInput_shared_local[(13)], placeholder_shared_local[(7)], DepthwiseConv2d[(2)]);
  DepthwiseConv2d[(2)] = __ocml_fma_f32(PaddedInput_shared_local[(14)], placeholder_shared_local[(8)], DepthwiseConv2d[(2)]);
  DepthwiseConv2d[(3)] = 0.000000e+00f;
  DepthwiseConv2d[(3)] = __ocml_fma_f32(PaddedInput_shared_local[(5)], placeholder_shared_local[(0)], DepthwiseConv2d[(3)]);
  DepthwiseConv2d[(3)] = __ocml_fma_f32(PaddedInput_shared_local[(6)], placeholder_shared_local[(1)], DepthwiseConv2d[(3)]);
  DepthwiseConv2d[(3)] = __ocml_fma_f32(PaddedInput_shared_local[(7)], placeholder_shared_local[(2)], DepthwiseConv2d[(3)]);
  DepthwiseConv2d[(3)] = __ocml_fma_f32(PaddedInput_shared_local[(9)], placeholder_shared_local[(3)], DepthwiseConv2d[(3)]);
  DepthwiseConv2d[(3)] = __ocml_fma_f32(PaddedInput_shared_local[(10)], placeholder_shared_local[(4)], DepthwiseConv2d[(3)]);
  DepthwiseConv2d[(3)] = __ocml_fma_f32(PaddedInput_shared_local[(11)], placeholder_shared_local[(5)], DepthwiseConv2d[(3)]);
  DepthwiseConv2d[(3)] = __ocml_fma_f32(PaddedInput_shared_local[(13)], placeholder_shared_local[(6)], DepthwiseConv2d[(3)]);
  DepthwiseConv2d[(3)] = __ocml_fma_f32(PaddedInput_shared_local[(14)], placeholder_shared_local[(7)], DepthwiseConv2d[(3)]);
  DepthwiseConv2d[(3)] = __ocml_fma_f32(PaddedInput_shared_local[(15)], placeholder_shared_local[(8)], DepthwiseConv2d[(3)]);
  T_relu[((((((int)blockIdx.z) * 3136) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 2)))] = max((DepthwiseConv2d[(0)] + placeholder2[(((int)blockIdx.z))]), 0.000000e+00f);
  T_relu[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 2)) + 1))] = max((DepthwiseConv2d[(1)] + placeholder2[(((int)blockIdx.z))]), 0.000000e+00f);
  T_relu[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 2)) + 56))] = max((DepthwiseConv2d[(2)] + placeholder2[(((int)blockIdx.z))]), 0.000000e+00f);
  T_relu[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 2)) + 57))] = max((DepthwiseConv2d[(3)] + placeholder2[(((int)blockIdx.z))]), 0.000000e+00f);
}

__device__ void fused_nn_conv2d_add_nn_relu_8_kernel0_device(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2){
  float compute[4];
  __shared__ float pad_temp_shared[1792];
  __shared__ float placeholder_shared[512];
  compute[(0)] = 0.000000e+00f;
  compute[(2)] = 0.000000e+00f;
  compute[(1)] = 0.000000e+00f;
  compute[(3)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 16; ++rc_outer) {
    __syncthreads();
    pad_temp_shared[((((((int)threadIdx.z) * 112) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 2)))] = placeholder[((((((rc_outer * 12544) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 2)))];
    pad_temp_shared[(((((((int)threadIdx.z) * 112) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 2)) + 1))] = placeholder[(((((((rc_outer * 12544) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 2)) + 1))];
    if (((((int)threadIdx.z) * 2) + (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) >> 4)) < 32) {
      if ((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x)) < 512) {
        if (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) < 32) {
          if (((int)threadIdx.x) < 8) {
            placeholder_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x)))] = placeholder1[((((((((int)blockIdx.z) * 8192) + (((int)threadIdx.z) * 512)) + ((((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) >> 4) * 256)) + (rc_outer * 16)) + (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) & 15)))];
          }
        }
      }
    }
    __syncthreads();
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)))], placeholder_shared[((((int)threadIdx.z) * 16))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)))], placeholder_shared[(((((int)threadIdx.z) * 16) + 256))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 1))], placeholder_shared[((((int)threadIdx.z) * 16))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 1))], placeholder_shared[(((((int)threadIdx.z) * 16) + 256))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 112))], placeholder_shared[(((((int)threadIdx.z) * 16) + 1))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 112))], placeholder_shared[(((((int)threadIdx.z) * 16) + 257))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 113))], placeholder_shared[(((((int)threadIdx.z) * 16) + 1))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 113))], placeholder_shared[(((((int)threadIdx.z) * 16) + 257))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 224))], placeholder_shared[(((((int)threadIdx.z) * 16) + 2))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 224))], placeholder_shared[(((((int)threadIdx.z) * 16) + 258))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 225))], placeholder_shared[(((((int)threadIdx.z) * 16) + 2))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 225))], placeholder_shared[(((((int)threadIdx.z) * 16) + 258))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 336))], placeholder_shared[(((((int)threadIdx.z) * 16) + 3))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 336))], placeholder_shared[(((((int)threadIdx.z) * 16) + 259))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 337))], placeholder_shared[(((((int)threadIdx.z) * 16) + 3))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 337))], placeholder_shared[(((((int)threadIdx.z) * 16) + 259))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 448))], placeholder_shared[(((((int)threadIdx.z) * 16) + 4))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 448))], placeholder_shared[(((((int)threadIdx.z) * 16) + 260))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 449))], placeholder_shared[(((((int)threadIdx.z) * 16) + 4))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 449))], placeholder_shared[(((((int)threadIdx.z) * 16) + 260))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 560))], placeholder_shared[(((((int)threadIdx.z) * 16) + 5))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 560))], placeholder_shared[(((((int)threadIdx.z) * 16) + 261))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 561))], placeholder_shared[(((((int)threadIdx.z) * 16) + 5))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 561))], placeholder_shared[(((((int)threadIdx.z) * 16) + 261))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 672))], placeholder_shared[(((((int)threadIdx.z) * 16) + 6))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 672))], placeholder_shared[(((((int)threadIdx.z) * 16) + 262))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 673))], placeholder_shared[(((((int)threadIdx.z) * 16) + 6))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 673))], placeholder_shared[(((((int)threadIdx.z) * 16) + 262))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 784))], placeholder_shared[(((((int)threadIdx.z) * 16) + 7))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 784))], placeholder_shared[(((((int)threadIdx.z) * 16) + 263))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 785))], placeholder_shared[(((((int)threadIdx.z) * 16) + 7))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 785))], placeholder_shared[(((((int)threadIdx.z) * 16) + 263))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 896))], placeholder_shared[(((((int)threadIdx.z) * 16) + 8))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 896))], placeholder_shared[(((((int)threadIdx.z) * 16) + 264))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 897))], placeholder_shared[(((((int)threadIdx.z) * 16) + 8))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 897))], placeholder_shared[(((((int)threadIdx.z) * 16) + 264))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 1008))], placeholder_shared[(((((int)threadIdx.z) * 16) + 9))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 1008))], placeholder_shared[(((((int)threadIdx.z) * 16) + 265))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 1009))], placeholder_shared[(((((int)threadIdx.z) * 16) + 9))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 1009))], placeholder_shared[(((((int)threadIdx.z) * 16) + 265))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 1120))], placeholder_shared[(((((int)threadIdx.z) * 16) + 10))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 1120))], placeholder_shared[(((((int)threadIdx.z) * 16) + 266))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 1121))], placeholder_shared[(((((int)threadIdx.z) * 16) + 10))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 1121))], placeholder_shared[(((((int)threadIdx.z) * 16) + 266))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 1232))], placeholder_shared[(((((int)threadIdx.z) * 16) + 11))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 1232))], placeholder_shared[(((((int)threadIdx.z) * 16) + 267))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 1233))], placeholder_shared[(((((int)threadIdx.z) * 16) + 11))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 1233))], placeholder_shared[(((((int)threadIdx.z) * 16) + 267))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 1344))], placeholder_shared[(((((int)threadIdx.z) * 16) + 12))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 1344))], placeholder_shared[(((((int)threadIdx.z) * 16) + 268))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 1345))], placeholder_shared[(((((int)threadIdx.z) * 16) + 12))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 1345))], placeholder_shared[(((((int)threadIdx.z) * 16) + 268))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 1456))], placeholder_shared[(((((int)threadIdx.z) * 16) + 13))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 1456))], placeholder_shared[(((((int)threadIdx.z) * 16) + 269))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 1457))], placeholder_shared[(((((int)threadIdx.z) * 16) + 13))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 1457))], placeholder_shared[(((((int)threadIdx.z) * 16) + 269))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 1568))], placeholder_shared[(((((int)threadIdx.z) * 16) + 14))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 1568))], placeholder_shared[(((((int)threadIdx.z) * 16) + 270))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 1569))], placeholder_shared[(((((int)threadIdx.z) * 16) + 14))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 1569))], placeholder_shared[(((((int)threadIdx.z) * 16) + 270))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 1680))], placeholder_shared[(((((int)threadIdx.z) * 16) + 15))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 1680))], placeholder_shared[(((((int)threadIdx.z) * 16) + 271))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 1681))], placeholder_shared[(((((int)threadIdx.z) * 16) + 15))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 1681))], placeholder_shared[(((((int)threadIdx.z) * 16) + 271))], compute[(3)]);
  }
  T_relu[((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 2)))] = max((compute[(0)] + placeholder2[(((((int)blockIdx.z) * 32) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 2)) + 12544))] = max((compute[(2)] + placeholder2[((((((int)blockIdx.z) * 32) + ((int)threadIdx.z)) + 16))]), 0.000000e+00f);
  T_relu[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 2)) + 1))] = max((compute[(1)] + placeholder2[(((((int)blockIdx.z) * 32) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 2)) + 12545))] = max((compute[(3)] + placeholder2[((((((int)blockIdx.z) * 32) + ((int)threadIdx.z)) + 16))]), 0.000000e+00f);
}

__device__ void fused_nn_global_avg_pool2d_kernel0_device(float* __restrict__ placeholder, float* __restrict__ tensor){
  float tensor1[1];
  tensor1[(0)] = 0.000000e+00f;
  for (int rv0 = 0; rv0 < 7; ++rv0) {
    for (int rv1 = 0; rv1 < 7; ++rv1) {
      if (((int)threadIdx.y) < 1) {
        tensor1[(0)] = (tensor1[(0)] + placeholder[((((((((int)threadIdx.y) * 50176) + (((int)blockIdx.x) * 392)) + (((int)threadIdx.x) * 49)) + (rv0 * 7)) + rv1))]);
      }
    }
  }
  if (((int)threadIdx.y) < 1) {
    tensor[((((((int)threadIdx.y) * 1024) + (((int)blockIdx.x) * 8)) + ((int)threadIdx.x)))] = (tensor1[(0)] * 2.040816e-02f);
  }
}

__device__ void fused_nn_conv2d_add_nn_relu_9_kernel0_device(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2){
  __shared__ float PaddedInput_shared[1024];
  __shared__ float placeholder_shared[36];
  float PaddedInput_shared_local[12];
  float placeholder_shared_local[9];
  float DepthwiseConv2d[2];
  PaddedInput_shared[((((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)))] = (((((1 <= ((((int)blockIdx.y) * 14) + (((((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) & 255) >> 4))) && (((((int)blockIdx.y) * 14) + (((((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) & 255) >> 4)) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + ((((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) & 15)))) && (((((int)blockIdx.x) * 14) + ((((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) & 15)) < 29)) ? placeholder[((((((((((int)blockIdx.z) * 3136) + (((((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) >> 8) * 784)) + (((int)blockIdx.y) * 392)) + ((((((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) & 255) >> 4) * 28)) + (((int)blockIdx.x) * 14)) + ((((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) & 15)) - 29))] : 0.000000e+00f);
  PaddedInput_shared[(((((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) + 392))] = (((((1 <= ((((int)blockIdx.y) * 14) + ((((((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) + 136) & 255) >> 4))) && (((((int)blockIdx.y) * 14) + ((((((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) + 136) & 255) >> 4)) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) + 8) & 15)))) && (((((int)blockIdx.x) * 14) + (((((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) + 8) & 15)) < 29)) ? placeholder[((((((((((int)blockIdx.z) * 3136) + ((((((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) + 392) >> 8) * 784)) + (((int)blockIdx.y) * 392)) + (((((((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) + 136) & 255) >> 4) * 28)) + (((int)blockIdx.x) * 14)) + (((((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) + 8) & 15)) - 29))] : 0.000000e+00f);
  if ((((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) < 240) {
    if (((((int)threadIdx.z) * 14) + ((int)threadIdx.y)) < 35) {
      if (((int)threadIdx.z) < 3) {
        PaddedInput_shared[(((((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) + 784))] = ((((((((int)blockIdx.y) * 14) + (((((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) + 16) >> 4)) < 29) && (1 <= ((((int)blockIdx.x) * 14) + ((((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) & 15)))) && (((((int)blockIdx.x) * 14) + ((((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) & 15)) < 29)) ? placeholder[((((((((((int)blockIdx.z) * 3136) + ((((((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) + 784) >> 8) * 784)) + (((int)blockIdx.y) * 392)) + ((((((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) + 16) >> 4) * 28)) + (((int)blockIdx.x) * 14)) + ((((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) & 15)) - 29))] : 0.000000e+00f);
      }
    }
  }
  if ((((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) < 36) {
    if (((((int)threadIdx.z) * 14) + ((int)threadIdx.y)) < 6) {
      if (((int)threadIdx.z) < 1) {
        placeholder_shared[((((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)))] = placeholder1[(((((((int)threadIdx.z) * 98) + (((int)blockIdx.z) * 36)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)))];
      }
    }
  }
  __syncthreads();
  PaddedInput_shared_local[(0)] = PaddedInput_shared[((((((int)threadIdx.z) * 256) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)))];
  PaddedInput_shared_local[(1)] = PaddedInput_shared[(((((((int)threadIdx.z) * 256) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 1))];
  PaddedInput_shared_local[(2)] = PaddedInput_shared[(((((((int)threadIdx.z) * 256) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 2))];
  PaddedInput_shared_local[(3)] = PaddedInput_shared[(((((((int)threadIdx.z) * 256) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 3))];
  PaddedInput_shared_local[(4)] = PaddedInput_shared[(((((((int)threadIdx.z) * 256) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 16))];
  PaddedInput_shared_local[(5)] = PaddedInput_shared[(((((((int)threadIdx.z) * 256) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 17))];
  PaddedInput_shared_local[(6)] = PaddedInput_shared[(((((((int)threadIdx.z) * 256) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 18))];
  PaddedInput_shared_local[(7)] = PaddedInput_shared[(((((((int)threadIdx.z) * 256) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 19))];
  PaddedInput_shared_local[(8)] = PaddedInput_shared[(((((((int)threadIdx.z) * 256) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 32))];
  PaddedInput_shared_local[(9)] = PaddedInput_shared[(((((((int)threadIdx.z) * 256) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 33))];
  PaddedInput_shared_local[(10)] = PaddedInput_shared[(((((((int)threadIdx.z) * 256) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 34))];
  PaddedInput_shared_local[(11)] = PaddedInput_shared[(((((((int)threadIdx.z) * 256) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 35))];
  placeholder_shared_local[(0)] = placeholder_shared[((((int)threadIdx.z) * 9))];
  placeholder_shared_local[(1)] = placeholder_shared[(((((int)threadIdx.z) * 9) + 1))];
  placeholder_shared_local[(2)] = placeholder_shared[(((((int)threadIdx.z) * 9) + 2))];
  placeholder_shared_local[(3)] = placeholder_shared[(((((int)threadIdx.z) * 9) + 3))];
  placeholder_shared_local[(4)] = placeholder_shared[(((((int)threadIdx.z) * 9) + 4))];
  placeholder_shared_local[(5)] = placeholder_shared[(((((int)threadIdx.z) * 9) + 5))];
  placeholder_shared_local[(6)] = placeholder_shared[(((((int)threadIdx.z) * 9) + 6))];
  placeholder_shared_local[(7)] = placeholder_shared[(((((int)threadIdx.z) * 9) + 7))];
  placeholder_shared_local[(8)] = placeholder_shared[(((((int)threadIdx.z) * 9) + 8))];
  DepthwiseConv2d[(0)] = 0.000000e+00f;
  DepthwiseConv2d[(0)] = __ocml_fma_f32(PaddedInput_shared_local[(0)], placeholder_shared_local[(0)], DepthwiseConv2d[(0)]);
  DepthwiseConv2d[(0)] = __ocml_fma_f32(PaddedInput_shared_local[(1)], placeholder_shared_local[(1)], DepthwiseConv2d[(0)]);
  DepthwiseConv2d[(0)] = __ocml_fma_f32(PaddedInput_shared_local[(2)], placeholder_shared_local[(2)], DepthwiseConv2d[(0)]);
  DepthwiseConv2d[(0)] = __ocml_fma_f32(PaddedInput_shared_local[(4)], placeholder_shared_local[(3)], DepthwiseConv2d[(0)]);
  DepthwiseConv2d[(0)] = __ocml_fma_f32(PaddedInput_shared_local[(5)], placeholder_shared_local[(4)], DepthwiseConv2d[(0)]);
  DepthwiseConv2d[(0)] = __ocml_fma_f32(PaddedInput_shared_local[(6)], placeholder_shared_local[(5)], DepthwiseConv2d[(0)]);
  DepthwiseConv2d[(0)] = __ocml_fma_f32(PaddedInput_shared_local[(8)], placeholder_shared_local[(6)], DepthwiseConv2d[(0)]);
  DepthwiseConv2d[(0)] = __ocml_fma_f32(PaddedInput_shared_local[(9)], placeholder_shared_local[(7)], DepthwiseConv2d[(0)]);
  DepthwiseConv2d[(0)] = __ocml_fma_f32(PaddedInput_shared_local[(10)], placeholder_shared_local[(8)], DepthwiseConv2d[(0)]);
  DepthwiseConv2d[(1)] = 0.000000e+00f;
  DepthwiseConv2d[(1)] = __ocml_fma_f32(PaddedInput_shared_local[(1)], placeholder_shared_local[(0)], DepthwiseConv2d[(1)]);
  DepthwiseConv2d[(1)] = __ocml_fma_f32(PaddedInput_shared_local[(2)], placeholder_shared_local[(1)], DepthwiseConv2d[(1)]);
  DepthwiseConv2d[(1)] = __ocml_fma_f32(PaddedInput_shared_local[(3)], placeholder_shared_local[(2)], DepthwiseConv2d[(1)]);
  DepthwiseConv2d[(1)] = __ocml_fma_f32(PaddedInput_shared_local[(5)], placeholder_shared_local[(3)], DepthwiseConv2d[(1)]);
  DepthwiseConv2d[(1)] = __ocml_fma_f32(PaddedInput_shared_local[(6)], placeholder_shared_local[(4)], DepthwiseConv2d[(1)]);
  DepthwiseConv2d[(1)] = __ocml_fma_f32(PaddedInput_shared_local[(7)], placeholder_shared_local[(5)], DepthwiseConv2d[(1)]);
  DepthwiseConv2d[(1)] = __ocml_fma_f32(PaddedInput_shared_local[(9)], placeholder_shared_local[(6)], DepthwiseConv2d[(1)]);
  DepthwiseConv2d[(1)] = __ocml_fma_f32(PaddedInput_shared_local[(10)], placeholder_shared_local[(7)], DepthwiseConv2d[(1)]);
  DepthwiseConv2d[(1)] = __ocml_fma_f32(PaddedInput_shared_local[(11)], placeholder_shared_local[(8)], DepthwiseConv2d[(1)]);
  T_relu[(((((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 392)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 14)) + (((int)threadIdx.x) * 2)))] = max((DepthwiseConv2d[(0)] + placeholder2[(((((int)blockIdx.z) * 4) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[((((((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 392)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 14)) + (((int)threadIdx.x) * 2)) + 1))] = max((DepthwiseConv2d[(1)] + placeholder2[(((((int)blockIdx.z) * 4) + ((int)threadIdx.z)))]), 0.000000e+00f);
}

__device__ void fused_nn_dense_add_kernel0_device(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_add, float* __restrict__ placeholder2){
  float T_dense_rf[1];
  float red_buf0[1];
  __shared__ float T_dense[1];
  T_dense_rf[(0)] = 0.000000e+00f;
  for (int k_outer = 0; k_outer < 16; ++k_outer) {
    T_dense_rf[(0)] = __ocml_fma_f32(placeholder[(((k_outer * 64) + ((int)threadIdx.x)))], placeholder1[((((((int)blockIdx.x) * 1024) + (k_outer * 64)) + ((int)threadIdx.x)))], T_dense_rf[(0)]);
  }
  unsigned int mask[1];
  float t0[1];
  red_buf0[(0)] = T_dense_rf[(0)];
  ((int*)mask)[(0)] = 0;
  t0[(0)] = __hip_ds_bpermute((((((__mbcnt_hi(-1, __mbcnt_lo(-1, 0)) & 63) + 32) >= 64) ? __mbcnt_hi(-1, __mbcnt_lo(-1, 0)) : (__mbcnt_hi(-1, __mbcnt_lo(-1, 0)) + 32)) << 2), red_buf0[(0)]);
  red_buf0[(0)] = (red_buf0[(0)] + t0[(0)]);
  t0[(0)] = __hip_ds_bpermute((((((__mbcnt_hi(-1, __mbcnt_lo(-1, 0)) & 63) + 16) >= 64) ? __mbcnt_hi(-1, __mbcnt_lo(-1, 0)) : (__mbcnt_hi(-1, __mbcnt_lo(-1, 0)) + 16)) << 2), red_buf0[(0)]);
  red_buf0[(0)] = (red_buf0[(0)] + t0[(0)]);
  t0[(0)] = __hip_ds_bpermute((((((__mbcnt_hi(-1, __mbcnt_lo(-1, 0)) & 63) + 8) >= 64) ? __mbcnt_hi(-1, __mbcnt_lo(-1, 0)) : (__mbcnt_hi(-1, __mbcnt_lo(-1, 0)) + 8)) << 2), red_buf0[(0)]);
  red_buf0[(0)] = (red_buf0[(0)] + t0[(0)]);
  t0[(0)] = __hip_ds_bpermute((((((__mbcnt_hi(-1, __mbcnt_lo(-1, 0)) & 63) + 4) >= 64) ? __mbcnt_hi(-1, __mbcnt_lo(-1, 0)) : (__mbcnt_hi(-1, __mbcnt_lo(-1, 0)) + 4)) << 2), red_buf0[(0)]);
  red_buf0[(0)] = (red_buf0[(0)] + t0[(0)]);
  t0[(0)] = __hip_ds_bpermute((((((__mbcnt_hi(-1, __mbcnt_lo(-1, 0)) & 63) + 2) >= 64) ? __mbcnt_hi(-1, __mbcnt_lo(-1, 0)) : (__mbcnt_hi(-1, __mbcnt_lo(-1, 0)) + 2)) << 2), red_buf0[(0)]);
  red_buf0[(0)] = (red_buf0[(0)] + t0[(0)]);
  t0[(0)] = __hip_ds_bpermute((((((__mbcnt_hi(-1, __mbcnt_lo(-1, 0)) & 63) + 1) >= 64) ? __mbcnt_hi(-1, __mbcnt_lo(-1, 0)) : (__mbcnt_hi(-1, __mbcnt_lo(-1, 0)) + 1)) << 2), red_buf0[(0)]);
  red_buf0[(0)] = (red_buf0[(0)] + t0[(0)]);
  red_buf0[(0)] = __hip_ds_bpermute(((__mbcnt_hi(-1, __mbcnt_lo(-1, 0)) & (~63)) << 2), red_buf0[(0)]);
  if (((int)threadIdx.x) == 0) {
    T_dense[(0)] = red_buf0[(0)];
  }
  if (((int)threadIdx.x) == 0) {
    T_add[(((int)blockIdx.x))] = (T_dense[(0)] + placeholder2[(((int)blockIdx.x))]);
  }
}

__device__ void fused_nn_conv2d_add_nn_relu_15_kernel0_device(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2){
  __shared__ float PaddedInput_shared[3277];
  __shared__ float placeholder_shared[9];
  float PaddedInput_shared_local[9];
  float placeholder_shared_local[9];
  float DepthwiseConv2d[1];
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 784) + (((int)threadIdx.y) * 56)) + ((int)threadIdx.x)) < 3277) {
      if (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 14) + ((int)threadIdx.y)) < 59) {
        PaddedInput_shared[((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 784) + (((int)threadIdx.y) * 56)) + ((int)threadIdx.x)))] = (((1 <= ((((int)blockIdx.y) * 28) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 784) + (((int)threadIdx.y) * 56)) + ((int)threadIdx.x)) / 113))) && (1 <= ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 784) + (((int)threadIdx.y) * 56)) + ((int)threadIdx.x)) % 113))) ? placeholder[((((((((int)blockIdx.z) * 12544) + (((int)blockIdx.y) * 3136)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 784) + (((int)threadIdx.y) * 56)) + ((int)threadIdx.x)) / 113) * 112)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 784) + (((int)threadIdx.y) * 56)) + ((int)threadIdx.x)) % 113)) - 113))] : 0.000000e+00f);
      }
    }
  }
  if (((((int)threadIdx.y) * 56) + ((int)threadIdx.x)) < 9) {
    if (((int)threadIdx.y) < 1) {
      placeholder_shared[(((((int)threadIdx.y) * 56) + ((int)threadIdx.x)))] = placeholder1[((((((int)threadIdx.y) * 56) + (((int)blockIdx.z) * 9)) + ((int)threadIdx.x)))];
    }
  }
  __syncthreads();
  for (int ax2 = 0; ax2 < 3; ++ax2) {
    for (int ax3 = 0; ax3 < 3; ++ax3) {
      PaddedInput_shared_local[(((ax2 * 3) + ax3))] = PaddedInput_shared[(((((((int)threadIdx.y) * 226) + (ax2 * 113)) + (((int)threadIdx.x) * 2)) + ax3))];
    }
  }
  for (int ax21 = 0; ax21 < 3; ++ax21) {
    for (int ax31 = 0; ax31 < 3; ++ax31) {
      placeholder_shared_local[(((ax21 * 3) + ax31))] = placeholder_shared[(((ax21 * 3) + ax31))];
    }
  }
  DepthwiseConv2d[(0)] = 0.000000e+00f;
  for (int di = 0; di < 3; ++di) {
    for (int dj = 0; dj < 3; ++dj) {
      DepthwiseConv2d[(0)] = __ocml_fma_f32(PaddedInput_shared_local[(((di * 3) + dj))], placeholder_shared_local[(((di * 3) + dj))], DepthwiseConv2d[(0)]);
    }
  }
  T_relu[(((((((int)blockIdx.z) * 3136) + (((int)blockIdx.y) * 784)) + (((int)threadIdx.y) * 56)) + ((int)threadIdx.x)))] = max((DepthwiseConv2d[(0)] + placeholder2[(((int)blockIdx.z))]), 0.000000e+00f);
}

__device__ void fused_nn_softmax_kernel0_device(float* __restrict__ placeholder, float* __restrict__ T_softmax_norm){
  float normal_reduce_temp0[1];
  float red_buf0[1];
  float T_softmax_exp[16];
  float normal_reduce_temp01[1];
  float red_buf01[1];
  normal_reduce_temp0[(0)] = -3.402823e+38f;
  for (int k_inner = 0; k_inner < 16; ++k_inner) {
    if (((((int)threadIdx.x) * 16) + k_inner) < 1000) {
      normal_reduce_temp0[(0)] = max(normal_reduce_temp0[(0)], placeholder[(((((int)threadIdx.x) * 16) + k_inner))]);
    }
  }
  unsigned int mask[1];
  float t0[1];
  red_buf0[(0)] = normal_reduce_temp0[(0)];
  ((int*)mask)[(0)] = 0;
  t0[(0)] = __hip_ds_bpermute((((((__mbcnt_hi(-1, __mbcnt_lo(-1, 0)) & 63) + 32) >= 64) ? __mbcnt_hi(-1, __mbcnt_lo(-1, 0)) : (__mbcnt_hi(-1, __mbcnt_lo(-1, 0)) + 32)) << 2), red_buf0[(0)]);
  red_buf0[(0)] = max(red_buf0[(0)], t0[(0)]);
  t0[(0)] = __hip_ds_bpermute((((((__mbcnt_hi(-1, __mbcnt_lo(-1, 0)) & 63) + 16) >= 64) ? __mbcnt_hi(-1, __mbcnt_lo(-1, 0)) : (__mbcnt_hi(-1, __mbcnt_lo(-1, 0)) + 16)) << 2), red_buf0[(0)]);
  red_buf0[(0)] = max(red_buf0[(0)], t0[(0)]);
  t0[(0)] = __hip_ds_bpermute((((((__mbcnt_hi(-1, __mbcnt_lo(-1, 0)) & 63) + 8) >= 64) ? __mbcnt_hi(-1, __mbcnt_lo(-1, 0)) : (__mbcnt_hi(-1, __mbcnt_lo(-1, 0)) + 8)) << 2), red_buf0[(0)]);
  red_buf0[(0)] = max(red_buf0[(0)], t0[(0)]);
  t0[(0)] = __hip_ds_bpermute((((((__mbcnt_hi(-1, __mbcnt_lo(-1, 0)) & 63) + 4) >= 64) ? __mbcnt_hi(-1, __mbcnt_lo(-1, 0)) : (__mbcnt_hi(-1, __mbcnt_lo(-1, 0)) + 4)) << 2), red_buf0[(0)]);
  red_buf0[(0)] = max(red_buf0[(0)], t0[(0)]);
  t0[(0)] = __hip_ds_bpermute((((((__mbcnt_hi(-1, __mbcnt_lo(-1, 0)) & 63) + 2) >= 64) ? __mbcnt_hi(-1, __mbcnt_lo(-1, 0)) : (__mbcnt_hi(-1, __mbcnt_lo(-1, 0)) + 2)) << 2), red_buf0[(0)]);
  red_buf0[(0)] = max(red_buf0[(0)], t0[(0)]);
  t0[(0)] = __hip_ds_bpermute((((((__mbcnt_hi(-1, __mbcnt_lo(-1, 0)) & 63) + 1) >= 64) ? __mbcnt_hi(-1, __mbcnt_lo(-1, 0)) : (__mbcnt_hi(-1, __mbcnt_lo(-1, 0)) + 1)) << 2), red_buf0[(0)]);
  red_buf0[(0)] = max(red_buf0[(0)], t0[(0)]);
  red_buf0[(0)] = __hip_ds_bpermute(((__mbcnt_hi(-1, __mbcnt_lo(-1, 0)) & (~63)) << 2), red_buf0[(0)]);
  for (int i1_inner_outer = 0; i1_inner_outer < 4; ++i1_inner_outer) {
    for (int i1_inner_inner_s = 0; i1_inner_inner_s < 4; ++i1_inner_inner_s) {
      if ((((((int)threadIdx.x) * 16) + (i1_inner_outer * 4)) + i1_inner_inner_s) < 1000) {
        T_softmax_exp[(((i1_inner_outer * 4) + i1_inner_inner_s))] = __ocml_exp_f32((placeholder[((((((int)threadIdx.x) * 16) + (i1_inner_outer * 4)) + i1_inner_inner_s))] - red_buf0[(0)]));
      }
    }
  }
  normal_reduce_temp01[(0)] = 0.000000e+00f;
  for (int k_inner1 = 0; k_inner1 < 16; ++k_inner1) {
    if (((((int)threadIdx.x) * 16) + k_inner1) < 1000) {
      normal_reduce_temp01[(0)] = (normal_reduce_temp01[(0)] + __hip_ds_bpermute(((((int)threadIdx.x) + (__mbcnt_hi(-1, __mbcnt_lo(-1, 0)) & (~63))) << 2), T_softmax_exp[(k_inner1)]));
    }
  }
  unsigned int mask1[1];
  float t01[1];
  red_buf01[(0)] = normal_reduce_temp01[(0)];
  ((int*)mask1)[(0)] = 0;
  t01[(0)] = __hip_ds_bpermute((((((__mbcnt_hi(-1, __mbcnt_lo(-1, 0)) & 63) + 32) >= 64) ? __mbcnt_hi(-1, __mbcnt_lo(-1, 0)) : (__mbcnt_hi(-1, __mbcnt_lo(-1, 0)) + 32)) << 2), red_buf01[(0)]);
  red_buf01[(0)] = (red_buf01[(0)] + t01[(0)]);
  t01[(0)] = __hip_ds_bpermute((((((__mbcnt_hi(-1, __mbcnt_lo(-1, 0)) & 63) + 16) >= 64) ? __mbcnt_hi(-1, __mbcnt_lo(-1, 0)) : (__mbcnt_hi(-1, __mbcnt_lo(-1, 0)) + 16)) << 2), red_buf01[(0)]);
  red_buf01[(0)] = (red_buf01[(0)] + t01[(0)]);
  t01[(0)] = __hip_ds_bpermute((((((__mbcnt_hi(-1, __mbcnt_lo(-1, 0)) & 63) + 8) >= 64) ? __mbcnt_hi(-1, __mbcnt_lo(-1, 0)) : (__mbcnt_hi(-1, __mbcnt_lo(-1, 0)) + 8)) << 2), red_buf01[(0)]);
  red_buf01[(0)] = (red_buf01[(0)] + t01[(0)]);
  t01[(0)] = __hip_ds_bpermute((((((__mbcnt_hi(-1, __mbcnt_lo(-1, 0)) & 63) + 4) >= 64) ? __mbcnt_hi(-1, __mbcnt_lo(-1, 0)) : (__mbcnt_hi(-1, __mbcnt_lo(-1, 0)) + 4)) << 2), red_buf01[(0)]);
  red_buf01[(0)] = (red_buf01[(0)] + t01[(0)]);
  t01[(0)] = __hip_ds_bpermute((((((__mbcnt_hi(-1, __mbcnt_lo(-1, 0)) & 63) + 2) >= 64) ? __mbcnt_hi(-1, __mbcnt_lo(-1, 0)) : (__mbcnt_hi(-1, __mbcnt_lo(-1, 0)) + 2)) << 2), red_buf01[(0)]);
  red_buf01[(0)] = (red_buf01[(0)] + t01[(0)]);
  t01[(0)] = __hip_ds_bpermute((((((__mbcnt_hi(-1, __mbcnt_lo(-1, 0)) & 63) + 1) >= 64) ? __mbcnt_hi(-1, __mbcnt_lo(-1, 0)) : (__mbcnt_hi(-1, __mbcnt_lo(-1, 0)) + 1)) << 2), red_buf01[(0)]);
  red_buf01[(0)] = (red_buf01[(0)] + t01[(0)]);
  red_buf01[(0)] = __hip_ds_bpermute(((__mbcnt_hi(-1, __mbcnt_lo(-1, 0)) & (~63)) << 2), red_buf01[(0)]);
  for (int i1_inner_outer1 = 0; i1_inner_outer1 < 4; ++i1_inner_outer1) {
    for (int i1_inner_inner_s1 = 0; i1_inner_inner_s1 < 4; ++i1_inner_inner_s1) {
      if ((((((int)threadIdx.x) * 16) + (i1_inner_outer1 * 4)) + i1_inner_inner_s1) < 1000) {
        T_softmax_norm[((((((int)threadIdx.x) * 16) + (i1_inner_outer1 * 4)) + i1_inner_inner_s1))] = (__hip_ds_bpermute(((((int)threadIdx.x) + (__mbcnt_hi(-1, __mbcnt_lo(-1, 0)) & (~63))) << 2), T_softmax_exp[(((i1_inner_outer1 * 4) + i1_inner_inner_s1))]) / red_buf01[(0)]);
      }
    }
  }
}

__device__ void fused_nn_conv2d_add_nn_relu_3_kernel0_device(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2){
  __shared__ float PaddedInput_shared[1800];
  __shared__ float placeholder_shared[72];
  float PaddedInput_shared_local[9];
  float placeholder_shared_local[9];
  float DepthwiseConv2d[1];
  PaddedInput_shared[((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)))] = (((15 <= ((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 225)) && (1 <= ((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 15))) ? placeholder[((((((((int)blockIdx.z) * 1568) + (((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) / 225) * 196)) + ((((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 225) / 15) * 14)) + ((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 15)) - 15))] : 0.000000e+00f);
  PaddedInput_shared[(((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) + 392))] = (((15 <= (((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) + 167) % 225)) && (1 <= (((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) + 2) % 15))) ? placeholder[((((((((int)blockIdx.z) * 1568) + ((((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) + 392) / 225) * 196)) + (((((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) + 167) % 225) / 15) * 14)) + (((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) + 2) % 15)) - 15))] : 0.000000e+00f);
  PaddedInput_shared[(((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) + 784))] = (((15 <= (((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) + 109) % 225)) && (1 <= (((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) + 4) % 15))) ? placeholder[((((((((int)blockIdx.z) * 1568) + ((((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) + 784) / 225) * 196)) + (((((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) + 109) % 225) / 15) * 14)) + (((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) + 4) % 15)) - 15))] : 0.000000e+00f);
  PaddedInput_shared[(((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) + 1176))] = (((15 <= (((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) + 51) % 225)) && (1 <= (((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) + 6) % 15))) ? placeholder[((((((((int)blockIdx.z) * 1568) + ((((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) + 1176) / 225) * 196)) + (((((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) + 51) % 225) / 15) * 14)) + (((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) + 6) % 15)) - 15))] : 0.000000e+00f);
  if ((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) < 232) {
    if (((((int)threadIdx.z) * 7) + ((int)threadIdx.y)) < 34) {
      if (((int)threadIdx.z) < 5) {
        PaddedInput_shared[(((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) + 1568))] = (((15 <= (((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) + 218) % 225)) && (1 <= (((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) + 8) % 15))) ? placeholder[((((((((int)blockIdx.z) * 1568) + ((((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) + 1568) / 225) * 196)) + (((((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) + 218) % 225) / 15) * 14)) + (((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) + 8) % 15)) - 15))] : 0.000000e+00f);
      }
    }
  }
  if ((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) < 72) {
    if (((((int)threadIdx.z) * 7) + ((int)threadIdx.y)) < 11) {
      if (((int)threadIdx.z) < 2) {
        placeholder_shared[((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)))] = placeholder1[(((((((int)blockIdx.z) * 72) + (((int)threadIdx.z) * 49)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)))];
      }
    }
  }
  __syncthreads();
  PaddedInput_shared_local[(0)] = PaddedInput_shared[((((((int)threadIdx.z) * 225) + (((int)threadIdx.y) * 30)) + (((int)threadIdx.x) * 2)))];
  PaddedInput_shared_local[(1)] = PaddedInput_shared[(((((((int)threadIdx.z) * 225) + (((int)threadIdx.y) * 30)) + (((int)threadIdx.x) * 2)) + 1))];
  PaddedInput_shared_local[(2)] = PaddedInput_shared[(((((((int)threadIdx.z) * 225) + (((int)threadIdx.y) * 30)) + (((int)threadIdx.x) * 2)) + 2))];
  PaddedInput_shared_local[(3)] = PaddedInput_shared[(((((((int)threadIdx.z) * 225) + (((int)threadIdx.y) * 30)) + (((int)threadIdx.x) * 2)) + 15))];
  PaddedInput_shared_local[(4)] = PaddedInput_shared[(((((((int)threadIdx.z) * 225) + (((int)threadIdx.y) * 30)) + (((int)threadIdx.x) * 2)) + 16))];
  PaddedInput_shared_local[(5)] = PaddedInput_shared[(((((((int)threadIdx.z) * 225) + (((int)threadIdx.y) * 30)) + (((int)threadIdx.x) * 2)) + 17))];
  PaddedInput_shared_local[(6)] = PaddedInput_shared[(((((((int)threadIdx.z) * 225) + (((int)threadIdx.y) * 30)) + (((int)threadIdx.x) * 2)) + 30))];
  PaddedInput_shared_local[(7)] = PaddedInput_shared[(((((((int)threadIdx.z) * 225) + (((int)threadIdx.y) * 30)) + (((int)threadIdx.x) * 2)) + 31))];
  PaddedInput_shared_local[(8)] = PaddedInput_shared[(((((((int)threadIdx.z) * 225) + (((int)threadIdx.y) * 30)) + (((int)threadIdx.x) * 2)) + 32))];
  placeholder_shared_local[(0)] = placeholder_shared[((((int)threadIdx.z) * 9))];
  placeholder_shared_local[(1)] = placeholder_shared[(((((int)threadIdx.z) * 9) + 1))];
  placeholder_shared_local[(2)] = placeholder_shared[(((((int)threadIdx.z) * 9) + 2))];
  placeholder_shared_local[(3)] = placeholder_shared[(((((int)threadIdx.z) * 9) + 3))];
  placeholder_shared_local[(4)] = placeholder_shared[(((((int)threadIdx.z) * 9) + 4))];
  placeholder_shared_local[(5)] = placeholder_shared[(((((int)threadIdx.z) * 9) + 5))];
  placeholder_shared_local[(6)] = placeholder_shared[(((((int)threadIdx.z) * 9) + 6))];
  placeholder_shared_local[(7)] = placeholder_shared[(((((int)threadIdx.z) * 9) + 7))];
  placeholder_shared_local[(8)] = placeholder_shared[(((((int)threadIdx.z) * 9) + 8))];
  DepthwiseConv2d[(0)] = 0.000000e+00f;
  DepthwiseConv2d[(0)] = __ocml_fma_f32(PaddedInput_shared_local[(0)], placeholder_shared_local[(0)], DepthwiseConv2d[(0)]);
  DepthwiseConv2d[(0)] = __ocml_fma_f32(PaddedInput_shared_local[(1)], placeholder_shared_local[(1)], DepthwiseConv2d[(0)]);
  DepthwiseConv2d[(0)] = __ocml_fma_f32(PaddedInput_shared_local[(2)], placeholder_shared_local[(2)], DepthwiseConv2d[(0)]);
  DepthwiseConv2d[(0)] = __ocml_fma_f32(PaddedInput_shared_local[(3)], placeholder_shared_local[(3)], DepthwiseConv2d[(0)]);
  DepthwiseConv2d[(0)] = __ocml_fma_f32(PaddedInput_shared_local[(4)], placeholder_shared_local[(4)], DepthwiseConv2d[(0)]);
  DepthwiseConv2d[(0)] = __ocml_fma_f32(PaddedInput_shared_local[(5)], placeholder_shared_local[(5)], DepthwiseConv2d[(0)]);
  DepthwiseConv2d[(0)] = __ocml_fma_f32(PaddedInput_shared_local[(6)], placeholder_shared_local[(6)], DepthwiseConv2d[(0)]);
  DepthwiseConv2d[(0)] = __ocml_fma_f32(PaddedInput_shared_local[(7)], placeholder_shared_local[(7)], DepthwiseConv2d[(0)]);
  DepthwiseConv2d[(0)] = __ocml_fma_f32(PaddedInput_shared_local[(8)], placeholder_shared_local[(8)], DepthwiseConv2d[(0)]);
  T_relu[(((((((int)blockIdx.z) * 392) + (((int)threadIdx.z) * 49)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)))] = max((DepthwiseConv2d[(0)] + placeholder2[(((((int)blockIdx.z) * 8) + ((int)threadIdx.z)))]), 0.000000e+00f);
}

__device__ void fused_nn_conv2d_add_nn_relu_4_kernel0_device(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2){
  float compute[2];
  __shared__ float pad_temp_shared[784];
  __shared__ float placeholder_shared[256];
  compute[(0)] = 0.000000e+00f;
  compute[(1)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 32; ++rc_outer) {
    __syncthreads();
    pad_temp_shared[((((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)))] = placeholder[((((((((rc_outer * 3136) + (((int)threadIdx.z) * 392)) + ((((((int)threadIdx.y) * 2) + ((((int)threadIdx.x) * 2) / 7)) / 7) * 196)) + (((int)blockIdx.y) * 98)) + ((((((int)threadIdx.y) * 2) + ((((int)threadIdx.x) * 2) / 7)) % 7) * 14)) + (((int)blockIdx.x) * 7)) + ((((int)threadIdx.x) * 2) % 7)))];
    pad_temp_shared[(((((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)) + 1))] = placeholder[((((((((rc_outer * 3136) + (((int)threadIdx.z) * 392)) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) / 7)) / 7) * 196)) + (((int)blockIdx.y) * 98)) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) / 7)) % 7) * 14)) + (((int)blockIdx.x) * 7)) + (((((int)threadIdx.x) * 2) + 1) % 7)))];
    if (((((int)threadIdx.z) * 2) + (((((int)threadIdx.y) * 5) + ((int)threadIdx.x)) >> 4)) < 16) {
      if ((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 5)) + ((int)threadIdx.x)) < 256) {
        if (((((int)threadIdx.y) * 5) + ((int)threadIdx.x)) < 32) {
          if (((int)threadIdx.x) < 5) {
            placeholder_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 5)) + ((int)threadIdx.x)))] = placeholder1[((((((((int)blockIdx.z) * 8192) + (((int)threadIdx.z) * 1024)) + ((((((int)threadIdx.y) * 5) + ((int)threadIdx.x)) >> 4) * 512)) + (rc_outer * 16)) + (((((int)threadIdx.y) * 5) + ((int)threadIdx.x)) & 15)))];
          }
        }
      }
    }
    __syncthreads();
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 7) + ((int)threadIdx.x)))], placeholder_shared[((((int)threadIdx.z) * 16))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 7) + ((int)threadIdx.x)))], placeholder_shared[(((((int)threadIdx.z) * 16) + 128))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 49))], placeholder_shared[(((((int)threadIdx.z) * 16) + 1))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 49))], placeholder_shared[(((((int)threadIdx.z) * 16) + 129))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 98))], placeholder_shared[(((((int)threadIdx.z) * 16) + 2))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 98))], placeholder_shared[(((((int)threadIdx.z) * 16) + 130))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 147))], placeholder_shared[(((((int)threadIdx.z) * 16) + 3))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 147))], placeholder_shared[(((((int)threadIdx.z) * 16) + 131))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 196))], placeholder_shared[(((((int)threadIdx.z) * 16) + 4))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 196))], placeholder_shared[(((((int)threadIdx.z) * 16) + 132))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 245))], placeholder_shared[(((((int)threadIdx.z) * 16) + 5))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 245))], placeholder_shared[(((((int)threadIdx.z) * 16) + 133))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 294))], placeholder_shared[(((((int)threadIdx.z) * 16) + 6))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 294))], placeholder_shared[(((((int)threadIdx.z) * 16) + 134))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 343))], placeholder_shared[(((((int)threadIdx.z) * 16) + 7))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 343))], placeholder_shared[(((((int)threadIdx.z) * 16) + 135))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 392))], placeholder_shared[(((((int)threadIdx.z) * 16) + 8))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 392))], placeholder_shared[(((((int)threadIdx.z) * 16) + 136))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 441))], placeholder_shared[(((((int)threadIdx.z) * 16) + 9))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 441))], placeholder_shared[(((((int)threadIdx.z) * 16) + 137))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 490))], placeholder_shared[(((((int)threadIdx.z) * 16) + 10))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 490))], placeholder_shared[(((((int)threadIdx.z) * 16) + 138))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 539))], placeholder_shared[(((((int)threadIdx.z) * 16) + 11))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 539))], placeholder_shared[(((((int)threadIdx.z) * 16) + 139))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 588))], placeholder_shared[(((((int)threadIdx.z) * 16) + 12))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 588))], placeholder_shared[(((((int)threadIdx.z) * 16) + 140))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 637))], placeholder_shared[(((((int)threadIdx.z) * 16) + 13))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 637))], placeholder_shared[(((((int)threadIdx.z) * 16) + 141))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 686))], placeholder_shared[(((((int)threadIdx.z) * 16) + 14))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 686))], placeholder_shared[(((((int)threadIdx.z) * 16) + 142))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 735))], placeholder_shared[(((((int)threadIdx.z) * 16) + 15))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 735))], placeholder_shared[(((((int)threadIdx.z) * 16) + 143))], compute[(1)]);
  }
  T_relu[(((((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)))] = max((compute[(0)] + placeholder2[(((((int)blockIdx.z) * 16) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[((((((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 1568))] = max((compute[(1)] + placeholder2[((((((int)blockIdx.z) * 16) + ((int)threadIdx.z)) + 8))]), 0.000000e+00f);
}

__device__ void fused_nn_conv2d_add_nn_relu_kernel0_device(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2){
  float compute[1];
  __shared__ float pad_temp_shared[1568];
  __shared__ float placeholder_shared[512];
  compute[(0)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 32; ++rc_outer) {
    __syncthreads();
    pad_temp_shared[((((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)))] = placeholder[(((((rc_outer * 1568) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)))];
    pad_temp_shared[(((((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)) + 1))] = placeholder[(((((rc_outer * 1568) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 14)) + ((((int)threadIdx.x) * 2) + 1)))];
    if (((((((int)threadIdx.y) * 5) + ((int)threadIdx.x)) >> 5) + ((int)threadIdx.z)) < 16) {
      if ((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 5)) + ((int)threadIdx.x)) < 512) {
        if (((((int)threadIdx.y) * 5) + ((int)threadIdx.x)) < 32) {
          if (((int)threadIdx.x) < 5) {
            placeholder_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 5)) + ((int)threadIdx.x)))] = placeholder1[((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 1024)) + (rc_outer * 32)) + (((int)threadIdx.y) * 5)) + ((int)threadIdx.x)))];
          }
        }
      }
    }
    __syncthreads();
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 7) + ((int)threadIdx.x)))], placeholder_shared[((((int)threadIdx.z) * 32))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 49))], placeholder_shared[(((((int)threadIdx.z) * 32) + 1))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 98))], placeholder_shared[(((((int)threadIdx.z) * 32) + 2))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 147))], placeholder_shared[(((((int)threadIdx.z) * 32) + 3))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 196))], placeholder_shared[(((((int)threadIdx.z) * 32) + 4))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 245))], placeholder_shared[(((((int)threadIdx.z) * 32) + 5))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 294))], placeholder_shared[(((((int)threadIdx.z) * 32) + 6))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 343))], placeholder_shared[(((((int)threadIdx.z) * 32) + 7))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 392))], placeholder_shared[(((((int)threadIdx.z) * 32) + 8))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 441))], placeholder_shared[(((((int)threadIdx.z) * 32) + 9))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 490))], placeholder_shared[(((((int)threadIdx.z) * 32) + 10))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 539))], placeholder_shared[(((((int)threadIdx.z) * 32) + 11))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 588))], placeholder_shared[(((((int)threadIdx.z) * 32) + 12))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 637))], placeholder_shared[(((((int)threadIdx.z) * 32) + 13))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 686))], placeholder_shared[(((((int)threadIdx.z) * 32) + 14))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 735))], placeholder_shared[(((((int)threadIdx.z) * 32) + 15))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 784))], placeholder_shared[(((((int)threadIdx.z) * 32) + 16))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 833))], placeholder_shared[(((((int)threadIdx.z) * 32) + 17))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 882))], placeholder_shared[(((((int)threadIdx.z) * 32) + 18))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 931))], placeholder_shared[(((((int)threadIdx.z) * 32) + 19))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 980))], placeholder_shared[(((((int)threadIdx.z) * 32) + 20))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1029))], placeholder_shared[(((((int)threadIdx.z) * 32) + 21))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1078))], placeholder_shared[(((((int)threadIdx.z) * 32) + 22))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1127))], placeholder_shared[(((((int)threadIdx.z) * 32) + 23))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1176))], placeholder_shared[(((((int)threadIdx.z) * 32) + 24))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1225))], placeholder_shared[(((((int)threadIdx.z) * 32) + 25))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1274))], placeholder_shared[(((((int)threadIdx.z) * 32) + 26))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1323))], placeholder_shared[(((((int)threadIdx.z) * 32) + 27))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1372))], placeholder_shared[(((((int)threadIdx.z) * 32) + 28))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1421))], placeholder_shared[(((((int)threadIdx.z) * 32) + 29))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1470))], placeholder_shared[(((((int)threadIdx.z) * 32) + 30))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1519))], placeholder_shared[(((((int)threadIdx.z) * 32) + 31))], compute[(0)]);
  }
  T_relu[(((((((int)blockIdx.z) * 784) + (((int)threadIdx.z) * 49)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)))] = max((compute[(0)] + placeholder2[(((((int)blockIdx.z) * 16) + ((int)threadIdx.z)))]), 0.000000e+00f);
}

__device__ void fused_nn_conv2d_add_nn_relu_1_kernel0_device(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2){
  __shared__ float PaddedInput_shared[324];
  __shared__ float placeholder_shared[36];
  float PaddedInput_shared_local[9];
  float placeholder_shared_local[9];
  float DepthwiseConv2d[1];
  PaddedInput_shared[((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)))] = (((((9 <= ((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 81)) && (((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 81) < 72)) && (1 <= ((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 9))) && (((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 9) < 8)) ? placeholder[((((((((int)blockIdx.z) * 196) + (((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) / 81) * 49)) + ((((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 81) / 9) * 7)) + ((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 9)) - 8))] : 0.000000e+00f);
  if ((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) < 128) {
    if (((((int)threadIdx.z) * 7) + ((int)threadIdx.y)) < 19) {
      if (((int)threadIdx.z) < 3) {
        PaddedInput_shared[(((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) + 196))] = (((((9 <= (((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) + 34) % 81)) && ((((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) + 34) % 81) < 72)) && (1 <= (((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) + 7) % 9))) && ((((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) + 7) % 9) < 8)) ? placeholder[((((((((int)blockIdx.z) * 196) + ((((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) + 196) / 81) * 49)) + (((((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) + 34) % 81) / 9) * 7)) + (((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) + 7) % 9)) - 8))] : 0.000000e+00f);
      }
    }
  }
  if ((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) < 36) {
    if (((((int)threadIdx.z) * 7) + ((int)threadIdx.y)) < 6) {
      if (((int)threadIdx.z) < 1) {
        placeholder_shared[((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)))] = placeholder1[(((((((int)threadIdx.z) * 49) + (((int)blockIdx.z) * 36)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)))];
      }
    }
  }
  __syncthreads();
  PaddedInput_shared_local[(0)] = PaddedInput_shared[((((((int)threadIdx.z) * 81) + (((int)threadIdx.y) * 9)) + ((int)threadIdx.x)))];
  PaddedInput_shared_local[(1)] = PaddedInput_shared[(((((((int)threadIdx.z) * 81) + (((int)threadIdx.y) * 9)) + ((int)threadIdx.x)) + 1))];
  PaddedInput_shared_local[(2)] = PaddedInput_shared[(((((((int)threadIdx.z) * 81) + (((int)threadIdx.y) * 9)) + ((int)threadIdx.x)) + 2))];
  PaddedInput_shared_local[(3)] = PaddedInput_shared[(((((((int)threadIdx.z) * 81) + (((int)threadIdx.y) * 9)) + ((int)threadIdx.x)) + 9))];
  PaddedInput_shared_local[(4)] = PaddedInput_shared[(((((((int)threadIdx.z) * 81) + (((int)threadIdx.y) * 9)) + ((int)threadIdx.x)) + 10))];
  PaddedInput_shared_local[(5)] = PaddedInput_shared[(((((((int)threadIdx.z) * 81) + (((int)threadIdx.y) * 9)) + ((int)threadIdx.x)) + 11))];
  PaddedInput_shared_local[(6)] = PaddedInput_shared[(((((((int)threadIdx.z) * 81) + (((int)threadIdx.y) * 9)) + ((int)threadIdx.x)) + 18))];
  PaddedInput_shared_local[(7)] = PaddedInput_shared[(((((((int)threadIdx.z) * 81) + (((int)threadIdx.y) * 9)) + ((int)threadIdx.x)) + 19))];
  PaddedInput_shared_local[(8)] = PaddedInput_shared[(((((((int)threadIdx.z) * 81) + (((int)threadIdx.y) * 9)) + ((int)threadIdx.x)) + 20))];
  placeholder_shared_local[(0)] = placeholder_shared[((((int)threadIdx.z) * 9))];
  placeholder_shared_local[(1)] = placeholder_shared[(((((int)threadIdx.z) * 9) + 1))];
  placeholder_shared_local[(2)] = placeholder_shared[(((((int)threadIdx.z) * 9) + 2))];
  placeholder_shared_local[(3)] = placeholder_shared[(((((int)threadIdx.z) * 9) + 3))];
  placeholder_shared_local[(4)] = placeholder_shared[(((((int)threadIdx.z) * 9) + 4))];
  placeholder_shared_local[(5)] = placeholder_shared[(((((int)threadIdx.z) * 9) + 5))];
  placeholder_shared_local[(6)] = placeholder_shared[(((((int)threadIdx.z) * 9) + 6))];
  placeholder_shared_local[(7)] = placeholder_shared[(((((int)threadIdx.z) * 9) + 7))];
  placeholder_shared_local[(8)] = placeholder_shared[(((((int)threadIdx.z) * 9) + 8))];
  DepthwiseConv2d[(0)] = 0.000000e+00f;
  DepthwiseConv2d[(0)] = __ocml_fma_f32(PaddedInput_shared_local[(0)], placeholder_shared_local[(0)], DepthwiseConv2d[(0)]);
  DepthwiseConv2d[(0)] = __ocml_fma_f32(PaddedInput_shared_local[(1)], placeholder_shared_local[(1)], DepthwiseConv2d[(0)]);
  DepthwiseConv2d[(0)] = __ocml_fma_f32(PaddedInput_shared_local[(2)], placeholder_shared_local[(2)], DepthwiseConv2d[(0)]);
  DepthwiseConv2d[(0)] = __ocml_fma_f32(PaddedInput_shared_local[(3)], placeholder_shared_local[(3)], DepthwiseConv2d[(0)]);
  DepthwiseConv2d[(0)] = __ocml_fma_f32(PaddedInput_shared_local[(4)], placeholder_shared_local[(4)], DepthwiseConv2d[(0)]);
  DepthwiseConv2d[(0)] = __ocml_fma_f32(PaddedInput_shared_local[(5)], placeholder_shared_local[(5)], DepthwiseConv2d[(0)]);
  DepthwiseConv2d[(0)] = __ocml_fma_f32(PaddedInput_shared_local[(6)], placeholder_shared_local[(6)], DepthwiseConv2d[(0)]);
  DepthwiseConv2d[(0)] = __ocml_fma_f32(PaddedInput_shared_local[(7)], placeholder_shared_local[(7)], DepthwiseConv2d[(0)]);
  DepthwiseConv2d[(0)] = __ocml_fma_f32(PaddedInput_shared_local[(8)], placeholder_shared_local[(8)], DepthwiseConv2d[(0)]);
  T_relu[(((((((int)blockIdx.z) * 196) + (((int)threadIdx.z) * 49)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)))] = max((DepthwiseConv2d[(0)] + placeholder2[(((((int)blockIdx.z) * 4) + ((int)threadIdx.z)))]), 0.000000e+00f);
}

__device__ void fused_nn_conv2d_add_nn_relu_5_kernel0_device(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2){
  __shared__ float PaddedInput_shared[512];
  __shared__ float placeholder_shared[18];
  float PaddedInput_shared_local[12];
  float placeholder_shared_local[9];
  float DepthwiseConv2d[2];
  PaddedInput_shared[((((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)))] = ((((16 <= (((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x))) && (1 <= ((((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) & 15))) && (((((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) & 15) < 15)) ? placeholder[(((((((int)blockIdx.z) * 392) + (((((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) >> 4) * 14)) + ((((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) & 15)) - 15))] : 0.000000e+00f);
  PaddedInput_shared[(((((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) + 196))] = (((((16 <= (((((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) + 196) & 255)) && ((((((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) + 196) & 255) < 240)) && (1 <= (((((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) + 4) & 15))) && ((((((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) + 4) & 15) < 15)) ? placeholder[((((((((int)blockIdx.z) * 392) + ((((((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) + 196) >> 8) * 196)) + (((((((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) + 196) & 255) >> 4) * 14)) + (((((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) + 4) & 15)) - 15))] : 0.000000e+00f);
  if ((((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) < 120) {
    if (((((int)threadIdx.z) * 14) + ((int)threadIdx.y)) < 18) {
      PaddedInput_shared[(((((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) + 392))] = (((((((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) < 104) && (1 <= (((((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) + 8) & 15))) && ((((((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) + 8) & 15) < 15)) ? placeholder[((((((((int)blockIdx.z) * 392) + ((((((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) + 392) >> 8) * 196)) + ((((((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) + 136) >> 4) * 14)) + (((((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) + 8) & 15)) - 15))] : 0.000000e+00f);
    }
  }
  if ((((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) < 18) {
    if (((((int)threadIdx.z) * 14) + ((int)threadIdx.y)) < 3) {
      if (((int)threadIdx.z) < 1) {
        placeholder_shared[((((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)))] = placeholder1[(((((((int)threadIdx.z) * 98) + (((int)blockIdx.z) * 18)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)))];
      }
    }
  }
  __syncthreads();
  PaddedInput_shared_local[(0)] = PaddedInput_shared[((((((int)threadIdx.z) * 256) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)))];
  PaddedInput_shared_local[(1)] = PaddedInput_shared[(((((((int)threadIdx.z) * 256) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 1))];
  PaddedInput_shared_local[(2)] = PaddedInput_shared[(((((((int)threadIdx.z) * 256) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 2))];
  PaddedInput_shared_local[(3)] = PaddedInput_shared[(((((((int)threadIdx.z) * 256) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 3))];
  PaddedInput_shared_local[(4)] = PaddedInput_shared[(((((((int)threadIdx.z) * 256) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 16))];
  PaddedInput_shared_local[(5)] = PaddedInput_shared[(((((((int)threadIdx.z) * 256) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 17))];
  PaddedInput_shared_local[(6)] = PaddedInput_shared[(((((((int)threadIdx.z) * 256) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 18))];
  PaddedInput_shared_local[(7)] = PaddedInput_shared[(((((((int)threadIdx.z) * 256) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 19))];
  PaddedInput_shared_local[(8)] = PaddedInput_shared[(((((((int)threadIdx.z) * 256) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 32))];
  PaddedInput_shared_local[(9)] = PaddedInput_shared[(((((((int)threadIdx.z) * 256) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 33))];
  PaddedInput_shared_local[(10)] = PaddedInput_shared[(((((((int)threadIdx.z) * 256) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 34))];
  PaddedInput_shared_local[(11)] = PaddedInput_shared[(((((((int)threadIdx.z) * 256) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 35))];
  placeholder_shared_local[(0)] = placeholder_shared[((((int)threadIdx.z) * 9))];
  placeholder_shared_local[(1)] = placeholder_shared[(((((int)threadIdx.z) * 9) + 1))];
  placeholder_shared_local[(2)] = placeholder_shared[(((((int)threadIdx.z) * 9) + 2))];
  placeholder_shared_local[(3)] = placeholder_shared[(((((int)threadIdx.z) * 9) + 3))];
  placeholder_shared_local[(4)] = placeholder_shared[(((((int)threadIdx.z) * 9) + 4))];
  placeholder_shared_local[(5)] = placeholder_shared[(((((int)threadIdx.z) * 9) + 5))];
  placeholder_shared_local[(6)] = placeholder_shared[(((((int)threadIdx.z) * 9) + 6))];
  placeholder_shared_local[(7)] = placeholder_shared[(((((int)threadIdx.z) * 9) + 7))];
  placeholder_shared_local[(8)] = placeholder_shared[(((((int)threadIdx.z) * 9) + 8))];
  DepthwiseConv2d[(0)] = 0.000000e+00f;
  DepthwiseConv2d[(0)] = __ocml_fma_f32(PaddedInput_shared_local[(0)], placeholder_shared_local[(0)], DepthwiseConv2d[(0)]);
  DepthwiseConv2d[(0)] = __ocml_fma_f32(PaddedInput_shared_local[(1)], placeholder_shared_local[(1)], DepthwiseConv2d[(0)]);
  DepthwiseConv2d[(0)] = __ocml_fma_f32(PaddedInput_shared_local[(2)], placeholder_shared_local[(2)], DepthwiseConv2d[(0)]);
  DepthwiseConv2d[(0)] = __ocml_fma_f32(PaddedInput_shared_local[(4)], placeholder_shared_local[(3)], DepthwiseConv2d[(0)]);
  DepthwiseConv2d[(0)] = __ocml_fma_f32(PaddedInput_shared_local[(5)], placeholder_shared_local[(4)], DepthwiseConv2d[(0)]);
  DepthwiseConv2d[(0)] = __ocml_fma_f32(PaddedInput_shared_local[(6)], placeholder_shared_local[(5)], DepthwiseConv2d[(0)]);
  DepthwiseConv2d[(0)] = __ocml_fma_f32(PaddedInput_shared_local[(8)], placeholder_shared_local[(6)], DepthwiseConv2d[(0)]);
  DepthwiseConv2d[(0)] = __ocml_fma_f32(PaddedInput_shared_local[(9)], placeholder_shared_local[(7)], DepthwiseConv2d[(0)]);
  DepthwiseConv2d[(0)] = __ocml_fma_f32(PaddedInput_shared_local[(10)], placeholder_shared_local[(8)], DepthwiseConv2d[(0)]);
  DepthwiseConv2d[(1)] = 0.000000e+00f;
  DepthwiseConv2d[(1)] = __ocml_fma_f32(PaddedInput_shared_local[(1)], placeholder_shared_local[(0)], DepthwiseConv2d[(1)]);
  DepthwiseConv2d[(1)] = __ocml_fma_f32(PaddedInput_shared_local[(2)], placeholder_shared_local[(1)], DepthwiseConv2d[(1)]);
  DepthwiseConv2d[(1)] = __ocml_fma_f32(PaddedInput_shared_local[(3)], placeholder_shared_local[(2)], DepthwiseConv2d[(1)]);
  DepthwiseConv2d[(1)] = __ocml_fma_f32(PaddedInput_shared_local[(5)], placeholder_shared_local[(3)], DepthwiseConv2d[(1)]);
  DepthwiseConv2d[(1)] = __ocml_fma_f32(PaddedInput_shared_local[(6)], placeholder_shared_local[(4)], DepthwiseConv2d[(1)]);
  DepthwiseConv2d[(1)] = __ocml_fma_f32(PaddedInput_shared_local[(7)], placeholder_shared_local[(5)], DepthwiseConv2d[(1)]);
  DepthwiseConv2d[(1)] = __ocml_fma_f32(PaddedInput_shared_local[(9)], placeholder_shared_local[(6)], DepthwiseConv2d[(1)]);
  DepthwiseConv2d[(1)] = __ocml_fma_f32(PaddedInput_shared_local[(10)], placeholder_shared_local[(7)], DepthwiseConv2d[(1)]);
  DepthwiseConv2d[(1)] = __ocml_fma_f32(PaddedInput_shared_local[(11)], placeholder_shared_local[(8)], DepthwiseConv2d[(1)]);
  T_relu[(((((((int)blockIdx.z) * 392) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)))] = max((DepthwiseConv2d[(0)] + placeholder2[(((((int)blockIdx.z) * 2) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 392) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)) + 1))] = max((DepthwiseConv2d[(1)] + placeholder2[(((((int)blockIdx.z) * 2) + ((int)threadIdx.z)))]), 0.000000e+00f);
}

__device__ void fused_nn_conv2d_add_nn_relu_10_kernel0_device(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2){
  float compute[4];
  __shared__ float pad_temp_shared[1792];
  __shared__ float placeholder_shared[512];
  compute[(0)] = 0.000000e+00f;
  compute[(2)] = 0.000000e+00f;
  compute[(1)] = 0.000000e+00f;
  compute[(3)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 8; ++rc_outer) {
    __syncthreads();
    pad_temp_shared[((((((int)threadIdx.z) * 112) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 2)))] = placeholder[((((((rc_outer * 12544) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 2)))];
    pad_temp_shared[(((((((int)threadIdx.z) * 112) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 2)) + 1))] = placeholder[(((((((rc_outer * 12544) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 2)) + 1))];
    if (((((int)threadIdx.z) * 2) + (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) >> 4)) < 32) {
      if ((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x)) < 512) {
        if (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) < 32) {
          if (((int)threadIdx.x) < 8) {
            placeholder_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x)))] = placeholder1[((((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) + ((((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) >> 4) * 128)) + (rc_outer * 16)) + (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) & 15)))];
          }
        }
      }
    }
    __syncthreads();
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)))], placeholder_shared[((((int)threadIdx.z) * 16))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)))], placeholder_shared[(((((int)threadIdx.z) * 16) + 256))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 1))], placeholder_shared[((((int)threadIdx.z) * 16))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 1))], placeholder_shared[(((((int)threadIdx.z) * 16) + 256))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 112))], placeholder_shared[(((((int)threadIdx.z) * 16) + 1))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 112))], placeholder_shared[(((((int)threadIdx.z) * 16) + 257))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 113))], placeholder_shared[(((((int)threadIdx.z) * 16) + 1))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 113))], placeholder_shared[(((((int)threadIdx.z) * 16) + 257))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 224))], placeholder_shared[(((((int)threadIdx.z) * 16) + 2))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 224))], placeholder_shared[(((((int)threadIdx.z) * 16) + 258))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 225))], placeholder_shared[(((((int)threadIdx.z) * 16) + 2))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 225))], placeholder_shared[(((((int)threadIdx.z) * 16) + 258))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 336))], placeholder_shared[(((((int)threadIdx.z) * 16) + 3))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 336))], placeholder_shared[(((((int)threadIdx.z) * 16) + 259))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 337))], placeholder_shared[(((((int)threadIdx.z) * 16) + 3))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 337))], placeholder_shared[(((((int)threadIdx.z) * 16) + 259))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 448))], placeholder_shared[(((((int)threadIdx.z) * 16) + 4))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 448))], placeholder_shared[(((((int)threadIdx.z) * 16) + 260))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 449))], placeholder_shared[(((((int)threadIdx.z) * 16) + 4))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 449))], placeholder_shared[(((((int)threadIdx.z) * 16) + 260))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 560))], placeholder_shared[(((((int)threadIdx.z) * 16) + 5))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 560))], placeholder_shared[(((((int)threadIdx.z) * 16) + 261))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 561))], placeholder_shared[(((((int)threadIdx.z) * 16) + 5))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 561))], placeholder_shared[(((((int)threadIdx.z) * 16) + 261))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 672))], placeholder_shared[(((((int)threadIdx.z) * 16) + 6))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 672))], placeholder_shared[(((((int)threadIdx.z) * 16) + 262))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 673))], placeholder_shared[(((((int)threadIdx.z) * 16) + 6))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 673))], placeholder_shared[(((((int)threadIdx.z) * 16) + 262))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 784))], placeholder_shared[(((((int)threadIdx.z) * 16) + 7))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 784))], placeholder_shared[(((((int)threadIdx.z) * 16) + 263))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 785))], placeholder_shared[(((((int)threadIdx.z) * 16) + 7))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 785))], placeholder_shared[(((((int)threadIdx.z) * 16) + 263))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 896))], placeholder_shared[(((((int)threadIdx.z) * 16) + 8))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 896))], placeholder_shared[(((((int)threadIdx.z) * 16) + 264))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 897))], placeholder_shared[(((((int)threadIdx.z) * 16) + 8))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 897))], placeholder_shared[(((((int)threadIdx.z) * 16) + 264))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 1008))], placeholder_shared[(((((int)threadIdx.z) * 16) + 9))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 1008))], placeholder_shared[(((((int)threadIdx.z) * 16) + 265))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 1009))], placeholder_shared[(((((int)threadIdx.z) * 16) + 9))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 1009))], placeholder_shared[(((((int)threadIdx.z) * 16) + 265))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 1120))], placeholder_shared[(((((int)threadIdx.z) * 16) + 10))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 1120))], placeholder_shared[(((((int)threadIdx.z) * 16) + 266))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 1121))], placeholder_shared[(((((int)threadIdx.z) * 16) + 10))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 1121))], placeholder_shared[(((((int)threadIdx.z) * 16) + 266))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 1232))], placeholder_shared[(((((int)threadIdx.z) * 16) + 11))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 1232))], placeholder_shared[(((((int)threadIdx.z) * 16) + 267))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 1233))], placeholder_shared[(((((int)threadIdx.z) * 16) + 11))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 1233))], placeholder_shared[(((((int)threadIdx.z) * 16) + 267))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 1344))], placeholder_shared[(((((int)threadIdx.z) * 16) + 12))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 1344))], placeholder_shared[(((((int)threadIdx.z) * 16) + 268))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 1345))], placeholder_shared[(((((int)threadIdx.z) * 16) + 12))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 1345))], placeholder_shared[(((((int)threadIdx.z) * 16) + 268))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 1456))], placeholder_shared[(((((int)threadIdx.z) * 16) + 13))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 1456))], placeholder_shared[(((((int)threadIdx.z) * 16) + 269))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 1457))], placeholder_shared[(((((int)threadIdx.z) * 16) + 13))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 1457))], placeholder_shared[(((((int)threadIdx.z) * 16) + 269))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 1568))], placeholder_shared[(((((int)threadIdx.z) * 16) + 14))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 1568))], placeholder_shared[(((((int)threadIdx.z) * 16) + 270))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 1569))], placeholder_shared[(((((int)threadIdx.z) * 16) + 14))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 1569))], placeholder_shared[(((((int)threadIdx.z) * 16) + 270))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 1680))], placeholder_shared[(((((int)threadIdx.z) * 16) + 15))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 1680))], placeholder_shared[(((((int)threadIdx.z) * 16) + 271))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 1681))], placeholder_shared[(((((int)threadIdx.z) * 16) + 15))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 1681))], placeholder_shared[(((((int)threadIdx.z) * 16) + 271))], compute[(3)]);
  }
  T_relu[((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 2)))] = max((compute[(0)] + placeholder2[(((((int)blockIdx.z) * 32) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 2)) + 12544))] = max((compute[(2)] + placeholder2[((((((int)blockIdx.z) * 32) + ((int)threadIdx.z)) + 16))]), 0.000000e+00f);
  T_relu[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 2)) + 1))] = max((compute[(1)] + placeholder2[(((((int)blockIdx.z) * 32) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 2)) + 12545))] = max((compute[(3)] + placeholder2[((((((int)blockIdx.z) * 32) + ((int)threadIdx.z)) + 16))]), 0.000000e+00f);
}


extern "C" __global__  __attribute__((amdgpu_num_vgpr(61))) __attribute__((amdgpu_num_sgpr(30))) void fused_nn_conv2d_add_nn_relu_12_kernel0_device_wrapper(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
    // Force the compiler to use all the index
    if (threadIdx.x + threadIdx.y * 4 + threadIdx.z * 4 * 4 >= 4 * 4 * 16) return;
    // if (blockIdx.x + blockIdx.y * 7 + blockIdx.z * 14 * 7 >= 7 * 14 * 2) return;
    fused_nn_conv2d_add_nn_relu_12_kernel0_device((float* __restrict__)placeholder,(float* __restrict__)placeholder1,(float* __restrict__)T_relu,(float* __restrict__)placeholder2);
    asm volatile(";; end_flag"); // jump back to the caller
}

extern "C" __global__  __attribute__((amdgpu_num_vgpr(25))) __attribute__((amdgpu_num_sgpr(30))) void fused_nn_conv2d_add_nn_relu_7_kernel0_device_wrapper(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
    // Force the compiler to use all the index
    if (threadIdx.x + threadIdx.y * 14 + threadIdx.z * 14 * 14 >= 14 * 14 * 1) return;
    // if (blockIdx.x + blockIdx.y * 1 + blockIdx.z * 1 * 1 >= 1 * 1 * 256) return;
    fused_nn_conv2d_add_nn_relu_7_kernel0_device((float* __restrict__)placeholder,(float* __restrict__)placeholder1,(float* __restrict__)T_relu,(float* __restrict__)placeholder2);
    asm volatile(";; end_flag"); // jump back to the caller
}

extern "C" __global__  __attribute__((amdgpu_num_vgpr(25))) __attribute__((amdgpu_num_sgpr(30))) void fused_nn_conv2d_add_nn_relu_6_kernel0_device_wrapper(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
    // Force the compiler to use all the index
    if (threadIdx.x + threadIdx.y * 7 + threadIdx.z * 2 * 7 >= 7 * 2 * 32) return;
    // if (blockIdx.x + blockIdx.y * 1 + blockIdx.z * 7 * 1 >= 1 * 7 * 16) return;
    fused_nn_conv2d_add_nn_relu_6_kernel0_device((float* __restrict__)placeholder,(float* __restrict__)placeholder1,(float* __restrict__)T_relu,(float* __restrict__)placeholder2);
    asm volatile(";; end_flag"); // jump back to the caller
}

extern "C" __global__  __attribute__((amdgpu_num_vgpr(37))) __attribute__((amdgpu_num_sgpr(30))) void fused_nn_conv2d_add_nn_relu_14_kernel0_device_wrapper(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
    // Force the compiler to use all the index
    if (threadIdx.x + threadIdx.y * 4 + threadIdx.z * 2 * 4 >= 4 * 2 * 32) return;
    // if (blockIdx.x + blockIdx.y * 7 + blockIdx.z * 14 * 7 >= 7 * 14 * 2) return;
    fused_nn_conv2d_add_nn_relu_14_kernel0_device((float* __restrict__)placeholder,(float* __restrict__)placeholder1,(float* __restrict__)T_relu,(float* __restrict__)placeholder2);
    asm volatile(";; end_flag"); // jump back to the caller
}

extern "C" __global__  __attribute__((amdgpu_num_vgpr(63))) __attribute__((amdgpu_num_sgpr(30))) void fused_nn_conv2d_add_nn_relu_18_kernel0_device_wrapper(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
    // Force the compiler to use all the index
    if (threadIdx.x + threadIdx.y * 16 + threadIdx.z * 1 * 16 >= 16 * 1 * 32) return;
    // if (blockIdx.x + blockIdx.y * 7 + blockIdx.z * 16 * 7 >= 7 * 16 * 1) return;
    fused_nn_conv2d_add_nn_relu_18_kernel0_device((float* __restrict__)placeholder,(float* __restrict__)placeholder1,(float* __restrict__)T_relu,(float* __restrict__)placeholder2);
    asm volatile(";; end_flag"); // jump back to the caller
}

extern "C" __global__  __attribute__((amdgpu_num_vgpr(25))) __attribute__((amdgpu_num_sgpr(30))) void fused_nn_conv2d_add_nn_relu_17_kernel0_device_wrapper(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
    // Force the compiler to use all the index
    if (threadIdx.x + threadIdx.y * 28 + threadIdx.z * 14 * 28 >= 28 * 14 * 1) return;
    // if (blockIdx.x + blockIdx.y * 1 + blockIdx.z * 8 * 1 >= 1 * 8 * 32) return;
    fused_nn_conv2d_add_nn_relu_17_kernel0_device((float* __restrict__)placeholder,(float* __restrict__)placeholder1,(float* __restrict__)T_relu,(float* __restrict__)placeholder2);
    asm volatile(";; end_flag"); // jump back to the caller
}

extern "C" __global__  __attribute__((amdgpu_num_vgpr(25))) __attribute__((amdgpu_num_sgpr(30))) void fused_nn_conv2d_add_nn_relu_11_kernel0_device_wrapper(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
    // Force the compiler to use all the index
    if (threadIdx.x + threadIdx.y * 28 + threadIdx.z * 28 * 28 >= 28 * 28 * 1) return;
    // if (blockIdx.x + blockIdx.y * 1 + blockIdx.z * 1 * 1 >= 1 * 1 * 128) return;
    fused_nn_conv2d_add_nn_relu_11_kernel0_device((float* __restrict__)placeholder,(float* __restrict__)placeholder1,(float* __restrict__)T_relu,(float* __restrict__)placeholder2);
    asm volatile(";; end_flag"); // jump back to the caller
}

extern "C" __global__  __attribute__((amdgpu_num_vgpr(63))) __attribute__((amdgpu_num_sgpr(30))) void fused_nn_conv2d_add_nn_relu_16_kernel0_device_wrapper(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
    // Force the compiler to use all the index
    if (threadIdx.x + threadIdx.y * 4 + threadIdx.z * 4 * 4 >= 4 * 4 * 16) return;
    // if (blockIdx.x + blockIdx.y * 7 + blockIdx.z * 28 * 7 >= 7 * 28 * 1) return;
    fused_nn_conv2d_add_nn_relu_16_kernel0_device((float* __restrict__)placeholder,(float* __restrict__)placeholder1,(float* __restrict__)T_relu,(float* __restrict__)placeholder2);
    asm volatile(";; end_flag"); // jump back to the caller
}

extern "C" __global__  __attribute__((amdgpu_num_vgpr(25))) __attribute__((amdgpu_num_sgpr(30))) void fused_nn_conv2d_add_nn_relu_2_kernel0_device_wrapper(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
    // Force the compiler to use all the index
    if (threadIdx.x + threadIdx.y * 7 + threadIdx.z * 7 * 7 >= 7 * 7 * 16) return;
    // if (blockIdx.x + blockIdx.y * 1 + blockIdx.z * 1 * 1 >= 1 * 1 * 64) return;
    fused_nn_conv2d_add_nn_relu_2_kernel0_device((float* __restrict__)placeholder,(float* __restrict__)placeholder1,(float* __restrict__)T_relu,(float* __restrict__)placeholder2);
    asm volatile(";; end_flag"); // jump back to the caller
}

extern "C" __global__  __attribute__((amdgpu_num_vgpr(25))) __attribute__((amdgpu_num_sgpr(30))) void fused_nn_batch_flatten_kernel0_device_wrapper(float* __restrict__ tensor, float* __restrict__ placeholder) {
    // Force the compiler to use all the index
    if (threadIdx.x + threadIdx.y * 256 + threadIdx.z * 1 * 256 >= 256 * 1 * 1) return;
    // if (blockIdx.x + blockIdx.y * 4 + blockIdx.z * 1 * 4 >= 4 * 1 * 1) return;
    fused_nn_batch_flatten_kernel0_device((float* __restrict__)tensor,(float* __restrict__)placeholder);
    asm volatile(";; end_flag"); // jump back to the caller
}

extern "C" __global__  __attribute__((amdgpu_num_vgpr(25))) __attribute__((amdgpu_num_sgpr(30))) void fused_nn_conv2d_add_nn_relu_13_kernel0_device_wrapper(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
    // Force the compiler to use all the index
    if (threadIdx.x + threadIdx.y * 28 + threadIdx.z * 28 * 28 >= 28 * 28 * 1) return;
    // if (blockIdx.x + blockIdx.y * 1 + blockIdx.z * 1 * 1 >= 1 * 1 * 128) return;
    fused_nn_conv2d_add_nn_relu_13_kernel0_device((float* __restrict__)placeholder,(float* __restrict__)placeholder1,(float* __restrict__)T_relu,(float* __restrict__)placeholder2);
    asm volatile(";; end_flag"); // jump back to the caller
}

extern "C" __global__  __attribute__((amdgpu_num_vgpr(33))) __attribute__((amdgpu_num_sgpr(30))) void fused_nn_conv2d_add_nn_relu_8_kernel0_device_wrapper(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
    // Force the compiler to use all the index
    if (threadIdx.x + threadIdx.y * 14 + threadIdx.z * 4 * 14 >= 14 * 4 * 16) return;
    // if (blockIdx.x + blockIdx.y * 1 + blockIdx.z * 7 * 1 >= 1 * 7 * 8) return;
    fused_nn_conv2d_add_nn_relu_8_kernel0_device((float* __restrict__)placeholder,(float* __restrict__)placeholder1,(float* __restrict__)T_relu,(float* __restrict__)placeholder2);
    asm volatile(";; end_flag"); // jump back to the caller
}

extern "C" __global__  __attribute__((amdgpu_num_vgpr(25))) __attribute__((amdgpu_num_sgpr(30))) void fused_nn_global_avg_pool2d_kernel0_device_wrapper(float* __restrict__ placeholder, float* __restrict__ tensor) {
    // Force the compiler to use all the index
    if (threadIdx.x + threadIdx.y * 8 + threadIdx.z * 8 * 8 >= 8 * 8 * 1) return;
    // if (blockIdx.x + blockIdx.y * 128 + blockIdx.z * 1 * 128 >= 128 * 1 * 1) return;
    fused_nn_global_avg_pool2d_kernel0_device((float* __restrict__)placeholder,(float* __restrict__)tensor);
    asm volatile(";; end_flag"); // jump back to the caller
}

extern "C" __global__  __attribute__((amdgpu_num_vgpr(25))) __attribute__((amdgpu_num_sgpr(30))) void fused_nn_conv2d_add_nn_relu_9_kernel0_device_wrapper(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
    // Force the compiler to use all the index
    if (threadIdx.x + threadIdx.y * 7 + threadIdx.z * 14 * 7 >= 7 * 14 * 4) return;
    // if (blockIdx.x + blockIdx.y * 2 + blockIdx.z * 2 * 2 >= 2 * 2 * 64) return;
    fused_nn_conv2d_add_nn_relu_9_kernel0_device((float* __restrict__)placeholder,(float* __restrict__)placeholder1,(float* __restrict__)T_relu,(float* __restrict__)placeholder2);
    asm volatile(";; end_flag"); // jump back to the caller
}

extern "C" __global__  __attribute__((amdgpu_num_vgpr(27))) __attribute__((amdgpu_num_sgpr(30))) void fused_nn_dense_add_kernel0_device_wrapper(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_add, float* __restrict__ placeholder2) {
    // Force the compiler to use all the index
    if (threadIdx.x + threadIdx.y * 64 + threadIdx.z * 1 * 64 >= 64 * 1 * 1) return;
    // if (blockIdx.x + blockIdx.y * 1000 + blockIdx.z * 1 * 1000 >= 1000 * 1 * 1) return;
    fused_nn_dense_add_kernel0_device((float* __restrict__)placeholder,(float* __restrict__)placeholder1,(float* __restrict__)T_add,(float* __restrict__)placeholder2);
    asm volatile(";; end_flag"); // jump back to the caller
}

extern "C" __global__  __attribute__((amdgpu_num_vgpr(25))) __attribute__((amdgpu_num_sgpr(30))) void fused_nn_conv2d_add_nn_relu_15_kernel0_device_wrapper(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
    // Force the compiler to use all the index
    if (threadIdx.x + threadIdx.y * 56 + threadIdx.z * 14 * 56 >= 56 * 14 * 1) return;
    // if (blockIdx.x + blockIdx.y * 1 + blockIdx.z * 4 * 1 >= 1 * 4 * 64) return;
    fused_nn_conv2d_add_nn_relu_15_kernel0_device((float* __restrict__)placeholder,(float* __restrict__)placeholder1,(float* __restrict__)T_relu,(float* __restrict__)placeholder2);
    asm volatile(";; end_flag"); // jump back to the caller
}

extern "C" __global__  __attribute__((amdgpu_num_vgpr(35))) __attribute__((amdgpu_num_sgpr(54))) void fused_nn_softmax_kernel0_device_wrapper(float* __restrict__ placeholder, float* __restrict__ T_softmax_norm) {
    // Force the compiler to use all the index
    if (threadIdx.x + threadIdx.y * 64 + threadIdx.z * 1 * 64 >= 64 * 1 * 1) return;
    // if (blockIdx.x + blockIdx.y * 1 + blockIdx.z * 1 * 1 >= 1 * 1 * 1) return;
    fused_nn_softmax_kernel0_device((float* __restrict__)placeholder,(float* __restrict__)T_softmax_norm);
    asm volatile(";; end_flag"); // jump back to the caller
}

extern "C" __global__  __attribute__((amdgpu_num_vgpr(25))) __attribute__((amdgpu_num_sgpr(30))) void fused_nn_conv2d_add_nn_relu_3_kernel0_device_wrapper(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
    // Force the compiler to use all the index
    if (threadIdx.x + threadIdx.y * 7 + threadIdx.z * 7 * 7 >= 7 * 7 * 8) return;
    // if (blockIdx.x + blockIdx.y * 1 + blockIdx.z * 1 * 1 >= 1 * 1 * 64) return;
    fused_nn_conv2d_add_nn_relu_3_kernel0_device((float* __restrict__)placeholder,(float* __restrict__)placeholder1,(float* __restrict__)T_relu,(float* __restrict__)placeholder2);
    asm volatile(";; end_flag"); // jump back to the caller
}

extern "C" __global__  __attribute__((amdgpu_num_vgpr(29))) __attribute__((amdgpu_num_sgpr(30))) void fused_nn_conv2d_add_nn_relu_4_kernel0_device_wrapper(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
    // Force the compiler to use all the index
    if (threadIdx.x + threadIdx.y * 7 + threadIdx.z * 7 * 7 >= 7 * 7 * 8) return;
    // if (blockIdx.x + blockIdx.y * 2 + blockIdx.z * 2 * 2 >= 2 * 2 * 32) return;
    fused_nn_conv2d_add_nn_relu_4_kernel0_device((float* __restrict__)placeholder,(float* __restrict__)placeholder1,(float* __restrict__)T_relu,(float* __restrict__)placeholder2);
    asm volatile(";; end_flag"); // jump back to the caller
}

extern "C" __global__  __attribute__((amdgpu_num_vgpr(25))) __attribute__((amdgpu_num_sgpr(30))) void fused_nn_conv2d_add_nn_relu_kernel0_device_wrapper(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
    // Force the compiler to use all the index
    if (threadIdx.x + threadIdx.y * 7 + threadIdx.z * 7 * 7 >= 7 * 7 * 16) return;
    // if (blockIdx.x + blockIdx.y * 1 + blockIdx.z * 1 * 1 >= 1 * 1 * 64) return;
    fused_nn_conv2d_add_nn_relu_kernel0_device((float* __restrict__)placeholder,(float* __restrict__)placeholder1,(float* __restrict__)T_relu,(float* __restrict__)placeholder2);
    asm volatile(";; end_flag"); // jump back to the caller
}

extern "C" __global__  __attribute__((amdgpu_num_vgpr(25))) __attribute__((amdgpu_num_sgpr(30))) void fused_nn_conv2d_add_nn_relu_1_kernel0_device_wrapper(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
    // Force the compiler to use all the index
    if (threadIdx.x + threadIdx.y * 7 + threadIdx.z * 7 * 7 >= 7 * 7 * 4) return;
    // if (blockIdx.x + blockIdx.y * 1 + blockIdx.z * 1 * 1 >= 1 * 1 * 256) return;
    fused_nn_conv2d_add_nn_relu_1_kernel0_device((float* __restrict__)placeholder,(float* __restrict__)placeholder1,(float* __restrict__)T_relu,(float* __restrict__)placeholder2);
    asm volatile(";; end_flag"); // jump back to the caller
}

extern "C" __global__  __attribute__((amdgpu_num_vgpr(25))) __attribute__((amdgpu_num_sgpr(30))) void fused_nn_conv2d_add_nn_relu_5_kernel0_device_wrapper(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
    // Force the compiler to use all the index
    if (threadIdx.x + threadIdx.y * 7 + threadIdx.z * 14 * 7 >= 7 * 14 * 2) return;
    // if (blockIdx.x + blockIdx.y * 1 + blockIdx.z * 1 * 1 >= 1 * 1 * 256) return;
    fused_nn_conv2d_add_nn_relu_5_kernel0_device((float* __restrict__)placeholder,(float* __restrict__)placeholder1,(float* __restrict__)T_relu,(float* __restrict__)placeholder2);
    asm volatile(";; end_flag"); // jump back to the caller
}

extern "C" __global__  __attribute__((amdgpu_num_vgpr(33))) __attribute__((amdgpu_num_sgpr(30))) void fused_nn_conv2d_add_nn_relu_10_kernel0_device_wrapper(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
    // Force the compiler to use all the index
    if (threadIdx.x + threadIdx.y * 14 + threadIdx.z * 4 * 14 >= 14 * 4 * 16) return;
    // if (blockIdx.x + blockIdx.y * 1 + blockIdx.z * 7 * 1 >= 1 * 7 * 8) return;
    fused_nn_conv2d_add_nn_relu_10_kernel0_device((float* __restrict__)placeholder,(float* __restrict__)placeholder1,(float* __restrict__)T_relu,(float* __restrict__)placeholder2);
    asm volatile(";; end_flag"); // jump back to the caller
}

extern "C" __global__  void fused_nn_conv2d_add_nn_relu_12_kernel0(
    void* func_l, int layers_l, int task_num_l, int task_offset_l, float** param_l,
    void* func_r, int layers_r, int task_num_r, int task_offset_r, float** param_r,
    int cu_partition) {
    asm volatile(";; caller_flag");
    return;
}

extern "C" __global__  void fused_nn_conv2d_add_nn_relu_7_kernel0(
    void* func_l, int layers_l, int task_num_l, int task_offset_l, float** param_l,
    void* func_r, int layers_r, int task_num_r, int task_offset_r, float** param_r,
    int cu_partition) {
    asm volatile(";; caller_flag");
    return;
}

extern "C" __global__  void fused_nn_conv2d_add_nn_relu_6_kernel0(
    void* func_l, int layers_l, int task_num_l, int task_offset_l, float** param_l,
    void* func_r, int layers_r, int task_num_r, int task_offset_r, float** param_r,
    int cu_partition) {
    asm volatile(";; caller_flag");
    return;
}

extern "C" __global__  void fused_nn_conv2d_add_nn_relu_14_kernel0(
    void* func_l, int layers_l, int task_num_l, int task_offset_l, float** param_l,
    void* func_r, int layers_r, int task_num_r, int task_offset_r, float** param_r,
    int cu_partition) {
    asm volatile(";; caller_flag");
    return;
}

extern "C" __global__  void fused_nn_conv2d_add_nn_relu_18_kernel0(
    void* func_l, int layers_l, int task_num_l, int task_offset_l, float** param_l,
    void* func_r, int layers_r, int task_num_r, int task_offset_r, float** param_r,
    int cu_partition) {
    asm volatile(";; caller_flag");
    return;
}

extern "C" __global__  void fused_nn_conv2d_add_nn_relu_17_kernel0(
    void* func_l, int layers_l, int task_num_l, int task_offset_l, float** param_l,
    void* func_r, int layers_r, int task_num_r, int task_offset_r, float** param_r,
    int cu_partition) {
    asm volatile(";; caller_flag");
    return;
}

extern "C" __global__  void fused_nn_conv2d_add_nn_relu_11_kernel0(
    void* func_l, int layers_l, int task_num_l, int task_offset_l, float** param_l,
    void* func_r, int layers_r, int task_num_r, int task_offset_r, float** param_r,
    int cu_partition) {
    asm volatile(";; caller_flag");
    return;
}

extern "C" __global__  void fused_nn_conv2d_add_nn_relu_16_kernel0(
    void* func_l, int layers_l, int task_num_l, int task_offset_l, float** param_l,
    void* func_r, int layers_r, int task_num_r, int task_offset_r, float** param_r,
    int cu_partition) {
    asm volatile(";; caller_flag");
    return;
}

extern "C" __global__  void fused_nn_conv2d_add_nn_relu_2_kernel0(
    void* func_l, int layers_l, int task_num_l, int task_offset_l, float** param_l,
    void* func_r, int layers_r, int task_num_r, int task_offset_r, float** param_r,
    int cu_partition) {
    asm volatile(";; caller_flag");
    return;
}

extern "C" __global__  void fused_nn_batch_flatten_kernel0(
    void* func_l, int layers_l, int task_num_l, int task_offset_l, float** param_l,
    void* func_r, int layers_r, int task_num_r, int task_offset_r, float** param_r,
    int cu_partition) {
    asm volatile(";; caller_flag");
    return;
}

extern "C" __global__  void fused_nn_conv2d_add_nn_relu_13_kernel0(
    void* func_l, int layers_l, int task_num_l, int task_offset_l, float** param_l,
    void* func_r, int layers_r, int task_num_r, int task_offset_r, float** param_r,
    int cu_partition) {
    asm volatile(";; caller_flag");
    return;
}

extern "C" __global__  void fused_nn_conv2d_add_nn_relu_8_kernel0(
    void* func_l, int layers_l, int task_num_l, int task_offset_l, float** param_l,
    void* func_r, int layers_r, int task_num_r, int task_offset_r, float** param_r,
    int cu_partition) {
    asm volatile(";; caller_flag");
    return;
}

extern "C" __global__  void fused_nn_global_avg_pool2d_kernel0(
    void* func_l, int layers_l, int task_num_l, int task_offset_l, float** param_l,
    void* func_r, int layers_r, int task_num_r, int task_offset_r, float** param_r,
    int cu_partition) {
    asm volatile(";; caller_flag");
    return;
}

extern "C" __global__  void fused_nn_conv2d_add_nn_relu_9_kernel0(
    void* func_l, int layers_l, int task_num_l, int task_offset_l, float** param_l,
    void* func_r, int layers_r, int task_num_r, int task_offset_r, float** param_r,
    int cu_partition) {
    asm volatile(";; caller_flag");
    return;
}

extern "C" __global__  void fused_nn_dense_add_kernel0(
    void* func_l, int layers_l, int task_num_l, int task_offset_l, float** param_l,
    void* func_r, int layers_r, int task_num_r, int task_offset_r, float** param_r,
    int cu_partition) {
    asm volatile(";; caller_flag");
    return;
}

extern "C" __global__  void fused_nn_conv2d_add_nn_relu_15_kernel0(
    void* func_l, int layers_l, int task_num_l, int task_offset_l, float** param_l,
    void* func_r, int layers_r, int task_num_r, int task_offset_r, float** param_r,
    int cu_partition) {
    asm volatile(";; caller_flag");
    return;
}

extern "C" __global__  void fused_nn_softmax_kernel0(
    void* func_l, int layers_l, int task_num_l, int task_offset_l, float** param_l,
    void* func_r, int layers_r, int task_num_r, int task_offset_r, float** param_r,
    int cu_partition) {
    asm volatile(";; caller_flag");
    return;
}

extern "C" __global__  void fused_nn_conv2d_add_nn_relu_3_kernel0(
    void* func_l, int layers_l, int task_num_l, int task_offset_l, float** param_l,
    void* func_r, int layers_r, int task_num_r, int task_offset_r, float** param_r,
    int cu_partition) {
    asm volatile(";; caller_flag");
    return;
}

extern "C" __global__  void fused_nn_conv2d_add_nn_relu_4_kernel0(
    void* func_l, int layers_l, int task_num_l, int task_offset_l, float** param_l,
    void* func_r, int layers_r, int task_num_r, int task_offset_r, float** param_r,
    int cu_partition) {
    asm volatile(";; caller_flag");
    return;
}

extern "C" __global__  void fused_nn_conv2d_add_nn_relu_kernel0(
    void* func_l, int layers_l, int task_num_l, int task_offset_l, float** param_l,
    void* func_r, int layers_r, int task_num_r, int task_offset_r, float** param_r,
    int cu_partition) {
    asm volatile(";; caller_flag");
    return;
}

extern "C" __global__  void fused_nn_conv2d_add_nn_relu_1_kernel0(
    void* func_l, int layers_l, int task_num_l, int task_offset_l, float** param_l,
    void* func_r, int layers_r, int task_num_r, int task_offset_r, float** param_r,
    int cu_partition) {
    asm volatile(";; caller_flag");
    return;
}

extern "C" __global__  void fused_nn_conv2d_add_nn_relu_5_kernel0(
    void* func_l, int layers_l, int task_num_l, int task_offset_l, float** param_l,
    void* func_r, int layers_r, int task_num_r, int task_offset_r, float** param_r,
    int cu_partition) {
    asm volatile(";; caller_flag");
    return;
}

extern "C" __global__  void fused_nn_conv2d_add_nn_relu_10_kernel0(
    void* func_l, int layers_l, int task_num_l, int task_offset_l, float** param_l,
    void* func_r, int layers_r, int task_num_r, int task_offset_r, float** param_r,
    int cu_partition) {
    asm volatile(";; caller_flag");
    return;
}

extern "C" __device__ __noinline__ dim3 get_3d_idx_128_1_1(int idx) {
  dim3 dim(128, 1, 1);
  dim3 result;
  result.x = idx % dim.x;
  result.y = idx / dim.x % dim.y;
  result.z = idx / (dim.x * dim.y);
  return result;
}

extern "C" __device__ __noinline__ dim3 get_3d_idx_256_1_1(int idx) {
  dim3 dim(256, 1, 1);
  dim3 result;
  result.x = idx % dim.x;
  result.y = idx / dim.x % dim.y;
  result.z = idx / (dim.x * dim.y);
  return result;
}

extern "C" __device__ __noinline__ dim3 get_3d_idx_7_28_1(int idx) {
  dim3 dim(7, 28, 1);
  dim3 result;
  result.x = idx % dim.x;
  result.y = idx / dim.x % dim.y;
  result.z = idx / (dim.x * dim.y);
  return result;
}

extern "C" __device__ __noinline__ dim3 get_3d_idx_16_1_32(int idx) {
  dim3 dim(16, 1, 32);
  dim3 result;
  result.x = idx % dim.x;
  result.y = idx / dim.x % dim.y;
  result.z = idx / (dim.x * dim.y);
  return result;
}

extern "C" __device__ __noinline__ dim3 get_3d_idx_2_2_32(int idx) {
  dim3 dim(2, 2, 32);
  dim3 result;
  result.x = idx % dim.x;
  result.y = idx / dim.x % dim.y;
  result.z = idx / (dim.x * dim.y);
  return result;
}

extern "C" __device__ __noinline__ dim3 get_3d_idx_4_2_32(int idx) {
  dim3 dim(4, 2, 32);
  dim3 result;
  result.x = idx % dim.x;
  result.y = idx / dim.x % dim.y;
  result.z = idx / (dim.x * dim.y);
  return result;
}

extern "C" __device__ __noinline__ dim3 get_3d_idx_7_7_8(int idx) {
  dim3 dim(7, 7, 8);
  dim3 result;
  result.x = idx % dim.x;
  result.y = idx / dim.x % dim.y;
  result.z = idx / (dim.x * dim.y);
  return result;
}

extern "C" __device__ __noinline__ dim3 get_3d_idx_1_7_16(int idx) {
  dim3 dim(1, 7, 16);
  dim3 result;
  result.x = idx % dim.x;
  result.y = idx / dim.x % dim.y;
  result.z = idx / (dim.x * dim.y);
  return result;
}

extern "C" __device__ __noinline__ dim3 get_3d_idx_1_1_64(int idx) {
  dim3 dim(1, 1, 64);
  dim3 result;
  result.x = idx % dim.x;
  result.y = idx / dim.x % dim.y;
  result.z = idx / (dim.x * dim.y);
  return result;
}

extern "C" __device__ __noinline__ dim3 get_3d_idx_28_14_1(int idx) {
  dim3 dim(28, 14, 1);
  dim3 result;
  result.x = idx % dim.x;
  result.y = idx / dim.x % dim.y;
  result.z = idx / (dim.x * dim.y);
  return result;
}

extern "C" __device__ __noinline__ dim3 get_3d_idx_2_2_64(int idx) {
  dim3 dim(2, 2, 64);
  dim3 result;
  result.x = idx % dim.x;
  result.y = idx / dim.x % dim.y;
  result.z = idx / (dim.x * dim.y);
  return result;
}

extern "C" __device__ __noinline__ dim3 get_3d_idx_1_8_32(int idx) {
  dim3 dim(1, 8, 32);
  dim3 result;
  result.x = idx % dim.x;
  result.y = idx / dim.x % dim.y;
  result.z = idx / (dim.x * dim.y);
  return result;
}

extern "C" __device__ __noinline__ dim3 get_3d_idx_7_14_2(int idx) {
  dim3 dim(7, 14, 2);
  dim3 result;
  result.x = idx % dim.x;
  result.y = idx / dim.x % dim.y;
  result.z = idx / (dim.x * dim.y);
  return result;
}

extern "C" __device__ __noinline__ dim3 get_3d_idx_64_1_1(int idx) {
  dim3 dim(64, 1, 1);
  dim3 result;
  result.x = idx % dim.x;
  result.y = idx / dim.x % dim.y;
  result.z = idx / (dim.x * dim.y);
  return result;
}

extern "C" __device__ __noinline__ dim3 get_3d_idx_28_28_1(int idx) {
  dim3 dim(28, 28, 1);
  dim3 result;
  result.x = idx % dim.x;
  result.y = idx / dim.x % dim.y;
  result.z = idx / (dim.x * dim.y);
  return result;
}

extern "C" __device__ __noinline__ dim3 get_3d_idx_4_1_1(int idx) {
  dim3 dim(4, 1, 1);
  dim3 result;
  result.x = idx % dim.x;
  result.y = idx / dim.x % dim.y;
  result.z = idx / (dim.x * dim.y);
  return result;
}

extern "C" __device__ __noinline__ dim3 get_3d_idx_56_14_1(int idx) {
  dim3 dim(56, 14, 1);
  dim3 result;
  result.x = idx % dim.x;
  result.y = idx / dim.x % dim.y;
  result.z = idx / (dim.x * dim.y);
  return result;
}

extern "C" __device__ __noinline__ dim3 get_3d_idx_1_1_1(int idx) {
  dim3 dim(1, 1, 1);
  dim3 result;
  result.x = idx % dim.x;
  result.y = idx / dim.x % dim.y;
  result.z = idx / (dim.x * dim.y);
  return result;
}

extern "C" __device__ __noinline__ dim3 get_3d_idx_7_14_4(int idx) {
  dim3 dim(7, 14, 4);
  dim3 result;
  result.x = idx % dim.x;
  result.y = idx / dim.x % dim.y;
  result.z = idx / (dim.x * dim.y);
  return result;
}

extern "C" __device__ __noinline__ dim3 get_3d_idx_7_16_1(int idx) {
  dim3 dim(7, 16, 1);
  dim3 result;
  result.x = idx % dim.x;
  result.y = idx / dim.x % dim.y;
  result.z = idx / (dim.x * dim.y);
  return result;
}

extern "C" __device__ __noinline__ dim3 get_3d_idx_8_8_1(int idx) {
  dim3 dim(8, 8, 1);
  dim3 result;
  result.x = idx % dim.x;
  result.y = idx / dim.x % dim.y;
  result.z = idx / (dim.x * dim.y);
  return result;
}

extern "C" __device__ __noinline__ dim3 get_3d_idx_1_1_128(int idx) {
  dim3 dim(1, 1, 128);
  dim3 result;
  result.x = idx % dim.x;
  result.y = idx / dim.x % dim.y;
  result.z = idx / (dim.x * dim.y);
  return result;
}

extern "C" __device__ __noinline__ dim3 get_3d_idx_1_1_256(int idx) {
  dim3 dim(1, 1, 256);
  dim3 result;
  result.x = idx % dim.x;
  result.y = idx / dim.x % dim.y;
  result.z = idx / (dim.x * dim.y);
  return result;
}

extern "C" __device__ __noinline__ dim3 get_3d_idx_7_7_16(int idx) {
  dim3 dim(7, 7, 16);
  dim3 result;
  result.x = idx % dim.x;
  result.y = idx / dim.x % dim.y;
  result.z = idx / (dim.x * dim.y);
  return result;
}

extern "C" __device__ __noinline__ dim3 get_3d_idx_1_4_64(int idx) {
  dim3 dim(1, 4, 64);
  dim3 result;
  result.x = idx % dim.x;
  result.y = idx / dim.x % dim.y;
  result.z = idx / (dim.x * dim.y);
  return result;
}

extern "C" __device__ __noinline__ dim3 get_3d_idx_14_14_1(int idx) {
  dim3 dim(14, 14, 1);
  dim3 result;
  result.x = idx % dim.x;
  result.y = idx / dim.x % dim.y;
  result.z = idx / (dim.x * dim.y);
  return result;
}

extern "C" __device__ __noinline__ dim3 get_3d_idx_7_7_4(int idx) {
  dim3 dim(7, 7, 4);
  dim3 result;
  result.x = idx % dim.x;
  result.y = idx / dim.x % dim.y;
  result.z = idx / (dim.x * dim.y);
  return result;
}

extern "C" __device__ __noinline__ dim3 get_3d_idx_7_2_32(int idx) {
  dim3 dim(7, 2, 32);
  dim3 result;
  result.x = idx % dim.x;
  result.y = idx / dim.x % dim.y;
  result.z = idx / (dim.x * dim.y);
  return result;
}

extern "C" __device__ __noinline__ dim3 get_3d_idx_14_4_16(int idx) {
  dim3 dim(14, 4, 16);
  dim3 result;
  result.x = idx % dim.x;
  result.y = idx / dim.x % dim.y;
  result.z = idx / (dim.x * dim.y);
  return result;
}

extern "C" __device__ __noinline__ dim3 get_3d_idx_1_7_8(int idx) {
  dim3 dim(1, 7, 8);
  dim3 result;
  result.x = idx % dim.x;
  result.y = idx / dim.x % dim.y;
  result.z = idx / (dim.x * dim.y);
  return result;
}

extern "C" __device__ __noinline__ dim3 get_3d_idx_1000_1_1(int idx) {
  dim3 dim(1000, 1, 1);
  dim3 result;
  result.x = idx % dim.x;
  result.y = idx / dim.x % dim.y;
  result.z = idx / (dim.x * dim.y);
  return result;
}

extern "C" __device__ __noinline__ dim3 get_3d_idx_4_4_16(int idx) {
  dim3 dim(4, 4, 16);
  dim3 result;
  result.x = idx % dim.x;
  result.y = idx / dim.x % dim.y;
  result.z = idx / (dim.x * dim.y);
  return result;
}

__global__ void get_3d_idx_caller(int* buf) {
    dim3 task_idx;

    task_idx = get_3d_idx_128_1_1(threadIdx.x);
    buf[task_idx.x] = task_idx.x;
    buf[task_idx.y] = task_idx.y;
    buf[task_idx.z] = task_idx.z;

    task_idx = get_3d_idx_256_1_1(threadIdx.x);
    buf[task_idx.x] = task_idx.x;
    buf[task_idx.y] = task_idx.y;
    buf[task_idx.z] = task_idx.z;

    task_idx = get_3d_idx_7_28_1(threadIdx.x);
    buf[task_idx.x] = task_idx.x;
    buf[task_idx.y] = task_idx.y;
    buf[task_idx.z] = task_idx.z;

    task_idx = get_3d_idx_16_1_32(threadIdx.x);
    buf[task_idx.x] = task_idx.x;
    buf[task_idx.y] = task_idx.y;
    buf[task_idx.z] = task_idx.z;

    task_idx = get_3d_idx_2_2_32(threadIdx.x);
    buf[task_idx.x] = task_idx.x;
    buf[task_idx.y] = task_idx.y;
    buf[task_idx.z] = task_idx.z;

    task_idx = get_3d_idx_4_2_32(threadIdx.x);
    buf[task_idx.x] = task_idx.x;
    buf[task_idx.y] = task_idx.y;
    buf[task_idx.z] = task_idx.z;

    task_idx = get_3d_idx_7_7_8(threadIdx.x);
    buf[task_idx.x] = task_idx.x;
    buf[task_idx.y] = task_idx.y;
    buf[task_idx.z] = task_idx.z;

    task_idx = get_3d_idx_1_7_16(threadIdx.x);
    buf[task_idx.x] = task_idx.x;
    buf[task_idx.y] = task_idx.y;
    buf[task_idx.z] = task_idx.z;

    task_idx = get_3d_idx_1_1_64(threadIdx.x);
    buf[task_idx.x] = task_idx.x;
    buf[task_idx.y] = task_idx.y;
    buf[task_idx.z] = task_idx.z;

    task_idx = get_3d_idx_28_14_1(threadIdx.x);
    buf[task_idx.x] = task_idx.x;
    buf[task_idx.y] = task_idx.y;
    buf[task_idx.z] = task_idx.z;

    task_idx = get_3d_idx_2_2_64(threadIdx.x);
    buf[task_idx.x] = task_idx.x;
    buf[task_idx.y] = task_idx.y;
    buf[task_idx.z] = task_idx.z;

    task_idx = get_3d_idx_1_8_32(threadIdx.x);
    buf[task_idx.x] = task_idx.x;
    buf[task_idx.y] = task_idx.y;
    buf[task_idx.z] = task_idx.z;

    task_idx = get_3d_idx_7_14_2(threadIdx.x);
    buf[task_idx.x] = task_idx.x;
    buf[task_idx.y] = task_idx.y;
    buf[task_idx.z] = task_idx.z;

    task_idx = get_3d_idx_64_1_1(threadIdx.x);
    buf[task_idx.x] = task_idx.x;
    buf[task_idx.y] = task_idx.y;
    buf[task_idx.z] = task_idx.z;

    task_idx = get_3d_idx_28_28_1(threadIdx.x);
    buf[task_idx.x] = task_idx.x;
    buf[task_idx.y] = task_idx.y;
    buf[task_idx.z] = task_idx.z;

    task_idx = get_3d_idx_4_1_1(threadIdx.x);
    buf[task_idx.x] = task_idx.x;
    buf[task_idx.y] = task_idx.y;
    buf[task_idx.z] = task_idx.z;

    task_idx = get_3d_idx_56_14_1(threadIdx.x);
    buf[task_idx.x] = task_idx.x;
    buf[task_idx.y] = task_idx.y;
    buf[task_idx.z] = task_idx.z;

    task_idx = get_3d_idx_1_1_1(threadIdx.x);
    buf[task_idx.x] = task_idx.x;
    buf[task_idx.y] = task_idx.y;
    buf[task_idx.z] = task_idx.z;

    task_idx = get_3d_idx_7_14_4(threadIdx.x);
    buf[task_idx.x] = task_idx.x;
    buf[task_idx.y] = task_idx.y;
    buf[task_idx.z] = task_idx.z;

    task_idx = get_3d_idx_7_16_1(threadIdx.x);
    buf[task_idx.x] = task_idx.x;
    buf[task_idx.y] = task_idx.y;
    buf[task_idx.z] = task_idx.z;

    task_idx = get_3d_idx_8_8_1(threadIdx.x);
    buf[task_idx.x] = task_idx.x;
    buf[task_idx.y] = task_idx.y;
    buf[task_idx.z] = task_idx.z;

    task_idx = get_3d_idx_1_1_128(threadIdx.x);
    buf[task_idx.x] = task_idx.x;
    buf[task_idx.y] = task_idx.y;
    buf[task_idx.z] = task_idx.z;

    task_idx = get_3d_idx_1_1_256(threadIdx.x);
    buf[task_idx.x] = task_idx.x;
    buf[task_idx.y] = task_idx.y;
    buf[task_idx.z] = task_idx.z;

    task_idx = get_3d_idx_7_7_16(threadIdx.x);
    buf[task_idx.x] = task_idx.x;
    buf[task_idx.y] = task_idx.y;
    buf[task_idx.z] = task_idx.z;

    task_idx = get_3d_idx_1_4_64(threadIdx.x);
    buf[task_idx.x] = task_idx.x;
    buf[task_idx.y] = task_idx.y;
    buf[task_idx.z] = task_idx.z;

    task_idx = get_3d_idx_14_14_1(threadIdx.x);
    buf[task_idx.x] = task_idx.x;
    buf[task_idx.y] = task_idx.y;
    buf[task_idx.z] = task_idx.z;

    task_idx = get_3d_idx_7_7_4(threadIdx.x);
    buf[task_idx.x] = task_idx.x;
    buf[task_idx.y] = task_idx.y;
    buf[task_idx.z] = task_idx.z;

    task_idx = get_3d_idx_7_2_32(threadIdx.x);
    buf[task_idx.x] = task_idx.x;
    buf[task_idx.y] = task_idx.y;
    buf[task_idx.z] = task_idx.z;

    task_idx = get_3d_idx_14_4_16(threadIdx.x);
    buf[task_idx.x] = task_idx.x;
    buf[task_idx.y] = task_idx.y;
    buf[task_idx.z] = task_idx.z;

    task_idx = get_3d_idx_1_7_8(threadIdx.x);
    buf[task_idx.x] = task_idx.x;
    buf[task_idx.y] = task_idx.y;
    buf[task_idx.z] = task_idx.z;

    task_idx = get_3d_idx_1000_1_1(threadIdx.x);
    buf[task_idx.x] = task_idx.x;
    buf[task_idx.y] = task_idx.y;
    buf[task_idx.z] = task_idx.z;

    task_idx = get_3d_idx_4_4_16(threadIdx.x);
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
