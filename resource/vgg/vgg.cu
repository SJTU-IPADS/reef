#include <hip/hip_runtime.h>

#ifdef _WIN32
  using uint = unsigned int;
  using uchar = unsigned char;
  using ushort = unsigned short;
  using int64_t = long long;
  using uint64_t = unsigned long long;
#else
  #define uint unsigned int
  #define uchar unsigned char
  #define ushort unsigned short
  #define int64_t long long
  #define uint64_t unsigned long long
#endif
extern "C" __global__ void tvmgen_default_fused_nn_conv2d_add_add_nn_relu_2_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2, float* __restrict__ placeholder3) {
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
  for (int rc_outer = 0; rc_outer < 4; ++rc_outer) {
    __syncthreads();
    pad_temp_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))] = placeholder[((((((rc_outer * 4096) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))];
    pad_temp_shared[(((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 1))] = placeholder[(((((((rc_outer * 4096) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 1))];
    pad_temp_shared[(((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 2))] = placeholder[(((((((rc_outer * 4096) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 2))];
    pad_temp_shared[(((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 3))] = placeholder[(((((((rc_outer * 4096) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 3))];
    placeholder_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))] = placeholder1[((((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 64)) + (rc_outer * 16)) + (((int)threadIdx.x) * 4)))];
    placeholder_shared[(((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 1))] = placeholder1[(((((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 64)) + (rc_outer * 16)) + (((int)threadIdx.x) * 4)) + 1))];
    placeholder_shared[(((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 2))] = placeholder1[(((((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 64)) + (rc_outer * 16)) + (((int)threadIdx.x) * 4)) + 2))];
    placeholder_shared[(((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 3))] = placeholder1[(((((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 64)) + (rc_outer * 16)) + (((int)threadIdx.x) * 4)) + 3))];
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
  T_relu[((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))] = max(((compute[(0)] + placeholder2[((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))]) + placeholder3[(((((int)blockIdx.z) * 64) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 4096))] = max(((compute[(4)] + placeholder2[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 4096))]) + placeholder3[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 16))]), 0.000000e+00f);
  T_relu[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 8192))] = max(((compute[(8)] + placeholder2[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 8192))]) + placeholder3[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 32))]), 0.000000e+00f);
  T_relu[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 12288))] = max(((compute[(12)] + placeholder2[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 12288))]) + placeholder3[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 48))]), 0.000000e+00f);
  T_relu[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 1))] = max(((compute[(1)] + placeholder2[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 1))]) + placeholder3[(((((int)blockIdx.z) * 64) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 4097))] = max(((compute[(5)] + placeholder2[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 4097))]) + placeholder3[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 16))]), 0.000000e+00f);
  T_relu[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 8193))] = max(((compute[(9)] + placeholder2[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 8193))]) + placeholder3[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 32))]), 0.000000e+00f);
  T_relu[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 12289))] = max(((compute[(13)] + placeholder2[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 12289))]) + placeholder3[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 48))]), 0.000000e+00f);
  T_relu[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 2))] = max(((compute[(2)] + placeholder2[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 2))]) + placeholder3[(((((int)blockIdx.z) * 64) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 4098))] = max(((compute[(6)] + placeholder2[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 4098))]) + placeholder3[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 16))]), 0.000000e+00f);
  T_relu[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 8194))] = max(((compute[(10)] + placeholder2[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 8194))]) + placeholder3[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 32))]), 0.000000e+00f);
  T_relu[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 12290))] = max(((compute[(14)] + placeholder2[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 12290))]) + placeholder3[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 48))]), 0.000000e+00f);
  T_relu[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 3))] = max(((compute[(3)] + placeholder2[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 3))]) + placeholder3[(((((int)blockIdx.z) * 64) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 4099))] = max(((compute[(7)] + placeholder2[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 4099))]) + placeholder3[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 16))]), 0.000000e+00f);
  T_relu[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 8195))] = max(((compute[(11)] + placeholder2[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 8195))]) + placeholder3[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 32))]), 0.000000e+00f);
  T_relu[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 12291))] = max(((compute[(15)] + placeholder2[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 12291))]) + placeholder3[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 48))]), 0.000000e+00f);
}

extern "C" __global__ void tvmgen_default_fused_nn_conv2d_add_add_nn_relu_1_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2, float* __restrict__ placeholder3) {
  float compute[1];
  __shared__ float pad_temp_shared[2048];
  __shared__ float placeholder_shared[2048];
  compute[(0)] = 0.000000e+00f;
  pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 8)))] = placeholder[(((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 128)) + ((((int)threadIdx.x) >> 1) * 64)) + (((int)blockIdx.y) * 32)) + ((((int)threadIdx.x) & 1) * 16)) + (((int)blockIdx.x) * 4)))];
  pad_temp_shared[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 8)) + 1))] = placeholder[((((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 128)) + ((((int)threadIdx.x) >> 1) * 64)) + (((int)blockIdx.y) * 32)) + ((((int)threadIdx.x) & 1) * 16)) + (((int)blockIdx.x) * 4)) + 1))];
  pad_temp_shared[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 8)) + 2))] = placeholder[((((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 128)) + ((((int)threadIdx.x) >> 1) * 64)) + (((int)blockIdx.y) * 32)) + ((((int)threadIdx.x) & 1) * 16)) + (((int)blockIdx.x) * 4)) + 2))];
  pad_temp_shared[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 8)) + 3))] = placeholder[((((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 128)) + ((((int)threadIdx.x) >> 1) * 64)) + (((int)blockIdx.y) * 32)) + ((((int)threadIdx.x) & 1) * 16)) + (((int)blockIdx.x) * 4)) + 3))];
  pad_temp_shared[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 8)) + 4))] = placeholder[(((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 128)) + ((((((int)threadIdx.x) * 2) + 1) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.x) * 2) + 1) & 3) * 8)) + (((int)blockIdx.x) * 4)))];
  pad_temp_shared[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 8)) + 5))] = placeholder[((((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 128)) + ((((((int)threadIdx.x) * 2) + 1) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.x) * 2) + 1) & 3) * 8)) + (((int)blockIdx.x) * 4)) + 1))];
  pad_temp_shared[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 8)) + 6))] = placeholder[((((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 128)) + ((((((int)threadIdx.x) * 2) + 1) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.x) * 2) + 1) & 3) * 8)) + (((int)blockIdx.x) * 4)) + 2))];
  pad_temp_shared[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 8)) + 7))] = placeholder[((((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 128)) + ((((((int)threadIdx.x) * 2) + 1) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.x) * 2) + 1) & 3) * 8)) + (((int)blockIdx.x) * 4)) + 3))];
  placeholder_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 8)))] = placeholder1[(((((((int)blockIdx.z) * 2048) + (((int)threadIdx.z) * 128)) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 8)))];
  placeholder_shared[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 8)) + 1))] = placeholder1[((((((((int)blockIdx.z) * 2048) + (((int)threadIdx.z) * 128)) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 8)) + 1))];
  placeholder_shared[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 8)) + 2))] = placeholder1[((((((((int)blockIdx.z) * 2048) + (((int)threadIdx.z) * 128)) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 8)) + 2))];
  placeholder_shared[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 8)) + 3))] = placeholder1[((((((((int)blockIdx.z) * 2048) + (((int)threadIdx.z) * 128)) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 8)) + 3))];
  placeholder_shared[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 8)) + 4))] = placeholder1[((((((((int)blockIdx.z) * 2048) + (((int)threadIdx.z) * 128)) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 8)) + 4))];
  placeholder_shared[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 8)) + 5))] = placeholder1[((((((((int)blockIdx.z) * 2048) + (((int)threadIdx.z) * 128)) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 8)) + 5))];
  placeholder_shared[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 8)) + 6))] = placeholder1[((((((((int)blockIdx.z) * 2048) + (((int)threadIdx.z) * 128)) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 8)) + 6))];
  placeholder_shared[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 8)) + 7))] = placeholder1[((((((((int)blockIdx.z) * 2048) + (((int)threadIdx.z) * 128)) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 8)) + 7))];
  __syncthreads();
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)))], placeholder_shared[((((int)threadIdx.z) * 128))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 16))], placeholder_shared[(((((int)threadIdx.z) * 128) + 1))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 32))], placeholder_shared[(((((int)threadIdx.z) * 128) + 2))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 48))], placeholder_shared[(((((int)threadIdx.z) * 128) + 3))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 64))], placeholder_shared[(((((int)threadIdx.z) * 128) + 4))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 80))], placeholder_shared[(((((int)threadIdx.z) * 128) + 5))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 96))], placeholder_shared[(((((int)threadIdx.z) * 128) + 6))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 112))], placeholder_shared[(((((int)threadIdx.z) * 128) + 7))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 128))], placeholder_shared[(((((int)threadIdx.z) * 128) + 8))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 144))], placeholder_shared[(((((int)threadIdx.z) * 128) + 9))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 160))], placeholder_shared[(((((int)threadIdx.z) * 128) + 10))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 176))], placeholder_shared[(((((int)threadIdx.z) * 128) + 11))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 192))], placeholder_shared[(((((int)threadIdx.z) * 128) + 12))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 208))], placeholder_shared[(((((int)threadIdx.z) * 128) + 13))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 224))], placeholder_shared[(((((int)threadIdx.z) * 128) + 14))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 240))], placeholder_shared[(((((int)threadIdx.z) * 128) + 15))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 256))], placeholder_shared[(((((int)threadIdx.z) * 128) + 16))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 272))], placeholder_shared[(((((int)threadIdx.z) * 128) + 17))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 288))], placeholder_shared[(((((int)threadIdx.z) * 128) + 18))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 304))], placeholder_shared[(((((int)threadIdx.z) * 128) + 19))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 320))], placeholder_shared[(((((int)threadIdx.z) * 128) + 20))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 336))], placeholder_shared[(((((int)threadIdx.z) * 128) + 21))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 352))], placeholder_shared[(((((int)threadIdx.z) * 128) + 22))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 368))], placeholder_shared[(((((int)threadIdx.z) * 128) + 23))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 384))], placeholder_shared[(((((int)threadIdx.z) * 128) + 24))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 400))], placeholder_shared[(((((int)threadIdx.z) * 128) + 25))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 416))], placeholder_shared[(((((int)threadIdx.z) * 128) + 26))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 432))], placeholder_shared[(((((int)threadIdx.z) * 128) + 27))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 448))], placeholder_shared[(((((int)threadIdx.z) * 128) + 28))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 464))], placeholder_shared[(((((int)threadIdx.z) * 128) + 29))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 480))], placeholder_shared[(((((int)threadIdx.z) * 128) + 30))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 496))], placeholder_shared[(((((int)threadIdx.z) * 128) + 31))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 512))], placeholder_shared[(((((int)threadIdx.z) * 128) + 32))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 528))], placeholder_shared[(((((int)threadIdx.z) * 128) + 33))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 544))], placeholder_shared[(((((int)threadIdx.z) * 128) + 34))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 560))], placeholder_shared[(((((int)threadIdx.z) * 128) + 35))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 576))], placeholder_shared[(((((int)threadIdx.z) * 128) + 36))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 592))], placeholder_shared[(((((int)threadIdx.z) * 128) + 37))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 608))], placeholder_shared[(((((int)threadIdx.z) * 128) + 38))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 624))], placeholder_shared[(((((int)threadIdx.z) * 128) + 39))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 640))], placeholder_shared[(((((int)threadIdx.z) * 128) + 40))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 656))], placeholder_shared[(((((int)threadIdx.z) * 128) + 41))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 672))], placeholder_shared[(((((int)threadIdx.z) * 128) + 42))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 688))], placeholder_shared[(((((int)threadIdx.z) * 128) + 43))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 704))], placeholder_shared[(((((int)threadIdx.z) * 128) + 44))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 720))], placeholder_shared[(((((int)threadIdx.z) * 128) + 45))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 736))], placeholder_shared[(((((int)threadIdx.z) * 128) + 46))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 752))], placeholder_shared[(((((int)threadIdx.z) * 128) + 47))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 768))], placeholder_shared[(((((int)threadIdx.z) * 128) + 48))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 784))], placeholder_shared[(((((int)threadIdx.z) * 128) + 49))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 800))], placeholder_shared[(((((int)threadIdx.z) * 128) + 50))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 816))], placeholder_shared[(((((int)threadIdx.z) * 128) + 51))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 832))], placeholder_shared[(((((int)threadIdx.z) * 128) + 52))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 848))], placeholder_shared[(((((int)threadIdx.z) * 128) + 53))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 864))], placeholder_shared[(((((int)threadIdx.z) * 128) + 54))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 880))], placeholder_shared[(((((int)threadIdx.z) * 128) + 55))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 896))], placeholder_shared[(((((int)threadIdx.z) * 128) + 56))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 912))], placeholder_shared[(((((int)threadIdx.z) * 128) + 57))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 928))], placeholder_shared[(((((int)threadIdx.z) * 128) + 58))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 944))], placeholder_shared[(((((int)threadIdx.z) * 128) + 59))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 960))], placeholder_shared[(((((int)threadIdx.z) * 128) + 60))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 976))], placeholder_shared[(((((int)threadIdx.z) * 128) + 61))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 992))], placeholder_shared[(((((int)threadIdx.z) * 128) + 62))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1008))], placeholder_shared[(((((int)threadIdx.z) * 128) + 63))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1024))], placeholder_shared[(((((int)threadIdx.z) * 128) + 64))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1040))], placeholder_shared[(((((int)threadIdx.z) * 128) + 65))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1056))], placeholder_shared[(((((int)threadIdx.z) * 128) + 66))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1072))], placeholder_shared[(((((int)threadIdx.z) * 128) + 67))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1088))], placeholder_shared[(((((int)threadIdx.z) * 128) + 68))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1104))], placeholder_shared[(((((int)threadIdx.z) * 128) + 69))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1120))], placeholder_shared[(((((int)threadIdx.z) * 128) + 70))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1136))], placeholder_shared[(((((int)threadIdx.z) * 128) + 71))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1152))], placeholder_shared[(((((int)threadIdx.z) * 128) + 72))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1168))], placeholder_shared[(((((int)threadIdx.z) * 128) + 73))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1184))], placeholder_shared[(((((int)threadIdx.z) * 128) + 74))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1200))], placeholder_shared[(((((int)threadIdx.z) * 128) + 75))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1216))], placeholder_shared[(((((int)threadIdx.z) * 128) + 76))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1232))], placeholder_shared[(((((int)threadIdx.z) * 128) + 77))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1248))], placeholder_shared[(((((int)threadIdx.z) * 128) + 78))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1264))], placeholder_shared[(((((int)threadIdx.z) * 128) + 79))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1280))], placeholder_shared[(((((int)threadIdx.z) * 128) + 80))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1296))], placeholder_shared[(((((int)threadIdx.z) * 128) + 81))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1312))], placeholder_shared[(((((int)threadIdx.z) * 128) + 82))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1328))], placeholder_shared[(((((int)threadIdx.z) * 128) + 83))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1344))], placeholder_shared[(((((int)threadIdx.z) * 128) + 84))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1360))], placeholder_shared[(((((int)threadIdx.z) * 128) + 85))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1376))], placeholder_shared[(((((int)threadIdx.z) * 128) + 86))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1392))], placeholder_shared[(((((int)threadIdx.z) * 128) + 87))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1408))], placeholder_shared[(((((int)threadIdx.z) * 128) + 88))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1424))], placeholder_shared[(((((int)threadIdx.z) * 128) + 89))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1440))], placeholder_shared[(((((int)threadIdx.z) * 128) + 90))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1456))], placeholder_shared[(((((int)threadIdx.z) * 128) + 91))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1472))], placeholder_shared[(((((int)threadIdx.z) * 128) + 92))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1488))], placeholder_shared[(((((int)threadIdx.z) * 128) + 93))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1504))], placeholder_shared[(((((int)threadIdx.z) * 128) + 94))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1520))], placeholder_shared[(((((int)threadIdx.z) * 128) + 95))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1536))], placeholder_shared[(((((int)threadIdx.z) * 128) + 96))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1552))], placeholder_shared[(((((int)threadIdx.z) * 128) + 97))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1568))], placeholder_shared[(((((int)threadIdx.z) * 128) + 98))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1584))], placeholder_shared[(((((int)threadIdx.z) * 128) + 99))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1600))], placeholder_shared[(((((int)threadIdx.z) * 128) + 100))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1616))], placeholder_shared[(((((int)threadIdx.z) * 128) + 101))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1632))], placeholder_shared[(((((int)threadIdx.z) * 128) + 102))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1648))], placeholder_shared[(((((int)threadIdx.z) * 128) + 103))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1664))], placeholder_shared[(((((int)threadIdx.z) * 128) + 104))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1680))], placeholder_shared[(((((int)threadIdx.z) * 128) + 105))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1696))], placeholder_shared[(((((int)threadIdx.z) * 128) + 106))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1712))], placeholder_shared[(((((int)threadIdx.z) * 128) + 107))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1728))], placeholder_shared[(((((int)threadIdx.z) * 128) + 108))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1744))], placeholder_shared[(((((int)threadIdx.z) * 128) + 109))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1760))], placeholder_shared[(((((int)threadIdx.z) * 128) + 110))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1776))], placeholder_shared[(((((int)threadIdx.z) * 128) + 111))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1792))], placeholder_shared[(((((int)threadIdx.z) * 128) + 112))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1808))], placeholder_shared[(((((int)threadIdx.z) * 128) + 113))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1824))], placeholder_shared[(((((int)threadIdx.z) * 128) + 114))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1840))], placeholder_shared[(((((int)threadIdx.z) * 128) + 115))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1856))], placeholder_shared[(((((int)threadIdx.z) * 128) + 116))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1872))], placeholder_shared[(((((int)threadIdx.z) * 128) + 117))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1888))], placeholder_shared[(((((int)threadIdx.z) * 128) + 118))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1904))], placeholder_shared[(((((int)threadIdx.z) * 128) + 119))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1920))], placeholder_shared[(((((int)threadIdx.z) * 128) + 120))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1936))], placeholder_shared[(((((int)threadIdx.z) * 128) + 121))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1952))], placeholder_shared[(((((int)threadIdx.z) * 128) + 122))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1968))], placeholder_shared[(((((int)threadIdx.z) * 128) + 123))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1984))], placeholder_shared[(((((int)threadIdx.z) * 128) + 124))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2000))], placeholder_shared[(((((int)threadIdx.z) * 128) + 125))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2016))], placeholder_shared[(((((int)threadIdx.z) * 128) + 126))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2032))], placeholder_shared[(((((int)threadIdx.z) * 128) + 127))], compute[(0)]);
  T_relu[(((((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 64)) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.y) * 8)) + (((int)blockIdx.x) * 4)) + ((int)threadIdx.x)))] = max(((compute[(0)] + placeholder2[(((((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 64)) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.y) * 8)) + (((int)blockIdx.x) * 4)) + ((int)threadIdx.x)))]) + placeholder3[(((((int)blockIdx.z) * 16) + ((int)threadIdx.z)))]), 0.000000e+00f);
}

extern "C" __global__ void tvmgen_default_fused_add_kernel0(float* __restrict__ T_add, float* __restrict__ placeholder, float* __restrict__ placeholder1) {
  T_add[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)))] = (placeholder[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)))] + placeholder1[((((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 6)) >> 6))]);
}

extern "C" __global__ void tvmgen_default_fused_nn_conv2d_add_nn_relu_4_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  float compute[4];
  __shared__ float pad_temp_shared[256];
  __shared__ float placeholder_shared[512];
  compute[(0)] = 0.000000e+00f;
  compute[(2)] = 0.000000e+00f;
  compute[(1)] = 0.000000e+00f;
  compute[(3)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 64; ++rc_outer) {
    __syncthreads();
    pad_temp_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 4)) + (((int)threadIdx.x) * 2)))] = placeholder[(((((rc_outer * 256) + (((int)threadIdx.z) * 16)) + (((int)threadIdx.y) * 4)) + (((int)threadIdx.x) * 2)))];
    pad_temp_shared[(((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 4)) + (((int)threadIdx.x) * 2)) + 1))] = placeholder[((((((rc_outer * 256) + (((int)threadIdx.z) * 16)) + (((int)threadIdx.y) * 4)) + (((int)threadIdx.x) * 2)) + 1))];
    placeholder_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 4)))] = placeholder1[((((((((int)blockIdx.z) * 32768) + (((int)threadIdx.z) * 2048)) + ((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 4)) >> 4) * 1024)) + (rc_outer * 16)) + (((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 4)) & 15)))];
    placeholder_shared[(((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 4)) + 1))] = placeholder1[((((((((int)blockIdx.z) * 32768) + (((int)threadIdx.z) * 2048)) + (((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 4)) + 1) >> 4) * 1024)) + (rc_outer * 16)) + ((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 4)) + 1) & 15)))];
    placeholder_shared[(((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 4)) + 2))] = placeholder1[((((((((int)blockIdx.z) * 32768) + (((int)threadIdx.z) * 2048)) + (((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 4)) + 2) >> 4) * 1024)) + (rc_outer * 16)) + ((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 4)) + 2) & 15)))];
    placeholder_shared[(((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 4)) + 3))] = placeholder1[((((((((int)blockIdx.z) * 32768) + (((int)threadIdx.z) * 2048)) + (((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 4)) + 3) >> 4) * 1024)) + (rc_outer * 16)) + ((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 4)) + 3) & 15)))];
    __syncthreads();
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)))], placeholder_shared[((((int)threadIdx.z) * 16))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)))], placeholder_shared[(((((int)threadIdx.z) * 16) + 256))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + 1))], placeholder_shared[((((int)threadIdx.z) * 16))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + 1))], placeholder_shared[(((((int)threadIdx.z) * 16) + 256))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + 16))], placeholder_shared[(((((int)threadIdx.z) * 16) + 1))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + 16))], placeholder_shared[(((((int)threadIdx.z) * 16) + 257))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + 17))], placeholder_shared[(((((int)threadIdx.z) * 16) + 1))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + 17))], placeholder_shared[(((((int)threadIdx.z) * 16) + 257))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + 32))], placeholder_shared[(((((int)threadIdx.z) * 16) + 2))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + 32))], placeholder_shared[(((((int)threadIdx.z) * 16) + 258))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + 33))], placeholder_shared[(((((int)threadIdx.z) * 16) + 2))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + 33))], placeholder_shared[(((((int)threadIdx.z) * 16) + 258))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + 48))], placeholder_shared[(((((int)threadIdx.z) * 16) + 3))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + 48))], placeholder_shared[(((((int)threadIdx.z) * 16) + 259))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + 49))], placeholder_shared[(((((int)threadIdx.z) * 16) + 3))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + 49))], placeholder_shared[(((((int)threadIdx.z) * 16) + 259))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + 64))], placeholder_shared[(((((int)threadIdx.z) * 16) + 4))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + 64))], placeholder_shared[(((((int)threadIdx.z) * 16) + 260))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + 65))], placeholder_shared[(((((int)threadIdx.z) * 16) + 4))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + 65))], placeholder_shared[(((((int)threadIdx.z) * 16) + 260))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + 80))], placeholder_shared[(((((int)threadIdx.z) * 16) + 5))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + 80))], placeholder_shared[(((((int)threadIdx.z) * 16) + 261))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + 81))], placeholder_shared[(((((int)threadIdx.z) * 16) + 5))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + 81))], placeholder_shared[(((((int)threadIdx.z) * 16) + 261))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + 96))], placeholder_shared[(((((int)threadIdx.z) * 16) + 6))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + 96))], placeholder_shared[(((((int)threadIdx.z) * 16) + 262))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + 97))], placeholder_shared[(((((int)threadIdx.z) * 16) + 6))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + 97))], placeholder_shared[(((((int)threadIdx.z) * 16) + 262))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + 112))], placeholder_shared[(((((int)threadIdx.z) * 16) + 7))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + 112))], placeholder_shared[(((((int)threadIdx.z) * 16) + 263))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + 113))], placeholder_shared[(((((int)threadIdx.z) * 16) + 7))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + 113))], placeholder_shared[(((((int)threadIdx.z) * 16) + 263))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + 128))], placeholder_shared[(((((int)threadIdx.z) * 16) + 8))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + 128))], placeholder_shared[(((((int)threadIdx.z) * 16) + 264))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + 129))], placeholder_shared[(((((int)threadIdx.z) * 16) + 8))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + 129))], placeholder_shared[(((((int)threadIdx.z) * 16) + 264))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + 144))], placeholder_shared[(((((int)threadIdx.z) * 16) + 9))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + 144))], placeholder_shared[(((((int)threadIdx.z) * 16) + 265))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + 145))], placeholder_shared[(((((int)threadIdx.z) * 16) + 9))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + 145))], placeholder_shared[(((((int)threadIdx.z) * 16) + 265))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + 160))], placeholder_shared[(((((int)threadIdx.z) * 16) + 10))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + 160))], placeholder_shared[(((((int)threadIdx.z) * 16) + 266))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + 161))], placeholder_shared[(((((int)threadIdx.z) * 16) + 10))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + 161))], placeholder_shared[(((((int)threadIdx.z) * 16) + 266))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + 176))], placeholder_shared[(((((int)threadIdx.z) * 16) + 11))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + 176))], placeholder_shared[(((((int)threadIdx.z) * 16) + 267))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + 177))], placeholder_shared[(((((int)threadIdx.z) * 16) + 11))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + 177))], placeholder_shared[(((((int)threadIdx.z) * 16) + 267))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + 192))], placeholder_shared[(((((int)threadIdx.z) * 16) + 12))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + 192))], placeholder_shared[(((((int)threadIdx.z) * 16) + 268))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + 193))], placeholder_shared[(((((int)threadIdx.z) * 16) + 12))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + 193))], placeholder_shared[(((((int)threadIdx.z) * 16) + 268))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + 208))], placeholder_shared[(((((int)threadIdx.z) * 16) + 13))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + 208))], placeholder_shared[(((((int)threadIdx.z) * 16) + 269))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + 209))], placeholder_shared[(((((int)threadIdx.z) * 16) + 13))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + 209))], placeholder_shared[(((((int)threadIdx.z) * 16) + 269))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + 224))], placeholder_shared[(((((int)threadIdx.z) * 16) + 14))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + 224))], placeholder_shared[(((((int)threadIdx.z) * 16) + 270))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + 225))], placeholder_shared[(((((int)threadIdx.z) * 16) + 14))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + 225))], placeholder_shared[(((((int)threadIdx.z) * 16) + 270))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + 240))], placeholder_shared[(((((int)threadIdx.z) * 16) + 15))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + 240))], placeholder_shared[(((((int)threadIdx.z) * 16) + 271))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + 241))], placeholder_shared[(((((int)threadIdx.z) * 16) + 15))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + 241))], placeholder_shared[(((((int)threadIdx.z) * 16) + 271))], compute[(3)]);
  }
  T_relu[(((((((int)blockIdx.z) * 512) + (((int)threadIdx.z) * 16)) + (((int)threadIdx.y) * 4)) + (((int)threadIdx.x) * 2)))] = max((compute[(0)] + placeholder2[(((((int)blockIdx.z) * 32) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 512) + (((int)threadIdx.z) * 16)) + (((int)threadIdx.y) * 4)) + (((int)threadIdx.x) * 2)) + 256))] = max((compute[(2)] + placeholder2[((((((int)blockIdx.z) * 32) + ((int)threadIdx.z)) + 16))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 512) + (((int)threadIdx.z) * 16)) + (((int)threadIdx.y) * 4)) + (((int)threadIdx.x) * 2)) + 1))] = max((compute[(1)] + placeholder2[(((((int)blockIdx.z) * 32) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 512) + (((int)threadIdx.z) * 16)) + (((int)threadIdx.y) * 4)) + (((int)threadIdx.x) * 2)) + 257))] = max((compute[(3)] + placeholder2[((((((int)blockIdx.z) * 32) + ((int)threadIdx.z)) + 16))]), 0.000000e+00f);
}

extern "C" __global__ void tvmgen_default_fused_nn_conv2d_add_nn_relu_9_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  float compute[1];
  __shared__ float pad_temp_shared[192];
  __shared__ float placeholder_shared[2304];
  compute[(0)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 8; ++rc_outer) {
    __syncthreads();
    if (((((((int)threadIdx.y) * 3) + ((int)threadIdx.x)) / 6) + ((int)threadIdx.z)) < 32) {
      if ((((((int)threadIdx.z) * 6) + (((int)threadIdx.y) * 3)) + ((int)threadIdx.x)) < 192) {
        if (((((int)threadIdx.y) * 3) + ((int)threadIdx.x)) < 6) {
          if (((int)threadIdx.x) < 3) {
            pad_temp_shared[((((((int)threadIdx.z) * 6) + (((int)threadIdx.y) * 3)) + ((int)threadIdx.x)))] = (((((1 <= ((((int)blockIdx.y) * 2) + (((int)threadIdx.z) & 3))) && (((((int)blockIdx.y) * 2) + (((int)threadIdx.z) & 3)) < 17)) && (1 <= (((((int)blockIdx.x) * 4) + (((int)threadIdx.y) * 3)) + ((int)threadIdx.x)))) && ((((((int)blockIdx.x) * 4) + (((int)threadIdx.y) * 3)) + ((int)threadIdx.x)) < 17)) ? placeholder[(((((((((rc_outer * 2048) + ((((int)threadIdx.z) >> 2) * 256)) + (((int)blockIdx.y) * 32)) + ((((int)threadIdx.z) & 3) * 16)) + (((int)blockIdx.x) * 4)) + (((int)threadIdx.y) * 3)) + ((int)threadIdx.x)) - 17))] : 0.000000e+00f);
          }
        }
      }
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 9; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      placeholder_shared[(((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 36)) + (((int)threadIdx.x) * 9)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))] = placeholder1[(((((((((int)blockIdx.z) * 18432) + (((int)threadIdx.z) * 576)) + (rc_outer * 72)) + (((int)threadIdx.y) * 36)) + (((int)threadIdx.x) * 9)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))];
    }
    __syncthreads();
    for (int rc_inner = 0; rc_inner < 8; ++rc_inner) {
      for (int ry_inner = 0; ry_inner < 3; ++ry_inner) {
        for (int rx_inner = 0; rx_inner < 3; ++rx_inner) {
          compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((rc_inner * 24) + (((int)threadIdx.y) * 6)) + (ry_inner * 6)) + ((int)threadIdx.x)) + rx_inner))], placeholder_shared[(((((((int)threadIdx.z) * 72) + (rc_inner * 9)) + (ry_inner * 3)) + rx_inner))], compute[(0)]);
        }
      }
    }
  }
  T_relu[(((((((((int)blockIdx.z) * 8192) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.y) * 16)) + (((int)blockIdx.x) * 4)) + ((int)threadIdx.x)))] = max((compute[(0)] + placeholder2[(((((int)blockIdx.z) * 32) + ((int)threadIdx.z)))]), 0.000000e+00f);
}

extern "C" __global__ void tvmgen_default_fused_nn_conv2d_1_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ compute) {
  float compute_local[1];
  __shared__ float pad_temp_shared[6272];
  __shared__ float placeholder_shared[2048];
  compute_local[(0)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 2; ++rc_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 25; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      if ((((((int)threadIdx.z) * 8) + (((int)threadIdx.y) * 2)) + (((((int)threadIdx.x) * 25) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 49)) < 128) {
        if ((((((int)threadIdx.z) * 56) + (((int)threadIdx.y) * 14)) + (((((int)threadIdx.x) * 25) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 7)) < 896) {
          if (((((((int)threadIdx.z) * 392) + (((int)threadIdx.y) * 98)) + (((int)threadIdx.x) * 25)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 6272) {
            if ((((((int)threadIdx.y) * 98) + (((int)threadIdx.x) * 25)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 392) {
              if (((((int)threadIdx.x) * 25) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 98) {
                pad_temp_shared[(((((((int)threadIdx.z) * 392) + (((int)threadIdx.y) * 98)) + (((int)threadIdx.x) * 25)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))] = placeholder[(((((((((rc_outer * 32768) + (((int)threadIdx.z) * 2048)) + (((int)threadIdx.y) * 512)) + ((((((int)threadIdx.x) * 25) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 49) * 256)) + (((int)blockIdx.y) * 128)) + (((((((int)threadIdx.x) * 25) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 49) / 7) * 16)) + (((int)blockIdx.x) * 8)) + (((((int)threadIdx.x) * 25) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 7)))];
              }
            }
          }
        }
      }
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 < 8; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) {
      placeholder_shared[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1))] = placeholder1[(((((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) + (rc_outer * 128)) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1))];
    }
    __syncthreads();
    for (int rc_inner = 0; rc_inner < 128; ++rc_inner) {
      compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((rc_inner * 49) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)))], placeholder_shared[(((((int)threadIdx.z) * 128) + rc_inner))], compute_local[(0)]);
    }
  }
  compute[(((((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 64)) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.y) * 8)) + (((int)blockIdx.x) * 4)) + ((int)threadIdx.x)))] = compute_local[(0)];
}

extern "C" __global__ void tvmgen_default_fused_nn_conv2d_add_nn_relu_5_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  float compute[4];
  __shared__ float pad_temp_shared[784];
  __shared__ float placeholder_shared[512];
  compute[(0)] = 0.000000e+00f;
  compute[(2)] = 0.000000e+00f;
  compute[(1)] = 0.000000e+00f;
  compute[(3)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 32; ++rc_outer) {
    __syncthreads();
    pad_temp_shared[((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 13)) + (((int)threadIdx.x) * 7)))] = placeholder[((((((rc_outer * 1024) + (((int)threadIdx.z) * 64)) + (((((int)threadIdx.y) * 13) / 7) * 8)) + (((int)threadIdx.x) * 8)) + ((((int)threadIdx.y) * 13) % 7)))];
    pad_temp_shared[(((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 13)) + (((int)threadIdx.x) * 7)) + 1))] = placeholder[((((((rc_outer * 1024) + (((int)threadIdx.z) * 64)) + ((((((int)threadIdx.y) * 13) + 1) / 7) * 8)) + (((int)threadIdx.x) * 8)) + (((((int)threadIdx.y) * 13) + 1) % 7)))];
    pad_temp_shared[(((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 13)) + (((int)threadIdx.x) * 7)) + 2))] = placeholder[((((((rc_outer * 1024) + (((int)threadIdx.z) * 64)) + ((((((int)threadIdx.y) * 13) + 2) / 7) * 8)) + (((int)threadIdx.x) * 8)) + (((((int)threadIdx.y) * 13) + 2) % 7)))];
    if (((((((((int)threadIdx.y) * 13) + 3) / 7) + ((int)threadIdx.x)) / 7) + ((int)threadIdx.z)) < 16) {
      if ((((((int)threadIdx.z) * 7) + (((((int)threadIdx.y) * 13) + 3) / 7)) + ((int)threadIdx.x)) < 112) {
        if ((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 13)) + (((int)threadIdx.x) * 7)) < 781) {
          if (((((int)threadIdx.y) * 13) + (((int)threadIdx.x) * 7)) < 46) {
            pad_temp_shared[(((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 13)) + (((int)threadIdx.x) * 7)) + 3))] = placeholder[((((((rc_outer * 1024) + ((((((((int)threadIdx.y) * 13) + 3) / 7) + ((int)threadIdx.x)) / 7) * 64)) + (((int)threadIdx.z) * 64)) + ((((((((int)threadIdx.y) * 13) + 3) / 7) + ((int)threadIdx.x)) % 7) * 8)) + (((((int)threadIdx.y) * 13) + 3) % 7)))];
          }
        }
      }
    }
    if (((((((((int)threadIdx.y) * 13) + 4) / 7) + ((int)threadIdx.x)) / 7) + ((int)threadIdx.z)) < 16) {
      if ((((((int)threadIdx.z) * 7) + (((((int)threadIdx.y) * 13) + 4) / 7)) + ((int)threadIdx.x)) < 112) {
        if ((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 13)) + (((int)threadIdx.x) * 7)) < 780) {
          if (((((int)threadIdx.y) * 13) + (((int)threadIdx.x) * 7)) < 45) {
            pad_temp_shared[(((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 13)) + (((int)threadIdx.x) * 7)) + 4))] = placeholder[((((((rc_outer * 1024) + ((((((((int)threadIdx.y) * 13) + 4) / 7) + ((int)threadIdx.x)) / 7) * 64)) + (((int)threadIdx.z) * 64)) + ((((((((int)threadIdx.y) * 13) + 4) / 7) + ((int)threadIdx.x)) % 7) * 8)) + (((((int)threadIdx.y) * 13) + 4) % 7)))];
          }
        }
      }
    }
    if (((((((((int)threadIdx.y) * 13) + 5) / 7) + ((int)threadIdx.x)) / 7) + ((int)threadIdx.z)) < 16) {
      if ((((((int)threadIdx.z) * 7) + (((((int)threadIdx.y) * 13) + 5) / 7)) + ((int)threadIdx.x)) < 112) {
        if ((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 13)) + (((int)threadIdx.x) * 7)) < 779) {
          if (((((int)threadIdx.y) * 13) + (((int)threadIdx.x) * 7)) < 44) {
            pad_temp_shared[(((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 13)) + (((int)threadIdx.x) * 7)) + 5))] = placeholder[((((((rc_outer * 1024) + ((((((((int)threadIdx.y) * 13) + 5) / 7) + ((int)threadIdx.x)) / 7) * 64)) + (((int)threadIdx.z) * 64)) + ((((((((int)threadIdx.y) * 13) + 5) / 7) + ((int)threadIdx.x)) % 7) * 8)) + (((((int)threadIdx.y) * 13) + 5) % 7)))];
          }
        }
      }
    }
    if (((((((((int)threadIdx.y) * 13) + 6) / 7) + ((int)threadIdx.x)) / 7) + ((int)threadIdx.z)) < 16) {
      if ((((((int)threadIdx.z) * 7) + (((((int)threadIdx.y) * 13) + 6) / 7)) + ((int)threadIdx.x)) < 112) {
        if ((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 13)) + (((int)threadIdx.x) * 7)) < 778) {
          if (((((int)threadIdx.y) * 13) + (((int)threadIdx.x) * 7)) < 43) {
            if (((int)threadIdx.x) < 1) {
              pad_temp_shared[(((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 13)) + (((int)threadIdx.x) * 7)) + 6))] = placeholder[((((((rc_outer * 1024) + (((int)threadIdx.z) * 64)) + ((((((int)threadIdx.y) * 13) + 6) / 7) * 8)) + (((int)threadIdx.x) * 8)) + (((((int)threadIdx.y) * 13) + 6) % 7)))];
            }
          }
        }
      }
    }
    placeholder_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 4)))] = placeholder1[((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 1024)) + ((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 4)) >> 4) * 512)) + (rc_outer * 16)) + (((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 4)) & 15)))];
    placeholder_shared[(((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 4)) + 1))] = placeholder1[((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 1024)) + (((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 4)) + 1) >> 4) * 512)) + (rc_outer * 16)) + ((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 4)) + 1) & 15)))];
    placeholder_shared[(((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 4)) + 2))] = placeholder1[((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 1024)) + (((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 4)) + 2) >> 4) * 512)) + (rc_outer * 16)) + ((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 4)) + 2) & 15)))];
    placeholder_shared[(((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 4)) + 3))] = placeholder1[((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 1024)) + (((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 4)) + 3) >> 4) * 512)) + (rc_outer * 16)) + ((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 4)) + 3) & 15)))];
    __syncthreads();
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)))], placeholder_shared[((((int)threadIdx.z) * 16))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)))], placeholder_shared[(((((int)threadIdx.z) * 16) + 256))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 2))], placeholder_shared[((((int)threadIdx.z) * 16))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 2))], placeholder_shared[(((((int)threadIdx.z) * 16) + 256))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 49))], placeholder_shared[(((((int)threadIdx.z) * 16) + 1))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 49))], placeholder_shared[(((((int)threadIdx.z) * 16) + 257))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 51))], placeholder_shared[(((((int)threadIdx.z) * 16) + 1))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 51))], placeholder_shared[(((((int)threadIdx.z) * 16) + 257))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 98))], placeholder_shared[(((((int)threadIdx.z) * 16) + 2))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 98))], placeholder_shared[(((((int)threadIdx.z) * 16) + 258))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 100))], placeholder_shared[(((((int)threadIdx.z) * 16) + 2))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 100))], placeholder_shared[(((((int)threadIdx.z) * 16) + 258))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 147))], placeholder_shared[(((((int)threadIdx.z) * 16) + 3))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 147))], placeholder_shared[(((((int)threadIdx.z) * 16) + 259))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 149))], placeholder_shared[(((((int)threadIdx.z) * 16) + 3))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 149))], placeholder_shared[(((((int)threadIdx.z) * 16) + 259))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 196))], placeholder_shared[(((((int)threadIdx.z) * 16) + 4))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 196))], placeholder_shared[(((((int)threadIdx.z) * 16) + 260))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 198))], placeholder_shared[(((((int)threadIdx.z) * 16) + 4))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 198))], placeholder_shared[(((((int)threadIdx.z) * 16) + 260))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 245))], placeholder_shared[(((((int)threadIdx.z) * 16) + 5))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 245))], placeholder_shared[(((((int)threadIdx.z) * 16) + 261))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 247))], placeholder_shared[(((((int)threadIdx.z) * 16) + 5))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 247))], placeholder_shared[(((((int)threadIdx.z) * 16) + 261))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 294))], placeholder_shared[(((((int)threadIdx.z) * 16) + 6))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 294))], placeholder_shared[(((((int)threadIdx.z) * 16) + 262))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 296))], placeholder_shared[(((((int)threadIdx.z) * 16) + 6))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 296))], placeholder_shared[(((((int)threadIdx.z) * 16) + 262))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 343))], placeholder_shared[(((((int)threadIdx.z) * 16) + 7))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 343))], placeholder_shared[(((((int)threadIdx.z) * 16) + 263))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 345))], placeholder_shared[(((((int)threadIdx.z) * 16) + 7))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 345))], placeholder_shared[(((((int)threadIdx.z) * 16) + 263))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 392))], placeholder_shared[(((((int)threadIdx.z) * 16) + 8))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 392))], placeholder_shared[(((((int)threadIdx.z) * 16) + 264))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 394))], placeholder_shared[(((((int)threadIdx.z) * 16) + 8))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 394))], placeholder_shared[(((((int)threadIdx.z) * 16) + 264))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 441))], placeholder_shared[(((((int)threadIdx.z) * 16) + 9))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 441))], placeholder_shared[(((((int)threadIdx.z) * 16) + 265))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 443))], placeholder_shared[(((((int)threadIdx.z) * 16) + 9))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 443))], placeholder_shared[(((((int)threadIdx.z) * 16) + 265))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 490))], placeholder_shared[(((((int)threadIdx.z) * 16) + 10))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 490))], placeholder_shared[(((((int)threadIdx.z) * 16) + 266))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 492))], placeholder_shared[(((((int)threadIdx.z) * 16) + 10))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 492))], placeholder_shared[(((((int)threadIdx.z) * 16) + 266))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 539))], placeholder_shared[(((((int)threadIdx.z) * 16) + 11))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 539))], placeholder_shared[(((((int)threadIdx.z) * 16) + 267))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 541))], placeholder_shared[(((((int)threadIdx.z) * 16) + 11))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 541))], placeholder_shared[(((((int)threadIdx.z) * 16) + 267))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 588))], placeholder_shared[(((((int)threadIdx.z) * 16) + 12))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 588))], placeholder_shared[(((((int)threadIdx.z) * 16) + 268))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 590))], placeholder_shared[(((((int)threadIdx.z) * 16) + 12))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 590))], placeholder_shared[(((((int)threadIdx.z) * 16) + 268))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 637))], placeholder_shared[(((((int)threadIdx.z) * 16) + 13))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 637))], placeholder_shared[(((((int)threadIdx.z) * 16) + 269))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 639))], placeholder_shared[(((((int)threadIdx.z) * 16) + 13))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 639))], placeholder_shared[(((((int)threadIdx.z) * 16) + 269))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 686))], placeholder_shared[(((((int)threadIdx.z) * 16) + 14))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 686))], placeholder_shared[(((((int)threadIdx.z) * 16) + 270))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 688))], placeholder_shared[(((((int)threadIdx.z) * 16) + 14))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 688))], placeholder_shared[(((((int)threadIdx.z) * 16) + 270))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 735))], placeholder_shared[(((((int)threadIdx.z) * 16) + 15))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 735))], placeholder_shared[(((((int)threadIdx.z) * 16) + 271))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 737))], placeholder_shared[(((((int)threadIdx.z) * 16) + 15))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 737))], placeholder_shared[(((((int)threadIdx.z) * 16) + 271))], compute[(3)]);
  }
  T_relu[(((((((int)blockIdx.z) * 512) + (((int)threadIdx.z) * 16)) + (((int)threadIdx.y) * 4)) + (((int)threadIdx.x) * 2)))] = max((compute[(0)] + placeholder2[(((((int)blockIdx.z) * 32) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 512) + (((int)threadIdx.z) * 16)) + (((int)threadIdx.y) * 4)) + (((int)threadIdx.x) * 2)) + 256))] = max((compute[(2)] + placeholder2[((((((int)blockIdx.z) * 32) + ((int)threadIdx.z)) + 16))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 512) + (((int)threadIdx.z) * 16)) + (((int)threadIdx.y) * 4)) + (((int)threadIdx.x) * 2)) + 1))] = max((compute[(1)] + placeholder2[(((((int)blockIdx.z) * 32) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 512) + (((int)threadIdx.z) * 16)) + (((int)threadIdx.y) * 4)) + (((int)threadIdx.x) * 2)) + 257))] = max((compute[(3)] + placeholder2[((((((int)blockIdx.z) * 32) + ((int)threadIdx.z)) + 16))]), 0.000000e+00f);
}

extern "C" __global__ void tvmgen_default_fused_add_nn_relu_3_kernel0(float* __restrict__ T_relu, float* __restrict__ placeholder, float* __restrict__ placeholder1) {
  T_relu[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)))] = max((placeholder[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)))] + placeholder1[(((int)blockIdx.x))]), 0.000000e+00f);
}

extern "C" __global__ void tvmgen_default_fused_nn_conv2d_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ compute) {
  float compute_local[16];
  __shared__ float pad_temp_shared[1024];
  __shared__ float placeholder_shared[1024];
  compute_local[(0)] = 0.000000e+00f;
  compute_local[(4)] = 0.000000e+00f;
  compute_local[(8)] = 0.000000e+00f;
  compute_local[(12)] = 0.000000e+00f;
  compute_local[(1)] = 0.000000e+00f;
  compute_local[(5)] = 0.000000e+00f;
  compute_local[(9)] = 0.000000e+00f;
  compute_local[(13)] = 0.000000e+00f;
  compute_local[(2)] = 0.000000e+00f;
  compute_local[(6)] = 0.000000e+00f;
  compute_local[(10)] = 0.000000e+00f;
  compute_local[(14)] = 0.000000e+00f;
  compute_local[(3)] = 0.000000e+00f;
  compute_local[(7)] = 0.000000e+00f;
  compute_local[(11)] = 0.000000e+00f;
  compute_local[(15)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 4; ++rc_outer) {
    __syncthreads();
    pad_temp_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))] = placeholder[((((((rc_outer * 4096) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))];
    pad_temp_shared[(((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 1))] = placeholder[(((((((rc_outer * 4096) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 1))];
    pad_temp_shared[(((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 2))] = placeholder[(((((((rc_outer * 4096) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 2))];
    pad_temp_shared[(((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 3))] = placeholder[(((((((rc_outer * 4096) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 3))];
    placeholder_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))] = placeholder1[((((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 64)) + (rc_outer * 16)) + (((int)threadIdx.x) * 4)))];
    placeholder_shared[(((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 1))] = placeholder1[(((((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 64)) + (rc_outer * 16)) + (((int)threadIdx.x) * 4)) + 1))];
    placeholder_shared[(((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 2))] = placeholder1[(((((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 64)) + (rc_outer * 16)) + (((int)threadIdx.x) * 4)) + 2))];
    placeholder_shared[(((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 3))] = placeholder1[(((((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 64)) + (rc_outer * 16)) + (((int)threadIdx.x) * 4)) + 3))];
    __syncthreads();
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)))], placeholder_shared[((((int)threadIdx.z) * 16))], compute_local[(0)]);
    compute_local[(4)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)))], placeholder_shared[(((((int)threadIdx.z) * 16) + 256))], compute_local[(4)]);
    compute_local[(8)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)))], placeholder_shared[(((((int)threadIdx.z) * 16) + 512))], compute_local[(8)]);
    compute_local[(12)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)))], placeholder_shared[(((((int)threadIdx.z) * 16) + 768))], compute_local[(12)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 1))], placeholder_shared[((((int)threadIdx.z) * 16))], compute_local[(1)]);
    compute_local[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 1))], placeholder_shared[(((((int)threadIdx.z) * 16) + 256))], compute_local[(5)]);
    compute_local[(9)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 1))], placeholder_shared[(((((int)threadIdx.z) * 16) + 512))], compute_local[(9)]);
    compute_local[(13)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 1))], placeholder_shared[(((((int)threadIdx.z) * 16) + 768))], compute_local[(13)]);
    compute_local[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 2))], placeholder_shared[((((int)threadIdx.z) * 16))], compute_local[(2)]);
    compute_local[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 2))], placeholder_shared[(((((int)threadIdx.z) * 16) + 256))], compute_local[(6)]);
    compute_local[(10)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 2))], placeholder_shared[(((((int)threadIdx.z) * 16) + 512))], compute_local[(10)]);
    compute_local[(14)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 2))], placeholder_shared[(((((int)threadIdx.z) * 16) + 768))], compute_local[(14)]);
    compute_local[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 3))], placeholder_shared[((((int)threadIdx.z) * 16))], compute_local[(3)]);
    compute_local[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 3))], placeholder_shared[(((((int)threadIdx.z) * 16) + 256))], compute_local[(7)]);
    compute_local[(11)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 3))], placeholder_shared[(((((int)threadIdx.z) * 16) + 512))], compute_local[(11)]);
    compute_local[(15)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 3))], placeholder_shared[(((((int)threadIdx.z) * 16) + 768))], compute_local[(15)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 64))], placeholder_shared[(((((int)threadIdx.z) * 16) + 1))], compute_local[(0)]);
    compute_local[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 64))], placeholder_shared[(((((int)threadIdx.z) * 16) + 257))], compute_local[(4)]);
    compute_local[(8)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 64))], placeholder_shared[(((((int)threadIdx.z) * 16) + 513))], compute_local[(8)]);
    compute_local[(12)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 64))], placeholder_shared[(((((int)threadIdx.z) * 16) + 769))], compute_local[(12)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 65))], placeholder_shared[(((((int)threadIdx.z) * 16) + 1))], compute_local[(1)]);
    compute_local[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 65))], placeholder_shared[(((((int)threadIdx.z) * 16) + 257))], compute_local[(5)]);
    compute_local[(9)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 65))], placeholder_shared[(((((int)threadIdx.z) * 16) + 513))], compute_local[(9)]);
    compute_local[(13)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 65))], placeholder_shared[(((((int)threadIdx.z) * 16) + 769))], compute_local[(13)]);
    compute_local[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 66))], placeholder_shared[(((((int)threadIdx.z) * 16) + 1))], compute_local[(2)]);
    compute_local[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 66))], placeholder_shared[(((((int)threadIdx.z) * 16) + 257))], compute_local[(6)]);
    compute_local[(10)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 66))], placeholder_shared[(((((int)threadIdx.z) * 16) + 513))], compute_local[(10)]);
    compute_local[(14)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 66))], placeholder_shared[(((((int)threadIdx.z) * 16) + 769))], compute_local[(14)]);
    compute_local[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 67))], placeholder_shared[(((((int)threadIdx.z) * 16) + 1))], compute_local[(3)]);
    compute_local[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 67))], placeholder_shared[(((((int)threadIdx.z) * 16) + 257))], compute_local[(7)]);
    compute_local[(11)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 67))], placeholder_shared[(((((int)threadIdx.z) * 16) + 513))], compute_local[(11)]);
    compute_local[(15)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 67))], placeholder_shared[(((((int)threadIdx.z) * 16) + 769))], compute_local[(15)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 128))], placeholder_shared[(((((int)threadIdx.z) * 16) + 2))], compute_local[(0)]);
    compute_local[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 128))], placeholder_shared[(((((int)threadIdx.z) * 16) + 258))], compute_local[(4)]);
    compute_local[(8)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 128))], placeholder_shared[(((((int)threadIdx.z) * 16) + 514))], compute_local[(8)]);
    compute_local[(12)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 128))], placeholder_shared[(((((int)threadIdx.z) * 16) + 770))], compute_local[(12)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 129))], placeholder_shared[(((((int)threadIdx.z) * 16) + 2))], compute_local[(1)]);
    compute_local[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 129))], placeholder_shared[(((((int)threadIdx.z) * 16) + 258))], compute_local[(5)]);
    compute_local[(9)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 129))], placeholder_shared[(((((int)threadIdx.z) * 16) + 514))], compute_local[(9)]);
    compute_local[(13)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 129))], placeholder_shared[(((((int)threadIdx.z) * 16) + 770))], compute_local[(13)]);
    compute_local[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 130))], placeholder_shared[(((((int)threadIdx.z) * 16) + 2))], compute_local[(2)]);
    compute_local[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 130))], placeholder_shared[(((((int)threadIdx.z) * 16) + 258))], compute_local[(6)]);
    compute_local[(10)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 130))], placeholder_shared[(((((int)threadIdx.z) * 16) + 514))], compute_local[(10)]);
    compute_local[(14)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 130))], placeholder_shared[(((((int)threadIdx.z) * 16) + 770))], compute_local[(14)]);
    compute_local[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 131))], placeholder_shared[(((((int)threadIdx.z) * 16) + 2))], compute_local[(3)]);
    compute_local[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 131))], placeholder_shared[(((((int)threadIdx.z) * 16) + 258))], compute_local[(7)]);
    compute_local[(11)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 131))], placeholder_shared[(((((int)threadIdx.z) * 16) + 514))], compute_local[(11)]);
    compute_local[(15)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 131))], placeholder_shared[(((((int)threadIdx.z) * 16) + 770))], compute_local[(15)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 192))], placeholder_shared[(((((int)threadIdx.z) * 16) + 3))], compute_local[(0)]);
    compute_local[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 192))], placeholder_shared[(((((int)threadIdx.z) * 16) + 259))], compute_local[(4)]);
    compute_local[(8)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 192))], placeholder_shared[(((((int)threadIdx.z) * 16) + 515))], compute_local[(8)]);
    compute_local[(12)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 192))], placeholder_shared[(((((int)threadIdx.z) * 16) + 771))], compute_local[(12)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 193))], placeholder_shared[(((((int)threadIdx.z) * 16) + 3))], compute_local[(1)]);
    compute_local[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 193))], placeholder_shared[(((((int)threadIdx.z) * 16) + 259))], compute_local[(5)]);
    compute_local[(9)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 193))], placeholder_shared[(((((int)threadIdx.z) * 16) + 515))], compute_local[(9)]);
    compute_local[(13)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 193))], placeholder_shared[(((((int)threadIdx.z) * 16) + 771))], compute_local[(13)]);
    compute_local[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 194))], placeholder_shared[(((((int)threadIdx.z) * 16) + 3))], compute_local[(2)]);
    compute_local[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 194))], placeholder_shared[(((((int)threadIdx.z) * 16) + 259))], compute_local[(6)]);
    compute_local[(10)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 194))], placeholder_shared[(((((int)threadIdx.z) * 16) + 515))], compute_local[(10)]);
    compute_local[(14)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 194))], placeholder_shared[(((((int)threadIdx.z) * 16) + 771))], compute_local[(14)]);
    compute_local[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 195))], placeholder_shared[(((((int)threadIdx.z) * 16) + 3))], compute_local[(3)]);
    compute_local[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 195))], placeholder_shared[(((((int)threadIdx.z) * 16) + 259))], compute_local[(7)]);
    compute_local[(11)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 195))], placeholder_shared[(((((int)threadIdx.z) * 16) + 515))], compute_local[(11)]);
    compute_local[(15)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 195))], placeholder_shared[(((((int)threadIdx.z) * 16) + 771))], compute_local[(15)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 256))], placeholder_shared[(((((int)threadIdx.z) * 16) + 4))], compute_local[(0)]);
    compute_local[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 256))], placeholder_shared[(((((int)threadIdx.z) * 16) + 260))], compute_local[(4)]);
    compute_local[(8)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 256))], placeholder_shared[(((((int)threadIdx.z) * 16) + 516))], compute_local[(8)]);
    compute_local[(12)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 256))], placeholder_shared[(((((int)threadIdx.z) * 16) + 772))], compute_local[(12)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 257))], placeholder_shared[(((((int)threadIdx.z) * 16) + 4))], compute_local[(1)]);
    compute_local[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 257))], placeholder_shared[(((((int)threadIdx.z) * 16) + 260))], compute_local[(5)]);
    compute_local[(9)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 257))], placeholder_shared[(((((int)threadIdx.z) * 16) + 516))], compute_local[(9)]);
    compute_local[(13)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 257))], placeholder_shared[(((((int)threadIdx.z) * 16) + 772))], compute_local[(13)]);
    compute_local[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 258))], placeholder_shared[(((((int)threadIdx.z) * 16) + 4))], compute_local[(2)]);
    compute_local[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 258))], placeholder_shared[(((((int)threadIdx.z) * 16) + 260))], compute_local[(6)]);
    compute_local[(10)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 258))], placeholder_shared[(((((int)threadIdx.z) * 16) + 516))], compute_local[(10)]);
    compute_local[(14)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 258))], placeholder_shared[(((((int)threadIdx.z) * 16) + 772))], compute_local[(14)]);
    compute_local[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 259))], placeholder_shared[(((((int)threadIdx.z) * 16) + 4))], compute_local[(3)]);
    compute_local[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 259))], placeholder_shared[(((((int)threadIdx.z) * 16) + 260))], compute_local[(7)]);
    compute_local[(11)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 259))], placeholder_shared[(((((int)threadIdx.z) * 16) + 516))], compute_local[(11)]);
    compute_local[(15)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 259))], placeholder_shared[(((((int)threadIdx.z) * 16) + 772))], compute_local[(15)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 320))], placeholder_shared[(((((int)threadIdx.z) * 16) + 5))], compute_local[(0)]);
    compute_local[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 320))], placeholder_shared[(((((int)threadIdx.z) * 16) + 261))], compute_local[(4)]);
    compute_local[(8)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 320))], placeholder_shared[(((((int)threadIdx.z) * 16) + 517))], compute_local[(8)]);
    compute_local[(12)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 320))], placeholder_shared[(((((int)threadIdx.z) * 16) + 773))], compute_local[(12)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 321))], placeholder_shared[(((((int)threadIdx.z) * 16) + 5))], compute_local[(1)]);
    compute_local[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 321))], placeholder_shared[(((((int)threadIdx.z) * 16) + 261))], compute_local[(5)]);
    compute_local[(9)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 321))], placeholder_shared[(((((int)threadIdx.z) * 16) + 517))], compute_local[(9)]);
    compute_local[(13)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 321))], placeholder_shared[(((((int)threadIdx.z) * 16) + 773))], compute_local[(13)]);
    compute_local[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 322))], placeholder_shared[(((((int)threadIdx.z) * 16) + 5))], compute_local[(2)]);
    compute_local[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 322))], placeholder_shared[(((((int)threadIdx.z) * 16) + 261))], compute_local[(6)]);
    compute_local[(10)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 322))], placeholder_shared[(((((int)threadIdx.z) * 16) + 517))], compute_local[(10)]);
    compute_local[(14)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 322))], placeholder_shared[(((((int)threadIdx.z) * 16) + 773))], compute_local[(14)]);
    compute_local[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 323))], placeholder_shared[(((((int)threadIdx.z) * 16) + 5))], compute_local[(3)]);
    compute_local[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 323))], placeholder_shared[(((((int)threadIdx.z) * 16) + 261))], compute_local[(7)]);
    compute_local[(11)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 323))], placeholder_shared[(((((int)threadIdx.z) * 16) + 517))], compute_local[(11)]);
    compute_local[(15)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 323))], placeholder_shared[(((((int)threadIdx.z) * 16) + 773))], compute_local[(15)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 384))], placeholder_shared[(((((int)threadIdx.z) * 16) + 6))], compute_local[(0)]);
    compute_local[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 384))], placeholder_shared[(((((int)threadIdx.z) * 16) + 262))], compute_local[(4)]);
    compute_local[(8)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 384))], placeholder_shared[(((((int)threadIdx.z) * 16) + 518))], compute_local[(8)]);
    compute_local[(12)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 384))], placeholder_shared[(((((int)threadIdx.z) * 16) + 774))], compute_local[(12)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 385))], placeholder_shared[(((((int)threadIdx.z) * 16) + 6))], compute_local[(1)]);
    compute_local[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 385))], placeholder_shared[(((((int)threadIdx.z) * 16) + 262))], compute_local[(5)]);
    compute_local[(9)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 385))], placeholder_shared[(((((int)threadIdx.z) * 16) + 518))], compute_local[(9)]);
    compute_local[(13)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 385))], placeholder_shared[(((((int)threadIdx.z) * 16) + 774))], compute_local[(13)]);
    compute_local[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 386))], placeholder_shared[(((((int)threadIdx.z) * 16) + 6))], compute_local[(2)]);
    compute_local[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 386))], placeholder_shared[(((((int)threadIdx.z) * 16) + 262))], compute_local[(6)]);
    compute_local[(10)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 386))], placeholder_shared[(((((int)threadIdx.z) * 16) + 518))], compute_local[(10)]);
    compute_local[(14)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 386))], placeholder_shared[(((((int)threadIdx.z) * 16) + 774))], compute_local[(14)]);
    compute_local[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 387))], placeholder_shared[(((((int)threadIdx.z) * 16) + 6))], compute_local[(3)]);
    compute_local[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 387))], placeholder_shared[(((((int)threadIdx.z) * 16) + 262))], compute_local[(7)]);
    compute_local[(11)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 387))], placeholder_shared[(((((int)threadIdx.z) * 16) + 518))], compute_local[(11)]);
    compute_local[(15)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 387))], placeholder_shared[(((((int)threadIdx.z) * 16) + 774))], compute_local[(15)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 448))], placeholder_shared[(((((int)threadIdx.z) * 16) + 7))], compute_local[(0)]);
    compute_local[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 448))], placeholder_shared[(((((int)threadIdx.z) * 16) + 263))], compute_local[(4)]);
    compute_local[(8)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 448))], placeholder_shared[(((((int)threadIdx.z) * 16) + 519))], compute_local[(8)]);
    compute_local[(12)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 448))], placeholder_shared[(((((int)threadIdx.z) * 16) + 775))], compute_local[(12)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 449))], placeholder_shared[(((((int)threadIdx.z) * 16) + 7))], compute_local[(1)]);
    compute_local[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 449))], placeholder_shared[(((((int)threadIdx.z) * 16) + 263))], compute_local[(5)]);
    compute_local[(9)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 449))], placeholder_shared[(((((int)threadIdx.z) * 16) + 519))], compute_local[(9)]);
    compute_local[(13)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 449))], placeholder_shared[(((((int)threadIdx.z) * 16) + 775))], compute_local[(13)]);
    compute_local[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 450))], placeholder_shared[(((((int)threadIdx.z) * 16) + 7))], compute_local[(2)]);
    compute_local[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 450))], placeholder_shared[(((((int)threadIdx.z) * 16) + 263))], compute_local[(6)]);
    compute_local[(10)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 450))], placeholder_shared[(((((int)threadIdx.z) * 16) + 519))], compute_local[(10)]);
    compute_local[(14)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 450))], placeholder_shared[(((((int)threadIdx.z) * 16) + 775))], compute_local[(14)]);
    compute_local[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 451))], placeholder_shared[(((((int)threadIdx.z) * 16) + 7))], compute_local[(3)]);
    compute_local[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 451))], placeholder_shared[(((((int)threadIdx.z) * 16) + 263))], compute_local[(7)]);
    compute_local[(11)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 451))], placeholder_shared[(((((int)threadIdx.z) * 16) + 519))], compute_local[(11)]);
    compute_local[(15)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 451))], placeholder_shared[(((((int)threadIdx.z) * 16) + 775))], compute_local[(15)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 512))], placeholder_shared[(((((int)threadIdx.z) * 16) + 8))], compute_local[(0)]);
    compute_local[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 512))], placeholder_shared[(((((int)threadIdx.z) * 16) + 264))], compute_local[(4)]);
    compute_local[(8)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 512))], placeholder_shared[(((((int)threadIdx.z) * 16) + 520))], compute_local[(8)]);
    compute_local[(12)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 512))], placeholder_shared[(((((int)threadIdx.z) * 16) + 776))], compute_local[(12)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 513))], placeholder_shared[(((((int)threadIdx.z) * 16) + 8))], compute_local[(1)]);
    compute_local[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 513))], placeholder_shared[(((((int)threadIdx.z) * 16) + 264))], compute_local[(5)]);
    compute_local[(9)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 513))], placeholder_shared[(((((int)threadIdx.z) * 16) + 520))], compute_local[(9)]);
    compute_local[(13)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 513))], placeholder_shared[(((((int)threadIdx.z) * 16) + 776))], compute_local[(13)]);
    compute_local[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 514))], placeholder_shared[(((((int)threadIdx.z) * 16) + 8))], compute_local[(2)]);
    compute_local[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 514))], placeholder_shared[(((((int)threadIdx.z) * 16) + 264))], compute_local[(6)]);
    compute_local[(10)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 514))], placeholder_shared[(((((int)threadIdx.z) * 16) + 520))], compute_local[(10)]);
    compute_local[(14)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 514))], placeholder_shared[(((((int)threadIdx.z) * 16) + 776))], compute_local[(14)]);
    compute_local[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 515))], placeholder_shared[(((((int)threadIdx.z) * 16) + 8))], compute_local[(3)]);
    compute_local[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 515))], placeholder_shared[(((((int)threadIdx.z) * 16) + 264))], compute_local[(7)]);
    compute_local[(11)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 515))], placeholder_shared[(((((int)threadIdx.z) * 16) + 520))], compute_local[(11)]);
    compute_local[(15)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 515))], placeholder_shared[(((((int)threadIdx.z) * 16) + 776))], compute_local[(15)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 576))], placeholder_shared[(((((int)threadIdx.z) * 16) + 9))], compute_local[(0)]);
    compute_local[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 576))], placeholder_shared[(((((int)threadIdx.z) * 16) + 265))], compute_local[(4)]);
    compute_local[(8)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 576))], placeholder_shared[(((((int)threadIdx.z) * 16) + 521))], compute_local[(8)]);
    compute_local[(12)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 576))], placeholder_shared[(((((int)threadIdx.z) * 16) + 777))], compute_local[(12)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 577))], placeholder_shared[(((((int)threadIdx.z) * 16) + 9))], compute_local[(1)]);
    compute_local[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 577))], placeholder_shared[(((((int)threadIdx.z) * 16) + 265))], compute_local[(5)]);
    compute_local[(9)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 577))], placeholder_shared[(((((int)threadIdx.z) * 16) + 521))], compute_local[(9)]);
    compute_local[(13)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 577))], placeholder_shared[(((((int)threadIdx.z) * 16) + 777))], compute_local[(13)]);
    compute_local[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 578))], placeholder_shared[(((((int)threadIdx.z) * 16) + 9))], compute_local[(2)]);
    compute_local[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 578))], placeholder_shared[(((((int)threadIdx.z) * 16) + 265))], compute_local[(6)]);
    compute_local[(10)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 578))], placeholder_shared[(((((int)threadIdx.z) * 16) + 521))], compute_local[(10)]);
    compute_local[(14)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 578))], placeholder_shared[(((((int)threadIdx.z) * 16) + 777))], compute_local[(14)]);
    compute_local[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 579))], placeholder_shared[(((((int)threadIdx.z) * 16) + 9))], compute_local[(3)]);
    compute_local[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 579))], placeholder_shared[(((((int)threadIdx.z) * 16) + 265))], compute_local[(7)]);
    compute_local[(11)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 579))], placeholder_shared[(((((int)threadIdx.z) * 16) + 521))], compute_local[(11)]);
    compute_local[(15)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 579))], placeholder_shared[(((((int)threadIdx.z) * 16) + 777))], compute_local[(15)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 640))], placeholder_shared[(((((int)threadIdx.z) * 16) + 10))], compute_local[(0)]);
    compute_local[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 640))], placeholder_shared[(((((int)threadIdx.z) * 16) + 266))], compute_local[(4)]);
    compute_local[(8)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 640))], placeholder_shared[(((((int)threadIdx.z) * 16) + 522))], compute_local[(8)]);
    compute_local[(12)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 640))], placeholder_shared[(((((int)threadIdx.z) * 16) + 778))], compute_local[(12)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 641))], placeholder_shared[(((((int)threadIdx.z) * 16) + 10))], compute_local[(1)]);
    compute_local[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 641))], placeholder_shared[(((((int)threadIdx.z) * 16) + 266))], compute_local[(5)]);
    compute_local[(9)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 641))], placeholder_shared[(((((int)threadIdx.z) * 16) + 522))], compute_local[(9)]);
    compute_local[(13)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 641))], placeholder_shared[(((((int)threadIdx.z) * 16) + 778))], compute_local[(13)]);
    compute_local[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 642))], placeholder_shared[(((((int)threadIdx.z) * 16) + 10))], compute_local[(2)]);
    compute_local[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 642))], placeholder_shared[(((((int)threadIdx.z) * 16) + 266))], compute_local[(6)]);
    compute_local[(10)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 642))], placeholder_shared[(((((int)threadIdx.z) * 16) + 522))], compute_local[(10)]);
    compute_local[(14)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 642))], placeholder_shared[(((((int)threadIdx.z) * 16) + 778))], compute_local[(14)]);
    compute_local[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 643))], placeholder_shared[(((((int)threadIdx.z) * 16) + 10))], compute_local[(3)]);
    compute_local[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 643))], placeholder_shared[(((((int)threadIdx.z) * 16) + 266))], compute_local[(7)]);
    compute_local[(11)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 643))], placeholder_shared[(((((int)threadIdx.z) * 16) + 522))], compute_local[(11)]);
    compute_local[(15)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 643))], placeholder_shared[(((((int)threadIdx.z) * 16) + 778))], compute_local[(15)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 704))], placeholder_shared[(((((int)threadIdx.z) * 16) + 11))], compute_local[(0)]);
    compute_local[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 704))], placeholder_shared[(((((int)threadIdx.z) * 16) + 267))], compute_local[(4)]);
    compute_local[(8)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 704))], placeholder_shared[(((((int)threadIdx.z) * 16) + 523))], compute_local[(8)]);
    compute_local[(12)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 704))], placeholder_shared[(((((int)threadIdx.z) * 16) + 779))], compute_local[(12)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 705))], placeholder_shared[(((((int)threadIdx.z) * 16) + 11))], compute_local[(1)]);
    compute_local[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 705))], placeholder_shared[(((((int)threadIdx.z) * 16) + 267))], compute_local[(5)]);
    compute_local[(9)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 705))], placeholder_shared[(((((int)threadIdx.z) * 16) + 523))], compute_local[(9)]);
    compute_local[(13)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 705))], placeholder_shared[(((((int)threadIdx.z) * 16) + 779))], compute_local[(13)]);
    compute_local[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 706))], placeholder_shared[(((((int)threadIdx.z) * 16) + 11))], compute_local[(2)]);
    compute_local[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 706))], placeholder_shared[(((((int)threadIdx.z) * 16) + 267))], compute_local[(6)]);
    compute_local[(10)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 706))], placeholder_shared[(((((int)threadIdx.z) * 16) + 523))], compute_local[(10)]);
    compute_local[(14)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 706))], placeholder_shared[(((((int)threadIdx.z) * 16) + 779))], compute_local[(14)]);
    compute_local[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 707))], placeholder_shared[(((((int)threadIdx.z) * 16) + 11))], compute_local[(3)]);
    compute_local[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 707))], placeholder_shared[(((((int)threadIdx.z) * 16) + 267))], compute_local[(7)]);
    compute_local[(11)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 707))], placeholder_shared[(((((int)threadIdx.z) * 16) + 523))], compute_local[(11)]);
    compute_local[(15)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 707))], placeholder_shared[(((((int)threadIdx.z) * 16) + 779))], compute_local[(15)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 768))], placeholder_shared[(((((int)threadIdx.z) * 16) + 12))], compute_local[(0)]);
    compute_local[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 768))], placeholder_shared[(((((int)threadIdx.z) * 16) + 268))], compute_local[(4)]);
    compute_local[(8)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 768))], placeholder_shared[(((((int)threadIdx.z) * 16) + 524))], compute_local[(8)]);
    compute_local[(12)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 768))], placeholder_shared[(((((int)threadIdx.z) * 16) + 780))], compute_local[(12)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 769))], placeholder_shared[(((((int)threadIdx.z) * 16) + 12))], compute_local[(1)]);
    compute_local[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 769))], placeholder_shared[(((((int)threadIdx.z) * 16) + 268))], compute_local[(5)]);
    compute_local[(9)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 769))], placeholder_shared[(((((int)threadIdx.z) * 16) + 524))], compute_local[(9)]);
    compute_local[(13)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 769))], placeholder_shared[(((((int)threadIdx.z) * 16) + 780))], compute_local[(13)]);
    compute_local[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 770))], placeholder_shared[(((((int)threadIdx.z) * 16) + 12))], compute_local[(2)]);
    compute_local[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 770))], placeholder_shared[(((((int)threadIdx.z) * 16) + 268))], compute_local[(6)]);
    compute_local[(10)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 770))], placeholder_shared[(((((int)threadIdx.z) * 16) + 524))], compute_local[(10)]);
    compute_local[(14)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 770))], placeholder_shared[(((((int)threadIdx.z) * 16) + 780))], compute_local[(14)]);
    compute_local[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 771))], placeholder_shared[(((((int)threadIdx.z) * 16) + 12))], compute_local[(3)]);
    compute_local[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 771))], placeholder_shared[(((((int)threadIdx.z) * 16) + 268))], compute_local[(7)]);
    compute_local[(11)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 771))], placeholder_shared[(((((int)threadIdx.z) * 16) + 524))], compute_local[(11)]);
    compute_local[(15)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 771))], placeholder_shared[(((((int)threadIdx.z) * 16) + 780))], compute_local[(15)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 832))], placeholder_shared[(((((int)threadIdx.z) * 16) + 13))], compute_local[(0)]);
    compute_local[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 832))], placeholder_shared[(((((int)threadIdx.z) * 16) + 269))], compute_local[(4)]);
    compute_local[(8)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 832))], placeholder_shared[(((((int)threadIdx.z) * 16) + 525))], compute_local[(8)]);
    compute_local[(12)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 832))], placeholder_shared[(((((int)threadIdx.z) * 16) + 781))], compute_local[(12)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 833))], placeholder_shared[(((((int)threadIdx.z) * 16) + 13))], compute_local[(1)]);
    compute_local[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 833))], placeholder_shared[(((((int)threadIdx.z) * 16) + 269))], compute_local[(5)]);
    compute_local[(9)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 833))], placeholder_shared[(((((int)threadIdx.z) * 16) + 525))], compute_local[(9)]);
    compute_local[(13)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 833))], placeholder_shared[(((((int)threadIdx.z) * 16) + 781))], compute_local[(13)]);
    compute_local[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 834))], placeholder_shared[(((((int)threadIdx.z) * 16) + 13))], compute_local[(2)]);
    compute_local[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 834))], placeholder_shared[(((((int)threadIdx.z) * 16) + 269))], compute_local[(6)]);
    compute_local[(10)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 834))], placeholder_shared[(((((int)threadIdx.z) * 16) + 525))], compute_local[(10)]);
    compute_local[(14)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 834))], placeholder_shared[(((((int)threadIdx.z) * 16) + 781))], compute_local[(14)]);
    compute_local[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 835))], placeholder_shared[(((((int)threadIdx.z) * 16) + 13))], compute_local[(3)]);
    compute_local[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 835))], placeholder_shared[(((((int)threadIdx.z) * 16) + 269))], compute_local[(7)]);
    compute_local[(11)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 835))], placeholder_shared[(((((int)threadIdx.z) * 16) + 525))], compute_local[(11)]);
    compute_local[(15)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 835))], placeholder_shared[(((((int)threadIdx.z) * 16) + 781))], compute_local[(15)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 896))], placeholder_shared[(((((int)threadIdx.z) * 16) + 14))], compute_local[(0)]);
    compute_local[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 896))], placeholder_shared[(((((int)threadIdx.z) * 16) + 270))], compute_local[(4)]);
    compute_local[(8)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 896))], placeholder_shared[(((((int)threadIdx.z) * 16) + 526))], compute_local[(8)]);
    compute_local[(12)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 896))], placeholder_shared[(((((int)threadIdx.z) * 16) + 782))], compute_local[(12)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 897))], placeholder_shared[(((((int)threadIdx.z) * 16) + 14))], compute_local[(1)]);
    compute_local[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 897))], placeholder_shared[(((((int)threadIdx.z) * 16) + 270))], compute_local[(5)]);
    compute_local[(9)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 897))], placeholder_shared[(((((int)threadIdx.z) * 16) + 526))], compute_local[(9)]);
    compute_local[(13)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 897))], placeholder_shared[(((((int)threadIdx.z) * 16) + 782))], compute_local[(13)]);
    compute_local[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 898))], placeholder_shared[(((((int)threadIdx.z) * 16) + 14))], compute_local[(2)]);
    compute_local[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 898))], placeholder_shared[(((((int)threadIdx.z) * 16) + 270))], compute_local[(6)]);
    compute_local[(10)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 898))], placeholder_shared[(((((int)threadIdx.z) * 16) + 526))], compute_local[(10)]);
    compute_local[(14)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 898))], placeholder_shared[(((((int)threadIdx.z) * 16) + 782))], compute_local[(14)]);
    compute_local[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 899))], placeholder_shared[(((((int)threadIdx.z) * 16) + 14))], compute_local[(3)]);
    compute_local[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 899))], placeholder_shared[(((((int)threadIdx.z) * 16) + 270))], compute_local[(7)]);
    compute_local[(11)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 899))], placeholder_shared[(((((int)threadIdx.z) * 16) + 526))], compute_local[(11)]);
    compute_local[(15)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 899))], placeholder_shared[(((((int)threadIdx.z) * 16) + 782))], compute_local[(15)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 960))], placeholder_shared[(((((int)threadIdx.z) * 16) + 15))], compute_local[(0)]);
    compute_local[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 960))], placeholder_shared[(((((int)threadIdx.z) * 16) + 271))], compute_local[(4)]);
    compute_local[(8)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 960))], placeholder_shared[(((((int)threadIdx.z) * 16) + 527))], compute_local[(8)]);
    compute_local[(12)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 960))], placeholder_shared[(((((int)threadIdx.z) * 16) + 783))], compute_local[(12)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 961))], placeholder_shared[(((((int)threadIdx.z) * 16) + 15))], compute_local[(1)]);
    compute_local[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 961))], placeholder_shared[(((((int)threadIdx.z) * 16) + 271))], compute_local[(5)]);
    compute_local[(9)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 961))], placeholder_shared[(((((int)threadIdx.z) * 16) + 527))], compute_local[(9)]);
    compute_local[(13)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 961))], placeholder_shared[(((((int)threadIdx.z) * 16) + 783))], compute_local[(13)]);
    compute_local[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 962))], placeholder_shared[(((((int)threadIdx.z) * 16) + 15))], compute_local[(2)]);
    compute_local[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 962))], placeholder_shared[(((((int)threadIdx.z) * 16) + 271))], compute_local[(6)]);
    compute_local[(10)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 962))], placeholder_shared[(((((int)threadIdx.z) * 16) + 527))], compute_local[(10)]);
    compute_local[(14)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 962))], placeholder_shared[(((((int)threadIdx.z) * 16) + 783))], compute_local[(14)]);
    compute_local[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 963))], placeholder_shared[(((((int)threadIdx.z) * 16) + 15))], compute_local[(3)]);
    compute_local[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 963))], placeholder_shared[(((((int)threadIdx.z) * 16) + 271))], compute_local[(7)]);
    compute_local[(11)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 963))], placeholder_shared[(((((int)threadIdx.z) * 16) + 527))], compute_local[(11)]);
    compute_local[(15)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + 963))], placeholder_shared[(((((int)threadIdx.z) * 16) + 783))], compute_local[(15)]);
  }
  compute[((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))] = compute_local[(0)];
  compute[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 4096))] = compute_local[(4)];
  compute[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 8192))] = compute_local[(8)];
  compute[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 12288))] = compute_local[(12)];
  compute[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 1))] = compute_local[(1)];
  compute[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 4097))] = compute_local[(5)];
  compute[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 8193))] = compute_local[(9)];
  compute[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 12289))] = compute_local[(13)];
  compute[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 2))] = compute_local[(2)];
  compute[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 4098))] = compute_local[(6)];
  compute[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 8194))] = compute_local[(10)];
  compute[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 12290))] = compute_local[(14)];
  compute[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 3))] = compute_local[(3)];
  compute[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 4099))] = compute_local[(7)];
  compute[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 8195))] = compute_local[(11)];
  compute[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 12291))] = compute_local[(15)];
}

extern "C" __global__ void tvmgen_default_fused_nn_max_pool2d_add_nn_relu_kernel0(float* __restrict__ placeholder, float* __restrict__ T_relu, float* __restrict__ placeholder1) {
  float tensor[1];
  tensor[(0)] = -3.402823e+38f;
  for (int rv0 = 0; rv0 < 3; ++rv0) {
    for (int rv1 = 0; rv1 < 3; ++rv1) {
      tensor[(0)] = max(tensor[(0)], (((1 <= (((((int)threadIdx.x) >> 4) * 2) + rv0)) && (1 <= (((((int)threadIdx.x) & 15) * 2) + rv1))) ? placeholder[(((((((((int)blockIdx.x) * 1024) + ((((int)threadIdx.x) >> 4) * 64)) + (rv0 * 32)) + ((((int)threadIdx.x) & 15) * 2)) + rv1) - 33))] : -3.402823e+38f));
    }
  }
  T_relu[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)))] = max((tensor[(0)] + placeholder1[(((int)blockIdx.x))]), 0.000000e+00f);
}

extern "C" __global__ void tvmgen_default_fused_nn_conv2d_add_nn_relu_1_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  float compute[2];
  __shared__ float pad_temp_shared[64];
  __shared__ float placeholder_shared[256];
  compute[(0)] = 0.000000e+00f;
  compute[(1)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 128; ++rc_outer) {
    __syncthreads();
    pad_temp_shared[((((((int)threadIdx.z) * 8) + (((int)threadIdx.y) * 4)) + (((int)threadIdx.x) * 2)))] = placeholder[(((((rc_outer * 64) + (((int)threadIdx.z) * 8)) + (((int)threadIdx.y) * 4)) + (((int)threadIdx.x) * 2)))];
    pad_temp_shared[(((((((int)threadIdx.z) * 8) + (((int)threadIdx.y) * 4)) + (((int)threadIdx.x) * 2)) + 1))] = placeholder[((((((rc_outer * 64) + (((int)threadIdx.z) * 8)) + (((int)threadIdx.y) * 4)) + (((int)threadIdx.x) * 2)) + 1))];
    placeholder_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 8)))] = placeholder1[((((((((int)blockIdx.z) * 32768) + (((int)threadIdx.z) * 4096)) + (((int)threadIdx.y) * 2048)) + (rc_outer * 16)) + (((int)threadIdx.x) * 8)))];
    placeholder_shared[(((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 8)) + 1))] = placeholder1[(((((((((int)blockIdx.z) * 32768) + (((int)threadIdx.z) * 4096)) + (((int)threadIdx.y) * 2048)) + (rc_outer * 16)) + (((int)threadIdx.x) * 8)) + 1))];
    placeholder_shared[(((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 8)) + 2))] = placeholder1[(((((((((int)blockIdx.z) * 32768) + (((int)threadIdx.z) * 4096)) + (((int)threadIdx.y) * 2048)) + (rc_outer * 16)) + (((int)threadIdx.x) * 8)) + 2))];
    placeholder_shared[(((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 8)) + 3))] = placeholder1[(((((((((int)blockIdx.z) * 32768) + (((int)threadIdx.z) * 4096)) + (((int)threadIdx.y) * 2048)) + (rc_outer * 16)) + (((int)threadIdx.x) * 8)) + 3))];
    placeholder_shared[(((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 8)) + 4))] = placeholder1[(((((((((int)blockIdx.z) * 32768) + (((int)threadIdx.z) * 4096)) + (((int)threadIdx.y) * 2048)) + (rc_outer * 16)) + (((int)threadIdx.x) * 8)) + 4))];
    placeholder_shared[(((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 8)) + 5))] = placeholder1[(((((((((int)blockIdx.z) * 32768) + (((int)threadIdx.z) * 4096)) + (((int)threadIdx.y) * 2048)) + (rc_outer * 16)) + (((int)threadIdx.x) * 8)) + 5))];
    placeholder_shared[(((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 8)) + 6))] = placeholder1[(((((((((int)blockIdx.z) * 32768) + (((int)threadIdx.z) * 4096)) + (((int)threadIdx.y) * 2048)) + (rc_outer * 16)) + (((int)threadIdx.x) * 8)) + 6))];
    placeholder_shared[(((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 8)) + 7))] = placeholder1[(((((((((int)blockIdx.z) * 32768) + (((int)threadIdx.z) * 4096)) + (((int)threadIdx.y) * 2048)) + (rc_outer * 16)) + (((int)threadIdx.x) * 8)) + 7))];
    __syncthreads();
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 2) + ((int)threadIdx.x)))], placeholder_shared[((((int)threadIdx.z) * 16))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 2) + ((int)threadIdx.x)))], placeholder_shared[(((((int)threadIdx.z) * 16) + 128))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) + 4))], placeholder_shared[(((((int)threadIdx.z) * 16) + 1))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) + 4))], placeholder_shared[(((((int)threadIdx.z) * 16) + 129))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) + 8))], placeholder_shared[(((((int)threadIdx.z) * 16) + 2))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) + 8))], placeholder_shared[(((((int)threadIdx.z) * 16) + 130))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) + 12))], placeholder_shared[(((((int)threadIdx.z) * 16) + 3))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) + 12))], placeholder_shared[(((((int)threadIdx.z) * 16) + 131))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) + 16))], placeholder_shared[(((((int)threadIdx.z) * 16) + 4))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) + 16))], placeholder_shared[(((((int)threadIdx.z) * 16) + 132))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) + 20))], placeholder_shared[(((((int)threadIdx.z) * 16) + 5))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) + 20))], placeholder_shared[(((((int)threadIdx.z) * 16) + 133))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) + 24))], placeholder_shared[(((((int)threadIdx.z) * 16) + 6))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) + 24))], placeholder_shared[(((((int)threadIdx.z) * 16) + 134))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) + 28))], placeholder_shared[(((((int)threadIdx.z) * 16) + 7))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) + 28))], placeholder_shared[(((((int)threadIdx.z) * 16) + 135))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) + 32))], placeholder_shared[(((((int)threadIdx.z) * 16) + 8))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) + 32))], placeholder_shared[(((((int)threadIdx.z) * 16) + 136))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) + 36))], placeholder_shared[(((((int)threadIdx.z) * 16) + 9))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) + 36))], placeholder_shared[(((((int)threadIdx.z) * 16) + 137))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) + 40))], placeholder_shared[(((((int)threadIdx.z) * 16) + 10))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) + 40))], placeholder_shared[(((((int)threadIdx.z) * 16) + 138))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) + 44))], placeholder_shared[(((((int)threadIdx.z) * 16) + 11))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) + 44))], placeholder_shared[(((((int)threadIdx.z) * 16) + 139))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) + 48))], placeholder_shared[(((((int)threadIdx.z) * 16) + 12))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) + 48))], placeholder_shared[(((((int)threadIdx.z) * 16) + 140))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) + 52))], placeholder_shared[(((((int)threadIdx.z) * 16) + 13))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) + 52))], placeholder_shared[(((((int)threadIdx.z) * 16) + 141))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) + 56))], placeholder_shared[(((((int)threadIdx.z) * 16) + 14))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) + 56))], placeholder_shared[(((((int)threadIdx.z) * 16) + 142))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) + 60))], placeholder_shared[(((((int)threadIdx.z) * 16) + 15))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) + 60))], placeholder_shared[(((((int)threadIdx.z) * 16) + 143))], compute[(1)]);
  }
  T_relu[(((((((int)blockIdx.z) * 64) + (((int)threadIdx.z) * 4)) + (((int)threadIdx.y) * 2)) + ((int)threadIdx.x)))] = max((compute[(0)] + placeholder2[(((((int)blockIdx.z) * 16) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 64) + (((int)threadIdx.z) * 4)) + (((int)threadIdx.y) * 2)) + ((int)threadIdx.x)) + 32))] = max((compute[(1)] + placeholder2[((((((int)blockIdx.z) * 16) + ((int)threadIdx.z)) + 8))]), 0.000000e+00f);
}

extern "C" __global__ void tvmgen_default_fused_add_nn_relu_1_kernel0(float* __restrict__ T_relu, float* __restrict__ placeholder, float* __restrict__ placeholder1) {
  T_relu[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)))] = max((placeholder[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)))] + placeholder1[(((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 4)))]), 0.000000e+00f);
}

extern "C" __global__ void tvmgen_default_fused_nn_dense_add_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_add, float* __restrict__ placeholder2) {
  float T_matmul_NT_rf[1];
  float red_buf0[1];
  __shared__ float T_matmul_NT[1];
  T_matmul_NT_rf[(0)] = 0.000000e+00f;
  for (int k_outer = 0; k_outer < 32; ++k_outer) {
    T_matmul_NT_rf[(0)] = __ocml_fma_f32(placeholder[(((k_outer * 64) + ((int)threadIdx.x)))], placeholder1[((((((int)blockIdx.x) * 2048) + (k_outer * 64)) + ((int)threadIdx.x)))], T_matmul_NT_rf[(0)]);
  }
  uint mask[1];
  float t0[1];
  red_buf0[(0)] = T_matmul_NT_rf[(0)];
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
    T_matmul_NT[(0)] = red_buf0[(0)];
  }
  if (((int)threadIdx.x) == 0) {
    T_add[(((int)blockIdx.x))] = (T_matmul_NT[(0)] + placeholder2[(((int)blockIdx.x))]);
  }
}

extern "C" __global__ void tvmgen_default_fused_nn_conv2d_add_nn_relu_12_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  float compute[8];
  __shared__ float pad_temp_shared[481];
  __shared__ float placeholder_shared[1568];
  compute[(0)] = 0.000000e+00f;
  compute[(4)] = 0.000000e+00f;
  compute[(1)] = 0.000000e+00f;
  compute[(5)] = 0.000000e+00f;
  compute[(2)] = 0.000000e+00f;
  compute[(6)] = 0.000000e+00f;
  compute[(3)] = 0.000000e+00f;
  compute[(7)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 3; ++rc_outer) {
    __syncthreads();
    if (((((int)threadIdx.z) * 31) + (((int)threadIdx.y) * 2)) < 481) {
      pad_temp_shared[(((((int)threadIdx.z) * 31) + (((int)threadIdx.y) * 2)))] = (((((3 <= ((((int)blockIdx.y) * 32) + (((((int)threadIdx.z) * 31) + (((int)threadIdx.y) * 2)) / 13))) && (((((int)blockIdx.y) * 32) + (((((int)threadIdx.z) * 31) + (((int)threadIdx.y) * 2)) / 13)) < 67)) && (3 <= ((((int)blockIdx.x) * 8) + (((((int)threadIdx.z) * 31) + (((int)threadIdx.y) * 2)) % 13)))) && (((((int)blockIdx.x) * 8) + (((((int)threadIdx.z) * 31) + (((int)threadIdx.y) * 2)) % 13)) < 67)) ? placeholder[(((((((rc_outer * 4096) + (((int)blockIdx.y) * 2048)) + ((((((int)threadIdx.z) * 31) + (((int)threadIdx.y) * 2)) / 13) * 64)) + (((int)blockIdx.x) * 8)) + (((((int)threadIdx.z) * 31) + (((int)threadIdx.y) * 2)) % 13)) - 195))] : 0.000000e+00f);
    }
    if (((((int)threadIdx.z) * 31) + (((int)threadIdx.y) * 2)) < 480) {
      if (((int)threadIdx.y) < 15) {
        pad_temp_shared[((((((int)threadIdx.z) * 31) + (((int)threadIdx.y) * 2)) + 1))] = (((((3 <= ((((int)blockIdx.y) * 32) + ((((((int)threadIdx.z) * 31) + (((int)threadIdx.y) * 2)) + 1) / 13))) && (((((int)blockIdx.y) * 32) + ((((((int)threadIdx.z) * 31) + (((int)threadIdx.y) * 2)) + 1) / 13)) < 67)) && (3 <= ((((int)blockIdx.x) * 8) + ((((((int)threadIdx.z) * 31) + (((int)threadIdx.y) * 2)) + 1) % 13)))) && (((((int)blockIdx.x) * 8) + ((((((int)threadIdx.z) * 31) + (((int)threadIdx.y) * 2)) + 1) % 13)) < 67)) ? placeholder[(((((((rc_outer * 4096) + (((int)blockIdx.y) * 2048)) + (((((((int)threadIdx.z) * 31) + (((int)threadIdx.y) * 2)) + 1) / 13) * 64)) + (((int)blockIdx.x) * 8)) + ((((((int)threadIdx.z) * 31) + (((int)threadIdx.y) * 2)) + 1) % 13)) - 195))] : 0.000000e+00f);
      }
    }
    if (((((int)threadIdx.z) * 2) + (((int)threadIdx.y) / 7)) < 32) {
      if (((((int)threadIdx.z) * 14) + ((int)threadIdx.y)) < 224) {
        if (((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 7)) < 1568) {
          if (((int)threadIdx.y) < 14) {
            placeholder_shared[(((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 7)))] = placeholder1[((((((((int)blockIdx.z) * 4704) + (((int)threadIdx.z) * 294)) + ((((int)threadIdx.y) / 7) * 147)) + (rc_outer * 49)) + ((((int)threadIdx.y) % 7) * 7)))];
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 2) + (((int)threadIdx.y) / 7)) < 32) {
      if (((((int)threadIdx.z) * 14) + ((int)threadIdx.y)) < 224) {
        if (((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 7)) < 1567) {
          if (((int)threadIdx.y) < 14) {
            placeholder_shared[((((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 7)) + 1))] = placeholder1[(((((((((int)blockIdx.z) * 4704) + (((int)threadIdx.z) * 294)) + ((((int)threadIdx.y) / 7) * 147)) + (rc_outer * 49)) + ((((int)threadIdx.y) % 7) * 7)) + 1))];
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 2) + (((int)threadIdx.y) / 7)) < 32) {
      if (((((int)threadIdx.z) * 14) + ((int)threadIdx.y)) < 224) {
        if (((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 7)) < 1566) {
          if (((int)threadIdx.y) < 14) {
            placeholder_shared[((((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 7)) + 2))] = placeholder1[(((((((((int)blockIdx.z) * 4704) + (((int)threadIdx.z) * 294)) + ((((int)threadIdx.y) / 7) * 147)) + (rc_outer * 49)) + ((((int)threadIdx.y) % 7) * 7)) + 2))];
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 2) + (((int)threadIdx.y) / 7)) < 32) {
      if (((((int)threadIdx.z) * 14) + ((int)threadIdx.y)) < 224) {
        if (((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 7)) < 1565) {
          if (((int)threadIdx.y) < 14) {
            placeholder_shared[((((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 7)) + 3))] = placeholder1[(((((((((int)blockIdx.z) * 4704) + (((int)threadIdx.z) * 294)) + ((((int)threadIdx.y) / 7) * 147)) + (rc_outer * 49)) + ((((int)threadIdx.y) % 7) * 7)) + 3))];
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 2) + (((int)threadIdx.y) / 7)) < 32) {
      if (((((int)threadIdx.z) * 14) + ((int)threadIdx.y)) < 224) {
        if (((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 7)) < 1564) {
          if (((int)threadIdx.y) < 14) {
            placeholder_shared[((((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 7)) + 4))] = placeholder1[(((((((((int)blockIdx.z) * 4704) + (((int)threadIdx.z) * 294)) + ((((int)threadIdx.y) / 7) * 147)) + (rc_outer * 49)) + ((((int)threadIdx.y) % 7) * 7)) + 4))];
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 2) + (((int)threadIdx.y) / 7)) < 32) {
      if (((((int)threadIdx.z) * 14) + ((int)threadIdx.y)) < 224) {
        if (((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 7)) < 1563) {
          if (((int)threadIdx.y) < 14) {
            placeholder_shared[((((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 7)) + 5))] = placeholder1[(((((((((int)blockIdx.z) * 4704) + (((int)threadIdx.z) * 294)) + ((((int)threadIdx.y) / 7) * 147)) + (rc_outer * 49)) + ((((int)threadIdx.y) % 7) * 7)) + 5))];
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 2) + (((int)threadIdx.y) / 7)) < 32) {
      if (((((int)threadIdx.z) * 14) + ((int)threadIdx.y)) < 224) {
        if (((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 7)) < 1562) {
          if (((int)threadIdx.y) < 14) {
            placeholder_shared[((((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 7)) + 6))] = placeholder1[(((((((((int)blockIdx.z) * 4704) + (((int)threadIdx.z) * 294)) + ((((int)threadIdx.y) / 7) * 147)) + (rc_outer * 49)) + ((((int)threadIdx.y) % 7) * 7)) + 6))];
          }
        }
      }
    }
    __syncthreads();
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.y) * 26))], placeholder_shared[((((int)threadIdx.z) * 98))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 4))], placeholder_shared[((((int)threadIdx.z) * 98))], compute[(4)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 2))], placeholder_shared[((((int)threadIdx.z) * 98))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 6))], placeholder_shared[((((int)threadIdx.z) * 98))], compute[(5)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.y) * 26))], placeholder_shared[(((((int)threadIdx.z) * 98) + 49))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 4))], placeholder_shared[(((((int)threadIdx.z) * 98) + 49))], compute[(6)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 2))], placeholder_shared[(((((int)threadIdx.z) * 98) + 49))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 6))], placeholder_shared[(((((int)threadIdx.z) * 98) + 49))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 1))], placeholder_shared[(((((int)threadIdx.z) * 98) + 1))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 5))], placeholder_shared[(((((int)threadIdx.z) * 98) + 1))], compute[(4)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 3))], placeholder_shared[(((((int)threadIdx.z) * 98) + 1))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 7))], placeholder_shared[(((((int)threadIdx.z) * 98) + 1))], compute[(5)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 1))], placeholder_shared[(((((int)threadIdx.z) * 98) + 50))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 5))], placeholder_shared[(((((int)threadIdx.z) * 98) + 50))], compute[(6)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 3))], placeholder_shared[(((((int)threadIdx.z) * 98) + 50))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 7))], placeholder_shared[(((((int)threadIdx.z) * 98) + 50))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 2))], placeholder_shared[(((((int)threadIdx.z) * 98) + 2))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 6))], placeholder_shared[(((((int)threadIdx.z) * 98) + 2))], compute[(4)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 4))], placeholder_shared[(((((int)threadIdx.z) * 98) + 2))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 8))], placeholder_shared[(((((int)threadIdx.z) * 98) + 2))], compute[(5)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 2))], placeholder_shared[(((((int)threadIdx.z) * 98) + 51))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 6))], placeholder_shared[(((((int)threadIdx.z) * 98) + 51))], compute[(6)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 4))], placeholder_shared[(((((int)threadIdx.z) * 98) + 51))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 8))], placeholder_shared[(((((int)threadIdx.z) * 98) + 51))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 3))], placeholder_shared[(((((int)threadIdx.z) * 98) + 3))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 7))], placeholder_shared[(((((int)threadIdx.z) * 98) + 3))], compute[(4)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 5))], placeholder_shared[(((((int)threadIdx.z) * 98) + 3))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 9))], placeholder_shared[(((((int)threadIdx.z) * 98) + 3))], compute[(5)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 3))], placeholder_shared[(((((int)threadIdx.z) * 98) + 52))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 7))], placeholder_shared[(((((int)threadIdx.z) * 98) + 52))], compute[(6)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 5))], placeholder_shared[(((((int)threadIdx.z) * 98) + 52))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 9))], placeholder_shared[(((((int)threadIdx.z) * 98) + 52))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 4))], placeholder_shared[(((((int)threadIdx.z) * 98) + 4))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 8))], placeholder_shared[(((((int)threadIdx.z) * 98) + 4))], compute[(4)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 6))], placeholder_shared[(((((int)threadIdx.z) * 98) + 4))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 10))], placeholder_shared[(((((int)threadIdx.z) * 98) + 4))], compute[(5)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 4))], placeholder_shared[(((((int)threadIdx.z) * 98) + 53))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 8))], placeholder_shared[(((((int)threadIdx.z) * 98) + 53))], compute[(6)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 6))], placeholder_shared[(((((int)threadIdx.z) * 98) + 53))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 10))], placeholder_shared[(((((int)threadIdx.z) * 98) + 53))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 5))], placeholder_shared[(((((int)threadIdx.z) * 98) + 5))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 9))], placeholder_shared[(((((int)threadIdx.z) * 98) + 5))], compute[(4)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 7))], placeholder_shared[(((((int)threadIdx.z) * 98) + 5))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 11))], placeholder_shared[(((((int)threadIdx.z) * 98) + 5))], compute[(5)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 5))], placeholder_shared[(((((int)threadIdx.z) * 98) + 54))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 9))], placeholder_shared[(((((int)threadIdx.z) * 98) + 54))], compute[(6)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 7))], placeholder_shared[(((((int)threadIdx.z) * 98) + 54))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 11))], placeholder_shared[(((((int)threadIdx.z) * 98) + 54))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 6))], placeholder_shared[(((((int)threadIdx.z) * 98) + 6))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 10))], placeholder_shared[(((((int)threadIdx.z) * 98) + 6))], compute[(4)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 8))], placeholder_shared[(((((int)threadIdx.z) * 98) + 6))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 12))], placeholder_shared[(((((int)threadIdx.z) * 98) + 6))], compute[(5)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 6))], placeholder_shared[(((((int)threadIdx.z) * 98) + 55))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 10))], placeholder_shared[(((((int)threadIdx.z) * 98) + 55))], compute[(6)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 8))], placeholder_shared[(((((int)threadIdx.z) * 98) + 55))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 12))], placeholder_shared[(((((int)threadIdx.z) * 98) + 55))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 13))], placeholder_shared[(((((int)threadIdx.z) * 98) + 7))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 17))], placeholder_shared[(((((int)threadIdx.z) * 98) + 7))], compute[(4)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 15))], placeholder_shared[(((((int)threadIdx.z) * 98) + 7))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 19))], placeholder_shared[(((((int)threadIdx.z) * 98) + 7))], compute[(5)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 13))], placeholder_shared[(((((int)threadIdx.z) * 98) + 56))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 17))], placeholder_shared[(((((int)threadIdx.z) * 98) + 56))], compute[(6)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 15))], placeholder_shared[(((((int)threadIdx.z) * 98) + 56))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 19))], placeholder_shared[(((((int)threadIdx.z) * 98) + 56))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 14))], placeholder_shared[(((((int)threadIdx.z) * 98) + 8))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 18))], placeholder_shared[(((((int)threadIdx.z) * 98) + 8))], compute[(4)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 16))], placeholder_shared[(((((int)threadIdx.z) * 98) + 8))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 20))], placeholder_shared[(((((int)threadIdx.z) * 98) + 8))], compute[(5)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 14))], placeholder_shared[(((((int)threadIdx.z) * 98) + 57))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 18))], placeholder_shared[(((((int)threadIdx.z) * 98) + 57))], compute[(6)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 16))], placeholder_shared[(((((int)threadIdx.z) * 98) + 57))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 20))], placeholder_shared[(((((int)threadIdx.z) * 98) + 57))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 15))], placeholder_shared[(((((int)threadIdx.z) * 98) + 9))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 19))], placeholder_shared[(((((int)threadIdx.z) * 98) + 9))], compute[(4)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 17))], placeholder_shared[(((((int)threadIdx.z) * 98) + 9))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 21))], placeholder_shared[(((((int)threadIdx.z) * 98) + 9))], compute[(5)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 15))], placeholder_shared[(((((int)threadIdx.z) * 98) + 58))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 19))], placeholder_shared[(((((int)threadIdx.z) * 98) + 58))], compute[(6)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 17))], placeholder_shared[(((((int)threadIdx.z) * 98) + 58))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 21))], placeholder_shared[(((((int)threadIdx.z) * 98) + 58))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 16))], placeholder_shared[(((((int)threadIdx.z) * 98) + 10))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 20))], placeholder_shared[(((((int)threadIdx.z) * 98) + 10))], compute[(4)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 18))], placeholder_shared[(((((int)threadIdx.z) * 98) + 10))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 22))], placeholder_shared[(((((int)threadIdx.z) * 98) + 10))], compute[(5)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 16))], placeholder_shared[(((((int)threadIdx.z) * 98) + 59))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 20))], placeholder_shared[(((((int)threadIdx.z) * 98) + 59))], compute[(6)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 18))], placeholder_shared[(((((int)threadIdx.z) * 98) + 59))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 22))], placeholder_shared[(((((int)threadIdx.z) * 98) + 59))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 17))], placeholder_shared[(((((int)threadIdx.z) * 98) + 11))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 21))], placeholder_shared[(((((int)threadIdx.z) * 98) + 11))], compute[(4)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 19))], placeholder_shared[(((((int)threadIdx.z) * 98) + 11))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 23))], placeholder_shared[(((((int)threadIdx.z) * 98) + 11))], compute[(5)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 17))], placeholder_shared[(((((int)threadIdx.z) * 98) + 60))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 21))], placeholder_shared[(((((int)threadIdx.z) * 98) + 60))], compute[(6)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 19))], placeholder_shared[(((((int)threadIdx.z) * 98) + 60))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 23))], placeholder_shared[(((((int)threadIdx.z) * 98) + 60))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 18))], placeholder_shared[(((((int)threadIdx.z) * 98) + 12))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 22))], placeholder_shared[(((((int)threadIdx.z) * 98) + 12))], compute[(4)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 20))], placeholder_shared[(((((int)threadIdx.z) * 98) + 12))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 24))], placeholder_shared[(((((int)threadIdx.z) * 98) + 12))], compute[(5)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 18))], placeholder_shared[(((((int)threadIdx.z) * 98) + 61))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 22))], placeholder_shared[(((((int)threadIdx.z) * 98) + 61))], compute[(6)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 20))], placeholder_shared[(((((int)threadIdx.z) * 98) + 61))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 24))], placeholder_shared[(((((int)threadIdx.z) * 98) + 61))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 19))], placeholder_shared[(((((int)threadIdx.z) * 98) + 13))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 23))], placeholder_shared[(((((int)threadIdx.z) * 98) + 13))], compute[(4)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 21))], placeholder_shared[(((((int)threadIdx.z) * 98) + 13))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 25))], placeholder_shared[(((((int)threadIdx.z) * 98) + 13))], compute[(5)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 19))], placeholder_shared[(((((int)threadIdx.z) * 98) + 62))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 23))], placeholder_shared[(((((int)threadIdx.z) * 98) + 62))], compute[(6)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 21))], placeholder_shared[(((((int)threadIdx.z) * 98) + 62))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 25))], placeholder_shared[(((((int)threadIdx.z) * 98) + 62))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 26))], placeholder_shared[(((((int)threadIdx.z) * 98) + 14))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 30))], placeholder_shared[(((((int)threadIdx.z) * 98) + 14))], compute[(4)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 28))], placeholder_shared[(((((int)threadIdx.z) * 98) + 14))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 32))], placeholder_shared[(((((int)threadIdx.z) * 98) + 14))], compute[(5)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 26))], placeholder_shared[(((((int)threadIdx.z) * 98) + 63))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 30))], placeholder_shared[(((((int)threadIdx.z) * 98) + 63))], compute[(6)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 28))], placeholder_shared[(((((int)threadIdx.z) * 98) + 63))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 32))], placeholder_shared[(((((int)threadIdx.z) * 98) + 63))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 27))], placeholder_shared[(((((int)threadIdx.z) * 98) + 15))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 31))], placeholder_shared[(((((int)threadIdx.z) * 98) + 15))], compute[(4)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 29))], placeholder_shared[(((((int)threadIdx.z) * 98) + 15))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 33))], placeholder_shared[(((((int)threadIdx.z) * 98) + 15))], compute[(5)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 27))], placeholder_shared[(((((int)threadIdx.z) * 98) + 64))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 31))], placeholder_shared[(((((int)threadIdx.z) * 98) + 64))], compute[(6)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 29))], placeholder_shared[(((((int)threadIdx.z) * 98) + 64))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 33))], placeholder_shared[(((((int)threadIdx.z) * 98) + 64))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 28))], placeholder_shared[(((((int)threadIdx.z) * 98) + 16))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 32))], placeholder_shared[(((((int)threadIdx.z) * 98) + 16))], compute[(4)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 30))], placeholder_shared[(((((int)threadIdx.z) * 98) + 16))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 34))], placeholder_shared[(((((int)threadIdx.z) * 98) + 16))], compute[(5)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 28))], placeholder_shared[(((((int)threadIdx.z) * 98) + 65))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 32))], placeholder_shared[(((((int)threadIdx.z) * 98) + 65))], compute[(6)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 30))], placeholder_shared[(((((int)threadIdx.z) * 98) + 65))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 34))], placeholder_shared[(((((int)threadIdx.z) * 98) + 65))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 29))], placeholder_shared[(((((int)threadIdx.z) * 98) + 17))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 33))], placeholder_shared[(((((int)threadIdx.z) * 98) + 17))], compute[(4)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 31))], placeholder_shared[(((((int)threadIdx.z) * 98) + 17))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 35))], placeholder_shared[(((((int)threadIdx.z) * 98) + 17))], compute[(5)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 29))], placeholder_shared[(((((int)threadIdx.z) * 98) + 66))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 33))], placeholder_shared[(((((int)threadIdx.z) * 98) + 66))], compute[(6)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 31))], placeholder_shared[(((((int)threadIdx.z) * 98) + 66))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 35))], placeholder_shared[(((((int)threadIdx.z) * 98) + 66))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 30))], placeholder_shared[(((((int)threadIdx.z) * 98) + 18))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 34))], placeholder_shared[(((((int)threadIdx.z) * 98) + 18))], compute[(4)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 32))], placeholder_shared[(((((int)threadIdx.z) * 98) + 18))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 36))], placeholder_shared[(((((int)threadIdx.z) * 98) + 18))], compute[(5)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 30))], placeholder_shared[(((((int)threadIdx.z) * 98) + 67))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 34))], placeholder_shared[(((((int)threadIdx.z) * 98) + 67))], compute[(6)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 32))], placeholder_shared[(((((int)threadIdx.z) * 98) + 67))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 36))], placeholder_shared[(((((int)threadIdx.z) * 98) + 67))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 31))], placeholder_shared[(((((int)threadIdx.z) * 98) + 19))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 35))], placeholder_shared[(((((int)threadIdx.z) * 98) + 19))], compute[(4)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 33))], placeholder_shared[(((((int)threadIdx.z) * 98) + 19))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 37))], placeholder_shared[(((((int)threadIdx.z) * 98) + 19))], compute[(5)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 31))], placeholder_shared[(((((int)threadIdx.z) * 98) + 68))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 35))], placeholder_shared[(((((int)threadIdx.z) * 98) + 68))], compute[(6)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 33))], placeholder_shared[(((((int)threadIdx.z) * 98) + 68))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 37))], placeholder_shared[(((((int)threadIdx.z) * 98) + 68))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 32))], placeholder_shared[(((((int)threadIdx.z) * 98) + 20))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 36))], placeholder_shared[(((((int)threadIdx.z) * 98) + 20))], compute[(4)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 34))], placeholder_shared[(((((int)threadIdx.z) * 98) + 20))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 38))], placeholder_shared[(((((int)threadIdx.z) * 98) + 20))], compute[(5)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 32))], placeholder_shared[(((((int)threadIdx.z) * 98) + 69))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 36))], placeholder_shared[(((((int)threadIdx.z) * 98) + 69))], compute[(6)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 34))], placeholder_shared[(((((int)threadIdx.z) * 98) + 69))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 38))], placeholder_shared[(((((int)threadIdx.z) * 98) + 69))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 39))], placeholder_shared[(((((int)threadIdx.z) * 98) + 21))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 43))], placeholder_shared[(((((int)threadIdx.z) * 98) + 21))], compute[(4)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 41))], placeholder_shared[(((((int)threadIdx.z) * 98) + 21))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 45))], placeholder_shared[(((((int)threadIdx.z) * 98) + 21))], compute[(5)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 39))], placeholder_shared[(((((int)threadIdx.z) * 98) + 70))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 43))], placeholder_shared[(((((int)threadIdx.z) * 98) + 70))], compute[(6)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 41))], placeholder_shared[(((((int)threadIdx.z) * 98) + 70))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 45))], placeholder_shared[(((((int)threadIdx.z) * 98) + 70))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 40))], placeholder_shared[(((((int)threadIdx.z) * 98) + 22))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 44))], placeholder_shared[(((((int)threadIdx.z) * 98) + 22))], compute[(4)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 42))], placeholder_shared[(((((int)threadIdx.z) * 98) + 22))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 46))], placeholder_shared[(((((int)threadIdx.z) * 98) + 22))], compute[(5)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 40))], placeholder_shared[(((((int)threadIdx.z) * 98) + 71))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 44))], placeholder_shared[(((((int)threadIdx.z) * 98) + 71))], compute[(6)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 42))], placeholder_shared[(((((int)threadIdx.z) * 98) + 71))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 46))], placeholder_shared[(((((int)threadIdx.z) * 98) + 71))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 41))], placeholder_shared[(((((int)threadIdx.z) * 98) + 23))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 45))], placeholder_shared[(((((int)threadIdx.z) * 98) + 23))], compute[(4)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 43))], placeholder_shared[(((((int)threadIdx.z) * 98) + 23))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 47))], placeholder_shared[(((((int)threadIdx.z) * 98) + 23))], compute[(5)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 41))], placeholder_shared[(((((int)threadIdx.z) * 98) + 72))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 45))], placeholder_shared[(((((int)threadIdx.z) * 98) + 72))], compute[(6)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 43))], placeholder_shared[(((((int)threadIdx.z) * 98) + 72))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 47))], placeholder_shared[(((((int)threadIdx.z) * 98) + 72))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 42))], placeholder_shared[(((((int)threadIdx.z) * 98) + 24))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 46))], placeholder_shared[(((((int)threadIdx.z) * 98) + 24))], compute[(4)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 44))], placeholder_shared[(((((int)threadIdx.z) * 98) + 24))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 48))], placeholder_shared[(((((int)threadIdx.z) * 98) + 24))], compute[(5)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 42))], placeholder_shared[(((((int)threadIdx.z) * 98) + 73))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 46))], placeholder_shared[(((((int)threadIdx.z) * 98) + 73))], compute[(6)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 44))], placeholder_shared[(((((int)threadIdx.z) * 98) + 73))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 48))], placeholder_shared[(((((int)threadIdx.z) * 98) + 73))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 43))], placeholder_shared[(((((int)threadIdx.z) * 98) + 25))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 47))], placeholder_shared[(((((int)threadIdx.z) * 98) + 25))], compute[(4)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 45))], placeholder_shared[(((((int)threadIdx.z) * 98) + 25))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 49))], placeholder_shared[(((((int)threadIdx.z) * 98) + 25))], compute[(5)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 43))], placeholder_shared[(((((int)threadIdx.z) * 98) + 74))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 47))], placeholder_shared[(((((int)threadIdx.z) * 98) + 74))], compute[(6)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 45))], placeholder_shared[(((((int)threadIdx.z) * 98) + 74))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 49))], placeholder_shared[(((((int)threadIdx.z) * 98) + 74))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 44))], placeholder_shared[(((((int)threadIdx.z) * 98) + 26))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 48))], placeholder_shared[(((((int)threadIdx.z) * 98) + 26))], compute[(4)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 46))], placeholder_shared[(((((int)threadIdx.z) * 98) + 26))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 50))], placeholder_shared[(((((int)threadIdx.z) * 98) + 26))], compute[(5)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 44))], placeholder_shared[(((((int)threadIdx.z) * 98) + 75))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 48))], placeholder_shared[(((((int)threadIdx.z) * 98) + 75))], compute[(6)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 46))], placeholder_shared[(((((int)threadIdx.z) * 98) + 75))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 50))], placeholder_shared[(((((int)threadIdx.z) * 98) + 75))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 45))], placeholder_shared[(((((int)threadIdx.z) * 98) + 27))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 49))], placeholder_shared[(((((int)threadIdx.z) * 98) + 27))], compute[(4)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 47))], placeholder_shared[(((((int)threadIdx.z) * 98) + 27))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 51))], placeholder_shared[(((((int)threadIdx.z) * 98) + 27))], compute[(5)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 45))], placeholder_shared[(((((int)threadIdx.z) * 98) + 76))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 49))], placeholder_shared[(((((int)threadIdx.z) * 98) + 76))], compute[(6)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 47))], placeholder_shared[(((((int)threadIdx.z) * 98) + 76))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 51))], placeholder_shared[(((((int)threadIdx.z) * 98) + 76))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 52))], placeholder_shared[(((((int)threadIdx.z) * 98) + 28))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 56))], placeholder_shared[(((((int)threadIdx.z) * 98) + 28))], compute[(4)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 54))], placeholder_shared[(((((int)threadIdx.z) * 98) + 28))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 58))], placeholder_shared[(((((int)threadIdx.z) * 98) + 28))], compute[(5)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 52))], placeholder_shared[(((((int)threadIdx.z) * 98) + 77))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 56))], placeholder_shared[(((((int)threadIdx.z) * 98) + 77))], compute[(6)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 54))], placeholder_shared[(((((int)threadIdx.z) * 98) + 77))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 58))], placeholder_shared[(((((int)threadIdx.z) * 98) + 77))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 53))], placeholder_shared[(((((int)threadIdx.z) * 98) + 29))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 57))], placeholder_shared[(((((int)threadIdx.z) * 98) + 29))], compute[(4)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 55))], placeholder_shared[(((((int)threadIdx.z) * 98) + 29))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 59))], placeholder_shared[(((((int)threadIdx.z) * 98) + 29))], compute[(5)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 53))], placeholder_shared[(((((int)threadIdx.z) * 98) + 78))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 57))], placeholder_shared[(((((int)threadIdx.z) * 98) + 78))], compute[(6)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 55))], placeholder_shared[(((((int)threadIdx.z) * 98) + 78))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 59))], placeholder_shared[(((((int)threadIdx.z) * 98) + 78))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 54))], placeholder_shared[(((((int)threadIdx.z) * 98) + 30))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 58))], placeholder_shared[(((((int)threadIdx.z) * 98) + 30))], compute[(4)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 56))], placeholder_shared[(((((int)threadIdx.z) * 98) + 30))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 60))], placeholder_shared[(((((int)threadIdx.z) * 98) + 30))], compute[(5)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 54))], placeholder_shared[(((((int)threadIdx.z) * 98) + 79))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 58))], placeholder_shared[(((((int)threadIdx.z) * 98) + 79))], compute[(6)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 56))], placeholder_shared[(((((int)threadIdx.z) * 98) + 79))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 60))], placeholder_shared[(((((int)threadIdx.z) * 98) + 79))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 55))], placeholder_shared[(((((int)threadIdx.z) * 98) + 31))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 59))], placeholder_shared[(((((int)threadIdx.z) * 98) + 31))], compute[(4)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 57))], placeholder_shared[(((((int)threadIdx.z) * 98) + 31))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 61))], placeholder_shared[(((((int)threadIdx.z) * 98) + 31))], compute[(5)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 55))], placeholder_shared[(((((int)threadIdx.z) * 98) + 80))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 59))], placeholder_shared[(((((int)threadIdx.z) * 98) + 80))], compute[(6)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 57))], placeholder_shared[(((((int)threadIdx.z) * 98) + 80))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 61))], placeholder_shared[(((((int)threadIdx.z) * 98) + 80))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 56))], placeholder_shared[(((((int)threadIdx.z) * 98) + 32))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 60))], placeholder_shared[(((((int)threadIdx.z) * 98) + 32))], compute[(4)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 58))], placeholder_shared[(((((int)threadIdx.z) * 98) + 32))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 62))], placeholder_shared[(((((int)threadIdx.z) * 98) + 32))], compute[(5)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 56))], placeholder_shared[(((((int)threadIdx.z) * 98) + 81))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 60))], placeholder_shared[(((((int)threadIdx.z) * 98) + 81))], compute[(6)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 58))], placeholder_shared[(((((int)threadIdx.z) * 98) + 81))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 62))], placeholder_shared[(((((int)threadIdx.z) * 98) + 81))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 57))], placeholder_shared[(((((int)threadIdx.z) * 98) + 33))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 61))], placeholder_shared[(((((int)threadIdx.z) * 98) + 33))], compute[(4)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 59))], placeholder_shared[(((((int)threadIdx.z) * 98) + 33))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 63))], placeholder_shared[(((((int)threadIdx.z) * 98) + 33))], compute[(5)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 57))], placeholder_shared[(((((int)threadIdx.z) * 98) + 82))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 61))], placeholder_shared[(((((int)threadIdx.z) * 98) + 82))], compute[(6)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 59))], placeholder_shared[(((((int)threadIdx.z) * 98) + 82))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 63))], placeholder_shared[(((((int)threadIdx.z) * 98) + 82))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 58))], placeholder_shared[(((((int)threadIdx.z) * 98) + 34))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 62))], placeholder_shared[(((((int)threadIdx.z) * 98) + 34))], compute[(4)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 60))], placeholder_shared[(((((int)threadIdx.z) * 98) + 34))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 64))], placeholder_shared[(((((int)threadIdx.z) * 98) + 34))], compute[(5)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 58))], placeholder_shared[(((((int)threadIdx.z) * 98) + 83))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 62))], placeholder_shared[(((((int)threadIdx.z) * 98) + 83))], compute[(6)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 60))], placeholder_shared[(((((int)threadIdx.z) * 98) + 83))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 64))], placeholder_shared[(((((int)threadIdx.z) * 98) + 83))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 65))], placeholder_shared[(((((int)threadIdx.z) * 98) + 35))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 69))], placeholder_shared[(((((int)threadIdx.z) * 98) + 35))], compute[(4)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 67))], placeholder_shared[(((((int)threadIdx.z) * 98) + 35))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 71))], placeholder_shared[(((((int)threadIdx.z) * 98) + 35))], compute[(5)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 65))], placeholder_shared[(((((int)threadIdx.z) * 98) + 84))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 69))], placeholder_shared[(((((int)threadIdx.z) * 98) + 84))], compute[(6)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 67))], placeholder_shared[(((((int)threadIdx.z) * 98) + 84))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 71))], placeholder_shared[(((((int)threadIdx.z) * 98) + 84))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 66))], placeholder_shared[(((((int)threadIdx.z) * 98) + 36))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 70))], placeholder_shared[(((((int)threadIdx.z) * 98) + 36))], compute[(4)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 68))], placeholder_shared[(((((int)threadIdx.z) * 98) + 36))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 72))], placeholder_shared[(((((int)threadIdx.z) * 98) + 36))], compute[(5)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 66))], placeholder_shared[(((((int)threadIdx.z) * 98) + 85))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 70))], placeholder_shared[(((((int)threadIdx.z) * 98) + 85))], compute[(6)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 68))], placeholder_shared[(((((int)threadIdx.z) * 98) + 85))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 72))], placeholder_shared[(((((int)threadIdx.z) * 98) + 85))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 67))], placeholder_shared[(((((int)threadIdx.z) * 98) + 37))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 71))], placeholder_shared[(((((int)threadIdx.z) * 98) + 37))], compute[(4)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 69))], placeholder_shared[(((((int)threadIdx.z) * 98) + 37))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 73))], placeholder_shared[(((((int)threadIdx.z) * 98) + 37))], compute[(5)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 67))], placeholder_shared[(((((int)threadIdx.z) * 98) + 86))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 71))], placeholder_shared[(((((int)threadIdx.z) * 98) + 86))], compute[(6)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 69))], placeholder_shared[(((((int)threadIdx.z) * 98) + 86))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 73))], placeholder_shared[(((((int)threadIdx.z) * 98) + 86))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 68))], placeholder_shared[(((((int)threadIdx.z) * 98) + 38))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 72))], placeholder_shared[(((((int)threadIdx.z) * 98) + 38))], compute[(4)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 70))], placeholder_shared[(((((int)threadIdx.z) * 98) + 38))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 74))], placeholder_shared[(((((int)threadIdx.z) * 98) + 38))], compute[(5)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 68))], placeholder_shared[(((((int)threadIdx.z) * 98) + 87))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 72))], placeholder_shared[(((((int)threadIdx.z) * 98) + 87))], compute[(6)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 70))], placeholder_shared[(((((int)threadIdx.z) * 98) + 87))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 74))], placeholder_shared[(((((int)threadIdx.z) * 98) + 87))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 69))], placeholder_shared[(((((int)threadIdx.z) * 98) + 39))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 73))], placeholder_shared[(((((int)threadIdx.z) * 98) + 39))], compute[(4)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 71))], placeholder_shared[(((((int)threadIdx.z) * 98) + 39))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 75))], placeholder_shared[(((((int)threadIdx.z) * 98) + 39))], compute[(5)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 69))], placeholder_shared[(((((int)threadIdx.z) * 98) + 88))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 73))], placeholder_shared[(((((int)threadIdx.z) * 98) + 88))], compute[(6)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 71))], placeholder_shared[(((((int)threadIdx.z) * 98) + 88))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 75))], placeholder_shared[(((((int)threadIdx.z) * 98) + 88))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 70))], placeholder_shared[(((((int)threadIdx.z) * 98) + 40))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 74))], placeholder_shared[(((((int)threadIdx.z) * 98) + 40))], compute[(4)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 72))], placeholder_shared[(((((int)threadIdx.z) * 98) + 40))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 76))], placeholder_shared[(((((int)threadIdx.z) * 98) + 40))], compute[(5)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 70))], placeholder_shared[(((((int)threadIdx.z) * 98) + 89))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 74))], placeholder_shared[(((((int)threadIdx.z) * 98) + 89))], compute[(6)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 72))], placeholder_shared[(((((int)threadIdx.z) * 98) + 89))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 76))], placeholder_shared[(((((int)threadIdx.z) * 98) + 89))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 71))], placeholder_shared[(((((int)threadIdx.z) * 98) + 41))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 75))], placeholder_shared[(((((int)threadIdx.z) * 98) + 41))], compute[(4)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 73))], placeholder_shared[(((((int)threadIdx.z) * 98) + 41))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 77))], placeholder_shared[(((((int)threadIdx.z) * 98) + 41))], compute[(5)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 71))], placeholder_shared[(((((int)threadIdx.z) * 98) + 90))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 75))], placeholder_shared[(((((int)threadIdx.z) * 98) + 90))], compute[(6)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 73))], placeholder_shared[(((((int)threadIdx.z) * 98) + 90))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 77))], placeholder_shared[(((((int)threadIdx.z) * 98) + 90))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 78))], placeholder_shared[(((((int)threadIdx.z) * 98) + 42))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 82))], placeholder_shared[(((((int)threadIdx.z) * 98) + 42))], compute[(4)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 80))], placeholder_shared[(((((int)threadIdx.z) * 98) + 42))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 84))], placeholder_shared[(((((int)threadIdx.z) * 98) + 42))], compute[(5)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 78))], placeholder_shared[(((((int)threadIdx.z) * 98) + 91))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 82))], placeholder_shared[(((((int)threadIdx.z) * 98) + 91))], compute[(6)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 80))], placeholder_shared[(((((int)threadIdx.z) * 98) + 91))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 84))], placeholder_shared[(((((int)threadIdx.z) * 98) + 91))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 79))], placeholder_shared[(((((int)threadIdx.z) * 98) + 43))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 83))], placeholder_shared[(((((int)threadIdx.z) * 98) + 43))], compute[(4)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 81))], placeholder_shared[(((((int)threadIdx.z) * 98) + 43))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 85))], placeholder_shared[(((((int)threadIdx.z) * 98) + 43))], compute[(5)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 79))], placeholder_shared[(((((int)threadIdx.z) * 98) + 92))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 83))], placeholder_shared[(((((int)threadIdx.z) * 98) + 92))], compute[(6)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 81))], placeholder_shared[(((((int)threadIdx.z) * 98) + 92))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 85))], placeholder_shared[(((((int)threadIdx.z) * 98) + 92))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 80))], placeholder_shared[(((((int)threadIdx.z) * 98) + 44))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 84))], placeholder_shared[(((((int)threadIdx.z) * 98) + 44))], compute[(4)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 82))], placeholder_shared[(((((int)threadIdx.z) * 98) + 44))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 86))], placeholder_shared[(((((int)threadIdx.z) * 98) + 44))], compute[(5)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 80))], placeholder_shared[(((((int)threadIdx.z) * 98) + 93))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 84))], placeholder_shared[(((((int)threadIdx.z) * 98) + 93))], compute[(6)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 82))], placeholder_shared[(((((int)threadIdx.z) * 98) + 93))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 86))], placeholder_shared[(((((int)threadIdx.z) * 98) + 93))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 81))], placeholder_shared[(((((int)threadIdx.z) * 98) + 45))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 85))], placeholder_shared[(((((int)threadIdx.z) * 98) + 45))], compute[(4)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 83))], placeholder_shared[(((((int)threadIdx.z) * 98) + 45))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 87))], placeholder_shared[(((((int)threadIdx.z) * 98) + 45))], compute[(5)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 81))], placeholder_shared[(((((int)threadIdx.z) * 98) + 94))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 85))], placeholder_shared[(((((int)threadIdx.z) * 98) + 94))], compute[(6)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 83))], placeholder_shared[(((((int)threadIdx.z) * 98) + 94))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 87))], placeholder_shared[(((((int)threadIdx.z) * 98) + 94))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 82))], placeholder_shared[(((((int)threadIdx.z) * 98) + 46))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 86))], placeholder_shared[(((((int)threadIdx.z) * 98) + 46))], compute[(4)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 84))], placeholder_shared[(((((int)threadIdx.z) * 98) + 46))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 88))], placeholder_shared[(((((int)threadIdx.z) * 98) + 46))], compute[(5)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 82))], placeholder_shared[(((((int)threadIdx.z) * 98) + 95))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 86))], placeholder_shared[(((((int)threadIdx.z) * 98) + 95))], compute[(6)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 84))], placeholder_shared[(((((int)threadIdx.z) * 98) + 95))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 88))], placeholder_shared[(((((int)threadIdx.z) * 98) + 95))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 83))], placeholder_shared[(((((int)threadIdx.z) * 98) + 47))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 87))], placeholder_shared[(((((int)threadIdx.z) * 98) + 47))], compute[(4)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 85))], placeholder_shared[(((((int)threadIdx.z) * 98) + 47))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 89))], placeholder_shared[(((((int)threadIdx.z) * 98) + 47))], compute[(5)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 83))], placeholder_shared[(((((int)threadIdx.z) * 98) + 96))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 87))], placeholder_shared[(((((int)threadIdx.z) * 98) + 96))], compute[(6)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 85))], placeholder_shared[(((((int)threadIdx.z) * 98) + 96))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 89))], placeholder_shared[(((((int)threadIdx.z) * 98) + 96))], compute[(7)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 84))], placeholder_shared[(((((int)threadIdx.z) * 98) + 48))], compute[(0)]);
    compute[(4)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 88))], placeholder_shared[(((((int)threadIdx.z) * 98) + 48))], compute[(4)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 86))], placeholder_shared[(((((int)threadIdx.z) * 98) + 48))], compute[(1)]);
    compute[(5)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 90))], placeholder_shared[(((((int)threadIdx.z) * 98) + 48))], compute[(5)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 84))], placeholder_shared[(((((int)threadIdx.z) * 98) + 97))], compute[(2)]);
    compute[(6)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 88))], placeholder_shared[(((((int)threadIdx.z) * 98) + 97))], compute[(6)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 86))], placeholder_shared[(((((int)threadIdx.z) * 98) + 97))], compute[(3)]);
    compute[(7)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + 90))], placeholder_shared[(((((int)threadIdx.z) * 98) + 97))], compute[(7)]);
  }
  T_relu[((((((((int)blockIdx.z) * 32768) + (((int)threadIdx.z) * 2048)) + (((int)blockIdx.y) * 512)) + (((int)threadIdx.y) * 32)) + (((int)blockIdx.x) * 4)))] = max((compute[(0)] + placeholder2[(((((int)blockIdx.z) * 32) + (((int)threadIdx.z) * 2)))]), 0.000000e+00f);
  T_relu[(((((((((int)blockIdx.z) * 32768) + (((int)threadIdx.z) * 2048)) + (((int)blockIdx.y) * 512)) + (((int)threadIdx.y) * 32)) + (((int)blockIdx.x) * 4)) + 2))] = max((compute[(4)] + placeholder2[(((((int)blockIdx.z) * 32) + (((int)threadIdx.z) * 2)))]), 0.000000e+00f);
  T_relu[(((((((((int)blockIdx.z) * 32768) + (((int)threadIdx.z) * 2048)) + (((int)blockIdx.y) * 512)) + (((int)threadIdx.y) * 32)) + (((int)blockIdx.x) * 4)) + 1))] = max((compute[(1)] + placeholder2[(((((int)blockIdx.z) * 32) + (((int)threadIdx.z) * 2)))]), 0.000000e+00f);
  T_relu[(((((((((int)blockIdx.z) * 32768) + (((int)threadIdx.z) * 2048)) + (((int)blockIdx.y) * 512)) + (((int)threadIdx.y) * 32)) + (((int)blockIdx.x) * 4)) + 3))] = max((compute[(5)] + placeholder2[(((((int)blockIdx.z) * 32) + (((int)threadIdx.z) * 2)))]), 0.000000e+00f);
  T_relu[(((((((((int)blockIdx.z) * 32768) + (((int)threadIdx.z) * 2048)) + (((int)blockIdx.y) * 512)) + (((int)threadIdx.y) * 32)) + (((int)blockIdx.x) * 4)) + 1024))] = max((compute[(2)] + placeholder2[((((((int)blockIdx.z) * 32) + (((int)threadIdx.z) * 2)) + 1))]), 0.000000e+00f);
  T_relu[(((((((((int)blockIdx.z) * 32768) + (((int)threadIdx.z) * 2048)) + (((int)blockIdx.y) * 512)) + (((int)threadIdx.y) * 32)) + (((int)blockIdx.x) * 4)) + 1026))] = max((compute[(6)] + placeholder2[((((((int)blockIdx.z) * 32) + (((int)threadIdx.z) * 2)) + 1))]), 0.000000e+00f);
  T_relu[(((((((((int)blockIdx.z) * 32768) + (((int)threadIdx.z) * 2048)) + (((int)blockIdx.y) * 512)) + (((int)threadIdx.y) * 32)) + (((int)blockIdx.x) * 4)) + 1025))] = max((compute[(3)] + placeholder2[((((((int)blockIdx.z) * 32) + (((int)threadIdx.z) * 2)) + 1))]), 0.000000e+00f);
  T_relu[(((((((((int)blockIdx.z) * 32768) + (((int)threadIdx.z) * 2048)) + (((int)blockIdx.y) * 512)) + (((int)threadIdx.y) * 32)) + (((int)blockIdx.x) * 4)) + 1027))] = max((compute[(7)] + placeholder2[((((((int)blockIdx.z) * 32) + (((int)threadIdx.z) * 2)) + 1))]), 0.000000e+00f);
}

extern "C" __global__ void tvmgen_default_fused_nn_softmax_kernel0(float* __restrict__ placeholder, float* __restrict__ T_softmax_norm) {
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
  uint mask[1];
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
      normal_reduce_temp01[(0)] = (normal_reduce_temp01[(0)] + T_softmax_exp[(k_inner1)]);
    }
  }
  uint mask1[1];
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
        T_softmax_norm[((((((int)threadIdx.x) * 16) + (i1_inner_outer1 * 4)) + i1_inner_inner_s1))] = (T_softmax_exp[(((i1_inner_outer1 * 4) + i1_inner_inner_s1))] / red_buf01[(0)]);
      }
    }
  }
}

extern "C" __global__ void tvmgen_default_fused_add_nn_relu_2_kernel0(float* __restrict__ T_relu, float* __restrict__ placeholder, float* __restrict__ placeholder1) {
  T_relu[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)))] = max((placeholder[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)))] + placeholder1[(((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 6)))]), 0.000000e+00f);
}

extern "C" __global__ void tvmgen_default_fused_nn_conv2d_add_add_nn_relu_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2, float* __restrict__ placeholder3) {
  float compute[1];
  __shared__ float pad_temp_shared[64];
  __shared__ float placeholder_shared[1024];
  compute[(0)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 16; ++rc_outer) {
    __syncthreads();
    if ((((int)threadIdx.x) + ((int)threadIdx.z)) < 64) {
      if (((int)threadIdx.x) < 1) {
        pad_temp_shared[((((int)threadIdx.x) + ((int)threadIdx.z)))] = placeholder[(((((rc_outer * 256) + (((((int)threadIdx.x) + ((int)threadIdx.z)) >> 2) * 16)) + (((int)blockIdx.y) * 4)) + ((((int)threadIdx.x) + ((int)threadIdx.z)) & 3)))];
      }
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      placeholder_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))] = placeholder1[((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (rc_outer * 16)) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))];
    }
    __syncthreads();
    for (int rc_inner = 0; rc_inner < 16; ++rc_inner) {
      compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((rc_inner * 4) + ((int)threadIdx.x)))], placeholder_shared[(((((int)threadIdx.z) * 16) + rc_inner))], compute[(0)]);
    }
  }
  T_relu[(((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 16)) + (((int)blockIdx.y) * 4)) + ((int)threadIdx.x)))] = max(((compute[(0)] + placeholder2[(((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 16)) + (((int)blockIdx.y) * 4)) + ((int)threadIdx.x)))]) + placeholder3[(((((int)blockIdx.z) * 64) + ((int)threadIdx.z)))]), 0.000000e+00f);
}

extern "C" __global__ void tvmgen_default_fused_nn_conv2d_add_nn_relu_7_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  float compute[1];
  __shared__ float pad_temp_shared[2048];
  __shared__ float placeholder_shared[2048];
  compute[(0)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 4; ++rc_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 8; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      pad_temp_shared[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))] = placeholder[(((((((((rc_outer * 8192) + (((int)threadIdx.z) * 512)) + (((int)threadIdx.y) * 128)) + ((((((int)threadIdx.x) * 2) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner >> 2)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.x) * 2) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner >> 2)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner & 3)))];
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 < 8; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) {
      placeholder_shared[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1))] = placeholder1[(((((((((int)blockIdx.z) * 8192) + (((int)threadIdx.z) * 512)) + (rc_outer * 128)) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1))];
    }
    __syncthreads();
    for (int rc_inner = 0; rc_inner < 128; ++rc_inner) {
      compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((rc_inner * 16) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)))], placeholder_shared[(((((int)threadIdx.z) * 128) + rc_inner))], compute[(0)]);
    }
  }
  T_relu[(((((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 64)) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.y) * 8)) + (((int)blockIdx.x) * 4)) + ((int)threadIdx.x)))] = max((compute[(0)] + placeholder2[(((((int)blockIdx.z) * 16) + ((int)threadIdx.z)))]), 0.000000e+00f);
}

extern "C" __global__ void tvmgen_default_fused_nn_conv2d_3_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ compute) {
  float compute_local[2];
  __shared__ float pad_temp_shared[144];
  __shared__ float placeholder_shared[256];
  compute_local[(0)] = 0.000000e+00f;
  compute_local[(1)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 64; ++rc_outer) {
    __syncthreads();
    pad_temp_shared[((((((int)threadIdx.z) * 18) + (((int)threadIdx.y) * 9)) + (((int)threadIdx.x) * 5)))] = placeholder[((((((rc_outer * 256) + (((int)threadIdx.z) * 32)) + (((int)threadIdx.y) * 16)) + (((((int)threadIdx.x) * 5) / 3) * 4)) + ((((int)threadIdx.x) * 5) % 3)))];
    pad_temp_shared[(((((((int)threadIdx.z) * 18) + (((int)threadIdx.y) * 9)) + (((int)threadIdx.x) * 5)) + 1))] = placeholder[((((((rc_outer * 256) + (((int)threadIdx.z) * 32)) + (((int)threadIdx.y) * 16)) + ((((((int)threadIdx.x) * 5) + 1) / 3) * 4)) + (((((int)threadIdx.x) * 5) + 1) % 3)))];
    pad_temp_shared[(((((((int)threadIdx.z) * 18) + (((int)threadIdx.y) * 9)) + (((int)threadIdx.x) * 5)) + 2))] = placeholder[((((((rc_outer * 256) + (((int)threadIdx.z) * 32)) + (((int)threadIdx.y) * 16)) + ((((((int)threadIdx.x) * 5) + 2) / 3) * 4)) + (((((int)threadIdx.x) * 5) + 2) % 3)))];
    pad_temp_shared[(((((((int)threadIdx.z) * 18) + (((int)threadIdx.y) * 9)) + (((int)threadIdx.x) * 5)) + 3))] = placeholder[(((((((rc_outer * 256) + (((int)threadIdx.z) * 32)) + (((int)threadIdx.y) * 16)) + (((((int)threadIdx.x) * 5) / 3) * 4)) + ((((int)threadIdx.x) * 5) % 3)) + 4))];
    if ((((((int)threadIdx.z) * 2) + (((((int)threadIdx.x) * 5) + 4) / 9)) + ((int)threadIdx.y)) < 16) {
      if ((((((int)threadIdx.z) * 6) + (((int)threadIdx.y) * 3)) + (((((int)threadIdx.x) * 5) + 4) / 3)) < 48) {
        if ((((((int)threadIdx.z) * 18) + (((int)threadIdx.y) * 9)) + (((int)threadIdx.x) * 5)) < 140) {
          if (((((int)threadIdx.y) * 9) + (((int)threadIdx.x) * 5)) < 14) {
            if (((int)threadIdx.x) < 1) {
              pad_temp_shared[(((((((int)threadIdx.z) * 18) + (((int)threadIdx.y) * 9)) + (((int)threadIdx.x) * 5)) + 4))] = placeholder[((((((rc_outer * 256) + (((int)threadIdx.z) * 32)) + (((int)threadIdx.y) * 16)) + ((((((int)threadIdx.x) * 5) + 4) / 3) * 4)) + ((((int)threadIdx.x) * 5) + 1)))];
            }
          }
        }
      }
    }
    placeholder_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 8)))] = placeholder1[((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 2048)) + (((int)threadIdx.y) * 1024)) + (rc_outer * 16)) + (((int)threadIdx.x) * 8)))];
    placeholder_shared[(((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 8)) + 1))] = placeholder1[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 2048)) + (((int)threadIdx.y) * 1024)) + (rc_outer * 16)) + (((int)threadIdx.x) * 8)) + 1))];
    placeholder_shared[(((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 8)) + 2))] = placeholder1[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 2048)) + (((int)threadIdx.y) * 1024)) + (rc_outer * 16)) + (((int)threadIdx.x) * 8)) + 2))];
    placeholder_shared[(((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 8)) + 3))] = placeholder1[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 2048)) + (((int)threadIdx.y) * 1024)) + (rc_outer * 16)) + (((int)threadIdx.x) * 8)) + 3))];
    placeholder_shared[(((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 8)) + 4))] = placeholder1[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 2048)) + (((int)threadIdx.y) * 1024)) + (rc_outer * 16)) + (((int)threadIdx.x) * 8)) + 4))];
    placeholder_shared[(((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 8)) + 5))] = placeholder1[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 2048)) + (((int)threadIdx.y) * 1024)) + (rc_outer * 16)) + (((int)threadIdx.x) * 8)) + 5))];
    placeholder_shared[(((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 8)) + 6))] = placeholder1[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 2048)) + (((int)threadIdx.y) * 1024)) + (rc_outer * 16)) + (((int)threadIdx.x) * 8)) + 6))];
    placeholder_shared[(((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 8)) + 7))] = placeholder1[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 2048)) + (((int)threadIdx.y) * 1024)) + (rc_outer * 16)) + (((int)threadIdx.x) * 8)) + 7))];
    __syncthreads();
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)))], placeholder_shared[((((int)threadIdx.z) * 16))], compute_local[(0)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)))], placeholder_shared[(((((int)threadIdx.z) * 16) + 128))], compute_local[(1)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 9))], placeholder_shared[(((((int)threadIdx.z) * 16) + 1))], compute_local[(0)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 9))], placeholder_shared[(((((int)threadIdx.z) * 16) + 129))], compute_local[(1)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 18))], placeholder_shared[(((((int)threadIdx.z) * 16) + 2))], compute_local[(0)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 18))], placeholder_shared[(((((int)threadIdx.z) * 16) + 130))], compute_local[(1)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 27))], placeholder_shared[(((((int)threadIdx.z) * 16) + 3))], compute_local[(0)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 27))], placeholder_shared[(((((int)threadIdx.z) * 16) + 131))], compute_local[(1)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 36))], placeholder_shared[(((((int)threadIdx.z) * 16) + 4))], compute_local[(0)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 36))], placeholder_shared[(((((int)threadIdx.z) * 16) + 132))], compute_local[(1)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 45))], placeholder_shared[(((((int)threadIdx.z) * 16) + 5))], compute_local[(0)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 45))], placeholder_shared[(((((int)threadIdx.z) * 16) + 133))], compute_local[(1)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 54))], placeholder_shared[(((((int)threadIdx.z) * 16) + 6))], compute_local[(0)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 54))], placeholder_shared[(((((int)threadIdx.z) * 16) + 134))], compute_local[(1)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 63))], placeholder_shared[(((((int)threadIdx.z) * 16) + 7))], compute_local[(0)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 63))], placeholder_shared[(((((int)threadIdx.z) * 16) + 135))], compute_local[(1)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 72))], placeholder_shared[(((((int)threadIdx.z) * 16) + 8))], compute_local[(0)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 72))], placeholder_shared[(((((int)threadIdx.z) * 16) + 136))], compute_local[(1)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 81))], placeholder_shared[(((((int)threadIdx.z) * 16) + 9))], compute_local[(0)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 81))], placeholder_shared[(((((int)threadIdx.z) * 16) + 137))], compute_local[(1)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 90))], placeholder_shared[(((((int)threadIdx.z) * 16) + 10))], compute_local[(0)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 90))], placeholder_shared[(((((int)threadIdx.z) * 16) + 138))], compute_local[(1)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 99))], placeholder_shared[(((((int)threadIdx.z) * 16) + 11))], compute_local[(0)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 99))], placeholder_shared[(((((int)threadIdx.z) * 16) + 139))], compute_local[(1)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 108))], placeholder_shared[(((((int)threadIdx.z) * 16) + 12))], compute_local[(0)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 108))], placeholder_shared[(((((int)threadIdx.z) * 16) + 140))], compute_local[(1)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 117))], placeholder_shared[(((((int)threadIdx.z) * 16) + 13))], compute_local[(0)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 117))], placeholder_shared[(((((int)threadIdx.z) * 16) + 141))], compute_local[(1)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 126))], placeholder_shared[(((((int)threadIdx.z) * 16) + 14))], compute_local[(0)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 126))], placeholder_shared[(((((int)threadIdx.z) * 16) + 142))], compute_local[(1)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 135))], placeholder_shared[(((((int)threadIdx.z) * 16) + 15))], compute_local[(0)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 135))], placeholder_shared[(((((int)threadIdx.z) * 16) + 143))], compute_local[(1)]);
  }
  compute[(((((((int)blockIdx.z) * 64) + (((int)threadIdx.z) * 4)) + (((int)threadIdx.y) * 2)) + ((int)threadIdx.x)))] = compute_local[(0)];
  compute[((((((((int)blockIdx.z) * 64) + (((int)threadIdx.z) * 4)) + (((int)threadIdx.y) * 2)) + ((int)threadIdx.x)) + 32))] = compute_local[(1)];
}

extern "C" __global__ void tvmgen_default_fused_nn_conv2d_add_nn_relu_2_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  float compute[2];
  __shared__ float pad_temp_shared[144];
  __shared__ float placeholder_shared[256];
  compute[(0)] = 0.000000e+00f;
  compute[(1)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 64; ++rc_outer) {
    __syncthreads();
    pad_temp_shared[((((((int)threadIdx.z) * 18) + (((int)threadIdx.y) * 9)) + (((int)threadIdx.x) * 5)))] = placeholder[((((((rc_outer * 256) + (((int)threadIdx.z) * 32)) + (((int)threadIdx.y) * 16)) + (((((int)threadIdx.x) * 5) / 3) * 4)) + ((((int)threadIdx.x) * 5) % 3)))];
    pad_temp_shared[(((((((int)threadIdx.z) * 18) + (((int)threadIdx.y) * 9)) + (((int)threadIdx.x) * 5)) + 1))] = placeholder[((((((rc_outer * 256) + (((int)threadIdx.z) * 32)) + (((int)threadIdx.y) * 16)) + ((((((int)threadIdx.x) * 5) + 1) / 3) * 4)) + (((((int)threadIdx.x) * 5) + 1) % 3)))];
    pad_temp_shared[(((((((int)threadIdx.z) * 18) + (((int)threadIdx.y) * 9)) + (((int)threadIdx.x) * 5)) + 2))] = placeholder[((((((rc_outer * 256) + (((int)threadIdx.z) * 32)) + (((int)threadIdx.y) * 16)) + ((((((int)threadIdx.x) * 5) + 2) / 3) * 4)) + (((((int)threadIdx.x) * 5) + 2) % 3)))];
    pad_temp_shared[(((((((int)threadIdx.z) * 18) + (((int)threadIdx.y) * 9)) + (((int)threadIdx.x) * 5)) + 3))] = placeholder[(((((((rc_outer * 256) + (((int)threadIdx.z) * 32)) + (((int)threadIdx.y) * 16)) + (((((int)threadIdx.x) * 5) / 3) * 4)) + ((((int)threadIdx.x) * 5) % 3)) + 4))];
    if ((((((int)threadIdx.z) * 2) + (((((int)threadIdx.x) * 5) + 4) / 9)) + ((int)threadIdx.y)) < 16) {
      if ((((((int)threadIdx.z) * 6) + (((int)threadIdx.y) * 3)) + (((((int)threadIdx.x) * 5) + 4) / 3)) < 48) {
        if ((((((int)threadIdx.z) * 18) + (((int)threadIdx.y) * 9)) + (((int)threadIdx.x) * 5)) < 140) {
          if (((((int)threadIdx.y) * 9) + (((int)threadIdx.x) * 5)) < 14) {
            if (((int)threadIdx.x) < 1) {
              pad_temp_shared[(((((((int)threadIdx.z) * 18) + (((int)threadIdx.y) * 9)) + (((int)threadIdx.x) * 5)) + 4))] = placeholder[((((((rc_outer * 256) + (((int)threadIdx.z) * 32)) + (((int)threadIdx.y) * 16)) + ((((((int)threadIdx.x) * 5) + 4) / 3) * 4)) + ((((int)threadIdx.x) * 5) + 1)))];
            }
          }
        }
      }
    }
    placeholder_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 8)))] = placeholder1[((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 2048)) + (((int)threadIdx.y) * 1024)) + (rc_outer * 16)) + (((int)threadIdx.x) * 8)))];
    placeholder_shared[(((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 8)) + 1))] = placeholder1[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 2048)) + (((int)threadIdx.y) * 1024)) + (rc_outer * 16)) + (((int)threadIdx.x) * 8)) + 1))];
    placeholder_shared[(((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 8)) + 2))] = placeholder1[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 2048)) + (((int)threadIdx.y) * 1024)) + (rc_outer * 16)) + (((int)threadIdx.x) * 8)) + 2))];
    placeholder_shared[(((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 8)) + 3))] = placeholder1[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 2048)) + (((int)threadIdx.y) * 1024)) + (rc_outer * 16)) + (((int)threadIdx.x) * 8)) + 3))];
    placeholder_shared[(((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 8)) + 4))] = placeholder1[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 2048)) + (((int)threadIdx.y) * 1024)) + (rc_outer * 16)) + (((int)threadIdx.x) * 8)) + 4))];
    placeholder_shared[(((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 8)) + 5))] = placeholder1[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 2048)) + (((int)threadIdx.y) * 1024)) + (rc_outer * 16)) + (((int)threadIdx.x) * 8)) + 5))];
    placeholder_shared[(((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 8)) + 6))] = placeholder1[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 2048)) + (((int)threadIdx.y) * 1024)) + (rc_outer * 16)) + (((int)threadIdx.x) * 8)) + 6))];
    placeholder_shared[(((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 8)) + 7))] = placeholder1[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 2048)) + (((int)threadIdx.y) * 1024)) + (rc_outer * 16)) + (((int)threadIdx.x) * 8)) + 7))];
    __syncthreads();
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)))], placeholder_shared[((((int)threadIdx.z) * 16))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)))], placeholder_shared[(((((int)threadIdx.z) * 16) + 128))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 9))], placeholder_shared[(((((int)threadIdx.z) * 16) + 1))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 9))], placeholder_shared[(((((int)threadIdx.z) * 16) + 129))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 18))], placeholder_shared[(((((int)threadIdx.z) * 16) + 2))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 18))], placeholder_shared[(((((int)threadIdx.z) * 16) + 130))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 27))], placeholder_shared[(((((int)threadIdx.z) * 16) + 3))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 27))], placeholder_shared[(((((int)threadIdx.z) * 16) + 131))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 36))], placeholder_shared[(((((int)threadIdx.z) * 16) + 4))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 36))], placeholder_shared[(((((int)threadIdx.z) * 16) + 132))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 45))], placeholder_shared[(((((int)threadIdx.z) * 16) + 5))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 45))], placeholder_shared[(((((int)threadIdx.z) * 16) + 133))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 54))], placeholder_shared[(((((int)threadIdx.z) * 16) + 6))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 54))], placeholder_shared[(((((int)threadIdx.z) * 16) + 134))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 63))], placeholder_shared[(((((int)threadIdx.z) * 16) + 7))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 63))], placeholder_shared[(((((int)threadIdx.z) * 16) + 135))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 72))], placeholder_shared[(((((int)threadIdx.z) * 16) + 8))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 72))], placeholder_shared[(((((int)threadIdx.z) * 16) + 136))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 81))], placeholder_shared[(((((int)threadIdx.z) * 16) + 9))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 81))], placeholder_shared[(((((int)threadIdx.z) * 16) + 137))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 90))], placeholder_shared[(((((int)threadIdx.z) * 16) + 10))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 90))], placeholder_shared[(((((int)threadIdx.z) * 16) + 138))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 99))], placeholder_shared[(((((int)threadIdx.z) * 16) + 11))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 99))], placeholder_shared[(((((int)threadIdx.z) * 16) + 139))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 108))], placeholder_shared[(((((int)threadIdx.z) * 16) + 12))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 108))], placeholder_shared[(((((int)threadIdx.z) * 16) + 140))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 117))], placeholder_shared[(((((int)threadIdx.z) * 16) + 13))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 117))], placeholder_shared[(((((int)threadIdx.z) * 16) + 141))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 126))], placeholder_shared[(((((int)threadIdx.z) * 16) + 14))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 126))], placeholder_shared[(((((int)threadIdx.z) * 16) + 142))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 135))], placeholder_shared[(((((int)threadIdx.z) * 16) + 15))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 135))], placeholder_shared[(((((int)threadIdx.z) * 16) + 143))], compute[(1)]);
  }
  T_relu[(((((((int)blockIdx.z) * 64) + (((int)threadIdx.z) * 4)) + (((int)threadIdx.y) * 2)) + ((int)threadIdx.x)))] = max((compute[(0)] + placeholder2[(((((int)blockIdx.z) * 16) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 64) + (((int)threadIdx.z) * 4)) + (((int)threadIdx.y) * 2)) + ((int)threadIdx.x)) + 32))] = max((compute[(1)] + placeholder2[((((((int)blockIdx.z) * 16) + ((int)threadIdx.z)) + 8))]), 0.000000e+00f);
}

extern "C" __global__ void tvmgen_default_fused_nn_conv2d_add_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_add, float* __restrict__ placeholder2) {
  float compute[2];
  __shared__ float pad_temp_shared[128];
  __shared__ float placeholder_shared[2048];
  for (int xx_init = 0; xx_init < 2; ++xx_init) {
    compute[(xx_init)] = 0.000000e+00f;
  }
  for (int rc_outer = 0; rc_outer < 16; ++rc_outer) {
    __syncthreads();
    pad_temp_shared[(((((int)threadIdx.z) * 2) + ((int)threadIdx.y)))] = placeholder[((((rc_outer * 128) + (((int)threadIdx.z) * 2)) + ((int)threadIdx.y)))];
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 16; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      placeholder_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))] = placeholder1[((((((((int)blockIdx.z) * 32768) + (((int)threadIdx.z) * 512)) + (rc_outer * 32)) + (((int)threadIdx.y) * 16)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))];
    }
    __syncthreads();
    for (int rc_inner = 0; rc_inner < 32; ++rc_inner) {
      for (int xx = 0; xx < 2; ++xx) {
        compute[(xx)] = __ocml_fma_f32(pad_temp_shared[((((rc_inner * 4) + (((int)threadIdx.y) * 2)) + xx))], placeholder_shared[(((((int)threadIdx.z) * 32) + rc_inner))], compute[(xx)]);
      }
    }
  }
  for (int ax3_inner_inner_inner = 0; ax3_inner_inner_inner < 2; ++ax3_inner_inner_inner) {
    T_add[(((((((int)blockIdx.z) * 256) + (((int)threadIdx.z) * 4)) + (((int)threadIdx.y) * 2)) + ax3_inner_inner_inner))] = (compute[(ax3_inner_inner_inner)] + placeholder2[(((((((int)blockIdx.z) * 256) + (((int)threadIdx.z) * 4)) + (((int)threadIdx.y) * 2)) + ax3_inner_inner_inner))]);
  }
}

extern "C" __global__ void tvmgen_default_fused_nn_conv2d_add_nn_relu_8_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  float compute[1];
  __shared__ float pad_temp_shared[6272];
  __shared__ float placeholder_shared[2048];
  compute[(0)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 2; ++rc_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 25; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      if ((((((int)threadIdx.z) * 8) + (((int)threadIdx.y) * 2)) + (((((int)threadIdx.x) * 25) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 49)) < 128) {
        if ((((((int)threadIdx.z) * 56) + (((int)threadIdx.y) * 14)) + (((((int)threadIdx.x) * 25) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 7)) < 896) {
          if (((((((int)threadIdx.z) * 392) + (((int)threadIdx.y) * 98)) + (((int)threadIdx.x) * 25)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 6272) {
            if ((((((int)threadIdx.y) * 98) + (((int)threadIdx.x) * 25)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 392) {
              if (((((int)threadIdx.x) * 25) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 98) {
                pad_temp_shared[(((((((int)threadIdx.z) * 392) + (((int)threadIdx.y) * 98)) + (((int)threadIdx.x) * 25)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))] = placeholder[(((((((((rc_outer * 32768) + (((int)threadIdx.z) * 2048)) + (((int)threadIdx.y) * 512)) + ((((((int)threadIdx.x) * 25) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 49) * 256)) + (((int)blockIdx.y) * 128)) + (((((((int)threadIdx.x) * 25) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 49) / 7) * 16)) + (((int)blockIdx.x) * 8)) + (((((int)threadIdx.x) * 25) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 7)))];
              }
            }
          }
        }
      }
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 < 8; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) {
      placeholder_shared[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1))] = placeholder1[(((((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) + (rc_outer * 128)) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1))];
    }
    __syncthreads();
    for (int rc_inner = 0; rc_inner < 128; ++rc_inner) {
      compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((rc_inner * 49) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)))], placeholder_shared[(((((int)threadIdx.z) * 128) + rc_inner))], compute[(0)]);
    }
  }
  T_relu[(((((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 64)) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.y) * 8)) + (((int)blockIdx.x) * 4)) + ((int)threadIdx.x)))] = max((compute[(0)] + placeholder2[(((((int)blockIdx.z) * 16) + ((int)threadIdx.z)))]), 0.000000e+00f);
}

extern "C" __global__ void tvmgen_default_fused_nn_conv2d_add_nn_relu_11_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
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
  for (int rc_outer = 0; rc_outer < 4; ++rc_outer) {
    __syncthreads();
    pad_temp_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))] = placeholder[((((((rc_outer * 4096) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))];
    pad_temp_shared[(((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 1))] = placeholder[(((((((rc_outer * 4096) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 1))];
    pad_temp_shared[(((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 2))] = placeholder[(((((((rc_outer * 4096) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 2))];
    pad_temp_shared[(((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 3))] = placeholder[(((((((rc_outer * 4096) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 3))];
    placeholder_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))] = placeholder1[(((((((int)threadIdx.z) * 256) + (((int)threadIdx.y) * 64)) + (rc_outer * 16)) + (((int)threadIdx.x) * 4)))];
    placeholder_shared[(((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 1))] = placeholder1[((((((((int)threadIdx.z) * 256) + (((int)threadIdx.y) * 64)) + (rc_outer * 16)) + (((int)threadIdx.x) * 4)) + 1))];
    placeholder_shared[(((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 2))] = placeholder1[((((((((int)threadIdx.z) * 256) + (((int)threadIdx.y) * 64)) + (rc_outer * 16)) + (((int)threadIdx.x) * 4)) + 2))];
    placeholder_shared[(((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 3))] = placeholder1[((((((((int)threadIdx.z) * 256) + (((int)threadIdx.y) * 64)) + (rc_outer * 16)) + (((int)threadIdx.x) * 4)) + 3))];
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
  T_relu[(((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))] = max((compute[(0)] + placeholder2[(((int)threadIdx.z))]), 0.000000e+00f);
  T_relu[((((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 4096))] = max((compute[(4)] + placeholder2[((((int)threadIdx.z) + 16))]), 0.000000e+00f);
  T_relu[((((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 8192))] = max((compute[(8)] + placeholder2[((((int)threadIdx.z) + 32))]), 0.000000e+00f);
  T_relu[((((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 12288))] = max((compute[(12)] + placeholder2[((((int)threadIdx.z) + 48))]), 0.000000e+00f);
  T_relu[((((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 1))] = max((compute[(1)] + placeholder2[(((int)threadIdx.z))]), 0.000000e+00f);
  T_relu[((((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 4097))] = max((compute[(5)] + placeholder2[((((int)threadIdx.z) + 16))]), 0.000000e+00f);
  T_relu[((((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 8193))] = max((compute[(9)] + placeholder2[((((int)threadIdx.z) + 32))]), 0.000000e+00f);
  T_relu[((((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 12289))] = max((compute[(13)] + placeholder2[((((int)threadIdx.z) + 48))]), 0.000000e+00f);
  T_relu[((((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 2))] = max((compute[(2)] + placeholder2[(((int)threadIdx.z))]), 0.000000e+00f);
  T_relu[((((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 4098))] = max((compute[(6)] + placeholder2[((((int)threadIdx.z) + 16))]), 0.000000e+00f);
  T_relu[((((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 8194))] = max((compute[(10)] + placeholder2[((((int)threadIdx.z) + 32))]), 0.000000e+00f);
  T_relu[((((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 12290))] = max((compute[(14)] + placeholder2[((((int)threadIdx.z) + 48))]), 0.000000e+00f);
  T_relu[((((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 3))] = max((compute[(3)] + placeholder2[(((int)threadIdx.z))]), 0.000000e+00f);
  T_relu[((((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 4099))] = max((compute[(7)] + placeholder2[((((int)threadIdx.z) + 16))]), 0.000000e+00f);
  T_relu[((((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 8195))] = max((compute[(11)] + placeholder2[((((int)threadIdx.z) + 32))]), 0.000000e+00f);
  T_relu[((((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 12291))] = max((compute[(15)] + placeholder2[((((int)threadIdx.z) + 48))]), 0.000000e+00f);
}

extern "C" __global__ void tvmgen_default_fused_nn_conv2d_add_nn_relu_3_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  float compute[4];
  __shared__ float pad_temp_shared[288];
  __shared__ float placeholder_shared[2304];
  compute[(0)] = 0.000000e+00f;
  compute[(1)] = 0.000000e+00f;
  compute[(2)] = 0.000000e+00f;
  compute[(3)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 32; ++rc_outer) {
    __syncthreads();
    if (((((int)threadIdx.z) * 9) + (((int)threadIdx.x) * 3)) < 288) {
      if (((int)threadIdx.x) < 3) {
        pad_temp_shared[(((((int)threadIdx.z) * 9) + (((int)threadIdx.x) * 3)))] = (((((6 <= (((((int)threadIdx.z) * 9) + (((int)threadIdx.x) * 3)) % 36)) && ((((((int)threadIdx.z) * 9) + (((int)threadIdx.x) * 3)) % 36) < 30)) && (1 <= (((((int)threadIdx.z) * 9) + (((int)threadIdx.x) * 3)) % 6))) && ((((((int)threadIdx.z) * 9) + (((int)threadIdx.x) * 3)) % 6) < 5)) ? placeholder[((((((rc_outer * 128) + ((((((int)threadIdx.z) * 9) + (((int)threadIdx.x) * 3)) / 36) * 16)) + (((((((int)threadIdx.z) * 9) + (((int)threadIdx.x) * 3)) % 36) / 6) * 4)) + (((((int)threadIdx.z) * 9) + (((int)threadIdx.x) * 3)) % 6)) - 5))] : 0.000000e+00f);
      }
    }
    if (((((int)threadIdx.z) * 9) + (((int)threadIdx.x) * 3)) < 287) {
      if (((int)threadIdx.x) < 3) {
        pad_temp_shared[((((((int)threadIdx.z) * 9) + (((int)threadIdx.x) * 3)) + 1))] = (((((6 <= ((((((int)threadIdx.z) * 9) + (((int)threadIdx.x) * 3)) + 1) % 36)) && (((((((int)threadIdx.z) * 9) + (((int)threadIdx.x) * 3)) + 1) % 36) < 30)) && (1 <= ((((((int)threadIdx.z) * 9) + (((int)threadIdx.x) * 3)) + 1) % 6))) && (((((((int)threadIdx.z) * 9) + (((int)threadIdx.x) * 3)) + 1) % 6) < 5)) ? placeholder[((((((rc_outer * 128) + (((((((int)threadIdx.z) * 9) + (((int)threadIdx.x) * 3)) + 1) / 36) * 16)) + ((((((((int)threadIdx.z) * 9) + (((int)threadIdx.x) * 3)) + 1) % 36) / 6) * 4)) + ((((((int)threadIdx.z) * 9) + (((int)threadIdx.x) * 3)) + 1) % 6)) - 5))] : 0.000000e+00f);
      }
    }
    if (((((int)threadIdx.z) * 9) + (((int)threadIdx.x) * 3)) < 286) {
      if (((int)threadIdx.x) < 3) {
        pad_temp_shared[((((((int)threadIdx.z) * 9) + (((int)threadIdx.x) * 3)) + 2))] = (((((6 <= ((((((int)threadIdx.z) * 9) + (((int)threadIdx.x) * 3)) + 2) % 36)) && (((((((int)threadIdx.z) * 9) + (((int)threadIdx.x) * 3)) + 2) % 36) < 30)) && (1 <= ((((((int)threadIdx.z) * 9) + (((int)threadIdx.x) * 3)) + 2) % 6))) && (((((((int)threadIdx.z) * 9) + (((int)threadIdx.x) * 3)) + 2) % 6) < 5)) ? placeholder[((((((rc_outer * 128) + (((((((int)threadIdx.z) * 9) + (((int)threadIdx.x) * 3)) + 2) / 36) * 16)) + ((((((((int)threadIdx.z) * 9) + (((int)threadIdx.x) * 3)) + 2) % 36) / 6) * 4)) + ((((((int)threadIdx.z) * 9) + (((int)threadIdx.x) * 3)) + 2) % 6)) - 5))] : 0.000000e+00f);
      }
    }
    placeholder_shared[(((((int)threadIdx.z) * 72) + (((int)threadIdx.x) * 18)))] = placeholder1[(((((((int)blockIdx.z) * 73728) + (((int)threadIdx.z) * 2304)) + (rc_outer * 72)) + (((int)threadIdx.x) * 18)))];
    placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.x) * 18)) + 1))] = placeholder1[((((((((int)blockIdx.z) * 73728) + (((int)threadIdx.z) * 2304)) + (rc_outer * 72)) + (((int)threadIdx.x) * 18)) + 1))];
    placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.x) * 18)) + 2))] = placeholder1[((((((((int)blockIdx.z) * 73728) + (((int)threadIdx.z) * 2304)) + (rc_outer * 72)) + (((int)threadIdx.x) * 18)) + 2))];
    placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.x) * 18)) + 3))] = placeholder1[((((((((int)blockIdx.z) * 73728) + (((int)threadIdx.z) * 2304)) + (rc_outer * 72)) + (((int)threadIdx.x) * 18)) + 3))];
    placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.x) * 18)) + 4))] = placeholder1[((((((((int)blockIdx.z) * 73728) + (((int)threadIdx.z) * 2304)) + (rc_outer * 72)) + (((int)threadIdx.x) * 18)) + 4))];
    placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.x) * 18)) + 5))] = placeholder1[((((((((int)blockIdx.z) * 73728) + (((int)threadIdx.z) * 2304)) + (rc_outer * 72)) + (((int)threadIdx.x) * 18)) + 5))];
    placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.x) * 18)) + 6))] = placeholder1[((((((((int)blockIdx.z) * 73728) + (((int)threadIdx.z) * 2304)) + (rc_outer * 72)) + (((int)threadIdx.x) * 18)) + 6))];
    placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.x) * 18)) + 7))] = placeholder1[((((((((int)blockIdx.z) * 73728) + (((int)threadIdx.z) * 2304)) + (rc_outer * 72)) + (((int)threadIdx.x) * 18)) + 7))];
    placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.x) * 18)) + 8))] = placeholder1[((((((((int)blockIdx.z) * 73728) + (((int)threadIdx.z) * 2304)) + (rc_outer * 72)) + (((int)threadIdx.x) * 18)) + 8))];
    placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.x) * 18)) + 9))] = placeholder1[((((((((int)blockIdx.z) * 73728) + (((int)threadIdx.z) * 2304)) + (rc_outer * 72)) + (((int)threadIdx.x) * 18)) + 9))];
    placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.x) * 18)) + 10))] = placeholder1[((((((((int)blockIdx.z) * 73728) + (((int)threadIdx.z) * 2304)) + (rc_outer * 72)) + (((int)threadIdx.x) * 18)) + 10))];
    placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.x) * 18)) + 11))] = placeholder1[((((((((int)blockIdx.z) * 73728) + (((int)threadIdx.z) * 2304)) + (rc_outer * 72)) + (((int)threadIdx.x) * 18)) + 11))];
    placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.x) * 18)) + 12))] = placeholder1[((((((((int)blockIdx.z) * 73728) + (((int)threadIdx.z) * 2304)) + (rc_outer * 72)) + (((int)threadIdx.x) * 18)) + 12))];
    placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.x) * 18)) + 13))] = placeholder1[((((((((int)blockIdx.z) * 73728) + (((int)threadIdx.z) * 2304)) + (rc_outer * 72)) + (((int)threadIdx.x) * 18)) + 13))];
    placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.x) * 18)) + 14))] = placeholder1[((((((((int)blockIdx.z) * 73728) + (((int)threadIdx.z) * 2304)) + (rc_outer * 72)) + (((int)threadIdx.x) * 18)) + 14))];
    placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.x) * 18)) + 15))] = placeholder1[((((((((int)blockIdx.z) * 73728) + (((int)threadIdx.z) * 2304)) + (rc_outer * 72)) + (((int)threadIdx.x) * 18)) + 15))];
    placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.x) * 18)) + 16))] = placeholder1[((((((((int)blockIdx.z) * 73728) + (((int)threadIdx.z) * 2304)) + (rc_outer * 72)) + (((int)threadIdx.x) * 18)) + 16))];
    placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.x) * 18)) + 17))] = placeholder1[((((((((int)blockIdx.z) * 73728) + (((int)threadIdx.z) * 2304)) + (rc_outer * 72)) + (((int)threadIdx.x) * 18)) + 17))];
    __syncthreads();
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((int)threadIdx.x))], placeholder_shared[((((int)threadIdx.z) * 72))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 6))], placeholder_shared[((((int)threadIdx.z) * 72))], compute[(1)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 12))], placeholder_shared[((((int)threadIdx.z) * 72))], compute[(2)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 18))], placeholder_shared[((((int)threadIdx.z) * 72))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 1))], placeholder_shared[(((((int)threadIdx.z) * 72) + 1))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 7))], placeholder_shared[(((((int)threadIdx.z) * 72) + 1))], compute[(1)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 13))], placeholder_shared[(((((int)threadIdx.z) * 72) + 1))], compute[(2)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 19))], placeholder_shared[(((((int)threadIdx.z) * 72) + 1))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 2))], placeholder_shared[(((((int)threadIdx.z) * 72) + 2))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 8))], placeholder_shared[(((((int)threadIdx.z) * 72) + 2))], compute[(1)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 14))], placeholder_shared[(((((int)threadIdx.z) * 72) + 2))], compute[(2)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 20))], placeholder_shared[(((((int)threadIdx.z) * 72) + 2))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 6))], placeholder_shared[(((((int)threadIdx.z) * 72) + 3))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 12))], placeholder_shared[(((((int)threadIdx.z) * 72) + 3))], compute[(1)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 18))], placeholder_shared[(((((int)threadIdx.z) * 72) + 3))], compute[(2)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 24))], placeholder_shared[(((((int)threadIdx.z) * 72) + 3))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 7))], placeholder_shared[(((((int)threadIdx.z) * 72) + 4))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 13))], placeholder_shared[(((((int)threadIdx.z) * 72) + 4))], compute[(1)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 19))], placeholder_shared[(((((int)threadIdx.z) * 72) + 4))], compute[(2)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 25))], placeholder_shared[(((((int)threadIdx.z) * 72) + 4))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 8))], placeholder_shared[(((((int)threadIdx.z) * 72) + 5))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 14))], placeholder_shared[(((((int)threadIdx.z) * 72) + 5))], compute[(1)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 20))], placeholder_shared[(((((int)threadIdx.z) * 72) + 5))], compute[(2)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 26))], placeholder_shared[(((((int)threadIdx.z) * 72) + 5))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 12))], placeholder_shared[(((((int)threadIdx.z) * 72) + 6))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 18))], placeholder_shared[(((((int)threadIdx.z) * 72) + 6))], compute[(1)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 24))], placeholder_shared[(((((int)threadIdx.z) * 72) + 6))], compute[(2)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 30))], placeholder_shared[(((((int)threadIdx.z) * 72) + 6))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 13))], placeholder_shared[(((((int)threadIdx.z) * 72) + 7))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 19))], placeholder_shared[(((((int)threadIdx.z) * 72) + 7))], compute[(1)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 25))], placeholder_shared[(((((int)threadIdx.z) * 72) + 7))], compute[(2)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 31))], placeholder_shared[(((((int)threadIdx.z) * 72) + 7))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 14))], placeholder_shared[(((((int)threadIdx.z) * 72) + 8))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 20))], placeholder_shared[(((((int)threadIdx.z) * 72) + 8))], compute[(1)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 26))], placeholder_shared[(((((int)threadIdx.z) * 72) + 8))], compute[(2)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 32))], placeholder_shared[(((((int)threadIdx.z) * 72) + 8))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 36))], placeholder_shared[(((((int)threadIdx.z) * 72) + 9))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 42))], placeholder_shared[(((((int)threadIdx.z) * 72) + 9))], compute[(1)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 48))], placeholder_shared[(((((int)threadIdx.z) * 72) + 9))], compute[(2)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 54))], placeholder_shared[(((((int)threadIdx.z) * 72) + 9))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 37))], placeholder_shared[(((((int)threadIdx.z) * 72) + 10))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 43))], placeholder_shared[(((((int)threadIdx.z) * 72) + 10))], compute[(1)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 49))], placeholder_shared[(((((int)threadIdx.z) * 72) + 10))], compute[(2)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 55))], placeholder_shared[(((((int)threadIdx.z) * 72) + 10))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 38))], placeholder_shared[(((((int)threadIdx.z) * 72) + 11))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 44))], placeholder_shared[(((((int)threadIdx.z) * 72) + 11))], compute[(1)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 50))], placeholder_shared[(((((int)threadIdx.z) * 72) + 11))], compute[(2)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 56))], placeholder_shared[(((((int)threadIdx.z) * 72) + 11))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 42))], placeholder_shared[(((((int)threadIdx.z) * 72) + 12))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 48))], placeholder_shared[(((((int)threadIdx.z) * 72) + 12))], compute[(1)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 54))], placeholder_shared[(((((int)threadIdx.z) * 72) + 12))], compute[(2)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 60))], placeholder_shared[(((((int)threadIdx.z) * 72) + 12))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 43))], placeholder_shared[(((((int)threadIdx.z) * 72) + 13))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 49))], placeholder_shared[(((((int)threadIdx.z) * 72) + 13))], compute[(1)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 55))], placeholder_shared[(((((int)threadIdx.z) * 72) + 13))], compute[(2)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 61))], placeholder_shared[(((((int)threadIdx.z) * 72) + 13))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 44))], placeholder_shared[(((((int)threadIdx.z) * 72) + 14))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 50))], placeholder_shared[(((((int)threadIdx.z) * 72) + 14))], compute[(1)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 56))], placeholder_shared[(((((int)threadIdx.z) * 72) + 14))], compute[(2)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 62))], placeholder_shared[(((((int)threadIdx.z) * 72) + 14))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 48))], placeholder_shared[(((((int)threadIdx.z) * 72) + 15))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 54))], placeholder_shared[(((((int)threadIdx.z) * 72) + 15))], compute[(1)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 60))], placeholder_shared[(((((int)threadIdx.z) * 72) + 15))], compute[(2)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 66))], placeholder_shared[(((((int)threadIdx.z) * 72) + 15))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 49))], placeholder_shared[(((((int)threadIdx.z) * 72) + 16))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 55))], placeholder_shared[(((((int)threadIdx.z) * 72) + 16))], compute[(1)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 61))], placeholder_shared[(((((int)threadIdx.z) * 72) + 16))], compute[(2)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 67))], placeholder_shared[(((((int)threadIdx.z) * 72) + 16))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 50))], placeholder_shared[(((((int)threadIdx.z) * 72) + 17))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 56))], placeholder_shared[(((((int)threadIdx.z) * 72) + 17))], compute[(1)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 62))], placeholder_shared[(((((int)threadIdx.z) * 72) + 17))], compute[(2)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 68))], placeholder_shared[(((((int)threadIdx.z) * 72) + 17))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 72))], placeholder_shared[(((((int)threadIdx.z) * 72) + 18))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 78))], placeholder_shared[(((((int)threadIdx.z) * 72) + 18))], compute[(1)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 84))], placeholder_shared[(((((int)threadIdx.z) * 72) + 18))], compute[(2)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 90))], placeholder_shared[(((((int)threadIdx.z) * 72) + 18))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 73))], placeholder_shared[(((((int)threadIdx.z) * 72) + 19))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 79))], placeholder_shared[(((((int)threadIdx.z) * 72) + 19))], compute[(1)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 85))], placeholder_shared[(((((int)threadIdx.z) * 72) + 19))], compute[(2)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 91))], placeholder_shared[(((((int)threadIdx.z) * 72) + 19))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 74))], placeholder_shared[(((((int)threadIdx.z) * 72) + 20))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 80))], placeholder_shared[(((((int)threadIdx.z) * 72) + 20))], compute[(1)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 86))], placeholder_shared[(((((int)threadIdx.z) * 72) + 20))], compute[(2)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 92))], placeholder_shared[(((((int)threadIdx.z) * 72) + 20))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 78))], placeholder_shared[(((((int)threadIdx.z) * 72) + 21))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 84))], placeholder_shared[(((((int)threadIdx.z) * 72) + 21))], compute[(1)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 90))], placeholder_shared[(((((int)threadIdx.z) * 72) + 21))], compute[(2)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 96))], placeholder_shared[(((((int)threadIdx.z) * 72) + 21))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 79))], placeholder_shared[(((((int)threadIdx.z) * 72) + 22))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 85))], placeholder_shared[(((((int)threadIdx.z) * 72) + 22))], compute[(1)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 91))], placeholder_shared[(((((int)threadIdx.z) * 72) + 22))], compute[(2)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 97))], placeholder_shared[(((((int)threadIdx.z) * 72) + 22))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 80))], placeholder_shared[(((((int)threadIdx.z) * 72) + 23))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 86))], placeholder_shared[(((((int)threadIdx.z) * 72) + 23))], compute[(1)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 92))], placeholder_shared[(((((int)threadIdx.z) * 72) + 23))], compute[(2)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 98))], placeholder_shared[(((((int)threadIdx.z) * 72) + 23))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 84))], placeholder_shared[(((((int)threadIdx.z) * 72) + 24))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 90))], placeholder_shared[(((((int)threadIdx.z) * 72) + 24))], compute[(1)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 96))], placeholder_shared[(((((int)threadIdx.z) * 72) + 24))], compute[(2)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 102))], placeholder_shared[(((((int)threadIdx.z) * 72) + 24))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 85))], placeholder_shared[(((((int)threadIdx.z) * 72) + 25))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 91))], placeholder_shared[(((((int)threadIdx.z) * 72) + 25))], compute[(1)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 97))], placeholder_shared[(((((int)threadIdx.z) * 72) + 25))], compute[(2)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 103))], placeholder_shared[(((((int)threadIdx.z) * 72) + 25))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 86))], placeholder_shared[(((((int)threadIdx.z) * 72) + 26))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 92))], placeholder_shared[(((((int)threadIdx.z) * 72) + 26))], compute[(1)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 98))], placeholder_shared[(((((int)threadIdx.z) * 72) + 26))], compute[(2)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 104))], placeholder_shared[(((((int)threadIdx.z) * 72) + 26))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 108))], placeholder_shared[(((((int)threadIdx.z) * 72) + 27))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 114))], placeholder_shared[(((((int)threadIdx.z) * 72) + 27))], compute[(1)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 120))], placeholder_shared[(((((int)threadIdx.z) * 72) + 27))], compute[(2)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 126))], placeholder_shared[(((((int)threadIdx.z) * 72) + 27))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 109))], placeholder_shared[(((((int)threadIdx.z) * 72) + 28))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 115))], placeholder_shared[(((((int)threadIdx.z) * 72) + 28))], compute[(1)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 121))], placeholder_shared[(((((int)threadIdx.z) * 72) + 28))], compute[(2)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 127))], placeholder_shared[(((((int)threadIdx.z) * 72) + 28))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 110))], placeholder_shared[(((((int)threadIdx.z) * 72) + 29))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 116))], placeholder_shared[(((((int)threadIdx.z) * 72) + 29))], compute[(1)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 122))], placeholder_shared[(((((int)threadIdx.z) * 72) + 29))], compute[(2)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 128))], placeholder_shared[(((((int)threadIdx.z) * 72) + 29))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 114))], placeholder_shared[(((((int)threadIdx.z) * 72) + 30))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 120))], placeholder_shared[(((((int)threadIdx.z) * 72) + 30))], compute[(1)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 126))], placeholder_shared[(((((int)threadIdx.z) * 72) + 30))], compute[(2)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 132))], placeholder_shared[(((((int)threadIdx.z) * 72) + 30))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 115))], placeholder_shared[(((((int)threadIdx.z) * 72) + 31))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 121))], placeholder_shared[(((((int)threadIdx.z) * 72) + 31))], compute[(1)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 127))], placeholder_shared[(((((int)threadIdx.z) * 72) + 31))], compute[(2)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 133))], placeholder_shared[(((((int)threadIdx.z) * 72) + 31))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 116))], placeholder_shared[(((((int)threadIdx.z) * 72) + 32))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 122))], placeholder_shared[(((((int)threadIdx.z) * 72) + 32))], compute[(1)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 128))], placeholder_shared[(((((int)threadIdx.z) * 72) + 32))], compute[(2)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 134))], placeholder_shared[(((((int)threadIdx.z) * 72) + 32))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 120))], placeholder_shared[(((((int)threadIdx.z) * 72) + 33))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 126))], placeholder_shared[(((((int)threadIdx.z) * 72) + 33))], compute[(1)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 132))], placeholder_shared[(((((int)threadIdx.z) * 72) + 33))], compute[(2)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 138))], placeholder_shared[(((((int)threadIdx.z) * 72) + 33))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 121))], placeholder_shared[(((((int)threadIdx.z) * 72) + 34))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 127))], placeholder_shared[(((((int)threadIdx.z) * 72) + 34))], compute[(1)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 133))], placeholder_shared[(((((int)threadIdx.z) * 72) + 34))], compute[(2)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 139))], placeholder_shared[(((((int)threadIdx.z) * 72) + 34))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 122))], placeholder_shared[(((((int)threadIdx.z) * 72) + 35))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 128))], placeholder_shared[(((((int)threadIdx.z) * 72) + 35))], compute[(1)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 134))], placeholder_shared[(((((int)threadIdx.z) * 72) + 35))], compute[(2)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 140))], placeholder_shared[(((((int)threadIdx.z) * 72) + 35))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 144))], placeholder_shared[(((((int)threadIdx.z) * 72) + 36))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 150))], placeholder_shared[(((((int)threadIdx.z) * 72) + 36))], compute[(1)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 156))], placeholder_shared[(((((int)threadIdx.z) * 72) + 36))], compute[(2)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 162))], placeholder_shared[(((((int)threadIdx.z) * 72) + 36))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 145))], placeholder_shared[(((((int)threadIdx.z) * 72) + 37))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 151))], placeholder_shared[(((((int)threadIdx.z) * 72) + 37))], compute[(1)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 157))], placeholder_shared[(((((int)threadIdx.z) * 72) + 37))], compute[(2)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 163))], placeholder_shared[(((((int)threadIdx.z) * 72) + 37))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 146))], placeholder_shared[(((((int)threadIdx.z) * 72) + 38))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 152))], placeholder_shared[(((((int)threadIdx.z) * 72) + 38))], compute[(1)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 158))], placeholder_shared[(((((int)threadIdx.z) * 72) + 38))], compute[(2)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 164))], placeholder_shared[(((((int)threadIdx.z) * 72) + 38))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 150))], placeholder_shared[(((((int)threadIdx.z) * 72) + 39))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 156))], placeholder_shared[(((((int)threadIdx.z) * 72) + 39))], compute[(1)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 162))], placeholder_shared[(((((int)threadIdx.z) * 72) + 39))], compute[(2)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 168))], placeholder_shared[(((((int)threadIdx.z) * 72) + 39))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 151))], placeholder_shared[(((((int)threadIdx.z) * 72) + 40))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 157))], placeholder_shared[(((((int)threadIdx.z) * 72) + 40))], compute[(1)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 163))], placeholder_shared[(((((int)threadIdx.z) * 72) + 40))], compute[(2)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 169))], placeholder_shared[(((((int)threadIdx.z) * 72) + 40))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 152))], placeholder_shared[(((((int)threadIdx.z) * 72) + 41))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 158))], placeholder_shared[(((((int)threadIdx.z) * 72) + 41))], compute[(1)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 164))], placeholder_shared[(((((int)threadIdx.z) * 72) + 41))], compute[(2)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 170))], placeholder_shared[(((((int)threadIdx.z) * 72) + 41))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 156))], placeholder_shared[(((((int)threadIdx.z) * 72) + 42))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 162))], placeholder_shared[(((((int)threadIdx.z) * 72) + 42))], compute[(1)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 168))], placeholder_shared[(((((int)threadIdx.z) * 72) + 42))], compute[(2)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 174))], placeholder_shared[(((((int)threadIdx.z) * 72) + 42))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 157))], placeholder_shared[(((((int)threadIdx.z) * 72) + 43))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 163))], placeholder_shared[(((((int)threadIdx.z) * 72) + 43))], compute[(1)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 169))], placeholder_shared[(((((int)threadIdx.z) * 72) + 43))], compute[(2)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 175))], placeholder_shared[(((((int)threadIdx.z) * 72) + 43))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 158))], placeholder_shared[(((((int)threadIdx.z) * 72) + 44))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 164))], placeholder_shared[(((((int)threadIdx.z) * 72) + 44))], compute[(1)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 170))], placeholder_shared[(((((int)threadIdx.z) * 72) + 44))], compute[(2)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 176))], placeholder_shared[(((((int)threadIdx.z) * 72) + 44))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 180))], placeholder_shared[(((((int)threadIdx.z) * 72) + 45))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 186))], placeholder_shared[(((((int)threadIdx.z) * 72) + 45))], compute[(1)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 192))], placeholder_shared[(((((int)threadIdx.z) * 72) + 45))], compute[(2)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 198))], placeholder_shared[(((((int)threadIdx.z) * 72) + 45))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 181))], placeholder_shared[(((((int)threadIdx.z) * 72) + 46))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 187))], placeholder_shared[(((((int)threadIdx.z) * 72) + 46))], compute[(1)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 193))], placeholder_shared[(((((int)threadIdx.z) * 72) + 46))], compute[(2)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 199))], placeholder_shared[(((((int)threadIdx.z) * 72) + 46))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 182))], placeholder_shared[(((((int)threadIdx.z) * 72) + 47))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 188))], placeholder_shared[(((((int)threadIdx.z) * 72) + 47))], compute[(1)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 194))], placeholder_shared[(((((int)threadIdx.z) * 72) + 47))], compute[(2)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 200))], placeholder_shared[(((((int)threadIdx.z) * 72) + 47))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 186))], placeholder_shared[(((((int)threadIdx.z) * 72) + 48))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 192))], placeholder_shared[(((((int)threadIdx.z) * 72) + 48))], compute[(1)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 198))], placeholder_shared[(((((int)threadIdx.z) * 72) + 48))], compute[(2)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 204))], placeholder_shared[(((((int)threadIdx.z) * 72) + 48))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 187))], placeholder_shared[(((((int)threadIdx.z) * 72) + 49))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 193))], placeholder_shared[(((((int)threadIdx.z) * 72) + 49))], compute[(1)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 199))], placeholder_shared[(((((int)threadIdx.z) * 72) + 49))], compute[(2)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 205))], placeholder_shared[(((((int)threadIdx.z) * 72) + 49))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 188))], placeholder_shared[(((((int)threadIdx.z) * 72) + 50))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 194))], placeholder_shared[(((((int)threadIdx.z) * 72) + 50))], compute[(1)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 200))], placeholder_shared[(((((int)threadIdx.z) * 72) + 50))], compute[(2)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 206))], placeholder_shared[(((((int)threadIdx.z) * 72) + 50))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 192))], placeholder_shared[(((((int)threadIdx.z) * 72) + 51))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 198))], placeholder_shared[(((((int)threadIdx.z) * 72) + 51))], compute[(1)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 204))], placeholder_shared[(((((int)threadIdx.z) * 72) + 51))], compute[(2)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 210))], placeholder_shared[(((((int)threadIdx.z) * 72) + 51))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 193))], placeholder_shared[(((((int)threadIdx.z) * 72) + 52))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 199))], placeholder_shared[(((((int)threadIdx.z) * 72) + 52))], compute[(1)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 205))], placeholder_shared[(((((int)threadIdx.z) * 72) + 52))], compute[(2)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 211))], placeholder_shared[(((((int)threadIdx.z) * 72) + 52))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 194))], placeholder_shared[(((((int)threadIdx.z) * 72) + 53))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 200))], placeholder_shared[(((((int)threadIdx.z) * 72) + 53))], compute[(1)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 206))], placeholder_shared[(((((int)threadIdx.z) * 72) + 53))], compute[(2)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 212))], placeholder_shared[(((((int)threadIdx.z) * 72) + 53))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 216))], placeholder_shared[(((((int)threadIdx.z) * 72) + 54))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 222))], placeholder_shared[(((((int)threadIdx.z) * 72) + 54))], compute[(1)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 228))], placeholder_shared[(((((int)threadIdx.z) * 72) + 54))], compute[(2)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 234))], placeholder_shared[(((((int)threadIdx.z) * 72) + 54))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 217))], placeholder_shared[(((((int)threadIdx.z) * 72) + 55))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 223))], placeholder_shared[(((((int)threadIdx.z) * 72) + 55))], compute[(1)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 229))], placeholder_shared[(((((int)threadIdx.z) * 72) + 55))], compute[(2)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 235))], placeholder_shared[(((((int)threadIdx.z) * 72) + 55))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 218))], placeholder_shared[(((((int)threadIdx.z) * 72) + 56))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 224))], placeholder_shared[(((((int)threadIdx.z) * 72) + 56))], compute[(1)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 230))], placeholder_shared[(((((int)threadIdx.z) * 72) + 56))], compute[(2)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 236))], placeholder_shared[(((((int)threadIdx.z) * 72) + 56))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 222))], placeholder_shared[(((((int)threadIdx.z) * 72) + 57))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 228))], placeholder_shared[(((((int)threadIdx.z) * 72) + 57))], compute[(1)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 234))], placeholder_shared[(((((int)threadIdx.z) * 72) + 57))], compute[(2)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 240))], placeholder_shared[(((((int)threadIdx.z) * 72) + 57))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 223))], placeholder_shared[(((((int)threadIdx.z) * 72) + 58))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 229))], placeholder_shared[(((((int)threadIdx.z) * 72) + 58))], compute[(1)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 235))], placeholder_shared[(((((int)threadIdx.z) * 72) + 58))], compute[(2)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 241))], placeholder_shared[(((((int)threadIdx.z) * 72) + 58))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 224))], placeholder_shared[(((((int)threadIdx.z) * 72) + 59))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 230))], placeholder_shared[(((((int)threadIdx.z) * 72) + 59))], compute[(1)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 236))], placeholder_shared[(((((int)threadIdx.z) * 72) + 59))], compute[(2)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 242))], placeholder_shared[(((((int)threadIdx.z) * 72) + 59))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 228))], placeholder_shared[(((((int)threadIdx.z) * 72) + 60))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 234))], placeholder_shared[(((((int)threadIdx.z) * 72) + 60))], compute[(1)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 240))], placeholder_shared[(((((int)threadIdx.z) * 72) + 60))], compute[(2)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 246))], placeholder_shared[(((((int)threadIdx.z) * 72) + 60))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 229))], placeholder_shared[(((((int)threadIdx.z) * 72) + 61))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 235))], placeholder_shared[(((((int)threadIdx.z) * 72) + 61))], compute[(1)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 241))], placeholder_shared[(((((int)threadIdx.z) * 72) + 61))], compute[(2)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 247))], placeholder_shared[(((((int)threadIdx.z) * 72) + 61))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 230))], placeholder_shared[(((((int)threadIdx.z) * 72) + 62))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 236))], placeholder_shared[(((((int)threadIdx.z) * 72) + 62))], compute[(1)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 242))], placeholder_shared[(((((int)threadIdx.z) * 72) + 62))], compute[(2)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 248))], placeholder_shared[(((((int)threadIdx.z) * 72) + 62))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 252))], placeholder_shared[(((((int)threadIdx.z) * 72) + 63))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 258))], placeholder_shared[(((((int)threadIdx.z) * 72) + 63))], compute[(1)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 264))], placeholder_shared[(((((int)threadIdx.z) * 72) + 63))], compute[(2)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 270))], placeholder_shared[(((((int)threadIdx.z) * 72) + 63))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 253))], placeholder_shared[(((((int)threadIdx.z) * 72) + 64))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 259))], placeholder_shared[(((((int)threadIdx.z) * 72) + 64))], compute[(1)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 265))], placeholder_shared[(((((int)threadIdx.z) * 72) + 64))], compute[(2)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 271))], placeholder_shared[(((((int)threadIdx.z) * 72) + 64))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 254))], placeholder_shared[(((((int)threadIdx.z) * 72) + 65))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 260))], placeholder_shared[(((((int)threadIdx.z) * 72) + 65))], compute[(1)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 266))], placeholder_shared[(((((int)threadIdx.z) * 72) + 65))], compute[(2)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 272))], placeholder_shared[(((((int)threadIdx.z) * 72) + 65))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 258))], placeholder_shared[(((((int)threadIdx.z) * 72) + 66))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 264))], placeholder_shared[(((((int)threadIdx.z) * 72) + 66))], compute[(1)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 270))], placeholder_shared[(((((int)threadIdx.z) * 72) + 66))], compute[(2)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 276))], placeholder_shared[(((((int)threadIdx.z) * 72) + 66))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 259))], placeholder_shared[(((((int)threadIdx.z) * 72) + 67))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 265))], placeholder_shared[(((((int)threadIdx.z) * 72) + 67))], compute[(1)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 271))], placeholder_shared[(((((int)threadIdx.z) * 72) + 67))], compute[(2)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 277))], placeholder_shared[(((((int)threadIdx.z) * 72) + 67))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 260))], placeholder_shared[(((((int)threadIdx.z) * 72) + 68))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 266))], placeholder_shared[(((((int)threadIdx.z) * 72) + 68))], compute[(1)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 272))], placeholder_shared[(((((int)threadIdx.z) * 72) + 68))], compute[(2)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 278))], placeholder_shared[(((((int)threadIdx.z) * 72) + 68))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 264))], placeholder_shared[(((((int)threadIdx.z) * 72) + 69))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 270))], placeholder_shared[(((((int)threadIdx.z) * 72) + 69))], compute[(1)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 276))], placeholder_shared[(((((int)threadIdx.z) * 72) + 69))], compute[(2)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 282))], placeholder_shared[(((((int)threadIdx.z) * 72) + 69))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 265))], placeholder_shared[(((((int)threadIdx.z) * 72) + 70))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 271))], placeholder_shared[(((((int)threadIdx.z) * 72) + 70))], compute[(1)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 277))], placeholder_shared[(((((int)threadIdx.z) * 72) + 70))], compute[(2)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 283))], placeholder_shared[(((((int)threadIdx.z) * 72) + 70))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 266))], placeholder_shared[(((((int)threadIdx.z) * 72) + 71))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 272))], placeholder_shared[(((((int)threadIdx.z) * 72) + 71))], compute[(1)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 278))], placeholder_shared[(((((int)threadIdx.z) * 72) + 71))], compute[(2)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) + 284))], placeholder_shared[(((((int)threadIdx.z) * 72) + 71))], compute[(3)]);
  }
  T_relu[((((((int)blockIdx.z) * 512) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.x)))] = max((compute[(0)] + placeholder2[(((((int)blockIdx.z) * 32) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[(((((((int)blockIdx.z) * 512) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.x)) + 4))] = max((compute[(1)] + placeholder2[(((((int)blockIdx.z) * 32) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[(((((((int)blockIdx.z) * 512) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.x)) + 8))] = max((compute[(2)] + placeholder2[(((((int)blockIdx.z) * 32) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[(((((((int)blockIdx.z) * 512) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.x)) + 12))] = max((compute[(3)] + placeholder2[(((((int)blockIdx.z) * 32) + ((int)threadIdx.z)))]), 0.000000e+00f);
}

extern "C" __global__ void tvmgen_default_fused_nn_conv2d_add_nn_relu_10_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
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
  for (int rc_outer = 0; rc_outer < 16; ++rc_outer) {
    __syncthreads();
    pad_temp_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))] = placeholder[((((((rc_outer * 4096) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))];
    pad_temp_shared[(((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 1))] = placeholder[(((((((rc_outer * 4096) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 1))];
    pad_temp_shared[(((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 2))] = placeholder[(((((((rc_outer * 4096) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 2))];
    pad_temp_shared[(((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 3))] = placeholder[(((((((rc_outer * 4096) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 3))];
    placeholder_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))] = placeholder1[(((((((int)threadIdx.z) * 1024) + (((int)threadIdx.y) * 256)) + (rc_outer * 16)) + (((int)threadIdx.x) * 4)))];
    placeholder_shared[(((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 1))] = placeholder1[((((((((int)threadIdx.z) * 1024) + (((int)threadIdx.y) * 256)) + (rc_outer * 16)) + (((int)threadIdx.x) * 4)) + 1))];
    placeholder_shared[(((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 2))] = placeholder1[((((((((int)threadIdx.z) * 1024) + (((int)threadIdx.y) * 256)) + (rc_outer * 16)) + (((int)threadIdx.x) * 4)) + 2))];
    placeholder_shared[(((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 3))] = placeholder1[((((((((int)threadIdx.z) * 1024) + (((int)threadIdx.y) * 256)) + (rc_outer * 16)) + (((int)threadIdx.x) * 4)) + 3))];
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
  T_relu[(((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))] = max((compute[(0)] + placeholder2[(((int)threadIdx.z))]), 0.000000e+00f);
  T_relu[((((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 4096))] = max((compute[(4)] + placeholder2[((((int)threadIdx.z) + 16))]), 0.000000e+00f);
  T_relu[((((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 8192))] = max((compute[(8)] + placeholder2[((((int)threadIdx.z) + 32))]), 0.000000e+00f);
  T_relu[((((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 12288))] = max((compute[(12)] + placeholder2[((((int)threadIdx.z) + 48))]), 0.000000e+00f);
  T_relu[((((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 1))] = max((compute[(1)] + placeholder2[(((int)threadIdx.z))]), 0.000000e+00f);
  T_relu[((((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 4097))] = max((compute[(5)] + placeholder2[((((int)threadIdx.z) + 16))]), 0.000000e+00f);
  T_relu[((((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 8193))] = max((compute[(9)] + placeholder2[((((int)threadIdx.z) + 32))]), 0.000000e+00f);
  T_relu[((((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 12289))] = max((compute[(13)] + placeholder2[((((int)threadIdx.z) + 48))]), 0.000000e+00f);
  T_relu[((((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 2))] = max((compute[(2)] + placeholder2[(((int)threadIdx.z))]), 0.000000e+00f);
  T_relu[((((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 4098))] = max((compute[(6)] + placeholder2[((((int)threadIdx.z) + 16))]), 0.000000e+00f);
  T_relu[((((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 8194))] = max((compute[(10)] + placeholder2[((((int)threadIdx.z) + 32))]), 0.000000e+00f);
  T_relu[((((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 12290))] = max((compute[(14)] + placeholder2[((((int)threadIdx.z) + 48))]), 0.000000e+00f);
  T_relu[((((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 3))] = max((compute[(3)] + placeholder2[(((int)threadIdx.z))]), 0.000000e+00f);
  T_relu[((((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 4099))] = max((compute[(7)] + placeholder2[((((int)threadIdx.z) + 16))]), 0.000000e+00f);
  T_relu[((((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 8195))] = max((compute[(11)] + placeholder2[((((int)threadIdx.z) + 32))]), 0.000000e+00f);
  T_relu[((((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 12291))] = max((compute[(15)] + placeholder2[((((int)threadIdx.z) + 48))]), 0.000000e+00f);
}

extern "C" __global__ void tvmgen_default_fused_nn_conv2d_2_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ compute) {
  float compute_local[4];
  __shared__ float pad_temp_shared[784];
  __shared__ float placeholder_shared[512];
  compute_local[(0)] = 0.000000e+00f;
  compute_local[(2)] = 0.000000e+00f;
  compute_local[(1)] = 0.000000e+00f;
  compute_local[(3)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 32; ++rc_outer) {
    __syncthreads();
    pad_temp_shared[((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 13)) + (((int)threadIdx.x) * 7)))] = placeholder[((((((rc_outer * 1024) + (((int)threadIdx.z) * 64)) + (((((int)threadIdx.y) * 13) / 7) * 8)) + (((int)threadIdx.x) * 8)) + ((((int)threadIdx.y) * 13) % 7)))];
    pad_temp_shared[(((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 13)) + (((int)threadIdx.x) * 7)) + 1))] = placeholder[((((((rc_outer * 1024) + (((int)threadIdx.z) * 64)) + ((((((int)threadIdx.y) * 13) + 1) / 7) * 8)) + (((int)threadIdx.x) * 8)) + (((((int)threadIdx.y) * 13) + 1) % 7)))];
    pad_temp_shared[(((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 13)) + (((int)threadIdx.x) * 7)) + 2))] = placeholder[((((((rc_outer * 1024) + (((int)threadIdx.z) * 64)) + ((((((int)threadIdx.y) * 13) + 2) / 7) * 8)) + (((int)threadIdx.x) * 8)) + (((((int)threadIdx.y) * 13) + 2) % 7)))];
    if (((((((((int)threadIdx.y) * 13) + 3) / 7) + ((int)threadIdx.x)) / 7) + ((int)threadIdx.z)) < 16) {
      if ((((((int)threadIdx.z) * 7) + (((((int)threadIdx.y) * 13) + 3) / 7)) + ((int)threadIdx.x)) < 112) {
        if ((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 13)) + (((int)threadIdx.x) * 7)) < 781) {
          if (((((int)threadIdx.y) * 13) + (((int)threadIdx.x) * 7)) < 46) {
            pad_temp_shared[(((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 13)) + (((int)threadIdx.x) * 7)) + 3))] = placeholder[((((((rc_outer * 1024) + ((((((((int)threadIdx.y) * 13) + 3) / 7) + ((int)threadIdx.x)) / 7) * 64)) + (((int)threadIdx.z) * 64)) + ((((((((int)threadIdx.y) * 13) + 3) / 7) + ((int)threadIdx.x)) % 7) * 8)) + (((((int)threadIdx.y) * 13) + 3) % 7)))];
          }
        }
      }
    }
    if (((((((((int)threadIdx.y) * 13) + 4) / 7) + ((int)threadIdx.x)) / 7) + ((int)threadIdx.z)) < 16) {
      if ((((((int)threadIdx.z) * 7) + (((((int)threadIdx.y) * 13) + 4) / 7)) + ((int)threadIdx.x)) < 112) {
        if ((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 13)) + (((int)threadIdx.x) * 7)) < 780) {
          if (((((int)threadIdx.y) * 13) + (((int)threadIdx.x) * 7)) < 45) {
            pad_temp_shared[(((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 13)) + (((int)threadIdx.x) * 7)) + 4))] = placeholder[((((((rc_outer * 1024) + ((((((((int)threadIdx.y) * 13) + 4) / 7) + ((int)threadIdx.x)) / 7) * 64)) + (((int)threadIdx.z) * 64)) + ((((((((int)threadIdx.y) * 13) + 4) / 7) + ((int)threadIdx.x)) % 7) * 8)) + (((((int)threadIdx.y) * 13) + 4) % 7)))];
          }
        }
      }
    }
    if (((((((((int)threadIdx.y) * 13) + 5) / 7) + ((int)threadIdx.x)) / 7) + ((int)threadIdx.z)) < 16) {
      if ((((((int)threadIdx.z) * 7) + (((((int)threadIdx.y) * 13) + 5) / 7)) + ((int)threadIdx.x)) < 112) {
        if ((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 13)) + (((int)threadIdx.x) * 7)) < 779) {
          if (((((int)threadIdx.y) * 13) + (((int)threadIdx.x) * 7)) < 44) {
            pad_temp_shared[(((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 13)) + (((int)threadIdx.x) * 7)) + 5))] = placeholder[((((((rc_outer * 1024) + ((((((((int)threadIdx.y) * 13) + 5) / 7) + ((int)threadIdx.x)) / 7) * 64)) + (((int)threadIdx.z) * 64)) + ((((((((int)threadIdx.y) * 13) + 5) / 7) + ((int)threadIdx.x)) % 7) * 8)) + (((((int)threadIdx.y) * 13) + 5) % 7)))];
          }
        }
      }
    }
    if (((((((((int)threadIdx.y) * 13) + 6) / 7) + ((int)threadIdx.x)) / 7) + ((int)threadIdx.z)) < 16) {
      if ((((((int)threadIdx.z) * 7) + (((((int)threadIdx.y) * 13) + 6) / 7)) + ((int)threadIdx.x)) < 112) {
        if ((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 13)) + (((int)threadIdx.x) * 7)) < 778) {
          if (((((int)threadIdx.y) * 13) + (((int)threadIdx.x) * 7)) < 43) {
            if (((int)threadIdx.x) < 1) {
              pad_temp_shared[(((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 13)) + (((int)threadIdx.x) * 7)) + 6))] = placeholder[((((((rc_outer * 1024) + (((int)threadIdx.z) * 64)) + ((((((int)threadIdx.y) * 13) + 6) / 7) * 8)) + (((int)threadIdx.x) * 8)) + (((((int)threadIdx.y) * 13) + 6) % 7)))];
            }
          }
        }
      }
    }
    placeholder_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 4)))] = placeholder1[((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 1024)) + ((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 4)) >> 4) * 512)) + (rc_outer * 16)) + (((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 4)) & 15)))];
    placeholder_shared[(((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 4)) + 1))] = placeholder1[((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 1024)) + (((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 4)) + 1) >> 4) * 512)) + (rc_outer * 16)) + ((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 4)) + 1) & 15)))];
    placeholder_shared[(((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 4)) + 2))] = placeholder1[((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 1024)) + (((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 4)) + 2) >> 4) * 512)) + (rc_outer * 16)) + ((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 4)) + 2) & 15)))];
    placeholder_shared[(((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 4)) + 3))] = placeholder1[((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 1024)) + (((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 4)) + 3) >> 4) * 512)) + (rc_outer * 16)) + ((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 4)) + 3) & 15)))];
    __syncthreads();
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)))], placeholder_shared[((((int)threadIdx.z) * 16))], compute_local[(0)]);
    compute_local[(2)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)))], placeholder_shared[(((((int)threadIdx.z) * 16) + 256))], compute_local[(2)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 2))], placeholder_shared[((((int)threadIdx.z) * 16))], compute_local[(1)]);
    compute_local[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 2))], placeholder_shared[(((((int)threadIdx.z) * 16) + 256))], compute_local[(3)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 49))], placeholder_shared[(((((int)threadIdx.z) * 16) + 1))], compute_local[(0)]);
    compute_local[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 49))], placeholder_shared[(((((int)threadIdx.z) * 16) + 257))], compute_local[(2)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 51))], placeholder_shared[(((((int)threadIdx.z) * 16) + 1))], compute_local[(1)]);
    compute_local[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 51))], placeholder_shared[(((((int)threadIdx.z) * 16) + 257))], compute_local[(3)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 98))], placeholder_shared[(((((int)threadIdx.z) * 16) + 2))], compute_local[(0)]);
    compute_local[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 98))], placeholder_shared[(((((int)threadIdx.z) * 16) + 258))], compute_local[(2)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 100))], placeholder_shared[(((((int)threadIdx.z) * 16) + 2))], compute_local[(1)]);
    compute_local[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 100))], placeholder_shared[(((((int)threadIdx.z) * 16) + 258))], compute_local[(3)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 147))], placeholder_shared[(((((int)threadIdx.z) * 16) + 3))], compute_local[(0)]);
    compute_local[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 147))], placeholder_shared[(((((int)threadIdx.z) * 16) + 259))], compute_local[(2)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 149))], placeholder_shared[(((((int)threadIdx.z) * 16) + 3))], compute_local[(1)]);
    compute_local[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 149))], placeholder_shared[(((((int)threadIdx.z) * 16) + 259))], compute_local[(3)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 196))], placeholder_shared[(((((int)threadIdx.z) * 16) + 4))], compute_local[(0)]);
    compute_local[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 196))], placeholder_shared[(((((int)threadIdx.z) * 16) + 260))], compute_local[(2)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 198))], placeholder_shared[(((((int)threadIdx.z) * 16) + 4))], compute_local[(1)]);
    compute_local[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 198))], placeholder_shared[(((((int)threadIdx.z) * 16) + 260))], compute_local[(3)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 245))], placeholder_shared[(((((int)threadIdx.z) * 16) + 5))], compute_local[(0)]);
    compute_local[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 245))], placeholder_shared[(((((int)threadIdx.z) * 16) + 261))], compute_local[(2)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 247))], placeholder_shared[(((((int)threadIdx.z) * 16) + 5))], compute_local[(1)]);
    compute_local[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 247))], placeholder_shared[(((((int)threadIdx.z) * 16) + 261))], compute_local[(3)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 294))], placeholder_shared[(((((int)threadIdx.z) * 16) + 6))], compute_local[(0)]);
    compute_local[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 294))], placeholder_shared[(((((int)threadIdx.z) * 16) + 262))], compute_local[(2)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 296))], placeholder_shared[(((((int)threadIdx.z) * 16) + 6))], compute_local[(1)]);
    compute_local[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 296))], placeholder_shared[(((((int)threadIdx.z) * 16) + 262))], compute_local[(3)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 343))], placeholder_shared[(((((int)threadIdx.z) * 16) + 7))], compute_local[(0)]);
    compute_local[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 343))], placeholder_shared[(((((int)threadIdx.z) * 16) + 263))], compute_local[(2)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 345))], placeholder_shared[(((((int)threadIdx.z) * 16) + 7))], compute_local[(1)]);
    compute_local[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 345))], placeholder_shared[(((((int)threadIdx.z) * 16) + 263))], compute_local[(3)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 392))], placeholder_shared[(((((int)threadIdx.z) * 16) + 8))], compute_local[(0)]);
    compute_local[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 392))], placeholder_shared[(((((int)threadIdx.z) * 16) + 264))], compute_local[(2)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 394))], placeholder_shared[(((((int)threadIdx.z) * 16) + 8))], compute_local[(1)]);
    compute_local[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 394))], placeholder_shared[(((((int)threadIdx.z) * 16) + 264))], compute_local[(3)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 441))], placeholder_shared[(((((int)threadIdx.z) * 16) + 9))], compute_local[(0)]);
    compute_local[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 441))], placeholder_shared[(((((int)threadIdx.z) * 16) + 265))], compute_local[(2)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 443))], placeholder_shared[(((((int)threadIdx.z) * 16) + 9))], compute_local[(1)]);
    compute_local[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 443))], placeholder_shared[(((((int)threadIdx.z) * 16) + 265))], compute_local[(3)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 490))], placeholder_shared[(((((int)threadIdx.z) * 16) + 10))], compute_local[(0)]);
    compute_local[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 490))], placeholder_shared[(((((int)threadIdx.z) * 16) + 266))], compute_local[(2)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 492))], placeholder_shared[(((((int)threadIdx.z) * 16) + 10))], compute_local[(1)]);
    compute_local[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 492))], placeholder_shared[(((((int)threadIdx.z) * 16) + 266))], compute_local[(3)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 539))], placeholder_shared[(((((int)threadIdx.z) * 16) + 11))], compute_local[(0)]);
    compute_local[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 539))], placeholder_shared[(((((int)threadIdx.z) * 16) + 267))], compute_local[(2)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 541))], placeholder_shared[(((((int)threadIdx.z) * 16) + 11))], compute_local[(1)]);
    compute_local[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 541))], placeholder_shared[(((((int)threadIdx.z) * 16) + 267))], compute_local[(3)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 588))], placeholder_shared[(((((int)threadIdx.z) * 16) + 12))], compute_local[(0)]);
    compute_local[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 588))], placeholder_shared[(((((int)threadIdx.z) * 16) + 268))], compute_local[(2)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 590))], placeholder_shared[(((((int)threadIdx.z) * 16) + 12))], compute_local[(1)]);
    compute_local[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 590))], placeholder_shared[(((((int)threadIdx.z) * 16) + 268))], compute_local[(3)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 637))], placeholder_shared[(((((int)threadIdx.z) * 16) + 13))], compute_local[(0)]);
    compute_local[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 637))], placeholder_shared[(((((int)threadIdx.z) * 16) + 269))], compute_local[(2)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 639))], placeholder_shared[(((((int)threadIdx.z) * 16) + 13))], compute_local[(1)]);
    compute_local[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 639))], placeholder_shared[(((((int)threadIdx.z) * 16) + 269))], compute_local[(3)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 686))], placeholder_shared[(((((int)threadIdx.z) * 16) + 14))], compute_local[(0)]);
    compute_local[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 686))], placeholder_shared[(((((int)threadIdx.z) * 16) + 270))], compute_local[(2)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 688))], placeholder_shared[(((((int)threadIdx.z) * 16) + 14))], compute_local[(1)]);
    compute_local[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 688))], placeholder_shared[(((((int)threadIdx.z) * 16) + 270))], compute_local[(3)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 735))], placeholder_shared[(((((int)threadIdx.z) * 16) + 15))], compute_local[(0)]);
    compute_local[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 735))], placeholder_shared[(((((int)threadIdx.z) * 16) + 271))], compute_local[(2)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 737))], placeholder_shared[(((((int)threadIdx.z) * 16) + 15))], compute_local[(1)]);
    compute_local[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 4)) + 737))], placeholder_shared[(((((int)threadIdx.z) * 16) + 271))], compute_local[(3)]);
  }
  compute[(((((((int)blockIdx.z) * 512) + (((int)threadIdx.z) * 16)) + (((int)threadIdx.y) * 4)) + (((int)threadIdx.x) * 2)))] = compute_local[(0)];
  compute[((((((((int)blockIdx.z) * 512) + (((int)threadIdx.z) * 16)) + (((int)threadIdx.y) * 4)) + (((int)threadIdx.x) * 2)) + 256))] = compute_local[(2)];
  compute[((((((((int)blockIdx.z) * 512) + (((int)threadIdx.z) * 16)) + (((int)threadIdx.y) * 4)) + (((int)threadIdx.x) * 2)) + 1))] = compute_local[(1)];
  compute[((((((((int)blockIdx.z) * 512) + (((int)threadIdx.z) * 16)) + (((int)threadIdx.y) * 4)) + (((int)threadIdx.x) * 2)) + 257))] = compute_local[(3)];
}

extern "C" __global__ void tvmgen_default_fused_add_nn_relu_kernel0(float* __restrict__ T_relu, float* __restrict__ placeholder, float* __restrict__ placeholder1) {
  T_relu[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)))] = max((placeholder[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)))] + placeholder1[(((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 2)))]), 0.000000e+00f);
}

extern "C" __global__ void tvmgen_default_fused_nn_conv2d_add_3_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_add, float* __restrict__ placeholder2) {
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
  for (int rc_outer = 0; rc_outer < 4; ++rc_outer) {
    __syncthreads();
    pad_temp_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))] = placeholder[((((((rc_outer * 4096) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))];
    pad_temp_shared[(((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 1))] = placeholder[(((((((rc_outer * 4096) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 1))];
    pad_temp_shared[(((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 2))] = placeholder[(((((((rc_outer * 4096) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 2))];
    pad_temp_shared[(((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 3))] = placeholder[(((((((rc_outer * 4096) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 3))];
    placeholder_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))] = placeholder1[((((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 64)) + (rc_outer * 16)) + (((int)threadIdx.x) * 4)))];
    placeholder_shared[(((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 1))] = placeholder1[(((((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 64)) + (rc_outer * 16)) + (((int)threadIdx.x) * 4)) + 1))];
    placeholder_shared[(((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 2))] = placeholder1[(((((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 64)) + (rc_outer * 16)) + (((int)threadIdx.x) * 4)) + 2))];
    placeholder_shared[(((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 3))] = placeholder1[(((((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 64)) + (rc_outer * 16)) + (((int)threadIdx.x) * 4)) + 3))];
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
  T_add[((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))] = (compute[(0)] + placeholder2[((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))]);
  T_add[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 4096))] = (compute[(4)] + placeholder2[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 4096))]);
  T_add[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 8192))] = (compute[(8)] + placeholder2[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 8192))]);
  T_add[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 12288))] = (compute[(12)] + placeholder2[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 12288))]);
  T_add[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 1))] = (compute[(1)] + placeholder2[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 1))]);
  T_add[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 4097))] = (compute[(5)] + placeholder2[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 4097))]);
  T_add[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 8193))] = (compute[(9)] + placeholder2[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 8193))]);
  T_add[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 12289))] = (compute[(13)] + placeholder2[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 12289))]);
  T_add[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 2))] = (compute[(2)] + placeholder2[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 2))]);
  T_add[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 4098))] = (compute[(6)] + placeholder2[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 4098))]);
  T_add[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 8194))] = (compute[(10)] + placeholder2[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 8194))]);
  T_add[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 12290))] = (compute[(14)] + placeholder2[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 12290))]);
  T_add[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 3))] = (compute[(3)] + placeholder2[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 3))]);
  T_add[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 4099))] = (compute[(7)] + placeholder2[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 4099))]);
  T_add[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 8195))] = (compute[(11)] + placeholder2[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 8195))]);
  T_add[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 12291))] = (compute[(15)] + placeholder2[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 12291))]);
}

extern "C" __global__ void tvmgen_default_fused_nn_conv2d_add_2_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_add, float* __restrict__ placeholder2) {
  float compute[1];
  __shared__ float pad_temp_shared[2048];
  __shared__ float placeholder_shared[2048];
  compute[(0)] = 0.000000e+00f;
  pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 8)))] = placeholder[(((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 128)) + ((((int)threadIdx.x) >> 1) * 64)) + (((int)blockIdx.y) * 32)) + ((((int)threadIdx.x) & 1) * 16)) + (((int)blockIdx.x) * 4)))];
  pad_temp_shared[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 8)) + 1))] = placeholder[((((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 128)) + ((((int)threadIdx.x) >> 1) * 64)) + (((int)blockIdx.y) * 32)) + ((((int)threadIdx.x) & 1) * 16)) + (((int)blockIdx.x) * 4)) + 1))];
  pad_temp_shared[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 8)) + 2))] = placeholder[((((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 128)) + ((((int)threadIdx.x) >> 1) * 64)) + (((int)blockIdx.y) * 32)) + ((((int)threadIdx.x) & 1) * 16)) + (((int)blockIdx.x) * 4)) + 2))];
  pad_temp_shared[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 8)) + 3))] = placeholder[((((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 128)) + ((((int)threadIdx.x) >> 1) * 64)) + (((int)blockIdx.y) * 32)) + ((((int)threadIdx.x) & 1) * 16)) + (((int)blockIdx.x) * 4)) + 3))];
  pad_temp_shared[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 8)) + 4))] = placeholder[(((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 128)) + ((((((int)threadIdx.x) * 2) + 1) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.x) * 2) + 1) & 3) * 8)) + (((int)blockIdx.x) * 4)))];
  pad_temp_shared[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 8)) + 5))] = placeholder[((((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 128)) + ((((((int)threadIdx.x) * 2) + 1) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.x) * 2) + 1) & 3) * 8)) + (((int)blockIdx.x) * 4)) + 1))];
  pad_temp_shared[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 8)) + 6))] = placeholder[((((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 128)) + ((((((int)threadIdx.x) * 2) + 1) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.x) * 2) + 1) & 3) * 8)) + (((int)blockIdx.x) * 4)) + 2))];
  pad_temp_shared[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 8)) + 7))] = placeholder[((((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 128)) + ((((((int)threadIdx.x) * 2) + 1) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.x) * 2) + 1) & 3) * 8)) + (((int)blockIdx.x) * 4)) + 3))];
  placeholder_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 8)))] = placeholder1[(((((((int)blockIdx.z) * 2048) + (((int)threadIdx.z) * 128)) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 8)))];
  placeholder_shared[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 8)) + 1))] = placeholder1[((((((((int)blockIdx.z) * 2048) + (((int)threadIdx.z) * 128)) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 8)) + 1))];
  placeholder_shared[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 8)) + 2))] = placeholder1[((((((((int)blockIdx.z) * 2048) + (((int)threadIdx.z) * 128)) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 8)) + 2))];
  placeholder_shared[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 8)) + 3))] = placeholder1[((((((((int)blockIdx.z) * 2048) + (((int)threadIdx.z) * 128)) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 8)) + 3))];
  placeholder_shared[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 8)) + 4))] = placeholder1[((((((((int)blockIdx.z) * 2048) + (((int)threadIdx.z) * 128)) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 8)) + 4))];
  placeholder_shared[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 8)) + 5))] = placeholder1[((((((((int)blockIdx.z) * 2048) + (((int)threadIdx.z) * 128)) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 8)) + 5))];
  placeholder_shared[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 8)) + 6))] = placeholder1[((((((((int)blockIdx.z) * 2048) + (((int)threadIdx.z) * 128)) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 8)) + 6))];
  placeholder_shared[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 8)) + 7))] = placeholder1[((((((((int)blockIdx.z) * 2048) + (((int)threadIdx.z) * 128)) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 8)) + 7))];
  __syncthreads();
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)))], placeholder_shared[((((int)threadIdx.z) * 128))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 16))], placeholder_shared[(((((int)threadIdx.z) * 128) + 1))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 32))], placeholder_shared[(((((int)threadIdx.z) * 128) + 2))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 48))], placeholder_shared[(((((int)threadIdx.z) * 128) + 3))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 64))], placeholder_shared[(((((int)threadIdx.z) * 128) + 4))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 80))], placeholder_shared[(((((int)threadIdx.z) * 128) + 5))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 96))], placeholder_shared[(((((int)threadIdx.z) * 128) + 6))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 112))], placeholder_shared[(((((int)threadIdx.z) * 128) + 7))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 128))], placeholder_shared[(((((int)threadIdx.z) * 128) + 8))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 144))], placeholder_shared[(((((int)threadIdx.z) * 128) + 9))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 160))], placeholder_shared[(((((int)threadIdx.z) * 128) + 10))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 176))], placeholder_shared[(((((int)threadIdx.z) * 128) + 11))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 192))], placeholder_shared[(((((int)threadIdx.z) * 128) + 12))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 208))], placeholder_shared[(((((int)threadIdx.z) * 128) + 13))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 224))], placeholder_shared[(((((int)threadIdx.z) * 128) + 14))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 240))], placeholder_shared[(((((int)threadIdx.z) * 128) + 15))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 256))], placeholder_shared[(((((int)threadIdx.z) * 128) + 16))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 272))], placeholder_shared[(((((int)threadIdx.z) * 128) + 17))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 288))], placeholder_shared[(((((int)threadIdx.z) * 128) + 18))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 304))], placeholder_shared[(((((int)threadIdx.z) * 128) + 19))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 320))], placeholder_shared[(((((int)threadIdx.z) * 128) + 20))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 336))], placeholder_shared[(((((int)threadIdx.z) * 128) + 21))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 352))], placeholder_shared[(((((int)threadIdx.z) * 128) + 22))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 368))], placeholder_shared[(((((int)threadIdx.z) * 128) + 23))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 384))], placeholder_shared[(((((int)threadIdx.z) * 128) + 24))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 400))], placeholder_shared[(((((int)threadIdx.z) * 128) + 25))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 416))], placeholder_shared[(((((int)threadIdx.z) * 128) + 26))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 432))], placeholder_shared[(((((int)threadIdx.z) * 128) + 27))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 448))], placeholder_shared[(((((int)threadIdx.z) * 128) + 28))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 464))], placeholder_shared[(((((int)threadIdx.z) * 128) + 29))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 480))], placeholder_shared[(((((int)threadIdx.z) * 128) + 30))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 496))], placeholder_shared[(((((int)threadIdx.z) * 128) + 31))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 512))], placeholder_shared[(((((int)threadIdx.z) * 128) + 32))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 528))], placeholder_shared[(((((int)threadIdx.z) * 128) + 33))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 544))], placeholder_shared[(((((int)threadIdx.z) * 128) + 34))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 560))], placeholder_shared[(((((int)threadIdx.z) * 128) + 35))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 576))], placeholder_shared[(((((int)threadIdx.z) * 128) + 36))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 592))], placeholder_shared[(((((int)threadIdx.z) * 128) + 37))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 608))], placeholder_shared[(((((int)threadIdx.z) * 128) + 38))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 624))], placeholder_shared[(((((int)threadIdx.z) * 128) + 39))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 640))], placeholder_shared[(((((int)threadIdx.z) * 128) + 40))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 656))], placeholder_shared[(((((int)threadIdx.z) * 128) + 41))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 672))], placeholder_shared[(((((int)threadIdx.z) * 128) + 42))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 688))], placeholder_shared[(((((int)threadIdx.z) * 128) + 43))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 704))], placeholder_shared[(((((int)threadIdx.z) * 128) + 44))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 720))], placeholder_shared[(((((int)threadIdx.z) * 128) + 45))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 736))], placeholder_shared[(((((int)threadIdx.z) * 128) + 46))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 752))], placeholder_shared[(((((int)threadIdx.z) * 128) + 47))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 768))], placeholder_shared[(((((int)threadIdx.z) * 128) + 48))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 784))], placeholder_shared[(((((int)threadIdx.z) * 128) + 49))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 800))], placeholder_shared[(((((int)threadIdx.z) * 128) + 50))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 816))], placeholder_shared[(((((int)threadIdx.z) * 128) + 51))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 832))], placeholder_shared[(((((int)threadIdx.z) * 128) + 52))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 848))], placeholder_shared[(((((int)threadIdx.z) * 128) + 53))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 864))], placeholder_shared[(((((int)threadIdx.z) * 128) + 54))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 880))], placeholder_shared[(((((int)threadIdx.z) * 128) + 55))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 896))], placeholder_shared[(((((int)threadIdx.z) * 128) + 56))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 912))], placeholder_shared[(((((int)threadIdx.z) * 128) + 57))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 928))], placeholder_shared[(((((int)threadIdx.z) * 128) + 58))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 944))], placeholder_shared[(((((int)threadIdx.z) * 128) + 59))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 960))], placeholder_shared[(((((int)threadIdx.z) * 128) + 60))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 976))], placeholder_shared[(((((int)threadIdx.z) * 128) + 61))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 992))], placeholder_shared[(((((int)threadIdx.z) * 128) + 62))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1008))], placeholder_shared[(((((int)threadIdx.z) * 128) + 63))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1024))], placeholder_shared[(((((int)threadIdx.z) * 128) + 64))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1040))], placeholder_shared[(((((int)threadIdx.z) * 128) + 65))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1056))], placeholder_shared[(((((int)threadIdx.z) * 128) + 66))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1072))], placeholder_shared[(((((int)threadIdx.z) * 128) + 67))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1088))], placeholder_shared[(((((int)threadIdx.z) * 128) + 68))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1104))], placeholder_shared[(((((int)threadIdx.z) * 128) + 69))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1120))], placeholder_shared[(((((int)threadIdx.z) * 128) + 70))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1136))], placeholder_shared[(((((int)threadIdx.z) * 128) + 71))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1152))], placeholder_shared[(((((int)threadIdx.z) * 128) + 72))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1168))], placeholder_shared[(((((int)threadIdx.z) * 128) + 73))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1184))], placeholder_shared[(((((int)threadIdx.z) * 128) + 74))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1200))], placeholder_shared[(((((int)threadIdx.z) * 128) + 75))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1216))], placeholder_shared[(((((int)threadIdx.z) * 128) + 76))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1232))], placeholder_shared[(((((int)threadIdx.z) * 128) + 77))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1248))], placeholder_shared[(((((int)threadIdx.z) * 128) + 78))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1264))], placeholder_shared[(((((int)threadIdx.z) * 128) + 79))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1280))], placeholder_shared[(((((int)threadIdx.z) * 128) + 80))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1296))], placeholder_shared[(((((int)threadIdx.z) * 128) + 81))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1312))], placeholder_shared[(((((int)threadIdx.z) * 128) + 82))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1328))], placeholder_shared[(((((int)threadIdx.z) * 128) + 83))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1344))], placeholder_shared[(((((int)threadIdx.z) * 128) + 84))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1360))], placeholder_shared[(((((int)threadIdx.z) * 128) + 85))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1376))], placeholder_shared[(((((int)threadIdx.z) * 128) + 86))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1392))], placeholder_shared[(((((int)threadIdx.z) * 128) + 87))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1408))], placeholder_shared[(((((int)threadIdx.z) * 128) + 88))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1424))], placeholder_shared[(((((int)threadIdx.z) * 128) + 89))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1440))], placeholder_shared[(((((int)threadIdx.z) * 128) + 90))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1456))], placeholder_shared[(((((int)threadIdx.z) * 128) + 91))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1472))], placeholder_shared[(((((int)threadIdx.z) * 128) + 92))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1488))], placeholder_shared[(((((int)threadIdx.z) * 128) + 93))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1504))], placeholder_shared[(((((int)threadIdx.z) * 128) + 94))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1520))], placeholder_shared[(((((int)threadIdx.z) * 128) + 95))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1536))], placeholder_shared[(((((int)threadIdx.z) * 128) + 96))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1552))], placeholder_shared[(((((int)threadIdx.z) * 128) + 97))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1568))], placeholder_shared[(((((int)threadIdx.z) * 128) + 98))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1584))], placeholder_shared[(((((int)threadIdx.z) * 128) + 99))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1600))], placeholder_shared[(((((int)threadIdx.z) * 128) + 100))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1616))], placeholder_shared[(((((int)threadIdx.z) * 128) + 101))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1632))], placeholder_shared[(((((int)threadIdx.z) * 128) + 102))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1648))], placeholder_shared[(((((int)threadIdx.z) * 128) + 103))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1664))], placeholder_shared[(((((int)threadIdx.z) * 128) + 104))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1680))], placeholder_shared[(((((int)threadIdx.z) * 128) + 105))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1696))], placeholder_shared[(((((int)threadIdx.z) * 128) + 106))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1712))], placeholder_shared[(((((int)threadIdx.z) * 128) + 107))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1728))], placeholder_shared[(((((int)threadIdx.z) * 128) + 108))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1744))], placeholder_shared[(((((int)threadIdx.z) * 128) + 109))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1760))], placeholder_shared[(((((int)threadIdx.z) * 128) + 110))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1776))], placeholder_shared[(((((int)threadIdx.z) * 128) + 111))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1792))], placeholder_shared[(((((int)threadIdx.z) * 128) + 112))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1808))], placeholder_shared[(((((int)threadIdx.z) * 128) + 113))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1824))], placeholder_shared[(((((int)threadIdx.z) * 128) + 114))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1840))], placeholder_shared[(((((int)threadIdx.z) * 128) + 115))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1856))], placeholder_shared[(((((int)threadIdx.z) * 128) + 116))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1872))], placeholder_shared[(((((int)threadIdx.z) * 128) + 117))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1888))], placeholder_shared[(((((int)threadIdx.z) * 128) + 118))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1904))], placeholder_shared[(((((int)threadIdx.z) * 128) + 119))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1920))], placeholder_shared[(((((int)threadIdx.z) * 128) + 120))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1936))], placeholder_shared[(((((int)threadIdx.z) * 128) + 121))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1952))], placeholder_shared[(((((int)threadIdx.z) * 128) + 122))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1968))], placeholder_shared[(((((int)threadIdx.z) * 128) + 123))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1984))], placeholder_shared[(((((int)threadIdx.z) * 128) + 124))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2000))], placeholder_shared[(((((int)threadIdx.z) * 128) + 125))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2016))], placeholder_shared[(((((int)threadIdx.z) * 128) + 126))], compute[(0)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2032))], placeholder_shared[(((((int)threadIdx.z) * 128) + 127))], compute[(0)]);
  T_add[(((((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 64)) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.y) * 8)) + (((int)blockIdx.x) * 4)) + ((int)threadIdx.x)))] = (compute[(0)] + placeholder2[(((((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 64)) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.y) * 8)) + (((int)blockIdx.x) * 4)) + ((int)threadIdx.x)))]);
}

extern "C" __global__ void tvmgen_default_fused_nn_conv2d_add_multiply_add_nn_relu_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2, float* __restrict__ placeholder3, float* __restrict__ placeholder4) {
  float compute[2];
  __shared__ float pad_temp_shared[128];
  __shared__ float placeholder_shared[2048];
  for (int xx_init = 0; xx_init < 2; ++xx_init) {
    compute[(xx_init)] = 0.000000e+00f;
  }
  for (int rc_outer = 0; rc_outer < 16; ++rc_outer) {
    __syncthreads();
    pad_temp_shared[(((((int)threadIdx.z) * 2) + ((int)threadIdx.y)))] = placeholder[((((rc_outer * 128) + (((int)threadIdx.z) * 2)) + ((int)threadIdx.y)))];
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 16; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      placeholder_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))] = placeholder1[((((((((int)blockIdx.z) * 32768) + (((int)threadIdx.z) * 512)) + (rc_outer * 32)) + (((int)threadIdx.y) * 16)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))];
    }
    __syncthreads();
    for (int rc_inner = 0; rc_inner < 32; ++rc_inner) {
      for (int xx = 0; xx < 2; ++xx) {
        compute[(xx)] = __ocml_fma_f32(pad_temp_shared[((((rc_inner * 4) + (((int)threadIdx.y) * 2)) + xx))], placeholder_shared[(((((int)threadIdx.z) * 32) + rc_inner))], compute[(xx)]);
      }
    }
  }
  for (int ax3_inner_inner_inner = 0; ax3_inner_inner_inner < 2; ++ax3_inner_inner_inner) {
    T_relu[(((((((int)blockIdx.z) * 256) + (((int)threadIdx.z) * 4)) + (((int)threadIdx.y) * 2)) + ax3_inner_inner_inner))] = max(__ocml_fma_f32((compute[(ax3_inner_inner_inner)] + placeholder2[(((((((int)blockIdx.z) * 256) + (((int)threadIdx.z) * 4)) + (((int)threadIdx.y) * 2)) + ax3_inner_inner_inner))]), placeholder3[(((((int)blockIdx.z) * 64) + ((int)threadIdx.z)))], placeholder4[(((((int)blockIdx.z) * 64) + ((int)threadIdx.z)))]), 0.000000e+00f);
  }
}

extern "C" __global__ void tvmgen_default_fused_nn_conv2d_add_nn_relu_6_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  float compute[1];
  __shared__ float pad_temp_shared[192];
  __shared__ float placeholder_shared[2304];
  compute[(0)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 16; ++rc_outer) {
    __syncthreads();
    if (((((((int)threadIdx.y) * 3) + ((int)threadIdx.x)) / 6) + ((int)threadIdx.z)) < 32) {
      if ((((((int)threadIdx.z) * 6) + (((int)threadIdx.y) * 3)) + ((int)threadIdx.x)) < 192) {
        if (((((int)threadIdx.y) * 3) + ((int)threadIdx.x)) < 6) {
          if (((int)threadIdx.x) < 3) {
            pad_temp_shared[((((((int)threadIdx.z) * 6) + (((int)threadIdx.y) * 3)) + ((int)threadIdx.x)))] = (((((1 <= ((((int)blockIdx.y) * 2) + (((int)threadIdx.z) & 3))) && (((((int)blockIdx.y) * 2) + (((int)threadIdx.z) & 3)) < 9)) && (1 <= (((((int)blockIdx.x) * 4) + (((int)threadIdx.y) * 3)) + ((int)threadIdx.x)))) && ((((((int)blockIdx.x) * 4) + (((int)threadIdx.y) * 3)) + ((int)threadIdx.x)) < 9)) ? placeholder[(((((((((rc_outer * 512) + ((((int)threadIdx.z) >> 2) * 64)) + (((int)blockIdx.y) * 16)) + ((((int)threadIdx.z) & 3) * 8)) + (((int)blockIdx.x) * 4)) + (((int)threadIdx.y) * 3)) + ((int)threadIdx.x)) - 9))] : 0.000000e+00f);
          }
        }
      }
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 9; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      placeholder_shared[(((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 36)) + (((int)threadIdx.x) * 9)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))] = placeholder1[(((((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 1152)) + (rc_outer * 72)) + (((int)threadIdx.y) * 36)) + (((int)threadIdx.x) * 9)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))];
    }
    __syncthreads();
    for (int rc_inner = 0; rc_inner < 8; ++rc_inner) {
      for (int ry_inner = 0; ry_inner < 3; ++ry_inner) {
        for (int rx_inner = 0; rx_inner < 3; ++rx_inner) {
          compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((rc_inner * 24) + (((int)threadIdx.y) * 6)) + (ry_inner * 6)) + ((int)threadIdx.x)) + rx_inner))], placeholder_shared[(((((((int)threadIdx.z) * 72) + (rc_inner * 9)) + (ry_inner * 3)) + rx_inner))], compute[(0)]);
        }
      }
    }
  }
  T_relu[(((((((((int)blockIdx.z) * 2048) + (((int)threadIdx.z) * 64)) + (((int)blockIdx.y) * 16)) + (((int)threadIdx.y) * 8)) + (((int)blockIdx.x) * 4)) + ((int)threadIdx.x)))] = max((compute[(0)] + placeholder2[(((((int)blockIdx.z) * 32) + ((int)threadIdx.z)))]), 0.000000e+00f);
}

extern "C" __global__ void tvmgen_default_fused_nn_global_avg_pool2d_kernel0(float* __restrict__ placeholder, float* __restrict__ tensor) {
  float tensor1[1];
  tensor1[(0)] = 0.000000e+00f;
  for (int rv0 = 0; rv0 < 2; ++rv0) {
    for (int rv1 = 0; rv1 < 2; ++rv1) {
      if (((int)threadIdx.y) < 1) {
        tensor1[(0)] = (tensor1[(0)] + placeholder[((((((((int)threadIdx.y) * 8192) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) * 4)) + (rv0 * 2)) + rv1))]);
      }
    }
  }
  if (((int)threadIdx.y) < 1) {
    tensor[((((((int)threadIdx.y) * 2048) + (((int)blockIdx.x) * 8)) + ((int)threadIdx.x)))] = (tensor1[(0)] * 2.500000e-01f);
  }
}

extern "C" __global__ void tvmgen_default_fused_nn_batch_flatten_kernel0(float* __restrict__ tensor, float* __restrict__ placeholder) {
  tensor[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)))] = placeholder[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)))];
}

extern "C" __global__ void tvmgen_default_fused_nn_conv2d_add_1_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_add, float* __restrict__ placeholder2) {
  float compute[1];
  __shared__ float pad_temp_shared[64];
  __shared__ float placeholder_shared[1024];
  compute[(0)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 16; ++rc_outer) {
    __syncthreads();
    if ((((int)threadIdx.x) + ((int)threadIdx.z)) < 64) {
      if (((int)threadIdx.x) < 1) {
        pad_temp_shared[((((int)threadIdx.x) + ((int)threadIdx.z)))] = placeholder[(((((rc_outer * 256) + (((((int)threadIdx.x) + ((int)threadIdx.z)) >> 2) * 16)) + (((int)blockIdx.y) * 4)) + ((((int)threadIdx.x) + ((int)threadIdx.z)) & 3)))];
      }
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      placeholder_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))] = placeholder1[((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (rc_outer * 16)) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))];
    }
    __syncthreads();
    for (int rc_inner = 0; rc_inner < 16; ++rc_inner) {
      compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((rc_inner * 4) + ((int)threadIdx.x)))], placeholder_shared[(((((int)threadIdx.z) * 16) + rc_inner))], compute[(0)]);
    }
  }
  T_add[(((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 16)) + (((int)blockIdx.y) * 4)) + ((int)threadIdx.x)))] = (compute[(0)] + placeholder2[(((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 16)) + (((int)blockIdx.y) * 4)) + ((int)threadIdx.x)))]);
}

extern "C" __global__ void tvmgen_default_fused_nn_conv2d_add_nn_relu_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  float compute[2];
  __shared__ float pad_temp_shared[128];
  __shared__ float placeholder_shared[2304];
  compute[(0)] = 0.000000e+00f;
  compute[(1)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 64; ++rc_outer) {
    __syncthreads();
    pad_temp_shared[(((((int)threadIdx.z) * 4) + (((int)threadIdx.y) * 2)))] = ((((1 <= (((int)threadIdx.z) & 3)) && ((((int)threadIdx.z) & 3) < 3)) && (1 <= ((int)threadIdx.y))) ? placeholder[((((((rc_outer * 32) + ((((int)threadIdx.z) >> 2) * 4)) + (((int)threadIdx.y) * 2)) + ((((int)threadIdx.z) & 3) * 2)) - 3))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 4) + (((int)threadIdx.y) * 2)) + 1))] = ((((1 <= (((int)threadIdx.z) & 3)) && ((((int)threadIdx.z) & 3) < 3)) && (((int)threadIdx.y) < 1)) ? placeholder[((((((rc_outer * 32) + ((((int)threadIdx.z) >> 2) * 4)) + (((int)threadIdx.y) * 2)) + ((((int)threadIdx.z) & 3) * 2)) - 2))] : 0.000000e+00f);
    placeholder_shared[(((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 36)))] = placeholder1[(((((((int)blockIdx.z) * 147456) + (((int)threadIdx.z) * 4608)) + (rc_outer * 72)) + (((int)threadIdx.y) * 36)))];
    placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 36)) + 1))] = placeholder1[((((((((int)blockIdx.z) * 147456) + (((int)threadIdx.z) * 4608)) + (rc_outer * 72)) + (((int)threadIdx.y) * 36)) + 1))];
    placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 36)) + 2))] = placeholder1[((((((((int)blockIdx.z) * 147456) + (((int)threadIdx.z) * 4608)) + (rc_outer * 72)) + (((int)threadIdx.y) * 36)) + 2))];
    placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 36)) + 3))] = placeholder1[((((((((int)blockIdx.z) * 147456) + (((int)threadIdx.z) * 4608)) + (rc_outer * 72)) + (((int)threadIdx.y) * 36)) + 3))];
    placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 36)) + 4))] = placeholder1[((((((((int)blockIdx.z) * 147456) + (((int)threadIdx.z) * 4608)) + (rc_outer * 72)) + (((int)threadIdx.y) * 36)) + 4))];
    placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 36)) + 5))] = placeholder1[((((((((int)blockIdx.z) * 147456) + (((int)threadIdx.z) * 4608)) + (rc_outer * 72)) + (((int)threadIdx.y) * 36)) + 5))];
    placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 36)) + 6))] = placeholder1[((((((((int)blockIdx.z) * 147456) + (((int)threadIdx.z) * 4608)) + (rc_outer * 72)) + (((int)threadIdx.y) * 36)) + 6))];
    placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 36)) + 7))] = placeholder1[((((((((int)blockIdx.z) * 147456) + (((int)threadIdx.z) * 4608)) + (rc_outer * 72)) + (((int)threadIdx.y) * 36)) + 7))];
    placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 36)) + 8))] = placeholder1[((((((((int)blockIdx.z) * 147456) + (((int)threadIdx.z) * 4608)) + (rc_outer * 72)) + (((int)threadIdx.y) * 36)) + 8))];
    placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 36)) + 9))] = placeholder1[((((((((int)blockIdx.z) * 147456) + (((int)threadIdx.z) * 4608)) + (rc_outer * 72)) + (((int)threadIdx.y) * 36)) + 9))];
    placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 36)) + 10))] = placeholder1[((((((((int)blockIdx.z) * 147456) + (((int)threadIdx.z) * 4608)) + (rc_outer * 72)) + (((int)threadIdx.y) * 36)) + 10))];
    placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 36)) + 11))] = placeholder1[((((((((int)blockIdx.z) * 147456) + (((int)threadIdx.z) * 4608)) + (rc_outer * 72)) + (((int)threadIdx.y) * 36)) + 11))];
    placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 36)) + 12))] = placeholder1[((((((((int)blockIdx.z) * 147456) + (((int)threadIdx.z) * 4608)) + (rc_outer * 72)) + (((int)threadIdx.y) * 36)) + 12))];
    placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 36)) + 13))] = placeholder1[((((((((int)blockIdx.z) * 147456) + (((int)threadIdx.z) * 4608)) + (rc_outer * 72)) + (((int)threadIdx.y) * 36)) + 13))];
    placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 36)) + 14))] = placeholder1[((((((((int)blockIdx.z) * 147456) + (((int)threadIdx.z) * 4608)) + (rc_outer * 72)) + (((int)threadIdx.y) * 36)) + 14))];
    placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 36)) + 15))] = placeholder1[((((((((int)blockIdx.z) * 147456) + (((int)threadIdx.z) * 4608)) + (rc_outer * 72)) + (((int)threadIdx.y) * 36)) + 15))];
    placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 36)) + 16))] = placeholder1[((((((((int)blockIdx.z) * 147456) + (((int)threadIdx.z) * 4608)) + (rc_outer * 72)) + (((int)threadIdx.y) * 36)) + 16))];
    placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 36)) + 17))] = placeholder1[((((((((int)blockIdx.z) * 147456) + (((int)threadIdx.z) * 4608)) + (rc_outer * 72)) + (((int)threadIdx.y) * 36)) + 17))];
    placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 36)) + 18))] = placeholder1[((((((((int)blockIdx.z) * 147456) + (((int)threadIdx.z) * 4608)) + (rc_outer * 72)) + (((int)threadIdx.y) * 36)) + 18))];
    placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 36)) + 19))] = placeholder1[((((((((int)blockIdx.z) * 147456) + (((int)threadIdx.z) * 4608)) + (rc_outer * 72)) + (((int)threadIdx.y) * 36)) + 19))];
    placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 36)) + 20))] = placeholder1[((((((((int)blockIdx.z) * 147456) + (((int)threadIdx.z) * 4608)) + (rc_outer * 72)) + (((int)threadIdx.y) * 36)) + 20))];
    placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 36)) + 21))] = placeholder1[((((((((int)blockIdx.z) * 147456) + (((int)threadIdx.z) * 4608)) + (rc_outer * 72)) + (((int)threadIdx.y) * 36)) + 21))];
    placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 36)) + 22))] = placeholder1[((((((((int)blockIdx.z) * 147456) + (((int)threadIdx.z) * 4608)) + (rc_outer * 72)) + (((int)threadIdx.y) * 36)) + 22))];
    placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 36)) + 23))] = placeholder1[((((((((int)blockIdx.z) * 147456) + (((int)threadIdx.z) * 4608)) + (rc_outer * 72)) + (((int)threadIdx.y) * 36)) + 23))];
    placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 36)) + 24))] = placeholder1[((((((((int)blockIdx.z) * 147456) + (((int)threadIdx.z) * 4608)) + (rc_outer * 72)) + (((int)threadIdx.y) * 36)) + 24))];
    placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 36)) + 25))] = placeholder1[((((((((int)blockIdx.z) * 147456) + (((int)threadIdx.z) * 4608)) + (rc_outer * 72)) + (((int)threadIdx.y) * 36)) + 25))];
    placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 36)) + 26))] = placeholder1[((((((((int)blockIdx.z) * 147456) + (((int)threadIdx.z) * 4608)) + (rc_outer * 72)) + (((int)threadIdx.y) * 36)) + 26))];
    placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 36)) + 27))] = placeholder1[((((((((int)blockIdx.z) * 147456) + (((int)threadIdx.z) * 4608)) + (rc_outer * 72)) + (((int)threadIdx.y) * 36)) + 27))];
    placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 36)) + 28))] = placeholder1[((((((((int)blockIdx.z) * 147456) + (((int)threadIdx.z) * 4608)) + (rc_outer * 72)) + (((int)threadIdx.y) * 36)) + 28))];
    placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 36)) + 29))] = placeholder1[((((((((int)blockIdx.z) * 147456) + (((int)threadIdx.z) * 4608)) + (rc_outer * 72)) + (((int)threadIdx.y) * 36)) + 29))];
    placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 36)) + 30))] = placeholder1[((((((((int)blockIdx.z) * 147456) + (((int)threadIdx.z) * 4608)) + (rc_outer * 72)) + (((int)threadIdx.y) * 36)) + 30))];
    placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 36)) + 31))] = placeholder1[((((((((int)blockIdx.z) * 147456) + (((int)threadIdx.z) * 4608)) + (rc_outer * 72)) + (((int)threadIdx.y) * 36)) + 31))];
    placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 36)) + 32))] = placeholder1[((((((((int)blockIdx.z) * 147456) + (((int)threadIdx.z) * 4608)) + (rc_outer * 72)) + (((int)threadIdx.y) * 36)) + 32))];
    placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 36)) + 33))] = placeholder1[((((((((int)blockIdx.z) * 147456) + (((int)threadIdx.z) * 4608)) + (rc_outer * 72)) + (((int)threadIdx.y) * 36)) + 33))];
    placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 36)) + 34))] = placeholder1[((((((((int)blockIdx.z) * 147456) + (((int)threadIdx.z) * 4608)) + (rc_outer * 72)) + (((int)threadIdx.y) * 36)) + 34))];
    placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 36)) + 35))] = placeholder1[((((((((int)blockIdx.z) * 147456) + (((int)threadIdx.z) * 4608)) + (rc_outer * 72)) + (((int)threadIdx.y) * 36)) + 35))];
    __syncthreads();
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.y) * 4))], placeholder_shared[((((int)threadIdx.z) * 72))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 1))], placeholder_shared[((((int)threadIdx.z) * 72))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 1))], placeholder_shared[(((((int)threadIdx.z) * 72) + 1))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 2))], placeholder_shared[(((((int)threadIdx.z) * 72) + 1))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 2))], placeholder_shared[(((((int)threadIdx.z) * 72) + 2))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 3))], placeholder_shared[(((((int)threadIdx.z) * 72) + 2))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 4))], placeholder_shared[(((((int)threadIdx.z) * 72) + 3))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 5))], placeholder_shared[(((((int)threadIdx.z) * 72) + 3))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 5))], placeholder_shared[(((((int)threadIdx.z) * 72) + 4))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 6))], placeholder_shared[(((((int)threadIdx.z) * 72) + 4))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 6))], placeholder_shared[(((((int)threadIdx.z) * 72) + 5))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 7))], placeholder_shared[(((((int)threadIdx.z) * 72) + 5))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 8))], placeholder_shared[(((((int)threadIdx.z) * 72) + 6))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 9))], placeholder_shared[(((((int)threadIdx.z) * 72) + 6))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 9))], placeholder_shared[(((((int)threadIdx.z) * 72) + 7))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 10))], placeholder_shared[(((((int)threadIdx.z) * 72) + 7))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 10))], placeholder_shared[(((((int)threadIdx.z) * 72) + 8))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 11))], placeholder_shared[(((((int)threadIdx.z) * 72) + 8))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 16))], placeholder_shared[(((((int)threadIdx.z) * 72) + 9))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 17))], placeholder_shared[(((((int)threadIdx.z) * 72) + 9))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 17))], placeholder_shared[(((((int)threadIdx.z) * 72) + 10))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 18))], placeholder_shared[(((((int)threadIdx.z) * 72) + 10))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 18))], placeholder_shared[(((((int)threadIdx.z) * 72) + 11))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 19))], placeholder_shared[(((((int)threadIdx.z) * 72) + 11))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 20))], placeholder_shared[(((((int)threadIdx.z) * 72) + 12))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 21))], placeholder_shared[(((((int)threadIdx.z) * 72) + 12))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 21))], placeholder_shared[(((((int)threadIdx.z) * 72) + 13))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 22))], placeholder_shared[(((((int)threadIdx.z) * 72) + 13))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 22))], placeholder_shared[(((((int)threadIdx.z) * 72) + 14))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 23))], placeholder_shared[(((((int)threadIdx.z) * 72) + 14))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 24))], placeholder_shared[(((((int)threadIdx.z) * 72) + 15))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 25))], placeholder_shared[(((((int)threadIdx.z) * 72) + 15))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 25))], placeholder_shared[(((((int)threadIdx.z) * 72) + 16))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 26))], placeholder_shared[(((((int)threadIdx.z) * 72) + 16))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 26))], placeholder_shared[(((((int)threadIdx.z) * 72) + 17))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 27))], placeholder_shared[(((((int)threadIdx.z) * 72) + 17))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 32))], placeholder_shared[(((((int)threadIdx.z) * 72) + 18))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 33))], placeholder_shared[(((((int)threadIdx.z) * 72) + 18))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 33))], placeholder_shared[(((((int)threadIdx.z) * 72) + 19))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 34))], placeholder_shared[(((((int)threadIdx.z) * 72) + 19))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 34))], placeholder_shared[(((((int)threadIdx.z) * 72) + 20))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 35))], placeholder_shared[(((((int)threadIdx.z) * 72) + 20))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 36))], placeholder_shared[(((((int)threadIdx.z) * 72) + 21))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 37))], placeholder_shared[(((((int)threadIdx.z) * 72) + 21))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 37))], placeholder_shared[(((((int)threadIdx.z) * 72) + 22))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 38))], placeholder_shared[(((((int)threadIdx.z) * 72) + 22))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 38))], placeholder_shared[(((((int)threadIdx.z) * 72) + 23))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 39))], placeholder_shared[(((((int)threadIdx.z) * 72) + 23))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 40))], placeholder_shared[(((((int)threadIdx.z) * 72) + 24))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 41))], placeholder_shared[(((((int)threadIdx.z) * 72) + 24))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 41))], placeholder_shared[(((((int)threadIdx.z) * 72) + 25))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 42))], placeholder_shared[(((((int)threadIdx.z) * 72) + 25))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 42))], placeholder_shared[(((((int)threadIdx.z) * 72) + 26))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 43))], placeholder_shared[(((((int)threadIdx.z) * 72) + 26))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 48))], placeholder_shared[(((((int)threadIdx.z) * 72) + 27))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 49))], placeholder_shared[(((((int)threadIdx.z) * 72) + 27))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 49))], placeholder_shared[(((((int)threadIdx.z) * 72) + 28))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 50))], placeholder_shared[(((((int)threadIdx.z) * 72) + 28))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 50))], placeholder_shared[(((((int)threadIdx.z) * 72) + 29))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 51))], placeholder_shared[(((((int)threadIdx.z) * 72) + 29))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 52))], placeholder_shared[(((((int)threadIdx.z) * 72) + 30))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 53))], placeholder_shared[(((((int)threadIdx.z) * 72) + 30))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 53))], placeholder_shared[(((((int)threadIdx.z) * 72) + 31))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 54))], placeholder_shared[(((((int)threadIdx.z) * 72) + 31))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 54))], placeholder_shared[(((((int)threadIdx.z) * 72) + 32))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 55))], placeholder_shared[(((((int)threadIdx.z) * 72) + 32))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 56))], placeholder_shared[(((((int)threadIdx.z) * 72) + 33))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 57))], placeholder_shared[(((((int)threadIdx.z) * 72) + 33))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 57))], placeholder_shared[(((((int)threadIdx.z) * 72) + 34))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 58))], placeholder_shared[(((((int)threadIdx.z) * 72) + 34))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 58))], placeholder_shared[(((((int)threadIdx.z) * 72) + 35))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 59))], placeholder_shared[(((((int)threadIdx.z) * 72) + 35))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 64))], placeholder_shared[(((((int)threadIdx.z) * 72) + 36))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 65))], placeholder_shared[(((((int)threadIdx.z) * 72) + 36))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 65))], placeholder_shared[(((((int)threadIdx.z) * 72) + 37))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 66))], placeholder_shared[(((((int)threadIdx.z) * 72) + 37))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 66))], placeholder_shared[(((((int)threadIdx.z) * 72) + 38))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 67))], placeholder_shared[(((((int)threadIdx.z) * 72) + 38))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 68))], placeholder_shared[(((((int)threadIdx.z) * 72) + 39))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 69))], placeholder_shared[(((((int)threadIdx.z) * 72) + 39))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 69))], placeholder_shared[(((((int)threadIdx.z) * 72) + 40))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 70))], placeholder_shared[(((((int)threadIdx.z) * 72) + 40))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 70))], placeholder_shared[(((((int)threadIdx.z) * 72) + 41))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 71))], placeholder_shared[(((((int)threadIdx.z) * 72) + 41))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 72))], placeholder_shared[(((((int)threadIdx.z) * 72) + 42))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 73))], placeholder_shared[(((((int)threadIdx.z) * 72) + 42))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 73))], placeholder_shared[(((((int)threadIdx.z) * 72) + 43))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 74))], placeholder_shared[(((((int)threadIdx.z) * 72) + 43))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 74))], placeholder_shared[(((((int)threadIdx.z) * 72) + 44))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 75))], placeholder_shared[(((((int)threadIdx.z) * 72) + 44))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 80))], placeholder_shared[(((((int)threadIdx.z) * 72) + 45))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 81))], placeholder_shared[(((((int)threadIdx.z) * 72) + 45))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 81))], placeholder_shared[(((((int)threadIdx.z) * 72) + 46))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 82))], placeholder_shared[(((((int)threadIdx.z) * 72) + 46))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 82))], placeholder_shared[(((((int)threadIdx.z) * 72) + 47))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 83))], placeholder_shared[(((((int)threadIdx.z) * 72) + 47))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 84))], placeholder_shared[(((((int)threadIdx.z) * 72) + 48))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 85))], placeholder_shared[(((((int)threadIdx.z) * 72) + 48))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 85))], placeholder_shared[(((((int)threadIdx.z) * 72) + 49))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 86))], placeholder_shared[(((((int)threadIdx.z) * 72) + 49))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 86))], placeholder_shared[(((((int)threadIdx.z) * 72) + 50))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 87))], placeholder_shared[(((((int)threadIdx.z) * 72) + 50))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 88))], placeholder_shared[(((((int)threadIdx.z) * 72) + 51))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 89))], placeholder_shared[(((((int)threadIdx.z) * 72) + 51))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 89))], placeholder_shared[(((((int)threadIdx.z) * 72) + 52))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 90))], placeholder_shared[(((((int)threadIdx.z) * 72) + 52))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 90))], placeholder_shared[(((((int)threadIdx.z) * 72) + 53))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 91))], placeholder_shared[(((((int)threadIdx.z) * 72) + 53))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 96))], placeholder_shared[(((((int)threadIdx.z) * 72) + 54))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 97))], placeholder_shared[(((((int)threadIdx.z) * 72) + 54))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 97))], placeholder_shared[(((((int)threadIdx.z) * 72) + 55))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 98))], placeholder_shared[(((((int)threadIdx.z) * 72) + 55))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 98))], placeholder_shared[(((((int)threadIdx.z) * 72) + 56))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 99))], placeholder_shared[(((((int)threadIdx.z) * 72) + 56))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 100))], placeholder_shared[(((((int)threadIdx.z) * 72) + 57))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 101))], placeholder_shared[(((((int)threadIdx.z) * 72) + 57))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 101))], placeholder_shared[(((((int)threadIdx.z) * 72) + 58))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 102))], placeholder_shared[(((((int)threadIdx.z) * 72) + 58))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 102))], placeholder_shared[(((((int)threadIdx.z) * 72) + 59))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 103))], placeholder_shared[(((((int)threadIdx.z) * 72) + 59))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 104))], placeholder_shared[(((((int)threadIdx.z) * 72) + 60))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 105))], placeholder_shared[(((((int)threadIdx.z) * 72) + 60))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 105))], placeholder_shared[(((((int)threadIdx.z) * 72) + 61))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 106))], placeholder_shared[(((((int)threadIdx.z) * 72) + 61))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 106))], placeholder_shared[(((((int)threadIdx.z) * 72) + 62))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 107))], placeholder_shared[(((((int)threadIdx.z) * 72) + 62))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 112))], placeholder_shared[(((((int)threadIdx.z) * 72) + 63))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 113))], placeholder_shared[(((((int)threadIdx.z) * 72) + 63))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 113))], placeholder_shared[(((((int)threadIdx.z) * 72) + 64))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 114))], placeholder_shared[(((((int)threadIdx.z) * 72) + 64))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 114))], placeholder_shared[(((((int)threadIdx.z) * 72) + 65))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 115))], placeholder_shared[(((((int)threadIdx.z) * 72) + 65))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 116))], placeholder_shared[(((((int)threadIdx.z) * 72) + 66))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 117))], placeholder_shared[(((((int)threadIdx.z) * 72) + 66))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 117))], placeholder_shared[(((((int)threadIdx.z) * 72) + 67))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 118))], placeholder_shared[(((((int)threadIdx.z) * 72) + 67))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 118))], placeholder_shared[(((((int)threadIdx.z) * 72) + 68))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 119))], placeholder_shared[(((((int)threadIdx.z) * 72) + 68))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 120))], placeholder_shared[(((((int)threadIdx.z) * 72) + 69))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 121))], placeholder_shared[(((((int)threadIdx.z) * 72) + 69))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 121))], placeholder_shared[(((((int)threadIdx.z) * 72) + 70))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 122))], placeholder_shared[(((((int)threadIdx.z) * 72) + 70))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 122))], placeholder_shared[(((((int)threadIdx.z) * 72) + 71))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 4) + 123))], placeholder_shared[(((((int)threadIdx.z) * 72) + 71))], compute[(1)]);
  }
  T_relu[((((((int)blockIdx.z) * 128) + (((int)threadIdx.z) * 4)) + (((int)threadIdx.y) * 2)))] = max((compute[(0)] + placeholder2[(((((int)blockIdx.z) * 32) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[(((((((int)blockIdx.z) * 128) + (((int)threadIdx.z) * 4)) + (((int)threadIdx.y) * 2)) + 1))] = max((compute[(1)] + placeholder2[(((((int)blockIdx.z) * 32) + ((int)threadIdx.z)))]), 0.000000e+00f);
}

