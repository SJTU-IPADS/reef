#include <hip/hip_runtime.h>
extern "C" __global__ void fused_nn_conv2d_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ compute) {
  float compute_local[2];
  __shared__ float pad_temp_shared[49];
  __shared__ float placeholder_shared[16];
  compute_local[(0)] = 0.000000e+00f;
  compute_local[(1)] = 0.000000e+00f;
  if ((((((int)threadIdx.x) + ((int)threadIdx.y)) / 7) + ((int)threadIdx.z)) < 7) {
    if ((((((int)threadIdx.z) * 7) + ((int)threadIdx.x)) + ((int)threadIdx.y)) < 49) {
      if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 7) {
        if (((int)threadIdx.x) < 1) {
          pad_temp_shared[((((((int)threadIdx.z) * 7) + ((int)threadIdx.x)) + ((int)threadIdx.y)))] = placeholder[((((((((((int)threadIdx.z) / 7) * 196) + (((int)blockIdx.y) * 98)) + ((((int)threadIdx.z) % 7) * 14)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + ((int)threadIdx.y)))];
        }
      }
    }
  }
  if ((((((int)threadIdx.z) * 2) + ((int)threadIdx.x)) + ((int)threadIdx.y)) < 16) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 2) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[((((((int)threadIdx.z) * 2) + ((int)threadIdx.x)) + ((int)threadIdx.y)))] = placeholder1[(((((((int)blockIdx.z) * 16) + (((int)threadIdx.z) * 2)) + ((int)threadIdx.x)) + ((int)threadIdx.y)))];
      }
    }
  }
  __syncthreads();
  compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 7) + ((int)threadIdx.x)))], placeholder_shared[(((int)threadIdx.z))], compute_local[(0)]);
  compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 7) + ((int)threadIdx.x)))], placeholder_shared[((((int)threadIdx.z) + 8))], compute_local[(1)]);
  compute[(((((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)))] = compute_local[(0)];
  compute[((((((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 1568))] = compute_local[(1)];
}

extern "C" __global__ void fused_nn_batch_flatten_kernel0(float* __restrict__ tensor, float* __restrict__ placeholder) {
  tensor[(0)] = placeholder[(0)];
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_8_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
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
            placeholder_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x)))] = placeholder1[(((((((int)threadIdx.z) * 256) + ((((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) >> 4) * 128)) + (rc_outer * 16)) + (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) & 15)))];
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
  T_relu[(((((((int)threadIdx.z) * 784) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 2)))] = max((compute[(0)] + placeholder2[(((int)threadIdx.z))]), 0.000000e+00f);
  T_relu[((((((((int)threadIdx.z) * 784) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 2)) + 12544))] = max((compute[(2)] + placeholder2[((((int)threadIdx.z) + 16))]), 0.000000e+00f);
  T_relu[((((((((int)threadIdx.z) * 784) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 2)) + 1))] = max((compute[(1)] + placeholder2[(((int)threadIdx.z))]), 0.000000e+00f);
  T_relu[((((((((int)threadIdx.z) * 784) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 2)) + 12545))] = max((compute[(3)] + placeholder2[((((int)threadIdx.z) + 16))]), 0.000000e+00f);
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_3_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  float compute[1];
  __shared__ float pad_temp_shared[128];
  __shared__ float placeholder_shared[72];
  compute[(0)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 4; ++rc_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 32; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      pad_temp_shared[((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 32)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner & 15) >> 2))) && (((((int)blockIdx.y) * 2) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner & 15) >> 2)) < 15)) && (1 <= ((((int)blockIdx.x) * 2) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner & 3)))) && (((((int)blockIdx.x) * 2) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner & 3)) < 15)) ? placeholder[((((((((((rc_outer * 1568) + (((int)threadIdx.y) * 784)) + (((int)threadIdx.x) * 392)) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner >> 4) * 196)) + (((int)blockIdx.y) * 28)) + (((ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner & 15) >> 2) * 14)) + (((int)blockIdx.x) * 2)) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner & 3)) - 15))] : 0.000000e+00f);
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 < 18; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) {
      placeholder_shared[((((((int)threadIdx.y) * 36) + (((int)threadIdx.x) * 18)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1))] = placeholder1[(((((rc_outer * 72) + (((int)threadIdx.y) * 36)) + (((int)threadIdx.x) * 18)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1))];
    }
    __syncthreads();
    for (int rc_inner = 0; rc_inner < 8; ++rc_inner) {
      for (int ry_inner = 0; ry_inner < 3; ++ry_inner) {
        for (int rx_inner = 0; rx_inner < 3; ++rx_inner) {
          compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((rc_inner * 16) + (((int)threadIdx.y) * 4)) + (ry_inner * 4)) + ((int)threadIdx.x)) + rx_inner))], placeholder_shared[((((rc_inner * 9) + (ry_inner * 3)) + rx_inner))], compute[(0)]);
        }
      }
    }
  }
  T_relu[(((((((int)blockIdx.y) * 28) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 2)) + ((int)threadIdx.x)))] = max((compute[(0)] + placeholder2[(0)]), 0.000000e+00f);
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_10_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  float compute[4];
  __shared__ float pad_temp_shared[32];
  __shared__ float placeholder_shared[32];
  compute[(0)] = 0.000000e+00f;
  compute[(2)] = 0.000000e+00f;
  compute[(1)] = 0.000000e+00f;
  compute[(3)] = 0.000000e+00f;
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 32) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        pad_temp_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder[(((((((int)blockIdx.y) * 224) + ((((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) >> 3) * 56)) + (((int)blockIdx.x) * 8)) + (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) & 7)))];
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 32) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)))], placeholder_shared[(((int)threadIdx.z))], compute[(0)]);
  compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 16))], placeholder_shared[(((int)threadIdx.z))], compute[(2)]);
  compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 1))], placeholder_shared[(((int)threadIdx.z))], compute[(1)]);
  compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 17))], placeholder_shared[(((int)threadIdx.z))], compute[(3)]);
  T_relu[((((((((int)threadIdx.z) * 3136) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)))] = max((compute[(0)] + placeholder2[(((int)threadIdx.z))]), 0.000000e+00f);
  T_relu[(((((((((int)threadIdx.z) * 3136) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 112))] = max((compute[(2)] + placeholder2[(((int)threadIdx.z))]), 0.000000e+00f);
  T_relu[(((((((((int)threadIdx.z) * 3136) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 1))] = max((compute[(1)] + placeholder2[(((int)threadIdx.z))]), 0.000000e+00f);
  T_relu[(((((((((int)threadIdx.z) * 3136) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 113))] = max((compute[(3)] + placeholder2[(((int)threadIdx.z))]), 0.000000e+00f);
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_7_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  float compute[4];
  __shared__ float pad_temp_shared[112];
  __shared__ float placeholder_shared[32];
  compute[(0)] = 0.000000e+00f;
  compute[(2)] = 0.000000e+00f;
  compute[(1)] = 0.000000e+00f;
  compute[(3)] = 0.000000e+00f;
  if ((((((int)threadIdx.z) * 7) + (((int)threadIdx.y) * 2)) + ((int)threadIdx.x)) < 112) {
    if (((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) < 7) {
      if (((int)threadIdx.x) < 2) {
        pad_temp_shared[((((((int)threadIdx.z) * 7) + (((int)threadIdx.y) * 2)) + ((int)threadIdx.x)))] = placeholder[(((((((int)blockIdx.y) * 112) + (((int)threadIdx.z) * 7)) + (((int)threadIdx.y) * 2)) + ((int)threadIdx.x)))];
      }
    }
  }
  if ((((((int)threadIdx.z) * 2) + ((int)threadIdx.x)) + ((int)threadIdx.y)) < 32) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 2) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[((((((int)threadIdx.z) * 2) + ((int)threadIdx.x)) + ((int)threadIdx.y)))] = placeholder1[((((((int)threadIdx.z) * 2) + ((int)threadIdx.x)) + ((int)threadIdx.y)))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)))], placeholder_shared[(((int)threadIdx.z))], compute[(0)]);
  compute[(2)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)))], placeholder_shared[((((int)threadIdx.z) + 16))], compute[(2)]);
  compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 1))], placeholder_shared[(((int)threadIdx.z))], compute[(1)]);
  compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 1))], placeholder_shared[((((int)threadIdx.z) + 16))], compute[(3)]);
  T_relu[(((((((int)threadIdx.z) * 784) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 2)))] = max((compute[(0)] + placeholder2[(((int)threadIdx.z))]), 0.000000e+00f);
  T_relu[((((((((int)threadIdx.z) * 784) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 2)) + 12544))] = max((compute[(2)] + placeholder2[((((int)threadIdx.z) + 16))]), 0.000000e+00f);
  T_relu[((((((((int)threadIdx.z) * 784) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 2)) + 1))] = max((compute[(1)] + placeholder2[(((int)threadIdx.z))]), 0.000000e+00f);
  T_relu[((((((((int)threadIdx.z) * 784) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 2)) + 12545))] = max((compute[(3)] + placeholder2[((((int)threadIdx.z) + 16))]), 0.000000e+00f);
}

extern "C" __global__ void fused_nn_conv2d_1_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ compute) {
  float compute_local[4];
  __shared__ float pad_temp_shared[112];
  __shared__ float placeholder_shared[32];
  compute_local[(0)] = 0.000000e+00f;
  compute_local[(2)] = 0.000000e+00f;
  compute_local[(1)] = 0.000000e+00f;
  compute_local[(3)] = 0.000000e+00f;
  if ((((((int)threadIdx.z) * 7) + (((int)threadIdx.y) * 2)) + ((int)threadIdx.x)) < 112) {
    if (((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) < 7) {
      if (((int)threadIdx.x) < 2) {
        pad_temp_shared[((((((int)threadIdx.z) * 7) + (((int)threadIdx.y) * 2)) + ((int)threadIdx.x)))] = placeholder[(((((((int)blockIdx.y) * 112) + (((int)threadIdx.z) * 7)) + (((int)threadIdx.y) * 2)) + ((int)threadIdx.x)))];
      }
    }
  }
  if ((((((int)threadIdx.z) * 2) + ((int)threadIdx.x)) + ((int)threadIdx.y)) < 32) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 2) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[((((((int)threadIdx.z) * 2) + ((int)threadIdx.x)) + ((int)threadIdx.y)))] = placeholder1[(((((((int)blockIdx.z) * 32) + (((int)threadIdx.z) * 2)) + ((int)threadIdx.x)) + ((int)threadIdx.y)))];
      }
    }
  }
  __syncthreads();
  compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)))], placeholder_shared[(((int)threadIdx.z))], compute_local[(0)]);
  compute_local[(2)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)))], placeholder_shared[((((int)threadIdx.z) + 16))], compute_local[(2)]);
  compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 1))], placeholder_shared[(((int)threadIdx.z))], compute_local[(1)]);
  compute_local[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 1))], placeholder_shared[((((int)threadIdx.z) + 16))], compute_local[(3)]);
  compute[((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 2)))] = compute_local[(0)];
  compute[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 2)) + 12544))] = compute_local[(2)];
  compute[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 2)) + 1))] = compute_local[(1)];
  compute[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 2)) + 12545))] = compute_local[(3)];
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_11_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  float compute[4];
  __shared__ float pad_temp_shared[512];
  __shared__ float placeholder_shared[512];
  compute[(0)] = 0.000000e+00f;
  compute[(2)] = 0.000000e+00f;
  compute[(1)] = 0.000000e+00f;
  compute[(3)] = 0.000000e+00f;
  pad_temp_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)))] = placeholder[(((((((((((int)threadIdx.z) * 2) + ((int)threadIdx.y)) >> 2) * 3136) + (((int)blockIdx.y) * 224)) + ((((((int)threadIdx.z) * 2) + ((int)threadIdx.y)) & 3) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)))];
  pad_temp_shared[(((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1))] = placeholder[((((((((((((int)threadIdx.z) * 2) + ((int)threadIdx.y)) >> 2) * 3136) + (((int)blockIdx.y) * 224)) + ((((((int)threadIdx.z) * 2) + ((int)threadIdx.y)) & 3) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 1))];
  placeholder_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)))] = placeholder1[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)))];
  placeholder_shared[(((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1))] = placeholder1[(((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1))];
  __syncthreads();
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)))], placeholder_shared[((((int)threadIdx.z) * 16))], compute[(0)]);
  compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 16))], placeholder_shared[((((int)threadIdx.z) * 16))], compute[(2)]);
  compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 1))], placeholder_shared[((((int)threadIdx.z) * 16))], compute[(1)]);
  compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 17))], placeholder_shared[((((int)threadIdx.z) * 16))], compute[(3)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 32))], placeholder_shared[(((((int)threadIdx.z) * 16) + 1))], compute[(0)]);
  compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 48))], placeholder_shared[(((((int)threadIdx.z) * 16) + 1))], compute[(2)]);
  compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 33))], placeholder_shared[(((((int)threadIdx.z) * 16) + 1))], compute[(1)]);
  compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 49))], placeholder_shared[(((((int)threadIdx.z) * 16) + 1))], compute[(3)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 64))], placeholder_shared[(((((int)threadIdx.z) * 16) + 2))], compute[(0)]);
  compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 80))], placeholder_shared[(((((int)threadIdx.z) * 16) + 2))], compute[(2)]);
  compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 65))], placeholder_shared[(((((int)threadIdx.z) * 16) + 2))], compute[(1)]);
  compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 81))], placeholder_shared[(((((int)threadIdx.z) * 16) + 2))], compute[(3)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 96))], placeholder_shared[(((((int)threadIdx.z) * 16) + 3))], compute[(0)]);
  compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 112))], placeholder_shared[(((((int)threadIdx.z) * 16) + 3))], compute[(2)]);
  compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 97))], placeholder_shared[(((((int)threadIdx.z) * 16) + 3))], compute[(1)]);
  compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 113))], placeholder_shared[(((((int)threadIdx.z) * 16) + 3))], compute[(3)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 128))], placeholder_shared[(((((int)threadIdx.z) * 16) + 4))], compute[(0)]);
  compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 144))], placeholder_shared[(((((int)threadIdx.z) * 16) + 4))], compute[(2)]);
  compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 129))], placeholder_shared[(((((int)threadIdx.z) * 16) + 4))], compute[(1)]);
  compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 145))], placeholder_shared[(((((int)threadIdx.z) * 16) + 4))], compute[(3)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 160))], placeholder_shared[(((((int)threadIdx.z) * 16) + 5))], compute[(0)]);
  compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 176))], placeholder_shared[(((((int)threadIdx.z) * 16) + 5))], compute[(2)]);
  compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 161))], placeholder_shared[(((((int)threadIdx.z) * 16) + 5))], compute[(1)]);
  compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 177))], placeholder_shared[(((((int)threadIdx.z) * 16) + 5))], compute[(3)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 192))], placeholder_shared[(((((int)threadIdx.z) * 16) + 6))], compute[(0)]);
  compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 208))], placeholder_shared[(((((int)threadIdx.z) * 16) + 6))], compute[(2)]);
  compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 193))], placeholder_shared[(((((int)threadIdx.z) * 16) + 6))], compute[(1)]);
  compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 209))], placeholder_shared[(((((int)threadIdx.z) * 16) + 6))], compute[(3)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 224))], placeholder_shared[(((((int)threadIdx.z) * 16) + 7))], compute[(0)]);
  compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 240))], placeholder_shared[(((((int)threadIdx.z) * 16) + 7))], compute[(2)]);
  compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 225))], placeholder_shared[(((((int)threadIdx.z) * 16) + 7))], compute[(1)]);
  compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 241))], placeholder_shared[(((((int)threadIdx.z) * 16) + 7))], compute[(3)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 256))], placeholder_shared[(((((int)threadIdx.z) * 16) + 8))], compute[(0)]);
  compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 272))], placeholder_shared[(((((int)threadIdx.z) * 16) + 8))], compute[(2)]);
  compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 257))], placeholder_shared[(((((int)threadIdx.z) * 16) + 8))], compute[(1)]);
  compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 273))], placeholder_shared[(((((int)threadIdx.z) * 16) + 8))], compute[(3)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 288))], placeholder_shared[(((((int)threadIdx.z) * 16) + 9))], compute[(0)]);
  compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 304))], placeholder_shared[(((((int)threadIdx.z) * 16) + 9))], compute[(2)]);
  compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 289))], placeholder_shared[(((((int)threadIdx.z) * 16) + 9))], compute[(1)]);
  compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 305))], placeholder_shared[(((((int)threadIdx.z) * 16) + 9))], compute[(3)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 320))], placeholder_shared[(((((int)threadIdx.z) * 16) + 10))], compute[(0)]);
  compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 336))], placeholder_shared[(((((int)threadIdx.z) * 16) + 10))], compute[(2)]);
  compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 321))], placeholder_shared[(((((int)threadIdx.z) * 16) + 10))], compute[(1)]);
  compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 337))], placeholder_shared[(((((int)threadIdx.z) * 16) + 10))], compute[(3)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 352))], placeholder_shared[(((((int)threadIdx.z) * 16) + 11))], compute[(0)]);
  compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 368))], placeholder_shared[(((((int)threadIdx.z) * 16) + 11))], compute[(2)]);
  compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 353))], placeholder_shared[(((((int)threadIdx.z) * 16) + 11))], compute[(1)]);
  compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 369))], placeholder_shared[(((((int)threadIdx.z) * 16) + 11))], compute[(3)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 384))], placeholder_shared[(((((int)threadIdx.z) * 16) + 12))], compute[(0)]);
  compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 400))], placeholder_shared[(((((int)threadIdx.z) * 16) + 12))], compute[(2)]);
  compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 385))], placeholder_shared[(((((int)threadIdx.z) * 16) + 12))], compute[(1)]);
  compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 401))], placeholder_shared[(((((int)threadIdx.z) * 16) + 12))], compute[(3)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 416))], placeholder_shared[(((((int)threadIdx.z) * 16) + 13))], compute[(0)]);
  compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 432))], placeholder_shared[(((((int)threadIdx.z) * 16) + 13))], compute[(2)]);
  compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 417))], placeholder_shared[(((((int)threadIdx.z) * 16) + 13))], compute[(1)]);
  compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 433))], placeholder_shared[(((((int)threadIdx.z) * 16) + 13))], compute[(3)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 448))], placeholder_shared[(((((int)threadIdx.z) * 16) + 14))], compute[(0)]);
  compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 464))], placeholder_shared[(((((int)threadIdx.z) * 16) + 14))], compute[(2)]);
  compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 449))], placeholder_shared[(((((int)threadIdx.z) * 16) + 14))], compute[(1)]);
  compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 465))], placeholder_shared[(((((int)threadIdx.z) * 16) + 14))], compute[(3)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 480))], placeholder_shared[(((((int)threadIdx.z) * 16) + 15))], compute[(0)]);
  compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 496))], placeholder_shared[(((((int)threadIdx.z) * 16) + 15))], compute[(2)]);
  compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 481))], placeholder_shared[(((((int)threadIdx.z) * 16) + 15))], compute[(1)]);
  compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 497))], placeholder_shared[(((((int)threadIdx.z) * 16) + 15))], compute[(3)]);
  __syncthreads();
  pad_temp_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)))] = placeholder[((((((((((((int)threadIdx.z) * 2) + ((int)threadIdx.y)) >> 2) * 3136) + (((int)blockIdx.y) * 224)) + ((((((int)threadIdx.z) * 2) + ((int)threadIdx.y)) & 3) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 50176))];
  pad_temp_shared[(((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1))] = placeholder[((((((((((((int)threadIdx.z) * 2) + ((int)threadIdx.y)) >> 2) * 3136) + (((int)blockIdx.y) * 224)) + ((((((int)threadIdx.z) * 2) + ((int)threadIdx.y)) & 3) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 50177))];
  placeholder_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)))] = placeholder1[(((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 16))];
  placeholder_shared[(((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1))] = placeholder1[(((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 17))];
  __syncthreads();
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)))], placeholder_shared[((((int)threadIdx.z) * 16))], compute[(0)]);
  compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 16))], placeholder_shared[((((int)threadIdx.z) * 16))], compute[(2)]);
  compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 1))], placeholder_shared[((((int)threadIdx.z) * 16))], compute[(1)]);
  compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 17))], placeholder_shared[((((int)threadIdx.z) * 16))], compute[(3)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 32))], placeholder_shared[(((((int)threadIdx.z) * 16) + 1))], compute[(0)]);
  compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 48))], placeholder_shared[(((((int)threadIdx.z) * 16) + 1))], compute[(2)]);
  compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 33))], placeholder_shared[(((((int)threadIdx.z) * 16) + 1))], compute[(1)]);
  compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 49))], placeholder_shared[(((((int)threadIdx.z) * 16) + 1))], compute[(3)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 64))], placeholder_shared[(((((int)threadIdx.z) * 16) + 2))], compute[(0)]);
  compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 80))], placeholder_shared[(((((int)threadIdx.z) * 16) + 2))], compute[(2)]);
  compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 65))], placeholder_shared[(((((int)threadIdx.z) * 16) + 2))], compute[(1)]);
  compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 81))], placeholder_shared[(((((int)threadIdx.z) * 16) + 2))], compute[(3)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 96))], placeholder_shared[(((((int)threadIdx.z) * 16) + 3))], compute[(0)]);
  compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 112))], placeholder_shared[(((((int)threadIdx.z) * 16) + 3))], compute[(2)]);
  compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 97))], placeholder_shared[(((((int)threadIdx.z) * 16) + 3))], compute[(1)]);
  compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 113))], placeholder_shared[(((((int)threadIdx.z) * 16) + 3))], compute[(3)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 128))], placeholder_shared[(((((int)threadIdx.z) * 16) + 4))], compute[(0)]);
  compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 144))], placeholder_shared[(((((int)threadIdx.z) * 16) + 4))], compute[(2)]);
  compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 129))], placeholder_shared[(((((int)threadIdx.z) * 16) + 4))], compute[(1)]);
  compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 145))], placeholder_shared[(((((int)threadIdx.z) * 16) + 4))], compute[(3)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 160))], placeholder_shared[(((((int)threadIdx.z) * 16) + 5))], compute[(0)]);
  compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 176))], placeholder_shared[(((((int)threadIdx.z) * 16) + 5))], compute[(2)]);
  compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 161))], placeholder_shared[(((((int)threadIdx.z) * 16) + 5))], compute[(1)]);
  compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 177))], placeholder_shared[(((((int)threadIdx.z) * 16) + 5))], compute[(3)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 192))], placeholder_shared[(((((int)threadIdx.z) * 16) + 6))], compute[(0)]);
  compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 208))], placeholder_shared[(((((int)threadIdx.z) * 16) + 6))], compute[(2)]);
  compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 193))], placeholder_shared[(((((int)threadIdx.z) * 16) + 6))], compute[(1)]);
  compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 209))], placeholder_shared[(((((int)threadIdx.z) * 16) + 6))], compute[(3)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 224))], placeholder_shared[(((((int)threadIdx.z) * 16) + 7))], compute[(0)]);
  compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 240))], placeholder_shared[(((((int)threadIdx.z) * 16) + 7))], compute[(2)]);
  compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 225))], placeholder_shared[(((((int)threadIdx.z) * 16) + 7))], compute[(1)]);
  compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 241))], placeholder_shared[(((((int)threadIdx.z) * 16) + 7))], compute[(3)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 256))], placeholder_shared[(((((int)threadIdx.z) * 16) + 8))], compute[(0)]);
  compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 272))], placeholder_shared[(((((int)threadIdx.z) * 16) + 8))], compute[(2)]);
  compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 257))], placeholder_shared[(((((int)threadIdx.z) * 16) + 8))], compute[(1)]);
  compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 273))], placeholder_shared[(((((int)threadIdx.z) * 16) + 8))], compute[(3)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 288))], placeholder_shared[(((((int)threadIdx.z) * 16) + 9))], compute[(0)]);
  compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 304))], placeholder_shared[(((((int)threadIdx.z) * 16) + 9))], compute[(2)]);
  compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 289))], placeholder_shared[(((((int)threadIdx.z) * 16) + 9))], compute[(1)]);
  compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 305))], placeholder_shared[(((((int)threadIdx.z) * 16) + 9))], compute[(3)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 320))], placeholder_shared[(((((int)threadIdx.z) * 16) + 10))], compute[(0)]);
  compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 336))], placeholder_shared[(((((int)threadIdx.z) * 16) + 10))], compute[(2)]);
  compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 321))], placeholder_shared[(((((int)threadIdx.z) * 16) + 10))], compute[(1)]);
  compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 337))], placeholder_shared[(((((int)threadIdx.z) * 16) + 10))], compute[(3)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 352))], placeholder_shared[(((((int)threadIdx.z) * 16) + 11))], compute[(0)]);
  compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 368))], placeholder_shared[(((((int)threadIdx.z) * 16) + 11))], compute[(2)]);
  compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 353))], placeholder_shared[(((((int)threadIdx.z) * 16) + 11))], compute[(1)]);
  compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 369))], placeholder_shared[(((((int)threadIdx.z) * 16) + 11))], compute[(3)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 384))], placeholder_shared[(((((int)threadIdx.z) * 16) + 12))], compute[(0)]);
  compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 400))], placeholder_shared[(((((int)threadIdx.z) * 16) + 12))], compute[(2)]);
  compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 385))], placeholder_shared[(((((int)threadIdx.z) * 16) + 12))], compute[(1)]);
  compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 401))], placeholder_shared[(((((int)threadIdx.z) * 16) + 12))], compute[(3)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 416))], placeholder_shared[(((((int)threadIdx.z) * 16) + 13))], compute[(0)]);
  compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 432))], placeholder_shared[(((((int)threadIdx.z) * 16) + 13))], compute[(2)]);
  compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 417))], placeholder_shared[(((((int)threadIdx.z) * 16) + 13))], compute[(1)]);
  compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 433))], placeholder_shared[(((((int)threadIdx.z) * 16) + 13))], compute[(3)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 448))], placeholder_shared[(((((int)threadIdx.z) * 16) + 14))], compute[(0)]);
  compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 464))], placeholder_shared[(((((int)threadIdx.z) * 16) + 14))], compute[(2)]);
  compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 449))], placeholder_shared[(((((int)threadIdx.z) * 16) + 14))], compute[(1)]);
  compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 465))], placeholder_shared[(((((int)threadIdx.z) * 16) + 14))], compute[(3)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 480))], placeholder_shared[(((((int)threadIdx.z) * 16) + 15))], compute[(0)]);
  compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 496))], placeholder_shared[(((((int)threadIdx.z) * 16) + 15))], compute[(2)]);
  compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 481))], placeholder_shared[(((((int)threadIdx.z) * 16) + 15))], compute[(1)]);
  compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 497))], placeholder_shared[(((((int)threadIdx.z) * 16) + 15))], compute[(3)]);
  __syncthreads();
  pad_temp_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)))] = placeholder[((((((((((((int)threadIdx.z) * 2) + ((int)threadIdx.y)) >> 2) * 3136) + (((int)blockIdx.y) * 224)) + ((((((int)threadIdx.z) * 2) + ((int)threadIdx.y)) & 3) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 100352))];
  pad_temp_shared[(((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1))] = placeholder[((((((((((((int)threadIdx.z) * 2) + ((int)threadIdx.y)) >> 2) * 3136) + (((int)blockIdx.y) * 224)) + ((((((int)threadIdx.z) * 2) + ((int)threadIdx.y)) & 3) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 100353))];
  placeholder_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)))] = placeholder1[(((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 32))];
  placeholder_shared[(((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1))] = placeholder1[(((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 33))];
  __syncthreads();
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)))], placeholder_shared[((((int)threadIdx.z) * 16))], compute[(0)]);
  compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 16))], placeholder_shared[((((int)threadIdx.z) * 16))], compute[(2)]);
  compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 1))], placeholder_shared[((((int)threadIdx.z) * 16))], compute[(1)]);
  compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 17))], placeholder_shared[((((int)threadIdx.z) * 16))], compute[(3)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 32))], placeholder_shared[(((((int)threadIdx.z) * 16) + 1))], compute[(0)]);
  compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 48))], placeholder_shared[(((((int)threadIdx.z) * 16) + 1))], compute[(2)]);
  compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 33))], placeholder_shared[(((((int)threadIdx.z) * 16) + 1))], compute[(1)]);
  compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 49))], placeholder_shared[(((((int)threadIdx.z) * 16) + 1))], compute[(3)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 64))], placeholder_shared[(((((int)threadIdx.z) * 16) + 2))], compute[(0)]);
  compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 80))], placeholder_shared[(((((int)threadIdx.z) * 16) + 2))], compute[(2)]);
  compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 65))], placeholder_shared[(((((int)threadIdx.z) * 16) + 2))], compute[(1)]);
  compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 81))], placeholder_shared[(((((int)threadIdx.z) * 16) + 2))], compute[(3)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 96))], placeholder_shared[(((((int)threadIdx.z) * 16) + 3))], compute[(0)]);
  compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 112))], placeholder_shared[(((((int)threadIdx.z) * 16) + 3))], compute[(2)]);
  compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 97))], placeholder_shared[(((((int)threadIdx.z) * 16) + 3))], compute[(1)]);
  compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 113))], placeholder_shared[(((((int)threadIdx.z) * 16) + 3))], compute[(3)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 128))], placeholder_shared[(((((int)threadIdx.z) * 16) + 4))], compute[(0)]);
  compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 144))], placeholder_shared[(((((int)threadIdx.z) * 16) + 4))], compute[(2)]);
  compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 129))], placeholder_shared[(((((int)threadIdx.z) * 16) + 4))], compute[(1)]);
  compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 145))], placeholder_shared[(((((int)threadIdx.z) * 16) + 4))], compute[(3)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 160))], placeholder_shared[(((((int)threadIdx.z) * 16) + 5))], compute[(0)]);
  compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 176))], placeholder_shared[(((((int)threadIdx.z) * 16) + 5))], compute[(2)]);
  compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 161))], placeholder_shared[(((((int)threadIdx.z) * 16) + 5))], compute[(1)]);
  compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 177))], placeholder_shared[(((((int)threadIdx.z) * 16) + 5))], compute[(3)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 192))], placeholder_shared[(((((int)threadIdx.z) * 16) + 6))], compute[(0)]);
  compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 208))], placeholder_shared[(((((int)threadIdx.z) * 16) + 6))], compute[(2)]);
  compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 193))], placeholder_shared[(((((int)threadIdx.z) * 16) + 6))], compute[(1)]);
  compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 209))], placeholder_shared[(((((int)threadIdx.z) * 16) + 6))], compute[(3)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 224))], placeholder_shared[(((((int)threadIdx.z) * 16) + 7))], compute[(0)]);
  compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 240))], placeholder_shared[(((((int)threadIdx.z) * 16) + 7))], compute[(2)]);
  compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 225))], placeholder_shared[(((((int)threadIdx.z) * 16) + 7))], compute[(1)]);
  compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 241))], placeholder_shared[(((((int)threadIdx.z) * 16) + 7))], compute[(3)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 256))], placeholder_shared[(((((int)threadIdx.z) * 16) + 8))], compute[(0)]);
  compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 272))], placeholder_shared[(((((int)threadIdx.z) * 16) + 8))], compute[(2)]);
  compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 257))], placeholder_shared[(((((int)threadIdx.z) * 16) + 8))], compute[(1)]);
  compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 273))], placeholder_shared[(((((int)threadIdx.z) * 16) + 8))], compute[(3)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 288))], placeholder_shared[(((((int)threadIdx.z) * 16) + 9))], compute[(0)]);
  compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 304))], placeholder_shared[(((((int)threadIdx.z) * 16) + 9))], compute[(2)]);
  compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 289))], placeholder_shared[(((((int)threadIdx.z) * 16) + 9))], compute[(1)]);
  compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 305))], placeholder_shared[(((((int)threadIdx.z) * 16) + 9))], compute[(3)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 320))], placeholder_shared[(((((int)threadIdx.z) * 16) + 10))], compute[(0)]);
  compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 336))], placeholder_shared[(((((int)threadIdx.z) * 16) + 10))], compute[(2)]);
  compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 321))], placeholder_shared[(((((int)threadIdx.z) * 16) + 10))], compute[(1)]);
  compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 337))], placeholder_shared[(((((int)threadIdx.z) * 16) + 10))], compute[(3)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 352))], placeholder_shared[(((((int)threadIdx.z) * 16) + 11))], compute[(0)]);
  compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 368))], placeholder_shared[(((((int)threadIdx.z) * 16) + 11))], compute[(2)]);
  compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 353))], placeholder_shared[(((((int)threadIdx.z) * 16) + 11))], compute[(1)]);
  compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 369))], placeholder_shared[(((((int)threadIdx.z) * 16) + 11))], compute[(3)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 384))], placeholder_shared[(((((int)threadIdx.z) * 16) + 12))], compute[(0)]);
  compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 400))], placeholder_shared[(((((int)threadIdx.z) * 16) + 12))], compute[(2)]);
  compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 385))], placeholder_shared[(((((int)threadIdx.z) * 16) + 12))], compute[(1)]);
  compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 401))], placeholder_shared[(((((int)threadIdx.z) * 16) + 12))], compute[(3)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 416))], placeholder_shared[(((((int)threadIdx.z) * 16) + 13))], compute[(0)]);
  compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 432))], placeholder_shared[(((((int)threadIdx.z) * 16) + 13))], compute[(2)]);
  compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 417))], placeholder_shared[(((((int)threadIdx.z) * 16) + 13))], compute[(1)]);
  compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 433))], placeholder_shared[(((((int)threadIdx.z) * 16) + 13))], compute[(3)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 448))], placeholder_shared[(((((int)threadIdx.z) * 16) + 14))], compute[(0)]);
  compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 464))], placeholder_shared[(((((int)threadIdx.z) * 16) + 14))], compute[(2)]);
  compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 449))], placeholder_shared[(((((int)threadIdx.z) * 16) + 14))], compute[(1)]);
  compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 465))], placeholder_shared[(((((int)threadIdx.z) * 16) + 14))], compute[(3)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 480))], placeholder_shared[(((((int)threadIdx.z) * 16) + 15))], compute[(0)]);
  compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 496))], placeholder_shared[(((((int)threadIdx.z) * 16) + 15))], compute[(2)]);
  compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 481))], placeholder_shared[(((((int)threadIdx.z) * 16) + 15))], compute[(1)]);
  compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 497))], placeholder_shared[(((((int)threadIdx.z) * 16) + 15))], compute[(3)]);
  __syncthreads();
  pad_temp_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)))] = placeholder[((((((((((((int)threadIdx.z) * 2) + ((int)threadIdx.y)) >> 2) * 3136) + (((int)blockIdx.y) * 224)) + ((((((int)threadIdx.z) * 2) + ((int)threadIdx.y)) & 3) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 150528))];
  pad_temp_shared[(((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1))] = placeholder[((((((((((((int)threadIdx.z) * 2) + ((int)threadIdx.y)) >> 2) * 3136) + (((int)blockIdx.y) * 224)) + ((((((int)threadIdx.z) * 2) + ((int)threadIdx.y)) & 3) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 150529))];
  placeholder_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)))] = placeholder1[(((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 48))];
  placeholder_shared[(((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1))] = placeholder1[(((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 49))];
  __syncthreads();
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)))], placeholder_shared[((((int)threadIdx.z) * 16))], compute[(0)]);
  compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 16))], placeholder_shared[((((int)threadIdx.z) * 16))], compute[(2)]);
  compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 1))], placeholder_shared[((((int)threadIdx.z) * 16))], compute[(1)]);
  compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 17))], placeholder_shared[((((int)threadIdx.z) * 16))], compute[(3)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 32))], placeholder_shared[(((((int)threadIdx.z) * 16) + 1))], compute[(0)]);
  compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 48))], placeholder_shared[(((((int)threadIdx.z) * 16) + 1))], compute[(2)]);
  compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 33))], placeholder_shared[(((((int)threadIdx.z) * 16) + 1))], compute[(1)]);
  compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 49))], placeholder_shared[(((((int)threadIdx.z) * 16) + 1))], compute[(3)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 64))], placeholder_shared[(((((int)threadIdx.z) * 16) + 2))], compute[(0)]);
  compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 80))], placeholder_shared[(((((int)threadIdx.z) * 16) + 2))], compute[(2)]);
  compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 65))], placeholder_shared[(((((int)threadIdx.z) * 16) + 2))], compute[(1)]);
  compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 81))], placeholder_shared[(((((int)threadIdx.z) * 16) + 2))], compute[(3)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 96))], placeholder_shared[(((((int)threadIdx.z) * 16) + 3))], compute[(0)]);
  compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 112))], placeholder_shared[(((((int)threadIdx.z) * 16) + 3))], compute[(2)]);
  compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 97))], placeholder_shared[(((((int)threadIdx.z) * 16) + 3))], compute[(1)]);
  compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 113))], placeholder_shared[(((((int)threadIdx.z) * 16) + 3))], compute[(3)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 128))], placeholder_shared[(((((int)threadIdx.z) * 16) + 4))], compute[(0)]);
  compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 144))], placeholder_shared[(((((int)threadIdx.z) * 16) + 4))], compute[(2)]);
  compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 129))], placeholder_shared[(((((int)threadIdx.z) * 16) + 4))], compute[(1)]);
  compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 145))], placeholder_shared[(((((int)threadIdx.z) * 16) + 4))], compute[(3)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 160))], placeholder_shared[(((((int)threadIdx.z) * 16) + 5))], compute[(0)]);
  compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 176))], placeholder_shared[(((((int)threadIdx.z) * 16) + 5))], compute[(2)]);
  compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 161))], placeholder_shared[(((((int)threadIdx.z) * 16) + 5))], compute[(1)]);
  compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 177))], placeholder_shared[(((((int)threadIdx.z) * 16) + 5))], compute[(3)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 192))], placeholder_shared[(((((int)threadIdx.z) * 16) + 6))], compute[(0)]);
  compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 208))], placeholder_shared[(((((int)threadIdx.z) * 16) + 6))], compute[(2)]);
  compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 193))], placeholder_shared[(((((int)threadIdx.z) * 16) + 6))], compute[(1)]);
  compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 209))], placeholder_shared[(((((int)threadIdx.z) * 16) + 6))], compute[(3)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 224))], placeholder_shared[(((((int)threadIdx.z) * 16) + 7))], compute[(0)]);
  compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 240))], placeholder_shared[(((((int)threadIdx.z) * 16) + 7))], compute[(2)]);
  compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 225))], placeholder_shared[(((((int)threadIdx.z) * 16) + 7))], compute[(1)]);
  compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 241))], placeholder_shared[(((((int)threadIdx.z) * 16) + 7))], compute[(3)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 256))], placeholder_shared[(((((int)threadIdx.z) * 16) + 8))], compute[(0)]);
  compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 272))], placeholder_shared[(((((int)threadIdx.z) * 16) + 8))], compute[(2)]);
  compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 257))], placeholder_shared[(((((int)threadIdx.z) * 16) + 8))], compute[(1)]);
  compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 273))], placeholder_shared[(((((int)threadIdx.z) * 16) + 8))], compute[(3)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 288))], placeholder_shared[(((((int)threadIdx.z) * 16) + 9))], compute[(0)]);
  compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 304))], placeholder_shared[(((((int)threadIdx.z) * 16) + 9))], compute[(2)]);
  compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 289))], placeholder_shared[(((((int)threadIdx.z) * 16) + 9))], compute[(1)]);
  compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 305))], placeholder_shared[(((((int)threadIdx.z) * 16) + 9))], compute[(3)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 320))], placeholder_shared[(((((int)threadIdx.z) * 16) + 10))], compute[(0)]);
  compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 336))], placeholder_shared[(((((int)threadIdx.z) * 16) + 10))], compute[(2)]);
  compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 321))], placeholder_shared[(((((int)threadIdx.z) * 16) + 10))], compute[(1)]);
  compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 337))], placeholder_shared[(((((int)threadIdx.z) * 16) + 10))], compute[(3)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 352))], placeholder_shared[(((((int)threadIdx.z) * 16) + 11))], compute[(0)]);
  compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 368))], placeholder_shared[(((((int)threadIdx.z) * 16) + 11))], compute[(2)]);
  compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 353))], placeholder_shared[(((((int)threadIdx.z) * 16) + 11))], compute[(1)]);
  compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 369))], placeholder_shared[(((((int)threadIdx.z) * 16) + 11))], compute[(3)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 384))], placeholder_shared[(((((int)threadIdx.z) * 16) + 12))], compute[(0)]);
  compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 400))], placeholder_shared[(((((int)threadIdx.z) * 16) + 12))], compute[(2)]);
  compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 385))], placeholder_shared[(((((int)threadIdx.z) * 16) + 12))], compute[(1)]);
  compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 401))], placeholder_shared[(((((int)threadIdx.z) * 16) + 12))], compute[(3)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 416))], placeholder_shared[(((((int)threadIdx.z) * 16) + 13))], compute[(0)]);
  compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 432))], placeholder_shared[(((((int)threadIdx.z) * 16) + 13))], compute[(2)]);
  compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 417))], placeholder_shared[(((((int)threadIdx.z) * 16) + 13))], compute[(1)]);
  compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 433))], placeholder_shared[(((((int)threadIdx.z) * 16) + 13))], compute[(3)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 448))], placeholder_shared[(((((int)threadIdx.z) * 16) + 14))], compute[(0)]);
  compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 464))], placeholder_shared[(((((int)threadIdx.z) * 16) + 14))], compute[(2)]);
  compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 449))], placeholder_shared[(((((int)threadIdx.z) * 16) + 14))], compute[(1)]);
  compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 465))], placeholder_shared[(((((int)threadIdx.z) * 16) + 14))], compute[(3)]);
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 480))], placeholder_shared[(((((int)threadIdx.z) * 16) + 15))], compute[(0)]);
  compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 496))], placeholder_shared[(((((int)threadIdx.z) * 16) + 15))], compute[(2)]);
  compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 481))], placeholder_shared[(((((int)threadIdx.z) * 16) + 15))], compute[(1)]);
  compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 497))], placeholder_shared[(((((int)threadIdx.z) * 16) + 15))], compute[(3)]);
  T_relu[((((((((int)threadIdx.z) * 3136) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)))] = max((compute[(0)] + placeholder2[(((int)threadIdx.z))]), 0.000000e+00f);
  T_relu[(((((((((int)threadIdx.z) * 3136) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 112))] = max((compute[(2)] + placeholder2[(((int)threadIdx.z))]), 0.000000e+00f);
  T_relu[(((((((((int)threadIdx.z) * 3136) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 1))] = max((compute[(1)] + placeholder2[(((int)threadIdx.z))]), 0.000000e+00f);
  T_relu[(((((((((int)threadIdx.z) * 3136) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 113))] = max((compute[(3)] + placeholder2[(((int)threadIdx.z))]), 0.000000e+00f);
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_4_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  float compute[2];
  __shared__ float pad_temp_shared[49];
  __shared__ float placeholder_shared[16];
  compute[(0)] = 0.000000e+00f;
  compute[(1)] = 0.000000e+00f;
  if ((((((int)threadIdx.x) + ((int)threadIdx.y)) / 7) + ((int)threadIdx.z)) < 7) {
    if ((((((int)threadIdx.z) * 7) + ((int)threadIdx.x)) + ((int)threadIdx.y)) < 49) {
      if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 7) {
        if (((int)threadIdx.x) < 1) {
          pad_temp_shared[((((((int)threadIdx.z) * 7) + ((int)threadIdx.x)) + ((int)threadIdx.y)))] = placeholder[((((((((((int)threadIdx.z) / 7) * 196) + (((int)blockIdx.y) * 98)) + ((((int)threadIdx.z) % 7) * 14)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + ((int)threadIdx.y)))];
        }
      }
    }
  }
  if ((((((int)threadIdx.z) * 2) + ((int)threadIdx.x)) + ((int)threadIdx.y)) < 16) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 2) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[((((((int)threadIdx.z) * 2) + ((int)threadIdx.x)) + ((int)threadIdx.y)))] = placeholder1[(((((((int)blockIdx.z) * 16) + (((int)threadIdx.z) * 2)) + ((int)threadIdx.x)) + ((int)threadIdx.y)))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 7) + ((int)threadIdx.x)))], placeholder_shared[(((int)threadIdx.z))], compute[(0)]);
  compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 7) + ((int)threadIdx.x)))], placeholder_shared[((((int)threadIdx.z) + 8))], compute[(1)]);
  T_relu[(((((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)))] = max((compute[(0)] + placeholder2[(((((int)blockIdx.z) * 16) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[((((((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 1568))] = max((compute[(1)] + placeholder2[((((((int)blockIdx.z) * 16) + ((int)threadIdx.z)) + 8))]), 0.000000e+00f);
}

extern "C" __global__ void fused_nn_dense_add_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_add, float* __restrict__ placeholder2) {
  float T_dense_rf[1];
  float red_buf0[1];
  __shared__ float T_dense[1];
  T_dense_rf[(0)] = 0.000000e+00f;
  if (((int)threadIdx.x) < 1) {
    T_dense_rf[(0)] = __ocml_fma_f32(placeholder[(((int)threadIdx.x))], placeholder1[((((int)blockIdx.x) + ((int)threadIdx.x)))], T_dense_rf[(0)]);
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

extern "C" __global__ void fused_nn_avg_pool2d_add_nn_relu_kernel0(float* __restrict__ placeholder, float* __restrict__ T_relu, float* __restrict__ placeholder1) {
  float tensor[1];
  tensor[(0)] = 0.000000e+00f;
  for (int dh = 0; dh < 2; ++dh) {
    for (int dw = 0; dw < 2; ++dw) {
      if (((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) < 43904) {
        tensor[(0)] = (tensor[(0)] + placeholder[((((((((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) / 7) * 28) + (dh * 14)) + ((((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) % 7) * 2)) + dw))]);
      }
    }
  }
  if (((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) < 43904) {
    T_relu[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)))] = max(__ocml_fma_f32(tensor[(0)], 2.500000e-01f, placeholder1[((((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) / 49))]), 0.000000e+00f);
  }
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  float compute[1];
  __shared__ float pad_temp_shared[162];
  __shared__ float placeholder_shared[18];
  compute[(0)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 16; ++rc_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      if ((((((int)threadIdx.y) * 24) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 162) {
        if (((((int)threadIdx.x) * 4) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 24) {
          pad_temp_shared[((((((int)threadIdx.y) * 24) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))] = (((((9 <= ((((((int)threadIdx.y) * 24) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 81)) && (((((((int)threadIdx.y) * 24) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 81) < 72)) && (1 <= ((((((int)threadIdx.y) * 24) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 9))) && (((((((int)threadIdx.y) * 24) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 9) < 8)) ? placeholder[((((((rc_outer * 98) + (((((((int)threadIdx.y) * 24) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 81) * 49)) + ((((((((int)threadIdx.y) * 24) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 81) / 9) * 7)) + ((((((int)threadIdx.y) * 24) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 9)) - 8))] : 0.000000e+00f);
        }
      }
    }
    if (((((int)threadIdx.x) / 3) + ((int)threadIdx.y)) < 6) {
      if (((((int)threadIdx.y) * 3) + ((int)threadIdx.x)) < 18) {
        if (((int)threadIdx.x) < 3) {
          placeholder_shared[(((((int)threadIdx.y) * 3) + ((int)threadIdx.x)))] = placeholder1[((((((((int)threadIdx.y) / 6) * 288) + (rc_outer * 18)) + ((((int)threadIdx.y) % 6) * 3)) + ((int)threadIdx.x)))];
        }
      }
    }
    __syncthreads();
    for (int rc_inner = 0; rc_inner < 2; ++rc_inner) {
      for (int ry_inner = 0; ry_inner < 3; ++ry_inner) {
        for (int rx_inner = 0; rx_inner < 3; ++rx_inner) {
          compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((rc_inner * 81) + (((int)threadIdx.y) * 9)) + (ry_inner * 9)) + ((int)threadIdx.x)) + rx_inner))], placeholder_shared[((((rc_inner * 9) + (ry_inner * 3)) + rx_inner))], compute[(0)]);
        }
      }
    }
  }
  T_relu[(((((int)threadIdx.y) * 7) + ((int)threadIdx.x)))] = max((compute[(0)] + placeholder2[(0)]), 0.000000e+00f);
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_9_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  float compute[1];
  __shared__ float pad_temp_shared[360];
  __shared__ float placeholder_shared[36];
  compute[(0)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 8; ++rc_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 7; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      if (((((int)threadIdx.y) * 5) + (((((int)threadIdx.x) * 7) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 9)) < 40) {
        if ((((((int)threadIdx.y) * 45) + (((int)threadIdx.x) * 7)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 360) {
          if (((((int)threadIdx.x) * 7) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 45) {
            pad_temp_shared[((((((int)threadIdx.y) * 45) + (((int)threadIdx.x) * 7)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))] = (((((1 <= ((((int)blockIdx.y) * 8) + (((((int)threadIdx.y) * 5) + (((((int)threadIdx.x) * 7) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 9)) % 10))) && (((((int)blockIdx.y) * 8) + (((((int)threadIdx.y) * 5) + (((((int)threadIdx.x) * 7) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 9)) % 10)) < 57)) && (1 <= ((((int)blockIdx.x) * 7) + (((((int)threadIdx.x) * 7) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 9)))) && (((((int)blockIdx.x) * 7) + (((((int)threadIdx.x) * 7) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 9)) < 57)) ? placeholder[((((((((rc_outer * 12544) + ((((((int)threadIdx.y) * 5) + (((((int)threadIdx.x) * 7) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 9)) / 10) * 3136)) + (((int)blockIdx.y) * 448)) + ((((((int)threadIdx.y) * 5) + (((((int)threadIdx.x) * 7) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 9)) % 10) * 56)) + (((int)blockIdx.x) * 7)) + (((((int)threadIdx.x) * 7) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 9)) - 57))] : 0.000000e+00f);
          }
        }
      }
    }
    if (((((int)threadIdx.y) * 5) + ((int)threadIdx.x)) < 36) {
      if (((int)threadIdx.x) < 5) {
        placeholder_shared[(((((int)threadIdx.y) * 5) + ((int)threadIdx.x)))] = placeholder1[((((rc_outer * 36) + (((int)threadIdx.y) * 5)) + ((int)threadIdx.x)))];
      }
    }
    __syncthreads();
    for (int rc_inner = 0; rc_inner < 4; ++rc_inner) {
      for (int ry_inner = 0; ry_inner < 3; ++ry_inner) {
        for (int rx_inner = 0; rx_inner < 3; ++rx_inner) {
          compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((rc_inner * 90) + (((int)threadIdx.y) * 9)) + (ry_inner * 9)) + ((int)threadIdx.x)) + rx_inner))], placeholder_shared[((((rc_inner * 9) + (ry_inner * 3)) + rx_inner))], compute[(0)]);
        }
      }
    }
  }
  T_relu[(((((((int)blockIdx.y) * 448) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)))] = max((compute[(0)] + placeholder2[(0)]), 0.000000e+00f);
}

extern "C" __global__ void fused_nn_avg_pool2d_add_nn_relu_1_kernel0(float* __restrict__ placeholder, float* __restrict__ T_relu, float* __restrict__ placeholder1) {
  float tensor[1];
  tensor[(0)] = 0.000000e+00f;
  for (int dh = 0; dh < 2; ++dh) {
    for (int dw = 0; dw < 2; ++dw) {
      tensor[(0)] = (tensor[(0)] + placeholder[((((((((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) / 14) * 56) + (dh * 28)) + ((((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) % 14) * 2)) + dw))]);
    }
  }
  T_relu[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)))] = max(__ocml_fma_f32(tensor[(0)], 2.500000e-01f, placeholder1[((((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) / 196))]), 0.000000e+00f);
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_5_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  float compute[2];
  __shared__ float pad_temp_shared[784];
  __shared__ float placeholder_shared[256];
  compute[(0)] = 0.000000e+00f;
  compute[(1)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 16; ++rc_outer) {
    __syncthreads();
    pad_temp_shared[((((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)))] = placeholder[((((((((rc_outer * 3136) + (((int)threadIdx.z) * 392)) + ((((((int)threadIdx.y) * 2) + ((((int)threadIdx.x) * 2) / 7)) / 7) * 196)) + (((int)blockIdx.y) * 98)) + ((((((int)threadIdx.y) * 2) + ((((int)threadIdx.x) * 2) / 7)) % 7) * 14)) + (((int)blockIdx.x) * 7)) + ((((int)threadIdx.x) * 2) % 7)))];
    pad_temp_shared[(((((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)) + 1))] = placeholder[((((((((rc_outer * 3136) + (((int)threadIdx.z) * 392)) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) / 7)) / 7) * 196)) + (((int)blockIdx.y) * 98)) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) / 7)) % 7) * 14)) + (((int)blockIdx.x) * 7)) + (((((int)threadIdx.x) * 2) + 1) % 7)))];
    if (((((int)threadIdx.z) * 2) + (((((int)threadIdx.y) * 5) + ((int)threadIdx.x)) >> 4)) < 16) {
      if ((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 5)) + ((int)threadIdx.x)) < 256) {
        if (((((int)threadIdx.y) * 5) + ((int)threadIdx.x)) < 32) {
          if (((int)threadIdx.x) < 5) {
            placeholder_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 5)) + ((int)threadIdx.x)))] = placeholder1[((((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 512)) + ((((((int)threadIdx.y) * 5) + ((int)threadIdx.x)) >> 4) * 256)) + (rc_outer * 16)) + (((((int)threadIdx.y) * 5) + ((int)threadIdx.x)) & 15)))];
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

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_1_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  float compute[1];
  __shared__ float pad_temp_shared[49];
  __shared__ float placeholder_shared[16];
  compute[(0)] = 0.000000e+00f;
  if ((((((int)threadIdx.z) * 4) + ((int)threadIdx.x)) + ((int)threadIdx.y)) < 49) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 4) {
      if (((int)threadIdx.x) < 1) {
        pad_temp_shared[((((((int)threadIdx.z) * 4) + ((int)threadIdx.x)) + ((int)threadIdx.y)))] = placeholder[((((((int)threadIdx.z) * 4) + ((int)threadIdx.x)) + ((int)threadIdx.y)))];
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 16) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[(((((((int)blockIdx.z) * 16) + ((int)threadIdx.x)) + ((int)threadIdx.y)) + ((int)threadIdx.z)))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 7) + ((int)threadIdx.x)))], placeholder_shared[(((int)threadIdx.z))], compute[(0)]);
  T_relu[(((((((int)blockIdx.z) * 784) + (((int)threadIdx.z) * 49)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)))] = max((compute[(0)] + placeholder2[(((((int)blockIdx.z) * 16) + ((int)threadIdx.z)))]), 0.000000e+00f);
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_2_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  float compute[1];
  __shared__ float pad_temp_shared[196];
  __shared__ float placeholder_shared[112];
  compute[(0)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 32; ++rc_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 7; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      pad_temp_shared[((((((int)threadIdx.z) * 49) + (((int)threadIdx.x) * 7)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))] = placeholder[((((((rc_outer * 1372) + (((int)threadIdx.z) * 343)) + (((int)threadIdx.x) * 49)) + (((int)blockIdx.y) * 7)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))];
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) {
      placeholder_shared[((((((int)threadIdx.z) * 28) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1))] = placeholder1[((((((((int)blockIdx.z) * 3584) + (((int)threadIdx.z) * 896)) + (rc_outer * 28)) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1))];
    }
    __syncthreads();
    for (int rc_inner = 0; rc_inner < 28; ++rc_inner) {
      compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((rc_inner * 7) + ((int)threadIdx.x)))], placeholder_shared[(((((int)threadIdx.z) * 28) + rc_inner))], compute[(0)]);
    }
  }
  T_relu[(((((((int)blockIdx.z) * 196) + (((int)threadIdx.z) * 49)) + (((int)blockIdx.y) * 7)) + ((int)threadIdx.x)))] = max((compute[(0)] + placeholder2[(((((int)blockIdx.z) * 4) + ((int)threadIdx.z)))]), 0.000000e+00f);
}

extern "C" __global__ void fused_nn_conv2d_2_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ compute) {
  float compute_local[8];
  __shared__ float pad_temp_shared[32];
  __shared__ float placeholder_shared[64];
  compute_local[(0)] = 0.000000e+00f;
  compute_local[(4)] = 0.000000e+00f;
  compute_local[(2)] = 0.000000e+00f;
  compute_local[(6)] = 0.000000e+00f;
  compute_local[(1)] = 0.000000e+00f;
  compute_local[(5)] = 0.000000e+00f;
  compute_local[(3)] = 0.000000e+00f;
  compute_local[(7)] = 0.000000e+00f;
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 32) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        pad_temp_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder[(((((((int)blockIdx.y) * 224) + ((((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) >> 3) * 56)) + (((int)blockIdx.x) * 8)) + (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) & 7)))];
      }
    }
  }
  if ((((((int)threadIdx.z) * 2) + ((int)threadIdx.x)) + ((int)threadIdx.y)) < 64) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 2) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[((((((int)threadIdx.z) * 2) + ((int)threadIdx.x)) + ((int)threadIdx.y)))] = placeholder1[(((((((int)blockIdx.z) * 64) + (((int)threadIdx.z) * 2)) + ((int)threadIdx.x)) + ((int)threadIdx.y)))];
      }
    }
  }
  __syncthreads();
  compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)))], placeholder_shared[(((int)threadIdx.z))], compute_local[(0)]);
  compute_local[(4)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)))], placeholder_shared[((((int)threadIdx.z) + 32))], compute_local[(4)]);
  compute_local[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 16))], placeholder_shared[(((int)threadIdx.z))], compute_local[(2)]);
  compute_local[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 16))], placeholder_shared[((((int)threadIdx.z) + 32))], compute_local[(6)]);
  compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 1))], placeholder_shared[(((int)threadIdx.z))], compute_local[(1)]);
  compute_local[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 1))], placeholder_shared[((((int)threadIdx.z) + 32))], compute_local[(5)]);
  compute_local[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 17))], placeholder_shared[(((int)threadIdx.z))], compute_local[(3)]);
  compute_local[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 17))], placeholder_shared[((((int)threadIdx.z) + 32))], compute_local[(7)]);
  compute[(((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)))] = compute_local[(0)];
  compute[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 100352))] = compute_local[(4)];
  compute[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 112))] = compute_local[(2)];
  compute[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 100464))] = compute_local[(6)];
  compute[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 1))] = compute_local[(1)];
  compute[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 100353))] = compute_local[(5)];
  compute[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 113))] = compute_local[(3)];
  compute[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 100465))] = compute_local[(7)];
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_12_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  float compute[1];
  __shared__ float pad_temp_shared[481];
  __shared__ float placeholder_shared[196];
  compute[(0)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 3; ++rc_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      if (((((((int)threadIdx.z) * 121) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 481) {
        if ((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 121) {
          pad_temp_shared[(((((((int)threadIdx.z) * 121) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))] = (((((3 <= ((((int)blockIdx.y) * 32) + (((((((int)threadIdx.z) * 121) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 13))) && (((((int)blockIdx.y) * 32) + (((((((int)threadIdx.z) * 121) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 13)) < 227)) && (3 <= ((((int)blockIdx.x) * 8) + (((((((int)threadIdx.z) * 121) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 13)))) && (((((int)blockIdx.x) * 8) + (((((((int)threadIdx.z) * 121) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 13)) < 227)) ? placeholder[(((((((rc_outer * 50176) + (((int)blockIdx.y) * 7168)) + ((((((((int)threadIdx.z) * 121) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 13) * 224)) + (((int)blockIdx.x) * 8)) + (((((((int)threadIdx.z) * 121) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 13)) - 675))] : 0.000000e+00f);
        }
      }
    }
    if (((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) / 49) + ((int)threadIdx.z)) < 4) {
      if (((((int)threadIdx.z) * 7) + (((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) / 7)) < 28) {
        if ((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) < 196) {
          if (((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) < 49) {
            placeholder_shared[((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)))] = placeholder1[((((((((int)blockIdx.z) * 588) + (((int)threadIdx.z) * 147)) + (rc_outer * 49)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)))];
          }
        }
      }
    }
    __syncthreads();
    for (int ry_inner = 0; ry_inner < 7; ++ry_inner) {
      for (int rx_inner = 0; rx_inner < 7; ++rx_inner) {
        compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((((int)threadIdx.y) * 26) + (ry_inner * 13)) + (((int)threadIdx.x) * 2)) + rx_inner))], placeholder_shared[((((((int)threadIdx.z) * 49) + (ry_inner * 7)) + rx_inner))], compute[(0)]);
      }
    }
  }
  T_relu[(((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 12544)) + (((int)blockIdx.y) * 1792)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 4)) + ((int)threadIdx.x)))] = max((compute[(0)] + placeholder2[(((((int)blockIdx.z) * 4) + ((int)threadIdx.z)))]), 0.000000e+00f);
}

extern "C" __global__ void fused_nn_avg_pool2d_add_nn_relu_2_kernel0(float* __restrict__ placeholder, float* __restrict__ T_relu, float* __restrict__ placeholder1) {
  float tensor[1];
  tensor[(0)] = 0.000000e+00f;
  for (int dh = 0; dh < 2; ++dh) {
    for (int dw = 0; dw < 2; ++dw) {
      tensor[(0)] = (tensor[(0)] + placeholder[((((((((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) / 28) * 112) + (dh * 56)) + ((((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) % 28) * 2)) + dw))]);
    }
  }
  T_relu[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)))] = max(__ocml_fma_f32(tensor[(0)], 2.500000e-01f, placeholder1[((((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) / 784))]), 0.000000e+00f);
}

extern "C" __global__ void fused_nn_avg_pool2d_kernel0(float* __restrict__ placeholder, float* __restrict__ tensor) {
  float tensor1[1];
  tensor1[(0)] = 0.000000e+00f;
  for (int dh = 0; dh < 7; ++dh) {
    for (int dw = 0; dw < 7; ++dw) {
      if (((int)threadIdx.x) < 1) {
        tensor1[(0)] = (tensor1[(0)] + placeholder[((((((int)threadIdx.x) * 49) + (dh * 7)) + dw))]);
      }
    }
  }
  if (((int)threadIdx.x) < 1) {
    tensor[(((int)threadIdx.x))] = (tensor1[(0)] * 2.040816e-02f);
  }
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_6_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  float compute[1];
  __shared__ float pad_temp_shared[384];
  __shared__ float placeholder_shared[36];
  compute[(0)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 8; ++rc_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 7; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      if (((((((int)threadIdx.x) * 7) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 96) + ((int)threadIdx.y)) < 4) {
        if (((((int)threadIdx.y) * 6) + (((((int)threadIdx.x) * 7) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) >> 4)) < 24) {
          if ((((((int)threadIdx.y) * 96) + (((int)threadIdx.x) * 7)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 384) {
            if (((((int)threadIdx.x) * 7) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 96) {
              pad_temp_shared[((((((int)threadIdx.y) * 96) + (((int)threadIdx.x) * 7)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))] = (((((1 <= ((((int)blockIdx.y) * 4) + (((((int)threadIdx.x) * 7) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) >> 4))) && (((((int)blockIdx.y) * 4) + (((((int)threadIdx.x) * 7) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) >> 4)) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 7) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 7) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) & 15)) < 29)) ? placeholder[((((((((rc_outer * 3136) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.y) * 112)) + ((((((int)threadIdx.x) * 7) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) >> 4) * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 7) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) & 15)) - 29))] : 0.000000e+00f);
            }
          }
        }
      }
    }
    if (((((int)threadIdx.x) / 9) + ((int)threadIdx.y)) < 4) {
      if (((((int)threadIdx.y) * 3) + (((int)threadIdx.x) / 3)) < 12) {
        if (((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) < 36) {
          if (((int)threadIdx.x) < 9) {
            placeholder_shared[(((((int)threadIdx.y) * 9) + ((int)threadIdx.x)))] = placeholder1[((((rc_outer * 36) + (((int)threadIdx.y) * 9)) + ((int)threadIdx.x)))];
          }
        }
      }
    }
    __syncthreads();
    for (int rc_inner = 0; rc_inner < 4; ++rc_inner) {
      for (int ry_inner = 0; ry_inner < 3; ++ry_inner) {
        for (int rx_inner = 0; rx_inner < 3; ++rx_inner) {
          compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((rc_inner * 96) + (((int)threadIdx.y) * 16)) + (ry_inner * 16)) + ((int)threadIdx.x)) + rx_inner))], placeholder_shared[((((rc_inner * 9) + (ry_inner * 3)) + rx_inner))], compute[(0)]);
        }
      }
    }
  }
  T_relu[(((((((int)blockIdx.y) * 112) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 14)) + ((int)threadIdx.x)))] = max((compute[(0)] + placeholder2[(0)]), 0.000000e+00f);
}

extern "C" __global__ void fused_nn_max_pool2d_add_nn_relu_kernel0(float* __restrict__ placeholder, float* __restrict__ T_relu, float* __restrict__ placeholder1) {
  float tensor[1];
  tensor[(0)] = -3.402823e+38f;
  for (int dh = 0; dh < 3; ++dh) {
    for (int dw = 0; dw < 3; ++dw) {
      tensor[(0)] = max(tensor[(0)], (((1 <= ((((((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) % 3136) / 56) * 2) + dh)) && (1 <= (((((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) % 56) * 2) + dw))) ? placeholder[(((((((((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) / 56) * 224) + (dh * 112)) + ((((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) % 56) * 2)) + dw) - 113))] : -3.402823e+38f));
    }
  }
  T_relu[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)))] = max((tensor[(0)] + placeholder1[((((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) / 3136))]), 0.000000e+00f);
}

