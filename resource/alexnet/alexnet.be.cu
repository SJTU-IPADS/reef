#include <hip/hip_runtime.h>

__device__ void fused_nn_max_pool2d_kernel0_device(float* __restrict__ placeholder, float* __restrict__ tensor){
  float tensor_local[1];
  tensor_local[(0)] = -3.402823e+38f;
  for (int dh = 0; dh < 3; ++dh) {
    for (int dw = 0; dw < 3; ++dw) {
      tensor_local[(0)] = max(tensor_local[(0)], placeholder[(((((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 25) * 144) + (((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 25) / 5) * 24)) + (dh * 12)) + ((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 5) * 2)) + dw))]);
    }
  }
  tensor[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)))] = tensor_local[(0)];
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

__device__ void fused_nn_conv2d_nn_relu_3_kernel0_device(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu){
  float compute[2];
  __shared__ float pad_temp_shared[36];
  __shared__ float placeholder_shared[200];
  for (int ff_init = 0; ff_init < 2; ++ff_init) {
    compute[(ff_init)] = 0.000000e+00f;
  }
  for (int rc_outer = 0; rc_outer < 96; ++rc_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 3; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      if (((((((int)threadIdx.z) * 9) + (((int)threadIdx.y) * 5)) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 36) {
        if ((((((int)threadIdx.y) * 5) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 9) {
          if (((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 5) {
            pad_temp_shared[(((((((int)threadIdx.z) * 9) + (((int)threadIdx.y) * 5)) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))] = (((((2 <= ((((int)blockIdx.y) * 2) + (((((((int)threadIdx.z) * 9) + (((int)threadIdx.y) * 5)) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 6))) && (((((int)blockIdx.y) * 2) + (((((((int)threadIdx.z) * 9) + (((int)threadIdx.y) * 5)) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 6)) < 28)) && (2 <= ((((int)blockIdx.x) * 2) + (((((((int)threadIdx.z) * 9) + (((int)threadIdx.y) * 5)) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 6)))) && (((((int)blockIdx.x) * 2) + (((((((int)threadIdx.z) * 9) + (((int)threadIdx.y) * 5)) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 6)) < 28)) ? placeholder[(((((((rc_outer * 676) + (((int)blockIdx.y) * 52)) + ((((((((int)threadIdx.z) * 9) + (((int)threadIdx.y) * 5)) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 6) * 26)) + (((int)blockIdx.x) * 2)) + (((((((int)threadIdx.z) * 9) + (((int)threadIdx.y) * 5)) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 6)) - 54))] : 0.000000e+00f);
          }
        }
      }
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 < 13; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) {
      if ((((((int)threadIdx.z) * 2) + (((((int)threadIdx.x) * 13) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) / 25)) + ((int)threadIdx.y)) < 8) {
        if ((((((int)threadIdx.z) * 10) + (((int)threadIdx.y) * 5)) + (((((int)threadIdx.x) * 13) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) / 5)) < 40) {
          if (((((((int)threadIdx.z) * 50) + (((int)threadIdx.y) * 25)) + (((int)threadIdx.x) * 13)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) < 200) {
            if ((((((int)threadIdx.y) * 25) + (((int)threadIdx.x) * 13)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) < 50) {
              if (((((int)threadIdx.x) * 13) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) < 25) {
                placeholder_shared[(((((((int)threadIdx.z) * 50) + (((int)threadIdx.y) * 25)) + (((int)threadIdx.x) * 13)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1))] = placeholder1[(((((((((int)blockIdx.z) * 19200) + (((int)threadIdx.z) * 4800)) + (((int)threadIdx.y) * 2400)) + (rc_outer * 25)) + (((int)threadIdx.x) * 13)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1))];
              }
            }
          }
        }
      }
    }
    __syncthreads();
    for (int ry_inner = 0; ry_inner < 5; ++ry_inner) {
      for (int rx_inner = 0; rx_inner < 5; ++rx_inner) {
        for (int ff = 0; ff < 2; ++ff) {
          compute[(ff)] = __ocml_fma_f32(pad_temp_shared[(((((((int)threadIdx.y) * 6) + (ry_inner * 6)) + ((int)threadIdx.x)) + rx_inner))], placeholder_shared[(((((((int)threadIdx.z) * 50) + (ff * 25)) + (ry_inner * 5)) + rx_inner))], compute[(ff)]);
        }
      }
    }
  }
  for (int ax1_inner_inner_inner = 0; ax1_inner_inner_inner < 2; ++ax1_inner_inner_inner) {
    T_relu[((((((((((int)blockIdx.z) * 5408) + (((int)threadIdx.z) * 1352)) + (ax1_inner_inner_inner * 676)) + (((int)blockIdx.y) * 52)) + (((int)threadIdx.y) * 26)) + (((int)blockIdx.x) * 2)) + ((int)threadIdx.x)))] = max(compute[(ax1_inner_inner_inner)], 0.000000e+00f);
  }
}

__device__ void fused_nn_conv2d_nn_relu_1_kernel0_device(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu){
  float compute[1];
  __shared__ float pad_temp_shared[192];
  __shared__ float placeholder_shared[2304];
  compute[(0)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 48; ++rc_outer) {
    __syncthreads();
    if (((((((int)threadIdx.y) * 3) + ((int)threadIdx.x)) / 6) + ((int)threadIdx.z)) < 32) {
      if ((((((int)threadIdx.z) * 6) + (((int)threadIdx.y) * 3)) + ((int)threadIdx.x)) < 192) {
        if (((((int)threadIdx.y) * 3) + ((int)threadIdx.x)) < 6) {
          if (((int)threadIdx.x) < 3) {
            pad_temp_shared[((((((int)threadIdx.z) * 6) + (((int)threadIdx.y) * 3)) + ((int)threadIdx.x)))] = (((((1 <= ((((int)blockIdx.y) * 2) + (((int)threadIdx.z) & 3))) && (((((int)blockIdx.y) * 2) + (((int)threadIdx.z) & 3)) < 13)) && (1 <= (((((int)blockIdx.x) * 4) + (((int)threadIdx.y) * 3)) + ((int)threadIdx.x)))) && ((((((int)blockIdx.x) * 4) + (((int)threadIdx.y) * 3)) + ((int)threadIdx.x)) < 13)) ? placeholder[(((((((((rc_outer * 1152) + ((((int)threadIdx.z) >> 2) * 144)) + (((int)blockIdx.y) * 24)) + ((((int)threadIdx.z) & 3) * 12)) + (((int)blockIdx.x) * 4)) + (((int)threadIdx.y) * 3)) + ((int)threadIdx.x)) - 13))] : 0.000000e+00f);
          }
        }
      }
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 9; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      placeholder_shared[(((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 36)) + (((int)threadIdx.x) * 9)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))] = placeholder1[(((((((((int)blockIdx.z) * 110592) + (((int)threadIdx.z) * 3456)) + (rc_outer * 72)) + (((int)threadIdx.y) * 36)) + (((int)threadIdx.x) * 9)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))];
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
  T_relu[(((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 144)) + (((int)blockIdx.y) * 24)) + (((int)threadIdx.y) * 12)) + (((int)blockIdx.x) * 4)) + ((int)threadIdx.x)))] = max(compute[(0)], 0.000000e+00f);
}

__device__ void fused_nn_dense_add_kernel0_device(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_add, float* __restrict__ placeholder2){
  float T_dense_rf[1];
  float red_buf0[1];
  __shared__ float T_dense[1];
  T_dense_rf[(0)] = 0.000000e+00f;
  for (int k_outer = 0; k_outer < 64; ++k_outer) {
    T_dense_rf[(0)] = __ocml_fma_f32(placeholder[(((k_outer * 64) + ((int)threadIdx.x)))], placeholder1[((((((int)blockIdx.x) * 4096) + (k_outer * 64)) + ((int)threadIdx.x)))], T_dense_rf[(0)]);
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

__device__ void fused_nn_max_pool2d_2_kernel0_device(float* __restrict__ placeholder, float* __restrict__ tensor){
  float tensor_local[1];
  tensor_local[(0)] = -3.402823e+38f;
  for (int dh = 0; dh < 3; ++dh) {
    for (int dw = 0; dw < 3; ++dw) {
      tensor_local[(0)] = max(tensor_local[(0)], placeholder[(((((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 676) * 2916) + (((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 676) / 26) * 108)) + (dh * 54)) + ((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 26) * 2)) + dw))]);
    }
  }
  tensor[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)))] = tensor_local[(0)];
}

__device__ void fused_nn_dense_add_nn_relu_1_kernel0_device(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2){
  float T_dense_rf[1];
  float red_buf0[1];
  __shared__ float T_dense[1];
  T_dense_rf[(0)] = 0.000000e+00f;
  for (int k_outer = 0; k_outer < 100; ++k_outer) {
    T_dense_rf[(0)] = __ocml_fma_f32(placeholder[(((k_outer * 64) + ((int)threadIdx.x)))], placeholder1[((((((int)blockIdx.x) * 6400) + (k_outer * 64)) + ((int)threadIdx.x)))], T_dense_rf[(0)]);
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
    T_relu[(((int)blockIdx.x))] = max((T_dense[(0)] + placeholder2[(((int)blockIdx.x))]), 0.000000e+00f);
  }
}

__device__ void fused_nn_conv2d_nn_relu_4_kernel0_device(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu){
  float compute[1];
  __shared__ float pad_temp_shared[315];
  __shared__ float placeholder_shared[24];
  compute[(0)] = 0.000000e+00f;
  for (int ry_outer = 0; ry_outer < 11; ++ry_outer) {
    __syncthreads();
    if (((((int)threadIdx.z) * 8) + (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) / 5)) < 63) {
      if ((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) < 315) {
        pad_temp_shared[((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)))] = placeholder[((((((((((((int)threadIdx.z) * 8) + (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) / 5)) / 21) * 50176) + (((int)blockIdx.y) * 5376)) + (ry_outer * 224)) + ((((((int)threadIdx.z) * 8) + (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) / 5)) % 21) * 224)) + (((int)blockIdx.x) * 8)) + (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) % 5)))];
      }
    }
    if (((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 1) / 5)) < 63) {
      if ((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) < 314) {
        if (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) < 39) {
          pad_temp_shared[(((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) + 1))] = placeholder[((((((((((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 1) / 5)) / 21) * 50176) + (((int)blockIdx.y) * 5376)) + (ry_outer * 224)) + ((((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 1) / 5)) % 21) * 224)) + (((int)blockIdx.x) * 8)) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 1) % 5)))];
        }
      }
    }
    if (((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 2) / 5)) < 63) {
      if ((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) < 313) {
        if (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) < 38) {
          pad_temp_shared[(((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) + 2))] = placeholder[((((((((((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 2) / 5)) / 21) * 50176) + (((int)blockIdx.y) * 5376)) + (ry_outer * 224)) + ((((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 2) / 5)) % 21) * 224)) + (((int)blockIdx.x) * 8)) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 2) % 5)))];
        }
      }
    }
    if (((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 3) / 5)) < 63) {
      if ((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) < 312) {
        if (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) < 37) {
          if (((int)threadIdx.x) < 1) {
            pad_temp_shared[(((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) + 3))] = placeholder[((((((((((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 3) / 5)) / 21) * 50176) + (((int)blockIdx.y) * 5376)) + (ry_outer * 224)) + ((((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 3) / 5)) % 21) * 224)) + (((int)blockIdx.x) * 8)) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 3) % 5)))];
          }
        }
      }
    }
    if ((((((int)threadIdx.x) + ((int)threadIdx.y)) / 3) + ((int)threadIdx.z)) < 8) {
      if ((((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) + ((int)threadIdx.y)) < 24) {
        if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 3) {
          if (((int)threadIdx.x) < 1) {
            placeholder_shared[((((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) + ((int)threadIdx.y)))] = placeholder1[((((((((int)blockIdx.z) * 2904) + (((int)threadIdx.z) * 363)) + (((int)threadIdx.x) * 121)) + (((int)threadIdx.y) * 121)) + (ry_outer * 11)))];
          }
        }
      }
    }
    __syncthreads();
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 4)))], placeholder_shared[((((int)threadIdx.z) * 3))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 4)) + 105))], placeholder_shared[(((((int)threadIdx.z) * 3) + 1))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 4)) + 210))], placeholder_shared[(((((int)threadIdx.z) * 3) + 2))], compute[(0)]);
    __syncthreads();
    if (((((int)threadIdx.z) * 8) + (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) / 5)) < 63) {
      if ((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) < 315) {
        pad_temp_shared[((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)))] = placeholder[(((((((((((((int)threadIdx.z) * 8) + (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) / 5)) / 21) * 50176) + (((int)blockIdx.y) * 5376)) + (ry_outer * 224)) + ((((((int)threadIdx.z) * 8) + (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) / 5)) % 21) * 224)) + (((int)blockIdx.x) * 8)) + (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) % 5)) + 1))];
      }
    }
    if (((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 1) / 5)) < 63) {
      if ((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) < 314) {
        if (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) < 39) {
          pad_temp_shared[(((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) + 1))] = placeholder[(((((((((((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 1) / 5)) / 21) * 50176) + (((int)blockIdx.y) * 5376)) + (ry_outer * 224)) + ((((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 1) / 5)) % 21) * 224)) + (((int)blockIdx.x) * 8)) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 1) % 5)) + 1))];
        }
      }
    }
    if (((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 2) / 5)) < 63) {
      if ((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) < 313) {
        if (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) < 38) {
          pad_temp_shared[(((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) + 2))] = placeholder[(((((((((((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 2) / 5)) / 21) * 50176) + (((int)blockIdx.y) * 5376)) + (ry_outer * 224)) + ((((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 2) / 5)) % 21) * 224)) + (((int)blockIdx.x) * 8)) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 2) % 5)) + 1))];
        }
      }
    }
    if (((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 3) / 5)) < 63) {
      if ((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) < 312) {
        if (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) < 37) {
          if (((int)threadIdx.x) < 1) {
            pad_temp_shared[(((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) + 3))] = placeholder[(((((((((((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 3) / 5)) / 21) * 50176) + (((int)blockIdx.y) * 5376)) + (ry_outer * 224)) + ((((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 3) / 5)) % 21) * 224)) + (((int)blockIdx.x) * 8)) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 3) % 5)) + 1))];
          }
        }
      }
    }
    if ((((((int)threadIdx.x) + ((int)threadIdx.y)) / 3) + ((int)threadIdx.z)) < 8) {
      if ((((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) + ((int)threadIdx.y)) < 24) {
        if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 3) {
          if (((int)threadIdx.x) < 1) {
            placeholder_shared[((((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) + ((int)threadIdx.y)))] = placeholder1[(((((((((int)blockIdx.z) * 2904) + (((int)threadIdx.z) * 363)) + (((int)threadIdx.x) * 121)) + (((int)threadIdx.y) * 121)) + (ry_outer * 11)) + 1))];
          }
        }
      }
    }
    __syncthreads();
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 4)))], placeholder_shared[((((int)threadIdx.z) * 3))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 4)) + 105))], placeholder_shared[(((((int)threadIdx.z) * 3) + 1))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 4)) + 210))], placeholder_shared[(((((int)threadIdx.z) * 3) + 2))], compute[(0)]);
    __syncthreads();
    if (((((int)threadIdx.z) * 8) + (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) / 5)) < 63) {
      if ((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) < 315) {
        pad_temp_shared[((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)))] = placeholder[(((((((((((((int)threadIdx.z) * 8) + (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) / 5)) / 21) * 50176) + (((int)blockIdx.y) * 5376)) + (ry_outer * 224)) + ((((((int)threadIdx.z) * 8) + (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) / 5)) % 21) * 224)) + (((int)blockIdx.x) * 8)) + (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) % 5)) + 2))];
      }
    }
    if (((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 1) / 5)) < 63) {
      if ((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) < 314) {
        if (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) < 39) {
          pad_temp_shared[(((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) + 1))] = placeholder[(((((((((((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 1) / 5)) / 21) * 50176) + (((int)blockIdx.y) * 5376)) + (ry_outer * 224)) + ((((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 1) / 5)) % 21) * 224)) + (((int)blockIdx.x) * 8)) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 1) % 5)) + 2))];
        }
      }
    }
    if (((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 2) / 5)) < 63) {
      if ((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) < 313) {
        if (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) < 38) {
          pad_temp_shared[(((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) + 2))] = placeholder[(((((((((((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 2) / 5)) / 21) * 50176) + (((int)blockIdx.y) * 5376)) + (ry_outer * 224)) + ((((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 2) / 5)) % 21) * 224)) + (((int)blockIdx.x) * 8)) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 2) % 5)) + 2))];
        }
      }
    }
    if (((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 3) / 5)) < 63) {
      if ((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) < 312) {
        if (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) < 37) {
          if (((int)threadIdx.x) < 1) {
            pad_temp_shared[(((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) + 3))] = placeholder[(((((((((((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 3) / 5)) / 21) * 50176) + (((int)blockIdx.y) * 5376)) + (ry_outer * 224)) + ((((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 3) / 5)) % 21) * 224)) + (((int)blockIdx.x) * 8)) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 3) % 5)) + 2))];
          }
        }
      }
    }
    if ((((((int)threadIdx.x) + ((int)threadIdx.y)) / 3) + ((int)threadIdx.z)) < 8) {
      if ((((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) + ((int)threadIdx.y)) < 24) {
        if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 3) {
          if (((int)threadIdx.x) < 1) {
            placeholder_shared[((((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) + ((int)threadIdx.y)))] = placeholder1[(((((((((int)blockIdx.z) * 2904) + (((int)threadIdx.z) * 363)) + (((int)threadIdx.x) * 121)) + (((int)threadIdx.y) * 121)) + (ry_outer * 11)) + 2))];
          }
        }
      }
    }
    __syncthreads();
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 4)))], placeholder_shared[((((int)threadIdx.z) * 3))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 4)) + 105))], placeholder_shared[(((((int)threadIdx.z) * 3) + 1))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 4)) + 210))], placeholder_shared[(((((int)threadIdx.z) * 3) + 2))], compute[(0)]);
    __syncthreads();
    if (((((int)threadIdx.z) * 8) + (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) / 5)) < 63) {
      if ((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) < 315) {
        pad_temp_shared[((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)))] = placeholder[(((((((((((((int)threadIdx.z) * 8) + (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) / 5)) / 21) * 50176) + (((int)blockIdx.y) * 5376)) + (ry_outer * 224)) + ((((((int)threadIdx.z) * 8) + (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) / 5)) % 21) * 224)) + (((int)blockIdx.x) * 8)) + (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) % 5)) + 3))];
      }
    }
    if (((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 1) / 5)) < 63) {
      if ((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) < 314) {
        if (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) < 39) {
          pad_temp_shared[(((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) + 1))] = placeholder[(((((((((((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 1) / 5)) / 21) * 50176) + (((int)blockIdx.y) * 5376)) + (ry_outer * 224)) + ((((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 1) / 5)) % 21) * 224)) + (((int)blockIdx.x) * 8)) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 1) % 5)) + 3))];
        }
      }
    }
    if (((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 2) / 5)) < 63) {
      if ((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) < 313) {
        if (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) < 38) {
          pad_temp_shared[(((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) + 2))] = placeholder[(((((((((((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 2) / 5)) / 21) * 50176) + (((int)blockIdx.y) * 5376)) + (ry_outer * 224)) + ((((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 2) / 5)) % 21) * 224)) + (((int)blockIdx.x) * 8)) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 2) % 5)) + 3))];
        }
      }
    }
    if (((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 3) / 5)) < 63) {
      if ((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) < 312) {
        if (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) < 37) {
          if (((int)threadIdx.x) < 1) {
            pad_temp_shared[(((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) + 3))] = placeholder[(((((((((((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 3) / 5)) / 21) * 50176) + (((int)blockIdx.y) * 5376)) + (ry_outer * 224)) + ((((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 3) / 5)) % 21) * 224)) + (((int)blockIdx.x) * 8)) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 3) % 5)) + 3))];
          }
        }
      }
    }
    if ((((((int)threadIdx.x) + ((int)threadIdx.y)) / 3) + ((int)threadIdx.z)) < 8) {
      if ((((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) + ((int)threadIdx.y)) < 24) {
        if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 3) {
          if (((int)threadIdx.x) < 1) {
            placeholder_shared[((((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) + ((int)threadIdx.y)))] = placeholder1[(((((((((int)blockIdx.z) * 2904) + (((int)threadIdx.z) * 363)) + (((int)threadIdx.x) * 121)) + (((int)threadIdx.y) * 121)) + (ry_outer * 11)) + 3))];
          }
        }
      }
    }
    __syncthreads();
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 4)))], placeholder_shared[((((int)threadIdx.z) * 3))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 4)) + 105))], placeholder_shared[(((((int)threadIdx.z) * 3) + 1))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 4)) + 210))], placeholder_shared[(((((int)threadIdx.z) * 3) + 2))], compute[(0)]);
    __syncthreads();
    if (((((int)threadIdx.z) * 8) + (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) / 5)) < 63) {
      if ((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) < 315) {
        pad_temp_shared[((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)))] = placeholder[(((((((((((((int)threadIdx.z) * 8) + (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) / 5)) / 21) * 50176) + (((int)blockIdx.y) * 5376)) + (ry_outer * 224)) + ((((((int)threadIdx.z) * 8) + (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) / 5)) % 21) * 224)) + (((int)blockIdx.x) * 8)) + (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) % 5)) + 4))];
      }
    }
    if (((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 1) / 5)) < 63) {
      if ((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) < 314) {
        if (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) < 39) {
          pad_temp_shared[(((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) + 1))] = placeholder[(((((((((((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 1) / 5)) / 21) * 50176) + (((int)blockIdx.y) * 5376)) + (ry_outer * 224)) + ((((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 1) / 5)) % 21) * 224)) + (((int)blockIdx.x) * 8)) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 1) % 5)) + 4))];
        }
      }
    }
    if (((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 2) / 5)) < 63) {
      if ((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) < 313) {
        if (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) < 38) {
          pad_temp_shared[(((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) + 2))] = placeholder[(((((((((((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 2) / 5)) / 21) * 50176) + (((int)blockIdx.y) * 5376)) + (ry_outer * 224)) + ((((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 2) / 5)) % 21) * 224)) + (((int)blockIdx.x) * 8)) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 2) % 5)) + 4))];
        }
      }
    }
    if (((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 3) / 5)) < 63) {
      if ((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) < 312) {
        if (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) < 37) {
          if (((int)threadIdx.x) < 1) {
            pad_temp_shared[(((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) + 3))] = placeholder[(((((((((((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 3) / 5)) / 21) * 50176) + (((int)blockIdx.y) * 5376)) + (ry_outer * 224)) + ((((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 3) / 5)) % 21) * 224)) + (((int)blockIdx.x) * 8)) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 3) % 5)) + 4))];
          }
        }
      }
    }
    if ((((((int)threadIdx.x) + ((int)threadIdx.y)) / 3) + ((int)threadIdx.z)) < 8) {
      if ((((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) + ((int)threadIdx.y)) < 24) {
        if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 3) {
          if (((int)threadIdx.x) < 1) {
            placeholder_shared[((((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) + ((int)threadIdx.y)))] = placeholder1[(((((((((int)blockIdx.z) * 2904) + (((int)threadIdx.z) * 363)) + (((int)threadIdx.x) * 121)) + (((int)threadIdx.y) * 121)) + (ry_outer * 11)) + 4))];
          }
        }
      }
    }
    __syncthreads();
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 4)))], placeholder_shared[((((int)threadIdx.z) * 3))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 4)) + 105))], placeholder_shared[(((((int)threadIdx.z) * 3) + 1))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 4)) + 210))], placeholder_shared[(((((int)threadIdx.z) * 3) + 2))], compute[(0)]);
    __syncthreads();
    if (((((int)threadIdx.z) * 8) + (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) / 5)) < 63) {
      if ((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) < 315) {
        pad_temp_shared[((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)))] = placeholder[(((((((((((((int)threadIdx.z) * 8) + (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) / 5)) / 21) * 50176) + (((int)blockIdx.y) * 5376)) + (ry_outer * 224)) + ((((((int)threadIdx.z) * 8) + (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) / 5)) % 21) * 224)) + (((int)blockIdx.x) * 8)) + (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) % 5)) + 5))];
      }
    }
    if (((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 1) / 5)) < 63) {
      if ((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) < 314) {
        if (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) < 39) {
          pad_temp_shared[(((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) + 1))] = placeholder[(((((((((((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 1) / 5)) / 21) * 50176) + (((int)blockIdx.y) * 5376)) + (ry_outer * 224)) + ((((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 1) / 5)) % 21) * 224)) + (((int)blockIdx.x) * 8)) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 1) % 5)) + 5))];
        }
      }
    }
    if (((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 2) / 5)) < 63) {
      if ((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) < 313) {
        if (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) < 38) {
          pad_temp_shared[(((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) + 2))] = placeholder[(((((((((((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 2) / 5)) / 21) * 50176) + (((int)blockIdx.y) * 5376)) + (ry_outer * 224)) + ((((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 2) / 5)) % 21) * 224)) + (((int)blockIdx.x) * 8)) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 2) % 5)) + 5))];
        }
      }
    }
    if (((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 3) / 5)) < 63) {
      if ((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) < 312) {
        if (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) < 37) {
          if (((int)threadIdx.x) < 1) {
            pad_temp_shared[(((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) + 3))] = placeholder[(((((((((((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 3) / 5)) / 21) * 50176) + (((int)blockIdx.y) * 5376)) + (ry_outer * 224)) + ((((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 3) / 5)) % 21) * 224)) + (((int)blockIdx.x) * 8)) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 3) % 5)) + 5))];
          }
        }
      }
    }
    if ((((((int)threadIdx.x) + ((int)threadIdx.y)) / 3) + ((int)threadIdx.z)) < 8) {
      if ((((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) + ((int)threadIdx.y)) < 24) {
        if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 3) {
          if (((int)threadIdx.x) < 1) {
            placeholder_shared[((((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) + ((int)threadIdx.y)))] = placeholder1[(((((((((int)blockIdx.z) * 2904) + (((int)threadIdx.z) * 363)) + (((int)threadIdx.x) * 121)) + (((int)threadIdx.y) * 121)) + (ry_outer * 11)) + 5))];
          }
        }
      }
    }
    __syncthreads();
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 4)))], placeholder_shared[((((int)threadIdx.z) * 3))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 4)) + 105))], placeholder_shared[(((((int)threadIdx.z) * 3) + 1))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 4)) + 210))], placeholder_shared[(((((int)threadIdx.z) * 3) + 2))], compute[(0)]);
    __syncthreads();
    if (((((int)threadIdx.z) * 8) + (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) / 5)) < 63) {
      if ((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) < 315) {
        pad_temp_shared[((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)))] = placeholder[(((((((((((((int)threadIdx.z) * 8) + (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) / 5)) / 21) * 50176) + (((int)blockIdx.y) * 5376)) + (ry_outer * 224)) + ((((((int)threadIdx.z) * 8) + (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) / 5)) % 21) * 224)) + (((int)blockIdx.x) * 8)) + (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) % 5)) + 6))];
      }
    }
    if (((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 1) / 5)) < 63) {
      if ((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) < 314) {
        if (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) < 39) {
          pad_temp_shared[(((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) + 1))] = placeholder[(((((((((((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 1) / 5)) / 21) * 50176) + (((int)blockIdx.y) * 5376)) + (ry_outer * 224)) + ((((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 1) / 5)) % 21) * 224)) + (((int)blockIdx.x) * 8)) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 1) % 5)) + 6))];
        }
      }
    }
    if (((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 2) / 5)) < 63) {
      if ((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) < 313) {
        if (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) < 38) {
          pad_temp_shared[(((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) + 2))] = placeholder[(((((((((((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 2) / 5)) / 21) * 50176) + (((int)blockIdx.y) * 5376)) + (ry_outer * 224)) + ((((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 2) / 5)) % 21) * 224)) + (((int)blockIdx.x) * 8)) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 2) % 5)) + 6))];
        }
      }
    }
    if (((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 3) / 5)) < 63) {
      if ((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) < 312) {
        if (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) < 37) {
          if (((int)threadIdx.x) < 1) {
            pad_temp_shared[(((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) + 3))] = placeholder[(((((((((((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 3) / 5)) / 21) * 50176) + (((int)blockIdx.y) * 5376)) + (ry_outer * 224)) + ((((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 3) / 5)) % 21) * 224)) + (((int)blockIdx.x) * 8)) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 3) % 5)) + 6))];
          }
        }
      }
    }
    if ((((((int)threadIdx.x) + ((int)threadIdx.y)) / 3) + ((int)threadIdx.z)) < 8) {
      if ((((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) + ((int)threadIdx.y)) < 24) {
        if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 3) {
          if (((int)threadIdx.x) < 1) {
            placeholder_shared[((((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) + ((int)threadIdx.y)))] = placeholder1[(((((((((int)blockIdx.z) * 2904) + (((int)threadIdx.z) * 363)) + (((int)threadIdx.x) * 121)) + (((int)threadIdx.y) * 121)) + (ry_outer * 11)) + 6))];
          }
        }
      }
    }
    __syncthreads();
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 4)))], placeholder_shared[((((int)threadIdx.z) * 3))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 4)) + 105))], placeholder_shared[(((((int)threadIdx.z) * 3) + 1))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 4)) + 210))], placeholder_shared[(((((int)threadIdx.z) * 3) + 2))], compute[(0)]);
    __syncthreads();
    if (((((int)threadIdx.z) * 8) + (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) / 5)) < 63) {
      if ((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) < 315) {
        pad_temp_shared[((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)))] = placeholder[(((((((((((((int)threadIdx.z) * 8) + (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) / 5)) / 21) * 50176) + (((int)blockIdx.y) * 5376)) + (ry_outer * 224)) + ((((((int)threadIdx.z) * 8) + (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) / 5)) % 21) * 224)) + (((int)blockIdx.x) * 8)) + (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) % 5)) + 7))];
      }
    }
    if (((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 1) / 5)) < 63) {
      if ((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) < 314) {
        if (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) < 39) {
          pad_temp_shared[(((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) + 1))] = placeholder[(((((((((((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 1) / 5)) / 21) * 50176) + (((int)blockIdx.y) * 5376)) + (ry_outer * 224)) + ((((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 1) / 5)) % 21) * 224)) + (((int)blockIdx.x) * 8)) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 1) % 5)) + 7))];
        }
      }
    }
    if (((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 2) / 5)) < 63) {
      if ((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) < 313) {
        if (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) < 38) {
          pad_temp_shared[(((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) + 2))] = placeholder[(((((((((((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 2) / 5)) / 21) * 50176) + (((int)blockIdx.y) * 5376)) + (ry_outer * 224)) + ((((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 2) / 5)) % 21) * 224)) + (((int)blockIdx.x) * 8)) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 2) % 5)) + 7))];
        }
      }
    }
    if (((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 3) / 5)) < 63) {
      if ((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) < 312) {
        if (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) < 37) {
          if (((int)threadIdx.x) < 1) {
            pad_temp_shared[(((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) + 3))] = placeholder[(((((((((((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 3) / 5)) / 21) * 50176) + (((int)blockIdx.y) * 5376)) + (ry_outer * 224)) + ((((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 3) / 5)) % 21) * 224)) + (((int)blockIdx.x) * 8)) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 3) % 5)) + 7))];
          }
        }
      }
    }
    if ((((((int)threadIdx.x) + ((int)threadIdx.y)) / 3) + ((int)threadIdx.z)) < 8) {
      if ((((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) + ((int)threadIdx.y)) < 24) {
        if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 3) {
          if (((int)threadIdx.x) < 1) {
            placeholder_shared[((((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) + ((int)threadIdx.y)))] = placeholder1[(((((((((int)blockIdx.z) * 2904) + (((int)threadIdx.z) * 363)) + (((int)threadIdx.x) * 121)) + (((int)threadIdx.y) * 121)) + (ry_outer * 11)) + 7))];
          }
        }
      }
    }
    __syncthreads();
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 4)))], placeholder_shared[((((int)threadIdx.z) * 3))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 4)) + 105))], placeholder_shared[(((((int)threadIdx.z) * 3) + 1))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 4)) + 210))], placeholder_shared[(((((int)threadIdx.z) * 3) + 2))], compute[(0)]);
    __syncthreads();
    if (((((int)threadIdx.z) * 8) + (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) / 5)) < 63) {
      if ((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) < 315) {
        pad_temp_shared[((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)))] = placeholder[(((((((((((((int)threadIdx.z) * 8) + (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) / 5)) / 21) * 50176) + (((int)blockIdx.y) * 5376)) + (ry_outer * 224)) + ((((((int)threadIdx.z) * 8) + (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) / 5)) % 21) * 224)) + (((int)blockIdx.x) * 8)) + (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) % 5)) + 8))];
      }
    }
    if (((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 1) / 5)) < 63) {
      if ((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) < 314) {
        if (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) < 39) {
          pad_temp_shared[(((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) + 1))] = placeholder[(((((((((((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 1) / 5)) / 21) * 50176) + (((int)blockIdx.y) * 5376)) + (ry_outer * 224)) + ((((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 1) / 5)) % 21) * 224)) + (((int)blockIdx.x) * 8)) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 1) % 5)) + 8))];
        }
      }
    }
    if (((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 2) / 5)) < 63) {
      if ((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) < 313) {
        if (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) < 38) {
          pad_temp_shared[(((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) + 2))] = placeholder[(((((((((((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 2) / 5)) / 21) * 50176) + (((int)blockIdx.y) * 5376)) + (ry_outer * 224)) + ((((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 2) / 5)) % 21) * 224)) + (((int)blockIdx.x) * 8)) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 2) % 5)) + 8))];
        }
      }
    }
    if (((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 3) / 5)) < 63) {
      if ((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) < 312) {
        if (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) < 37) {
          if (((int)threadIdx.x) < 1) {
            pad_temp_shared[(((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) + 3))] = placeholder[(((((((((((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 3) / 5)) / 21) * 50176) + (((int)blockIdx.y) * 5376)) + (ry_outer * 224)) + ((((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 3) / 5)) % 21) * 224)) + (((int)blockIdx.x) * 8)) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 3) % 5)) + 8))];
          }
        }
      }
    }
    if ((((((int)threadIdx.x) + ((int)threadIdx.y)) / 3) + ((int)threadIdx.z)) < 8) {
      if ((((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) + ((int)threadIdx.y)) < 24) {
        if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 3) {
          if (((int)threadIdx.x) < 1) {
            placeholder_shared[((((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) + ((int)threadIdx.y)))] = placeholder1[(((((((((int)blockIdx.z) * 2904) + (((int)threadIdx.z) * 363)) + (((int)threadIdx.x) * 121)) + (((int)threadIdx.y) * 121)) + (ry_outer * 11)) + 8))];
          }
        }
      }
    }
    __syncthreads();
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 4)))], placeholder_shared[((((int)threadIdx.z) * 3))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 4)) + 105))], placeholder_shared[(((((int)threadIdx.z) * 3) + 1))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 4)) + 210))], placeholder_shared[(((((int)threadIdx.z) * 3) + 2))], compute[(0)]);
    __syncthreads();
    if (((((int)threadIdx.z) * 8) + (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) / 5)) < 63) {
      if ((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) < 315) {
        pad_temp_shared[((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)))] = placeholder[(((((((((((((int)threadIdx.z) * 8) + (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) / 5)) / 21) * 50176) + (((int)blockIdx.y) * 5376)) + (ry_outer * 224)) + ((((((int)threadIdx.z) * 8) + (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) / 5)) % 21) * 224)) + (((int)blockIdx.x) * 8)) + (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) % 5)) + 9))];
      }
    }
    if (((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 1) / 5)) < 63) {
      if ((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) < 314) {
        if (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) < 39) {
          pad_temp_shared[(((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) + 1))] = placeholder[(((((((((((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 1) / 5)) / 21) * 50176) + (((int)blockIdx.y) * 5376)) + (ry_outer * 224)) + ((((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 1) / 5)) % 21) * 224)) + (((int)blockIdx.x) * 8)) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 1) % 5)) + 9))];
        }
      }
    }
    if (((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 2) / 5)) < 63) {
      if ((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) < 313) {
        if (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) < 38) {
          pad_temp_shared[(((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) + 2))] = placeholder[(((((((((((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 2) / 5)) / 21) * 50176) + (((int)blockIdx.y) * 5376)) + (ry_outer * 224)) + ((((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 2) / 5)) % 21) * 224)) + (((int)blockIdx.x) * 8)) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 2) % 5)) + 9))];
        }
      }
    }
    if (((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 3) / 5)) < 63) {
      if ((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) < 312) {
        if (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) < 37) {
          if (((int)threadIdx.x) < 1) {
            pad_temp_shared[(((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) + 3))] = placeholder[(((((((((((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 3) / 5)) / 21) * 50176) + (((int)blockIdx.y) * 5376)) + (ry_outer * 224)) + ((((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 3) / 5)) % 21) * 224)) + (((int)blockIdx.x) * 8)) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 3) % 5)) + 9))];
          }
        }
      }
    }
    if ((((((int)threadIdx.x) + ((int)threadIdx.y)) / 3) + ((int)threadIdx.z)) < 8) {
      if ((((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) + ((int)threadIdx.y)) < 24) {
        if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 3) {
          if (((int)threadIdx.x) < 1) {
            placeholder_shared[((((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) + ((int)threadIdx.y)))] = placeholder1[(((((((((int)blockIdx.z) * 2904) + (((int)threadIdx.z) * 363)) + (((int)threadIdx.x) * 121)) + (((int)threadIdx.y) * 121)) + (ry_outer * 11)) + 9))];
          }
        }
      }
    }
    __syncthreads();
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 4)))], placeholder_shared[((((int)threadIdx.z) * 3))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 4)) + 105))], placeholder_shared[(((((int)threadIdx.z) * 3) + 1))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 4)) + 210))], placeholder_shared[(((((int)threadIdx.z) * 3) + 2))], compute[(0)]);
    __syncthreads();
    if (((((int)threadIdx.z) * 8) + (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) / 5)) < 63) {
      if ((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) < 315) {
        pad_temp_shared[((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)))] = placeholder[(((((((((((((int)threadIdx.z) * 8) + (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) / 5)) / 21) * 50176) + (((int)blockIdx.y) * 5376)) + (ry_outer * 224)) + ((((((int)threadIdx.z) * 8) + (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) / 5)) % 21) * 224)) + (((int)blockIdx.x) * 8)) + (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) % 5)) + 10))];
      }
    }
    if (((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 1) / 5)) < 63) {
      if ((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) < 314) {
        if (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) < 39) {
          pad_temp_shared[(((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) + 1))] = placeholder[(((((((((((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 1) / 5)) / 21) * 50176) + (((int)blockIdx.y) * 5376)) + (ry_outer * 224)) + ((((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 1) / 5)) % 21) * 224)) + (((int)blockIdx.x) * 8)) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 1) % 5)) + 10))];
        }
      }
    }
    if (((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 2) / 5)) < 63) {
      if ((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) < 313) {
        if (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) < 38) {
          pad_temp_shared[(((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) + 2))] = placeholder[(((((((((((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 2) / 5)) / 21) * 50176) + (((int)blockIdx.y) * 5376)) + (ry_outer * 224)) + ((((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 2) / 5)) % 21) * 224)) + (((int)blockIdx.x) * 8)) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 2) % 5)) + 10))];
        }
      }
    }
    if (((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 3) / 5)) < 63) {
      if ((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) < 312) {
        if (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) < 37) {
          if (((int)threadIdx.x) < 1) {
            pad_temp_shared[(((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) * 4)) + 3))] = placeholder[(((((((((((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 3) / 5)) / 21) * 50176) + (((int)blockIdx.y) * 5376)) + (ry_outer * 224)) + ((((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 3) / 5)) % 21) * 224)) + (((int)blockIdx.x) * 8)) + ((((((int)threadIdx.y) * 7) + (((int)threadIdx.x) * 4)) + 3) % 5)) + 10))];
          }
        }
      }
    }
    if ((((((int)threadIdx.x) + ((int)threadIdx.y)) / 3) + ((int)threadIdx.z)) < 8) {
      if ((((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) + ((int)threadIdx.y)) < 24) {
        if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 3) {
          if (((int)threadIdx.x) < 1) {
            placeholder_shared[((((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) + ((int)threadIdx.y)))] = placeholder1[(((((((((int)blockIdx.z) * 2904) + (((int)threadIdx.z) * 363)) + (((int)threadIdx.x) * 121)) + (((int)threadIdx.y) * 121)) + (ry_outer * 11)) + 10))];
          }
        }
      }
    }
    __syncthreads();
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 4)))], placeholder_shared[((((int)threadIdx.z) * 3))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 4)) + 105))], placeholder_shared[(((((int)threadIdx.z) * 3) + 1))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 4)) + 210))], placeholder_shared[(((((int)threadIdx.z) * 3) + 2))], compute[(0)]);
  }
  T_relu[(((((((((int)blockIdx.z) * 23328) + (((int)threadIdx.z) * 2916)) + (((int)blockIdx.y) * 324)) + (((int)threadIdx.y) * 54)) + (((int)blockIdx.x) * 2)) + ((int)threadIdx.x)))] = max(compute[(0)], 0.000000e+00f);
}

__device__ void fused_nn_max_pool2d_1_kernel0_device(float* __restrict__ placeholder, float* __restrict__ tensor){
  float tensor_local[1];
  tensor_local[(0)] = -3.402823e+38f;
  for (int dh = 0; dh < 3; ++dh) {
    for (int dw = 0; dw < 3; ++dw) {
      tensor_local[(0)] = max(tensor_local[(0)], placeholder[(((((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 144) * 676) + (((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 144) / 12) * 52)) + (dh * 26)) + ((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 12) * 2)) + dw))]);
    }
  }
  tensor[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)))] = tensor_local[(0)];
}

__device__ void fused_nn_conv2d_nn_relu_2_kernel0_device(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu){
  float compute[1];
  __shared__ float pad_temp_shared[192];
  __shared__ float placeholder_shared[2304];
  compute[(0)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 32; ++rc_outer) {
    __syncthreads();
    if (((((((int)threadIdx.y) * 3) + ((int)threadIdx.x)) / 6) + ((int)threadIdx.z)) < 32) {
      if ((((((int)threadIdx.z) * 6) + (((int)threadIdx.y) * 3)) + ((int)threadIdx.x)) < 192) {
        if (((((int)threadIdx.y) * 3) + ((int)threadIdx.x)) < 6) {
          if (((int)threadIdx.x) < 3) {
            pad_temp_shared[((((((int)threadIdx.z) * 6) + (((int)threadIdx.y) * 3)) + ((int)threadIdx.x)))] = (((((1 <= ((((int)blockIdx.y) * 2) + (((int)threadIdx.z) & 3))) && (((((int)blockIdx.y) * 2) + (((int)threadIdx.z) & 3)) < 13)) && (1 <= (((((int)blockIdx.x) * 4) + (((int)threadIdx.y) * 3)) + ((int)threadIdx.x)))) && ((((((int)blockIdx.x) * 4) + (((int)threadIdx.y) * 3)) + ((int)threadIdx.x)) < 13)) ? placeholder[(((((((((rc_outer * 1152) + ((((int)threadIdx.z) >> 2) * 144)) + (((int)blockIdx.y) * 24)) + ((((int)threadIdx.z) & 3) * 12)) + (((int)blockIdx.x) * 4)) + (((int)threadIdx.y) * 3)) + ((int)threadIdx.x)) - 13))] : 0.000000e+00f);
          }
        }
      }
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 9; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      placeholder_shared[(((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 36)) + (((int)threadIdx.x) * 9)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))] = placeholder1[(((((((((int)blockIdx.z) * 73728) + (((int)threadIdx.z) * 2304)) + (rc_outer * 72)) + (((int)threadIdx.y) * 36)) + (((int)threadIdx.x) * 9)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))];
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
  T_relu[(((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 144)) + (((int)blockIdx.y) * 24)) + (((int)threadIdx.y) * 12)) + (((int)blockIdx.x) * 4)) + ((int)threadIdx.x)))] = max(compute[(0)], 0.000000e+00f);
}

__device__ void fused_nn_batch_flatten_kernel0_device(float* __restrict__ tensor, float* __restrict__ placeholder){
  tensor[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)))] = placeholder[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)))];
}

__device__ void fused_nn_conv2d_nn_relu_kernel0_device(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu){
  float compute[1];
  __shared__ float pad_temp_shared[192];
  __shared__ float placeholder_shared[2304];
  compute[(0)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 48; ++rc_outer) {
    __syncthreads();
    if (((((((int)threadIdx.y) * 3) + ((int)threadIdx.x)) / 6) + ((int)threadIdx.z)) < 32) {
      if ((((((int)threadIdx.z) * 6) + (((int)threadIdx.y) * 3)) + ((int)threadIdx.x)) < 192) {
        if (((((int)threadIdx.y) * 3) + ((int)threadIdx.x)) < 6) {
          if (((int)threadIdx.x) < 3) {
            pad_temp_shared[((((((int)threadIdx.z) * 6) + (((int)threadIdx.y) * 3)) + ((int)threadIdx.x)))] = (((((1 <= ((((int)blockIdx.y) * 2) + (((int)threadIdx.z) & 3))) && (((((int)blockIdx.y) * 2) + (((int)threadIdx.z) & 3)) < 13)) && (1 <= (((((int)blockIdx.x) * 4) + (((int)threadIdx.y) * 3)) + ((int)threadIdx.x)))) && ((((((int)blockIdx.x) * 4) + (((int)threadIdx.y) * 3)) + ((int)threadIdx.x)) < 13)) ? placeholder[(((((((((rc_outer * 1152) + ((((int)threadIdx.z) >> 2) * 144)) + (((int)blockIdx.y) * 24)) + ((((int)threadIdx.z) & 3) * 12)) + (((int)blockIdx.x) * 4)) + (((int)threadIdx.y) * 3)) + ((int)threadIdx.x)) - 13))] : 0.000000e+00f);
          }
        }
      }
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 9; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      placeholder_shared[(((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 36)) + (((int)threadIdx.x) * 9)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))] = placeholder1[(((((((((int)blockIdx.z) * 110592) + (((int)threadIdx.z) * 3456)) + (rc_outer * 72)) + (((int)threadIdx.y) * 36)) + (((int)threadIdx.x) * 9)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))];
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
  T_relu[(((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 144)) + (((int)blockIdx.y) * 24)) + (((int)threadIdx.y) * 12)) + (((int)blockIdx.x) * 4)) + ((int)threadIdx.x)))] = max(compute[(0)], 0.000000e+00f);
}

__device__ void fused_nn_dense_add_nn_relu_kernel0_device(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2){
  float T_dense_rf[1];
  float red_buf0[1];
  __shared__ float T_dense[1];
  T_dense_rf[(0)] = 0.000000e+00f;
  for (int k_outer = 0; k_outer < 64; ++k_outer) {
    T_dense_rf[(0)] = __ocml_fma_f32(placeholder[(((k_outer * 64) + ((int)threadIdx.x)))], placeholder1[((((((int)blockIdx.x) * 4096) + (k_outer * 64)) + ((int)threadIdx.x)))], T_dense_rf[(0)]);
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
    T_relu[(((int)blockIdx.x))] = max((T_dense[(0)] + placeholder2[(((int)blockIdx.x))]), 0.000000e+00f);
  }
}


extern "C" __global__ void fused_nn_max_pool2d_kernel0(int* preempted, int* task_slot, float* __restrict__ placeholder, float* __restrict__ tensor) {
    if (*preempted) return;
    fused_nn_max_pool2d_kernel0_device(placeholder, tensor);
    if (threadIdx.x + threadIdx.y + threadIdx.z == 0)
        atomicAdd(task_slot, 1);
}        

extern "C" __global__ void fused_nn_softmax_kernel0(int* preempted, int* task_slot, float* __restrict__ placeholder, float* __restrict__ T_softmax_norm) {
    if (*preempted) return;
    fused_nn_softmax_kernel0_device(placeholder, T_softmax_norm);
    if (threadIdx.x + threadIdx.y + threadIdx.z == 0)
        atomicAdd(task_slot, 1);
}        

extern "C" __global__ void fused_nn_conv2d_nn_relu_3_kernel0(int* preempted, int* task_slot, float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu) {
    if (*preempted) return;
    fused_nn_conv2d_nn_relu_3_kernel0_device(placeholder, placeholder1, T_relu);
    if (threadIdx.x + threadIdx.y + threadIdx.z == 0)
        atomicAdd(task_slot, 1);
}        

extern "C" __global__ void fused_nn_conv2d_nn_relu_1_kernel0(int* preempted, int* task_slot, float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu) {
    if (*preempted) return;
    fused_nn_conv2d_nn_relu_1_kernel0_device(placeholder, placeholder1, T_relu);
    if (threadIdx.x + threadIdx.y + threadIdx.z == 0)
        atomicAdd(task_slot, 1);
}        

extern "C" __global__ void fused_nn_dense_add_kernel0(int* preempted, int* task_slot, float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_add, float* __restrict__ placeholder2) {
    if (*preempted) return;
    fused_nn_dense_add_kernel0_device(placeholder, placeholder1, T_add, placeholder2);
    if (threadIdx.x + threadIdx.y + threadIdx.z == 0)
        atomicAdd(task_slot, 1);
}        

extern "C" __global__ void fused_nn_max_pool2d_2_kernel0(int* preempted, int* task_slot, float* __restrict__ placeholder, float* __restrict__ tensor) {
    if (*preempted) return;
    fused_nn_max_pool2d_2_kernel0_device(placeholder, tensor);
    if (threadIdx.x + threadIdx.y + threadIdx.z == 0)
        atomicAdd(task_slot, 1);
}        

extern "C" __global__ void fused_nn_dense_add_nn_relu_1_kernel0(int* preempted, int* task_slot, float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
    if (*preempted) return;
    fused_nn_dense_add_nn_relu_1_kernel0_device(placeholder, placeholder1, T_relu, placeholder2);
    if (threadIdx.x + threadIdx.y + threadIdx.z == 0)
        atomicAdd(task_slot, 1);
}        

extern "C" __global__ void fused_nn_conv2d_nn_relu_4_kernel0(int* preempted, int* task_slot, float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu) {
    if (*preempted) return;
    fused_nn_conv2d_nn_relu_4_kernel0_device(placeholder, placeholder1, T_relu);
    if (threadIdx.x + threadIdx.y + threadIdx.z == 0)
        atomicAdd(task_slot, 1);
}        

extern "C" __global__ void fused_nn_max_pool2d_1_kernel0(int* preempted, int* task_slot, float* __restrict__ placeholder, float* __restrict__ tensor) {
    if (*preempted) return;
    fused_nn_max_pool2d_1_kernel0_device(placeholder, tensor);
    if (threadIdx.x + threadIdx.y + threadIdx.z == 0)
        atomicAdd(task_slot, 1);
}        

extern "C" __global__ void fused_nn_conv2d_nn_relu_2_kernel0(int* preempted, int* task_slot, float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu) {
    if (*preempted) return;
    fused_nn_conv2d_nn_relu_2_kernel0_device(placeholder, placeholder1, T_relu);
    if (threadIdx.x + threadIdx.y + threadIdx.z == 0)
        atomicAdd(task_slot, 1);
}        

extern "C" __global__ void fused_nn_batch_flatten_kernel0(int* preempted, int* task_slot, float* __restrict__ tensor, float* __restrict__ placeholder) {
    if (*preempted) return;
    fused_nn_batch_flatten_kernel0_device(tensor, placeholder);
    if (threadIdx.x + threadIdx.y + threadIdx.z == 0)
        atomicAdd(task_slot, 1);
}        

extern "C" __global__ void fused_nn_conv2d_nn_relu_kernel0(int* preempted, int* task_slot, float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu) {
    if (*preempted) return;
    fused_nn_conv2d_nn_relu_kernel0_device(placeholder, placeholder1, T_relu);
    if (threadIdx.x + threadIdx.y + threadIdx.z == 0)
        atomicAdd(task_slot, 1);
}        

extern "C" __global__ void fused_nn_dense_add_nn_relu_kernel0(int* preempted, int* task_slot, float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
    if (*preempted) return;
    fused_nn_dense_add_nn_relu_kernel0_device(placeholder, placeholder1, T_relu, placeholder2);
    if (threadIdx.x + threadIdx.y + threadIdx.z == 0)
        atomicAdd(task_slot, 1);
}        
