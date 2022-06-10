#include <hip/hip_runtime.h>

extern "C" __global__ void fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_1_kernel0(float* __restrict__ placeholder, float* __restrict__ data_pack) {
  float d[16];
  float data_pack_local[16];
  for (int eps = 0; eps < 4; ++eps) {
    for (int nu = 0; nu < 4; ++nu) {
      d[(((eps * 4) + nu))] = (((((1 <= ((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 49) / 7) * 2) + eps)) && (((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 49) / 7) * 2) + eps) < 15)) && (1 <= (((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 7) * 2) + nu))) && ((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 7) * 2) + nu) < 15)) ? placeholder[(((((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 7) * 28) + (eps * 14)) + ((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 7) * 2)) + nu) - 15))] : 0.000000e+00f);
    }
  }
  data_pack_local[(0)] = 0.000000e+00f;
  data_pack_local[(0)] = (data_pack_local[(0)] + d[(0)]);
  data_pack_local[(0)] = __ocml_fma_f32(d[(2)], -1.000000e+00f, data_pack_local[(0)]);
  data_pack_local[(0)] = __ocml_fma_f32(d[(8)], -1.000000e+00f, data_pack_local[(0)]);
  data_pack_local[(0)] = __ocml_fma_f32((d[(10)] * -1.000000e+00f), -1.000000e+00f, data_pack_local[(0)]);
  data_pack_local[(1)] = 0.000000e+00f;
  data_pack_local[(1)] = __ocml_fma_f32(d[(1)], -1.000000e+00f, data_pack_local[(1)]);
  data_pack_local[(1)] = (data_pack_local[(1)] + d[(2)]);
  data_pack_local[(1)] = __ocml_fma_f32((d[(9)] * -1.000000e+00f), -1.000000e+00f, data_pack_local[(1)]);
  data_pack_local[(1)] = __ocml_fma_f32(d[(10)], -1.000000e+00f, data_pack_local[(1)]);
  data_pack_local[(2)] = 0.000000e+00f;
  data_pack_local[(2)] = (data_pack_local[(2)] + d[(1)]);
  data_pack_local[(2)] = (data_pack_local[(2)] + d[(2)]);
  data_pack_local[(2)] = __ocml_fma_f32(d[(9)], -1.000000e+00f, data_pack_local[(2)]);
  data_pack_local[(2)] = __ocml_fma_f32(d[(10)], -1.000000e+00f, data_pack_local[(2)]);
  data_pack_local[(3)] = 0.000000e+00f;
  data_pack_local[(3)] = __ocml_fma_f32(d[(1)], -1.000000e+00f, data_pack_local[(3)]);
  data_pack_local[(3)] = (data_pack_local[(3)] + d[(3)]);
  data_pack_local[(3)] = __ocml_fma_f32((d[(9)] * -1.000000e+00f), -1.000000e+00f, data_pack_local[(3)]);
  data_pack_local[(3)] = __ocml_fma_f32(d[(11)], -1.000000e+00f, data_pack_local[(3)]);
  data_pack_local[(4)] = 0.000000e+00f;
  data_pack_local[(4)] = __ocml_fma_f32(d[(4)], -1.000000e+00f, data_pack_local[(4)]);
  data_pack_local[(4)] = __ocml_fma_f32((d[(6)] * -1.000000e+00f), -1.000000e+00f, data_pack_local[(4)]);
  data_pack_local[(4)] = (data_pack_local[(4)] + d[(8)]);
  data_pack_local[(4)] = __ocml_fma_f32(d[(10)], -1.000000e+00f, data_pack_local[(4)]);
  data_pack_local[(5)] = 0.000000e+00f;
  data_pack_local[(5)] = __ocml_fma_f32((d[(5)] * -1.000000e+00f), -1.000000e+00f, data_pack_local[(5)]);
  data_pack_local[(5)] = __ocml_fma_f32(d[(6)], -1.000000e+00f, data_pack_local[(5)]);
  data_pack_local[(5)] = __ocml_fma_f32(d[(9)], -1.000000e+00f, data_pack_local[(5)]);
  data_pack_local[(5)] = (data_pack_local[(5)] + d[(10)]);
  data_pack_local[(6)] = 0.000000e+00f;
  data_pack_local[(6)] = __ocml_fma_f32(d[(5)], -1.000000e+00f, data_pack_local[(6)]);
  data_pack_local[(6)] = __ocml_fma_f32(d[(6)], -1.000000e+00f, data_pack_local[(6)]);
  data_pack_local[(6)] = (data_pack_local[(6)] + d[(9)]);
  data_pack_local[(6)] = (data_pack_local[(6)] + d[(10)]);
  data_pack_local[(7)] = 0.000000e+00f;
  data_pack_local[(7)] = __ocml_fma_f32((d[(5)] * -1.000000e+00f), -1.000000e+00f, data_pack_local[(7)]);
  data_pack_local[(7)] = __ocml_fma_f32(d[(7)], -1.000000e+00f, data_pack_local[(7)]);
  data_pack_local[(7)] = __ocml_fma_f32(d[(9)], -1.000000e+00f, data_pack_local[(7)]);
  data_pack_local[(7)] = (data_pack_local[(7)] + d[(11)]);
  data_pack_local[(8)] = 0.000000e+00f;
  data_pack_local[(8)] = (data_pack_local[(8)] + d[(4)]);
  data_pack_local[(8)] = __ocml_fma_f32(d[(6)], -1.000000e+00f, data_pack_local[(8)]);
  data_pack_local[(8)] = (data_pack_local[(8)] + d[(8)]);
  data_pack_local[(8)] = __ocml_fma_f32(d[(10)], -1.000000e+00f, data_pack_local[(8)]);
  data_pack_local[(9)] = 0.000000e+00f;
  data_pack_local[(9)] = __ocml_fma_f32(d[(5)], -1.000000e+00f, data_pack_local[(9)]);
  data_pack_local[(9)] = (data_pack_local[(9)] + d[(6)]);
  data_pack_local[(9)] = __ocml_fma_f32(d[(9)], -1.000000e+00f, data_pack_local[(9)]);
  data_pack_local[(9)] = (data_pack_local[(9)] + d[(10)]);
  data_pack_local[(10)] = 0.000000e+00f;
  data_pack_local[(10)] = (data_pack_local[(10)] + d[(5)]);
  data_pack_local[(10)] = (data_pack_local[(10)] + d[(6)]);
  data_pack_local[(10)] = (data_pack_local[(10)] + d[(9)]);
  data_pack_local[(10)] = (data_pack_local[(10)] + d[(10)]);
  data_pack_local[(11)] = 0.000000e+00f;
  data_pack_local[(11)] = __ocml_fma_f32(d[(5)], -1.000000e+00f, data_pack_local[(11)]);
  data_pack_local[(11)] = (data_pack_local[(11)] + d[(7)]);
  data_pack_local[(11)] = __ocml_fma_f32(d[(9)], -1.000000e+00f, data_pack_local[(11)]);
  data_pack_local[(11)] = (data_pack_local[(11)] + d[(11)]);
  data_pack_local[(12)] = 0.000000e+00f;
  data_pack_local[(12)] = __ocml_fma_f32(d[(4)], -1.000000e+00f, data_pack_local[(12)]);
  data_pack_local[(12)] = __ocml_fma_f32((d[(6)] * -1.000000e+00f), -1.000000e+00f, data_pack_local[(12)]);
  data_pack_local[(12)] = (data_pack_local[(12)] + d[(12)]);
  data_pack_local[(12)] = __ocml_fma_f32(d[(14)], -1.000000e+00f, data_pack_local[(12)]);
  data_pack_local[(13)] = 0.000000e+00f;
  data_pack_local[(13)] = __ocml_fma_f32((d[(5)] * -1.000000e+00f), -1.000000e+00f, data_pack_local[(13)]);
  data_pack_local[(13)] = __ocml_fma_f32(d[(6)], -1.000000e+00f, data_pack_local[(13)]);
  data_pack_local[(13)] = __ocml_fma_f32(d[(13)], -1.000000e+00f, data_pack_local[(13)]);
  data_pack_local[(13)] = (data_pack_local[(13)] + d[(14)]);
  data_pack_local[(14)] = 0.000000e+00f;
  data_pack_local[(14)] = __ocml_fma_f32(d[(5)], -1.000000e+00f, data_pack_local[(14)]);
  data_pack_local[(14)] = __ocml_fma_f32(d[(6)], -1.000000e+00f, data_pack_local[(14)]);
  data_pack_local[(14)] = (data_pack_local[(14)] + d[(13)]);
  data_pack_local[(14)] = (data_pack_local[(14)] + d[(14)]);
  data_pack_local[(15)] = 0.000000e+00f;
  data_pack_local[(15)] = __ocml_fma_f32((d[(5)] * -1.000000e+00f), -1.000000e+00f, data_pack_local[(15)]);
  data_pack_local[(15)] = __ocml_fma_f32(d[(7)], -1.000000e+00f, data_pack_local[(15)]);
  data_pack_local[(15)] = __ocml_fma_f32(d[(13)], -1.000000e+00f, data_pack_local[(15)]);
  data_pack_local[(15)] = (data_pack_local[(15)] + d[(15)]);
  for (int eps1 = 0; eps1 < 4; ++eps1) {
    for (int nu1 = 0; nu1 < 4; ++nu1) {
      data_pack[(((((eps1 * 50176) + (nu1 * 12544)) + (((int)blockIdx.x) * 128)) + ((int)threadIdx.x)))] = data_pack_local[(((eps1 * 4) + nu1))];
    }
  }
}

extern "C" __global__ void fused_nn_softmax_kernel0(float* __restrict__ placeholder, float* __restrict__ T_softmax_norm) {
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

extern "C" __global__ void fused_add_nn_relu_1_kernel0(float* __restrict__ T_relu, float* __restrict__ placeholder, float* __restrict__ placeholder1) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) < 200704) {
      T_relu[((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)))] = max((placeholder[((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)))] + placeholder1[(((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) / 196))]), 0.000000e+00f);
    }
  }
}

extern "C" __global__ void fused_nn_global_avg_pool2d_kernel0(float* __restrict__ placeholder, float* __restrict__ tensor) {
  float tensor1[1];
  tensor1[(0)] = 0.000000e+00f;
  for (int rv0 = 0; rv0 < 7; ++rv0) {
    for (int rv1 = 0; rv1 < 7; ++rv1) {
      if (((int)threadIdx.y) < 1) {
        tensor1[(0)] = (tensor1[(0)] + placeholder[((((((((int)threadIdx.y) * 100352) + (((int)blockIdx.x) * 392)) + (((int)threadIdx.x) * 49)) + (rv0 * 7)) + rv1))]);
      }
    }
  }
  if (((int)threadIdx.y) < 1) {
    tensor[((((((int)threadIdx.y) * 2048) + (((int)blockIdx.x) * 8)) + ((int)threadIdx.x)))] = (tensor1[(0)] * 2.040816e-02f);
  }
}

extern "C" __global__ void fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_2_kernel2(float* __restrict__ bgemm, float* __restrict__ T_relu, float* __restrict__ placeholder) {
  float inverse[4];
  inverse[(0)] = 0.000000e+00f;
  inverse[(0)] = (inverse[(0)] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 25088))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 50176))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 150528))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 225792))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880))]);
  inverse[(1)] = 0.000000e+00f;
  inverse[(1)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 25088))], -1.000000e+00f, inverse[(1)]);
  inverse[(1)] = (inverse[(1)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 50176))]);
  inverse[(1)] = (inverse[(1)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 75264))]);
  inverse[(1)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440))], -1.000000e+00f, inverse[(1)]);
  inverse[(1)] = (inverse[(1)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 150528))]);
  inverse[(1)] = (inverse[(1)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616))]);
  inverse[(1)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 225792))], -1.000000e+00f, inverse[(1)]);
  inverse[(1)] = (inverse[(1)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880))]);
  inverse[(1)] = (inverse[(1)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968))]);
  inverse[(2)] = 0.000000e+00f;
  inverse[(2)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352))], -1.000000e+00f, inverse[(2)]);
  inverse[(2)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440))], -1.000000e+00f, inverse[(2)]);
  inverse[(2)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 150528))], -1.000000e+00f, inverse[(2)]);
  inverse[(2)] = (inverse[(2)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704))]);
  inverse[(2)] = (inverse[(2)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 225792))]);
  inverse[(2)] = (inverse[(2)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880))]);
  inverse[(2)] = (inverse[(2)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 301056))]);
  inverse[(2)] = (inverse[(2)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144))]);
  inverse[(2)] = (inverse[(2)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232))]);
  inverse[(3)] = 0.000000e+00f;
  inverse[(3)] = __ocml_fma_f32((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440))] * -1.000000e+00f), -1.000000e+00f, inverse[(3)]);
  inverse[(3)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 150528))], -1.000000e+00f, inverse[(3)]);
  inverse[(3)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616))], -1.000000e+00f, inverse[(3)]);
  inverse[(3)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 225792))], -1.000000e+00f, inverse[(3)]);
  inverse[(3)] = (inverse[(3)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880))]);
  inverse[(3)] = (inverse[(3)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968))]);
  inverse[(3)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144))], -1.000000e+00f, inverse[(3)]);
  inverse[(3)] = (inverse[(3)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232))]);
  inverse[(3)] = (inverse[(3)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 376320))]);
  for (int ax2_inner = 0; ax2_inner < 2; ++ax2_inner) {
    for (int ax3_inner = 0; ax3_inner < 2; ++ax3_inner) {
      T_relu[((((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 14) * 56) + (ax2_inner * 28)) + ((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 14) * 2)) + ax3_inner))] = max((inverse[(((ax2_inner * 2) + ax3_inner))] + placeholder[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 196))]), 0.000000e+00f);
    }
  }
}

extern "C" __global__ void fused_add_14_kernel0(float* __restrict__ T_add, float* __restrict__ placeholder, float* __restrict__ placeholder1) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer < 3; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) < 150528) {
      T_add[((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)))] = (placeholder[((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)))] + placeholder1[(((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) / 50176))]);
    }
  }
}

extern "C" __global__ void fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_kernel0(float* __restrict__ placeholder, float* __restrict__ data_pack) {
  float d[16];
  float data_pack_local[16];
  for (int eps = 0; eps < 4; ++eps) {
    for (int nu = 0; nu < 4; ++nu) {
      d[(((eps * 4) + nu))] = (((((1 <= ((((((int)threadIdx.x) & 15) >> 2) * 2) + eps)) && (((((((int)threadIdx.x) & 15) >> 2) * 2) + eps) < 8)) && (1 <= (((((int)threadIdx.x) & 3) * 2) + nu))) && ((((((int)threadIdx.x) & 3) * 2) + nu) < 8)) ? placeholder[((((((((((int)blockIdx.x) * 392) + ((((int)threadIdx.x) >> 4) * 49)) + (((((int)threadIdx.x) & 15) >> 2) * 14)) + (eps * 7)) + ((((int)threadIdx.x) & 3) * 2)) + nu) - 8))] : 0.000000e+00f);
    }
  }
  data_pack_local[(0)] = 0.000000e+00f;
  data_pack_local[(0)] = (data_pack_local[(0)] + d[(0)]);
  data_pack_local[(0)] = __ocml_fma_f32(d[(2)], -1.000000e+00f, data_pack_local[(0)]);
  data_pack_local[(0)] = __ocml_fma_f32(d[(8)], -1.000000e+00f, data_pack_local[(0)]);
  data_pack_local[(0)] = __ocml_fma_f32((d[(10)] * -1.000000e+00f), -1.000000e+00f, data_pack_local[(0)]);
  data_pack_local[(1)] = 0.000000e+00f;
  data_pack_local[(1)] = __ocml_fma_f32(d[(1)], -1.000000e+00f, data_pack_local[(1)]);
  data_pack_local[(1)] = (data_pack_local[(1)] + d[(2)]);
  data_pack_local[(1)] = __ocml_fma_f32((d[(9)] * -1.000000e+00f), -1.000000e+00f, data_pack_local[(1)]);
  data_pack_local[(1)] = __ocml_fma_f32(d[(10)], -1.000000e+00f, data_pack_local[(1)]);
  data_pack_local[(2)] = 0.000000e+00f;
  data_pack_local[(2)] = (data_pack_local[(2)] + d[(1)]);
  data_pack_local[(2)] = (data_pack_local[(2)] + d[(2)]);
  data_pack_local[(2)] = __ocml_fma_f32(d[(9)], -1.000000e+00f, data_pack_local[(2)]);
  data_pack_local[(2)] = __ocml_fma_f32(d[(10)], -1.000000e+00f, data_pack_local[(2)]);
  data_pack_local[(3)] = 0.000000e+00f;
  data_pack_local[(3)] = __ocml_fma_f32(d[(1)], -1.000000e+00f, data_pack_local[(3)]);
  data_pack_local[(3)] = (data_pack_local[(3)] + d[(3)]);
  data_pack_local[(3)] = __ocml_fma_f32((d[(9)] * -1.000000e+00f), -1.000000e+00f, data_pack_local[(3)]);
  data_pack_local[(3)] = __ocml_fma_f32(d[(11)], -1.000000e+00f, data_pack_local[(3)]);
  data_pack_local[(4)] = 0.000000e+00f;
  data_pack_local[(4)] = __ocml_fma_f32(d[(4)], -1.000000e+00f, data_pack_local[(4)]);
  data_pack_local[(4)] = __ocml_fma_f32((d[(6)] * -1.000000e+00f), -1.000000e+00f, data_pack_local[(4)]);
  data_pack_local[(4)] = (data_pack_local[(4)] + d[(8)]);
  data_pack_local[(4)] = __ocml_fma_f32(d[(10)], -1.000000e+00f, data_pack_local[(4)]);
  data_pack_local[(5)] = 0.000000e+00f;
  data_pack_local[(5)] = __ocml_fma_f32((d[(5)] * -1.000000e+00f), -1.000000e+00f, data_pack_local[(5)]);
  data_pack_local[(5)] = __ocml_fma_f32(d[(6)], -1.000000e+00f, data_pack_local[(5)]);
  data_pack_local[(5)] = __ocml_fma_f32(d[(9)], -1.000000e+00f, data_pack_local[(5)]);
  data_pack_local[(5)] = (data_pack_local[(5)] + d[(10)]);
  data_pack_local[(6)] = 0.000000e+00f;
  data_pack_local[(6)] = __ocml_fma_f32(d[(5)], -1.000000e+00f, data_pack_local[(6)]);
  data_pack_local[(6)] = __ocml_fma_f32(d[(6)], -1.000000e+00f, data_pack_local[(6)]);
  data_pack_local[(6)] = (data_pack_local[(6)] + d[(9)]);
  data_pack_local[(6)] = (data_pack_local[(6)] + d[(10)]);
  data_pack_local[(7)] = 0.000000e+00f;
  data_pack_local[(7)] = __ocml_fma_f32((d[(5)] * -1.000000e+00f), -1.000000e+00f, data_pack_local[(7)]);
  data_pack_local[(7)] = __ocml_fma_f32(d[(7)], -1.000000e+00f, data_pack_local[(7)]);
  data_pack_local[(7)] = __ocml_fma_f32(d[(9)], -1.000000e+00f, data_pack_local[(7)]);
  data_pack_local[(7)] = (data_pack_local[(7)] + d[(11)]);
  data_pack_local[(8)] = 0.000000e+00f;
  data_pack_local[(8)] = (data_pack_local[(8)] + d[(4)]);
  data_pack_local[(8)] = __ocml_fma_f32(d[(6)], -1.000000e+00f, data_pack_local[(8)]);
  data_pack_local[(8)] = (data_pack_local[(8)] + d[(8)]);
  data_pack_local[(8)] = __ocml_fma_f32(d[(10)], -1.000000e+00f, data_pack_local[(8)]);
  data_pack_local[(9)] = 0.000000e+00f;
  data_pack_local[(9)] = __ocml_fma_f32(d[(5)], -1.000000e+00f, data_pack_local[(9)]);
  data_pack_local[(9)] = (data_pack_local[(9)] + d[(6)]);
  data_pack_local[(9)] = __ocml_fma_f32(d[(9)], -1.000000e+00f, data_pack_local[(9)]);
  data_pack_local[(9)] = (data_pack_local[(9)] + d[(10)]);
  data_pack_local[(10)] = 0.000000e+00f;
  data_pack_local[(10)] = (data_pack_local[(10)] + d[(5)]);
  data_pack_local[(10)] = (data_pack_local[(10)] + d[(6)]);
  data_pack_local[(10)] = (data_pack_local[(10)] + d[(9)]);
  data_pack_local[(10)] = (data_pack_local[(10)] + d[(10)]);
  data_pack_local[(11)] = 0.000000e+00f;
  data_pack_local[(11)] = __ocml_fma_f32(d[(5)], -1.000000e+00f, data_pack_local[(11)]);
  data_pack_local[(11)] = (data_pack_local[(11)] + d[(7)]);
  data_pack_local[(11)] = __ocml_fma_f32(d[(9)], -1.000000e+00f, data_pack_local[(11)]);
  data_pack_local[(11)] = (data_pack_local[(11)] + d[(11)]);
  data_pack_local[(12)] = 0.000000e+00f;
  data_pack_local[(12)] = __ocml_fma_f32(d[(4)], -1.000000e+00f, data_pack_local[(12)]);
  data_pack_local[(12)] = __ocml_fma_f32((d[(6)] * -1.000000e+00f), -1.000000e+00f, data_pack_local[(12)]);
  data_pack_local[(12)] = (data_pack_local[(12)] + d[(12)]);
  data_pack_local[(12)] = __ocml_fma_f32(d[(14)], -1.000000e+00f, data_pack_local[(12)]);
  data_pack_local[(13)] = 0.000000e+00f;
  data_pack_local[(13)] = __ocml_fma_f32((d[(5)] * -1.000000e+00f), -1.000000e+00f, data_pack_local[(13)]);
  data_pack_local[(13)] = __ocml_fma_f32(d[(6)], -1.000000e+00f, data_pack_local[(13)]);
  data_pack_local[(13)] = __ocml_fma_f32(d[(13)], -1.000000e+00f, data_pack_local[(13)]);
  data_pack_local[(13)] = (data_pack_local[(13)] + d[(14)]);
  data_pack_local[(14)] = 0.000000e+00f;
  data_pack_local[(14)] = __ocml_fma_f32(d[(5)], -1.000000e+00f, data_pack_local[(14)]);
  data_pack_local[(14)] = __ocml_fma_f32(d[(6)], -1.000000e+00f, data_pack_local[(14)]);
  data_pack_local[(14)] = (data_pack_local[(14)] + d[(13)]);
  data_pack_local[(14)] = (data_pack_local[(14)] + d[(14)]);
  data_pack_local[(15)] = 0.000000e+00f;
  data_pack_local[(15)] = __ocml_fma_f32((d[(5)] * -1.000000e+00f), -1.000000e+00f, data_pack_local[(15)]);
  data_pack_local[(15)] = __ocml_fma_f32(d[(7)], -1.000000e+00f, data_pack_local[(15)]);
  data_pack_local[(15)] = __ocml_fma_f32(d[(13)], -1.000000e+00f, data_pack_local[(15)]);
  data_pack_local[(15)] = (data_pack_local[(15)] + d[(15)]);
  for (int eps1 = 0; eps1 < 4; ++eps1) {
    for (int nu1 = 0; nu1 < 4; ++nu1) {
      data_pack[(((((eps1 * 32768) + (nu1 * 8192)) + (((int)blockIdx.x) * 128)) + ((int)threadIdx.x)))] = data_pack_local[(((eps1 * 4) + nu1))];
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_7_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  float compute[4];
  __shared__ float pad_temp_shared[256];
  __shared__ float placeholder_shared[512];
  compute[(0)] = 0.000000e+00f;
  compute[(2)] = 0.000000e+00f;
  compute[(1)] = 0.000000e+00f;
  compute[(3)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 4; ++rc_outer) {
    __syncthreads();
    pad_temp_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)))] = placeholder[(((((((rc_outer * 50176) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)))];
    pad_temp_shared[(((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1))] = placeholder[((((((((rc_outer * 50176) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 1))];
    placeholder_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))] = placeholder1[((((((((int)blockIdx.z) * 2048) + (((int)threadIdx.z) * 128)) + (((int)threadIdx.y) * 64)) + (rc_outer * 16)) + (((int)threadIdx.x) * 4)))];
    placeholder_shared[(((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 1))] = placeholder1[(((((((((int)blockIdx.z) * 2048) + (((int)threadIdx.z) * 128)) + (((int)threadIdx.y) * 64)) + (rc_outer * 16)) + (((int)threadIdx.x) * 4)) + 1))];
    placeholder_shared[(((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 2))] = placeholder1[(((((((((int)blockIdx.z) * 2048) + (((int)threadIdx.z) * 128)) + (((int)threadIdx.y) * 64)) + (rc_outer * 16)) + (((int)threadIdx.x) * 4)) + 2))];
    placeholder_shared[(((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 3))] = placeholder1[(((((((((int)blockIdx.z) * 2048) + (((int)threadIdx.z) * 128)) + (((int)threadIdx.y) * 64)) + (rc_outer * 16)) + (((int)threadIdx.x) * 4)) + 3))];
    __syncthreads();
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)))], placeholder_shared[((((int)threadIdx.z) * 16))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)))], placeholder_shared[(((((int)threadIdx.z) * 16) + 256))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 1))], placeholder_shared[((((int)threadIdx.z) * 16))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 1))], placeholder_shared[(((((int)threadIdx.z) * 16) + 256))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 16))], placeholder_shared[(((((int)threadIdx.z) * 16) + 1))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 16))], placeholder_shared[(((((int)threadIdx.z) * 16) + 257))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 17))], placeholder_shared[(((((int)threadIdx.z) * 16) + 1))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 17))], placeholder_shared[(((((int)threadIdx.z) * 16) + 257))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 32))], placeholder_shared[(((((int)threadIdx.z) * 16) + 2))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 32))], placeholder_shared[(((((int)threadIdx.z) * 16) + 258))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 33))], placeholder_shared[(((((int)threadIdx.z) * 16) + 2))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 33))], placeholder_shared[(((((int)threadIdx.z) * 16) + 258))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 48))], placeholder_shared[(((((int)threadIdx.z) * 16) + 3))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 48))], placeholder_shared[(((((int)threadIdx.z) * 16) + 259))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 49))], placeholder_shared[(((((int)threadIdx.z) * 16) + 3))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 49))], placeholder_shared[(((((int)threadIdx.z) * 16) + 259))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 64))], placeholder_shared[(((((int)threadIdx.z) * 16) + 4))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 64))], placeholder_shared[(((((int)threadIdx.z) * 16) + 260))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 65))], placeholder_shared[(((((int)threadIdx.z) * 16) + 4))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 65))], placeholder_shared[(((((int)threadIdx.z) * 16) + 260))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 80))], placeholder_shared[(((((int)threadIdx.z) * 16) + 5))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 80))], placeholder_shared[(((((int)threadIdx.z) * 16) + 261))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 81))], placeholder_shared[(((((int)threadIdx.z) * 16) + 5))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 81))], placeholder_shared[(((((int)threadIdx.z) * 16) + 261))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 96))], placeholder_shared[(((((int)threadIdx.z) * 16) + 6))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 96))], placeholder_shared[(((((int)threadIdx.z) * 16) + 262))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 97))], placeholder_shared[(((((int)threadIdx.z) * 16) + 6))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 97))], placeholder_shared[(((((int)threadIdx.z) * 16) + 262))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 112))], placeholder_shared[(((((int)threadIdx.z) * 16) + 7))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 112))], placeholder_shared[(((((int)threadIdx.z) * 16) + 263))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 113))], placeholder_shared[(((((int)threadIdx.z) * 16) + 7))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 113))], placeholder_shared[(((((int)threadIdx.z) * 16) + 263))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 128))], placeholder_shared[(((((int)threadIdx.z) * 16) + 8))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 128))], placeholder_shared[(((((int)threadIdx.z) * 16) + 264))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 129))], placeholder_shared[(((((int)threadIdx.z) * 16) + 8))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 129))], placeholder_shared[(((((int)threadIdx.z) * 16) + 264))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 144))], placeholder_shared[(((((int)threadIdx.z) * 16) + 9))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 144))], placeholder_shared[(((((int)threadIdx.z) * 16) + 265))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 145))], placeholder_shared[(((((int)threadIdx.z) * 16) + 9))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 145))], placeholder_shared[(((((int)threadIdx.z) * 16) + 265))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 160))], placeholder_shared[(((((int)threadIdx.z) * 16) + 10))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 160))], placeholder_shared[(((((int)threadIdx.z) * 16) + 266))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 161))], placeholder_shared[(((((int)threadIdx.z) * 16) + 10))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 161))], placeholder_shared[(((((int)threadIdx.z) * 16) + 266))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 176))], placeholder_shared[(((((int)threadIdx.z) * 16) + 11))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 176))], placeholder_shared[(((((int)threadIdx.z) * 16) + 267))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 177))], placeholder_shared[(((((int)threadIdx.z) * 16) + 11))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 177))], placeholder_shared[(((((int)threadIdx.z) * 16) + 267))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 192))], placeholder_shared[(((((int)threadIdx.z) * 16) + 12))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 192))], placeholder_shared[(((((int)threadIdx.z) * 16) + 268))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 193))], placeholder_shared[(((((int)threadIdx.z) * 16) + 12))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 193))], placeholder_shared[(((((int)threadIdx.z) * 16) + 268))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 208))], placeholder_shared[(((((int)threadIdx.z) * 16) + 13))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 208))], placeholder_shared[(((((int)threadIdx.z) * 16) + 269))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 209))], placeholder_shared[(((((int)threadIdx.z) * 16) + 13))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 209))], placeholder_shared[(((((int)threadIdx.z) * 16) + 269))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 224))], placeholder_shared[(((((int)threadIdx.z) * 16) + 14))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 224))], placeholder_shared[(((((int)threadIdx.z) * 16) + 270))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 225))], placeholder_shared[(((((int)threadIdx.z) * 16) + 14))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 225))], placeholder_shared[(((((int)threadIdx.z) * 16) + 270))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 240))], placeholder_shared[(((((int)threadIdx.z) * 16) + 15))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 240))], placeholder_shared[(((((int)threadIdx.z) * 16) + 271))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 241))], placeholder_shared[(((((int)threadIdx.z) * 16) + 15))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 241))], placeholder_shared[(((((int)threadIdx.z) * 16) + 271))], compute[(3)]);
  }
  T_relu[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)))] = max((compute[(0)] + placeholder2[(((((int)blockIdx.z) * 32) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[((((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 50176))] = max((compute[(2)] + placeholder2[((((((int)blockIdx.z) * 32) + ((int)threadIdx.z)) + 16))]), 0.000000e+00f);
  T_relu[((((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 1))] = max((compute[(1)] + placeholder2[(((((int)blockIdx.z) * 32) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[((((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 50177))] = max((compute[(3)] + placeholder2[((((((int)blockIdx.z) * 32) + ((int)threadIdx.z)) + 16))]), 0.000000e+00f);
}

extern "C" __global__ void fused_nn_conv2d_add_3_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_add, float* __restrict__ placeholder2) {
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
  for (int rc_outer = 0; rc_outer < 4; ++rc_outer) {
    __syncthreads();
    pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)))] = placeholder[(((((((rc_outer * 50176) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)))];
    pad_temp_shared[(((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1))] = placeholder[((((((((rc_outer * 50176) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 1))];
    placeholder_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))] = placeholder1[((((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 64)) + (rc_outer * 16)) + (((int)threadIdx.x) * 4)))];
    placeholder_shared[(((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 1))] = placeholder1[(((((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 64)) + (rc_outer * 16)) + (((int)threadIdx.x) * 4)) + 1))];
    placeholder_shared[(((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 2))] = placeholder1[(((((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 64)) + (rc_outer * 16)) + (((int)threadIdx.x) * 4)) + 2))];
    placeholder_shared[(((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 3))] = placeholder1[(((((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 64)) + (rc_outer * 16)) + (((int)threadIdx.x) * 4)) + 3))];
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
  T_add[(((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)))] = (compute[(0)] + placeholder2[(((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)))]);
  T_add[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 50176))] = (compute[(2)] + placeholder2[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 50176))]);
  T_add[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 100352))] = (compute[(4)] + placeholder2[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 100352))]);
  T_add[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 150528))] = (compute[(6)] + placeholder2[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 150528))]);
  T_add[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 1))] = (compute[(1)] + placeholder2[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 1))]);
  T_add[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 50177))] = (compute[(3)] + placeholder2[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 50177))]);
  T_add[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 100353))] = (compute[(5)] + placeholder2[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 100353))]);
  T_add[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 150529))] = (compute[(7)] + placeholder2[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 150529))]);
}

extern "C" __global__ void fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_kernel2(float* __restrict__ bgemm, float* __restrict__ T_relu, float* __restrict__ placeholder) {
  float inverse[4];
  inverse[(0)] = 0.000000e+00f;
  inverse[(0)] = (inverse[(0)] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 8192))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 16384))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 32768))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 40960))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 49152))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 65536))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 73728))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 81920))]);
  inverse[(1)] = 0.000000e+00f;
  inverse[(1)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 8192))], -1.000000e+00f, inverse[(1)]);
  inverse[(1)] = (inverse[(1)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 16384))]);
  inverse[(1)] = (inverse[(1)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 24576))]);
  inverse[(1)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 40960))], -1.000000e+00f, inverse[(1)]);
  inverse[(1)] = (inverse[(1)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 49152))]);
  inverse[(1)] = (inverse[(1)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 57344))]);
  inverse[(1)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 73728))], -1.000000e+00f, inverse[(1)]);
  inverse[(1)] = (inverse[(1)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 81920))]);
  inverse[(1)] = (inverse[(1)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 90112))]);
  inverse[(2)] = 0.000000e+00f;
  inverse[(2)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 32768))], -1.000000e+00f, inverse[(2)]);
  inverse[(2)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 40960))], -1.000000e+00f, inverse[(2)]);
  inverse[(2)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 49152))], -1.000000e+00f, inverse[(2)]);
  inverse[(2)] = (inverse[(2)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 65536))]);
  inverse[(2)] = (inverse[(2)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 73728))]);
  inverse[(2)] = (inverse[(2)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 81920))]);
  inverse[(2)] = (inverse[(2)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 98304))]);
  inverse[(2)] = (inverse[(2)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 106496))]);
  inverse[(2)] = (inverse[(2)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 114688))]);
  inverse[(3)] = 0.000000e+00f;
  inverse[(3)] = __ocml_fma_f32((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 40960))] * -1.000000e+00f), -1.000000e+00f, inverse[(3)]);
  inverse[(3)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 49152))], -1.000000e+00f, inverse[(3)]);
  inverse[(3)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 57344))], -1.000000e+00f, inverse[(3)]);
  inverse[(3)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 73728))], -1.000000e+00f, inverse[(3)]);
  inverse[(3)] = (inverse[(3)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 81920))]);
  inverse[(3)] = (inverse[(3)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 90112))]);
  inverse[(3)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 106496))], -1.000000e+00f, inverse[(3)]);
  inverse[(3)] = (inverse[(3)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 114688))]);
  inverse[(3)] = (inverse[(3)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 122880))]);
  for (int ax2_inner = 0; ax2_inner < 2; ++ax2_inner) {
    for (int ax3_inner = 0; ax3_inner < 2; ++ax3_inner) {
      if (((((((int)threadIdx.x) & 15) >> 2) * 2) + ax2_inner) < 7) {
        if ((((((int)threadIdx.x) & 3) * 2) + ax3_inner) < 7) {
          T_relu[(((((((((int)blockIdx.x) * 392) + ((((int)threadIdx.x) >> 4) * 49)) + (((((int)threadIdx.x) & 15) >> 2) * 14)) + (ax2_inner * 7)) + ((((int)threadIdx.x) & 3) * 2)) + ax3_inner))] = max((inverse[(((ax2_inner * 2) + ax3_inner))] + placeholder[(((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 4)))]), 0.000000e+00f);
        }
      }
    }
  }
}

extern "C" __global__ void fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_1_kernel1(float* __restrict__ placeholder, float* __restrict__ data_pack, float* __restrict__ bgemm) {
  float bgemm_local[4];
  __shared__ float placeholder_shared[512];
  __shared__ float data_pack_shared[1568];
  bgemm_local[(0)] = 0.000000e+00f;
  bgemm_local[(1)] = 0.000000e+00f;
  bgemm_local[(2)] = 0.000000e+00f;
  bgemm_local[(3)] = 0.000000e+00f;
  for (int ci_outer = 0; ci_outer < 8; ++ci_outer) {
    __syncthreads();
    placeholder_shared[(((((int)threadIdx.y) * 49) + ((int)threadIdx.x)))] = placeholder[((((((((int)blockIdx.z) * 65536) + (ci_outer * 8192)) + ((((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) >> 4) * 256)) + (((int)blockIdx.y) * 16)) + (((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) & 15)))];
    placeholder_shared[((((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) + 196))] = placeholder[((((((((int)blockIdx.z) * 65536) + (ci_outer * 8192)) + (((((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) + 196) >> 4) * 256)) + (((int)blockIdx.y) * 16)) + ((((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) + 4) & 15)))];
    if (((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) < 120) {
      if (((int)threadIdx.y) < 3) {
        placeholder_shared[((((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) + 392))] = placeholder[((((((((int)blockIdx.z) * 65536) + (ci_outer * 8192)) + (((((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) + 392) >> 4) * 256)) + (((int)blockIdx.y) * 16)) + ((((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) + 8) & 15)))];
      }
    }
    data_pack_shared[(((((int)threadIdx.y) * 49) + ((int)threadIdx.x)))] = data_pack[(((((((int)blockIdx.z) * 12544) + (ci_outer * 1568)) + (((int)threadIdx.y) * 49)) + ((int)threadIdx.x)))];
    data_pack_shared[((((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) + 196))] = data_pack[((((((((int)blockIdx.z) * 12544) + (ci_outer * 1568)) + (((int)threadIdx.y) * 49)) + ((int)threadIdx.x)) + 196))];
    data_pack_shared[((((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) + 392))] = data_pack[((((((((int)blockIdx.z) * 12544) + (ci_outer * 1568)) + (((int)threadIdx.y) * 49)) + ((int)threadIdx.x)) + 392))];
    data_pack_shared[((((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) + 588))] = data_pack[((((((((int)blockIdx.z) * 12544) + (ci_outer * 1568)) + (((int)threadIdx.y) * 49)) + ((int)threadIdx.x)) + 588))];
    data_pack_shared[((((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) + 784))] = data_pack[((((((((int)blockIdx.z) * 12544) + (ci_outer * 1568)) + (((int)threadIdx.y) * 49)) + ((int)threadIdx.x)) + 784))];
    data_pack_shared[((((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) + 980))] = data_pack[((((((((int)blockIdx.z) * 12544) + (ci_outer * 1568)) + (((int)threadIdx.y) * 49)) + ((int)threadIdx.x)) + 980))];
    data_pack_shared[((((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) + 1176))] = data_pack[((((((((int)blockIdx.z) * 12544) + (ci_outer * 1568)) + (((int)threadIdx.y) * 49)) + ((int)threadIdx.x)) + 1176))];
    data_pack_shared[((((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) + 1372))] = data_pack[((((((((int)blockIdx.z) * 12544) + (ci_outer * 1568)) + (((int)threadIdx.y) * 49)) + ((int)threadIdx.x)) + 1372))];
    __syncthreads();
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[((((int)threadIdx.y) * 4))], data_pack_shared[(((int)threadIdx.x))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 1))], data_pack_shared[(((int)threadIdx.x))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 2))], data_pack_shared[(((int)threadIdx.x))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 3))], data_pack_shared[(((int)threadIdx.x))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 16))], data_pack_shared[((((int)threadIdx.x) + 49))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 17))], data_pack_shared[((((int)threadIdx.x) + 49))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 18))], data_pack_shared[((((int)threadIdx.x) + 49))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 19))], data_pack_shared[((((int)threadIdx.x) + 49))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 32))], data_pack_shared[((((int)threadIdx.x) + 98))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 33))], data_pack_shared[((((int)threadIdx.x) + 98))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 34))], data_pack_shared[((((int)threadIdx.x) + 98))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 35))], data_pack_shared[((((int)threadIdx.x) + 98))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 48))], data_pack_shared[((((int)threadIdx.x) + 147))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 49))], data_pack_shared[((((int)threadIdx.x) + 147))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 50))], data_pack_shared[((((int)threadIdx.x) + 147))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 51))], data_pack_shared[((((int)threadIdx.x) + 147))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 64))], data_pack_shared[((((int)threadIdx.x) + 196))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 65))], data_pack_shared[((((int)threadIdx.x) + 196))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 66))], data_pack_shared[((((int)threadIdx.x) + 196))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 67))], data_pack_shared[((((int)threadIdx.x) + 196))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 80))], data_pack_shared[((((int)threadIdx.x) + 245))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 81))], data_pack_shared[((((int)threadIdx.x) + 245))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 82))], data_pack_shared[((((int)threadIdx.x) + 245))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 83))], data_pack_shared[((((int)threadIdx.x) + 245))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 96))], data_pack_shared[((((int)threadIdx.x) + 294))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 97))], data_pack_shared[((((int)threadIdx.x) + 294))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 98))], data_pack_shared[((((int)threadIdx.x) + 294))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 99))], data_pack_shared[((((int)threadIdx.x) + 294))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 112))], data_pack_shared[((((int)threadIdx.x) + 343))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 113))], data_pack_shared[((((int)threadIdx.x) + 343))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 114))], data_pack_shared[((((int)threadIdx.x) + 343))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 115))], data_pack_shared[((((int)threadIdx.x) + 343))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 128))], data_pack_shared[((((int)threadIdx.x) + 392))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 129))], data_pack_shared[((((int)threadIdx.x) + 392))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 130))], data_pack_shared[((((int)threadIdx.x) + 392))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 131))], data_pack_shared[((((int)threadIdx.x) + 392))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 144))], data_pack_shared[((((int)threadIdx.x) + 441))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 145))], data_pack_shared[((((int)threadIdx.x) + 441))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 146))], data_pack_shared[((((int)threadIdx.x) + 441))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 147))], data_pack_shared[((((int)threadIdx.x) + 441))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 160))], data_pack_shared[((((int)threadIdx.x) + 490))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 161))], data_pack_shared[((((int)threadIdx.x) + 490))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 162))], data_pack_shared[((((int)threadIdx.x) + 490))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 163))], data_pack_shared[((((int)threadIdx.x) + 490))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 176))], data_pack_shared[((((int)threadIdx.x) + 539))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 177))], data_pack_shared[((((int)threadIdx.x) + 539))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 178))], data_pack_shared[((((int)threadIdx.x) + 539))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 179))], data_pack_shared[((((int)threadIdx.x) + 539))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 192))], data_pack_shared[((((int)threadIdx.x) + 588))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 193))], data_pack_shared[((((int)threadIdx.x) + 588))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 194))], data_pack_shared[((((int)threadIdx.x) + 588))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 195))], data_pack_shared[((((int)threadIdx.x) + 588))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 208))], data_pack_shared[((((int)threadIdx.x) + 637))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 209))], data_pack_shared[((((int)threadIdx.x) + 637))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 210))], data_pack_shared[((((int)threadIdx.x) + 637))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 211))], data_pack_shared[((((int)threadIdx.x) + 637))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 224))], data_pack_shared[((((int)threadIdx.x) + 686))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 225))], data_pack_shared[((((int)threadIdx.x) + 686))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 226))], data_pack_shared[((((int)threadIdx.x) + 686))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 227))], data_pack_shared[((((int)threadIdx.x) + 686))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 240))], data_pack_shared[((((int)threadIdx.x) + 735))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 241))], data_pack_shared[((((int)threadIdx.x) + 735))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 242))], data_pack_shared[((((int)threadIdx.x) + 735))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 243))], data_pack_shared[((((int)threadIdx.x) + 735))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 256))], data_pack_shared[((((int)threadIdx.x) + 784))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 257))], data_pack_shared[((((int)threadIdx.x) + 784))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 258))], data_pack_shared[((((int)threadIdx.x) + 784))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 259))], data_pack_shared[((((int)threadIdx.x) + 784))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 272))], data_pack_shared[((((int)threadIdx.x) + 833))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 273))], data_pack_shared[((((int)threadIdx.x) + 833))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 274))], data_pack_shared[((((int)threadIdx.x) + 833))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 275))], data_pack_shared[((((int)threadIdx.x) + 833))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 288))], data_pack_shared[((((int)threadIdx.x) + 882))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 289))], data_pack_shared[((((int)threadIdx.x) + 882))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 290))], data_pack_shared[((((int)threadIdx.x) + 882))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 291))], data_pack_shared[((((int)threadIdx.x) + 882))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 304))], data_pack_shared[((((int)threadIdx.x) + 931))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 305))], data_pack_shared[((((int)threadIdx.x) + 931))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 306))], data_pack_shared[((((int)threadIdx.x) + 931))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 307))], data_pack_shared[((((int)threadIdx.x) + 931))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 320))], data_pack_shared[((((int)threadIdx.x) + 980))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 321))], data_pack_shared[((((int)threadIdx.x) + 980))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 322))], data_pack_shared[((((int)threadIdx.x) + 980))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 323))], data_pack_shared[((((int)threadIdx.x) + 980))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 336))], data_pack_shared[((((int)threadIdx.x) + 1029))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 337))], data_pack_shared[((((int)threadIdx.x) + 1029))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 338))], data_pack_shared[((((int)threadIdx.x) + 1029))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 339))], data_pack_shared[((((int)threadIdx.x) + 1029))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 352))], data_pack_shared[((((int)threadIdx.x) + 1078))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 353))], data_pack_shared[((((int)threadIdx.x) + 1078))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 354))], data_pack_shared[((((int)threadIdx.x) + 1078))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 355))], data_pack_shared[((((int)threadIdx.x) + 1078))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 368))], data_pack_shared[((((int)threadIdx.x) + 1127))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 369))], data_pack_shared[((((int)threadIdx.x) + 1127))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 370))], data_pack_shared[((((int)threadIdx.x) + 1127))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 371))], data_pack_shared[((((int)threadIdx.x) + 1127))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 384))], data_pack_shared[((((int)threadIdx.x) + 1176))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 385))], data_pack_shared[((((int)threadIdx.x) + 1176))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 386))], data_pack_shared[((((int)threadIdx.x) + 1176))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 387))], data_pack_shared[((((int)threadIdx.x) + 1176))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 400))], data_pack_shared[((((int)threadIdx.x) + 1225))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 401))], data_pack_shared[((((int)threadIdx.x) + 1225))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 402))], data_pack_shared[((((int)threadIdx.x) + 1225))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 403))], data_pack_shared[((((int)threadIdx.x) + 1225))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 416))], data_pack_shared[((((int)threadIdx.x) + 1274))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 417))], data_pack_shared[((((int)threadIdx.x) + 1274))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 418))], data_pack_shared[((((int)threadIdx.x) + 1274))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 419))], data_pack_shared[((((int)threadIdx.x) + 1274))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 432))], data_pack_shared[((((int)threadIdx.x) + 1323))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 433))], data_pack_shared[((((int)threadIdx.x) + 1323))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 434))], data_pack_shared[((((int)threadIdx.x) + 1323))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 435))], data_pack_shared[((((int)threadIdx.x) + 1323))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 448))], data_pack_shared[((((int)threadIdx.x) + 1372))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 449))], data_pack_shared[((((int)threadIdx.x) + 1372))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 450))], data_pack_shared[((((int)threadIdx.x) + 1372))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 451))], data_pack_shared[((((int)threadIdx.x) + 1372))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 464))], data_pack_shared[((((int)threadIdx.x) + 1421))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 465))], data_pack_shared[((((int)threadIdx.x) + 1421))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 466))], data_pack_shared[((((int)threadIdx.x) + 1421))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 467))], data_pack_shared[((((int)threadIdx.x) + 1421))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 480))], data_pack_shared[((((int)threadIdx.x) + 1470))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 481))], data_pack_shared[((((int)threadIdx.x) + 1470))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 482))], data_pack_shared[((((int)threadIdx.x) + 1470))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 483))], data_pack_shared[((((int)threadIdx.x) + 1470))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 496))], data_pack_shared[((((int)threadIdx.x) + 1519))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 497))], data_pack_shared[((((int)threadIdx.x) + 1519))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 498))], data_pack_shared[((((int)threadIdx.x) + 1519))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 4) + 499))], data_pack_shared[((((int)threadIdx.x) + 1519))], bgemm_local[(3)]);
  }
  bgemm[(((((((int)blockIdx.z) * 12544) + (((int)blockIdx.y) * 784)) + (((int)threadIdx.y) * 196)) + ((int)threadIdx.x)))] = bgemm_local[(0)];
  bgemm[((((((((int)blockIdx.z) * 12544) + (((int)blockIdx.y) * 784)) + (((int)threadIdx.y) * 196)) + ((int)threadIdx.x)) + 49))] = bgemm_local[(1)];
  bgemm[((((((((int)blockIdx.z) * 12544) + (((int)blockIdx.y) * 784)) + (((int)threadIdx.y) * 196)) + ((int)threadIdx.x)) + 98))] = bgemm_local[(2)];
  bgemm[((((((((int)blockIdx.z) * 12544) + (((int)blockIdx.y) * 784)) + (((int)threadIdx.y) * 196)) + ((int)threadIdx.x)) + 147))] = bgemm_local[(3)];
}

extern "C" __global__ void fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_2_kernel0(float* __restrict__ placeholder, float* __restrict__ data_pack) {
  float d[16];
  float data_pack_local[16];
  for (int eps = 0; eps < 4; ++eps) {
    for (int nu = 0; nu < 4; ++nu) {
      d[(((eps * 4) + nu))] = (((((1 <= ((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 196) / 14) * 2) + eps)) && (((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 196) / 14) * 2) + eps) < 29)) && (1 <= (((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 14) * 2) + nu))) && ((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 14) * 2) + nu) < 29)) ? placeholder[(((((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 14) * 56) + (eps * 28)) + ((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 14) * 2)) + nu) - 29))] : 0.000000e+00f);
    }
  }
  data_pack_local[(0)] = 0.000000e+00f;
  data_pack_local[(0)] = (data_pack_local[(0)] + d[(0)]);
  data_pack_local[(0)] = __ocml_fma_f32(d[(2)], -1.000000e+00f, data_pack_local[(0)]);
  data_pack_local[(0)] = __ocml_fma_f32(d[(8)], -1.000000e+00f, data_pack_local[(0)]);
  data_pack_local[(0)] = __ocml_fma_f32((d[(10)] * -1.000000e+00f), -1.000000e+00f, data_pack_local[(0)]);
  data_pack_local[(1)] = 0.000000e+00f;
  data_pack_local[(1)] = __ocml_fma_f32(d[(1)], -1.000000e+00f, data_pack_local[(1)]);
  data_pack_local[(1)] = (data_pack_local[(1)] + d[(2)]);
  data_pack_local[(1)] = __ocml_fma_f32((d[(9)] * -1.000000e+00f), -1.000000e+00f, data_pack_local[(1)]);
  data_pack_local[(1)] = __ocml_fma_f32(d[(10)], -1.000000e+00f, data_pack_local[(1)]);
  data_pack_local[(2)] = 0.000000e+00f;
  data_pack_local[(2)] = (data_pack_local[(2)] + d[(1)]);
  data_pack_local[(2)] = (data_pack_local[(2)] + d[(2)]);
  data_pack_local[(2)] = __ocml_fma_f32(d[(9)], -1.000000e+00f, data_pack_local[(2)]);
  data_pack_local[(2)] = __ocml_fma_f32(d[(10)], -1.000000e+00f, data_pack_local[(2)]);
  data_pack_local[(3)] = 0.000000e+00f;
  data_pack_local[(3)] = __ocml_fma_f32(d[(1)], -1.000000e+00f, data_pack_local[(3)]);
  data_pack_local[(3)] = (data_pack_local[(3)] + d[(3)]);
  data_pack_local[(3)] = __ocml_fma_f32((d[(9)] * -1.000000e+00f), -1.000000e+00f, data_pack_local[(3)]);
  data_pack_local[(3)] = __ocml_fma_f32(d[(11)], -1.000000e+00f, data_pack_local[(3)]);
  data_pack_local[(4)] = 0.000000e+00f;
  data_pack_local[(4)] = __ocml_fma_f32(d[(4)], -1.000000e+00f, data_pack_local[(4)]);
  data_pack_local[(4)] = __ocml_fma_f32((d[(6)] * -1.000000e+00f), -1.000000e+00f, data_pack_local[(4)]);
  data_pack_local[(4)] = (data_pack_local[(4)] + d[(8)]);
  data_pack_local[(4)] = __ocml_fma_f32(d[(10)], -1.000000e+00f, data_pack_local[(4)]);
  data_pack_local[(5)] = 0.000000e+00f;
  data_pack_local[(5)] = __ocml_fma_f32((d[(5)] * -1.000000e+00f), -1.000000e+00f, data_pack_local[(5)]);
  data_pack_local[(5)] = __ocml_fma_f32(d[(6)], -1.000000e+00f, data_pack_local[(5)]);
  data_pack_local[(5)] = __ocml_fma_f32(d[(9)], -1.000000e+00f, data_pack_local[(5)]);
  data_pack_local[(5)] = (data_pack_local[(5)] + d[(10)]);
  data_pack_local[(6)] = 0.000000e+00f;
  data_pack_local[(6)] = __ocml_fma_f32(d[(5)], -1.000000e+00f, data_pack_local[(6)]);
  data_pack_local[(6)] = __ocml_fma_f32(d[(6)], -1.000000e+00f, data_pack_local[(6)]);
  data_pack_local[(6)] = (data_pack_local[(6)] + d[(9)]);
  data_pack_local[(6)] = (data_pack_local[(6)] + d[(10)]);
  data_pack_local[(7)] = 0.000000e+00f;
  data_pack_local[(7)] = __ocml_fma_f32((d[(5)] * -1.000000e+00f), -1.000000e+00f, data_pack_local[(7)]);
  data_pack_local[(7)] = __ocml_fma_f32(d[(7)], -1.000000e+00f, data_pack_local[(7)]);
  data_pack_local[(7)] = __ocml_fma_f32(d[(9)], -1.000000e+00f, data_pack_local[(7)]);
  data_pack_local[(7)] = (data_pack_local[(7)] + d[(11)]);
  data_pack_local[(8)] = 0.000000e+00f;
  data_pack_local[(8)] = (data_pack_local[(8)] + d[(4)]);
  data_pack_local[(8)] = __ocml_fma_f32(d[(6)], -1.000000e+00f, data_pack_local[(8)]);
  data_pack_local[(8)] = (data_pack_local[(8)] + d[(8)]);
  data_pack_local[(8)] = __ocml_fma_f32(d[(10)], -1.000000e+00f, data_pack_local[(8)]);
  data_pack_local[(9)] = 0.000000e+00f;
  data_pack_local[(9)] = __ocml_fma_f32(d[(5)], -1.000000e+00f, data_pack_local[(9)]);
  data_pack_local[(9)] = (data_pack_local[(9)] + d[(6)]);
  data_pack_local[(9)] = __ocml_fma_f32(d[(9)], -1.000000e+00f, data_pack_local[(9)]);
  data_pack_local[(9)] = (data_pack_local[(9)] + d[(10)]);
  data_pack_local[(10)] = 0.000000e+00f;
  data_pack_local[(10)] = (data_pack_local[(10)] + d[(5)]);
  data_pack_local[(10)] = (data_pack_local[(10)] + d[(6)]);
  data_pack_local[(10)] = (data_pack_local[(10)] + d[(9)]);
  data_pack_local[(10)] = (data_pack_local[(10)] + d[(10)]);
  data_pack_local[(11)] = 0.000000e+00f;
  data_pack_local[(11)] = __ocml_fma_f32(d[(5)], -1.000000e+00f, data_pack_local[(11)]);
  data_pack_local[(11)] = (data_pack_local[(11)] + d[(7)]);
  data_pack_local[(11)] = __ocml_fma_f32(d[(9)], -1.000000e+00f, data_pack_local[(11)]);
  data_pack_local[(11)] = (data_pack_local[(11)] + d[(11)]);
  data_pack_local[(12)] = 0.000000e+00f;
  data_pack_local[(12)] = __ocml_fma_f32(d[(4)], -1.000000e+00f, data_pack_local[(12)]);
  data_pack_local[(12)] = __ocml_fma_f32((d[(6)] * -1.000000e+00f), -1.000000e+00f, data_pack_local[(12)]);
  data_pack_local[(12)] = (data_pack_local[(12)] + d[(12)]);
  data_pack_local[(12)] = __ocml_fma_f32(d[(14)], -1.000000e+00f, data_pack_local[(12)]);
  data_pack_local[(13)] = 0.000000e+00f;
  data_pack_local[(13)] = __ocml_fma_f32((d[(5)] * -1.000000e+00f), -1.000000e+00f, data_pack_local[(13)]);
  data_pack_local[(13)] = __ocml_fma_f32(d[(6)], -1.000000e+00f, data_pack_local[(13)]);
  data_pack_local[(13)] = __ocml_fma_f32(d[(13)], -1.000000e+00f, data_pack_local[(13)]);
  data_pack_local[(13)] = (data_pack_local[(13)] + d[(14)]);
  data_pack_local[(14)] = 0.000000e+00f;
  data_pack_local[(14)] = __ocml_fma_f32(d[(5)], -1.000000e+00f, data_pack_local[(14)]);
  data_pack_local[(14)] = __ocml_fma_f32(d[(6)], -1.000000e+00f, data_pack_local[(14)]);
  data_pack_local[(14)] = (data_pack_local[(14)] + d[(13)]);
  data_pack_local[(14)] = (data_pack_local[(14)] + d[(14)]);
  data_pack_local[(15)] = 0.000000e+00f;
  data_pack_local[(15)] = __ocml_fma_f32((d[(5)] * -1.000000e+00f), -1.000000e+00f, data_pack_local[(15)]);
  data_pack_local[(15)] = __ocml_fma_f32(d[(7)], -1.000000e+00f, data_pack_local[(15)]);
  data_pack_local[(15)] = __ocml_fma_f32(d[(13)], -1.000000e+00f, data_pack_local[(15)]);
  data_pack_local[(15)] = (data_pack_local[(15)] + d[(15)]);
  for (int eps1 = 0; eps1 < 4; ++eps1) {
    for (int nu1 = 0; nu1 < 4; ++nu1) {
      data_pack[(((((eps1 * 100352) + (nu1 * 25088)) + (((int)blockIdx.x) * 128)) + ((int)threadIdx.x)))] = data_pack_local[(((eps1 * 4) + nu1))];
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_add_add_nn_relu_1_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2, float* __restrict__ placeholder3) {
  float compute[8];
  __shared__ float pad_temp_shared[896];
  __shared__ float placeholder_shared[256];
  for (int xx_init = 0; xx_init < 2; ++xx_init) {
    compute[(xx_init)] = 0.000000e+00f;
    compute[((xx_init + 2))] = 0.000000e+00f;
    compute[((xx_init + 4))] = 0.000000e+00f;
    compute[((xx_init + 6))] = 0.000000e+00f;
  }
  for (int rc_outer = 0; rc_outer < 16; ++rc_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      pad_temp_shared[(((((((int)threadIdx.z) * 112) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))] = placeholder[(((((((rc_outer * 6272) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))];
    }
    if ((((((int)threadIdx.z) * 4) + (((int)threadIdx.x) >> 3)) + ((int)threadIdx.y)) < 32) {
      if ((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x)) < 256) {
        if (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) < 32) {
          if (((int)threadIdx.x) < 8) {
            placeholder_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x)))] = placeholder1[((((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 512)) + (((int)threadIdx.y) * 128)) + (rc_outer * 8)) + ((int)threadIdx.x)))];
          }
        }
      }
    }
    __syncthreads();
    for (int rc_inner = 0; rc_inner < 8; ++rc_inner) {
      for (int xx = 0; xx < 2; ++xx) {
        compute[(xx)] = __ocml_fma_f32(pad_temp_shared[(((((rc_inner * 112) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 2)) + xx))], placeholder_shared[(((((int)threadIdx.z) * 8) + rc_inner))], compute[(xx)]);
        compute[((xx + 2))] = __ocml_fma_f32(pad_temp_shared[(((((rc_inner * 112) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 2)) + xx))], placeholder_shared[((((((int)threadIdx.z) * 8) + rc_inner) + 64))], compute[((xx + 2))]);
        compute[((xx + 4))] = __ocml_fma_f32(pad_temp_shared[(((((rc_inner * 112) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 2)) + xx))], placeholder_shared[((((((int)threadIdx.z) * 8) + rc_inner) + 128))], compute[((xx + 4))]);
        compute[((xx + 6))] = __ocml_fma_f32(pad_temp_shared[(((((rc_inner * 112) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 2)) + xx))], placeholder_shared[((((((int)threadIdx.z) * 8) + rc_inner) + 192))], compute[((xx + 6))]);
      }
    }
  }
  for (int ax3_inner_inner_inner = 0; ax3_inner_inner_inner < 2; ++ax3_inner_inner_inner) {
    T_relu[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 2)) + ax3_inner_inner_inner))] = max(((compute[(ax3_inner_inner_inner)] + placeholder2[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 2)) + ax3_inner_inner_inner))]) + placeholder3[(((((int)blockIdx.z) * 32) + ((int)threadIdx.z)))]), 0.000000e+00f);
    T_relu[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 2)) + ax3_inner_inner_inner) + 6272))] = max(((compute[((ax3_inner_inner_inner + 2))] + placeholder2[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 2)) + ax3_inner_inner_inner) + 6272))]) + placeholder3[((((((int)blockIdx.z) * 32) + ((int)threadIdx.z)) + 8))]), 0.000000e+00f);
    T_relu[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 2)) + ax3_inner_inner_inner) + 12544))] = max(((compute[((ax3_inner_inner_inner + 4))] + placeholder2[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 2)) + ax3_inner_inner_inner) + 12544))]) + placeholder3[((((((int)blockIdx.z) * 32) + ((int)threadIdx.z)) + 16))]), 0.000000e+00f);
    T_relu[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 2)) + ax3_inner_inner_inner) + 18816))] = max(((compute[((ax3_inner_inner_inner + 6))] + placeholder2[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 2)) + ax3_inner_inner_inner) + 18816))]) + placeholder3[((((((int)blockIdx.z) * 32) + ((int)threadIdx.z)) + 24))]), 0.000000e+00f);
  }
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_4_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  float compute[2];
  __shared__ float pad_temp_shared[896];
  __shared__ float placeholder_shared[256];
  compute[(0)] = 0.000000e+00f;
  compute[(1)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 32; ++rc_outer) {
    __syncthreads();
    pad_temp_shared[((((((int)threadIdx.z) * 56) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 2)))] = placeholder[((((((rc_outer * 12544) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 56)) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 2)))];
    pad_temp_shared[(((((((int)threadIdx.z) * 56) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 2)) + 1))] = placeholder[(((((((rc_outer * 12544) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 56)) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 2)) + 1))];
    if (((((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) >> 4) + ((int)threadIdx.z)) < 16) {
      if ((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x)) < 256) {
        if (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) < 16) {
          if (((int)threadIdx.x) < 8) {
            placeholder_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x)))] = placeholder1[((((((((int)blockIdx.z) * 8192) + (((int)threadIdx.z) * 512)) + (rc_outer * 16)) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x)))];
          }
        }
      }
    }
    __syncthreads();
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)))], placeholder_shared[((((int)threadIdx.z) * 16))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 1))], placeholder_shared[((((int)threadIdx.z) * 16))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 56))], placeholder_shared[(((((int)threadIdx.z) * 16) + 1))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 57))], placeholder_shared[(((((int)threadIdx.z) * 16) + 1))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 112))], placeholder_shared[(((((int)threadIdx.z) * 16) + 2))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 113))], placeholder_shared[(((((int)threadIdx.z) * 16) + 2))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 168))], placeholder_shared[(((((int)threadIdx.z) * 16) + 3))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 169))], placeholder_shared[(((((int)threadIdx.z) * 16) + 3))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 224))], placeholder_shared[(((((int)threadIdx.z) * 16) + 4))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 225))], placeholder_shared[(((((int)threadIdx.z) * 16) + 4))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 280))], placeholder_shared[(((((int)threadIdx.z) * 16) + 5))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 281))], placeholder_shared[(((((int)threadIdx.z) * 16) + 5))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 336))], placeholder_shared[(((((int)threadIdx.z) * 16) + 6))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 337))], placeholder_shared[(((((int)threadIdx.z) * 16) + 6))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 392))], placeholder_shared[(((((int)threadIdx.z) * 16) + 7))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 393))], placeholder_shared[(((((int)threadIdx.z) * 16) + 7))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 448))], placeholder_shared[(((((int)threadIdx.z) * 16) + 8))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 449))], placeholder_shared[(((((int)threadIdx.z) * 16) + 8))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 504))], placeholder_shared[(((((int)threadIdx.z) * 16) + 9))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 505))], placeholder_shared[(((((int)threadIdx.z) * 16) + 9))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 560))], placeholder_shared[(((((int)threadIdx.z) * 16) + 10))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 561))], placeholder_shared[(((((int)threadIdx.z) * 16) + 10))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 616))], placeholder_shared[(((((int)threadIdx.z) * 16) + 11))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 617))], placeholder_shared[(((((int)threadIdx.z) * 16) + 11))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 672))], placeholder_shared[(((((int)threadIdx.z) * 16) + 12))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 673))], placeholder_shared[(((((int)threadIdx.z) * 16) + 12))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 728))], placeholder_shared[(((((int)threadIdx.z) * 16) + 13))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 729))], placeholder_shared[(((((int)threadIdx.z) * 16) + 13))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 784))], placeholder_shared[(((((int)threadIdx.z) * 16) + 14))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 785))], placeholder_shared[(((((int)threadIdx.z) * 16) + 14))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 840))], placeholder_shared[(((((int)threadIdx.z) * 16) + 15))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 841))], placeholder_shared[(((((int)threadIdx.z) * 16) + 15))], compute[(1)]);
  }
  T_relu[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 56)) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 2)))] = max((compute[(0)] + placeholder2[(((((int)blockIdx.z) * 16) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[(((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 56)) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 2)) + 1))] = max((compute[(1)] + placeholder2[(((((int)blockIdx.z) * 16) + ((int)threadIdx.z)))]), 0.000000e+00f);
}

extern "C" __global__ void fused_nn_conv2d_add_2_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_add, float* __restrict__ placeholder2) {
  float compute[8];
  __shared__ float pad_temp_shared[896];
  __shared__ float placeholder_shared[256];
  for (int xx_init = 0; xx_init < 2; ++xx_init) {
    compute[(xx_init)] = 0.000000e+00f;
    compute[((xx_init + 2))] = 0.000000e+00f;
    compute[((xx_init + 4))] = 0.000000e+00f;
    compute[((xx_init + 6))] = 0.000000e+00f;
  }
  for (int rc_outer = 0; rc_outer < 16; ++rc_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      pad_temp_shared[(((((((int)threadIdx.z) * 112) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))] = placeholder[(((((((rc_outer * 6272) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))];
    }
    if ((((((int)threadIdx.z) * 4) + (((int)threadIdx.x) >> 3)) + ((int)threadIdx.y)) < 32) {
      if ((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x)) < 256) {
        if (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) < 32) {
          if (((int)threadIdx.x) < 8) {
            placeholder_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x)))] = placeholder1[((((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 512)) + (((int)threadIdx.y) * 128)) + (rc_outer * 8)) + ((int)threadIdx.x)))];
          }
        }
      }
    }
    __syncthreads();
    for (int rc_inner = 0; rc_inner < 8; ++rc_inner) {
      for (int xx = 0; xx < 2; ++xx) {
        compute[(xx)] = __ocml_fma_f32(pad_temp_shared[(((((rc_inner * 112) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 2)) + xx))], placeholder_shared[(((((int)threadIdx.z) * 8) + rc_inner))], compute[(xx)]);
        compute[((xx + 2))] = __ocml_fma_f32(pad_temp_shared[(((((rc_inner * 112) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 2)) + xx))], placeholder_shared[((((((int)threadIdx.z) * 8) + rc_inner) + 64))], compute[((xx + 2))]);
        compute[((xx + 4))] = __ocml_fma_f32(pad_temp_shared[(((((rc_inner * 112) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 2)) + xx))], placeholder_shared[((((((int)threadIdx.z) * 8) + rc_inner) + 128))], compute[((xx + 4))]);
        compute[((xx + 6))] = __ocml_fma_f32(pad_temp_shared[(((((rc_inner * 112) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 2)) + xx))], placeholder_shared[((((((int)threadIdx.z) * 8) + rc_inner) + 192))], compute[((xx + 6))]);
      }
    }
  }
  for (int ax3_inner_inner_inner = 0; ax3_inner_inner_inner < 2; ++ax3_inner_inner_inner) {
    T_add[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 2)) + ax3_inner_inner_inner))] = (compute[(ax3_inner_inner_inner)] + placeholder2[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 2)) + ax3_inner_inner_inner))]);
    T_add[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 2)) + ax3_inner_inner_inner) + 6272))] = (compute[((ax3_inner_inner_inner + 2))] + placeholder2[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 2)) + ax3_inner_inner_inner) + 6272))]);
    T_add[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 2)) + ax3_inner_inner_inner) + 12544))] = (compute[((ax3_inner_inner_inner + 4))] + placeholder2[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 2)) + ax3_inner_inner_inner) + 12544))]);
    T_add[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 2)) + ax3_inner_inner_inner) + 18816))] = (compute[((ax3_inner_inner_inner + 6))] + placeholder2[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 2)) + ax3_inner_inner_inner) + 18816))]);
  }
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_3_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  float compute[1];
  __shared__ float pad_temp_shared[864];
  __shared__ float placeholder_shared[1024];
  compute[(0)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 16; ++rc_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      if (((((((int)threadIdx.x) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 27) + ((int)threadIdx.z)) < 32) {
        if ((((((int)threadIdx.z) * 27) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 864) {
          if (((((int)threadIdx.x) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 27) {
            pad_temp_shared[((((((int)threadIdx.z) * 27) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))] = placeholder[((((((rc_outer * 25088) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 56)) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))];
          }
        }
      }
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 < 3; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) {
      if (((((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) >> 5) + ((int)threadIdx.z)) < 32) {
        if ((((((int)threadIdx.z) * 32) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) < 1024) {
          if (((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) < 32) {
            placeholder_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1))] = placeholder1[((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 512)) + (rc_outer * 32)) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1))];
          }
        }
      }
    }
    __syncthreads();
    for (int rc_inner = 0; rc_inner < 32; ++rc_inner) {
      compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((rc_inner * 27) + (((int)threadIdx.x) * 2)))], placeholder_shared[(((((int)threadIdx.z) * 32) + rc_inner))], compute[(0)]);
    }
  }
  T_relu[(((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 14)) + ((int)threadIdx.x)))] = max((compute[(0)] + placeholder2[(((((int)blockIdx.z) * 32) + ((int)threadIdx.z)))]), 0.000000e+00f);
}

extern "C" __global__ void fused_nn_conv2d_add_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_add, float* __restrict__ placeholder2) {
  float compute[2];
  __shared__ float pad_temp_shared[784];
  __shared__ float placeholder_shared[256];
  compute[(0)] = 0.000000e+00f;
  compute[(1)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 32; ++rc_outer) {
    __syncthreads();
    pad_temp_shared[((((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)))] = placeholder[(((((rc_outer * 784) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)))];
    pad_temp_shared[(((((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)) + 1))] = placeholder[(((((rc_outer * 784) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 14)) + ((((int)threadIdx.x) * 2) + 1)))];
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
  T_add[(((((((int)blockIdx.z) * 784) + (((int)threadIdx.z) * 49)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)))] = (compute[(0)] + placeholder2[(((((((int)blockIdx.z) * 784) + (((int)threadIdx.z) * 49)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)))]);
  T_add[((((((((int)blockIdx.z) * 784) + (((int)threadIdx.z) * 49)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) + 392))] = (compute[(1)] + placeholder2[((((((((int)blockIdx.z) * 784) + (((int)threadIdx.z) * 49)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) + 392))]);
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_1_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  float compute[1];
  __shared__ float pad_temp_shared[416];
  __shared__ float placeholder_shared[1024];
  compute[(0)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 32; ++rc_outer) {
    __syncthreads();
    pad_temp_shared[(((((int)threadIdx.z) * 13) + (((int)threadIdx.x) * 2)))] = placeholder[(((((rc_outer * 6272) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 28)) + (((int)threadIdx.x) * 2)))];
    if (((((((int)threadIdx.x) * 2) + 1) / 13) + ((int)threadIdx.z)) < 32) {
      if (((((int)threadIdx.z) * 13) + (((int)threadIdx.x) * 2)) < 415) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[((((((int)threadIdx.z) * 13) + (((int)threadIdx.x) * 2)) + 1))] = placeholder[((((((rc_outer * 6272) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 28)) + (((int)threadIdx.x) * 2)) + 1))];
        }
      }
    }
    placeholder_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.x) * 5)))] = placeholder1[(((((((int)blockIdx.z) * 32768) + (((int)threadIdx.z) * 1024)) + (rc_outer * 32)) + (((int)threadIdx.x) * 5)))];
    placeholder_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.x) * 5)) + 1))] = placeholder1[((((((((int)blockIdx.z) * 32768) + (((int)threadIdx.z) * 1024)) + (rc_outer * 32)) + (((int)threadIdx.x) * 5)) + 1))];
    if (((((((int)threadIdx.x) * 5) + 2) >> 5) + ((int)threadIdx.z)) < 32) {
      if (((((int)threadIdx.z) * 32) + (((int)threadIdx.x) * 5)) < 1022) {
        if (((int)threadIdx.x) < 6) {
          placeholder_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.x) * 5)) + 2))] = placeholder1[((((((((int)blockIdx.z) * 32768) + (((int)threadIdx.z) * 1024)) + (rc_outer * 32)) + (((int)threadIdx.x) * 5)) + 2))];
        }
      }
    }
    if (((((((int)threadIdx.x) * 5) + 3) >> 5) + ((int)threadIdx.z)) < 32) {
      if (((((int)threadIdx.z) * 32) + (((int)threadIdx.x) * 5)) < 1021) {
        if (((int)threadIdx.x) < 6) {
          placeholder_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.x) * 5)) + 3))] = placeholder1[((((((((int)blockIdx.z) * 32768) + (((int)threadIdx.z) * 1024)) + (rc_outer * 32)) + (((int)threadIdx.x) * 5)) + 3))];
        }
      }
    }
    if (((((((int)threadIdx.x) * 5) + 4) >> 5) + ((int)threadIdx.z)) < 32) {
      if (((((int)threadIdx.z) * 32) + (((int)threadIdx.x) * 5)) < 1020) {
        if (((int)threadIdx.x) < 6) {
          placeholder_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.x) * 5)) + 4))] = placeholder1[((((((((int)blockIdx.z) * 32768) + (((int)threadIdx.z) * 1024)) + (rc_outer * 32)) + (((int)threadIdx.x) * 5)) + 4))];
        }
      }
    }
    __syncthreads();
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) * 2))], placeholder_shared[((((int)threadIdx.z) * 32))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.x) * 2) + 13))], placeholder_shared[(((((int)threadIdx.z) * 32) + 1))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.x) * 2) + 26))], placeholder_shared[(((((int)threadIdx.z) * 32) + 2))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.x) * 2) + 39))], placeholder_shared[(((((int)threadIdx.z) * 32) + 3))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.x) * 2) + 52))], placeholder_shared[(((((int)threadIdx.z) * 32) + 4))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.x) * 2) + 65))], placeholder_shared[(((((int)threadIdx.z) * 32) + 5))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.x) * 2) + 78))], placeholder_shared[(((((int)threadIdx.z) * 32) + 6))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.x) * 2) + 91))], placeholder_shared[(((((int)threadIdx.z) * 32) + 7))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.x) * 2) + 104))], placeholder_shared[(((((int)threadIdx.z) * 32) + 8))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.x) * 2) + 117))], placeholder_shared[(((((int)threadIdx.z) * 32) + 9))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.x) * 2) + 130))], placeholder_shared[(((((int)threadIdx.z) * 32) + 10))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.x) * 2) + 143))], placeholder_shared[(((((int)threadIdx.z) * 32) + 11))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.x) * 2) + 156))], placeholder_shared[(((((int)threadIdx.z) * 32) + 12))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.x) * 2) + 169))], placeholder_shared[(((((int)threadIdx.z) * 32) + 13))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.x) * 2) + 182))], placeholder_shared[(((((int)threadIdx.z) * 32) + 14))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.x) * 2) + 195))], placeholder_shared[(((((int)threadIdx.z) * 32) + 15))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.x) * 2) + 208))], placeholder_shared[(((((int)threadIdx.z) * 32) + 16))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.x) * 2) + 221))], placeholder_shared[(((((int)threadIdx.z) * 32) + 17))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.x) * 2) + 234))], placeholder_shared[(((((int)threadIdx.z) * 32) + 18))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.x) * 2) + 247))], placeholder_shared[(((((int)threadIdx.z) * 32) + 19))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.x) * 2) + 260))], placeholder_shared[(((((int)threadIdx.z) * 32) + 20))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.x) * 2) + 273))], placeholder_shared[(((((int)threadIdx.z) * 32) + 21))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.x) * 2) + 286))], placeholder_shared[(((((int)threadIdx.z) * 32) + 22))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.x) * 2) + 299))], placeholder_shared[(((((int)threadIdx.z) * 32) + 23))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.x) * 2) + 312))], placeholder_shared[(((((int)threadIdx.z) * 32) + 24))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.x) * 2) + 325))], placeholder_shared[(((((int)threadIdx.z) * 32) + 25))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.x) * 2) + 338))], placeholder_shared[(((((int)threadIdx.z) * 32) + 26))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.x) * 2) + 351))], placeholder_shared[(((((int)threadIdx.z) * 32) + 27))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.x) * 2) + 364))], placeholder_shared[(((((int)threadIdx.z) * 32) + 28))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.x) * 2) + 377))], placeholder_shared[(((((int)threadIdx.z) * 32) + 29))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.x) * 2) + 390))], placeholder_shared[(((((int)threadIdx.z) * 32) + 30))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.x) * 2) + 403))], placeholder_shared[(((((int)threadIdx.z) * 32) + 31))], compute[(0)]);
  }
  T_relu[(((((((int)blockIdx.z) * 1568) + (((int)threadIdx.z) * 49)) + (((int)blockIdx.y) * 7)) + ((int)threadIdx.x)))] = max((compute[(0)] + placeholder2[(((((int)blockIdx.z) * 32) + ((int)threadIdx.z)))]), 0.000000e+00f);
}

extern "C" __global__ void fused_add_nn_relu_kernel0(float* __restrict__ T_relu, float* __restrict__ placeholder, float* __restrict__ placeholder1) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) < 100352) {
      T_relu[((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)))] = max((placeholder[((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)))] + placeholder1[(((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) / 49))]), 0.000000e+00f);
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_2_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  float compute[1];
  __shared__ float pad_temp_shared[448];
  __shared__ float placeholder_shared[256];
  compute[(0)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 64; ++rc_outer) {
    __syncthreads();
    pad_temp_shared[((((((int)threadIdx.z) * 28) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)))] = placeholder[((((((rc_outer * 3136) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 28)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)))];
    if (((((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) >> 4) + ((int)threadIdx.z)) < 16) {
      if ((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x)) < 256) {
        if (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) < 16) {
          if (((int)threadIdx.x) < 8) {
            placeholder_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x)))] = placeholder1[((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 1024)) + (rc_outer * 16)) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x)))];
          }
        }
      }
    }
    __syncthreads();
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 14) + ((int)threadIdx.x)))], placeholder_shared[((((int)threadIdx.z) * 16))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) + 28))], placeholder_shared[(((((int)threadIdx.z) * 16) + 1))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) + 56))], placeholder_shared[(((((int)threadIdx.z) * 16) + 2))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) + 84))], placeholder_shared[(((((int)threadIdx.z) * 16) + 3))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) + 112))], placeholder_shared[(((((int)threadIdx.z) * 16) + 4))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) + 140))], placeholder_shared[(((((int)threadIdx.z) * 16) + 5))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) + 168))], placeholder_shared[(((((int)threadIdx.z) * 16) + 6))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) + 196))], placeholder_shared[(((((int)threadIdx.z) * 16) + 7))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) + 224))], placeholder_shared[(((((int)threadIdx.z) * 16) + 8))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) + 252))], placeholder_shared[(((((int)threadIdx.z) * 16) + 9))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) + 280))], placeholder_shared[(((((int)threadIdx.z) * 16) + 10))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) + 308))], placeholder_shared[(((((int)threadIdx.z) * 16) + 11))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) + 336))], placeholder_shared[(((((int)threadIdx.z) * 16) + 12))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) + 364))], placeholder_shared[(((((int)threadIdx.z) * 16) + 13))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) + 392))], placeholder_shared[(((((int)threadIdx.z) * 16) + 14))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) + 420))], placeholder_shared[(((((int)threadIdx.z) * 16) + 15))], compute[(0)]);
  }
  T_relu[((((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 28)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)))] = max((compute[(0)] + placeholder2[(((((int)blockIdx.z) * 16) + ((int)threadIdx.z)))]), 0.000000e+00f);
}

extern "C" __global__ void fused_nn_conv2d_add_add_nn_relu_2_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2, float* __restrict__ placeholder3) {
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
  for (int rc_outer = 0; rc_outer < 4; ++rc_outer) {
    __syncthreads();
    pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)))] = placeholder[(((((((rc_outer * 50176) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)))];
    pad_temp_shared[(((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1))] = placeholder[((((((((rc_outer * 50176) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 1))];
    placeholder_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))] = placeholder1[((((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 64)) + (rc_outer * 16)) + (((int)threadIdx.x) * 4)))];
    placeholder_shared[(((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 1))] = placeholder1[(((((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 64)) + (rc_outer * 16)) + (((int)threadIdx.x) * 4)) + 1))];
    placeholder_shared[(((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 2))] = placeholder1[(((((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 64)) + (rc_outer * 16)) + (((int)threadIdx.x) * 4)) + 2))];
    placeholder_shared[(((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 3))] = placeholder1[(((((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 64)) + (rc_outer * 16)) + (((int)threadIdx.x) * 4)) + 3))];
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
  T_relu[(((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)))] = max(((compute[(0)] + placeholder2[(((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)))]) + placeholder3[(((((int)blockIdx.z) * 64) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 50176))] = max(((compute[(2)] + placeholder2[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 50176))]) + placeholder3[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 16))]), 0.000000e+00f);
  T_relu[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 100352))] = max(((compute[(4)] + placeholder2[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 100352))]) + placeholder3[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 32))]), 0.000000e+00f);
  T_relu[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 150528))] = max(((compute[(6)] + placeholder2[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 150528))]) + placeholder3[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 48))]), 0.000000e+00f);
  T_relu[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 1))] = max(((compute[(1)] + placeholder2[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 1))]) + placeholder3[(((((int)blockIdx.z) * 64) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 50177))] = max(((compute[(3)] + placeholder2[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 50177))]) + placeholder3[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 16))]), 0.000000e+00f);
  T_relu[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 100353))] = max(((compute[(5)] + placeholder2[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 100353))]) + placeholder3[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 32))]), 0.000000e+00f);
  T_relu[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 150529))] = max(((compute[(7)] + placeholder2[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 150529))]) + placeholder3[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 48))]), 0.000000e+00f);
}

extern "C" __global__ void fused_nn_conv2d_2_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ compute) {
  float compute_local[2];
  __shared__ float pad_temp_shared[1296];
  __shared__ float placeholder_shared[1024];
  compute_local[(0)] = 0.000000e+00f;
  compute_local[(1)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 32; ++rc_outer) {
    __syncthreads();
    if ((((((int)threadIdx.z) * 21) + (((int)threadIdx.y) * 11)) + (((int)threadIdx.x) * 2)) < 1296) {
      if (((((int)threadIdx.y) * 11) + (((int)threadIdx.x) * 2)) < 21) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[((((((int)threadIdx.z) * 21) + (((int)threadIdx.y) * 11)) + (((int)threadIdx.x) * 2)))] = placeholder[((((((rc_outer * 12544) + (((((((int)threadIdx.z) * 21) + (((int)threadIdx.y) * 11)) + (((int)threadIdx.x) * 2)) / 81) * 784)) + (((int)blockIdx.y) * 112)) + ((((((((int)threadIdx.z) * 21) + (((int)threadIdx.y) * 11)) + (((int)threadIdx.x) * 2)) % 81) / 27) * 28)) + ((((((int)threadIdx.z) * 21) + (((int)threadIdx.y) * 11)) + (((int)threadIdx.x) * 2)) % 27)))];
        }
      }
    }
    if ((((((int)threadIdx.z) * 21) + (((int)threadIdx.y) * 11)) + (((int)threadIdx.x) * 2)) < 1295) {
      if (((((int)threadIdx.y) * 11) + (((int)threadIdx.x) * 2)) < 20) {
        if (((int)threadIdx.x) < 5) {
          pad_temp_shared[(((((((int)threadIdx.z) * 21) + (((int)threadIdx.y) * 11)) + (((int)threadIdx.x) * 2)) + 1))] = placeholder[((((((rc_outer * 12544) + ((((((((int)threadIdx.z) * 21) + (((int)threadIdx.y) * 11)) + (((int)threadIdx.x) * 2)) + 1) / 81) * 784)) + (((int)blockIdx.y) * 112)) + (((((((((int)threadIdx.z) * 21) + (((int)threadIdx.y) * 11)) + (((int)threadIdx.x) * 2)) + 1) % 81) / 27) * 28)) + (((((((int)threadIdx.z) * 21) + (((int)threadIdx.y) * 11)) + (((int)threadIdx.x) * 2)) + 1) % 27)))];
        }
      }
    }
    if (((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) >> 4) + ((int)threadIdx.z)) < 64) {
      if ((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) < 1024) {
        if (((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) < 16) {
          if (((int)threadIdx.x) < 4) {
            placeholder_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)))] = placeholder1[((((((((int)blockIdx.z) * 32768) + (((int)threadIdx.z) * 512)) + (rc_outer * 16)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)))];
          }
        }
      }
    }
    if ((((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 1) >> 4) + ((int)threadIdx.z)) < 64) {
      if ((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) < 1023) {
        if (((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) < 15) {
          if (((int)threadIdx.x) < 4) {
            placeholder_shared[(((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1))] = placeholder1[(((((((((int)blockIdx.z) * 32768) + (((int)threadIdx.z) * 512)) + (rc_outer * 16)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1))];
          }
        }
      }
    }
    __syncthreads();
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 54) + (((int)threadIdx.x) * 4)))], placeholder_shared[((((int)threadIdx.z) * 16))], compute_local[(0)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 54) + (((int)threadIdx.x) * 4)) + 2))], placeholder_shared[((((int)threadIdx.z) * 16))], compute_local[(1)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 54) + (((int)threadIdx.x) * 4)) + 81))], placeholder_shared[(((((int)threadIdx.z) * 16) + 1))], compute_local[(0)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 54) + (((int)threadIdx.x) * 4)) + 83))], placeholder_shared[(((((int)threadIdx.z) * 16) + 1))], compute_local[(1)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 54) + (((int)threadIdx.x) * 4)) + 162))], placeholder_shared[(((((int)threadIdx.z) * 16) + 2))], compute_local[(0)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 54) + (((int)threadIdx.x) * 4)) + 164))], placeholder_shared[(((((int)threadIdx.z) * 16) + 2))], compute_local[(1)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 54) + (((int)threadIdx.x) * 4)) + 243))], placeholder_shared[(((((int)threadIdx.z) * 16) + 3))], compute_local[(0)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 54) + (((int)threadIdx.x) * 4)) + 245))], placeholder_shared[(((((int)threadIdx.z) * 16) + 3))], compute_local[(1)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 54) + (((int)threadIdx.x) * 4)) + 324))], placeholder_shared[(((((int)threadIdx.z) * 16) + 4))], compute_local[(0)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 54) + (((int)threadIdx.x) * 4)) + 326))], placeholder_shared[(((((int)threadIdx.z) * 16) + 4))], compute_local[(1)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 54) + (((int)threadIdx.x) * 4)) + 405))], placeholder_shared[(((((int)threadIdx.z) * 16) + 5))], compute_local[(0)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 54) + (((int)threadIdx.x) * 4)) + 407))], placeholder_shared[(((((int)threadIdx.z) * 16) + 5))], compute_local[(1)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 54) + (((int)threadIdx.x) * 4)) + 486))], placeholder_shared[(((((int)threadIdx.z) * 16) + 6))], compute_local[(0)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 54) + (((int)threadIdx.x) * 4)) + 488))], placeholder_shared[(((((int)threadIdx.z) * 16) + 6))], compute_local[(1)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 54) + (((int)threadIdx.x) * 4)) + 567))], placeholder_shared[(((((int)threadIdx.z) * 16) + 7))], compute_local[(0)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 54) + (((int)threadIdx.x) * 4)) + 569))], placeholder_shared[(((((int)threadIdx.z) * 16) + 7))], compute_local[(1)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 54) + (((int)threadIdx.x) * 4)) + 648))], placeholder_shared[(((((int)threadIdx.z) * 16) + 8))], compute_local[(0)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 54) + (((int)threadIdx.x) * 4)) + 650))], placeholder_shared[(((((int)threadIdx.z) * 16) + 8))], compute_local[(1)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 54) + (((int)threadIdx.x) * 4)) + 729))], placeholder_shared[(((((int)threadIdx.z) * 16) + 9))], compute_local[(0)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 54) + (((int)threadIdx.x) * 4)) + 731))], placeholder_shared[(((((int)threadIdx.z) * 16) + 9))], compute_local[(1)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 54) + (((int)threadIdx.x) * 4)) + 810))], placeholder_shared[(((((int)threadIdx.z) * 16) + 10))], compute_local[(0)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 54) + (((int)threadIdx.x) * 4)) + 812))], placeholder_shared[(((((int)threadIdx.z) * 16) + 10))], compute_local[(1)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 54) + (((int)threadIdx.x) * 4)) + 891))], placeholder_shared[(((((int)threadIdx.z) * 16) + 11))], compute_local[(0)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 54) + (((int)threadIdx.x) * 4)) + 893))], placeholder_shared[(((((int)threadIdx.z) * 16) + 11))], compute_local[(1)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 54) + (((int)threadIdx.x) * 4)) + 972))], placeholder_shared[(((((int)threadIdx.z) * 16) + 12))], compute_local[(0)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 54) + (((int)threadIdx.x) * 4)) + 974))], placeholder_shared[(((((int)threadIdx.z) * 16) + 12))], compute_local[(1)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 54) + (((int)threadIdx.x) * 4)) + 1053))], placeholder_shared[(((((int)threadIdx.z) * 16) + 13))], compute_local[(0)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 54) + (((int)threadIdx.x) * 4)) + 1055))], placeholder_shared[(((((int)threadIdx.z) * 16) + 13))], compute_local[(1)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 54) + (((int)threadIdx.x) * 4)) + 1134))], placeholder_shared[(((((int)threadIdx.z) * 16) + 14))], compute_local[(0)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 54) + (((int)threadIdx.x) * 4)) + 1136))], placeholder_shared[(((((int)threadIdx.z) * 16) + 14))], compute_local[(1)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 54) + (((int)threadIdx.x) * 4)) + 1215))], placeholder_shared[(((((int)threadIdx.z) * 16) + 15))], compute_local[(0)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 54) + (((int)threadIdx.x) * 4)) + 1217))], placeholder_shared[(((((int)threadIdx.z) * 16) + 15))], compute_local[(1)]);
  }
  compute[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 28)) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)))] = compute_local[(0)];
  compute[(((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 28)) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)) + 1))] = compute_local[(1)];
}

extern "C" __global__ void fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_3_kernel0(float* __restrict__ placeholder, float* __restrict__ data_pack) {
  float d[36];
  float data_pack_local[36];
  for (int eps = 0; eps < 6; ++eps) {
    for (int nu = 0; nu < 6; ++nu) {
      d[(((eps * 6) + nu))] = (((((1 <= ((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 196) / 14) * 4) + eps)) && (((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 196) / 14) * 4) + eps) < 57)) && (1 <= (((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 14) * 4) + nu))) && ((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 14) * 4) + nu) < 57)) ? placeholder[(((((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 14) * 224) + (eps * 56)) + ((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 14) * 4)) + nu) - 57))] : 0.000000e+00f);
    }
  }
  data_pack_local[(0)] = 0.000000e+00f;
  data_pack_local[(0)] = (data_pack_local[(0)] + d[(0)]);
  data_pack_local[(0)] = __ocml_fma_f32(d[(1)], -1.500000e+00f, data_pack_local[(0)]);
  data_pack_local[(0)] = __ocml_fma_f32(d[(2)], -2.000000e+00f, data_pack_local[(0)]);
  data_pack_local[(0)] = __ocml_fma_f32(d[(3)], 1.500000e+00f, data_pack_local[(0)]);
  data_pack_local[(0)] = (data_pack_local[(0)] + d[(4)]);
  data_pack_local[(0)] = __ocml_fma_f32(d[(6)], -1.500000e+00f, data_pack_local[(0)]);
  data_pack_local[(0)] = __ocml_fma_f32((d[(7)] * -1.500000e+00f), -1.500000e+00f, data_pack_local[(0)]);
  data_pack_local[(0)] = __ocml_fma_f32((d[(8)] * -1.500000e+00f), -2.000000e+00f, data_pack_local[(0)]);
  data_pack_local[(0)] = __ocml_fma_f32((d[(9)] * -1.500000e+00f), 1.500000e+00f, data_pack_local[(0)]);
  data_pack_local[(0)] = __ocml_fma_f32(d[(10)], -1.500000e+00f, data_pack_local[(0)]);
  data_pack_local[(0)] = __ocml_fma_f32(d[(12)], -2.000000e+00f, data_pack_local[(0)]);
  data_pack_local[(0)] = __ocml_fma_f32((d[(13)] * -2.000000e+00f), -1.500000e+00f, data_pack_local[(0)]);
  data_pack_local[(0)] = __ocml_fma_f32((d[(14)] * -2.000000e+00f), -2.000000e+00f, data_pack_local[(0)]);
  data_pack_local[(0)] = __ocml_fma_f32((d[(15)] * -2.000000e+00f), 1.500000e+00f, data_pack_local[(0)]);
  data_pack_local[(0)] = __ocml_fma_f32(d[(16)], -2.000000e+00f, data_pack_local[(0)]);
  data_pack_local[(0)] = __ocml_fma_f32(d[(18)], 1.500000e+00f, data_pack_local[(0)]);
  data_pack_local[(0)] = __ocml_fma_f32((d[(19)] * 1.500000e+00f), -1.500000e+00f, data_pack_local[(0)]);
  data_pack_local[(0)] = __ocml_fma_f32((d[(20)] * 1.500000e+00f), -2.000000e+00f, data_pack_local[(0)]);
  data_pack_local[(0)] = __ocml_fma_f32((d[(21)] * 1.500000e+00f), 1.500000e+00f, data_pack_local[(0)]);
  data_pack_local[(0)] = __ocml_fma_f32(d[(22)], 1.500000e+00f, data_pack_local[(0)]);
  data_pack_local[(0)] = (data_pack_local[(0)] + d[(24)]);
  data_pack_local[(0)] = __ocml_fma_f32(d[(25)], -1.500000e+00f, data_pack_local[(0)]);
  data_pack_local[(0)] = __ocml_fma_f32(d[(26)], -2.000000e+00f, data_pack_local[(0)]);
  data_pack_local[(0)] = __ocml_fma_f32(d[(27)], 1.500000e+00f, data_pack_local[(0)]);
  data_pack_local[(0)] = (data_pack_local[(0)] + d[(28)]);
  data_pack_local[(1)] = 0.000000e+00f;
  data_pack_local[(1)] = (data_pack_local[(1)] + d[(1)]);
  data_pack_local[(1)] = __ocml_fma_f32(d[(2)], -2.500000e+00f, data_pack_local[(1)]);
  data_pack_local[(1)] = __ocml_fma_f32(d[(3)], 5.000000e-01f, data_pack_local[(1)]);
  data_pack_local[(1)] = (data_pack_local[(1)] + d[(4)]);
  data_pack_local[(1)] = __ocml_fma_f32(d[(7)], -1.500000e+00f, data_pack_local[(1)]);
  data_pack_local[(1)] = __ocml_fma_f32((d[(8)] * -1.500000e+00f), -2.500000e+00f, data_pack_local[(1)]);
  data_pack_local[(1)] = __ocml_fma_f32((d[(9)] * -1.500000e+00f), 5.000000e-01f, data_pack_local[(1)]);
  data_pack_local[(1)] = __ocml_fma_f32(d[(10)], -1.500000e+00f, data_pack_local[(1)]);
  data_pack_local[(1)] = __ocml_fma_f32(d[(13)], -2.000000e+00f, data_pack_local[(1)]);
  data_pack_local[(1)] = __ocml_fma_f32((d[(14)] * -2.000000e+00f), -2.500000e+00f, data_pack_local[(1)]);
  data_pack_local[(1)] = __ocml_fma_f32((d[(15)] * -2.000000e+00f), 5.000000e-01f, data_pack_local[(1)]);
  data_pack_local[(1)] = __ocml_fma_f32(d[(16)], -2.000000e+00f, data_pack_local[(1)]);
  data_pack_local[(1)] = __ocml_fma_f32(d[(19)], 1.500000e+00f, data_pack_local[(1)]);
  data_pack_local[(1)] = __ocml_fma_f32((d[(20)] * 1.500000e+00f), -2.500000e+00f, data_pack_local[(1)]);
  data_pack_local[(1)] = __ocml_fma_f32((d[(21)] * 1.500000e+00f), 5.000000e-01f, data_pack_local[(1)]);
  data_pack_local[(1)] = __ocml_fma_f32(d[(22)], 1.500000e+00f, data_pack_local[(1)]);
  data_pack_local[(1)] = (data_pack_local[(1)] + d[(25)]);
  data_pack_local[(1)] = __ocml_fma_f32(d[(26)], -2.500000e+00f, data_pack_local[(1)]);
  data_pack_local[(1)] = __ocml_fma_f32(d[(27)], 5.000000e-01f, data_pack_local[(1)]);
  data_pack_local[(1)] = (data_pack_local[(1)] + d[(28)]);
  data_pack_local[(2)] = 0.000000e+00f;
  data_pack_local[(2)] = __ocml_fma_f32(d[(1)], -1.000000e+00f, data_pack_local[(2)]);
  data_pack_local[(2)] = __ocml_fma_f32(d[(2)], 5.000000e-01f, data_pack_local[(2)]);
  data_pack_local[(2)] = __ocml_fma_f32(d[(3)], 2.500000e+00f, data_pack_local[(2)]);
  data_pack_local[(2)] = (data_pack_local[(2)] + d[(4)]);
  data_pack_local[(2)] = __ocml_fma_f32((d[(7)] * -1.500000e+00f), -1.000000e+00f, data_pack_local[(2)]);
  data_pack_local[(2)] = __ocml_fma_f32((d[(8)] * -1.500000e+00f), 5.000000e-01f, data_pack_local[(2)]);
  data_pack_local[(2)] = __ocml_fma_f32((d[(9)] * -1.500000e+00f), 2.500000e+00f, data_pack_local[(2)]);
  data_pack_local[(2)] = __ocml_fma_f32(d[(10)], -1.500000e+00f, data_pack_local[(2)]);
  data_pack_local[(2)] = __ocml_fma_f32((d[(13)] * -2.000000e+00f), -1.000000e+00f, data_pack_local[(2)]);
  data_pack_local[(2)] = __ocml_fma_f32((d[(14)] * -2.000000e+00f), 5.000000e-01f, data_pack_local[(2)]);
  data_pack_local[(2)] = __ocml_fma_f32((d[(15)] * -2.000000e+00f), 2.500000e+00f, data_pack_local[(2)]);
  data_pack_local[(2)] = __ocml_fma_f32(d[(16)], -2.000000e+00f, data_pack_local[(2)]);
  data_pack_local[(2)] = __ocml_fma_f32((d[(19)] * 1.500000e+00f), -1.000000e+00f, data_pack_local[(2)]);
  data_pack_local[(2)] = __ocml_fma_f32((d[(20)] * 1.500000e+00f), 5.000000e-01f, data_pack_local[(2)]);
  data_pack_local[(2)] = __ocml_fma_f32((d[(21)] * 1.500000e+00f), 2.500000e+00f, data_pack_local[(2)]);
  data_pack_local[(2)] = __ocml_fma_f32(d[(22)], 1.500000e+00f, data_pack_local[(2)]);
  data_pack_local[(2)] = __ocml_fma_f32(d[(25)], -1.000000e+00f, data_pack_local[(2)]);
  data_pack_local[(2)] = __ocml_fma_f32(d[(26)], 5.000000e-01f, data_pack_local[(2)]);
  data_pack_local[(2)] = __ocml_fma_f32(d[(27)], 2.500000e+00f, data_pack_local[(2)]);
  data_pack_local[(2)] = (data_pack_local[(2)] + d[(28)]);
  data_pack_local[(3)] = 0.000000e+00f;
  data_pack_local[(3)] = __ocml_fma_f32(d[(1)], -2.000000e+00f, data_pack_local[(3)]);
  data_pack_local[(3)] = __ocml_fma_f32(d[(2)], -1.000000e+00f, data_pack_local[(3)]);
  data_pack_local[(3)] = __ocml_fma_f32(d[(3)], 2.000000e+00f, data_pack_local[(3)]);
  data_pack_local[(3)] = (data_pack_local[(3)] + d[(4)]);
  data_pack_local[(3)] = __ocml_fma_f32((d[(7)] * -1.500000e+00f), -2.000000e+00f, data_pack_local[(3)]);
  data_pack_local[(3)] = __ocml_fma_f32((d[(8)] * -1.500000e+00f), -1.000000e+00f, data_pack_local[(3)]);
  data_pack_local[(3)] = __ocml_fma_f32((d[(9)] * -1.500000e+00f), 2.000000e+00f, data_pack_local[(3)]);
  data_pack_local[(3)] = __ocml_fma_f32(d[(10)], -1.500000e+00f, data_pack_local[(3)]);
  data_pack_local[(3)] = __ocml_fma_f32((d[(13)] * -2.000000e+00f), -2.000000e+00f, data_pack_local[(3)]);
  data_pack_local[(3)] = __ocml_fma_f32((d[(14)] * -2.000000e+00f), -1.000000e+00f, data_pack_local[(3)]);
  data_pack_local[(3)] = __ocml_fma_f32((d[(15)] * -2.000000e+00f), 2.000000e+00f, data_pack_local[(3)]);
  data_pack_local[(3)] = __ocml_fma_f32(d[(16)], -2.000000e+00f, data_pack_local[(3)]);
  data_pack_local[(3)] = __ocml_fma_f32((d[(19)] * 1.500000e+00f), -2.000000e+00f, data_pack_local[(3)]);
  data_pack_local[(3)] = __ocml_fma_f32((d[(20)] * 1.500000e+00f), -1.000000e+00f, data_pack_local[(3)]);
  data_pack_local[(3)] = __ocml_fma_f32((d[(21)] * 1.500000e+00f), 2.000000e+00f, data_pack_local[(3)]);
  data_pack_local[(3)] = __ocml_fma_f32(d[(22)], 1.500000e+00f, data_pack_local[(3)]);
  data_pack_local[(3)] = __ocml_fma_f32(d[(25)], -2.000000e+00f, data_pack_local[(3)]);
  data_pack_local[(3)] = __ocml_fma_f32(d[(26)], -1.000000e+00f, data_pack_local[(3)]);
  data_pack_local[(3)] = __ocml_fma_f32(d[(27)], 2.000000e+00f, data_pack_local[(3)]);
  data_pack_local[(3)] = (data_pack_local[(3)] + d[(28)]);
  data_pack_local[(4)] = 0.000000e+00f;
  data_pack_local[(4)] = __ocml_fma_f32(d[(1)], 5.000000e-01f, data_pack_local[(4)]);
  data_pack_local[(4)] = __ocml_fma_f32(d[(2)], -1.000000e+00f, data_pack_local[(4)]);
  data_pack_local[(4)] = __ocml_fma_f32(d[(3)], -5.000000e-01f, data_pack_local[(4)]);
  data_pack_local[(4)] = (data_pack_local[(4)] + d[(4)]);
  data_pack_local[(4)] = __ocml_fma_f32((d[(7)] * -1.500000e+00f), 5.000000e-01f, data_pack_local[(4)]);
  data_pack_local[(4)] = __ocml_fma_f32((d[(8)] * -1.500000e+00f), -1.000000e+00f, data_pack_local[(4)]);
  data_pack_local[(4)] = __ocml_fma_f32((d[(9)] * -1.500000e+00f), -5.000000e-01f, data_pack_local[(4)]);
  data_pack_local[(4)] = __ocml_fma_f32(d[(10)], -1.500000e+00f, data_pack_local[(4)]);
  data_pack_local[(4)] = __ocml_fma_f32((d[(13)] * -2.000000e+00f), 5.000000e-01f, data_pack_local[(4)]);
  data_pack_local[(4)] = __ocml_fma_f32((d[(14)] * -2.000000e+00f), -1.000000e+00f, data_pack_local[(4)]);
  data_pack_local[(4)] = __ocml_fma_f32((d[(15)] * -2.000000e+00f), -5.000000e-01f, data_pack_local[(4)]);
  data_pack_local[(4)] = __ocml_fma_f32(d[(16)], -2.000000e+00f, data_pack_local[(4)]);
  data_pack_local[(4)] = __ocml_fma_f32((d[(19)] * 1.500000e+00f), 5.000000e-01f, data_pack_local[(4)]);
  data_pack_local[(4)] = __ocml_fma_f32((d[(20)] * 1.500000e+00f), -1.000000e+00f, data_pack_local[(4)]);
  data_pack_local[(4)] = __ocml_fma_f32((d[(21)] * 1.500000e+00f), -5.000000e-01f, data_pack_local[(4)]);
  data_pack_local[(4)] = __ocml_fma_f32(d[(22)], 1.500000e+00f, data_pack_local[(4)]);
  data_pack_local[(4)] = __ocml_fma_f32(d[(25)], 5.000000e-01f, data_pack_local[(4)]);
  data_pack_local[(4)] = __ocml_fma_f32(d[(26)], -1.000000e+00f, data_pack_local[(4)]);
  data_pack_local[(4)] = __ocml_fma_f32(d[(27)], -5.000000e-01f, data_pack_local[(4)]);
  data_pack_local[(4)] = (data_pack_local[(4)] + d[(28)]);
  data_pack_local[(5)] = 0.000000e+00f;
  data_pack_local[(5)] = (data_pack_local[(5)] + d[(1)]);
  data_pack_local[(5)] = __ocml_fma_f32(d[(2)], -1.500000e+00f, data_pack_local[(5)]);
  data_pack_local[(5)] = __ocml_fma_f32(d[(3)], -2.000000e+00f, data_pack_local[(5)]);
  data_pack_local[(5)] = __ocml_fma_f32(d[(4)], 1.500000e+00f, data_pack_local[(5)]);
  data_pack_local[(5)] = (data_pack_local[(5)] + d[(5)]);
  data_pack_local[(5)] = __ocml_fma_f32(d[(7)], -1.500000e+00f, data_pack_local[(5)]);
  data_pack_local[(5)] = __ocml_fma_f32((d[(8)] * -1.500000e+00f), -1.500000e+00f, data_pack_local[(5)]);
  data_pack_local[(5)] = __ocml_fma_f32((d[(9)] * -1.500000e+00f), -2.000000e+00f, data_pack_local[(5)]);
  data_pack_local[(5)] = __ocml_fma_f32((d[(10)] * -1.500000e+00f), 1.500000e+00f, data_pack_local[(5)]);
  data_pack_local[(5)] = __ocml_fma_f32(d[(11)], -1.500000e+00f, data_pack_local[(5)]);
  data_pack_local[(5)] = __ocml_fma_f32(d[(13)], -2.000000e+00f, data_pack_local[(5)]);
  data_pack_local[(5)] = __ocml_fma_f32((d[(14)] * -2.000000e+00f), -1.500000e+00f, data_pack_local[(5)]);
  data_pack_local[(5)] = __ocml_fma_f32((d[(15)] * -2.000000e+00f), -2.000000e+00f, data_pack_local[(5)]);
  data_pack_local[(5)] = __ocml_fma_f32((d[(16)] * -2.000000e+00f), 1.500000e+00f, data_pack_local[(5)]);
  data_pack_local[(5)] = __ocml_fma_f32(d[(17)], -2.000000e+00f, data_pack_local[(5)]);
  data_pack_local[(5)] = __ocml_fma_f32(d[(19)], 1.500000e+00f, data_pack_local[(5)]);
  data_pack_local[(5)] = __ocml_fma_f32((d[(20)] * 1.500000e+00f), -1.500000e+00f, data_pack_local[(5)]);
  data_pack_local[(5)] = __ocml_fma_f32((d[(21)] * 1.500000e+00f), -2.000000e+00f, data_pack_local[(5)]);
  data_pack_local[(5)] = __ocml_fma_f32((d[(22)] * 1.500000e+00f), 1.500000e+00f, data_pack_local[(5)]);
  data_pack_local[(5)] = __ocml_fma_f32(d[(23)], 1.500000e+00f, data_pack_local[(5)]);
  data_pack_local[(5)] = (data_pack_local[(5)] + d[(25)]);
  data_pack_local[(5)] = __ocml_fma_f32(d[(26)], -1.500000e+00f, data_pack_local[(5)]);
  data_pack_local[(5)] = __ocml_fma_f32(d[(27)], -2.000000e+00f, data_pack_local[(5)]);
  data_pack_local[(5)] = __ocml_fma_f32(d[(28)], 1.500000e+00f, data_pack_local[(5)]);
  data_pack_local[(5)] = (data_pack_local[(5)] + d[(29)]);
  data_pack_local[(6)] = 0.000000e+00f;
  data_pack_local[(6)] = (data_pack_local[(6)] + d[(6)]);
  data_pack_local[(6)] = __ocml_fma_f32(d[(7)], -1.500000e+00f, data_pack_local[(6)]);
  data_pack_local[(6)] = __ocml_fma_f32(d[(8)], -2.000000e+00f, data_pack_local[(6)]);
  data_pack_local[(6)] = __ocml_fma_f32(d[(9)], 1.500000e+00f, data_pack_local[(6)]);
  data_pack_local[(6)] = (data_pack_local[(6)] + d[(10)]);
  data_pack_local[(6)] = __ocml_fma_f32(d[(12)], -2.500000e+00f, data_pack_local[(6)]);
  data_pack_local[(6)] = __ocml_fma_f32((d[(13)] * -2.500000e+00f), -1.500000e+00f, data_pack_local[(6)]);
  data_pack_local[(6)] = __ocml_fma_f32((d[(14)] * -2.500000e+00f), -2.000000e+00f, data_pack_local[(6)]);
  data_pack_local[(6)] = __ocml_fma_f32((d[(15)] * -2.500000e+00f), 1.500000e+00f, data_pack_local[(6)]);
  data_pack_local[(6)] = __ocml_fma_f32(d[(16)], -2.500000e+00f, data_pack_local[(6)]);
  data_pack_local[(6)] = __ocml_fma_f32(d[(18)], 5.000000e-01f, data_pack_local[(6)]);
  data_pack_local[(6)] = __ocml_fma_f32((d[(19)] * 5.000000e-01f), -1.500000e+00f, data_pack_local[(6)]);
  data_pack_local[(6)] = __ocml_fma_f32((d[(20)] * 5.000000e-01f), -2.000000e+00f, data_pack_local[(6)]);
  data_pack_local[(6)] = __ocml_fma_f32((d[(21)] * 5.000000e-01f), 1.500000e+00f, data_pack_local[(6)]);
  data_pack_local[(6)] = __ocml_fma_f32(d[(22)], 5.000000e-01f, data_pack_local[(6)]);
  data_pack_local[(6)] = (data_pack_local[(6)] + d[(24)]);
  data_pack_local[(6)] = __ocml_fma_f32(d[(25)], -1.500000e+00f, data_pack_local[(6)]);
  data_pack_local[(6)] = __ocml_fma_f32(d[(26)], -2.000000e+00f, data_pack_local[(6)]);
  data_pack_local[(6)] = __ocml_fma_f32(d[(27)], 1.500000e+00f, data_pack_local[(6)]);
  data_pack_local[(6)] = (data_pack_local[(6)] + d[(28)]);
  data_pack_local[(7)] = 0.000000e+00f;
  data_pack_local[(7)] = (data_pack_local[(7)] + d[(7)]);
  data_pack_local[(7)] = __ocml_fma_f32(d[(8)], -2.500000e+00f, data_pack_local[(7)]);
  data_pack_local[(7)] = __ocml_fma_f32(d[(9)], 5.000000e-01f, data_pack_local[(7)]);
  data_pack_local[(7)] = (data_pack_local[(7)] + d[(10)]);
  data_pack_local[(7)] = __ocml_fma_f32(d[(13)], -2.500000e+00f, data_pack_local[(7)]);
  data_pack_local[(7)] = __ocml_fma_f32((d[(14)] * -2.500000e+00f), -2.500000e+00f, data_pack_local[(7)]);
  data_pack_local[(7)] = __ocml_fma_f32((d[(15)] * -2.500000e+00f), 5.000000e-01f, data_pack_local[(7)]);
  data_pack_local[(7)] = __ocml_fma_f32(d[(16)], -2.500000e+00f, data_pack_local[(7)]);
  data_pack_local[(7)] = __ocml_fma_f32(d[(19)], 5.000000e-01f, data_pack_local[(7)]);
  data_pack_local[(7)] = __ocml_fma_f32((d[(20)] * 5.000000e-01f), -2.500000e+00f, data_pack_local[(7)]);
  data_pack_local[(7)] = __ocml_fma_f32((d[(21)] * 5.000000e-01f), 5.000000e-01f, data_pack_local[(7)]);
  data_pack_local[(7)] = __ocml_fma_f32(d[(22)], 5.000000e-01f, data_pack_local[(7)]);
  data_pack_local[(7)] = (data_pack_local[(7)] + d[(25)]);
  data_pack_local[(7)] = __ocml_fma_f32(d[(26)], -2.500000e+00f, data_pack_local[(7)]);
  data_pack_local[(7)] = __ocml_fma_f32(d[(27)], 5.000000e-01f, data_pack_local[(7)]);
  data_pack_local[(7)] = (data_pack_local[(7)] + d[(28)]);
  data_pack_local[(8)] = 0.000000e+00f;
  data_pack_local[(8)] = __ocml_fma_f32(d[(7)], -1.000000e+00f, data_pack_local[(8)]);
  data_pack_local[(8)] = __ocml_fma_f32(d[(8)], 5.000000e-01f, data_pack_local[(8)]);
  data_pack_local[(8)] = __ocml_fma_f32(d[(9)], 2.500000e+00f, data_pack_local[(8)]);
  data_pack_local[(8)] = (data_pack_local[(8)] + d[(10)]);
  data_pack_local[(8)] = __ocml_fma_f32((d[(13)] * -2.500000e+00f), -1.000000e+00f, data_pack_local[(8)]);
  data_pack_local[(8)] = __ocml_fma_f32((d[(14)] * -2.500000e+00f), 5.000000e-01f, data_pack_local[(8)]);
  data_pack_local[(8)] = __ocml_fma_f32((d[(15)] * -2.500000e+00f), 2.500000e+00f, data_pack_local[(8)]);
  data_pack_local[(8)] = __ocml_fma_f32(d[(16)], -2.500000e+00f, data_pack_local[(8)]);
  data_pack_local[(8)] = __ocml_fma_f32((d[(19)] * 5.000000e-01f), -1.000000e+00f, data_pack_local[(8)]);
  data_pack_local[(8)] = __ocml_fma_f32((d[(20)] * 5.000000e-01f), 5.000000e-01f, data_pack_local[(8)]);
  data_pack_local[(8)] = __ocml_fma_f32((d[(21)] * 5.000000e-01f), 2.500000e+00f, data_pack_local[(8)]);
  data_pack_local[(8)] = __ocml_fma_f32(d[(22)], 5.000000e-01f, data_pack_local[(8)]);
  data_pack_local[(8)] = __ocml_fma_f32(d[(25)], -1.000000e+00f, data_pack_local[(8)]);
  data_pack_local[(8)] = __ocml_fma_f32(d[(26)], 5.000000e-01f, data_pack_local[(8)]);
  data_pack_local[(8)] = __ocml_fma_f32(d[(27)], 2.500000e+00f, data_pack_local[(8)]);
  data_pack_local[(8)] = (data_pack_local[(8)] + d[(28)]);
  data_pack_local[(9)] = 0.000000e+00f;
  data_pack_local[(9)] = __ocml_fma_f32(d[(7)], -2.000000e+00f, data_pack_local[(9)]);
  data_pack_local[(9)] = __ocml_fma_f32(d[(8)], -1.000000e+00f, data_pack_local[(9)]);
  data_pack_local[(9)] = __ocml_fma_f32(d[(9)], 2.000000e+00f, data_pack_local[(9)]);
  data_pack_local[(9)] = (data_pack_local[(9)] + d[(10)]);
  data_pack_local[(9)] = __ocml_fma_f32((d[(13)] * -2.500000e+00f), -2.000000e+00f, data_pack_local[(9)]);
  data_pack_local[(9)] = __ocml_fma_f32((d[(14)] * -2.500000e+00f), -1.000000e+00f, data_pack_local[(9)]);
  data_pack_local[(9)] = __ocml_fma_f32((d[(15)] * -2.500000e+00f), 2.000000e+00f, data_pack_local[(9)]);
  data_pack_local[(9)] = __ocml_fma_f32(d[(16)], -2.500000e+00f, data_pack_local[(9)]);
  data_pack_local[(9)] = __ocml_fma_f32((d[(19)] * 5.000000e-01f), -2.000000e+00f, data_pack_local[(9)]);
  data_pack_local[(9)] = __ocml_fma_f32((d[(20)] * 5.000000e-01f), -1.000000e+00f, data_pack_local[(9)]);
  data_pack_local[(9)] = __ocml_fma_f32((d[(21)] * 5.000000e-01f), 2.000000e+00f, data_pack_local[(9)]);
  data_pack_local[(9)] = __ocml_fma_f32(d[(22)], 5.000000e-01f, data_pack_local[(9)]);
  data_pack_local[(9)] = __ocml_fma_f32(d[(25)], -2.000000e+00f, data_pack_local[(9)]);
  data_pack_local[(9)] = __ocml_fma_f32(d[(26)], -1.000000e+00f, data_pack_local[(9)]);
  data_pack_local[(9)] = __ocml_fma_f32(d[(27)], 2.000000e+00f, data_pack_local[(9)]);
  data_pack_local[(9)] = (data_pack_local[(9)] + d[(28)]);
  data_pack_local[(10)] = 0.000000e+00f;
  data_pack_local[(10)] = __ocml_fma_f32(d[(7)], 5.000000e-01f, data_pack_local[(10)]);
  data_pack_local[(10)] = __ocml_fma_f32(d[(8)], -1.000000e+00f, data_pack_local[(10)]);
  data_pack_local[(10)] = __ocml_fma_f32(d[(9)], -5.000000e-01f, data_pack_local[(10)]);
  data_pack_local[(10)] = (data_pack_local[(10)] + d[(10)]);
  data_pack_local[(10)] = __ocml_fma_f32((d[(13)] * -2.500000e+00f), 5.000000e-01f, data_pack_local[(10)]);
  data_pack_local[(10)] = __ocml_fma_f32((d[(14)] * -2.500000e+00f), -1.000000e+00f, data_pack_local[(10)]);
  data_pack_local[(10)] = __ocml_fma_f32((d[(15)] * -2.500000e+00f), -5.000000e-01f, data_pack_local[(10)]);
  data_pack_local[(10)] = __ocml_fma_f32(d[(16)], -2.500000e+00f, data_pack_local[(10)]);
  data_pack_local[(10)] = __ocml_fma_f32((d[(19)] * 5.000000e-01f), 5.000000e-01f, data_pack_local[(10)]);
  data_pack_local[(10)] = __ocml_fma_f32((d[(20)] * 5.000000e-01f), -1.000000e+00f, data_pack_local[(10)]);
  data_pack_local[(10)] = __ocml_fma_f32((d[(21)] * 5.000000e-01f), -5.000000e-01f, data_pack_local[(10)]);
  data_pack_local[(10)] = __ocml_fma_f32(d[(22)], 5.000000e-01f, data_pack_local[(10)]);
  data_pack_local[(10)] = __ocml_fma_f32(d[(25)], 5.000000e-01f, data_pack_local[(10)]);
  data_pack_local[(10)] = __ocml_fma_f32(d[(26)], -1.000000e+00f, data_pack_local[(10)]);
  data_pack_local[(10)] = __ocml_fma_f32(d[(27)], -5.000000e-01f, data_pack_local[(10)]);
  data_pack_local[(10)] = (data_pack_local[(10)] + d[(28)]);
  data_pack_local[(11)] = 0.000000e+00f;
  data_pack_local[(11)] = (data_pack_local[(11)] + d[(7)]);
  data_pack_local[(11)] = __ocml_fma_f32(d[(8)], -1.500000e+00f, data_pack_local[(11)]);
  data_pack_local[(11)] = __ocml_fma_f32(d[(9)], -2.000000e+00f, data_pack_local[(11)]);
  data_pack_local[(11)] = __ocml_fma_f32(d[(10)], 1.500000e+00f, data_pack_local[(11)]);
  data_pack_local[(11)] = (data_pack_local[(11)] + d[(11)]);
  data_pack_local[(11)] = __ocml_fma_f32(d[(13)], -2.500000e+00f, data_pack_local[(11)]);
  data_pack_local[(11)] = __ocml_fma_f32((d[(14)] * -2.500000e+00f), -1.500000e+00f, data_pack_local[(11)]);
  data_pack_local[(11)] = __ocml_fma_f32((d[(15)] * -2.500000e+00f), -2.000000e+00f, data_pack_local[(11)]);
  data_pack_local[(11)] = __ocml_fma_f32((d[(16)] * -2.500000e+00f), 1.500000e+00f, data_pack_local[(11)]);
  data_pack_local[(11)] = __ocml_fma_f32(d[(17)], -2.500000e+00f, data_pack_local[(11)]);
  data_pack_local[(11)] = __ocml_fma_f32(d[(19)], 5.000000e-01f, data_pack_local[(11)]);
  data_pack_local[(11)] = __ocml_fma_f32((d[(20)] * 5.000000e-01f), -1.500000e+00f, data_pack_local[(11)]);
  data_pack_local[(11)] = __ocml_fma_f32((d[(21)] * 5.000000e-01f), -2.000000e+00f, data_pack_local[(11)]);
  data_pack_local[(11)] = __ocml_fma_f32((d[(22)] * 5.000000e-01f), 1.500000e+00f, data_pack_local[(11)]);
  data_pack_local[(11)] = __ocml_fma_f32(d[(23)], 5.000000e-01f, data_pack_local[(11)]);
  data_pack_local[(11)] = (data_pack_local[(11)] + d[(25)]);
  data_pack_local[(11)] = __ocml_fma_f32(d[(26)], -1.500000e+00f, data_pack_local[(11)]);
  data_pack_local[(11)] = __ocml_fma_f32(d[(27)], -2.000000e+00f, data_pack_local[(11)]);
  data_pack_local[(11)] = __ocml_fma_f32(d[(28)], 1.500000e+00f, data_pack_local[(11)]);
  data_pack_local[(11)] = (data_pack_local[(11)] + d[(29)]);
  data_pack_local[(12)] = 0.000000e+00f;
  data_pack_local[(12)] = __ocml_fma_f32(d[(6)], -1.000000e+00f, data_pack_local[(12)]);
  data_pack_local[(12)] = __ocml_fma_f32((d[(7)] * -1.000000e+00f), -1.500000e+00f, data_pack_local[(12)]);
  data_pack_local[(12)] = __ocml_fma_f32((d[(8)] * -1.000000e+00f), -2.000000e+00f, data_pack_local[(12)]);
  data_pack_local[(12)] = __ocml_fma_f32((d[(9)] * -1.000000e+00f), 1.500000e+00f, data_pack_local[(12)]);
  data_pack_local[(12)] = __ocml_fma_f32(d[(10)], -1.000000e+00f, data_pack_local[(12)]);
  data_pack_local[(12)] = __ocml_fma_f32(d[(12)], 5.000000e-01f, data_pack_local[(12)]);
  data_pack_local[(12)] = __ocml_fma_f32((d[(13)] * 5.000000e-01f), -1.500000e+00f, data_pack_local[(12)]);
  data_pack_local[(12)] = __ocml_fma_f32((d[(14)] * 5.000000e-01f), -2.000000e+00f, data_pack_local[(12)]);
  data_pack_local[(12)] = __ocml_fma_f32((d[(15)] * 5.000000e-01f), 1.500000e+00f, data_pack_local[(12)]);
  data_pack_local[(12)] = __ocml_fma_f32(d[(16)], 5.000000e-01f, data_pack_local[(12)]);
  data_pack_local[(12)] = __ocml_fma_f32(d[(18)], 2.500000e+00f, data_pack_local[(12)]);
  data_pack_local[(12)] = __ocml_fma_f32((d[(19)] * 2.500000e+00f), -1.500000e+00f, data_pack_local[(12)]);
  data_pack_local[(12)] = __ocml_fma_f32((d[(20)] * 2.500000e+00f), -2.000000e+00f, data_pack_local[(12)]);
  data_pack_local[(12)] = __ocml_fma_f32((d[(21)] * 2.500000e+00f), 1.500000e+00f, data_pack_local[(12)]);
  data_pack_local[(12)] = __ocml_fma_f32(d[(22)], 2.500000e+00f, data_pack_local[(12)]);
  data_pack_local[(12)] = (data_pack_local[(12)] + d[(24)]);
  data_pack_local[(12)] = __ocml_fma_f32(d[(25)], -1.500000e+00f, data_pack_local[(12)]);
  data_pack_local[(12)] = __ocml_fma_f32(d[(26)], -2.000000e+00f, data_pack_local[(12)]);
  data_pack_local[(12)] = __ocml_fma_f32(d[(27)], 1.500000e+00f, data_pack_local[(12)]);
  data_pack_local[(12)] = (data_pack_local[(12)] + d[(28)]);
  data_pack_local[(13)] = 0.000000e+00f;
  data_pack_local[(13)] = __ocml_fma_f32(d[(7)], -1.000000e+00f, data_pack_local[(13)]);
  data_pack_local[(13)] = __ocml_fma_f32((d[(8)] * -1.000000e+00f), -2.500000e+00f, data_pack_local[(13)]);
  data_pack_local[(13)] = __ocml_fma_f32((d[(9)] * -1.000000e+00f), 5.000000e-01f, data_pack_local[(13)]);
  data_pack_local[(13)] = __ocml_fma_f32(d[(10)], -1.000000e+00f, data_pack_local[(13)]);
  data_pack_local[(13)] = __ocml_fma_f32(d[(13)], 5.000000e-01f, data_pack_local[(13)]);
  data_pack_local[(13)] = __ocml_fma_f32((d[(14)] * 5.000000e-01f), -2.500000e+00f, data_pack_local[(13)]);
  data_pack_local[(13)] = __ocml_fma_f32((d[(15)] * 5.000000e-01f), 5.000000e-01f, data_pack_local[(13)]);
  data_pack_local[(13)] = __ocml_fma_f32(d[(16)], 5.000000e-01f, data_pack_local[(13)]);
  data_pack_local[(13)] = __ocml_fma_f32(d[(19)], 2.500000e+00f, data_pack_local[(13)]);
  data_pack_local[(13)] = __ocml_fma_f32((d[(20)] * 2.500000e+00f), -2.500000e+00f, data_pack_local[(13)]);
  data_pack_local[(13)] = __ocml_fma_f32((d[(21)] * 2.500000e+00f), 5.000000e-01f, data_pack_local[(13)]);
  data_pack_local[(13)] = __ocml_fma_f32(d[(22)], 2.500000e+00f, data_pack_local[(13)]);
  data_pack_local[(13)] = (data_pack_local[(13)] + d[(25)]);
  data_pack_local[(13)] = __ocml_fma_f32(d[(26)], -2.500000e+00f, data_pack_local[(13)]);
  data_pack_local[(13)] = __ocml_fma_f32(d[(27)], 5.000000e-01f, data_pack_local[(13)]);
  data_pack_local[(13)] = (data_pack_local[(13)] + d[(28)]);
  data_pack_local[(14)] = 0.000000e+00f;
  data_pack_local[(14)] = __ocml_fma_f32((d[(7)] * -1.000000e+00f), -1.000000e+00f, data_pack_local[(14)]);
  data_pack_local[(14)] = __ocml_fma_f32((d[(8)] * -1.000000e+00f), 5.000000e-01f, data_pack_local[(14)]);
  data_pack_local[(14)] = __ocml_fma_f32((d[(9)] * -1.000000e+00f), 2.500000e+00f, data_pack_local[(14)]);
  data_pack_local[(14)] = __ocml_fma_f32(d[(10)], -1.000000e+00f, data_pack_local[(14)]);
  data_pack_local[(14)] = __ocml_fma_f32((d[(13)] * 5.000000e-01f), -1.000000e+00f, data_pack_local[(14)]);
  data_pack_local[(14)] = __ocml_fma_f32((d[(14)] * 5.000000e-01f), 5.000000e-01f, data_pack_local[(14)]);
  data_pack_local[(14)] = __ocml_fma_f32((d[(15)] * 5.000000e-01f), 2.500000e+00f, data_pack_local[(14)]);
  data_pack_local[(14)] = __ocml_fma_f32(d[(16)], 5.000000e-01f, data_pack_local[(14)]);
  data_pack_local[(14)] = __ocml_fma_f32((d[(19)] * 2.500000e+00f), -1.000000e+00f, data_pack_local[(14)]);
  data_pack_local[(14)] = __ocml_fma_f32((d[(20)] * 2.500000e+00f), 5.000000e-01f, data_pack_local[(14)]);
  data_pack_local[(14)] = __ocml_fma_f32((d[(21)] * 2.500000e+00f), 2.500000e+00f, data_pack_local[(14)]);
  data_pack_local[(14)] = __ocml_fma_f32(d[(22)], 2.500000e+00f, data_pack_local[(14)]);
  data_pack_local[(14)] = __ocml_fma_f32(d[(25)], -1.000000e+00f, data_pack_local[(14)]);
  data_pack_local[(14)] = __ocml_fma_f32(d[(26)], 5.000000e-01f, data_pack_local[(14)]);
  data_pack_local[(14)] = __ocml_fma_f32(d[(27)], 2.500000e+00f, data_pack_local[(14)]);
  data_pack_local[(14)] = (data_pack_local[(14)] + d[(28)]);
  data_pack_local[(15)] = 0.000000e+00f;
  data_pack_local[(15)] = __ocml_fma_f32((d[(7)] * -1.000000e+00f), -2.000000e+00f, data_pack_local[(15)]);
  data_pack_local[(15)] = __ocml_fma_f32((d[(8)] * -1.000000e+00f), -1.000000e+00f, data_pack_local[(15)]);
  data_pack_local[(15)] = __ocml_fma_f32((d[(9)] * -1.000000e+00f), 2.000000e+00f, data_pack_local[(15)]);
  data_pack_local[(15)] = __ocml_fma_f32(d[(10)], -1.000000e+00f, data_pack_local[(15)]);
  data_pack_local[(15)] = __ocml_fma_f32((d[(13)] * 5.000000e-01f), -2.000000e+00f, data_pack_local[(15)]);
  data_pack_local[(15)] = __ocml_fma_f32((d[(14)] * 5.000000e-01f), -1.000000e+00f, data_pack_local[(15)]);
  data_pack_local[(15)] = __ocml_fma_f32((d[(15)] * 5.000000e-01f), 2.000000e+00f, data_pack_local[(15)]);
  data_pack_local[(15)] = __ocml_fma_f32(d[(16)], 5.000000e-01f, data_pack_local[(15)]);
  data_pack_local[(15)] = __ocml_fma_f32((d[(19)] * 2.500000e+00f), -2.000000e+00f, data_pack_local[(15)]);
  data_pack_local[(15)] = __ocml_fma_f32((d[(20)] * 2.500000e+00f), -1.000000e+00f, data_pack_local[(15)]);
  data_pack_local[(15)] = __ocml_fma_f32((d[(21)] * 2.500000e+00f), 2.000000e+00f, data_pack_local[(15)]);
  data_pack_local[(15)] = __ocml_fma_f32(d[(22)], 2.500000e+00f, data_pack_local[(15)]);
  data_pack_local[(15)] = __ocml_fma_f32(d[(25)], -2.000000e+00f, data_pack_local[(15)]);
  data_pack_local[(15)] = __ocml_fma_f32(d[(26)], -1.000000e+00f, data_pack_local[(15)]);
  data_pack_local[(15)] = __ocml_fma_f32(d[(27)], 2.000000e+00f, data_pack_local[(15)]);
  data_pack_local[(15)] = (data_pack_local[(15)] + d[(28)]);
  data_pack_local[(16)] = 0.000000e+00f;
  data_pack_local[(16)] = __ocml_fma_f32((d[(7)] * -1.000000e+00f), 5.000000e-01f, data_pack_local[(16)]);
  data_pack_local[(16)] = __ocml_fma_f32((d[(8)] * -1.000000e+00f), -1.000000e+00f, data_pack_local[(16)]);
  data_pack_local[(16)] = __ocml_fma_f32((d[(9)] * -1.000000e+00f), -5.000000e-01f, data_pack_local[(16)]);
  data_pack_local[(16)] = __ocml_fma_f32(d[(10)], -1.000000e+00f, data_pack_local[(16)]);
  data_pack_local[(16)] = __ocml_fma_f32((d[(13)] * 5.000000e-01f), 5.000000e-01f, data_pack_local[(16)]);
  data_pack_local[(16)] = __ocml_fma_f32((d[(14)] * 5.000000e-01f), -1.000000e+00f, data_pack_local[(16)]);
  data_pack_local[(16)] = __ocml_fma_f32((d[(15)] * 5.000000e-01f), -5.000000e-01f, data_pack_local[(16)]);
  data_pack_local[(16)] = __ocml_fma_f32(d[(16)], 5.000000e-01f, data_pack_local[(16)]);
  data_pack_local[(16)] = __ocml_fma_f32((d[(19)] * 2.500000e+00f), 5.000000e-01f, data_pack_local[(16)]);
  data_pack_local[(16)] = __ocml_fma_f32((d[(20)] * 2.500000e+00f), -1.000000e+00f, data_pack_local[(16)]);
  data_pack_local[(16)] = __ocml_fma_f32((d[(21)] * 2.500000e+00f), -5.000000e-01f, data_pack_local[(16)]);
  data_pack_local[(16)] = __ocml_fma_f32(d[(22)], 2.500000e+00f, data_pack_local[(16)]);
  data_pack_local[(16)] = __ocml_fma_f32(d[(25)], 5.000000e-01f, data_pack_local[(16)]);
  data_pack_local[(16)] = __ocml_fma_f32(d[(26)], -1.000000e+00f, data_pack_local[(16)]);
  data_pack_local[(16)] = __ocml_fma_f32(d[(27)], -5.000000e-01f, data_pack_local[(16)]);
  data_pack_local[(16)] = (data_pack_local[(16)] + d[(28)]);
  data_pack_local[(17)] = 0.000000e+00f;
  data_pack_local[(17)] = __ocml_fma_f32(d[(7)], -1.000000e+00f, data_pack_local[(17)]);
  data_pack_local[(17)] = __ocml_fma_f32((d[(8)] * -1.000000e+00f), -1.500000e+00f, data_pack_local[(17)]);
  data_pack_local[(17)] = __ocml_fma_f32((d[(9)] * -1.000000e+00f), -2.000000e+00f, data_pack_local[(17)]);
  data_pack_local[(17)] = __ocml_fma_f32((d[(10)] * -1.000000e+00f), 1.500000e+00f, data_pack_local[(17)]);
  data_pack_local[(17)] = __ocml_fma_f32(d[(11)], -1.000000e+00f, data_pack_local[(17)]);
  data_pack_local[(17)] = __ocml_fma_f32(d[(13)], 5.000000e-01f, data_pack_local[(17)]);
  data_pack_local[(17)] = __ocml_fma_f32((d[(14)] * 5.000000e-01f), -1.500000e+00f, data_pack_local[(17)]);
  data_pack_local[(17)] = __ocml_fma_f32((d[(15)] * 5.000000e-01f), -2.000000e+00f, data_pack_local[(17)]);
  data_pack_local[(17)] = __ocml_fma_f32((d[(16)] * 5.000000e-01f), 1.500000e+00f, data_pack_local[(17)]);
  data_pack_local[(17)] = __ocml_fma_f32(d[(17)], 5.000000e-01f, data_pack_local[(17)]);
  data_pack_local[(17)] = __ocml_fma_f32(d[(19)], 2.500000e+00f, data_pack_local[(17)]);
  data_pack_local[(17)] = __ocml_fma_f32((d[(20)] * 2.500000e+00f), -1.500000e+00f, data_pack_local[(17)]);
  data_pack_local[(17)] = __ocml_fma_f32((d[(21)] * 2.500000e+00f), -2.000000e+00f, data_pack_local[(17)]);
  data_pack_local[(17)] = __ocml_fma_f32((d[(22)] * 2.500000e+00f), 1.500000e+00f, data_pack_local[(17)]);
  data_pack_local[(17)] = __ocml_fma_f32(d[(23)], 2.500000e+00f, data_pack_local[(17)]);
  data_pack_local[(17)] = (data_pack_local[(17)] + d[(25)]);
  data_pack_local[(17)] = __ocml_fma_f32(d[(26)], -1.500000e+00f, data_pack_local[(17)]);
  data_pack_local[(17)] = __ocml_fma_f32(d[(27)], -2.000000e+00f, data_pack_local[(17)]);
  data_pack_local[(17)] = __ocml_fma_f32(d[(28)], 1.500000e+00f, data_pack_local[(17)]);
  data_pack_local[(17)] = (data_pack_local[(17)] + d[(29)]);
  data_pack_local[(18)] = 0.000000e+00f;
  data_pack_local[(18)] = __ocml_fma_f32(d[(6)], -2.000000e+00f, data_pack_local[(18)]);
  data_pack_local[(18)] = __ocml_fma_f32((d[(7)] * -2.000000e+00f), -1.500000e+00f, data_pack_local[(18)]);
  data_pack_local[(18)] = __ocml_fma_f32((d[(8)] * -2.000000e+00f), -2.000000e+00f, data_pack_local[(18)]);
  data_pack_local[(18)] = __ocml_fma_f32((d[(9)] * -2.000000e+00f), 1.500000e+00f, data_pack_local[(18)]);
  data_pack_local[(18)] = __ocml_fma_f32(d[(10)], -2.000000e+00f, data_pack_local[(18)]);
  data_pack_local[(18)] = __ocml_fma_f32(d[(12)], -1.000000e+00f, data_pack_local[(18)]);
  data_pack_local[(18)] = __ocml_fma_f32((d[(13)] * -1.000000e+00f), -1.500000e+00f, data_pack_local[(18)]);
  data_pack_local[(18)] = __ocml_fma_f32((d[(14)] * -1.000000e+00f), -2.000000e+00f, data_pack_local[(18)]);
  data_pack_local[(18)] = __ocml_fma_f32((d[(15)] * -1.000000e+00f), 1.500000e+00f, data_pack_local[(18)]);
  data_pack_local[(18)] = __ocml_fma_f32(d[(16)], -1.000000e+00f, data_pack_local[(18)]);
  data_pack_local[(18)] = __ocml_fma_f32(d[(18)], 2.000000e+00f, data_pack_local[(18)]);
  data_pack_local[(18)] = __ocml_fma_f32((d[(19)] * 2.000000e+00f), -1.500000e+00f, data_pack_local[(18)]);
  data_pack_local[(18)] = __ocml_fma_f32((d[(20)] * 2.000000e+00f), -2.000000e+00f, data_pack_local[(18)]);
  data_pack_local[(18)] = __ocml_fma_f32((d[(21)] * 2.000000e+00f), 1.500000e+00f, data_pack_local[(18)]);
  data_pack_local[(18)] = __ocml_fma_f32(d[(22)], 2.000000e+00f, data_pack_local[(18)]);
  data_pack_local[(18)] = (data_pack_local[(18)] + d[(24)]);
  data_pack_local[(18)] = __ocml_fma_f32(d[(25)], -1.500000e+00f, data_pack_local[(18)]);
  data_pack_local[(18)] = __ocml_fma_f32(d[(26)], -2.000000e+00f, data_pack_local[(18)]);
  data_pack_local[(18)] = __ocml_fma_f32(d[(27)], 1.500000e+00f, data_pack_local[(18)]);
  data_pack_local[(18)] = (data_pack_local[(18)] + d[(28)]);
  data_pack_local[(19)] = 0.000000e+00f;
  data_pack_local[(19)] = __ocml_fma_f32(d[(7)], -2.000000e+00f, data_pack_local[(19)]);
  data_pack_local[(19)] = __ocml_fma_f32((d[(8)] * -2.000000e+00f), -2.500000e+00f, data_pack_local[(19)]);
  data_pack_local[(19)] = __ocml_fma_f32((d[(9)] * -2.000000e+00f), 5.000000e-01f, data_pack_local[(19)]);
  data_pack_local[(19)] = __ocml_fma_f32(d[(10)], -2.000000e+00f, data_pack_local[(19)]);
  data_pack_local[(19)] = __ocml_fma_f32(d[(13)], -1.000000e+00f, data_pack_local[(19)]);
  data_pack_local[(19)] = __ocml_fma_f32((d[(14)] * -1.000000e+00f), -2.500000e+00f, data_pack_local[(19)]);
  data_pack_local[(19)] = __ocml_fma_f32((d[(15)] * -1.000000e+00f), 5.000000e-01f, data_pack_local[(19)]);
  data_pack_local[(19)] = __ocml_fma_f32(d[(16)], -1.000000e+00f, data_pack_local[(19)]);
  data_pack_local[(19)] = __ocml_fma_f32(d[(19)], 2.000000e+00f, data_pack_local[(19)]);
  data_pack_local[(19)] = __ocml_fma_f32((d[(20)] * 2.000000e+00f), -2.500000e+00f, data_pack_local[(19)]);
  data_pack_local[(19)] = __ocml_fma_f32((d[(21)] * 2.000000e+00f), 5.000000e-01f, data_pack_local[(19)]);
  data_pack_local[(19)] = __ocml_fma_f32(d[(22)], 2.000000e+00f, data_pack_local[(19)]);
  data_pack_local[(19)] = (data_pack_local[(19)] + d[(25)]);
  data_pack_local[(19)] = __ocml_fma_f32(d[(26)], -2.500000e+00f, data_pack_local[(19)]);
  data_pack_local[(19)] = __ocml_fma_f32(d[(27)], 5.000000e-01f, data_pack_local[(19)]);
  data_pack_local[(19)] = (data_pack_local[(19)] + d[(28)]);
  data_pack_local[(20)] = 0.000000e+00f;
  data_pack_local[(20)] = __ocml_fma_f32((d[(7)] * -2.000000e+00f), -1.000000e+00f, data_pack_local[(20)]);
  data_pack_local[(20)] = __ocml_fma_f32((d[(8)] * -2.000000e+00f), 5.000000e-01f, data_pack_local[(20)]);
  data_pack_local[(20)] = __ocml_fma_f32((d[(9)] * -2.000000e+00f), 2.500000e+00f, data_pack_local[(20)]);
  data_pack_local[(20)] = __ocml_fma_f32(d[(10)], -2.000000e+00f, data_pack_local[(20)]);
  data_pack_local[(20)] = __ocml_fma_f32((d[(13)] * -1.000000e+00f), -1.000000e+00f, data_pack_local[(20)]);
  data_pack_local[(20)] = __ocml_fma_f32((d[(14)] * -1.000000e+00f), 5.000000e-01f, data_pack_local[(20)]);
  data_pack_local[(20)] = __ocml_fma_f32((d[(15)] * -1.000000e+00f), 2.500000e+00f, data_pack_local[(20)]);
  data_pack_local[(20)] = __ocml_fma_f32(d[(16)], -1.000000e+00f, data_pack_local[(20)]);
  data_pack_local[(20)] = __ocml_fma_f32((d[(19)] * 2.000000e+00f), -1.000000e+00f, data_pack_local[(20)]);
  data_pack_local[(20)] = __ocml_fma_f32((d[(20)] * 2.000000e+00f), 5.000000e-01f, data_pack_local[(20)]);
  data_pack_local[(20)] = __ocml_fma_f32((d[(21)] * 2.000000e+00f), 2.500000e+00f, data_pack_local[(20)]);
  data_pack_local[(20)] = __ocml_fma_f32(d[(22)], 2.000000e+00f, data_pack_local[(20)]);
  data_pack_local[(20)] = __ocml_fma_f32(d[(25)], -1.000000e+00f, data_pack_local[(20)]);
  data_pack_local[(20)] = __ocml_fma_f32(d[(26)], 5.000000e-01f, data_pack_local[(20)]);
  data_pack_local[(20)] = __ocml_fma_f32(d[(27)], 2.500000e+00f, data_pack_local[(20)]);
  data_pack_local[(20)] = (data_pack_local[(20)] + d[(28)]);
  data_pack_local[(21)] = 0.000000e+00f;
  data_pack_local[(21)] = __ocml_fma_f32((d[(7)] * -2.000000e+00f), -2.000000e+00f, data_pack_local[(21)]);
  data_pack_local[(21)] = __ocml_fma_f32((d[(8)] * -2.000000e+00f), -1.000000e+00f, data_pack_local[(21)]);
  data_pack_local[(21)] = __ocml_fma_f32((d[(9)] * -2.000000e+00f), 2.000000e+00f, data_pack_local[(21)]);
  data_pack_local[(21)] = __ocml_fma_f32(d[(10)], -2.000000e+00f, data_pack_local[(21)]);
  data_pack_local[(21)] = __ocml_fma_f32((d[(13)] * -1.000000e+00f), -2.000000e+00f, data_pack_local[(21)]);
  data_pack_local[(21)] = __ocml_fma_f32((d[(14)] * -1.000000e+00f), -1.000000e+00f, data_pack_local[(21)]);
  data_pack_local[(21)] = __ocml_fma_f32((d[(15)] * -1.000000e+00f), 2.000000e+00f, data_pack_local[(21)]);
  data_pack_local[(21)] = __ocml_fma_f32(d[(16)], -1.000000e+00f, data_pack_local[(21)]);
  data_pack_local[(21)] = __ocml_fma_f32((d[(19)] * 2.000000e+00f), -2.000000e+00f, data_pack_local[(21)]);
  data_pack_local[(21)] = __ocml_fma_f32((d[(20)] * 2.000000e+00f), -1.000000e+00f, data_pack_local[(21)]);
  data_pack_local[(21)] = __ocml_fma_f32((d[(21)] * 2.000000e+00f), 2.000000e+00f, data_pack_local[(21)]);
  data_pack_local[(21)] = __ocml_fma_f32(d[(22)], 2.000000e+00f, data_pack_local[(21)]);
  data_pack_local[(21)] = __ocml_fma_f32(d[(25)], -2.000000e+00f, data_pack_local[(21)]);
  data_pack_local[(21)] = __ocml_fma_f32(d[(26)], -1.000000e+00f, data_pack_local[(21)]);
  data_pack_local[(21)] = __ocml_fma_f32(d[(27)], 2.000000e+00f, data_pack_local[(21)]);
  data_pack_local[(21)] = (data_pack_local[(21)] + d[(28)]);
  data_pack_local[(22)] = 0.000000e+00f;
  data_pack_local[(22)] = __ocml_fma_f32((d[(7)] * -2.000000e+00f), 5.000000e-01f, data_pack_local[(22)]);
  data_pack_local[(22)] = __ocml_fma_f32((d[(8)] * -2.000000e+00f), -1.000000e+00f, data_pack_local[(22)]);
  data_pack_local[(22)] = __ocml_fma_f32((d[(9)] * -2.000000e+00f), -5.000000e-01f, data_pack_local[(22)]);
  data_pack_local[(22)] = __ocml_fma_f32(d[(10)], -2.000000e+00f, data_pack_local[(22)]);
  data_pack_local[(22)] = __ocml_fma_f32((d[(13)] * -1.000000e+00f), 5.000000e-01f, data_pack_local[(22)]);
  data_pack_local[(22)] = __ocml_fma_f32((d[(14)] * -1.000000e+00f), -1.000000e+00f, data_pack_local[(22)]);
  data_pack_local[(22)] = __ocml_fma_f32((d[(15)] * -1.000000e+00f), -5.000000e-01f, data_pack_local[(22)]);
  data_pack_local[(22)] = __ocml_fma_f32(d[(16)], -1.000000e+00f, data_pack_local[(22)]);
  data_pack_local[(22)] = __ocml_fma_f32((d[(19)] * 2.000000e+00f), 5.000000e-01f, data_pack_local[(22)]);
  data_pack_local[(22)] = __ocml_fma_f32((d[(20)] * 2.000000e+00f), -1.000000e+00f, data_pack_local[(22)]);
  data_pack_local[(22)] = __ocml_fma_f32((d[(21)] * 2.000000e+00f), -5.000000e-01f, data_pack_local[(22)]);
  data_pack_local[(22)] = __ocml_fma_f32(d[(22)], 2.000000e+00f, data_pack_local[(22)]);
  data_pack_local[(22)] = __ocml_fma_f32(d[(25)], 5.000000e-01f, data_pack_local[(22)]);
  data_pack_local[(22)] = __ocml_fma_f32(d[(26)], -1.000000e+00f, data_pack_local[(22)]);
  data_pack_local[(22)] = __ocml_fma_f32(d[(27)], -5.000000e-01f, data_pack_local[(22)]);
  data_pack_local[(22)] = (data_pack_local[(22)] + d[(28)]);
  data_pack_local[(23)] = 0.000000e+00f;
  data_pack_local[(23)] = __ocml_fma_f32(d[(7)], -2.000000e+00f, data_pack_local[(23)]);
  data_pack_local[(23)] = __ocml_fma_f32((d[(8)] * -2.000000e+00f), -1.500000e+00f, data_pack_local[(23)]);
  data_pack_local[(23)] = __ocml_fma_f32((d[(9)] * -2.000000e+00f), -2.000000e+00f, data_pack_local[(23)]);
  data_pack_local[(23)] = __ocml_fma_f32((d[(10)] * -2.000000e+00f), 1.500000e+00f, data_pack_local[(23)]);
  data_pack_local[(23)] = __ocml_fma_f32(d[(11)], -2.000000e+00f, data_pack_local[(23)]);
  data_pack_local[(23)] = __ocml_fma_f32(d[(13)], -1.000000e+00f, data_pack_local[(23)]);
  data_pack_local[(23)] = __ocml_fma_f32((d[(14)] * -1.000000e+00f), -1.500000e+00f, data_pack_local[(23)]);
  data_pack_local[(23)] = __ocml_fma_f32((d[(15)] * -1.000000e+00f), -2.000000e+00f, data_pack_local[(23)]);
  data_pack_local[(23)] = __ocml_fma_f32((d[(16)] * -1.000000e+00f), 1.500000e+00f, data_pack_local[(23)]);
  data_pack_local[(23)] = __ocml_fma_f32(d[(17)], -1.000000e+00f, data_pack_local[(23)]);
  data_pack_local[(23)] = __ocml_fma_f32(d[(19)], 2.000000e+00f, data_pack_local[(23)]);
  data_pack_local[(23)] = __ocml_fma_f32((d[(20)] * 2.000000e+00f), -1.500000e+00f, data_pack_local[(23)]);
  data_pack_local[(23)] = __ocml_fma_f32((d[(21)] * 2.000000e+00f), -2.000000e+00f, data_pack_local[(23)]);
  data_pack_local[(23)] = __ocml_fma_f32((d[(22)] * 2.000000e+00f), 1.500000e+00f, data_pack_local[(23)]);
  data_pack_local[(23)] = __ocml_fma_f32(d[(23)], 2.000000e+00f, data_pack_local[(23)]);
  data_pack_local[(23)] = (data_pack_local[(23)] + d[(25)]);
  data_pack_local[(23)] = __ocml_fma_f32(d[(26)], -1.500000e+00f, data_pack_local[(23)]);
  data_pack_local[(23)] = __ocml_fma_f32(d[(27)], -2.000000e+00f, data_pack_local[(23)]);
  data_pack_local[(23)] = __ocml_fma_f32(d[(28)], 1.500000e+00f, data_pack_local[(23)]);
  data_pack_local[(23)] = (data_pack_local[(23)] + d[(29)]);
  data_pack_local[(24)] = 0.000000e+00f;
  data_pack_local[(24)] = __ocml_fma_f32(d[(6)], 5.000000e-01f, data_pack_local[(24)]);
  data_pack_local[(24)] = __ocml_fma_f32((d[(7)] * 5.000000e-01f), -1.500000e+00f, data_pack_local[(24)]);
  data_pack_local[(24)] = __ocml_fma_f32((d[(8)] * 5.000000e-01f), -2.000000e+00f, data_pack_local[(24)]);
  data_pack_local[(24)] = __ocml_fma_f32((d[(9)] * 5.000000e-01f), 1.500000e+00f, data_pack_local[(24)]);
  data_pack_local[(24)] = __ocml_fma_f32(d[(10)], 5.000000e-01f, data_pack_local[(24)]);
  data_pack_local[(24)] = __ocml_fma_f32(d[(12)], -1.000000e+00f, data_pack_local[(24)]);
  data_pack_local[(24)] = __ocml_fma_f32((d[(13)] * -1.000000e+00f), -1.500000e+00f, data_pack_local[(24)]);
  data_pack_local[(24)] = __ocml_fma_f32((d[(14)] * -1.000000e+00f), -2.000000e+00f, data_pack_local[(24)]);
  data_pack_local[(24)] = __ocml_fma_f32((d[(15)] * -1.000000e+00f), 1.500000e+00f, data_pack_local[(24)]);
  data_pack_local[(24)] = __ocml_fma_f32(d[(16)], -1.000000e+00f, data_pack_local[(24)]);
  data_pack_local[(24)] = __ocml_fma_f32(d[(18)], -5.000000e-01f, data_pack_local[(24)]);
  data_pack_local[(24)] = __ocml_fma_f32((d[(19)] * -5.000000e-01f), -1.500000e+00f, data_pack_local[(24)]);
  data_pack_local[(24)] = __ocml_fma_f32((d[(20)] * -5.000000e-01f), -2.000000e+00f, data_pack_local[(24)]);
  data_pack_local[(24)] = __ocml_fma_f32((d[(21)] * -5.000000e-01f), 1.500000e+00f, data_pack_local[(24)]);
  data_pack_local[(24)] = __ocml_fma_f32(d[(22)], -5.000000e-01f, data_pack_local[(24)]);
  data_pack_local[(24)] = (data_pack_local[(24)] + d[(24)]);
  data_pack_local[(24)] = __ocml_fma_f32(d[(25)], -1.500000e+00f, data_pack_local[(24)]);
  data_pack_local[(24)] = __ocml_fma_f32(d[(26)], -2.000000e+00f, data_pack_local[(24)]);
  data_pack_local[(24)] = __ocml_fma_f32(d[(27)], 1.500000e+00f, data_pack_local[(24)]);
  data_pack_local[(24)] = (data_pack_local[(24)] + d[(28)]);
  data_pack_local[(25)] = 0.000000e+00f;
  data_pack_local[(25)] = __ocml_fma_f32(d[(7)], 5.000000e-01f, data_pack_local[(25)]);
  data_pack_local[(25)] = __ocml_fma_f32((d[(8)] * 5.000000e-01f), -2.500000e+00f, data_pack_local[(25)]);
  data_pack_local[(25)] = __ocml_fma_f32((d[(9)] * 5.000000e-01f), 5.000000e-01f, data_pack_local[(25)]);
  data_pack_local[(25)] = __ocml_fma_f32(d[(10)], 5.000000e-01f, data_pack_local[(25)]);
  data_pack_local[(25)] = __ocml_fma_f32(d[(13)], -1.000000e+00f, data_pack_local[(25)]);
  data_pack_local[(25)] = __ocml_fma_f32((d[(14)] * -1.000000e+00f), -2.500000e+00f, data_pack_local[(25)]);
  data_pack_local[(25)] = __ocml_fma_f32((d[(15)] * -1.000000e+00f), 5.000000e-01f, data_pack_local[(25)]);
  data_pack_local[(25)] = __ocml_fma_f32(d[(16)], -1.000000e+00f, data_pack_local[(25)]);
  data_pack_local[(25)] = __ocml_fma_f32(d[(19)], -5.000000e-01f, data_pack_local[(25)]);
  data_pack_local[(25)] = __ocml_fma_f32((d[(20)] * -5.000000e-01f), -2.500000e+00f, data_pack_local[(25)]);
  data_pack_local[(25)] = __ocml_fma_f32((d[(21)] * -5.000000e-01f), 5.000000e-01f, data_pack_local[(25)]);
  data_pack_local[(25)] = __ocml_fma_f32(d[(22)], -5.000000e-01f, data_pack_local[(25)]);
  data_pack_local[(25)] = (data_pack_local[(25)] + d[(25)]);
  data_pack_local[(25)] = __ocml_fma_f32(d[(26)], -2.500000e+00f, data_pack_local[(25)]);
  data_pack_local[(25)] = __ocml_fma_f32(d[(27)], 5.000000e-01f, data_pack_local[(25)]);
  data_pack_local[(25)] = (data_pack_local[(25)] + d[(28)]);
  data_pack_local[(26)] = 0.000000e+00f;
  data_pack_local[(26)] = __ocml_fma_f32((d[(7)] * 5.000000e-01f), -1.000000e+00f, data_pack_local[(26)]);
  data_pack_local[(26)] = __ocml_fma_f32((d[(8)] * 5.000000e-01f), 5.000000e-01f, data_pack_local[(26)]);
  data_pack_local[(26)] = __ocml_fma_f32((d[(9)] * 5.000000e-01f), 2.500000e+00f, data_pack_local[(26)]);
  data_pack_local[(26)] = __ocml_fma_f32(d[(10)], 5.000000e-01f, data_pack_local[(26)]);
  data_pack_local[(26)] = __ocml_fma_f32((d[(13)] * -1.000000e+00f), -1.000000e+00f, data_pack_local[(26)]);
  data_pack_local[(26)] = __ocml_fma_f32((d[(14)] * -1.000000e+00f), 5.000000e-01f, data_pack_local[(26)]);
  data_pack_local[(26)] = __ocml_fma_f32((d[(15)] * -1.000000e+00f), 2.500000e+00f, data_pack_local[(26)]);
  data_pack_local[(26)] = __ocml_fma_f32(d[(16)], -1.000000e+00f, data_pack_local[(26)]);
  data_pack_local[(26)] = __ocml_fma_f32((d[(19)] * -5.000000e-01f), -1.000000e+00f, data_pack_local[(26)]);
  data_pack_local[(26)] = __ocml_fma_f32((d[(20)] * -5.000000e-01f), 5.000000e-01f, data_pack_local[(26)]);
  data_pack_local[(26)] = __ocml_fma_f32((d[(21)] * -5.000000e-01f), 2.500000e+00f, data_pack_local[(26)]);
  data_pack_local[(26)] = __ocml_fma_f32(d[(22)], -5.000000e-01f, data_pack_local[(26)]);
  data_pack_local[(26)] = __ocml_fma_f32(d[(25)], -1.000000e+00f, data_pack_local[(26)]);
  data_pack_local[(26)] = __ocml_fma_f32(d[(26)], 5.000000e-01f, data_pack_local[(26)]);
  data_pack_local[(26)] = __ocml_fma_f32(d[(27)], 2.500000e+00f, data_pack_local[(26)]);
  data_pack_local[(26)] = (data_pack_local[(26)] + d[(28)]);
  data_pack_local[(27)] = 0.000000e+00f;
  data_pack_local[(27)] = __ocml_fma_f32((d[(7)] * 5.000000e-01f), -2.000000e+00f, data_pack_local[(27)]);
  data_pack_local[(27)] = __ocml_fma_f32((d[(8)] * 5.000000e-01f), -1.000000e+00f, data_pack_local[(27)]);
  data_pack_local[(27)] = __ocml_fma_f32((d[(9)] * 5.000000e-01f), 2.000000e+00f, data_pack_local[(27)]);
  data_pack_local[(27)] = __ocml_fma_f32(d[(10)], 5.000000e-01f, data_pack_local[(27)]);
  data_pack_local[(27)] = __ocml_fma_f32((d[(13)] * -1.000000e+00f), -2.000000e+00f, data_pack_local[(27)]);
  data_pack_local[(27)] = __ocml_fma_f32((d[(14)] * -1.000000e+00f), -1.000000e+00f, data_pack_local[(27)]);
  data_pack_local[(27)] = __ocml_fma_f32((d[(15)] * -1.000000e+00f), 2.000000e+00f, data_pack_local[(27)]);
  data_pack_local[(27)] = __ocml_fma_f32(d[(16)], -1.000000e+00f, data_pack_local[(27)]);
  data_pack_local[(27)] = __ocml_fma_f32((d[(19)] * -5.000000e-01f), -2.000000e+00f, data_pack_local[(27)]);
  data_pack_local[(27)] = __ocml_fma_f32((d[(20)] * -5.000000e-01f), -1.000000e+00f, data_pack_local[(27)]);
  data_pack_local[(27)] = __ocml_fma_f32((d[(21)] * -5.000000e-01f), 2.000000e+00f, data_pack_local[(27)]);
  data_pack_local[(27)] = __ocml_fma_f32(d[(22)], -5.000000e-01f, data_pack_local[(27)]);
  data_pack_local[(27)] = __ocml_fma_f32(d[(25)], -2.000000e+00f, data_pack_local[(27)]);
  data_pack_local[(27)] = __ocml_fma_f32(d[(26)], -1.000000e+00f, data_pack_local[(27)]);
  data_pack_local[(27)] = __ocml_fma_f32(d[(27)], 2.000000e+00f, data_pack_local[(27)]);
  data_pack_local[(27)] = (data_pack_local[(27)] + d[(28)]);
  data_pack_local[(28)] = 0.000000e+00f;
  data_pack_local[(28)] = __ocml_fma_f32((d[(7)] * 5.000000e-01f), 5.000000e-01f, data_pack_local[(28)]);
  data_pack_local[(28)] = __ocml_fma_f32((d[(8)] * 5.000000e-01f), -1.000000e+00f, data_pack_local[(28)]);
  data_pack_local[(28)] = __ocml_fma_f32((d[(9)] * 5.000000e-01f), -5.000000e-01f, data_pack_local[(28)]);
  data_pack_local[(28)] = __ocml_fma_f32(d[(10)], 5.000000e-01f, data_pack_local[(28)]);
  data_pack_local[(28)] = __ocml_fma_f32((d[(13)] * -1.000000e+00f), 5.000000e-01f, data_pack_local[(28)]);
  data_pack_local[(28)] = __ocml_fma_f32((d[(14)] * -1.000000e+00f), -1.000000e+00f, data_pack_local[(28)]);
  data_pack_local[(28)] = __ocml_fma_f32((d[(15)] * -1.000000e+00f), -5.000000e-01f, data_pack_local[(28)]);
  data_pack_local[(28)] = __ocml_fma_f32(d[(16)], -1.000000e+00f, data_pack_local[(28)]);
  data_pack_local[(28)] = __ocml_fma_f32((d[(19)] * -5.000000e-01f), 5.000000e-01f, data_pack_local[(28)]);
  data_pack_local[(28)] = __ocml_fma_f32((d[(20)] * -5.000000e-01f), -1.000000e+00f, data_pack_local[(28)]);
  data_pack_local[(28)] = __ocml_fma_f32((d[(21)] * -5.000000e-01f), -5.000000e-01f, data_pack_local[(28)]);
  data_pack_local[(28)] = __ocml_fma_f32(d[(22)], -5.000000e-01f, data_pack_local[(28)]);
  data_pack_local[(28)] = __ocml_fma_f32(d[(25)], 5.000000e-01f, data_pack_local[(28)]);
  data_pack_local[(28)] = __ocml_fma_f32(d[(26)], -1.000000e+00f, data_pack_local[(28)]);
  data_pack_local[(28)] = __ocml_fma_f32(d[(27)], -5.000000e-01f, data_pack_local[(28)]);
  data_pack_local[(28)] = (data_pack_local[(28)] + d[(28)]);
  data_pack_local[(29)] = 0.000000e+00f;
  data_pack_local[(29)] = __ocml_fma_f32(d[(7)], 5.000000e-01f, data_pack_local[(29)]);
  data_pack_local[(29)] = __ocml_fma_f32((d[(8)] * 5.000000e-01f), -1.500000e+00f, data_pack_local[(29)]);
  data_pack_local[(29)] = __ocml_fma_f32((d[(9)] * 5.000000e-01f), -2.000000e+00f, data_pack_local[(29)]);
  data_pack_local[(29)] = __ocml_fma_f32((d[(10)] * 5.000000e-01f), 1.500000e+00f, data_pack_local[(29)]);
  data_pack_local[(29)] = __ocml_fma_f32(d[(11)], 5.000000e-01f, data_pack_local[(29)]);
  data_pack_local[(29)] = __ocml_fma_f32(d[(13)], -1.000000e+00f, data_pack_local[(29)]);
  data_pack_local[(29)] = __ocml_fma_f32((d[(14)] * -1.000000e+00f), -1.500000e+00f, data_pack_local[(29)]);
  data_pack_local[(29)] = __ocml_fma_f32((d[(15)] * -1.000000e+00f), -2.000000e+00f, data_pack_local[(29)]);
  data_pack_local[(29)] = __ocml_fma_f32((d[(16)] * -1.000000e+00f), 1.500000e+00f, data_pack_local[(29)]);
  data_pack_local[(29)] = __ocml_fma_f32(d[(17)], -1.000000e+00f, data_pack_local[(29)]);
  data_pack_local[(29)] = __ocml_fma_f32(d[(19)], -5.000000e-01f, data_pack_local[(29)]);
  data_pack_local[(29)] = __ocml_fma_f32((d[(20)] * -5.000000e-01f), -1.500000e+00f, data_pack_local[(29)]);
  data_pack_local[(29)] = __ocml_fma_f32((d[(21)] * -5.000000e-01f), -2.000000e+00f, data_pack_local[(29)]);
  data_pack_local[(29)] = __ocml_fma_f32((d[(22)] * -5.000000e-01f), 1.500000e+00f, data_pack_local[(29)]);
  data_pack_local[(29)] = __ocml_fma_f32(d[(23)], -5.000000e-01f, data_pack_local[(29)]);
  data_pack_local[(29)] = (data_pack_local[(29)] + d[(25)]);
  data_pack_local[(29)] = __ocml_fma_f32(d[(26)], -1.500000e+00f, data_pack_local[(29)]);
  data_pack_local[(29)] = __ocml_fma_f32(d[(27)], -2.000000e+00f, data_pack_local[(29)]);
  data_pack_local[(29)] = __ocml_fma_f32(d[(28)], 1.500000e+00f, data_pack_local[(29)]);
  data_pack_local[(29)] = (data_pack_local[(29)] + d[(29)]);
  data_pack_local[(30)] = 0.000000e+00f;
  data_pack_local[(30)] = (data_pack_local[(30)] + d[(6)]);
  data_pack_local[(30)] = __ocml_fma_f32(d[(7)], -1.500000e+00f, data_pack_local[(30)]);
  data_pack_local[(30)] = __ocml_fma_f32(d[(8)], -2.000000e+00f, data_pack_local[(30)]);
  data_pack_local[(30)] = __ocml_fma_f32(d[(9)], 1.500000e+00f, data_pack_local[(30)]);
  data_pack_local[(30)] = (data_pack_local[(30)] + d[(10)]);
  data_pack_local[(30)] = __ocml_fma_f32(d[(12)], -1.500000e+00f, data_pack_local[(30)]);
  data_pack_local[(30)] = __ocml_fma_f32((d[(13)] * -1.500000e+00f), -1.500000e+00f, data_pack_local[(30)]);
  data_pack_local[(30)] = __ocml_fma_f32((d[(14)] * -1.500000e+00f), -2.000000e+00f, data_pack_local[(30)]);
  data_pack_local[(30)] = __ocml_fma_f32((d[(15)] * -1.500000e+00f), 1.500000e+00f, data_pack_local[(30)]);
  data_pack_local[(30)] = __ocml_fma_f32(d[(16)], -1.500000e+00f, data_pack_local[(30)]);
  data_pack_local[(30)] = __ocml_fma_f32(d[(18)], -2.000000e+00f, data_pack_local[(30)]);
  data_pack_local[(30)] = __ocml_fma_f32((d[(19)] * -2.000000e+00f), -1.500000e+00f, data_pack_local[(30)]);
  data_pack_local[(30)] = __ocml_fma_f32((d[(20)] * -2.000000e+00f), -2.000000e+00f, data_pack_local[(30)]);
  data_pack_local[(30)] = __ocml_fma_f32((d[(21)] * -2.000000e+00f), 1.500000e+00f, data_pack_local[(30)]);
  data_pack_local[(30)] = __ocml_fma_f32(d[(22)], -2.000000e+00f, data_pack_local[(30)]);
  data_pack_local[(30)] = __ocml_fma_f32(d[(24)], 1.500000e+00f, data_pack_local[(30)]);
  data_pack_local[(30)] = __ocml_fma_f32((d[(25)] * 1.500000e+00f), -1.500000e+00f, data_pack_local[(30)]);
  data_pack_local[(30)] = __ocml_fma_f32((d[(26)] * 1.500000e+00f), -2.000000e+00f, data_pack_local[(30)]);
  data_pack_local[(30)] = __ocml_fma_f32((d[(27)] * 1.500000e+00f), 1.500000e+00f, data_pack_local[(30)]);
  data_pack_local[(30)] = __ocml_fma_f32(d[(28)], 1.500000e+00f, data_pack_local[(30)]);
  data_pack_local[(30)] = (data_pack_local[(30)] + d[(30)]);
  data_pack_local[(30)] = __ocml_fma_f32(d[(31)], -1.500000e+00f, data_pack_local[(30)]);
  data_pack_local[(30)] = __ocml_fma_f32(d[(32)], -2.000000e+00f, data_pack_local[(30)]);
  data_pack_local[(30)] = __ocml_fma_f32(d[(33)], 1.500000e+00f, data_pack_local[(30)]);
  data_pack_local[(30)] = (data_pack_local[(30)] + d[(34)]);
  data_pack_local[(31)] = 0.000000e+00f;
  data_pack_local[(31)] = (data_pack_local[(31)] + d[(7)]);
  data_pack_local[(31)] = __ocml_fma_f32(d[(8)], -2.500000e+00f, data_pack_local[(31)]);
  data_pack_local[(31)] = __ocml_fma_f32(d[(9)], 5.000000e-01f, data_pack_local[(31)]);
  data_pack_local[(31)] = (data_pack_local[(31)] + d[(10)]);
  data_pack_local[(31)] = __ocml_fma_f32(d[(13)], -1.500000e+00f, data_pack_local[(31)]);
  data_pack_local[(31)] = __ocml_fma_f32((d[(14)] * -1.500000e+00f), -2.500000e+00f, data_pack_local[(31)]);
  data_pack_local[(31)] = __ocml_fma_f32((d[(15)] * -1.500000e+00f), 5.000000e-01f, data_pack_local[(31)]);
  data_pack_local[(31)] = __ocml_fma_f32(d[(16)], -1.500000e+00f, data_pack_local[(31)]);
  data_pack_local[(31)] = __ocml_fma_f32(d[(19)], -2.000000e+00f, data_pack_local[(31)]);
  data_pack_local[(31)] = __ocml_fma_f32((d[(20)] * -2.000000e+00f), -2.500000e+00f, data_pack_local[(31)]);
  data_pack_local[(31)] = __ocml_fma_f32((d[(21)] * -2.000000e+00f), 5.000000e-01f, data_pack_local[(31)]);
  data_pack_local[(31)] = __ocml_fma_f32(d[(22)], -2.000000e+00f, data_pack_local[(31)]);
  data_pack_local[(31)] = __ocml_fma_f32(d[(25)], 1.500000e+00f, data_pack_local[(31)]);
  data_pack_local[(31)] = __ocml_fma_f32((d[(26)] * 1.500000e+00f), -2.500000e+00f, data_pack_local[(31)]);
  data_pack_local[(31)] = __ocml_fma_f32((d[(27)] * 1.500000e+00f), 5.000000e-01f, data_pack_local[(31)]);
  data_pack_local[(31)] = __ocml_fma_f32(d[(28)], 1.500000e+00f, data_pack_local[(31)]);
  data_pack_local[(31)] = (data_pack_local[(31)] + d[(31)]);
  data_pack_local[(31)] = __ocml_fma_f32(d[(32)], -2.500000e+00f, data_pack_local[(31)]);
  data_pack_local[(31)] = __ocml_fma_f32(d[(33)], 5.000000e-01f, data_pack_local[(31)]);
  data_pack_local[(31)] = (data_pack_local[(31)] + d[(34)]);
  data_pack_local[(32)] = 0.000000e+00f;
  data_pack_local[(32)] = __ocml_fma_f32(d[(7)], -1.000000e+00f, data_pack_local[(32)]);
  data_pack_local[(32)] = __ocml_fma_f32(d[(8)], 5.000000e-01f, data_pack_local[(32)]);
  data_pack_local[(32)] = __ocml_fma_f32(d[(9)], 2.500000e+00f, data_pack_local[(32)]);
  data_pack_local[(32)] = (data_pack_local[(32)] + d[(10)]);
  data_pack_local[(32)] = __ocml_fma_f32((d[(13)] * -1.500000e+00f), -1.000000e+00f, data_pack_local[(32)]);
  data_pack_local[(32)] = __ocml_fma_f32((d[(14)] * -1.500000e+00f), 5.000000e-01f, data_pack_local[(32)]);
  data_pack_local[(32)] = __ocml_fma_f32((d[(15)] * -1.500000e+00f), 2.500000e+00f, data_pack_local[(32)]);
  data_pack_local[(32)] = __ocml_fma_f32(d[(16)], -1.500000e+00f, data_pack_local[(32)]);
  data_pack_local[(32)] = __ocml_fma_f32((d[(19)] * -2.000000e+00f), -1.000000e+00f, data_pack_local[(32)]);
  data_pack_local[(32)] = __ocml_fma_f32((d[(20)] * -2.000000e+00f), 5.000000e-01f, data_pack_local[(32)]);
  data_pack_local[(32)] = __ocml_fma_f32((d[(21)] * -2.000000e+00f), 2.500000e+00f, data_pack_local[(32)]);
  data_pack_local[(32)] = __ocml_fma_f32(d[(22)], -2.000000e+00f, data_pack_local[(32)]);
  data_pack_local[(32)] = __ocml_fma_f32((d[(25)] * 1.500000e+00f), -1.000000e+00f, data_pack_local[(32)]);
  data_pack_local[(32)] = __ocml_fma_f32((d[(26)] * 1.500000e+00f), 5.000000e-01f, data_pack_local[(32)]);
  data_pack_local[(32)] = __ocml_fma_f32((d[(27)] * 1.500000e+00f), 2.500000e+00f, data_pack_local[(32)]);
  data_pack_local[(32)] = __ocml_fma_f32(d[(28)], 1.500000e+00f, data_pack_local[(32)]);
  data_pack_local[(32)] = __ocml_fma_f32(d[(31)], -1.000000e+00f, data_pack_local[(32)]);
  data_pack_local[(32)] = __ocml_fma_f32(d[(32)], 5.000000e-01f, data_pack_local[(32)]);
  data_pack_local[(32)] = __ocml_fma_f32(d[(33)], 2.500000e+00f, data_pack_local[(32)]);
  data_pack_local[(32)] = (data_pack_local[(32)] + d[(34)]);
  data_pack_local[(33)] = 0.000000e+00f;
  data_pack_local[(33)] = __ocml_fma_f32(d[(7)], -2.000000e+00f, data_pack_local[(33)]);
  data_pack_local[(33)] = __ocml_fma_f32(d[(8)], -1.000000e+00f, data_pack_local[(33)]);
  data_pack_local[(33)] = __ocml_fma_f32(d[(9)], 2.000000e+00f, data_pack_local[(33)]);
  data_pack_local[(33)] = (data_pack_local[(33)] + d[(10)]);
  data_pack_local[(33)] = __ocml_fma_f32((d[(13)] * -1.500000e+00f), -2.000000e+00f, data_pack_local[(33)]);
  data_pack_local[(33)] = __ocml_fma_f32((d[(14)] * -1.500000e+00f), -1.000000e+00f, data_pack_local[(33)]);
  data_pack_local[(33)] = __ocml_fma_f32((d[(15)] * -1.500000e+00f), 2.000000e+00f, data_pack_local[(33)]);
  data_pack_local[(33)] = __ocml_fma_f32(d[(16)], -1.500000e+00f, data_pack_local[(33)]);
  data_pack_local[(33)] = __ocml_fma_f32((d[(19)] * -2.000000e+00f), -2.000000e+00f, data_pack_local[(33)]);
  data_pack_local[(33)] = __ocml_fma_f32((d[(20)] * -2.000000e+00f), -1.000000e+00f, data_pack_local[(33)]);
  data_pack_local[(33)] = __ocml_fma_f32((d[(21)] * -2.000000e+00f), 2.000000e+00f, data_pack_local[(33)]);
  data_pack_local[(33)] = __ocml_fma_f32(d[(22)], -2.000000e+00f, data_pack_local[(33)]);
  data_pack_local[(33)] = __ocml_fma_f32((d[(25)] * 1.500000e+00f), -2.000000e+00f, data_pack_local[(33)]);
  data_pack_local[(33)] = __ocml_fma_f32((d[(26)] * 1.500000e+00f), -1.000000e+00f, data_pack_local[(33)]);
  data_pack_local[(33)] = __ocml_fma_f32((d[(27)] * 1.500000e+00f), 2.000000e+00f, data_pack_local[(33)]);
  data_pack_local[(33)] = __ocml_fma_f32(d[(28)], 1.500000e+00f, data_pack_local[(33)]);
  data_pack_local[(33)] = __ocml_fma_f32(d[(31)], -2.000000e+00f, data_pack_local[(33)]);
  data_pack_local[(33)] = __ocml_fma_f32(d[(32)], -1.000000e+00f, data_pack_local[(33)]);
  data_pack_local[(33)] = __ocml_fma_f32(d[(33)], 2.000000e+00f, data_pack_local[(33)]);
  data_pack_local[(33)] = (data_pack_local[(33)] + d[(34)]);
  data_pack_local[(34)] = 0.000000e+00f;
  data_pack_local[(34)] = __ocml_fma_f32(d[(7)], 5.000000e-01f, data_pack_local[(34)]);
  data_pack_local[(34)] = __ocml_fma_f32(d[(8)], -1.000000e+00f, data_pack_local[(34)]);
  data_pack_local[(34)] = __ocml_fma_f32(d[(9)], -5.000000e-01f, data_pack_local[(34)]);
  data_pack_local[(34)] = (data_pack_local[(34)] + d[(10)]);
  data_pack_local[(34)] = __ocml_fma_f32((d[(13)] * -1.500000e+00f), 5.000000e-01f, data_pack_local[(34)]);
  data_pack_local[(34)] = __ocml_fma_f32((d[(14)] * -1.500000e+00f), -1.000000e+00f, data_pack_local[(34)]);
  data_pack_local[(34)] = __ocml_fma_f32((d[(15)] * -1.500000e+00f), -5.000000e-01f, data_pack_local[(34)]);
  data_pack_local[(34)] = __ocml_fma_f32(d[(16)], -1.500000e+00f, data_pack_local[(34)]);
  data_pack_local[(34)] = __ocml_fma_f32((d[(19)] * -2.000000e+00f), 5.000000e-01f, data_pack_local[(34)]);
  data_pack_local[(34)] = __ocml_fma_f32((d[(20)] * -2.000000e+00f), -1.000000e+00f, data_pack_local[(34)]);
  data_pack_local[(34)] = __ocml_fma_f32((d[(21)] * -2.000000e+00f), -5.000000e-01f, data_pack_local[(34)]);
  data_pack_local[(34)] = __ocml_fma_f32(d[(22)], -2.000000e+00f, data_pack_local[(34)]);
  data_pack_local[(34)] = __ocml_fma_f32((d[(25)] * 1.500000e+00f), 5.000000e-01f, data_pack_local[(34)]);
  data_pack_local[(34)] = __ocml_fma_f32((d[(26)] * 1.500000e+00f), -1.000000e+00f, data_pack_local[(34)]);
  data_pack_local[(34)] = __ocml_fma_f32((d[(27)] * 1.500000e+00f), -5.000000e-01f, data_pack_local[(34)]);
  data_pack_local[(34)] = __ocml_fma_f32(d[(28)], 1.500000e+00f, data_pack_local[(34)]);
  data_pack_local[(34)] = __ocml_fma_f32(d[(31)], 5.000000e-01f, data_pack_local[(34)]);
  data_pack_local[(34)] = __ocml_fma_f32(d[(32)], -1.000000e+00f, data_pack_local[(34)]);
  data_pack_local[(34)] = __ocml_fma_f32(d[(33)], -5.000000e-01f, data_pack_local[(34)]);
  data_pack_local[(34)] = (data_pack_local[(34)] + d[(34)]);
  data_pack_local[(35)] = 0.000000e+00f;
  data_pack_local[(35)] = (data_pack_local[(35)] + d[(7)]);
  data_pack_local[(35)] = __ocml_fma_f32(d[(8)], -1.500000e+00f, data_pack_local[(35)]);
  data_pack_local[(35)] = __ocml_fma_f32(d[(9)], -2.000000e+00f, data_pack_local[(35)]);
  data_pack_local[(35)] = __ocml_fma_f32(d[(10)], 1.500000e+00f, data_pack_local[(35)]);
  data_pack_local[(35)] = (data_pack_local[(35)] + d[(11)]);
  data_pack_local[(35)] = __ocml_fma_f32(d[(13)], -1.500000e+00f, data_pack_local[(35)]);
  data_pack_local[(35)] = __ocml_fma_f32((d[(14)] * -1.500000e+00f), -1.500000e+00f, data_pack_local[(35)]);
  data_pack_local[(35)] = __ocml_fma_f32((d[(15)] * -1.500000e+00f), -2.000000e+00f, data_pack_local[(35)]);
  data_pack_local[(35)] = __ocml_fma_f32((d[(16)] * -1.500000e+00f), 1.500000e+00f, data_pack_local[(35)]);
  data_pack_local[(35)] = __ocml_fma_f32(d[(17)], -1.500000e+00f, data_pack_local[(35)]);
  data_pack_local[(35)] = __ocml_fma_f32(d[(19)], -2.000000e+00f, data_pack_local[(35)]);
  data_pack_local[(35)] = __ocml_fma_f32((d[(20)] * -2.000000e+00f), -1.500000e+00f, data_pack_local[(35)]);
  data_pack_local[(35)] = __ocml_fma_f32((d[(21)] * -2.000000e+00f), -2.000000e+00f, data_pack_local[(35)]);
  data_pack_local[(35)] = __ocml_fma_f32((d[(22)] * -2.000000e+00f), 1.500000e+00f, data_pack_local[(35)]);
  data_pack_local[(35)] = __ocml_fma_f32(d[(23)], -2.000000e+00f, data_pack_local[(35)]);
  data_pack_local[(35)] = __ocml_fma_f32(d[(25)], 1.500000e+00f, data_pack_local[(35)]);
  data_pack_local[(35)] = __ocml_fma_f32((d[(26)] * 1.500000e+00f), -1.500000e+00f, data_pack_local[(35)]);
  data_pack_local[(35)] = __ocml_fma_f32((d[(27)] * 1.500000e+00f), -2.000000e+00f, data_pack_local[(35)]);
  data_pack_local[(35)] = __ocml_fma_f32((d[(28)] * 1.500000e+00f), 1.500000e+00f, data_pack_local[(35)]);
  data_pack_local[(35)] = __ocml_fma_f32(d[(29)], 1.500000e+00f, data_pack_local[(35)]);
  data_pack_local[(35)] = (data_pack_local[(35)] + d[(31)]);
  data_pack_local[(35)] = __ocml_fma_f32(d[(32)], -1.500000e+00f, data_pack_local[(35)]);
  data_pack_local[(35)] = __ocml_fma_f32(d[(33)], -2.000000e+00f, data_pack_local[(35)]);
  data_pack_local[(35)] = __ocml_fma_f32(d[(34)], 1.500000e+00f, data_pack_local[(35)]);
  data_pack_local[(35)] = (data_pack_local[(35)] + d[(35)]);
  for (int eps1 = 0; eps1 < 6; ++eps1) {
    for (int nu1 = 0; nu1 < 6; ++nu1) {
      data_pack[(((((eps1 * 75264) + (nu1 * 12544)) + (((int)blockIdx.x) * 128)) + ((int)threadIdx.x)))] = data_pack_local[(((eps1 * 6) + nu1))];
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_add_add_nn_relu_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2, float* __restrict__ placeholder3) {
  float compute[4];
  __shared__ float pad_temp_shared[1568];
  __shared__ float placeholder_shared[512];
  compute[(0)] = 0.000000e+00f;
  compute[(2)] = 0.000000e+00f;
  compute[(1)] = 0.000000e+00f;
  compute[(3)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 16; ++rc_outer) {
    __syncthreads();
    pad_temp_shared[((((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)))] = placeholder[((((((rc_outer * 3136) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)))];
    pad_temp_shared[(((((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)) + 1))] = placeholder[(((((((rc_outer * 3136) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)) + 1))];
    if (((((int)threadIdx.z) * 2) + (((((int)threadIdx.y) * 5) + ((int)threadIdx.x)) >> 4)) < 32) {
      if ((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 5)) + ((int)threadIdx.x)) < 512) {
        if (((((int)threadIdx.y) * 5) + ((int)threadIdx.x)) < 32) {
          if (((int)threadIdx.x) < 5) {
            placeholder_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 5)) + ((int)threadIdx.x)))] = placeholder1[((((((((int)blockIdx.z) * 8192) + (((int)threadIdx.z) * 512)) + ((((((int)threadIdx.y) * 5) + ((int)threadIdx.x)) >> 4) * 256)) + (rc_outer * 16)) + (((((int)threadIdx.y) * 5) + ((int)threadIdx.x)) & 15)))];
          }
        }
      }
    }
    __syncthreads();
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)))], placeholder_shared[((((int)threadIdx.z) * 16))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)))], placeholder_shared[(((((int)threadIdx.z) * 16) + 256))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 1))], placeholder_shared[((((int)threadIdx.z) * 16))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 1))], placeholder_shared[(((((int)threadIdx.z) * 16) + 256))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 98))], placeholder_shared[(((((int)threadIdx.z) * 16) + 1))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 98))], placeholder_shared[(((((int)threadIdx.z) * 16) + 257))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 99))], placeholder_shared[(((((int)threadIdx.z) * 16) + 1))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 99))], placeholder_shared[(((((int)threadIdx.z) * 16) + 257))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 196))], placeholder_shared[(((((int)threadIdx.z) * 16) + 2))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 196))], placeholder_shared[(((((int)threadIdx.z) * 16) + 258))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 197))], placeholder_shared[(((((int)threadIdx.z) * 16) + 2))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 197))], placeholder_shared[(((((int)threadIdx.z) * 16) + 258))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 294))], placeholder_shared[(((((int)threadIdx.z) * 16) + 3))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 294))], placeholder_shared[(((((int)threadIdx.z) * 16) + 259))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 295))], placeholder_shared[(((((int)threadIdx.z) * 16) + 3))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 295))], placeholder_shared[(((((int)threadIdx.z) * 16) + 259))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 392))], placeholder_shared[(((((int)threadIdx.z) * 16) + 4))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 392))], placeholder_shared[(((((int)threadIdx.z) * 16) + 260))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 393))], placeholder_shared[(((((int)threadIdx.z) * 16) + 4))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 393))], placeholder_shared[(((((int)threadIdx.z) * 16) + 260))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 490))], placeholder_shared[(((((int)threadIdx.z) * 16) + 5))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 490))], placeholder_shared[(((((int)threadIdx.z) * 16) + 261))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 491))], placeholder_shared[(((((int)threadIdx.z) * 16) + 5))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 491))], placeholder_shared[(((((int)threadIdx.z) * 16) + 261))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 588))], placeholder_shared[(((((int)threadIdx.z) * 16) + 6))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 588))], placeholder_shared[(((((int)threadIdx.z) * 16) + 262))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 589))], placeholder_shared[(((((int)threadIdx.z) * 16) + 6))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 589))], placeholder_shared[(((((int)threadIdx.z) * 16) + 262))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 686))], placeholder_shared[(((((int)threadIdx.z) * 16) + 7))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 686))], placeholder_shared[(((((int)threadIdx.z) * 16) + 263))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 687))], placeholder_shared[(((((int)threadIdx.z) * 16) + 7))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 687))], placeholder_shared[(((((int)threadIdx.z) * 16) + 263))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 784))], placeholder_shared[(((((int)threadIdx.z) * 16) + 8))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 784))], placeholder_shared[(((((int)threadIdx.z) * 16) + 264))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 785))], placeholder_shared[(((((int)threadIdx.z) * 16) + 8))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 785))], placeholder_shared[(((((int)threadIdx.z) * 16) + 264))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 882))], placeholder_shared[(((((int)threadIdx.z) * 16) + 9))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 882))], placeholder_shared[(((((int)threadIdx.z) * 16) + 265))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 883))], placeholder_shared[(((((int)threadIdx.z) * 16) + 9))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 883))], placeholder_shared[(((((int)threadIdx.z) * 16) + 265))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 980))], placeholder_shared[(((((int)threadIdx.z) * 16) + 10))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 980))], placeholder_shared[(((((int)threadIdx.z) * 16) + 266))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 981))], placeholder_shared[(((((int)threadIdx.z) * 16) + 10))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 981))], placeholder_shared[(((((int)threadIdx.z) * 16) + 266))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 1078))], placeholder_shared[(((((int)threadIdx.z) * 16) + 11))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 1078))], placeholder_shared[(((((int)threadIdx.z) * 16) + 267))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 1079))], placeholder_shared[(((((int)threadIdx.z) * 16) + 11))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 1079))], placeholder_shared[(((((int)threadIdx.z) * 16) + 267))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 1176))], placeholder_shared[(((((int)threadIdx.z) * 16) + 12))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 1176))], placeholder_shared[(((((int)threadIdx.z) * 16) + 268))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 1177))], placeholder_shared[(((((int)threadIdx.z) * 16) + 12))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 1177))], placeholder_shared[(((((int)threadIdx.z) * 16) + 268))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 1274))], placeholder_shared[(((((int)threadIdx.z) * 16) + 13))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 1274))], placeholder_shared[(((((int)threadIdx.z) * 16) + 269))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 1275))], placeholder_shared[(((((int)threadIdx.z) * 16) + 13))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 1275))], placeholder_shared[(((((int)threadIdx.z) * 16) + 269))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 1372))], placeholder_shared[(((((int)threadIdx.z) * 16) + 14))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 1372))], placeholder_shared[(((((int)threadIdx.z) * 16) + 270))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 1373))], placeholder_shared[(((((int)threadIdx.z) * 16) + 14))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 1373))], placeholder_shared[(((((int)threadIdx.z) * 16) + 270))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 1470))], placeholder_shared[(((((int)threadIdx.z) * 16) + 15))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 1470))], placeholder_shared[(((((int)threadIdx.z) * 16) + 271))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 1471))], placeholder_shared[(((((int)threadIdx.z) * 16) + 15))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 1471))], placeholder_shared[(((((int)threadIdx.z) * 16) + 271))], compute[(3)]);
  }
  T_relu[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)))] = max(((compute[(0)] + placeholder2[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)))]) + placeholder3[(((((int)blockIdx.z) * 32) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[(((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)) + 3136))] = max(((compute[(2)] + placeholder2[(((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)) + 3136))]) + placeholder3[((((((int)blockIdx.z) * 32) + ((int)threadIdx.z)) + 16))]), 0.000000e+00f);
  T_relu[(((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)) + 1))] = max(((compute[(1)] + placeholder2[(((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)) + 1))]) + placeholder3[(((((int)blockIdx.z) * 32) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[(((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)) + 3137))] = max(((compute[(3)] + placeholder2[(((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)) + 3137))]) + placeholder3[((((((int)blockIdx.z) * 32) + ((int)threadIdx.z)) + 16))]), 0.000000e+00f);
}

extern "C" __global__ void fused_nn_dense_add_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_add, float* __restrict__ placeholder2) {
  float T_dense_rf[1];
  float red_buf0[1];
  __shared__ float T_dense[1];
  T_dense_rf[(0)] = 0.000000e+00f;
  for (int k_outer = 0; k_outer < 32; ++k_outer) {
    T_dense_rf[(0)] = __ocml_fma_f32(placeholder[(((k_outer * 64) + ((int)threadIdx.x)))], placeholder1[((((((int)blockIdx.x) * 2048) + (k_outer * 64)) + ((int)threadIdx.x)))], T_dense_rf[(0)]);
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

extern "C" __global__ void fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_3_kernel1(float* __restrict__ placeholder, float* __restrict__ data_pack, float* __restrict__ bgemm) {
  float bgemm_local[4];
  __shared__ float placeholder_shared[2048];
  __shared__ float data_pack_shared[896];
  bgemm_local[(0)] = 0.000000e+00f;
  bgemm_local[(1)] = 0.000000e+00f;
  bgemm_local[(2)] = 0.000000e+00f;
  bgemm_local[(3)] = 0.000000e+00f;
  for (int ci_outer = 0; ci_outer < 2; ++ci_outer) {
    __syncthreads();
    placeholder_shared[(((((int)threadIdx.y) * 14) + ((int)threadIdx.x)))] = placeholder[((((((int)blockIdx.z) * 4096) + (ci_outer * 2048)) + ((((int)threadIdx.y) * 14) + ((int)threadIdx.x))))];
    placeholder_shared[((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) + 448))] = placeholder[(((((((int)blockIdx.z) * 4096) + (ci_outer * 2048)) + ((((int)threadIdx.y) * 14) + ((int)threadIdx.x))) + 448))];
    placeholder_shared[((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) + 896))] = placeholder[(((((((int)blockIdx.z) * 4096) + (ci_outer * 2048)) + ((((int)threadIdx.y) * 14) + ((int)threadIdx.x))) + 896))];
    placeholder_shared[((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) + 1344))] = placeholder[(((((((int)blockIdx.z) * 4096) + (ci_outer * 2048)) + ((((int)threadIdx.y) * 14) + ((int)threadIdx.x))) + 1344))];
    if (((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) < 256) {
      if (((int)threadIdx.y) < 19) {
        placeholder_shared[((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) + 1792))] = placeholder[(((((((int)blockIdx.z) * 4096) + (ci_outer * 2048)) + ((((int)threadIdx.y) * 14) + ((int)threadIdx.x))) + 1792))];
      }
    }
    data_pack_shared[(((((int)threadIdx.y) * 14) + ((int)threadIdx.x)))] = data_pack[((((((((int)blockIdx.z) * 12544) + (ci_outer * 6272)) + ((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) / 28) * 196)) + (((int)blockIdx.x) * 28)) + (((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) % 28)))];
    data_pack_shared[((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) + 448))] = data_pack[(((((((((int)blockIdx.z) * 12544) + (ci_outer * 6272)) + ((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) / 28) * 196)) + (((int)blockIdx.x) * 28)) + (((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) % 28)) + 3136))];
    __syncthreads();
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[((((int)threadIdx.y) * 2))], data_pack_shared[((((int)threadIdx.x) * 2))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[((((int)threadIdx.y) * 2))], data_pack_shared[(((((int)threadIdx.x) * 2) + 1))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 1))], data_pack_shared[((((int)threadIdx.x) * 2))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 1))], data_pack_shared[(((((int)threadIdx.x) * 2) + 1))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 64))], data_pack_shared[(((((int)threadIdx.x) * 2) + 28))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 64))], data_pack_shared[(((((int)threadIdx.x) * 2) + 29))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 65))], data_pack_shared[(((((int)threadIdx.x) * 2) + 28))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 65))], data_pack_shared[(((((int)threadIdx.x) * 2) + 29))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 128))], data_pack_shared[(((((int)threadIdx.x) * 2) + 56))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 128))], data_pack_shared[(((((int)threadIdx.x) * 2) + 57))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 129))], data_pack_shared[(((((int)threadIdx.x) * 2) + 56))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 129))], data_pack_shared[(((((int)threadIdx.x) * 2) + 57))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 192))], data_pack_shared[(((((int)threadIdx.x) * 2) + 84))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 192))], data_pack_shared[(((((int)threadIdx.x) * 2) + 85))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 193))], data_pack_shared[(((((int)threadIdx.x) * 2) + 84))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 193))], data_pack_shared[(((((int)threadIdx.x) * 2) + 85))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 256))], data_pack_shared[(((((int)threadIdx.x) * 2) + 112))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 256))], data_pack_shared[(((((int)threadIdx.x) * 2) + 113))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 257))], data_pack_shared[(((((int)threadIdx.x) * 2) + 112))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 257))], data_pack_shared[(((((int)threadIdx.x) * 2) + 113))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 320))], data_pack_shared[(((((int)threadIdx.x) * 2) + 140))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 320))], data_pack_shared[(((((int)threadIdx.x) * 2) + 141))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 321))], data_pack_shared[(((((int)threadIdx.x) * 2) + 140))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 321))], data_pack_shared[(((((int)threadIdx.x) * 2) + 141))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 384))], data_pack_shared[(((((int)threadIdx.x) * 2) + 168))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 384))], data_pack_shared[(((((int)threadIdx.x) * 2) + 169))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 385))], data_pack_shared[(((((int)threadIdx.x) * 2) + 168))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 385))], data_pack_shared[(((((int)threadIdx.x) * 2) + 169))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 448))], data_pack_shared[(((((int)threadIdx.x) * 2) + 196))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 448))], data_pack_shared[(((((int)threadIdx.x) * 2) + 197))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 449))], data_pack_shared[(((((int)threadIdx.x) * 2) + 196))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 449))], data_pack_shared[(((((int)threadIdx.x) * 2) + 197))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 512))], data_pack_shared[(((((int)threadIdx.x) * 2) + 224))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 512))], data_pack_shared[(((((int)threadIdx.x) * 2) + 225))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 513))], data_pack_shared[(((((int)threadIdx.x) * 2) + 224))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 513))], data_pack_shared[(((((int)threadIdx.x) * 2) + 225))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 576))], data_pack_shared[(((((int)threadIdx.x) * 2) + 252))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 576))], data_pack_shared[(((((int)threadIdx.x) * 2) + 253))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 577))], data_pack_shared[(((((int)threadIdx.x) * 2) + 252))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 577))], data_pack_shared[(((((int)threadIdx.x) * 2) + 253))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 640))], data_pack_shared[(((((int)threadIdx.x) * 2) + 280))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 640))], data_pack_shared[(((((int)threadIdx.x) * 2) + 281))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 641))], data_pack_shared[(((((int)threadIdx.x) * 2) + 280))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 641))], data_pack_shared[(((((int)threadIdx.x) * 2) + 281))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 704))], data_pack_shared[(((((int)threadIdx.x) * 2) + 308))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 704))], data_pack_shared[(((((int)threadIdx.x) * 2) + 309))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 705))], data_pack_shared[(((((int)threadIdx.x) * 2) + 308))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 705))], data_pack_shared[(((((int)threadIdx.x) * 2) + 309))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 768))], data_pack_shared[(((((int)threadIdx.x) * 2) + 336))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 768))], data_pack_shared[(((((int)threadIdx.x) * 2) + 337))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 769))], data_pack_shared[(((((int)threadIdx.x) * 2) + 336))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 769))], data_pack_shared[(((((int)threadIdx.x) * 2) + 337))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 832))], data_pack_shared[(((((int)threadIdx.x) * 2) + 364))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 832))], data_pack_shared[(((((int)threadIdx.x) * 2) + 365))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 833))], data_pack_shared[(((((int)threadIdx.x) * 2) + 364))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 833))], data_pack_shared[(((((int)threadIdx.x) * 2) + 365))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 896))], data_pack_shared[(((((int)threadIdx.x) * 2) + 392))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 896))], data_pack_shared[(((((int)threadIdx.x) * 2) + 393))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 897))], data_pack_shared[(((((int)threadIdx.x) * 2) + 392))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 897))], data_pack_shared[(((((int)threadIdx.x) * 2) + 393))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 960))], data_pack_shared[(((((int)threadIdx.x) * 2) + 420))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 960))], data_pack_shared[(((((int)threadIdx.x) * 2) + 421))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 961))], data_pack_shared[(((((int)threadIdx.x) * 2) + 420))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 961))], data_pack_shared[(((((int)threadIdx.x) * 2) + 421))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 1024))], data_pack_shared[(((((int)threadIdx.x) * 2) + 448))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 1024))], data_pack_shared[(((((int)threadIdx.x) * 2) + 449))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 1025))], data_pack_shared[(((((int)threadIdx.x) * 2) + 448))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 1025))], data_pack_shared[(((((int)threadIdx.x) * 2) + 449))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 1088))], data_pack_shared[(((((int)threadIdx.x) * 2) + 476))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 1088))], data_pack_shared[(((((int)threadIdx.x) * 2) + 477))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 1089))], data_pack_shared[(((((int)threadIdx.x) * 2) + 476))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 1089))], data_pack_shared[(((((int)threadIdx.x) * 2) + 477))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 1152))], data_pack_shared[(((((int)threadIdx.x) * 2) + 504))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 1152))], data_pack_shared[(((((int)threadIdx.x) * 2) + 505))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 1153))], data_pack_shared[(((((int)threadIdx.x) * 2) + 504))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 1153))], data_pack_shared[(((((int)threadIdx.x) * 2) + 505))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 1216))], data_pack_shared[(((((int)threadIdx.x) * 2) + 532))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 1216))], data_pack_shared[(((((int)threadIdx.x) * 2) + 533))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 1217))], data_pack_shared[(((((int)threadIdx.x) * 2) + 532))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 1217))], data_pack_shared[(((((int)threadIdx.x) * 2) + 533))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 1280))], data_pack_shared[(((((int)threadIdx.x) * 2) + 560))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 1280))], data_pack_shared[(((((int)threadIdx.x) * 2) + 561))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 1281))], data_pack_shared[(((((int)threadIdx.x) * 2) + 560))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 1281))], data_pack_shared[(((((int)threadIdx.x) * 2) + 561))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 1344))], data_pack_shared[(((((int)threadIdx.x) * 2) + 588))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 1344))], data_pack_shared[(((((int)threadIdx.x) * 2) + 589))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 1345))], data_pack_shared[(((((int)threadIdx.x) * 2) + 588))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 1345))], data_pack_shared[(((((int)threadIdx.x) * 2) + 589))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 1408))], data_pack_shared[(((((int)threadIdx.x) * 2) + 616))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 1408))], data_pack_shared[(((((int)threadIdx.x) * 2) + 617))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 1409))], data_pack_shared[(((((int)threadIdx.x) * 2) + 616))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 1409))], data_pack_shared[(((((int)threadIdx.x) * 2) + 617))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 1472))], data_pack_shared[(((((int)threadIdx.x) * 2) + 644))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 1472))], data_pack_shared[(((((int)threadIdx.x) * 2) + 645))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 1473))], data_pack_shared[(((((int)threadIdx.x) * 2) + 644))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 1473))], data_pack_shared[(((((int)threadIdx.x) * 2) + 645))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 1536))], data_pack_shared[(((((int)threadIdx.x) * 2) + 672))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 1536))], data_pack_shared[(((((int)threadIdx.x) * 2) + 673))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 1537))], data_pack_shared[(((((int)threadIdx.x) * 2) + 672))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 1537))], data_pack_shared[(((((int)threadIdx.x) * 2) + 673))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 1600))], data_pack_shared[(((((int)threadIdx.x) * 2) + 700))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 1600))], data_pack_shared[(((((int)threadIdx.x) * 2) + 701))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 1601))], data_pack_shared[(((((int)threadIdx.x) * 2) + 700))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 1601))], data_pack_shared[(((((int)threadIdx.x) * 2) + 701))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 1664))], data_pack_shared[(((((int)threadIdx.x) * 2) + 728))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 1664))], data_pack_shared[(((((int)threadIdx.x) * 2) + 729))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 1665))], data_pack_shared[(((((int)threadIdx.x) * 2) + 728))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 1665))], data_pack_shared[(((((int)threadIdx.x) * 2) + 729))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 1728))], data_pack_shared[(((((int)threadIdx.x) * 2) + 756))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 1728))], data_pack_shared[(((((int)threadIdx.x) * 2) + 757))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 1729))], data_pack_shared[(((((int)threadIdx.x) * 2) + 756))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 1729))], data_pack_shared[(((((int)threadIdx.x) * 2) + 757))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 1792))], data_pack_shared[(((((int)threadIdx.x) * 2) + 784))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 1792))], data_pack_shared[(((((int)threadIdx.x) * 2) + 785))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 1793))], data_pack_shared[(((((int)threadIdx.x) * 2) + 784))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 1793))], data_pack_shared[(((((int)threadIdx.x) * 2) + 785))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 1856))], data_pack_shared[(((((int)threadIdx.x) * 2) + 812))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 1856))], data_pack_shared[(((((int)threadIdx.x) * 2) + 813))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 1857))], data_pack_shared[(((((int)threadIdx.x) * 2) + 812))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 1857))], data_pack_shared[(((((int)threadIdx.x) * 2) + 813))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 1920))], data_pack_shared[(((((int)threadIdx.x) * 2) + 840))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 1920))], data_pack_shared[(((((int)threadIdx.x) * 2) + 841))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 1921))], data_pack_shared[(((((int)threadIdx.x) * 2) + 840))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 1921))], data_pack_shared[(((((int)threadIdx.x) * 2) + 841))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 1984))], data_pack_shared[(((((int)threadIdx.x) * 2) + 868))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 1984))], data_pack_shared[(((((int)threadIdx.x) * 2) + 869))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 1985))], data_pack_shared[(((((int)threadIdx.x) * 2) + 868))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 1985))], data_pack_shared[(((((int)threadIdx.x) * 2) + 869))], bgemm_local[(3)]);
  }
  bgemm[(((((((int)blockIdx.z) * 12544) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 28)) + (((int)threadIdx.x) * 2)))] = bgemm_local[(0)];
  bgemm[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 28)) + (((int)threadIdx.x) * 2)) + 1))] = bgemm_local[(1)];
  bgemm[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 28)) + (((int)threadIdx.x) * 2)) + 196))] = bgemm_local[(2)];
  bgemm[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 28)) + (((int)threadIdx.x) * 2)) + 197))] = bgemm_local[(3)];
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_6_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  float compute[4];
  __shared__ float pad_temp_shared[256];
  __shared__ float placeholder_shared[512];
  compute[(0)] = 0.000000e+00f;
  compute[(2)] = 0.000000e+00f;
  compute[(1)] = 0.000000e+00f;
  compute[(3)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 16; ++rc_outer) {
    __syncthreads();
    pad_temp_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)))] = placeholder[(((((((rc_outer * 50176) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)))];
    pad_temp_shared[(((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1))] = placeholder[((((((((rc_outer * 50176) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 1))];
    placeholder_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))] = placeholder1[((((((((int)blockIdx.z) * 8192) + (((int)threadIdx.z) * 512)) + (((int)threadIdx.y) * 256)) + (rc_outer * 16)) + (((int)threadIdx.x) * 4)))];
    placeholder_shared[(((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 1))] = placeholder1[(((((((((int)blockIdx.z) * 8192) + (((int)threadIdx.z) * 512)) + (((int)threadIdx.y) * 256)) + (rc_outer * 16)) + (((int)threadIdx.x) * 4)) + 1))];
    placeholder_shared[(((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 2))] = placeholder1[(((((((((int)blockIdx.z) * 8192) + (((int)threadIdx.z) * 512)) + (((int)threadIdx.y) * 256)) + (rc_outer * 16)) + (((int)threadIdx.x) * 4)) + 2))];
    placeholder_shared[(((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 3))] = placeholder1[(((((((((int)blockIdx.z) * 8192) + (((int)threadIdx.z) * 512)) + (((int)threadIdx.y) * 256)) + (rc_outer * 16)) + (((int)threadIdx.x) * 4)) + 3))];
    __syncthreads();
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)))], placeholder_shared[((((int)threadIdx.z) * 16))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)))], placeholder_shared[(((((int)threadIdx.z) * 16) + 256))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 1))], placeholder_shared[((((int)threadIdx.z) * 16))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 1))], placeholder_shared[(((((int)threadIdx.z) * 16) + 256))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 16))], placeholder_shared[(((((int)threadIdx.z) * 16) + 1))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 16))], placeholder_shared[(((((int)threadIdx.z) * 16) + 257))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 17))], placeholder_shared[(((((int)threadIdx.z) * 16) + 1))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 17))], placeholder_shared[(((((int)threadIdx.z) * 16) + 257))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 32))], placeholder_shared[(((((int)threadIdx.z) * 16) + 2))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 32))], placeholder_shared[(((((int)threadIdx.z) * 16) + 258))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 33))], placeholder_shared[(((((int)threadIdx.z) * 16) + 2))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 33))], placeholder_shared[(((((int)threadIdx.z) * 16) + 258))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 48))], placeholder_shared[(((((int)threadIdx.z) * 16) + 3))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 48))], placeholder_shared[(((((int)threadIdx.z) * 16) + 259))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 49))], placeholder_shared[(((((int)threadIdx.z) * 16) + 3))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 49))], placeholder_shared[(((((int)threadIdx.z) * 16) + 259))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 64))], placeholder_shared[(((((int)threadIdx.z) * 16) + 4))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 64))], placeholder_shared[(((((int)threadIdx.z) * 16) + 260))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 65))], placeholder_shared[(((((int)threadIdx.z) * 16) + 4))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 65))], placeholder_shared[(((((int)threadIdx.z) * 16) + 260))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 80))], placeholder_shared[(((((int)threadIdx.z) * 16) + 5))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 80))], placeholder_shared[(((((int)threadIdx.z) * 16) + 261))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 81))], placeholder_shared[(((((int)threadIdx.z) * 16) + 5))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 81))], placeholder_shared[(((((int)threadIdx.z) * 16) + 261))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 96))], placeholder_shared[(((((int)threadIdx.z) * 16) + 6))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 96))], placeholder_shared[(((((int)threadIdx.z) * 16) + 262))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 97))], placeholder_shared[(((((int)threadIdx.z) * 16) + 6))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 97))], placeholder_shared[(((((int)threadIdx.z) * 16) + 262))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 112))], placeholder_shared[(((((int)threadIdx.z) * 16) + 7))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 112))], placeholder_shared[(((((int)threadIdx.z) * 16) + 263))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 113))], placeholder_shared[(((((int)threadIdx.z) * 16) + 7))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 113))], placeholder_shared[(((((int)threadIdx.z) * 16) + 263))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 128))], placeholder_shared[(((((int)threadIdx.z) * 16) + 8))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 128))], placeholder_shared[(((((int)threadIdx.z) * 16) + 264))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 129))], placeholder_shared[(((((int)threadIdx.z) * 16) + 8))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 129))], placeholder_shared[(((((int)threadIdx.z) * 16) + 264))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 144))], placeholder_shared[(((((int)threadIdx.z) * 16) + 9))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 144))], placeholder_shared[(((((int)threadIdx.z) * 16) + 265))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 145))], placeholder_shared[(((((int)threadIdx.z) * 16) + 9))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 145))], placeholder_shared[(((((int)threadIdx.z) * 16) + 265))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 160))], placeholder_shared[(((((int)threadIdx.z) * 16) + 10))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 160))], placeholder_shared[(((((int)threadIdx.z) * 16) + 266))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 161))], placeholder_shared[(((((int)threadIdx.z) * 16) + 10))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 161))], placeholder_shared[(((((int)threadIdx.z) * 16) + 266))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 176))], placeholder_shared[(((((int)threadIdx.z) * 16) + 11))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 176))], placeholder_shared[(((((int)threadIdx.z) * 16) + 267))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 177))], placeholder_shared[(((((int)threadIdx.z) * 16) + 11))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 177))], placeholder_shared[(((((int)threadIdx.z) * 16) + 267))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 192))], placeholder_shared[(((((int)threadIdx.z) * 16) + 12))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 192))], placeholder_shared[(((((int)threadIdx.z) * 16) + 268))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 193))], placeholder_shared[(((((int)threadIdx.z) * 16) + 12))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 193))], placeholder_shared[(((((int)threadIdx.z) * 16) + 268))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 208))], placeholder_shared[(((((int)threadIdx.z) * 16) + 13))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 208))], placeholder_shared[(((((int)threadIdx.z) * 16) + 269))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 209))], placeholder_shared[(((((int)threadIdx.z) * 16) + 13))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 209))], placeholder_shared[(((((int)threadIdx.z) * 16) + 269))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 224))], placeholder_shared[(((((int)threadIdx.z) * 16) + 14))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 224))], placeholder_shared[(((((int)threadIdx.z) * 16) + 270))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 225))], placeholder_shared[(((((int)threadIdx.z) * 16) + 14))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 225))], placeholder_shared[(((((int)threadIdx.z) * 16) + 270))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 240))], placeholder_shared[(((((int)threadIdx.z) * 16) + 15))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 240))], placeholder_shared[(((((int)threadIdx.z) * 16) + 271))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 241))], placeholder_shared[(((((int)threadIdx.z) * 16) + 15))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 241))], placeholder_shared[(((((int)threadIdx.z) * 16) + 271))], compute[(3)]);
  }
  T_relu[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)))] = max((compute[(0)] + placeholder2[(((((int)blockIdx.z) * 32) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[((((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 50176))] = max((compute[(2)] + placeholder2[((((((int)blockIdx.z) * 32) + ((int)threadIdx.z)) + 16))]), 0.000000e+00f);
  T_relu[((((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 1))] = max((compute[(1)] + placeholder2[(((((int)blockIdx.z) * 32) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[((((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 50177))] = max((compute[(3)] + placeholder2[((((((int)blockIdx.z) * 32) + ((int)threadIdx.z)) + 16))]), 0.000000e+00f);
}

extern "C" __global__ void fused_nn_conv2d_1_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ compute) {
  float compute_local[8];
  __shared__ float pad_temp_shared[1320];
  __shared__ float placeholder_shared[512];
  compute_local[(0)] = 0.000000e+00f;
  compute_local[(2)] = 0.000000e+00f;
  compute_local[(4)] = 0.000000e+00f;
  compute_local[(6)] = 0.000000e+00f;
  compute_local[(1)] = 0.000000e+00f;
  compute_local[(3)] = 0.000000e+00f;
  compute_local[(5)] = 0.000000e+00f;
  compute_local[(7)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 32; ++rc_outer) {
    __syncthreads();
    if ((((((int)threadIdx.z) * 42) + (((int)threadIdx.y) * 21)) + (((int)threadIdx.x) * 3)) < 1320) {
      pad_temp_shared[((((((int)threadIdx.z) * 42) + (((int)threadIdx.y) * 21)) + (((int)threadIdx.x) * 3)))] = placeholder[((((((rc_outer * 25088) + (((((((int)threadIdx.z) * 42) + (((int)threadIdx.y) * 21)) + (((int)threadIdx.x) * 3)) / 165) * 3136)) + (((int)blockIdx.y) * 224)) + ((((((((int)threadIdx.z) * 42) + (((int)threadIdx.y) * 21)) + (((int)threadIdx.x) * 3)) % 165) / 55) * 56)) + ((((((int)threadIdx.z) * 42) + (((int)threadIdx.y) * 21)) + (((int)threadIdx.x) * 3)) % 55)))];
    }
    if ((((((int)threadIdx.z) * 42) + (((int)threadIdx.y) * 21)) + (((int)threadIdx.x) * 3)) < 1319) {
      pad_temp_shared[(((((((int)threadIdx.z) * 42) + (((int)threadIdx.y) * 21)) + (((int)threadIdx.x) * 3)) + 1))] = placeholder[((((((rc_outer * 25088) + ((((((((int)threadIdx.z) * 42) + (((int)threadIdx.y) * 21)) + (((int)threadIdx.x) * 3)) + 1) / 165) * 3136)) + (((int)blockIdx.y) * 224)) + (((((((((int)threadIdx.z) * 42) + (((int)threadIdx.y) * 21)) + (((int)threadIdx.x) * 3)) + 1) % 165) / 55) * 56)) + (((((((int)threadIdx.z) * 42) + (((int)threadIdx.y) * 21)) + (((int)threadIdx.x) * 3)) + 1) % 55)))];
    }
    if ((((((int)threadIdx.z) * 42) + (((int)threadIdx.y) * 21)) + (((int)threadIdx.x) * 3)) < 1318) {
      pad_temp_shared[(((((((int)threadIdx.z) * 42) + (((int)threadIdx.y) * 21)) + (((int)threadIdx.x) * 3)) + 2))] = placeholder[((((((rc_outer * 25088) + ((((((((int)threadIdx.z) * 42) + (((int)threadIdx.y) * 21)) + (((int)threadIdx.x) * 3)) + 2) / 165) * 3136)) + (((int)blockIdx.y) * 224)) + (((((((((int)threadIdx.z) * 42) + (((int)threadIdx.y) * 21)) + (((int)threadIdx.x) * 3)) + 2) % 165) / 55) * 56)) + (((((((int)threadIdx.z) * 42) + (((int)threadIdx.y) * 21)) + (((int)threadIdx.x) * 3)) + 2) % 55)))];
    }
    if ((((((int)threadIdx.z) * 2) + (((int)threadIdx.x) >> 2)) + ((int)threadIdx.y)) < 64) {
      if ((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) < 512) {
        if (((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) < 16) {
          if (((int)threadIdx.x) < 4) {
            placeholder_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)))] = placeholder1[((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 512)) + (((int)threadIdx.y) * 256)) + (rc_outer * 8)) + (((int)threadIdx.x) * 2)))];
          }
        }
      }
    }
    if ((((((int)threadIdx.z) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 3)) + ((int)threadIdx.y)) < 64) {
      if ((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) < 511) {
        if (((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) < 15) {
          if (((int)threadIdx.x) < 4) {
            placeholder_shared[(((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1))] = placeholder1[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 512)) + (((int)threadIdx.y) * 256)) + (rc_outer * 8)) + (((int)threadIdx.x) * 2)) + 1))];
          }
        }
      }
    }
    __syncthreads();
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 2)))], placeholder_shared[((((int)threadIdx.z) * 16))], compute_local[(0)]);
    compute_local[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 2)) + 14))], placeholder_shared[((((int)threadIdx.z) * 16))], compute_local[(2)]);
    compute_local[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 2)) + 28))], placeholder_shared[((((int)threadIdx.z) * 16))], compute_local[(4)]);
    compute_local[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 2)) + 42))], placeholder_shared[((((int)threadIdx.z) * 16))], compute_local[(6)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 2)))], placeholder_shared[(((((int)threadIdx.z) * 16) + 8))], compute_local[(1)]);
    compute_local[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 2)) + 14))], placeholder_shared[(((((int)threadIdx.z) * 16) + 8))], compute_local[(3)]);
    compute_local[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 2)) + 28))], placeholder_shared[(((((int)threadIdx.z) * 16) + 8))], compute_local[(5)]);
    compute_local[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 2)) + 42))], placeholder_shared[(((((int)threadIdx.z) * 16) + 8))], compute_local[(7)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 2)) + 165))], placeholder_shared[(((((int)threadIdx.z) * 16) + 1))], compute_local[(0)]);
    compute_local[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 2)) + 179))], placeholder_shared[(((((int)threadIdx.z) * 16) + 1))], compute_local[(2)]);
    compute_local[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 2)) + 193))], placeholder_shared[(((((int)threadIdx.z) * 16) + 1))], compute_local[(4)]);
    compute_local[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 2)) + 207))], placeholder_shared[(((((int)threadIdx.z) * 16) + 1))], compute_local[(6)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 2)) + 165))], placeholder_shared[(((((int)threadIdx.z) * 16) + 9))], compute_local[(1)]);
    compute_local[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 2)) + 179))], placeholder_shared[(((((int)threadIdx.z) * 16) + 9))], compute_local[(3)]);
    compute_local[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 2)) + 193))], placeholder_shared[(((((int)threadIdx.z) * 16) + 9))], compute_local[(5)]);
    compute_local[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 2)) + 207))], placeholder_shared[(((((int)threadIdx.z) * 16) + 9))], compute_local[(7)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 2)) + 330))], placeholder_shared[(((((int)threadIdx.z) * 16) + 2))], compute_local[(0)]);
    compute_local[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 2)) + 344))], placeholder_shared[(((((int)threadIdx.z) * 16) + 2))], compute_local[(2)]);
    compute_local[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 2)) + 358))], placeholder_shared[(((((int)threadIdx.z) * 16) + 2))], compute_local[(4)]);
    compute_local[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 2)) + 372))], placeholder_shared[(((((int)threadIdx.z) * 16) + 2))], compute_local[(6)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 2)) + 330))], placeholder_shared[(((((int)threadIdx.z) * 16) + 10))], compute_local[(1)]);
    compute_local[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 2)) + 344))], placeholder_shared[(((((int)threadIdx.z) * 16) + 10))], compute_local[(3)]);
    compute_local[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 2)) + 358))], placeholder_shared[(((((int)threadIdx.z) * 16) + 10))], compute_local[(5)]);
    compute_local[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 2)) + 372))], placeholder_shared[(((((int)threadIdx.z) * 16) + 10))], compute_local[(7)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 2)) + 495))], placeholder_shared[(((((int)threadIdx.z) * 16) + 3))], compute_local[(0)]);
    compute_local[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 2)) + 509))], placeholder_shared[(((((int)threadIdx.z) * 16) + 3))], compute_local[(2)]);
    compute_local[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 2)) + 523))], placeholder_shared[(((((int)threadIdx.z) * 16) + 3))], compute_local[(4)]);
    compute_local[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 2)) + 537))], placeholder_shared[(((((int)threadIdx.z) * 16) + 3))], compute_local[(6)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 2)) + 495))], placeholder_shared[(((((int)threadIdx.z) * 16) + 11))], compute_local[(1)]);
    compute_local[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 2)) + 509))], placeholder_shared[(((((int)threadIdx.z) * 16) + 11))], compute_local[(3)]);
    compute_local[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 2)) + 523))], placeholder_shared[(((((int)threadIdx.z) * 16) + 11))], compute_local[(5)]);
    compute_local[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 2)) + 537))], placeholder_shared[(((((int)threadIdx.z) * 16) + 11))], compute_local[(7)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 2)) + 660))], placeholder_shared[(((((int)threadIdx.z) * 16) + 4))], compute_local[(0)]);
    compute_local[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 2)) + 674))], placeholder_shared[(((((int)threadIdx.z) * 16) + 4))], compute_local[(2)]);
    compute_local[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 2)) + 688))], placeholder_shared[(((((int)threadIdx.z) * 16) + 4))], compute_local[(4)]);
    compute_local[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 2)) + 702))], placeholder_shared[(((((int)threadIdx.z) * 16) + 4))], compute_local[(6)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 2)) + 660))], placeholder_shared[(((((int)threadIdx.z) * 16) + 12))], compute_local[(1)]);
    compute_local[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 2)) + 674))], placeholder_shared[(((((int)threadIdx.z) * 16) + 12))], compute_local[(3)]);
    compute_local[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 2)) + 688))], placeholder_shared[(((((int)threadIdx.z) * 16) + 12))], compute_local[(5)]);
    compute_local[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 2)) + 702))], placeholder_shared[(((((int)threadIdx.z) * 16) + 12))], compute_local[(7)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 2)) + 825))], placeholder_shared[(((((int)threadIdx.z) * 16) + 5))], compute_local[(0)]);
    compute_local[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 2)) + 839))], placeholder_shared[(((((int)threadIdx.z) * 16) + 5))], compute_local[(2)]);
    compute_local[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 2)) + 853))], placeholder_shared[(((((int)threadIdx.z) * 16) + 5))], compute_local[(4)]);
    compute_local[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 2)) + 867))], placeholder_shared[(((((int)threadIdx.z) * 16) + 5))], compute_local[(6)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 2)) + 825))], placeholder_shared[(((((int)threadIdx.z) * 16) + 13))], compute_local[(1)]);
    compute_local[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 2)) + 839))], placeholder_shared[(((((int)threadIdx.z) * 16) + 13))], compute_local[(3)]);
    compute_local[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 2)) + 853))], placeholder_shared[(((((int)threadIdx.z) * 16) + 13))], compute_local[(5)]);
    compute_local[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 2)) + 867))], placeholder_shared[(((((int)threadIdx.z) * 16) + 13))], compute_local[(7)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 2)) + 990))], placeholder_shared[(((((int)threadIdx.z) * 16) + 6))], compute_local[(0)]);
    compute_local[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 2)) + 1004))], placeholder_shared[(((((int)threadIdx.z) * 16) + 6))], compute_local[(2)]);
    compute_local[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 2)) + 1018))], placeholder_shared[(((((int)threadIdx.z) * 16) + 6))], compute_local[(4)]);
    compute_local[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 2)) + 1032))], placeholder_shared[(((((int)threadIdx.z) * 16) + 6))], compute_local[(6)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 2)) + 990))], placeholder_shared[(((((int)threadIdx.z) * 16) + 14))], compute_local[(1)]);
    compute_local[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 2)) + 1004))], placeholder_shared[(((((int)threadIdx.z) * 16) + 14))], compute_local[(3)]);
    compute_local[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 2)) + 1018))], placeholder_shared[(((((int)threadIdx.z) * 16) + 14))], compute_local[(5)]);
    compute_local[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 2)) + 1032))], placeholder_shared[(((((int)threadIdx.z) * 16) + 14))], compute_local[(7)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 2)) + 1155))], placeholder_shared[(((((int)threadIdx.z) * 16) + 7))], compute_local[(0)]);
    compute_local[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 2)) + 1169))], placeholder_shared[(((((int)threadIdx.z) * 16) + 7))], compute_local[(2)]);
    compute_local[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 2)) + 1183))], placeholder_shared[(((((int)threadIdx.z) * 16) + 7))], compute_local[(4)]);
    compute_local[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 2)) + 1197))], placeholder_shared[(((((int)threadIdx.z) * 16) + 7))], compute_local[(6)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 2)) + 1155))], placeholder_shared[(((((int)threadIdx.z) * 16) + 15))], compute_local[(1)]);
    compute_local[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 2)) + 1169))], placeholder_shared[(((((int)threadIdx.z) * 16) + 15))], compute_local[(3)]);
    compute_local[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 2)) + 1183))], placeholder_shared[(((((int)threadIdx.z) * 16) + 15))], compute_local[(5)]);
    compute_local[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 2)) + 1197))], placeholder_shared[(((((int)threadIdx.z) * 16) + 15))], compute_local[(7)]);
  }
  compute[((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (((int)blockIdx.y) * 56)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)))] = compute_local[(0)];
  compute[(((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (((int)blockIdx.y) * 56)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 7))] = compute_local[(2)];
  compute[(((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (((int)blockIdx.y) * 56)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 14))] = compute_local[(4)];
  compute[(((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (((int)blockIdx.y) * 56)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 21))] = compute_local[(6)];
  compute[(((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (((int)blockIdx.y) * 56)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 784))] = compute_local[(1)];
  compute[(((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (((int)blockIdx.y) * 56)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 791))] = compute_local[(3)];
  compute[(((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (((int)blockIdx.y) * 56)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 798))] = compute_local[(5)];
  compute[(((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (((int)blockIdx.y) * 56)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 805))] = compute_local[(7)];
}

extern "C" __global__ void fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_kernel1(float* __restrict__ placeholder, float* __restrict__ data_pack, float* __restrict__ bgemm) {
  float bgemm_local[4];
  __shared__ float placeholder_shared[1024];
  __shared__ float data_pack_shared[256];
  bgemm_local[(0)] = 0.000000e+00f;
  bgemm_local[(1)] = 0.000000e+00f;
  bgemm_local[(2)] = 0.000000e+00f;
  bgemm_local[(3)] = 0.000000e+00f;
  for (int ci_outer = 0; ci_outer < 32; ++ci_outer) {
    __syncthreads();
    placeholder_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)))] = placeholder[((((((((int)blockIdx.z) * 262144) + (ci_outer * 8192)) + ((((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) >> 6) * 512)) + (((int)blockIdx.y) * 64)) + (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) & 63)))];
    placeholder_shared[((((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 256))] = placeholder[(((((((((int)blockIdx.z) * 262144) + (ci_outer * 8192)) + ((((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) >> 6) * 512)) + (((int)blockIdx.y) * 64)) + (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) & 63)) + 2048))];
    placeholder_shared[((((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 512))] = placeholder[(((((((((int)blockIdx.z) * 262144) + (ci_outer * 8192)) + ((((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) >> 6) * 512)) + (((int)blockIdx.y) * 64)) + (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) & 63)) + 4096))];
    placeholder_shared[((((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 768))] = placeholder[(((((((((int)blockIdx.z) * 262144) + (ci_outer * 8192)) + ((((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) >> 6) * 512)) + (((int)blockIdx.y) * 64)) + (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) & 63)) + 6144))];
    data_pack_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)))] = data_pack[(((((((int)blockIdx.z) * 8192) + (ci_outer * 256)) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x)))];
    __syncthreads();
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[((((int)threadIdx.y) * 2))], data_pack_shared[((((int)threadIdx.x) * 2))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[((((int)threadIdx.y) * 2))], data_pack_shared[(((((int)threadIdx.x) * 2) + 1))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 1))], data_pack_shared[((((int)threadIdx.x) * 2))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 1))], data_pack_shared[(((((int)threadIdx.x) * 2) + 1))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 64))], data_pack_shared[(((((int)threadIdx.x) * 2) + 16))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 64))], data_pack_shared[(((((int)threadIdx.x) * 2) + 17))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 65))], data_pack_shared[(((((int)threadIdx.x) * 2) + 16))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 65))], data_pack_shared[(((((int)threadIdx.x) * 2) + 17))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 128))], data_pack_shared[(((((int)threadIdx.x) * 2) + 32))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 128))], data_pack_shared[(((((int)threadIdx.x) * 2) + 33))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 129))], data_pack_shared[(((((int)threadIdx.x) * 2) + 32))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 129))], data_pack_shared[(((((int)threadIdx.x) * 2) + 33))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 192))], data_pack_shared[(((((int)threadIdx.x) * 2) + 48))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 192))], data_pack_shared[(((((int)threadIdx.x) * 2) + 49))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 193))], data_pack_shared[(((((int)threadIdx.x) * 2) + 48))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 193))], data_pack_shared[(((((int)threadIdx.x) * 2) + 49))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 256))], data_pack_shared[(((((int)threadIdx.x) * 2) + 64))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 256))], data_pack_shared[(((((int)threadIdx.x) * 2) + 65))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 257))], data_pack_shared[(((((int)threadIdx.x) * 2) + 64))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 257))], data_pack_shared[(((((int)threadIdx.x) * 2) + 65))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 320))], data_pack_shared[(((((int)threadIdx.x) * 2) + 80))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 320))], data_pack_shared[(((((int)threadIdx.x) * 2) + 81))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 321))], data_pack_shared[(((((int)threadIdx.x) * 2) + 80))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 321))], data_pack_shared[(((((int)threadIdx.x) * 2) + 81))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 384))], data_pack_shared[(((((int)threadIdx.x) * 2) + 96))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 384))], data_pack_shared[(((((int)threadIdx.x) * 2) + 97))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 385))], data_pack_shared[(((((int)threadIdx.x) * 2) + 96))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 385))], data_pack_shared[(((((int)threadIdx.x) * 2) + 97))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 448))], data_pack_shared[(((((int)threadIdx.x) * 2) + 112))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 448))], data_pack_shared[(((((int)threadIdx.x) * 2) + 113))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 449))], data_pack_shared[(((((int)threadIdx.x) * 2) + 112))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 449))], data_pack_shared[(((((int)threadIdx.x) * 2) + 113))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 512))], data_pack_shared[(((((int)threadIdx.x) * 2) + 128))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 512))], data_pack_shared[(((((int)threadIdx.x) * 2) + 129))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 513))], data_pack_shared[(((((int)threadIdx.x) * 2) + 128))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 513))], data_pack_shared[(((((int)threadIdx.x) * 2) + 129))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 576))], data_pack_shared[(((((int)threadIdx.x) * 2) + 144))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 576))], data_pack_shared[(((((int)threadIdx.x) * 2) + 145))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 577))], data_pack_shared[(((((int)threadIdx.x) * 2) + 144))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 577))], data_pack_shared[(((((int)threadIdx.x) * 2) + 145))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 640))], data_pack_shared[(((((int)threadIdx.x) * 2) + 160))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 640))], data_pack_shared[(((((int)threadIdx.x) * 2) + 161))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 641))], data_pack_shared[(((((int)threadIdx.x) * 2) + 160))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 641))], data_pack_shared[(((((int)threadIdx.x) * 2) + 161))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 704))], data_pack_shared[(((((int)threadIdx.x) * 2) + 176))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 704))], data_pack_shared[(((((int)threadIdx.x) * 2) + 177))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 705))], data_pack_shared[(((((int)threadIdx.x) * 2) + 176))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 705))], data_pack_shared[(((((int)threadIdx.x) * 2) + 177))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 768))], data_pack_shared[(((((int)threadIdx.x) * 2) + 192))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 768))], data_pack_shared[(((((int)threadIdx.x) * 2) + 193))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 769))], data_pack_shared[(((((int)threadIdx.x) * 2) + 192))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 769))], data_pack_shared[(((((int)threadIdx.x) * 2) + 193))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 832))], data_pack_shared[(((((int)threadIdx.x) * 2) + 208))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 832))], data_pack_shared[(((((int)threadIdx.x) * 2) + 209))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 833))], data_pack_shared[(((((int)threadIdx.x) * 2) + 208))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 833))], data_pack_shared[(((((int)threadIdx.x) * 2) + 209))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 896))], data_pack_shared[(((((int)threadIdx.x) * 2) + 224))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 896))], data_pack_shared[(((((int)threadIdx.x) * 2) + 225))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 897))], data_pack_shared[(((((int)threadIdx.x) * 2) + 224))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 897))], data_pack_shared[(((((int)threadIdx.x) * 2) + 225))], bgemm_local[(3)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 960))], data_pack_shared[(((((int)threadIdx.x) * 2) + 240))], bgemm_local[(0)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 960))], data_pack_shared[(((((int)threadIdx.x) * 2) + 241))], bgemm_local[(1)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 961))], data_pack_shared[(((((int)threadIdx.x) * 2) + 240))], bgemm_local[(2)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 961))], data_pack_shared[(((((int)threadIdx.x) * 2) + 241))], bgemm_local[(3)]);
  }
  bgemm[(((((((int)blockIdx.z) * 8192) + (((int)blockIdx.y) * 1024)) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 2)))] = bgemm_local[(0)];
  bgemm[((((((((int)blockIdx.z) * 8192) + (((int)blockIdx.y) * 1024)) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 2)) + 1))] = bgemm_local[(1)];
  bgemm[((((((((int)blockIdx.z) * 8192) + (((int)blockIdx.y) * 1024)) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 2)) + 16))] = bgemm_local[(2)];
  bgemm[((((((((int)blockIdx.z) * 8192) + (((int)blockIdx.y) * 1024)) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 2)) + 17))] = bgemm_local[(3)];
}

extern "C" __global__ void fused_add_nn_relu_2_kernel0(float* __restrict__ T_relu, float* __restrict__ placeholder, float* __restrict__ placeholder1) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer < 7; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) < 401408) {
      T_relu[((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)))] = max((placeholder[((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)))] + placeholder1[(((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) / 784))]), 0.000000e+00f);
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  float compute[1];
  __shared__ float pad_temp_shared[6272];
  __shared__ float placeholder_shared[1024];
  compute[(0)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 16; ++rc_outer) {
    __syncthreads();
    pad_temp_shared[((((((int)threadIdx.z) * 784) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 16)))] = placeholder[(((((rc_outer * 6272) + (((int)threadIdx.z) * 784)) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 16)))];
    pad_temp_shared[(((((((int)threadIdx.z) * 784) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 16)) + 1))] = placeholder[(((((rc_outer * 6272) + (((int)threadIdx.z) * 784)) + (((int)threadIdx.y) * 112)) + ((((int)threadIdx.x) * 16) + 1)))];
    pad_temp_shared[(((((((int)threadIdx.z) * 784) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 16)) + 2))] = placeholder[(((((rc_outer * 6272) + (((int)threadIdx.z) * 784)) + (((int)threadIdx.y) * 112)) + ((((int)threadIdx.x) * 16) + 2)))];
    pad_temp_shared[(((((((int)threadIdx.z) * 784) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 16)) + 3))] = placeholder[(((((rc_outer * 6272) + (((int)threadIdx.z) * 784)) + (((int)threadIdx.y) * 112)) + ((((int)threadIdx.x) * 16) + 3)))];
    pad_temp_shared[(((((((int)threadIdx.z) * 784) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 16)) + 4))] = placeholder[(((((rc_outer * 6272) + (((int)threadIdx.z) * 784)) + (((int)threadIdx.y) * 112)) + ((((int)threadIdx.x) * 16) + 4)))];
    pad_temp_shared[(((((((int)threadIdx.z) * 784) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 16)) + 5))] = placeholder[(((((rc_outer * 6272) + (((int)threadIdx.z) * 784)) + (((int)threadIdx.y) * 112)) + ((((int)threadIdx.x) * 16) + 5)))];
    pad_temp_shared[(((((((int)threadIdx.z) * 784) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 16)) + 6))] = placeholder[(((((rc_outer * 6272) + (((int)threadIdx.z) * 784)) + (((int)threadIdx.y) * 112)) + ((((int)threadIdx.x) * 16) + 6)))];
    pad_temp_shared[(((((((int)threadIdx.z) * 784) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 16)) + 7))] = placeholder[((((((rc_outer * 6272) + (((int)threadIdx.z) * 784)) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 16)) + 7))];
    pad_temp_shared[(((((((int)threadIdx.z) * 784) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 16)) + 8))] = placeholder[((((((rc_outer * 6272) + (((int)threadIdx.z) * 784)) + (((int)threadIdx.y) * 112)) + ((((((int)threadIdx.x) * 16) + 8) / 7) * 7)) + (((((int)threadIdx.x) * 16) + 1) % 7)))];
    pad_temp_shared[(((((((int)threadIdx.z) * 784) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 16)) + 9))] = placeholder[((((((rc_outer * 6272) + (((int)threadIdx.z) * 784)) + (((int)threadIdx.y) * 112)) + ((((((int)threadIdx.x) * 16) + 9) / 7) * 7)) + (((((int)threadIdx.x) * 16) + 2) % 7)))];
    pad_temp_shared[(((((((int)threadIdx.z) * 784) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 16)) + 10))] = placeholder[((((((rc_outer * 6272) + (((int)threadIdx.z) * 784)) + (((int)threadIdx.y) * 112)) + ((((((int)threadIdx.x) * 16) + 10) / 7) * 7)) + (((((int)threadIdx.x) * 16) + 3) % 7)))];
    pad_temp_shared[(((((((int)threadIdx.z) * 784) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 16)) + 11))] = placeholder[((((((rc_outer * 6272) + (((int)threadIdx.z) * 784)) + (((int)threadIdx.y) * 112)) + ((((((int)threadIdx.x) * 16) + 11) / 7) * 7)) + (((((int)threadIdx.x) * 16) + 4) % 7)))];
    pad_temp_shared[(((((((int)threadIdx.z) * 784) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 16)) + 12))] = placeholder[((((((rc_outer * 6272) + (((int)threadIdx.z) * 784)) + (((int)threadIdx.y) * 112)) + ((((((int)threadIdx.x) * 16) + 12) / 7) * 7)) + (((((int)threadIdx.x) * 16) + 5) % 7)))];
    pad_temp_shared[(((((((int)threadIdx.z) * 784) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 16)) + 13))] = placeholder[((((((rc_outer * 6272) + (((int)threadIdx.z) * 784)) + (((int)threadIdx.y) * 112)) + ((((((int)threadIdx.x) * 16) + 13) / 7) * 7)) + (((((int)threadIdx.x) * 16) + 6) % 7)))];
    pad_temp_shared[(((((((int)threadIdx.z) * 784) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 16)) + 14))] = placeholder[((((((rc_outer * 6272) + (((int)threadIdx.z) * 784)) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 16)) + 14))];
    pad_temp_shared[(((((((int)threadIdx.z) * 784) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 16)) + 15))] = placeholder[((((((rc_outer * 6272) + (((int)threadIdx.z) * 784)) + (((int)threadIdx.y) * 112)) + ((((((int)threadIdx.x) * 16) + 15) / 7) * 7)) + (((((int)threadIdx.x) * 16) + 1) % 7)))];
    if (((((((int)threadIdx.y) * 19) + (((int)threadIdx.x) * 3)) >> 7) + ((int)threadIdx.z)) < 8) {
      if ((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 19)) + (((int)threadIdx.x) * 3)) < 1024) {
        if (((((int)threadIdx.y) * 19) + (((int)threadIdx.x) * 3)) < 128) {
          placeholder_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 19)) + (((int)threadIdx.x) * 3)))] = placeholder1[((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 2048)) + (rc_outer * 128)) + (((int)threadIdx.y) * 19)) + (((int)threadIdx.x) * 3)))];
        }
      }
    }
    if ((((((((int)threadIdx.y) * 19) + (((int)threadIdx.x) * 3)) + 1) >> 7) + ((int)threadIdx.z)) < 8) {
      if ((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 19)) + (((int)threadIdx.x) * 3)) < 1023) {
        if (((((int)threadIdx.y) * 19) + (((int)threadIdx.x) * 3)) < 127) {
          if (((int)threadIdx.x) < 6) {
            placeholder_shared[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 19)) + (((int)threadIdx.x) * 3)) + 1))] = placeholder1[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 2048)) + (rc_outer * 128)) + (((int)threadIdx.y) * 19)) + (((int)threadIdx.x) * 3)) + 1))];
          }
        }
      }
    }
    if ((((((((int)threadIdx.y) * 19) + (((int)threadIdx.x) * 3)) + 2) >> 7) + ((int)threadIdx.z)) < 8) {
      if ((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 19)) + (((int)threadIdx.x) * 3)) < 1022) {
        if (((((int)threadIdx.y) * 19) + (((int)threadIdx.x) * 3)) < 126) {
          if (((int)threadIdx.x) < 6) {
            placeholder_shared[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 19)) + (((int)threadIdx.x) * 3)) + 2))] = placeholder1[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 2048)) + (rc_outer * 128)) + (((int)threadIdx.y) * 19)) + (((int)threadIdx.x) * 3)) + 2))];
          }
        }
      }
    }
    __syncthreads();
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 7) + ((int)threadIdx.x)))], placeholder_shared[((((int)threadIdx.z) * 128))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 49))], placeholder_shared[(((((int)threadIdx.z) * 128) + 1))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 98))], placeholder_shared[(((((int)threadIdx.z) * 128) + 2))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 147))], placeholder_shared[(((((int)threadIdx.z) * 128) + 3))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 196))], placeholder_shared[(((((int)threadIdx.z) * 128) + 4))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 245))], placeholder_shared[(((((int)threadIdx.z) * 128) + 5))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 294))], placeholder_shared[(((((int)threadIdx.z) * 128) + 6))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 343))], placeholder_shared[(((((int)threadIdx.z) * 128) + 7))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 392))], placeholder_shared[(((((int)threadIdx.z) * 128) + 8))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 441))], placeholder_shared[(((((int)threadIdx.z) * 128) + 9))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 490))], placeholder_shared[(((((int)threadIdx.z) * 128) + 10))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 539))], placeholder_shared[(((((int)threadIdx.z) * 128) + 11))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 588))], placeholder_shared[(((((int)threadIdx.z) * 128) + 12))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 637))], placeholder_shared[(((((int)threadIdx.z) * 128) + 13))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 686))], placeholder_shared[(((((int)threadIdx.z) * 128) + 14))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 735))], placeholder_shared[(((((int)threadIdx.z) * 128) + 15))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 784))], placeholder_shared[(((((int)threadIdx.z) * 128) + 16))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 833))], placeholder_shared[(((((int)threadIdx.z) * 128) + 17))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 882))], placeholder_shared[(((((int)threadIdx.z) * 128) + 18))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 931))], placeholder_shared[(((((int)threadIdx.z) * 128) + 19))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 980))], placeholder_shared[(((((int)threadIdx.z) * 128) + 20))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1029))], placeholder_shared[(((((int)threadIdx.z) * 128) + 21))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1078))], placeholder_shared[(((((int)threadIdx.z) * 128) + 22))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1127))], placeholder_shared[(((((int)threadIdx.z) * 128) + 23))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1176))], placeholder_shared[(((((int)threadIdx.z) * 128) + 24))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1225))], placeholder_shared[(((((int)threadIdx.z) * 128) + 25))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1274))], placeholder_shared[(((((int)threadIdx.z) * 128) + 26))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1323))], placeholder_shared[(((((int)threadIdx.z) * 128) + 27))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1372))], placeholder_shared[(((((int)threadIdx.z) * 128) + 28))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1421))], placeholder_shared[(((((int)threadIdx.z) * 128) + 29))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1470))], placeholder_shared[(((((int)threadIdx.z) * 128) + 30))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1519))], placeholder_shared[(((((int)threadIdx.z) * 128) + 31))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1568))], placeholder_shared[(((((int)threadIdx.z) * 128) + 32))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1617))], placeholder_shared[(((((int)threadIdx.z) * 128) + 33))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1666))], placeholder_shared[(((((int)threadIdx.z) * 128) + 34))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1715))], placeholder_shared[(((((int)threadIdx.z) * 128) + 35))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1764))], placeholder_shared[(((((int)threadIdx.z) * 128) + 36))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1813))], placeholder_shared[(((((int)threadIdx.z) * 128) + 37))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1862))], placeholder_shared[(((((int)threadIdx.z) * 128) + 38))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1911))], placeholder_shared[(((((int)threadIdx.z) * 128) + 39))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1960))], placeholder_shared[(((((int)threadIdx.z) * 128) + 40))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 2009))], placeholder_shared[(((((int)threadIdx.z) * 128) + 41))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 2058))], placeholder_shared[(((((int)threadIdx.z) * 128) + 42))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 2107))], placeholder_shared[(((((int)threadIdx.z) * 128) + 43))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 2156))], placeholder_shared[(((((int)threadIdx.z) * 128) + 44))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 2205))], placeholder_shared[(((((int)threadIdx.z) * 128) + 45))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 2254))], placeholder_shared[(((((int)threadIdx.z) * 128) + 46))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 2303))], placeholder_shared[(((((int)threadIdx.z) * 128) + 47))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 2352))], placeholder_shared[(((((int)threadIdx.z) * 128) + 48))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 2401))], placeholder_shared[(((((int)threadIdx.z) * 128) + 49))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 2450))], placeholder_shared[(((((int)threadIdx.z) * 128) + 50))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 2499))], placeholder_shared[(((((int)threadIdx.z) * 128) + 51))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 2548))], placeholder_shared[(((((int)threadIdx.z) * 128) + 52))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 2597))], placeholder_shared[(((((int)threadIdx.z) * 128) + 53))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 2646))], placeholder_shared[(((((int)threadIdx.z) * 128) + 54))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 2695))], placeholder_shared[(((((int)threadIdx.z) * 128) + 55))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 2744))], placeholder_shared[(((((int)threadIdx.z) * 128) + 56))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 2793))], placeholder_shared[(((((int)threadIdx.z) * 128) + 57))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 2842))], placeholder_shared[(((((int)threadIdx.z) * 128) + 58))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 2891))], placeholder_shared[(((((int)threadIdx.z) * 128) + 59))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 2940))], placeholder_shared[(((((int)threadIdx.z) * 128) + 60))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 2989))], placeholder_shared[(((((int)threadIdx.z) * 128) + 61))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 3038))], placeholder_shared[(((((int)threadIdx.z) * 128) + 62))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 3087))], placeholder_shared[(((((int)threadIdx.z) * 128) + 63))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 3136))], placeholder_shared[(((((int)threadIdx.z) * 128) + 64))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 3185))], placeholder_shared[(((((int)threadIdx.z) * 128) + 65))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 3234))], placeholder_shared[(((((int)threadIdx.z) * 128) + 66))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 3283))], placeholder_shared[(((((int)threadIdx.z) * 128) + 67))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 3332))], placeholder_shared[(((((int)threadIdx.z) * 128) + 68))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 3381))], placeholder_shared[(((((int)threadIdx.z) * 128) + 69))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 3430))], placeholder_shared[(((((int)threadIdx.z) * 128) + 70))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 3479))], placeholder_shared[(((((int)threadIdx.z) * 128) + 71))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 3528))], placeholder_shared[(((((int)threadIdx.z) * 128) + 72))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 3577))], placeholder_shared[(((((int)threadIdx.z) * 128) + 73))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 3626))], placeholder_shared[(((((int)threadIdx.z) * 128) + 74))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 3675))], placeholder_shared[(((((int)threadIdx.z) * 128) + 75))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 3724))], placeholder_shared[(((((int)threadIdx.z) * 128) + 76))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 3773))], placeholder_shared[(((((int)threadIdx.z) * 128) + 77))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 3822))], placeholder_shared[(((((int)threadIdx.z) * 128) + 78))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 3871))], placeholder_shared[(((((int)threadIdx.z) * 128) + 79))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 3920))], placeholder_shared[(((((int)threadIdx.z) * 128) + 80))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 3969))], placeholder_shared[(((((int)threadIdx.z) * 128) + 81))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 4018))], placeholder_shared[(((((int)threadIdx.z) * 128) + 82))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 4067))], placeholder_shared[(((((int)threadIdx.z) * 128) + 83))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 4116))], placeholder_shared[(((((int)threadIdx.z) * 128) + 84))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 4165))], placeholder_shared[(((((int)threadIdx.z) * 128) + 85))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 4214))], placeholder_shared[(((((int)threadIdx.z) * 128) + 86))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 4263))], placeholder_shared[(((((int)threadIdx.z) * 128) + 87))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 4312))], placeholder_shared[(((((int)threadIdx.z) * 128) + 88))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 4361))], placeholder_shared[(((((int)threadIdx.z) * 128) + 89))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 4410))], placeholder_shared[(((((int)threadIdx.z) * 128) + 90))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 4459))], placeholder_shared[(((((int)threadIdx.z) * 128) + 91))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 4508))], placeholder_shared[(((((int)threadIdx.z) * 128) + 92))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 4557))], placeholder_shared[(((((int)threadIdx.z) * 128) + 93))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 4606))], placeholder_shared[(((((int)threadIdx.z) * 128) + 94))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 4655))], placeholder_shared[(((((int)threadIdx.z) * 128) + 95))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 4704))], placeholder_shared[(((((int)threadIdx.z) * 128) + 96))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 4753))], placeholder_shared[(((((int)threadIdx.z) * 128) + 97))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 4802))], placeholder_shared[(((((int)threadIdx.z) * 128) + 98))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 4851))], placeholder_shared[(((((int)threadIdx.z) * 128) + 99))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 4900))], placeholder_shared[(((((int)threadIdx.z) * 128) + 100))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 4949))], placeholder_shared[(((((int)threadIdx.z) * 128) + 101))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 4998))], placeholder_shared[(((((int)threadIdx.z) * 128) + 102))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 5047))], placeholder_shared[(((((int)threadIdx.z) * 128) + 103))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 5096))], placeholder_shared[(((((int)threadIdx.z) * 128) + 104))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 5145))], placeholder_shared[(((((int)threadIdx.z) * 128) + 105))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 5194))], placeholder_shared[(((((int)threadIdx.z) * 128) + 106))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 5243))], placeholder_shared[(((((int)threadIdx.z) * 128) + 107))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 5292))], placeholder_shared[(((((int)threadIdx.z) * 128) + 108))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 5341))], placeholder_shared[(((((int)threadIdx.z) * 128) + 109))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 5390))], placeholder_shared[(((((int)threadIdx.z) * 128) + 110))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 5439))], placeholder_shared[(((((int)threadIdx.z) * 128) + 111))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 5488))], placeholder_shared[(((((int)threadIdx.z) * 128) + 112))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 5537))], placeholder_shared[(((((int)threadIdx.z) * 128) + 113))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 5586))], placeholder_shared[(((((int)threadIdx.z) * 128) + 114))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 5635))], placeholder_shared[(((((int)threadIdx.z) * 128) + 115))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 5684))], placeholder_shared[(((((int)threadIdx.z) * 128) + 116))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 5733))], placeholder_shared[(((((int)threadIdx.z) * 128) + 117))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 5782))], placeholder_shared[(((((int)threadIdx.z) * 128) + 118))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 5831))], placeholder_shared[(((((int)threadIdx.z) * 128) + 119))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 5880))], placeholder_shared[(((((int)threadIdx.z) * 128) + 120))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 5929))], placeholder_shared[(((((int)threadIdx.z) * 128) + 121))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 5978))], placeholder_shared[(((((int)threadIdx.z) * 128) + 122))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 6027))], placeholder_shared[(((((int)threadIdx.z) * 128) + 123))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 6076))], placeholder_shared[(((((int)threadIdx.z) * 128) + 124))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 6125))], placeholder_shared[(((((int)threadIdx.z) * 128) + 125))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 6174))], placeholder_shared[(((((int)threadIdx.z) * 128) + 126))], compute[(0)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 6223))], placeholder_shared[(((((int)threadIdx.z) * 128) + 127))], compute[(0)]);
  }
  T_relu[(((((((int)blockIdx.z) * 392) + (((int)threadIdx.z) * 49)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)))] = max((compute[(0)] + placeholder2[(((((int)blockIdx.z) * 8) + ((int)threadIdx.z)))]), 0.000000e+00f);
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_8_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  float compute[16];
  __shared__ float pad_temp_shared[793];
  __shared__ float placeholder_shared[3136];
  for (int ff_init = 0; ff_init < 4; ++ff_init) {
    for (int xx_init = 0; xx_init < 4; ++xx_init) {
      compute[(((ff_init * 4) + xx_init))] = 0.000000e+00f;
    }
  }
  for (int rc_outer = 0; rc_outer < 3; ++rc_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      if (((((((int)threadIdx.z) * 50) + (((int)threadIdx.y) * 13)) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 793) {
        if ((((((int)threadIdx.y) * 13) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 50) {
          if (((((int)threadIdx.x) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 13) {
            pad_temp_shared[(((((((int)threadIdx.z) * 50) + (((int)threadIdx.y) * 13)) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))] = (((((3 <= ((((int)blockIdx.y) * 8) + (((((((int)threadIdx.z) * 50) + (((int)threadIdx.y) * 13)) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 61))) && (((((int)blockIdx.y) * 8) + (((((((int)threadIdx.z) * 50) + (((int)threadIdx.y) * 13)) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 61)) < 227)) && (3 <= ((((int)blockIdx.x) * 56) + (((((((int)threadIdx.z) * 50) + (((int)threadIdx.y) * 13)) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 61)))) && (((((int)blockIdx.x) * 56) + (((((((int)threadIdx.z) * 50) + (((int)threadIdx.y) * 13)) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 61)) < 227)) ? placeholder[(((((((rc_outer * 50176) + (((int)blockIdx.y) * 1792)) + ((((((((int)threadIdx.z) * 50) + (((int)threadIdx.y) * 13)) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 61) * 224)) + (((int)blockIdx.x) * 56)) + (((((((int)threadIdx.z) * 50) + (((int)threadIdx.y) * 13)) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 61)) - 675))] : 0.000000e+00f);
          }
        }
      }
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 < 7; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) {
      placeholder_shared[(((((((int)threadIdx.z) * 196) + (((int)threadIdx.y) * 49)) + (((int)threadIdx.x) * 7)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1))] = placeholder1[((((((((int)threadIdx.z) * 588) + (((int)threadIdx.y) * 147)) + (rc_outer * 49)) + (((int)threadIdx.x) * 7)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1))];
    }
    __syncthreads();
    for (int ry_inner = 0; ry_inner < 7; ++ry_inner) {
      for (int rx_inner = 0; rx_inner < 7; ++rx_inner) {
        for (int ff = 0; ff < 4; ++ff) {
          for (int xx = 0; xx < 4; ++xx) {
            compute[(((ff * 4) + xx))] = __ocml_fma_f32(pad_temp_shared[((((((((int)threadIdx.y) * 122) + (ry_inner * 61)) + (((int)threadIdx.x) * 8)) + (xx * 2)) + rx_inner))], placeholder_shared[(((((((int)threadIdx.z) * 196) + (ff * 49)) + (ry_inner * 7)) + rx_inner))], compute[(((ff * 4) + xx))]);
          }
        }
      }
    }
  }
  for (int ax1_inner_inner_inner = 0; ax1_inner_inner_inner < 4; ++ax1_inner_inner_inner) {
    for (int ax3_inner_inner_inner = 0; ax3_inner_inner_inner < 4; ++ax3_inner_inner_inner) {
      T_relu[((((((((((int)threadIdx.z) * 50176) + (ax1_inner_inner_inner * 12544)) + (((int)blockIdx.y) * 448)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 28)) + (((int)threadIdx.x) * 4)) + ax3_inner_inner_inner))] = max((compute[(((ax1_inner_inner_inner * 4) + ax3_inner_inner_inner))] + placeholder2[(((((int)threadIdx.z) * 4) + ax1_inner_inner_inner))]), 0.000000e+00f);
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_5_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  float compute[2];
  __shared__ float pad_temp_shared[880];
  __shared__ float placeholder_shared[1024];
  compute[(0)] = 0.000000e+00f;
  compute[(1)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 16; ++rc_outer) {
    __syncthreads();
    if (((((int)threadIdx.z) * 14) + ((int)threadIdx.x)) < 880) {
      pad_temp_shared[(((((int)threadIdx.z) * 14) + ((int)threadIdx.x)))] = placeholder[(((((rc_outer * 50176) + ((((((int)threadIdx.z) * 14) + ((int)threadIdx.x)) / 55) * 3136)) + (((int)blockIdx.y) * 112)) + (((((int)threadIdx.z) * 14) + ((int)threadIdx.x)) % 55)))];
    }
    if (((((int)threadIdx.x) >> 3) + ((int)threadIdx.z)) < 64) {
      if (((((int)threadIdx.z) * 16) + (((int)threadIdx.x) * 2)) < 1024) {
        if (((int)threadIdx.x) < 8) {
          placeholder_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.x) * 2)))] = placeholder1[(((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (rc_outer * 16)) + (((int)threadIdx.x) * 2)))];
        }
      }
    }
    if (((((((int)threadIdx.x) * 2) + 1) >> 4) + ((int)threadIdx.z)) < 64) {
      if (((((int)threadIdx.z) * 16) + (((int)threadIdx.x) * 2)) < 1023) {
        if (((int)threadIdx.x) < 8) {
          placeholder_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.x) * 2)) + 1))] = placeholder1[((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 256)) + (rc_outer * 16)) + (((int)threadIdx.x) * 2)) + 1))];
        }
      }
    }
    __syncthreads();
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((int)threadIdx.x) * 2))], placeholder_shared[((((int)threadIdx.z) * 16))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.x) * 2) + 28))], placeholder_shared[((((int)threadIdx.z) * 16))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.x) * 2) + 55))], placeholder_shared[(((((int)threadIdx.z) * 16) + 1))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.x) * 2) + 83))], placeholder_shared[(((((int)threadIdx.z) * 16) + 1))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.x) * 2) + 110))], placeholder_shared[(((((int)threadIdx.z) * 16) + 2))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.x) * 2) + 138))], placeholder_shared[(((((int)threadIdx.z) * 16) + 2))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.x) * 2) + 165))], placeholder_shared[(((((int)threadIdx.z) * 16) + 3))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.x) * 2) + 193))], placeholder_shared[(((((int)threadIdx.z) * 16) + 3))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.x) * 2) + 220))], placeholder_shared[(((((int)threadIdx.z) * 16) + 4))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.x) * 2) + 248))], placeholder_shared[(((((int)threadIdx.z) * 16) + 4))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.x) * 2) + 275))], placeholder_shared[(((((int)threadIdx.z) * 16) + 5))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.x) * 2) + 303))], placeholder_shared[(((((int)threadIdx.z) * 16) + 5))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.x) * 2) + 330))], placeholder_shared[(((((int)threadIdx.z) * 16) + 6))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.x) * 2) + 358))], placeholder_shared[(((((int)threadIdx.z) * 16) + 6))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.x) * 2) + 385))], placeholder_shared[(((((int)threadIdx.z) * 16) + 7))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.x) * 2) + 413))], placeholder_shared[(((((int)threadIdx.z) * 16) + 7))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.x) * 2) + 440))], placeholder_shared[(((((int)threadIdx.z) * 16) + 8))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.x) * 2) + 468))], placeholder_shared[(((((int)threadIdx.z) * 16) + 8))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.x) * 2) + 495))], placeholder_shared[(((((int)threadIdx.z) * 16) + 9))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.x) * 2) + 523))], placeholder_shared[(((((int)threadIdx.z) * 16) + 9))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.x) * 2) + 550))], placeholder_shared[(((((int)threadIdx.z) * 16) + 10))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.x) * 2) + 578))], placeholder_shared[(((((int)threadIdx.z) * 16) + 10))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.x) * 2) + 605))], placeholder_shared[(((((int)threadIdx.z) * 16) + 11))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.x) * 2) + 633))], placeholder_shared[(((((int)threadIdx.z) * 16) + 11))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.x) * 2) + 660))], placeholder_shared[(((((int)threadIdx.z) * 16) + 12))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.x) * 2) + 688))], placeholder_shared[(((((int)threadIdx.z) * 16) + 12))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.x) * 2) + 715))], placeholder_shared[(((((int)threadIdx.z) * 16) + 13))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.x) * 2) + 743))], placeholder_shared[(((((int)threadIdx.z) * 16) + 13))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.x) * 2) + 770))], placeholder_shared[(((((int)threadIdx.z) * 16) + 14))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.x) * 2) + 798))], placeholder_shared[(((((int)threadIdx.z) * 16) + 14))], compute[(1)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.x) * 2) + 825))], placeholder_shared[(((((int)threadIdx.z) * 16) + 15))], compute[(0)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.x) * 2) + 853))], placeholder_shared[(((((int)threadIdx.z) * 16) + 15))], compute[(1)]);
  }
  T_relu[(((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 28)) + ((int)threadIdx.x)))] = max((compute[(0)] + placeholder2[(((((int)blockIdx.z) * 64) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 28)) + ((int)threadIdx.x)) + 14))] = max((compute[(1)] + placeholder2[(((((int)blockIdx.z) * 64) + ((int)threadIdx.z)))]), 0.000000e+00f);
}

extern "C" __global__ void fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_1_kernel2(float* __restrict__ bgemm, float* __restrict__ T_relu, float* __restrict__ placeholder) {
  float inverse[4];
  inverse[(0)] = 0.000000e+00f;
  inverse[(0)] = (inverse[(0)] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 12544))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 25088))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 50176))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 62720))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 75264))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440))]);
  inverse[(1)] = 0.000000e+00f;
  inverse[(1)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 12544))], -1.000000e+00f, inverse[(1)]);
  inverse[(1)] = (inverse[(1)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 25088))]);
  inverse[(1)] = (inverse[(1)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 37632))]);
  inverse[(1)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 62720))], -1.000000e+00f, inverse[(1)]);
  inverse[(1)] = (inverse[(1)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 75264))]);
  inverse[(1)] = (inverse[(1)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808))]);
  inverse[(1)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896))], -1.000000e+00f, inverse[(1)]);
  inverse[(1)] = (inverse[(1)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440))]);
  inverse[(1)] = (inverse[(1)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 137984))]);
  inverse[(2)] = 0.000000e+00f;
  inverse[(2)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 50176))], -1.000000e+00f, inverse[(2)]);
  inverse[(2)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 62720))], -1.000000e+00f, inverse[(2)]);
  inverse[(2)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 75264))], -1.000000e+00f, inverse[(2)]);
  inverse[(2)] = (inverse[(2)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352))]);
  inverse[(2)] = (inverse[(2)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896))]);
  inverse[(2)] = (inverse[(2)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440))]);
  inverse[(2)] = (inverse[(2)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 150528))]);
  inverse[(2)] = (inverse[(2)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072))]);
  inverse[(2)] = (inverse[(2)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616))]);
  inverse[(3)] = 0.000000e+00f;
  inverse[(3)] = __ocml_fma_f32((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 62720))] * -1.000000e+00f), -1.000000e+00f, inverse[(3)]);
  inverse[(3)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 75264))], -1.000000e+00f, inverse[(3)]);
  inverse[(3)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808))], -1.000000e+00f, inverse[(3)]);
  inverse[(3)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896))], -1.000000e+00f, inverse[(3)]);
  inverse[(3)] = (inverse[(3)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440))]);
  inverse[(3)] = (inverse[(3)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 137984))]);
  inverse[(3)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072))], -1.000000e+00f, inverse[(3)]);
  inverse[(3)] = (inverse[(3)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616))]);
  inverse[(3)] = (inverse[(3)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160))]);
  for (int ax2_inner = 0; ax2_inner < 2; ++ax2_inner) {
    for (int ax3_inner = 0; ax3_inner < 2; ++ax3_inner) {
      T_relu[((((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 7) * 28) + (ax2_inner * 14)) + ((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 7) * 2)) + ax3_inner))] = max((inverse[(((ax2_inner * 2) + ax3_inner))] + placeholder[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49))]), 0.000000e+00f);
    }
  }
}

extern "C" __global__ void fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_3_kernel2(float* __restrict__ bgemm, float* __restrict__ T_relu, float* __restrict__ placeholder) {
  float inverse[16];
  inverse[(0)] = 0.000000e+00f;
  inverse[(0)] = (inverse[(0)] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 12544))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 25088))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 37632))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 50176))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 75264))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 150528))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 225792))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 301056))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232))]);
  inverse[(1)] = 0.000000e+00f;
  inverse[(1)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 12544))], -1.000000e+00f, inverse[(1)]);
  inverse[(1)] = (inverse[(1)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 25088))]);
  inverse[(1)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 37632))], 5.000000e-01f, inverse[(1)]);
  inverse[(1)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 50176))], -2.000000e+00f, inverse[(1)]);
  inverse[(1)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808))], -1.000000e+00f, inverse[(1)]);
  inverse[(1)] = (inverse[(1)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352))]);
  inverse[(1)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896))], 5.000000e-01f, inverse[(1)]);
  inverse[(1)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440))], -2.000000e+00f, inverse[(1)]);
  inverse[(1)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072))], -1.000000e+00f, inverse[(1)]);
  inverse[(1)] = (inverse[(1)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616))]);
  inverse[(1)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160))], 5.000000e-01f, inverse[(1)]);
  inverse[(1)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704))], -2.000000e+00f, inverse[(1)]);
  inverse[(1)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336))], -1.000000e+00f, inverse[(1)]);
  inverse[(1)] = (inverse[(1)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880))]);
  inverse[(1)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424))], 5.000000e-01f, inverse[(1)]);
  inverse[(1)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968))], -2.000000e+00f, inverse[(1)]);
  inverse[(1)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600))], -1.000000e+00f, inverse[(1)]);
  inverse[(1)] = (inverse[(1)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144))]);
  inverse[(1)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688))], 5.000000e-01f, inverse[(1)]);
  inverse[(1)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232))], -2.000000e+00f, inverse[(1)]);
  inverse[(2)] = 0.000000e+00f;
  inverse[(2)] = (inverse[(2)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 12544))]);
  inverse[(2)] = (inverse[(2)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 25088))]);
  inverse[(2)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 37632))], 2.500000e-01f, inverse[(2)]);
  inverse[(2)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 50176))], 4.000000e+00f, inverse[(2)]);
  inverse[(2)] = (inverse[(2)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808))]);
  inverse[(2)] = (inverse[(2)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352))]);
  inverse[(2)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896))], 2.500000e-01f, inverse[(2)]);
  inverse[(2)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440))], 4.000000e+00f, inverse[(2)]);
  inverse[(2)] = (inverse[(2)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072))]);
  inverse[(2)] = (inverse[(2)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616))]);
  inverse[(2)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160))], 2.500000e-01f, inverse[(2)]);
  inverse[(2)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704))], 4.000000e+00f, inverse[(2)]);
  inverse[(2)] = (inverse[(2)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336))]);
  inverse[(2)] = (inverse[(2)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880))]);
  inverse[(2)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424))], 2.500000e-01f, inverse[(2)]);
  inverse[(2)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968))], 4.000000e+00f, inverse[(2)]);
  inverse[(2)] = (inverse[(2)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600))]);
  inverse[(2)] = (inverse[(2)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144))]);
  inverse[(2)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688))], 2.500000e-01f, inverse[(2)]);
  inverse[(2)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232))], 4.000000e+00f, inverse[(2)]);
  inverse[(3)] = 0.000000e+00f;
  inverse[(3)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 12544))], -1.000000e+00f, inverse[(3)]);
  inverse[(3)] = (inverse[(3)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 25088))]);
  inverse[(3)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 37632))], 1.250000e-01f, inverse[(3)]);
  inverse[(3)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 50176))], -8.000000e+00f, inverse[(3)]);
  inverse[(3)] = (inverse[(3)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 62720))]);
  inverse[(3)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808))], -1.000000e+00f, inverse[(3)]);
  inverse[(3)] = (inverse[(3)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352))]);
  inverse[(3)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896))], 1.250000e-01f, inverse[(3)]);
  inverse[(3)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440))], -8.000000e+00f, inverse[(3)]);
  inverse[(3)] = (inverse[(3)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 137984))]);
  inverse[(3)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072))], -1.000000e+00f, inverse[(3)]);
  inverse[(3)] = (inverse[(3)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616))]);
  inverse[(3)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160))], 1.250000e-01f, inverse[(3)]);
  inverse[(3)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704))], -8.000000e+00f, inverse[(3)]);
  inverse[(3)] = (inverse[(3)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 213248))]);
  inverse[(3)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336))], -1.000000e+00f, inverse[(3)]);
  inverse[(3)] = (inverse[(3)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880))]);
  inverse[(3)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424))], 1.250000e-01f, inverse[(3)]);
  inverse[(3)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968))], -8.000000e+00f, inverse[(3)]);
  inverse[(3)] = (inverse[(3)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 288512))]);
  inverse[(3)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600))], -1.000000e+00f, inverse[(3)]);
  inverse[(3)] = (inverse[(3)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144))]);
  inverse[(3)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688))], 1.250000e-01f, inverse[(3)]);
  inverse[(3)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232))], -8.000000e+00f, inverse[(3)]);
  inverse[(3)] = (inverse[(3)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 363776))]);
  inverse[(4)] = 0.000000e+00f;
  inverse[(4)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 75264))], -1.000000e+00f, inverse[(4)]);
  inverse[(4)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808))], -1.000000e+00f, inverse[(4)]);
  inverse[(4)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352))], -1.000000e+00f, inverse[(4)]);
  inverse[(4)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896))], -1.000000e+00f, inverse[(4)]);
  inverse[(4)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440))], -1.000000e+00f, inverse[(4)]);
  inverse[(4)] = (inverse[(4)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 150528))]);
  inverse[(4)] = (inverse[(4)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072))]);
  inverse[(4)] = (inverse[(4)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616))]);
  inverse[(4)] = (inverse[(4)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160))]);
  inverse[(4)] = (inverse[(4)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704))]);
  inverse[(4)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 225792))], 5.000000e-01f, inverse[(4)]);
  inverse[(4)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336))], 5.000000e-01f, inverse[(4)]);
  inverse[(4)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880))], 5.000000e-01f, inverse[(4)]);
  inverse[(4)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424))], 5.000000e-01f, inverse[(4)]);
  inverse[(4)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968))], 5.000000e-01f, inverse[(4)]);
  inverse[(4)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 301056))], -2.000000e+00f, inverse[(4)]);
  inverse[(4)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600))], -2.000000e+00f, inverse[(4)]);
  inverse[(4)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144))], -2.000000e+00f, inverse[(4)]);
  inverse[(4)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688))], -2.000000e+00f, inverse[(4)]);
  inverse[(4)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232))], -2.000000e+00f, inverse[(4)]);
  inverse[(5)] = 0.000000e+00f;
  inverse[(5)] = __ocml_fma_f32((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808))] * -1.000000e+00f), -1.000000e+00f, inverse[(5)]);
  inverse[(5)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352))], -1.000000e+00f, inverse[(5)]);
  inverse[(5)] = __ocml_fma_f32((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896))] * -1.000000e+00f), 5.000000e-01f, inverse[(5)]);
  inverse[(5)] = __ocml_fma_f32((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440))] * -1.000000e+00f), -2.000000e+00f, inverse[(5)]);
  inverse[(5)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072))], -1.000000e+00f, inverse[(5)]);
  inverse[(5)] = (inverse[(5)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616))]);
  inverse[(5)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160))], 5.000000e-01f, inverse[(5)]);
  inverse[(5)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704))], -2.000000e+00f, inverse[(5)]);
  inverse[(5)] = __ocml_fma_f32((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336))] * 5.000000e-01f), -1.000000e+00f, inverse[(5)]);
  inverse[(5)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880))], 5.000000e-01f, inverse[(5)]);
  inverse[(5)] = __ocml_fma_f32((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424))] * 5.000000e-01f), 5.000000e-01f, inverse[(5)]);
  inverse[(5)] = __ocml_fma_f32((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968))] * 5.000000e-01f), -2.000000e+00f, inverse[(5)]);
  inverse[(5)] = __ocml_fma_f32((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600))] * -2.000000e+00f), -1.000000e+00f, inverse[(5)]);
  inverse[(5)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144))], -2.000000e+00f, inverse[(5)]);
  inverse[(5)] = __ocml_fma_f32((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688))] * -2.000000e+00f), 5.000000e-01f, inverse[(5)]);
  inverse[(5)] = __ocml_fma_f32((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232))] * -2.000000e+00f), -2.000000e+00f, inverse[(5)]);
  inverse[(6)] = 0.000000e+00f;
  inverse[(6)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808))], -1.000000e+00f, inverse[(6)]);
  inverse[(6)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352))], -1.000000e+00f, inverse[(6)]);
  inverse[(6)] = __ocml_fma_f32((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896))] * -1.000000e+00f), 2.500000e-01f, inverse[(6)]);
  inverse[(6)] = __ocml_fma_f32((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440))] * -1.000000e+00f), 4.000000e+00f, inverse[(6)]);
  inverse[(6)] = (inverse[(6)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072))]);
  inverse[(6)] = (inverse[(6)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616))]);
  inverse[(6)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160))], 2.500000e-01f, inverse[(6)]);
  inverse[(6)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704))], 4.000000e+00f, inverse[(6)]);
  inverse[(6)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336))], 5.000000e-01f, inverse[(6)]);
  inverse[(6)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880))], 5.000000e-01f, inverse[(6)]);
  inverse[(6)] = __ocml_fma_f32((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424))] * 5.000000e-01f), 2.500000e-01f, inverse[(6)]);
  inverse[(6)] = __ocml_fma_f32((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968))] * 5.000000e-01f), 4.000000e+00f, inverse[(6)]);
  inverse[(6)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600))], -2.000000e+00f, inverse[(6)]);
  inverse[(6)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144))], -2.000000e+00f, inverse[(6)]);
  inverse[(6)] = __ocml_fma_f32((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688))] * -2.000000e+00f), 2.500000e-01f, inverse[(6)]);
  inverse[(6)] = __ocml_fma_f32((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232))] * -2.000000e+00f), 4.000000e+00f, inverse[(6)]);
  inverse[(7)] = 0.000000e+00f;
  inverse[(7)] = __ocml_fma_f32((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808))] * -1.000000e+00f), -1.000000e+00f, inverse[(7)]);
  inverse[(7)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352))], -1.000000e+00f, inverse[(7)]);
  inverse[(7)] = __ocml_fma_f32((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896))] * -1.000000e+00f), 1.250000e-01f, inverse[(7)]);
  inverse[(7)] = __ocml_fma_f32((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440))] * -1.000000e+00f), -8.000000e+00f, inverse[(7)]);
  inverse[(7)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 137984))], -1.000000e+00f, inverse[(7)]);
  inverse[(7)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072))], -1.000000e+00f, inverse[(7)]);
  inverse[(7)] = (inverse[(7)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616))]);
  inverse[(7)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160))], 1.250000e-01f, inverse[(7)]);
  inverse[(7)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704))], -8.000000e+00f, inverse[(7)]);
  inverse[(7)] = (inverse[(7)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 213248))]);
  inverse[(7)] = __ocml_fma_f32((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336))] * 5.000000e-01f), -1.000000e+00f, inverse[(7)]);
  inverse[(7)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880))], 5.000000e-01f, inverse[(7)]);
  inverse[(7)] = __ocml_fma_f32((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424))] * 5.000000e-01f), 1.250000e-01f, inverse[(7)]);
  inverse[(7)] = __ocml_fma_f32((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968))] * 5.000000e-01f), -8.000000e+00f, inverse[(7)]);
  inverse[(7)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 288512))], 5.000000e-01f, inverse[(7)]);
  inverse[(7)] = __ocml_fma_f32((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600))] * -2.000000e+00f), -1.000000e+00f, inverse[(7)]);
  inverse[(7)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144))], -2.000000e+00f, inverse[(7)]);
  inverse[(7)] = __ocml_fma_f32((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688))] * -2.000000e+00f), 1.250000e-01f, inverse[(7)]);
  inverse[(7)] = __ocml_fma_f32((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232))] * -2.000000e+00f), -8.000000e+00f, inverse[(7)]);
  inverse[(7)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 363776))], -2.000000e+00f, inverse[(7)]);
  inverse[(8)] = 0.000000e+00f;
  inverse[(8)] = (inverse[(8)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 75264))]);
  inverse[(8)] = (inverse[(8)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808))]);
  inverse[(8)] = (inverse[(8)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352))]);
  inverse[(8)] = (inverse[(8)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896))]);
  inverse[(8)] = (inverse[(8)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440))]);
  inverse[(8)] = (inverse[(8)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 150528))]);
  inverse[(8)] = (inverse[(8)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072))]);
  inverse[(8)] = (inverse[(8)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616))]);
  inverse[(8)] = (inverse[(8)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160))]);
  inverse[(8)] = (inverse[(8)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704))]);
  inverse[(8)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 225792))], 2.500000e-01f, inverse[(8)]);
  inverse[(8)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336))], 2.500000e-01f, inverse[(8)]);
  inverse[(8)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880))], 2.500000e-01f, inverse[(8)]);
  inverse[(8)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424))], 2.500000e-01f, inverse[(8)]);
  inverse[(8)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968))], 2.500000e-01f, inverse[(8)]);
  inverse[(8)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 301056))], 4.000000e+00f, inverse[(8)]);
  inverse[(8)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600))], 4.000000e+00f, inverse[(8)]);
  inverse[(8)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144))], 4.000000e+00f, inverse[(8)]);
  inverse[(8)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688))], 4.000000e+00f, inverse[(8)]);
  inverse[(8)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232))], 4.000000e+00f, inverse[(8)]);
  inverse[(9)] = 0.000000e+00f;
  inverse[(9)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808))], -1.000000e+00f, inverse[(9)]);
  inverse[(9)] = (inverse[(9)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352))]);
  inverse[(9)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896))], 5.000000e-01f, inverse[(9)]);
  inverse[(9)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440))], -2.000000e+00f, inverse[(9)]);
  inverse[(9)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072))], -1.000000e+00f, inverse[(9)]);
  inverse[(9)] = (inverse[(9)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616))]);
  inverse[(9)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160))], 5.000000e-01f, inverse[(9)]);
  inverse[(9)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704))], -2.000000e+00f, inverse[(9)]);
  inverse[(9)] = __ocml_fma_f32((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336))] * 2.500000e-01f), -1.000000e+00f, inverse[(9)]);
  inverse[(9)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880))], 2.500000e-01f, inverse[(9)]);
  inverse[(9)] = __ocml_fma_f32((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424))] * 2.500000e-01f), 5.000000e-01f, inverse[(9)]);
  inverse[(9)] = __ocml_fma_f32((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968))] * 2.500000e-01f), -2.000000e+00f, inverse[(9)]);
  inverse[(9)] = __ocml_fma_f32((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600))] * 4.000000e+00f), -1.000000e+00f, inverse[(9)]);
  inverse[(9)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144))], 4.000000e+00f, inverse[(9)]);
  inverse[(9)] = __ocml_fma_f32((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688))] * 4.000000e+00f), 5.000000e-01f, inverse[(9)]);
  inverse[(9)] = __ocml_fma_f32((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232))] * 4.000000e+00f), -2.000000e+00f, inverse[(9)]);
  inverse[(10)] = 0.000000e+00f;
  inverse[(10)] = (inverse[(10)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808))]);
  inverse[(10)] = (inverse[(10)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352))]);
  inverse[(10)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896))], 2.500000e-01f, inverse[(10)]);
  inverse[(10)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440))], 4.000000e+00f, inverse[(10)]);
  inverse[(10)] = (inverse[(10)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072))]);
  inverse[(10)] = (inverse[(10)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616))]);
  inverse[(10)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160))], 2.500000e-01f, inverse[(10)]);
  inverse[(10)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704))], 4.000000e+00f, inverse[(10)]);
  inverse[(10)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336))], 2.500000e-01f, inverse[(10)]);
  inverse[(10)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880))], 2.500000e-01f, inverse[(10)]);
  inverse[(10)] = __ocml_fma_f32((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424))] * 2.500000e-01f), 2.500000e-01f, inverse[(10)]);
  inverse[(10)] = __ocml_fma_f32((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968))] * 2.500000e-01f), 4.000000e+00f, inverse[(10)]);
  inverse[(10)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600))], 4.000000e+00f, inverse[(10)]);
  inverse[(10)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144))], 4.000000e+00f, inverse[(10)]);
  inverse[(10)] = __ocml_fma_f32((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688))] * 4.000000e+00f), 2.500000e-01f, inverse[(10)]);
  inverse[(10)] = __ocml_fma_f32((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232))] * 4.000000e+00f), 4.000000e+00f, inverse[(10)]);
  inverse[(11)] = 0.000000e+00f;
  inverse[(11)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808))], -1.000000e+00f, inverse[(11)]);
  inverse[(11)] = (inverse[(11)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352))]);
  inverse[(11)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896))], 1.250000e-01f, inverse[(11)]);
  inverse[(11)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440))], -8.000000e+00f, inverse[(11)]);
  inverse[(11)] = (inverse[(11)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 137984))]);
  inverse[(11)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072))], -1.000000e+00f, inverse[(11)]);
  inverse[(11)] = (inverse[(11)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616))]);
  inverse[(11)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160))], 1.250000e-01f, inverse[(11)]);
  inverse[(11)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704))], -8.000000e+00f, inverse[(11)]);
  inverse[(11)] = (inverse[(11)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 213248))]);
  inverse[(11)] = __ocml_fma_f32((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336))] * 2.500000e-01f), -1.000000e+00f, inverse[(11)]);
  inverse[(11)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880))], 2.500000e-01f, inverse[(11)]);
  inverse[(11)] = __ocml_fma_f32((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424))] * 2.500000e-01f), 1.250000e-01f, inverse[(11)]);
  inverse[(11)] = __ocml_fma_f32((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968))] * 2.500000e-01f), -8.000000e+00f, inverse[(11)]);
  inverse[(11)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 288512))], 2.500000e-01f, inverse[(11)]);
  inverse[(11)] = __ocml_fma_f32((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600))] * 4.000000e+00f), -1.000000e+00f, inverse[(11)]);
  inverse[(11)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144))], 4.000000e+00f, inverse[(11)]);
  inverse[(11)] = __ocml_fma_f32((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688))] * 4.000000e+00f), 1.250000e-01f, inverse[(11)]);
  inverse[(11)] = __ocml_fma_f32((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232))] * 4.000000e+00f), -8.000000e+00f, inverse[(11)]);
  inverse[(11)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 363776))], 4.000000e+00f, inverse[(11)]);
  inverse[(12)] = 0.000000e+00f;
  inverse[(12)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 75264))], -1.000000e+00f, inverse[(12)]);
  inverse[(12)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808))], -1.000000e+00f, inverse[(12)]);
  inverse[(12)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352))], -1.000000e+00f, inverse[(12)]);
  inverse[(12)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896))], -1.000000e+00f, inverse[(12)]);
  inverse[(12)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440))], -1.000000e+00f, inverse[(12)]);
  inverse[(12)] = (inverse[(12)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 150528))]);
  inverse[(12)] = (inverse[(12)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072))]);
  inverse[(12)] = (inverse[(12)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616))]);
  inverse[(12)] = (inverse[(12)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160))]);
  inverse[(12)] = (inverse[(12)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704))]);
  inverse[(12)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 225792))], 1.250000e-01f, inverse[(12)]);
  inverse[(12)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336))], 1.250000e-01f, inverse[(12)]);
  inverse[(12)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880))], 1.250000e-01f, inverse[(12)]);
  inverse[(12)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424))], 1.250000e-01f, inverse[(12)]);
  inverse[(12)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968))], 1.250000e-01f, inverse[(12)]);
  inverse[(12)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 301056))], -8.000000e+00f, inverse[(12)]);
  inverse[(12)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600))], -8.000000e+00f, inverse[(12)]);
  inverse[(12)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144))], -8.000000e+00f, inverse[(12)]);
  inverse[(12)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688))], -8.000000e+00f, inverse[(12)]);
  inverse[(12)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232))], -8.000000e+00f, inverse[(12)]);
  inverse[(12)] = (inverse[(12)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 376320))]);
  inverse[(12)] = (inverse[(12)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 388864))]);
  inverse[(12)] = (inverse[(12)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 401408))]);
  inverse[(12)] = (inverse[(12)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 413952))]);
  inverse[(12)] = (inverse[(12)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 426496))]);
  inverse[(13)] = 0.000000e+00f;
  inverse[(13)] = __ocml_fma_f32((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808))] * -1.000000e+00f), -1.000000e+00f, inverse[(13)]);
  inverse[(13)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352))], -1.000000e+00f, inverse[(13)]);
  inverse[(13)] = __ocml_fma_f32((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896))] * -1.000000e+00f), 5.000000e-01f, inverse[(13)]);
  inverse[(13)] = __ocml_fma_f32((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440))] * -1.000000e+00f), -2.000000e+00f, inverse[(13)]);
  inverse[(13)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072))], -1.000000e+00f, inverse[(13)]);
  inverse[(13)] = (inverse[(13)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616))]);
  inverse[(13)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160))], 5.000000e-01f, inverse[(13)]);
  inverse[(13)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704))], -2.000000e+00f, inverse[(13)]);
  inverse[(13)] = __ocml_fma_f32((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336))] * 1.250000e-01f), -1.000000e+00f, inverse[(13)]);
  inverse[(13)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880))], 1.250000e-01f, inverse[(13)]);
  inverse[(13)] = __ocml_fma_f32((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424))] * 1.250000e-01f), 5.000000e-01f, inverse[(13)]);
  inverse[(13)] = __ocml_fma_f32((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968))] * 1.250000e-01f), -2.000000e+00f, inverse[(13)]);
  inverse[(13)] = __ocml_fma_f32((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600))] * -8.000000e+00f), -1.000000e+00f, inverse[(13)]);
  inverse[(13)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144))], -8.000000e+00f, inverse[(13)]);
  inverse[(13)] = __ocml_fma_f32((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688))] * -8.000000e+00f), 5.000000e-01f, inverse[(13)]);
  inverse[(13)] = __ocml_fma_f32((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232))] * -8.000000e+00f), -2.000000e+00f, inverse[(13)]);
  inverse[(13)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 388864))], -1.000000e+00f, inverse[(13)]);
  inverse[(13)] = (inverse[(13)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 401408))]);
  inverse[(13)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 413952))], 5.000000e-01f, inverse[(13)]);
  inverse[(13)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 426496))], -2.000000e+00f, inverse[(13)]);
  inverse[(14)] = 0.000000e+00f;
  inverse[(14)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808))], -1.000000e+00f, inverse[(14)]);
  inverse[(14)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352))], -1.000000e+00f, inverse[(14)]);
  inverse[(14)] = __ocml_fma_f32((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896))] * -1.000000e+00f), 2.500000e-01f, inverse[(14)]);
  inverse[(14)] = __ocml_fma_f32((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440))] * -1.000000e+00f), 4.000000e+00f, inverse[(14)]);
  inverse[(14)] = (inverse[(14)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072))]);
  inverse[(14)] = (inverse[(14)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616))]);
  inverse[(14)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160))], 2.500000e-01f, inverse[(14)]);
  inverse[(14)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704))], 4.000000e+00f, inverse[(14)]);
  inverse[(14)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336))], 1.250000e-01f, inverse[(14)]);
  inverse[(14)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880))], 1.250000e-01f, inverse[(14)]);
  inverse[(14)] = __ocml_fma_f32((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424))] * 1.250000e-01f), 2.500000e-01f, inverse[(14)]);
  inverse[(14)] = __ocml_fma_f32((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968))] * 1.250000e-01f), 4.000000e+00f, inverse[(14)]);
  inverse[(14)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600))], -8.000000e+00f, inverse[(14)]);
  inverse[(14)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144))], -8.000000e+00f, inverse[(14)]);
  inverse[(14)] = __ocml_fma_f32((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688))] * -8.000000e+00f), 2.500000e-01f, inverse[(14)]);
  inverse[(14)] = __ocml_fma_f32((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232))] * -8.000000e+00f), 4.000000e+00f, inverse[(14)]);
  inverse[(14)] = (inverse[(14)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 388864))]);
  inverse[(14)] = (inverse[(14)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 401408))]);
  inverse[(14)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 413952))], 2.500000e-01f, inverse[(14)]);
  inverse[(14)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 426496))], 4.000000e+00f, inverse[(14)]);
  inverse[(15)] = 0.000000e+00f;
  inverse[(15)] = __ocml_fma_f32((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808))] * -1.000000e+00f), -1.000000e+00f, inverse[(15)]);
  inverse[(15)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352))], -1.000000e+00f, inverse[(15)]);
  inverse[(15)] = __ocml_fma_f32((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896))] * -1.000000e+00f), 1.250000e-01f, inverse[(15)]);
  inverse[(15)] = __ocml_fma_f32((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440))] * -1.000000e+00f), -8.000000e+00f, inverse[(15)]);
  inverse[(15)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 137984))], -1.000000e+00f, inverse[(15)]);
  inverse[(15)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072))], -1.000000e+00f, inverse[(15)]);
  inverse[(15)] = (inverse[(15)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616))]);
  inverse[(15)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160))], 1.250000e-01f, inverse[(15)]);
  inverse[(15)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704))], -8.000000e+00f, inverse[(15)]);
  inverse[(15)] = (inverse[(15)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 213248))]);
  inverse[(15)] = __ocml_fma_f32((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336))] * 1.250000e-01f), -1.000000e+00f, inverse[(15)]);
  inverse[(15)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880))], 1.250000e-01f, inverse[(15)]);
  inverse[(15)] = __ocml_fma_f32((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424))] * 1.250000e-01f), 1.250000e-01f, inverse[(15)]);
  inverse[(15)] = __ocml_fma_f32((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968))] * 1.250000e-01f), -8.000000e+00f, inverse[(15)]);
  inverse[(15)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 288512))], 1.250000e-01f, inverse[(15)]);
  inverse[(15)] = __ocml_fma_f32((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600))] * -8.000000e+00f), -1.000000e+00f, inverse[(15)]);
  inverse[(15)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144))], -8.000000e+00f, inverse[(15)]);
  inverse[(15)] = __ocml_fma_f32((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688))] * -8.000000e+00f), 1.250000e-01f, inverse[(15)]);
  inverse[(15)] = __ocml_fma_f32((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232))] * -8.000000e+00f), -8.000000e+00f, inverse[(15)]);
  inverse[(15)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 363776))], -8.000000e+00f, inverse[(15)]);
  inverse[(15)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 388864))], -1.000000e+00f, inverse[(15)]);
  inverse[(15)] = (inverse[(15)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 401408))]);
  inverse[(15)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 413952))], 1.250000e-01f, inverse[(15)]);
  inverse[(15)] = __ocml_fma_f32(bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 426496))], -8.000000e+00f, inverse[(15)]);
  inverse[(15)] = (inverse[(15)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 439040))]);
  for (int ax2_inner = 0; ax2_inner < 4; ++ax2_inner) {
    for (int ax3_inner = 0; ax3_inner < 4; ++ax3_inner) {
      T_relu[((((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 14) * 224) + (ax2_inner * 56)) + ((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 14) * 4)) + ax3_inner))] = max((inverse[(((ax2_inner * 4) + ax3_inner))] + placeholder[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 196))]), 0.000000e+00f);
    }
  }
}

extern "C" __global__ void fused_add_nn_relu_3_kernel0(float* __restrict__ T_relu, float* __restrict__ placeholder, float* __restrict__ placeholder1) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer < 13; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) < 802816) {
      T_relu[((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)))] = max((placeholder[((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)))] + placeholder1[(((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) / 3136))]), 0.000000e+00f);
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_add_multiply_add_nn_relu_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2, float* __restrict__ placeholder3, float* __restrict__ placeholder4) {
  float compute[2];
  __shared__ float pad_temp_shared[784];
  __shared__ float placeholder_shared[256];
  compute[(0)] = 0.000000e+00f;
  compute[(1)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 32; ++rc_outer) {
    __syncthreads();
    pad_temp_shared[((((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)))] = placeholder[(((((rc_outer * 784) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)))];
    pad_temp_shared[(((((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)) + 1))] = placeholder[(((((rc_outer * 784) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 14)) + ((((int)threadIdx.x) * 2) + 1)))];
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
  T_relu[(((((((int)blockIdx.z) * 784) + (((int)threadIdx.z) * 49)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)))] = max(__ocml_fma_f32((compute[(0)] + placeholder2[(((((((int)blockIdx.z) * 784) + (((int)threadIdx.z) * 49)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)))]), placeholder3[(((((int)blockIdx.z) * 16) + ((int)threadIdx.z)))], placeholder4[(((((int)blockIdx.z) * 16) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 784) + (((int)threadIdx.z) * 49)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) + 392))] = max(__ocml_fma_f32((compute[(1)] + placeholder2[((((((((int)blockIdx.z) * 784) + (((int)threadIdx.z) * 49)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) + 392))]), placeholder3[((((((int)blockIdx.z) * 16) + ((int)threadIdx.z)) + 8))], placeholder4[((((((int)blockIdx.z) * 16) + ((int)threadIdx.z)) + 8))]), 0.000000e+00f);
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

extern "C" __global__ void fused_nn_conv2d_3_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ compute) {
  float compute_local[1];
  __shared__ float pad_temp_shared[2704];
  __shared__ float placeholder_shared[256];
  compute_local[(0)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 64; ++rc_outer) {
    __syncthreads();
    if (((((((int)threadIdx.y) * 25) + (((int)threadIdx.x) * 4)) / 169) + ((int)threadIdx.z)) < 16) {
      if (((((int)threadIdx.z) * 13) + (((((int)threadIdx.y) * 25) + (((int)threadIdx.x) * 4)) / 13)) < 208) {
        if ((((((int)threadIdx.z) * 169) + (((int)threadIdx.y) * 25)) + (((int)threadIdx.x) * 4)) < 2704) {
          if (((((int)threadIdx.y) * 25) + (((int)threadIdx.x) * 4)) < 169) {
            pad_temp_shared[((((((int)threadIdx.z) * 169) + (((int)threadIdx.y) * 25)) + (((int)threadIdx.x) * 4)))] = placeholder[(((((rc_outer * 3136) + (((int)threadIdx.z) * 196)) + ((((((int)threadIdx.y) * 25) + (((int)threadIdx.x) * 4)) / 13) * 14)) + (((((int)threadIdx.y) * 25) + (((int)threadIdx.x) * 4)) % 13)))];
          }
        }
      }
    }
    if ((((((((int)threadIdx.y) * 25) + (((int)threadIdx.x) * 4)) + 1) / 169) + ((int)threadIdx.z)) < 16) {
      if (((((int)threadIdx.z) * 13) + ((((((int)threadIdx.y) * 25) + (((int)threadIdx.x) * 4)) + 1) / 13)) < 208) {
        if ((((((int)threadIdx.z) * 169) + (((int)threadIdx.y) * 25)) + (((int)threadIdx.x) * 4)) < 2703) {
          if (((((int)threadIdx.y) * 25) + (((int)threadIdx.x) * 4)) < 168) {
            if (((int)threadIdx.x) < 6) {
              pad_temp_shared[(((((((int)threadIdx.z) * 169) + (((int)threadIdx.y) * 25)) + (((int)threadIdx.x) * 4)) + 1))] = placeholder[(((((rc_outer * 3136) + (((int)threadIdx.z) * 196)) + (((((((int)threadIdx.y) * 25) + (((int)threadIdx.x) * 4)) + 1) / 13) * 14)) + ((((((int)threadIdx.y) * 25) + (((int)threadIdx.x) * 4)) + 1) % 13)))];
            }
          }
        }
      }
    }
    if ((((((((int)threadIdx.y) * 25) + (((int)threadIdx.x) * 4)) + 2) / 169) + ((int)threadIdx.z)) < 16) {
      if (((((int)threadIdx.z) * 13) + ((((((int)threadIdx.y) * 25) + (((int)threadIdx.x) * 4)) + 2) / 13)) < 208) {
        if ((((((int)threadIdx.z) * 169) + (((int)threadIdx.y) * 25)) + (((int)threadIdx.x) * 4)) < 2702) {
          if (((((int)threadIdx.y) * 25) + (((int)threadIdx.x) * 4)) < 167) {
            if (((int)threadIdx.x) < 6) {
              pad_temp_shared[(((((((int)threadIdx.z) * 169) + (((int)threadIdx.y) * 25)) + (((int)threadIdx.x) * 4)) + 2))] = placeholder[(((((rc_outer * 3136) + (((int)threadIdx.z) * 196)) + (((((((int)threadIdx.y) * 25) + (((int)threadIdx.x) * 4)) + 2) / 13) * 14)) + ((((((int)threadIdx.y) * 25) + (((int)threadIdx.x) * 4)) + 2) % 13)))];
            }
          }
        }
      }
    }
    if ((((((((int)threadIdx.y) * 25) + (((int)threadIdx.x) * 4)) + 3) / 169) + ((int)threadIdx.z)) < 16) {
      if (((((int)threadIdx.z) * 13) + ((((((int)threadIdx.y) * 25) + (((int)threadIdx.x) * 4)) + 3) / 13)) < 208) {
        if ((((((int)threadIdx.z) * 169) + (((int)threadIdx.y) * 25)) + (((int)threadIdx.x) * 4)) < 2701) {
          if (((((int)threadIdx.y) * 25) + (((int)threadIdx.x) * 4)) < 166) {
            if (((int)threadIdx.x) < 6) {
              pad_temp_shared[(((((((int)threadIdx.z) * 169) + (((int)threadIdx.y) * 25)) + (((int)threadIdx.x) * 4)) + 3))] = placeholder[(((((rc_outer * 3136) + (((int)threadIdx.z) * 196)) + (((((((int)threadIdx.y) * 25) + (((int)threadIdx.x) * 4)) + 3) / 13) * 14)) + ((((((int)threadIdx.y) * 25) + (((int)threadIdx.x) * 4)) + 3) % 13)))];
            }
          }
        }
      }
    }
    if (((((((int)threadIdx.y) * 3) + ((int)threadIdx.x)) >> 4) + ((int)threadIdx.z)) < 16) {
      if ((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 3)) + ((int)threadIdx.x)) < 256) {
        if (((((int)threadIdx.y) * 3) + ((int)threadIdx.x)) < 16) {
          if (((int)threadIdx.x) < 3) {
            placeholder_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 3)) + ((int)threadIdx.x)))] = placeholder1[((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 1024)) + (rc_outer * 16)) + (((int)threadIdx.y) * 3)) + ((int)threadIdx.x)))];
          }
        }
      }
    }
    __syncthreads();
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 26) + (((int)threadIdx.x) * 2)))], placeholder_shared[((((int)threadIdx.z) * 16))], compute_local[(0)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 26) + (((int)threadIdx.x) * 2)) + 169))], placeholder_shared[(((((int)threadIdx.z) * 16) + 1))], compute_local[(0)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 26) + (((int)threadIdx.x) * 2)) + 338))], placeholder_shared[(((((int)threadIdx.z) * 16) + 2))], compute_local[(0)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 26) + (((int)threadIdx.x) * 2)) + 507))], placeholder_shared[(((((int)threadIdx.z) * 16) + 3))], compute_local[(0)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 26) + (((int)threadIdx.x) * 2)) + 676))], placeholder_shared[(((((int)threadIdx.z) * 16) + 4))], compute_local[(0)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 26) + (((int)threadIdx.x) * 2)) + 845))], placeholder_shared[(((((int)threadIdx.z) * 16) + 5))], compute_local[(0)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 26) + (((int)threadIdx.x) * 2)) + 1014))], placeholder_shared[(((((int)threadIdx.z) * 16) + 6))], compute_local[(0)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 26) + (((int)threadIdx.x) * 2)) + 1183))], placeholder_shared[(((((int)threadIdx.z) * 16) + 7))], compute_local[(0)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 26) + (((int)threadIdx.x) * 2)) + 1352))], placeholder_shared[(((((int)threadIdx.z) * 16) + 8))], compute_local[(0)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 26) + (((int)threadIdx.x) * 2)) + 1521))], placeholder_shared[(((((int)threadIdx.z) * 16) + 9))], compute_local[(0)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 26) + (((int)threadIdx.x) * 2)) + 1690))], placeholder_shared[(((((int)threadIdx.z) * 16) + 10))], compute_local[(0)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 26) + (((int)threadIdx.x) * 2)) + 1859))], placeholder_shared[(((((int)threadIdx.z) * 16) + 11))], compute_local[(0)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 26) + (((int)threadIdx.x) * 2)) + 2028))], placeholder_shared[(((((int)threadIdx.z) * 16) + 12))], compute_local[(0)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 26) + (((int)threadIdx.x) * 2)) + 2197))], placeholder_shared[(((((int)threadIdx.z) * 16) + 13))], compute_local[(0)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 26) + (((int)threadIdx.x) * 2)) + 2366))], placeholder_shared[(((((int)threadIdx.z) * 16) + 14))], compute_local[(0)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 26) + (((int)threadIdx.x) * 2)) + 2535))], placeholder_shared[(((((int)threadIdx.z) * 16) + 15))], compute_local[(0)]);
  }
  compute[(((((((int)blockIdx.z) * 784) + (((int)threadIdx.z) * 49)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)))] = compute_local[(0)];
}

extern "C" __global__ void fused_nn_batch_flatten_kernel0(float* __restrict__ tensor, float* __restrict__ placeholder) {
  tensor[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)))] = placeholder[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)))];
}

extern "C" __global__ void fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_2_kernel1(float* __restrict__ placeholder, float* __restrict__ data_pack, float* __restrict__ bgemm) {
  float bgemm_local[8];
  __shared__ float placeholder_shared[512];
  __shared__ float data_pack_shared[3136];
  bgemm_local[(0)] = 0.000000e+00f;
  bgemm_local[(4)] = 0.000000e+00f;
  bgemm_local[(2)] = 0.000000e+00f;
  bgemm_local[(6)] = 0.000000e+00f;
  bgemm_local[(1)] = 0.000000e+00f;
  bgemm_local[(5)] = 0.000000e+00f;
  bgemm_local[(3)] = 0.000000e+00f;
  bgemm_local[(7)] = 0.000000e+00f;
  for (int ci_outer = 0; ci_outer < 8; ++ci_outer) {
    __syncthreads();
    if (((((int)threadIdx.y) * 98) + ((int)threadIdx.x)) < 512) {
      if (((int)threadIdx.y) < 6) {
        placeholder_shared[(((((int)threadIdx.y) * 98) + ((int)threadIdx.x)))] = placeholder[((((((((int)blockIdx.z) * 16384) + (ci_outer * 2048)) + ((((((int)threadIdx.y) * 98) + ((int)threadIdx.x)) >> 5) * 128)) + (((int)blockIdx.y) * 32)) + (((((int)threadIdx.y) * 98) + ((int)threadIdx.x)) & 31)))];
      }
    }
    data_pack_shared[(((((int)threadIdx.y) * 98) + ((int)threadIdx.x)))] = data_pack[(((((((int)blockIdx.z) * 25088) + (ci_outer * 3136)) + (((int)threadIdx.y) * 98)) + ((int)threadIdx.x)))];
    data_pack_shared[((((((int)threadIdx.y) * 98) + ((int)threadIdx.x)) + 784))] = data_pack[((((((((int)blockIdx.z) * 25088) + (ci_outer * 3136)) + (((int)threadIdx.y) * 98)) + ((int)threadIdx.x)) + 784))];
    data_pack_shared[((((((int)threadIdx.y) * 98) + ((int)threadIdx.x)) + 1568))] = data_pack[((((((((int)blockIdx.z) * 25088) + (ci_outer * 3136)) + (((int)threadIdx.y) * 98)) + ((int)threadIdx.x)) + 1568))];
    data_pack_shared[((((((int)threadIdx.y) * 98) + ((int)threadIdx.x)) + 2352))] = data_pack[((((((((int)blockIdx.z) * 25088) + (ci_outer * 3136)) + (((int)threadIdx.y) * 98)) + ((int)threadIdx.x)) + 2352))];
    __syncthreads();
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[((((int)threadIdx.y) * 2))], data_pack_shared[(((int)threadIdx.x))], bgemm_local[(0)]);
    bgemm_local[(4)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 16))], data_pack_shared[(((int)threadIdx.x))], bgemm_local[(4)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[((((int)threadIdx.y) * 2))], data_pack_shared[((((int)threadIdx.x) + 98))], bgemm_local[(2)]);
    bgemm_local[(6)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 16))], data_pack_shared[((((int)threadIdx.x) + 98))], bgemm_local[(6)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 1))], data_pack_shared[(((int)threadIdx.x))], bgemm_local[(1)]);
    bgemm_local[(5)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 17))], data_pack_shared[(((int)threadIdx.x))], bgemm_local[(5)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 1))], data_pack_shared[((((int)threadIdx.x) + 98))], bgemm_local[(3)]);
    bgemm_local[(7)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 17))], data_pack_shared[((((int)threadIdx.x) + 98))], bgemm_local[(7)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 32))], data_pack_shared[((((int)threadIdx.x) + 196))], bgemm_local[(0)]);
    bgemm_local[(4)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 48))], data_pack_shared[((((int)threadIdx.x) + 196))], bgemm_local[(4)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 32))], data_pack_shared[((((int)threadIdx.x) + 294))], bgemm_local[(2)]);
    bgemm_local[(6)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 48))], data_pack_shared[((((int)threadIdx.x) + 294))], bgemm_local[(6)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 33))], data_pack_shared[((((int)threadIdx.x) + 196))], bgemm_local[(1)]);
    bgemm_local[(5)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 49))], data_pack_shared[((((int)threadIdx.x) + 196))], bgemm_local[(5)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 33))], data_pack_shared[((((int)threadIdx.x) + 294))], bgemm_local[(3)]);
    bgemm_local[(7)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 49))], data_pack_shared[((((int)threadIdx.x) + 294))], bgemm_local[(7)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 64))], data_pack_shared[((((int)threadIdx.x) + 392))], bgemm_local[(0)]);
    bgemm_local[(4)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 80))], data_pack_shared[((((int)threadIdx.x) + 392))], bgemm_local[(4)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 64))], data_pack_shared[((((int)threadIdx.x) + 490))], bgemm_local[(2)]);
    bgemm_local[(6)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 80))], data_pack_shared[((((int)threadIdx.x) + 490))], bgemm_local[(6)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 65))], data_pack_shared[((((int)threadIdx.x) + 392))], bgemm_local[(1)]);
    bgemm_local[(5)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 81))], data_pack_shared[((((int)threadIdx.x) + 392))], bgemm_local[(5)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 65))], data_pack_shared[((((int)threadIdx.x) + 490))], bgemm_local[(3)]);
    bgemm_local[(7)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 81))], data_pack_shared[((((int)threadIdx.x) + 490))], bgemm_local[(7)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 96))], data_pack_shared[((((int)threadIdx.x) + 588))], bgemm_local[(0)]);
    bgemm_local[(4)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 112))], data_pack_shared[((((int)threadIdx.x) + 588))], bgemm_local[(4)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 96))], data_pack_shared[((((int)threadIdx.x) + 686))], bgemm_local[(2)]);
    bgemm_local[(6)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 112))], data_pack_shared[((((int)threadIdx.x) + 686))], bgemm_local[(6)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 97))], data_pack_shared[((((int)threadIdx.x) + 588))], bgemm_local[(1)]);
    bgemm_local[(5)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 113))], data_pack_shared[((((int)threadIdx.x) + 588))], bgemm_local[(5)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 97))], data_pack_shared[((((int)threadIdx.x) + 686))], bgemm_local[(3)]);
    bgemm_local[(7)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 113))], data_pack_shared[((((int)threadIdx.x) + 686))], bgemm_local[(7)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 128))], data_pack_shared[((((int)threadIdx.x) + 784))], bgemm_local[(0)]);
    bgemm_local[(4)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 144))], data_pack_shared[((((int)threadIdx.x) + 784))], bgemm_local[(4)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 128))], data_pack_shared[((((int)threadIdx.x) + 882))], bgemm_local[(2)]);
    bgemm_local[(6)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 144))], data_pack_shared[((((int)threadIdx.x) + 882))], bgemm_local[(6)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 129))], data_pack_shared[((((int)threadIdx.x) + 784))], bgemm_local[(1)]);
    bgemm_local[(5)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 145))], data_pack_shared[((((int)threadIdx.x) + 784))], bgemm_local[(5)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 129))], data_pack_shared[((((int)threadIdx.x) + 882))], bgemm_local[(3)]);
    bgemm_local[(7)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 145))], data_pack_shared[((((int)threadIdx.x) + 882))], bgemm_local[(7)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 160))], data_pack_shared[((((int)threadIdx.x) + 980))], bgemm_local[(0)]);
    bgemm_local[(4)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 176))], data_pack_shared[((((int)threadIdx.x) + 980))], bgemm_local[(4)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 160))], data_pack_shared[((((int)threadIdx.x) + 1078))], bgemm_local[(2)]);
    bgemm_local[(6)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 176))], data_pack_shared[((((int)threadIdx.x) + 1078))], bgemm_local[(6)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 161))], data_pack_shared[((((int)threadIdx.x) + 980))], bgemm_local[(1)]);
    bgemm_local[(5)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 177))], data_pack_shared[((((int)threadIdx.x) + 980))], bgemm_local[(5)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 161))], data_pack_shared[((((int)threadIdx.x) + 1078))], bgemm_local[(3)]);
    bgemm_local[(7)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 177))], data_pack_shared[((((int)threadIdx.x) + 1078))], bgemm_local[(7)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 192))], data_pack_shared[((((int)threadIdx.x) + 1176))], bgemm_local[(0)]);
    bgemm_local[(4)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 208))], data_pack_shared[((((int)threadIdx.x) + 1176))], bgemm_local[(4)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 192))], data_pack_shared[((((int)threadIdx.x) + 1274))], bgemm_local[(2)]);
    bgemm_local[(6)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 208))], data_pack_shared[((((int)threadIdx.x) + 1274))], bgemm_local[(6)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 193))], data_pack_shared[((((int)threadIdx.x) + 1176))], bgemm_local[(1)]);
    bgemm_local[(5)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 209))], data_pack_shared[((((int)threadIdx.x) + 1176))], bgemm_local[(5)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 193))], data_pack_shared[((((int)threadIdx.x) + 1274))], bgemm_local[(3)]);
    bgemm_local[(7)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 209))], data_pack_shared[((((int)threadIdx.x) + 1274))], bgemm_local[(7)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 224))], data_pack_shared[((((int)threadIdx.x) + 1372))], bgemm_local[(0)]);
    bgemm_local[(4)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 240))], data_pack_shared[((((int)threadIdx.x) + 1372))], bgemm_local[(4)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 224))], data_pack_shared[((((int)threadIdx.x) + 1470))], bgemm_local[(2)]);
    bgemm_local[(6)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 240))], data_pack_shared[((((int)threadIdx.x) + 1470))], bgemm_local[(6)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 225))], data_pack_shared[((((int)threadIdx.x) + 1372))], bgemm_local[(1)]);
    bgemm_local[(5)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 241))], data_pack_shared[((((int)threadIdx.x) + 1372))], bgemm_local[(5)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 225))], data_pack_shared[((((int)threadIdx.x) + 1470))], bgemm_local[(3)]);
    bgemm_local[(7)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 241))], data_pack_shared[((((int)threadIdx.x) + 1470))], bgemm_local[(7)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 256))], data_pack_shared[((((int)threadIdx.x) + 1568))], bgemm_local[(0)]);
    bgemm_local[(4)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 272))], data_pack_shared[((((int)threadIdx.x) + 1568))], bgemm_local[(4)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 256))], data_pack_shared[((((int)threadIdx.x) + 1666))], bgemm_local[(2)]);
    bgemm_local[(6)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 272))], data_pack_shared[((((int)threadIdx.x) + 1666))], bgemm_local[(6)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 257))], data_pack_shared[((((int)threadIdx.x) + 1568))], bgemm_local[(1)]);
    bgemm_local[(5)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 273))], data_pack_shared[((((int)threadIdx.x) + 1568))], bgemm_local[(5)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 257))], data_pack_shared[((((int)threadIdx.x) + 1666))], bgemm_local[(3)]);
    bgemm_local[(7)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 273))], data_pack_shared[((((int)threadIdx.x) + 1666))], bgemm_local[(7)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 288))], data_pack_shared[((((int)threadIdx.x) + 1764))], bgemm_local[(0)]);
    bgemm_local[(4)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 304))], data_pack_shared[((((int)threadIdx.x) + 1764))], bgemm_local[(4)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 288))], data_pack_shared[((((int)threadIdx.x) + 1862))], bgemm_local[(2)]);
    bgemm_local[(6)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 304))], data_pack_shared[((((int)threadIdx.x) + 1862))], bgemm_local[(6)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 289))], data_pack_shared[((((int)threadIdx.x) + 1764))], bgemm_local[(1)]);
    bgemm_local[(5)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 305))], data_pack_shared[((((int)threadIdx.x) + 1764))], bgemm_local[(5)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 289))], data_pack_shared[((((int)threadIdx.x) + 1862))], bgemm_local[(3)]);
    bgemm_local[(7)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 305))], data_pack_shared[((((int)threadIdx.x) + 1862))], bgemm_local[(7)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 320))], data_pack_shared[((((int)threadIdx.x) + 1960))], bgemm_local[(0)]);
    bgemm_local[(4)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 336))], data_pack_shared[((((int)threadIdx.x) + 1960))], bgemm_local[(4)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 320))], data_pack_shared[((((int)threadIdx.x) + 2058))], bgemm_local[(2)]);
    bgemm_local[(6)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 336))], data_pack_shared[((((int)threadIdx.x) + 2058))], bgemm_local[(6)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 321))], data_pack_shared[((((int)threadIdx.x) + 1960))], bgemm_local[(1)]);
    bgemm_local[(5)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 337))], data_pack_shared[((((int)threadIdx.x) + 1960))], bgemm_local[(5)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 321))], data_pack_shared[((((int)threadIdx.x) + 2058))], bgemm_local[(3)]);
    bgemm_local[(7)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 337))], data_pack_shared[((((int)threadIdx.x) + 2058))], bgemm_local[(7)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 352))], data_pack_shared[((((int)threadIdx.x) + 2156))], bgemm_local[(0)]);
    bgemm_local[(4)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 368))], data_pack_shared[((((int)threadIdx.x) + 2156))], bgemm_local[(4)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 352))], data_pack_shared[((((int)threadIdx.x) + 2254))], bgemm_local[(2)]);
    bgemm_local[(6)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 368))], data_pack_shared[((((int)threadIdx.x) + 2254))], bgemm_local[(6)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 353))], data_pack_shared[((((int)threadIdx.x) + 2156))], bgemm_local[(1)]);
    bgemm_local[(5)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 369))], data_pack_shared[((((int)threadIdx.x) + 2156))], bgemm_local[(5)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 353))], data_pack_shared[((((int)threadIdx.x) + 2254))], bgemm_local[(3)]);
    bgemm_local[(7)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 369))], data_pack_shared[((((int)threadIdx.x) + 2254))], bgemm_local[(7)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 384))], data_pack_shared[((((int)threadIdx.x) + 2352))], bgemm_local[(0)]);
    bgemm_local[(4)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 400))], data_pack_shared[((((int)threadIdx.x) + 2352))], bgemm_local[(4)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 384))], data_pack_shared[((((int)threadIdx.x) + 2450))], bgemm_local[(2)]);
    bgemm_local[(6)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 400))], data_pack_shared[((((int)threadIdx.x) + 2450))], bgemm_local[(6)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 385))], data_pack_shared[((((int)threadIdx.x) + 2352))], bgemm_local[(1)]);
    bgemm_local[(5)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 401))], data_pack_shared[((((int)threadIdx.x) + 2352))], bgemm_local[(5)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 385))], data_pack_shared[((((int)threadIdx.x) + 2450))], bgemm_local[(3)]);
    bgemm_local[(7)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 401))], data_pack_shared[((((int)threadIdx.x) + 2450))], bgemm_local[(7)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 416))], data_pack_shared[((((int)threadIdx.x) + 2548))], bgemm_local[(0)]);
    bgemm_local[(4)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 432))], data_pack_shared[((((int)threadIdx.x) + 2548))], bgemm_local[(4)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 416))], data_pack_shared[((((int)threadIdx.x) + 2646))], bgemm_local[(2)]);
    bgemm_local[(6)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 432))], data_pack_shared[((((int)threadIdx.x) + 2646))], bgemm_local[(6)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 417))], data_pack_shared[((((int)threadIdx.x) + 2548))], bgemm_local[(1)]);
    bgemm_local[(5)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 433))], data_pack_shared[((((int)threadIdx.x) + 2548))], bgemm_local[(5)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 417))], data_pack_shared[((((int)threadIdx.x) + 2646))], bgemm_local[(3)]);
    bgemm_local[(7)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 433))], data_pack_shared[((((int)threadIdx.x) + 2646))], bgemm_local[(7)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 448))], data_pack_shared[((((int)threadIdx.x) + 2744))], bgemm_local[(0)]);
    bgemm_local[(4)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 464))], data_pack_shared[((((int)threadIdx.x) + 2744))], bgemm_local[(4)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 448))], data_pack_shared[((((int)threadIdx.x) + 2842))], bgemm_local[(2)]);
    bgemm_local[(6)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 464))], data_pack_shared[((((int)threadIdx.x) + 2842))], bgemm_local[(6)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 449))], data_pack_shared[((((int)threadIdx.x) + 2744))], bgemm_local[(1)]);
    bgemm_local[(5)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 465))], data_pack_shared[((((int)threadIdx.x) + 2744))], bgemm_local[(5)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 449))], data_pack_shared[((((int)threadIdx.x) + 2842))], bgemm_local[(3)]);
    bgemm_local[(7)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 465))], data_pack_shared[((((int)threadIdx.x) + 2842))], bgemm_local[(7)]);
    bgemm_local[(0)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 480))], data_pack_shared[((((int)threadIdx.x) + 2940))], bgemm_local[(0)]);
    bgemm_local[(4)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 496))], data_pack_shared[((((int)threadIdx.x) + 2940))], bgemm_local[(4)]);
    bgemm_local[(2)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 480))], data_pack_shared[((((int)threadIdx.x) + 3038))], bgemm_local[(2)]);
    bgemm_local[(6)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 496))], data_pack_shared[((((int)threadIdx.x) + 3038))], bgemm_local[(6)]);
    bgemm_local[(1)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 481))], data_pack_shared[((((int)threadIdx.x) + 2940))], bgemm_local[(1)]);
    bgemm_local[(5)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 497))], data_pack_shared[((((int)threadIdx.x) + 2940))], bgemm_local[(5)]);
    bgemm_local[(3)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 481))], data_pack_shared[((((int)threadIdx.x) + 3038))], bgemm_local[(3)]);
    bgemm_local[(7)] = __ocml_fma_f32(placeholder_shared[(((((int)threadIdx.y) * 2) + 497))], data_pack_shared[((((int)threadIdx.x) + 3038))], bgemm_local[(7)]);
  }
  bgemm[(((((((int)blockIdx.z) * 25088) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 392)) + ((int)threadIdx.x)))] = bgemm_local[(0)];
  bgemm[((((((((int)blockIdx.z) * 25088) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 392)) + ((int)threadIdx.x)) + 3136))] = bgemm_local[(4)];
  bgemm[((((((((int)blockIdx.z) * 25088) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 392)) + ((int)threadIdx.x)) + 98))] = bgemm_local[(2)];
  bgemm[((((((((int)blockIdx.z) * 25088) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 392)) + ((int)threadIdx.x)) + 3234))] = bgemm_local[(6)];
  bgemm[((((((((int)blockIdx.z) * 25088) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 392)) + ((int)threadIdx.x)) + 196))] = bgemm_local[(1)];
  bgemm[((((((((int)blockIdx.z) * 25088) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 392)) + ((int)threadIdx.x)) + 3332))] = bgemm_local[(5)];
  bgemm[((((((((int)blockIdx.z) * 25088) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 392)) + ((int)threadIdx.x)) + 294))] = bgemm_local[(3)];
  bgemm[((((((((int)blockIdx.z) * 25088) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 392)) + ((int)threadIdx.x)) + 3430))] = bgemm_local[(7)];
}

extern "C" __global__ void fused_nn_conv2d_add_1_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_add, float* __restrict__ placeholder2) {
  float compute[4];
  __shared__ float pad_temp_shared[1568];
  __shared__ float placeholder_shared[512];
  compute[(0)] = 0.000000e+00f;
  compute[(2)] = 0.000000e+00f;
  compute[(1)] = 0.000000e+00f;
  compute[(3)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 16; ++rc_outer) {
    __syncthreads();
    pad_temp_shared[((((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)))] = placeholder[((((((rc_outer * 3136) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)))];
    pad_temp_shared[(((((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)) + 1))] = placeholder[(((((((rc_outer * 3136) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)) + 1))];
    if (((((int)threadIdx.z) * 2) + (((((int)threadIdx.y) * 5) + ((int)threadIdx.x)) >> 4)) < 32) {
      if ((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 5)) + ((int)threadIdx.x)) < 512) {
        if (((((int)threadIdx.y) * 5) + ((int)threadIdx.x)) < 32) {
          if (((int)threadIdx.x) < 5) {
            placeholder_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 5)) + ((int)threadIdx.x)))] = placeholder1[((((((((int)blockIdx.z) * 8192) + (((int)threadIdx.z) * 512)) + ((((((int)threadIdx.y) * 5) + ((int)threadIdx.x)) >> 4) * 256)) + (rc_outer * 16)) + (((((int)threadIdx.y) * 5) + ((int)threadIdx.x)) & 15)))];
          }
        }
      }
    }
    __syncthreads();
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)))], placeholder_shared[((((int)threadIdx.z) * 16))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)))], placeholder_shared[(((((int)threadIdx.z) * 16) + 256))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 1))], placeholder_shared[((((int)threadIdx.z) * 16))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 1))], placeholder_shared[(((((int)threadIdx.z) * 16) + 256))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 98))], placeholder_shared[(((((int)threadIdx.z) * 16) + 1))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 98))], placeholder_shared[(((((int)threadIdx.z) * 16) + 257))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 99))], placeholder_shared[(((((int)threadIdx.z) * 16) + 1))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 99))], placeholder_shared[(((((int)threadIdx.z) * 16) + 257))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 196))], placeholder_shared[(((((int)threadIdx.z) * 16) + 2))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 196))], placeholder_shared[(((((int)threadIdx.z) * 16) + 258))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 197))], placeholder_shared[(((((int)threadIdx.z) * 16) + 2))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 197))], placeholder_shared[(((((int)threadIdx.z) * 16) + 258))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 294))], placeholder_shared[(((((int)threadIdx.z) * 16) + 3))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 294))], placeholder_shared[(((((int)threadIdx.z) * 16) + 259))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 295))], placeholder_shared[(((((int)threadIdx.z) * 16) + 3))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 295))], placeholder_shared[(((((int)threadIdx.z) * 16) + 259))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 392))], placeholder_shared[(((((int)threadIdx.z) * 16) + 4))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 392))], placeholder_shared[(((((int)threadIdx.z) * 16) + 260))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 393))], placeholder_shared[(((((int)threadIdx.z) * 16) + 4))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 393))], placeholder_shared[(((((int)threadIdx.z) * 16) + 260))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 490))], placeholder_shared[(((((int)threadIdx.z) * 16) + 5))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 490))], placeholder_shared[(((((int)threadIdx.z) * 16) + 261))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 491))], placeholder_shared[(((((int)threadIdx.z) * 16) + 5))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 491))], placeholder_shared[(((((int)threadIdx.z) * 16) + 261))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 588))], placeholder_shared[(((((int)threadIdx.z) * 16) + 6))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 588))], placeholder_shared[(((((int)threadIdx.z) * 16) + 262))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 589))], placeholder_shared[(((((int)threadIdx.z) * 16) + 6))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 589))], placeholder_shared[(((((int)threadIdx.z) * 16) + 262))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 686))], placeholder_shared[(((((int)threadIdx.z) * 16) + 7))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 686))], placeholder_shared[(((((int)threadIdx.z) * 16) + 263))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 687))], placeholder_shared[(((((int)threadIdx.z) * 16) + 7))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 687))], placeholder_shared[(((((int)threadIdx.z) * 16) + 263))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 784))], placeholder_shared[(((((int)threadIdx.z) * 16) + 8))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 784))], placeholder_shared[(((((int)threadIdx.z) * 16) + 264))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 785))], placeholder_shared[(((((int)threadIdx.z) * 16) + 8))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 785))], placeholder_shared[(((((int)threadIdx.z) * 16) + 264))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 882))], placeholder_shared[(((((int)threadIdx.z) * 16) + 9))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 882))], placeholder_shared[(((((int)threadIdx.z) * 16) + 265))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 883))], placeholder_shared[(((((int)threadIdx.z) * 16) + 9))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 883))], placeholder_shared[(((((int)threadIdx.z) * 16) + 265))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 980))], placeholder_shared[(((((int)threadIdx.z) * 16) + 10))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 980))], placeholder_shared[(((((int)threadIdx.z) * 16) + 266))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 981))], placeholder_shared[(((((int)threadIdx.z) * 16) + 10))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 981))], placeholder_shared[(((((int)threadIdx.z) * 16) + 266))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 1078))], placeholder_shared[(((((int)threadIdx.z) * 16) + 11))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 1078))], placeholder_shared[(((((int)threadIdx.z) * 16) + 267))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 1079))], placeholder_shared[(((((int)threadIdx.z) * 16) + 11))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 1079))], placeholder_shared[(((((int)threadIdx.z) * 16) + 267))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 1176))], placeholder_shared[(((((int)threadIdx.z) * 16) + 12))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 1176))], placeholder_shared[(((((int)threadIdx.z) * 16) + 268))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 1177))], placeholder_shared[(((((int)threadIdx.z) * 16) + 12))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 1177))], placeholder_shared[(((((int)threadIdx.z) * 16) + 268))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 1274))], placeholder_shared[(((((int)threadIdx.z) * 16) + 13))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 1274))], placeholder_shared[(((((int)threadIdx.z) * 16) + 269))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 1275))], placeholder_shared[(((((int)threadIdx.z) * 16) + 13))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 1275))], placeholder_shared[(((((int)threadIdx.z) * 16) + 269))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 1372))], placeholder_shared[(((((int)threadIdx.z) * 16) + 14))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 1372))], placeholder_shared[(((((int)threadIdx.z) * 16) + 270))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 1373))], placeholder_shared[(((((int)threadIdx.z) * 16) + 14))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 1373))], placeholder_shared[(((((int)threadIdx.z) * 16) + 270))], compute[(3)]);
    compute[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 1470))], placeholder_shared[(((((int)threadIdx.z) * 16) + 15))], compute[(0)]);
    compute[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 1470))], placeholder_shared[(((((int)threadIdx.z) * 16) + 271))], compute[(2)]);
    compute[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 1471))], placeholder_shared[(((((int)threadIdx.z) * 16) + 15))], compute[(1)]);
    compute[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 1471))], placeholder_shared[(((((int)threadIdx.z) * 16) + 271))], compute[(3)]);
  }
  T_add[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)))] = (compute[(0)] + placeholder2[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)))]);
  T_add[(((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)) + 3136))] = (compute[(2)] + placeholder2[(((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)) + 3136))]);
  T_add[(((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)) + 1))] = (compute[(1)] + placeholder2[(((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)) + 1))]);
  T_add[(((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)) + 3137))] = (compute[(3)] + placeholder2[(((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)) + 3137))]);
}

extern "C" __global__ void fused_nn_conv2d_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ compute) {
  float compute_local[8];
  __shared__ float pad_temp_shared[512];
  __shared__ float placeholder_shared[1024];
  compute_local[(0)] = 0.000000e+00f;
  compute_local[(2)] = 0.000000e+00f;
  compute_local[(4)] = 0.000000e+00f;
  compute_local[(6)] = 0.000000e+00f;
  compute_local[(1)] = 0.000000e+00f;
  compute_local[(3)] = 0.000000e+00f;
  compute_local[(5)] = 0.000000e+00f;
  compute_local[(7)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 4; ++rc_outer) {
    __syncthreads();
    pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)))] = placeholder[(((((((rc_outer * 50176) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)))];
    pad_temp_shared[(((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1))] = placeholder[((((((((rc_outer * 50176) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 1))];
    placeholder_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))] = placeholder1[((((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 64)) + (rc_outer * 16)) + (((int)threadIdx.x) * 4)))];
    placeholder_shared[(((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 1))] = placeholder1[(((((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 64)) + (rc_outer * 16)) + (((int)threadIdx.x) * 4)) + 1))];
    placeholder_shared[(((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 2))] = placeholder1[(((((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 64)) + (rc_outer * 16)) + (((int)threadIdx.x) * 4)) + 2))];
    placeholder_shared[(((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + 3))] = placeholder1[(((((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 64)) + (rc_outer * 16)) + (((int)threadIdx.x) * 4)) + 3))];
    __syncthreads();
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)))], placeholder_shared[((((int)threadIdx.z) * 16))], compute_local[(0)]);
    compute_local[(2)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)))], placeholder_shared[(((((int)threadIdx.z) * 16) + 256))], compute_local[(2)]);
    compute_local[(4)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)))], placeholder_shared[(((((int)threadIdx.z) * 16) + 512))], compute_local[(4)]);
    compute_local[(6)] = __ocml_fma_f32(pad_temp_shared[(((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)))], placeholder_shared[(((((int)threadIdx.z) * 16) + 768))], compute_local[(6)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 1))], placeholder_shared[((((int)threadIdx.z) * 16))], compute_local[(1)]);
    compute_local[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 1))], placeholder_shared[(((((int)threadIdx.z) * 16) + 256))], compute_local[(3)]);
    compute_local[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 1))], placeholder_shared[(((((int)threadIdx.z) * 16) + 512))], compute_local[(5)]);
    compute_local[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 1))], placeholder_shared[(((((int)threadIdx.z) * 16) + 768))], compute_local[(7)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 32))], placeholder_shared[(((((int)threadIdx.z) * 16) + 1))], compute_local[(0)]);
    compute_local[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 32))], placeholder_shared[(((((int)threadIdx.z) * 16) + 257))], compute_local[(2)]);
    compute_local[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 32))], placeholder_shared[(((((int)threadIdx.z) * 16) + 513))], compute_local[(4)]);
    compute_local[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 32))], placeholder_shared[(((((int)threadIdx.z) * 16) + 769))], compute_local[(6)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 33))], placeholder_shared[(((((int)threadIdx.z) * 16) + 1))], compute_local[(1)]);
    compute_local[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 33))], placeholder_shared[(((((int)threadIdx.z) * 16) + 257))], compute_local[(3)]);
    compute_local[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 33))], placeholder_shared[(((((int)threadIdx.z) * 16) + 513))], compute_local[(5)]);
    compute_local[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 33))], placeholder_shared[(((((int)threadIdx.z) * 16) + 769))], compute_local[(7)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 64))], placeholder_shared[(((((int)threadIdx.z) * 16) + 2))], compute_local[(0)]);
    compute_local[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 64))], placeholder_shared[(((((int)threadIdx.z) * 16) + 258))], compute_local[(2)]);
    compute_local[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 64))], placeholder_shared[(((((int)threadIdx.z) * 16) + 514))], compute_local[(4)]);
    compute_local[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 64))], placeholder_shared[(((((int)threadIdx.z) * 16) + 770))], compute_local[(6)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 65))], placeholder_shared[(((((int)threadIdx.z) * 16) + 2))], compute_local[(1)]);
    compute_local[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 65))], placeholder_shared[(((((int)threadIdx.z) * 16) + 258))], compute_local[(3)]);
    compute_local[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 65))], placeholder_shared[(((((int)threadIdx.z) * 16) + 514))], compute_local[(5)]);
    compute_local[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 65))], placeholder_shared[(((((int)threadIdx.z) * 16) + 770))], compute_local[(7)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 96))], placeholder_shared[(((((int)threadIdx.z) * 16) + 3))], compute_local[(0)]);
    compute_local[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 96))], placeholder_shared[(((((int)threadIdx.z) * 16) + 259))], compute_local[(2)]);
    compute_local[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 96))], placeholder_shared[(((((int)threadIdx.z) * 16) + 515))], compute_local[(4)]);
    compute_local[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 96))], placeholder_shared[(((((int)threadIdx.z) * 16) + 771))], compute_local[(6)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 97))], placeholder_shared[(((((int)threadIdx.z) * 16) + 3))], compute_local[(1)]);
    compute_local[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 97))], placeholder_shared[(((((int)threadIdx.z) * 16) + 259))], compute_local[(3)]);
    compute_local[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 97))], placeholder_shared[(((((int)threadIdx.z) * 16) + 515))], compute_local[(5)]);
    compute_local[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 97))], placeholder_shared[(((((int)threadIdx.z) * 16) + 771))], compute_local[(7)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 128))], placeholder_shared[(((((int)threadIdx.z) * 16) + 4))], compute_local[(0)]);
    compute_local[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 128))], placeholder_shared[(((((int)threadIdx.z) * 16) + 260))], compute_local[(2)]);
    compute_local[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 128))], placeholder_shared[(((((int)threadIdx.z) * 16) + 516))], compute_local[(4)]);
    compute_local[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 128))], placeholder_shared[(((((int)threadIdx.z) * 16) + 772))], compute_local[(6)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 129))], placeholder_shared[(((((int)threadIdx.z) * 16) + 4))], compute_local[(1)]);
    compute_local[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 129))], placeholder_shared[(((((int)threadIdx.z) * 16) + 260))], compute_local[(3)]);
    compute_local[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 129))], placeholder_shared[(((((int)threadIdx.z) * 16) + 516))], compute_local[(5)]);
    compute_local[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 129))], placeholder_shared[(((((int)threadIdx.z) * 16) + 772))], compute_local[(7)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 160))], placeholder_shared[(((((int)threadIdx.z) * 16) + 5))], compute_local[(0)]);
    compute_local[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 160))], placeholder_shared[(((((int)threadIdx.z) * 16) + 261))], compute_local[(2)]);
    compute_local[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 160))], placeholder_shared[(((((int)threadIdx.z) * 16) + 517))], compute_local[(4)]);
    compute_local[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 160))], placeholder_shared[(((((int)threadIdx.z) * 16) + 773))], compute_local[(6)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 161))], placeholder_shared[(((((int)threadIdx.z) * 16) + 5))], compute_local[(1)]);
    compute_local[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 161))], placeholder_shared[(((((int)threadIdx.z) * 16) + 261))], compute_local[(3)]);
    compute_local[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 161))], placeholder_shared[(((((int)threadIdx.z) * 16) + 517))], compute_local[(5)]);
    compute_local[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 161))], placeholder_shared[(((((int)threadIdx.z) * 16) + 773))], compute_local[(7)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 192))], placeholder_shared[(((((int)threadIdx.z) * 16) + 6))], compute_local[(0)]);
    compute_local[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 192))], placeholder_shared[(((((int)threadIdx.z) * 16) + 262))], compute_local[(2)]);
    compute_local[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 192))], placeholder_shared[(((((int)threadIdx.z) * 16) + 518))], compute_local[(4)]);
    compute_local[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 192))], placeholder_shared[(((((int)threadIdx.z) * 16) + 774))], compute_local[(6)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 193))], placeholder_shared[(((((int)threadIdx.z) * 16) + 6))], compute_local[(1)]);
    compute_local[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 193))], placeholder_shared[(((((int)threadIdx.z) * 16) + 262))], compute_local[(3)]);
    compute_local[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 193))], placeholder_shared[(((((int)threadIdx.z) * 16) + 518))], compute_local[(5)]);
    compute_local[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 193))], placeholder_shared[(((((int)threadIdx.z) * 16) + 774))], compute_local[(7)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 224))], placeholder_shared[(((((int)threadIdx.z) * 16) + 7))], compute_local[(0)]);
    compute_local[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 224))], placeholder_shared[(((((int)threadIdx.z) * 16) + 263))], compute_local[(2)]);
    compute_local[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 224))], placeholder_shared[(((((int)threadIdx.z) * 16) + 519))], compute_local[(4)]);
    compute_local[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 224))], placeholder_shared[(((((int)threadIdx.z) * 16) + 775))], compute_local[(6)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 225))], placeholder_shared[(((((int)threadIdx.z) * 16) + 7))], compute_local[(1)]);
    compute_local[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 225))], placeholder_shared[(((((int)threadIdx.z) * 16) + 263))], compute_local[(3)]);
    compute_local[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 225))], placeholder_shared[(((((int)threadIdx.z) * 16) + 519))], compute_local[(5)]);
    compute_local[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 225))], placeholder_shared[(((((int)threadIdx.z) * 16) + 775))], compute_local[(7)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 256))], placeholder_shared[(((((int)threadIdx.z) * 16) + 8))], compute_local[(0)]);
    compute_local[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 256))], placeholder_shared[(((((int)threadIdx.z) * 16) + 264))], compute_local[(2)]);
    compute_local[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 256))], placeholder_shared[(((((int)threadIdx.z) * 16) + 520))], compute_local[(4)]);
    compute_local[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 256))], placeholder_shared[(((((int)threadIdx.z) * 16) + 776))], compute_local[(6)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 257))], placeholder_shared[(((((int)threadIdx.z) * 16) + 8))], compute_local[(1)]);
    compute_local[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 257))], placeholder_shared[(((((int)threadIdx.z) * 16) + 264))], compute_local[(3)]);
    compute_local[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 257))], placeholder_shared[(((((int)threadIdx.z) * 16) + 520))], compute_local[(5)]);
    compute_local[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 257))], placeholder_shared[(((((int)threadIdx.z) * 16) + 776))], compute_local[(7)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 288))], placeholder_shared[(((((int)threadIdx.z) * 16) + 9))], compute_local[(0)]);
    compute_local[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 288))], placeholder_shared[(((((int)threadIdx.z) * 16) + 265))], compute_local[(2)]);
    compute_local[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 288))], placeholder_shared[(((((int)threadIdx.z) * 16) + 521))], compute_local[(4)]);
    compute_local[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 288))], placeholder_shared[(((((int)threadIdx.z) * 16) + 777))], compute_local[(6)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 289))], placeholder_shared[(((((int)threadIdx.z) * 16) + 9))], compute_local[(1)]);
    compute_local[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 289))], placeholder_shared[(((((int)threadIdx.z) * 16) + 265))], compute_local[(3)]);
    compute_local[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 289))], placeholder_shared[(((((int)threadIdx.z) * 16) + 521))], compute_local[(5)]);
    compute_local[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 289))], placeholder_shared[(((((int)threadIdx.z) * 16) + 777))], compute_local[(7)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 320))], placeholder_shared[(((((int)threadIdx.z) * 16) + 10))], compute_local[(0)]);
    compute_local[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 320))], placeholder_shared[(((((int)threadIdx.z) * 16) + 266))], compute_local[(2)]);
    compute_local[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 320))], placeholder_shared[(((((int)threadIdx.z) * 16) + 522))], compute_local[(4)]);
    compute_local[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 320))], placeholder_shared[(((((int)threadIdx.z) * 16) + 778))], compute_local[(6)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 321))], placeholder_shared[(((((int)threadIdx.z) * 16) + 10))], compute_local[(1)]);
    compute_local[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 321))], placeholder_shared[(((((int)threadIdx.z) * 16) + 266))], compute_local[(3)]);
    compute_local[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 321))], placeholder_shared[(((((int)threadIdx.z) * 16) + 522))], compute_local[(5)]);
    compute_local[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 321))], placeholder_shared[(((((int)threadIdx.z) * 16) + 778))], compute_local[(7)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 352))], placeholder_shared[(((((int)threadIdx.z) * 16) + 11))], compute_local[(0)]);
    compute_local[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 352))], placeholder_shared[(((((int)threadIdx.z) * 16) + 267))], compute_local[(2)]);
    compute_local[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 352))], placeholder_shared[(((((int)threadIdx.z) * 16) + 523))], compute_local[(4)]);
    compute_local[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 352))], placeholder_shared[(((((int)threadIdx.z) * 16) + 779))], compute_local[(6)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 353))], placeholder_shared[(((((int)threadIdx.z) * 16) + 11))], compute_local[(1)]);
    compute_local[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 353))], placeholder_shared[(((((int)threadIdx.z) * 16) + 267))], compute_local[(3)]);
    compute_local[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 353))], placeholder_shared[(((((int)threadIdx.z) * 16) + 523))], compute_local[(5)]);
    compute_local[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 353))], placeholder_shared[(((((int)threadIdx.z) * 16) + 779))], compute_local[(7)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 384))], placeholder_shared[(((((int)threadIdx.z) * 16) + 12))], compute_local[(0)]);
    compute_local[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 384))], placeholder_shared[(((((int)threadIdx.z) * 16) + 268))], compute_local[(2)]);
    compute_local[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 384))], placeholder_shared[(((((int)threadIdx.z) * 16) + 524))], compute_local[(4)]);
    compute_local[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 384))], placeholder_shared[(((((int)threadIdx.z) * 16) + 780))], compute_local[(6)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 385))], placeholder_shared[(((((int)threadIdx.z) * 16) + 12))], compute_local[(1)]);
    compute_local[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 385))], placeholder_shared[(((((int)threadIdx.z) * 16) + 268))], compute_local[(3)]);
    compute_local[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 385))], placeholder_shared[(((((int)threadIdx.z) * 16) + 524))], compute_local[(5)]);
    compute_local[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 385))], placeholder_shared[(((((int)threadIdx.z) * 16) + 780))], compute_local[(7)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 416))], placeholder_shared[(((((int)threadIdx.z) * 16) + 13))], compute_local[(0)]);
    compute_local[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 416))], placeholder_shared[(((((int)threadIdx.z) * 16) + 269))], compute_local[(2)]);
    compute_local[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 416))], placeholder_shared[(((((int)threadIdx.z) * 16) + 525))], compute_local[(4)]);
    compute_local[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 416))], placeholder_shared[(((((int)threadIdx.z) * 16) + 781))], compute_local[(6)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 417))], placeholder_shared[(((((int)threadIdx.z) * 16) + 13))], compute_local[(1)]);
    compute_local[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 417))], placeholder_shared[(((((int)threadIdx.z) * 16) + 269))], compute_local[(3)]);
    compute_local[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 417))], placeholder_shared[(((((int)threadIdx.z) * 16) + 525))], compute_local[(5)]);
    compute_local[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 417))], placeholder_shared[(((((int)threadIdx.z) * 16) + 781))], compute_local[(7)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 448))], placeholder_shared[(((((int)threadIdx.z) * 16) + 14))], compute_local[(0)]);
    compute_local[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 448))], placeholder_shared[(((((int)threadIdx.z) * 16) + 270))], compute_local[(2)]);
    compute_local[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 448))], placeholder_shared[(((((int)threadIdx.z) * 16) + 526))], compute_local[(4)]);
    compute_local[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 448))], placeholder_shared[(((((int)threadIdx.z) * 16) + 782))], compute_local[(6)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 449))], placeholder_shared[(((((int)threadIdx.z) * 16) + 14))], compute_local[(1)]);
    compute_local[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 449))], placeholder_shared[(((((int)threadIdx.z) * 16) + 270))], compute_local[(3)]);
    compute_local[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 449))], placeholder_shared[(((((int)threadIdx.z) * 16) + 526))], compute_local[(5)]);
    compute_local[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 449))], placeholder_shared[(((((int)threadIdx.z) * 16) + 782))], compute_local[(7)]);
    compute_local[(0)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 480))], placeholder_shared[(((((int)threadIdx.z) * 16) + 15))], compute_local[(0)]);
    compute_local[(2)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 480))], placeholder_shared[(((((int)threadIdx.z) * 16) + 271))], compute_local[(2)]);
    compute_local[(4)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 480))], placeholder_shared[(((((int)threadIdx.z) * 16) + 527))], compute_local[(4)]);
    compute_local[(6)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 480))], placeholder_shared[(((((int)threadIdx.z) * 16) + 783))], compute_local[(6)]);
    compute_local[(1)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 481))], placeholder_shared[(((((int)threadIdx.z) * 16) + 15))], compute_local[(1)]);
    compute_local[(3)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 481))], placeholder_shared[(((((int)threadIdx.z) * 16) + 271))], compute_local[(3)]);
    compute_local[(5)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 481))], placeholder_shared[(((((int)threadIdx.z) * 16) + 527))], compute_local[(5)]);
    compute_local[(7)] = __ocml_fma_f32(pad_temp_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + 481))], placeholder_shared[(((((int)threadIdx.z) * 16) + 783))], compute_local[(7)]);
  }
  compute[(((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)))] = compute_local[(0)];
  compute[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 50176))] = compute_local[(2)];
  compute[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 100352))] = compute_local[(4)];
  compute[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 150528))] = compute_local[(6)];
  compute[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 1))] = compute_local[(1)];
  compute[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 50177))] = compute_local[(3)];
  compute[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 100353))] = compute_local[(5)];
  compute[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 150529))] = compute_local[(7)];
}

