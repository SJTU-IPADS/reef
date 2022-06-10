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

__device__ void fused_nn_softmax_1_kernel0_device(float* __restrict__ T_softmax_maxelem, float* __restrict__ placeholder){
  if (((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) < 5760) {
    T_softmax_maxelem[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)))] = -3.402823e+38f;
  }
  for (int k = 0; k < 480; ++k) {
    if (((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) < 5760) {
      T_softmax_maxelem[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)))] = max(T_softmax_maxelem[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)))], placeholder[((((((int)blockIdx.x) * 122880) + (((int)threadIdx.x) * 480)) + k))]);
    }
  }
}

__device__ void fused_reshape_add_add_kernel0_device(float* __restrict__ T_add, float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ placeholder2){
  for (int ax0_ax1_fused_ax2_fused_outer = 0; ax0_ax1_fused_ax2_fused_outer < 4; ++ax0_ax1_fused_ax2_fused_outer) {
    if ((((ax0_ax1_fused_ax2_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) < 230400) {
      T_add[((((ax0_ax1_fused_ax2_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)))] = ((placeholder[((((ax0_ax1_fused_ax2_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)))] + placeholder1[(((((ax0_ax1_fused_ax2_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) / 480))]) + placeholder2[((((ax0_ax1_fused_ax2_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)))]);
    }
  }
}

__device__ void fused_nn_batch_matmul_4_kernel0_device(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ compute){
  float compute_local[8];
  __shared__ float placeholder_shared[720];
  __shared__ float placeholder_d_shared[300];
  float placeholder_shared_local[2];
  float placeholder_d_shared_local[4];
  for (int i_c_init = 0; i_c_init < 2; ++i_c_init) {
    for (int j_c_init = 0; j_c_init < 4; ++j_c_init) {
      compute_local[(((i_c_init * 4) + j_c_init))] = 0.000000e+00f;
    }
  }
  for (int k_outer = 0; k_outer < 32; ++k_outer) {
    __syncthreads();
    placeholder_shared[(((((int)threadIdx.y) * 30) + (((int)threadIdx.x) * 3)))] = placeholder[((((((((int)blockIdx.z) * 230400) + (((int)blockIdx.y) * 23040)) + (((int)threadIdx.y) * 960)) + (k_outer * 15)) + (((int)threadIdx.x) * 3)))];
    placeholder_shared[((((((int)threadIdx.y) * 30) + (((int)threadIdx.x) * 3)) + 1))] = placeholder[(((((((((int)blockIdx.z) * 230400) + (((int)blockIdx.y) * 23040)) + (((int)threadIdx.y) * 960)) + (k_outer * 15)) + (((int)threadIdx.x) * 3)) + 1))];
    placeholder_shared[((((((int)threadIdx.y) * 30) + (((int)threadIdx.x) * 3)) + 2))] = placeholder[(((((((((int)blockIdx.z) * 230400) + (((int)blockIdx.y) * 23040)) + (((int)threadIdx.y) * 960)) + (k_outer * 15)) + (((int)threadIdx.x) * 3)) + 2))];
    placeholder_shared[((((((int)threadIdx.y) * 30) + (((int)threadIdx.x) * 3)) + 15))] = placeholder[(((((((((int)blockIdx.z) * 230400) + (((int)blockIdx.y) * 23040)) + (((int)threadIdx.y) * 960)) + (k_outer * 15)) + (((int)threadIdx.x) * 3)) + 480))];
    placeholder_shared[((((((int)threadIdx.y) * 30) + (((int)threadIdx.x) * 3)) + 16))] = placeholder[(((((((((int)blockIdx.z) * 230400) + (((int)blockIdx.y) * 23040)) + (((int)threadIdx.y) * 960)) + (k_outer * 15)) + (((int)threadIdx.x) * 3)) + 481))];
    placeholder_shared[((((((int)threadIdx.y) * 30) + (((int)threadIdx.x) * 3)) + 17))] = placeholder[(((((((((int)blockIdx.z) * 230400) + (((int)blockIdx.y) * 23040)) + (((int)threadIdx.y) * 960)) + (k_outer * 15)) + (((int)threadIdx.x) * 3)) + 482))];
    if (((int)threadIdx.y) < 20) {
      placeholder_d_shared[(((((int)threadIdx.y) * 15) + (((int)threadIdx.x) * 3)))] = placeholder1[((((((((int)blockIdx.z) * 19200) + (((int)blockIdx.x) * 9600)) + (((int)threadIdx.y) * 480)) + (k_outer * 15)) + (((int)threadIdx.x) * 3)))];
    }
    if (((int)threadIdx.y) < 20) {
      placeholder_d_shared[((((((int)threadIdx.y) * 15) + (((int)threadIdx.x) * 3)) + 1))] = placeholder1[(((((((((int)blockIdx.z) * 19200) + (((int)blockIdx.x) * 9600)) + (((int)threadIdx.y) * 480)) + (k_outer * 15)) + (((int)threadIdx.x) * 3)) + 1))];
    }
    if (((int)threadIdx.y) < 20) {
      placeholder_d_shared[((((((int)threadIdx.y) * 15) + (((int)threadIdx.x) * 3)) + 2))] = placeholder1[(((((((((int)blockIdx.z) * 19200) + (((int)blockIdx.x) * 9600)) + (((int)threadIdx.y) * 480)) + (k_outer * 15)) + (((int)threadIdx.x) * 3)) + 2))];
    }
    __syncthreads();
    for (int k_inner = 0; k_inner < 15; ++k_inner) {
      placeholder_shared_local[(0)] = placeholder_shared[(((((int)threadIdx.y) * 30) + k_inner))];
      placeholder_shared_local[(1)] = placeholder_shared[((((((int)threadIdx.y) * 30) + k_inner) + 15))];
      placeholder_d_shared_local[(0)] = placeholder_d_shared[(((((int)threadIdx.x) * 60) + k_inner))];
      placeholder_d_shared_local[(1)] = placeholder_d_shared[((((((int)threadIdx.x) * 60) + k_inner) + 15))];
      placeholder_d_shared_local[(2)] = placeholder_d_shared[((((((int)threadIdx.x) * 60) + k_inner) + 30))];
      placeholder_d_shared_local[(3)] = placeholder_d_shared[((((((int)threadIdx.x) * 60) + k_inner) + 45))];
      compute_local[(0)] = __ocml_fma_f32(placeholder_shared_local[(0)], placeholder_d_shared_local[(0)], compute_local[(0)]);
      compute_local[(1)] = __ocml_fma_f32(placeholder_shared_local[(0)], placeholder_d_shared_local[(1)], compute_local[(1)]);
      compute_local[(2)] = __ocml_fma_f32(placeholder_shared_local[(0)], placeholder_d_shared_local[(2)], compute_local[(2)]);
      compute_local[(3)] = __ocml_fma_f32(placeholder_shared_local[(0)], placeholder_d_shared_local[(3)], compute_local[(3)]);
      compute_local[(4)] = __ocml_fma_f32(placeholder_shared_local[(1)], placeholder_d_shared_local[(0)], compute_local[(4)]);
      compute_local[(5)] = __ocml_fma_f32(placeholder_shared_local[(1)], placeholder_d_shared_local[(1)], compute_local[(5)]);
      compute_local[(6)] = __ocml_fma_f32(placeholder_shared_local[(1)], placeholder_d_shared_local[(2)], compute_local[(6)]);
      compute_local[(7)] = __ocml_fma_f32(placeholder_shared_local[(1)], placeholder_d_shared_local[(3)], compute_local[(7)]);
    }
  }
  compute[((((((((int)blockIdx.z) * 19200) + (((int)blockIdx.y) * 1920)) + (((int)threadIdx.y) * 80)) + (((int)blockIdx.x) * 20)) + (((int)threadIdx.x) * 4)))] = compute_local[(0)];
  compute[(((((((((int)blockIdx.z) * 19200) + (((int)blockIdx.y) * 1920)) + (((int)threadIdx.y) * 80)) + (((int)blockIdx.x) * 20)) + (((int)threadIdx.x) * 4)) + 1))] = compute_local[(1)];
  compute[(((((((((int)blockIdx.z) * 19200) + (((int)blockIdx.y) * 1920)) + (((int)threadIdx.y) * 80)) + (((int)blockIdx.x) * 20)) + (((int)threadIdx.x) * 4)) + 2))] = compute_local[(2)];
  compute[(((((((((int)blockIdx.z) * 19200) + (((int)blockIdx.y) * 1920)) + (((int)threadIdx.y) * 80)) + (((int)blockIdx.x) * 20)) + (((int)threadIdx.x) * 4)) + 3))] = compute_local[(3)];
  compute[(((((((((int)blockIdx.z) * 19200) + (((int)blockIdx.y) * 1920)) + (((int)threadIdx.y) * 80)) + (((int)blockIdx.x) * 20)) + (((int)threadIdx.x) * 4)) + 40))] = compute_local[(4)];
  compute[(((((((((int)blockIdx.z) * 19200) + (((int)blockIdx.y) * 1920)) + (((int)threadIdx.y) * 80)) + (((int)blockIdx.x) * 20)) + (((int)threadIdx.x) * 4)) + 41))] = compute_local[(5)];
  compute[(((((((((int)blockIdx.z) * 19200) + (((int)blockIdx.y) * 1920)) + (((int)threadIdx.y) * 80)) + (((int)blockIdx.x) * 20)) + (((int)threadIdx.x) * 4)) + 42))] = compute_local[(6)];
  compute[(((((((((int)blockIdx.z) * 19200) + (((int)blockIdx.y) * 1920)) + (((int)threadIdx.y) * 80)) + (((int)blockIdx.x) * 20)) + (((int)threadIdx.x) * 4)) + 43))] = compute_local[(7)];
}

__device__ void fused_nn_batch_matmul_5_kernel0_device(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ compute){
  float compute_local[15];
  __shared__ float placeholder_shared[800];
  __shared__ float placeholder_d_shared[480];
  float placeholder_shared_local[5];
  float placeholder_d_shared_local[3];
  for (int i_c_init = 0; i_c_init < 5; ++i_c_init) {
    for (int j_c_init = 0; j_c_init < 3; ++j_c_init) {
      compute_local[(((i_c_init * 3) + j_c_init))] = 0.000000e+00f;
    }
  }
  for (int k_outer = 0; k_outer < 4; ++k_outer) {
    __syncthreads();
    if (((int)threadIdx.x) < 10) {
      placeholder_shared[(((((int)threadIdx.y) * 50) + ((int)threadIdx.x)))] = placeholder[((((((((int)blockIdx.z) * 19200) + (((int)blockIdx.y) * 3200)) + (((int)threadIdx.y) * 200)) + (k_outer * 10)) + ((int)threadIdx.x)))];
    }
    if (((int)threadIdx.x) < 10) {
      placeholder_shared[((((((int)threadIdx.y) * 50) + ((int)threadIdx.x)) + 10))] = placeholder[(((((((((int)blockIdx.z) * 19200) + (((int)blockIdx.y) * 3200)) + (((int)threadIdx.y) * 200)) + (k_outer * 10)) + ((int)threadIdx.x)) + 40))];
    }
    if (((int)threadIdx.x) < 10) {
      placeholder_shared[((((((int)threadIdx.y) * 50) + ((int)threadIdx.x)) + 20))] = placeholder[(((((((((int)blockIdx.z) * 19200) + (((int)blockIdx.y) * 3200)) + (((int)threadIdx.y) * 200)) + (k_outer * 10)) + ((int)threadIdx.x)) + 80))];
    }
    if (((int)threadIdx.x) < 10) {
      placeholder_shared[((((((int)threadIdx.y) * 50) + ((int)threadIdx.x)) + 30))] = placeholder[(((((((((int)blockIdx.z) * 19200) + (((int)blockIdx.y) * 3200)) + (((int)threadIdx.y) * 200)) + (k_outer * 10)) + ((int)threadIdx.x)) + 120))];
    }
    if (((int)threadIdx.x) < 10) {
      placeholder_shared[((((((int)threadIdx.y) * 50) + ((int)threadIdx.x)) + 40))] = placeholder[(((((((((int)blockIdx.z) * 19200) + (((int)blockIdx.y) * 3200)) + (((int)threadIdx.y) * 200)) + (k_outer * 10)) + ((int)threadIdx.x)) + 160))];
    }
    if (((int)threadIdx.x) < 10) {
      placeholder_d_shared[(((((int)threadIdx.y) * 30) + ((int)threadIdx.x)))] = placeholder1[((((((((int)blockIdx.z) * 19200) + (((int)blockIdx.x) * 1920)) + (((int)threadIdx.y) * 120)) + (k_outer * 10)) + ((int)threadIdx.x)))];
    }
    if (((int)threadIdx.x) < 10) {
      placeholder_d_shared[((((((int)threadIdx.y) * 30) + ((int)threadIdx.x)) + 10))] = placeholder1[(((((((((int)blockIdx.z) * 19200) + (((int)blockIdx.x) * 1920)) + (((int)threadIdx.y) * 120)) + (k_outer * 10)) + ((int)threadIdx.x)) + 40))];
    }
    if (((int)threadIdx.x) < 10) {
      placeholder_d_shared[((((((int)threadIdx.y) * 30) + ((int)threadIdx.x)) + 20))] = placeholder1[(((((((((int)blockIdx.z) * 19200) + (((int)blockIdx.x) * 1920)) + (((int)threadIdx.y) * 120)) + (k_outer * 10)) + ((int)threadIdx.x)) + 80))];
    }
    __syncthreads();
    for (int k_inner = 0; k_inner < 10; ++k_inner) {
      placeholder_shared_local[(0)] = placeholder_shared[(((((int)threadIdx.y) * 50) + k_inner))];
      placeholder_shared_local[(1)] = placeholder_shared[((((((int)threadIdx.y) * 50) + k_inner) + 10))];
      placeholder_shared_local[(2)] = placeholder_shared[((((((int)threadIdx.y) * 50) + k_inner) + 20))];
      placeholder_shared_local[(3)] = placeholder_shared[((((((int)threadIdx.y) * 50) + k_inner) + 30))];
      placeholder_shared_local[(4)] = placeholder_shared[((((((int)threadIdx.y) * 50) + k_inner) + 40))];
      placeholder_d_shared_local[(0)] = placeholder_d_shared[(((((int)threadIdx.x) * 30) + k_inner))];
      placeholder_d_shared_local[(1)] = placeholder_d_shared[((((((int)threadIdx.x) * 30) + k_inner) + 10))];
      placeholder_d_shared_local[(2)] = placeholder_d_shared[((((((int)threadIdx.x) * 30) + k_inner) + 20))];
      compute_local[(0)] = __ocml_fma_f32(placeholder_shared_local[(0)], placeholder_d_shared_local[(0)], compute_local[(0)]);
      compute_local[(1)] = __ocml_fma_f32(placeholder_shared_local[(0)], placeholder_d_shared_local[(1)], compute_local[(1)]);
      compute_local[(2)] = __ocml_fma_f32(placeholder_shared_local[(0)], placeholder_d_shared_local[(2)], compute_local[(2)]);
      compute_local[(3)] = __ocml_fma_f32(placeholder_shared_local[(1)], placeholder_d_shared_local[(0)], compute_local[(3)]);
      compute_local[(4)] = __ocml_fma_f32(placeholder_shared_local[(1)], placeholder_d_shared_local[(1)], compute_local[(4)]);
      compute_local[(5)] = __ocml_fma_f32(placeholder_shared_local[(1)], placeholder_d_shared_local[(2)], compute_local[(5)]);
      compute_local[(6)] = __ocml_fma_f32(placeholder_shared_local[(2)], placeholder_d_shared_local[(0)], compute_local[(6)]);
      compute_local[(7)] = __ocml_fma_f32(placeholder_shared_local[(2)], placeholder_d_shared_local[(1)], compute_local[(7)]);
      compute_local[(8)] = __ocml_fma_f32(placeholder_shared_local[(2)], placeholder_d_shared_local[(2)], compute_local[(8)]);
      compute_local[(9)] = __ocml_fma_f32(placeholder_shared_local[(3)], placeholder_d_shared_local[(0)], compute_local[(9)]);
      compute_local[(10)] = __ocml_fma_f32(placeholder_shared_local[(3)], placeholder_d_shared_local[(1)], compute_local[(10)]);
      compute_local[(11)] = __ocml_fma_f32(placeholder_shared_local[(3)], placeholder_d_shared_local[(2)], compute_local[(11)]);
      compute_local[(12)] = __ocml_fma_f32(placeholder_shared_local[(4)], placeholder_d_shared_local[(0)], compute_local[(12)]);
      compute_local[(13)] = __ocml_fma_f32(placeholder_shared_local[(4)], placeholder_d_shared_local[(1)], compute_local[(13)]);
      compute_local[(14)] = __ocml_fma_f32(placeholder_shared_local[(4)], placeholder_d_shared_local[(2)], compute_local[(14)]);
    }
  }
  compute[((((((((int)blockIdx.z) * 230400) + (((int)blockIdx.y) * 38400)) + (((int)threadIdx.y) * 2400)) + (((int)blockIdx.x) * 48)) + (((int)threadIdx.x) * 3)))] = compute_local[(0)];
  compute[(((((((((int)blockIdx.z) * 230400) + (((int)blockIdx.y) * 38400)) + (((int)threadIdx.y) * 2400)) + (((int)blockIdx.x) * 48)) + (((int)threadIdx.x) * 3)) + 1))] = compute_local[(1)];
  compute[(((((((((int)blockIdx.z) * 230400) + (((int)blockIdx.y) * 38400)) + (((int)threadIdx.y) * 2400)) + (((int)blockIdx.x) * 48)) + (((int)threadIdx.x) * 3)) + 2))] = compute_local[(2)];
  compute[(((((((((int)blockIdx.z) * 230400) + (((int)blockIdx.y) * 38400)) + (((int)threadIdx.y) * 2400)) + (((int)blockIdx.x) * 48)) + (((int)threadIdx.x) * 3)) + 480))] = compute_local[(3)];
  compute[(((((((((int)blockIdx.z) * 230400) + (((int)blockIdx.y) * 38400)) + (((int)threadIdx.y) * 2400)) + (((int)blockIdx.x) * 48)) + (((int)threadIdx.x) * 3)) + 481))] = compute_local[(4)];
  compute[(((((((((int)blockIdx.z) * 230400) + (((int)blockIdx.y) * 38400)) + (((int)threadIdx.y) * 2400)) + (((int)blockIdx.x) * 48)) + (((int)threadIdx.x) * 3)) + 482))] = compute_local[(5)];
  compute[(((((((((int)blockIdx.z) * 230400) + (((int)blockIdx.y) * 38400)) + (((int)threadIdx.y) * 2400)) + (((int)blockIdx.x) * 48)) + (((int)threadIdx.x) * 3)) + 960))] = compute_local[(6)];
  compute[(((((((((int)blockIdx.z) * 230400) + (((int)blockIdx.y) * 38400)) + (((int)threadIdx.y) * 2400)) + (((int)blockIdx.x) * 48)) + (((int)threadIdx.x) * 3)) + 961))] = compute_local[(7)];
  compute[(((((((((int)blockIdx.z) * 230400) + (((int)blockIdx.y) * 38400)) + (((int)threadIdx.y) * 2400)) + (((int)blockIdx.x) * 48)) + (((int)threadIdx.x) * 3)) + 962))] = compute_local[(8)];
  compute[(((((((((int)blockIdx.z) * 230400) + (((int)blockIdx.y) * 38400)) + (((int)threadIdx.y) * 2400)) + (((int)blockIdx.x) * 48)) + (((int)threadIdx.x) * 3)) + 1440))] = compute_local[(9)];
  compute[(((((((((int)blockIdx.z) * 230400) + (((int)blockIdx.y) * 38400)) + (((int)threadIdx.y) * 2400)) + (((int)blockIdx.x) * 48)) + (((int)threadIdx.x) * 3)) + 1441))] = compute_local[(10)];
  compute[(((((((((int)blockIdx.z) * 230400) + (((int)blockIdx.y) * 38400)) + (((int)threadIdx.y) * 2400)) + (((int)blockIdx.x) * 48)) + (((int)threadIdx.x) * 3)) + 1442))] = compute_local[(11)];
  compute[(((((((((int)blockIdx.z) * 230400) + (((int)blockIdx.y) * 38400)) + (((int)threadIdx.y) * 2400)) + (((int)blockIdx.x) * 48)) + (((int)threadIdx.x) * 3)) + 1920))] = compute_local[(12)];
  compute[(((((((((int)blockIdx.z) * 230400) + (((int)blockIdx.y) * 38400)) + (((int)threadIdx.y) * 2400)) + (((int)blockIdx.x) * 48)) + (((int)threadIdx.x) * 3)) + 1921))] = compute_local[(13)];
  compute[(((((((((int)blockIdx.z) * 230400) + (((int)blockIdx.y) * 38400)) + (((int)threadIdx.y) * 2400)) + (((int)blockIdx.x) * 48)) + (((int)threadIdx.x) * 3)) + 1922))] = compute_local[(14)];
}

__device__ void fused_nn_softmax_1_kernel1_device(float* __restrict__ T_softmax_exp, float* __restrict__ placeholder, float* __restrict__ T_softmax_maxelem){
  for (int i0_i1_fused_i2_fused_i3_fused_outer = 0; i0_i1_fused_i2_fused_i3_fused_outer < 43; ++i0_i1_fused_i2_fused_i3_fused_outer) {
    if ((((i0_i1_fused_i2_fused_i3_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) < 2764800) {
      T_softmax_exp[((((i0_i1_fused_i2_fused_i3_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)))] = __ocml_exp_f32((placeholder[((((i0_i1_fused_i2_fused_i3_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)))] - T_softmax_maxelem[(((((i0_i1_fused_i2_fused_i3_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) / 480))]));
    }
  }
}

__device__ void fused_reshape_add_multiply_erf_multiply_add_multiply_reshape_kernel0_device(float* __restrict__ T_reshape, float* __restrict__ placeholder, float* __restrict__ placeholder1){
  for (int ax0_ax1_fused_ax2_fused_outer = 0; ax0_ax1_fused_ax2_fused_outer < 4; ++ax0_ax1_fused_ax2_fused_outer) {
    if ((((ax0_ax1_fused_ax2_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) < 230400) {
      T_reshape[((((ax0_ax1_fused_ax2_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)))] = ((placeholder[((((ax0_ax1_fused_ax2_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)))] + placeholder1[(((((ax0_ax1_fused_ax2_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) / 480))]) * __ocml_fma_f32(__ocml_erf_f32(((placeholder[((((ax0_ax1_fused_ax2_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)))] + placeholder1[(((((ax0_ax1_fused_ax2_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) / 480))]) * 7.071068e-01f)), 5.000000e-01f, 5.000000e-01f));
    }
  }
}

__device__ void fused_mean_1_kernel0_device(float* __restrict__ placeholder, float* __restrict__ placeholder_red){
  float placeholder_red_rf[1];
  __shared__ float red_buf0[1024];
  placeholder_red_rf[(0)] = 0.000000e+00f;
  for (int k2_outer = 0; k2_outer < 15; ++k2_outer) {
    placeholder_red_rf[(0)] = (placeholder_red_rf[(0)] + placeholder[(((((((int)blockIdx.x) * 15360) + (((int)threadIdx.y) * 480)) + (k2_outer * 32)) + ((int)threadIdx.x)))]);
  }
  __syncthreads();
  ((volatile float*)red_buf0)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)))] = placeholder_red_rf[(0)];
  __syncthreads();
  if (((int)threadIdx.x) < 16) {
    ((volatile float*)red_buf0)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)))] = (((volatile float*)red_buf0)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)))] + ((volatile float*)red_buf0)[((((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 16))]);
    ((volatile float*)red_buf0)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)))] = (((volatile float*)red_buf0)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)))] + ((volatile float*)red_buf0)[((((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 8))]);
    ((volatile float*)red_buf0)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)))] = (((volatile float*)red_buf0)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)))] + ((volatile float*)red_buf0)[((((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 4))]);
    ((volatile float*)red_buf0)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)))] = (((volatile float*)red_buf0)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)))] + ((volatile float*)red_buf0)[((((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 2))]);
    ((volatile float*)red_buf0)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)))] = (((volatile float*)red_buf0)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)))] + ((volatile float*)red_buf0)[((((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1))]);
  }
  __syncthreads();
  if (((int)threadIdx.x) == 0) {
    placeholder_red[(((((int)blockIdx.x) * 32) + ((int)threadIdx.y)))] = ((volatile float*)red_buf0)[((((int)threadIdx.y) * 32))];
  }
}

__device__ void fused_subtract_add_sqrt_divide_multiply_add_1_kernel0_device(float* __restrict__ T_add, float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ placeholder2, float* __restrict__ placeholder3, float* __restrict__ placeholder4){
  for (int ax0_ax1_fused_ax2_fused_outer = 0; ax0_ax1_fused_ax2_fused_outer < 4; ++ax0_ax1_fused_ax2_fused_outer) {
    if ((((ax0_ax1_fused_ax2_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) < 230400) {
      T_add[((((ax0_ax1_fused_ax2_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)))] = __ocml_fma_f32(((placeholder[((((ax0_ax1_fused_ax2_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)))] - placeholder1[(((((ax0_ax1_fused_ax2_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) / 480))]) / __ocml_sqrt_f32((placeholder2[(((((ax0_ax1_fused_ax2_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) / 480))] + 1.000000e-12f))), placeholder3[(((((ax0_ax1_fused_ax2_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) % 480))], placeholder4[(((((ax0_ax1_fused_ax2_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) % 480))]);
    }
  }
}

__device__ void fused_nn_batch_matmul_3_kernel0_device(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ compute){
  float compute_local[9];
  __shared__ float placeholder_shared[2400];
  __shared__ float placeholder_d_shared[300];
  float placeholder_shared_local[3];
  float placeholder_d_shared_local[3];
  for (int i_c_init = 0; i_c_init < 3; ++i_c_init) {
    for (int j_c_init = 0; j_c_init < 3; ++j_c_init) {
      compute_local[(((i_c_init * 3) + j_c_init))] = 0.000000e+00f;
    }
  }
  for (int k_outer = 0; k_outer < 24; ++k_outer) {
    __syncthreads();
    placeholder_shared[(((((int)threadIdx.y) * 60) + (((int)threadIdx.x) * 4)))] = placeholder[(((((((int)blockIdx.y) * 57600) + (((int)threadIdx.y) * 1440)) + (k_outer * 20)) + (((int)threadIdx.x) * 4)))];
    placeholder_shared[((((((int)threadIdx.y) * 60) + (((int)threadIdx.x) * 4)) + 1))] = placeholder[((((((((int)blockIdx.y) * 57600) + (((int)threadIdx.y) * 1440)) + (k_outer * 20)) + (((int)threadIdx.x) * 4)) + 1))];
    placeholder_shared[((((((int)threadIdx.y) * 60) + (((int)threadIdx.x) * 4)) + 2))] = placeholder[((((((((int)blockIdx.y) * 57600) + (((int)threadIdx.y) * 1440)) + (k_outer * 20)) + (((int)threadIdx.x) * 4)) + 2))];
    placeholder_shared[((((((int)threadIdx.y) * 60) + (((int)threadIdx.x) * 4)) + 3))] = placeholder[((((((((int)blockIdx.y) * 57600) + (((int)threadIdx.y) * 1440)) + (k_outer * 20)) + (((int)threadIdx.x) * 4)) + 3))];
    placeholder_shared[((((((int)threadIdx.y) * 60) + (((int)threadIdx.x) * 4)) + 20))] = placeholder[((((((((int)blockIdx.y) * 57600) + (((int)threadIdx.y) * 1440)) + (k_outer * 20)) + (((int)threadIdx.x) * 4)) + 480))];
    placeholder_shared[((((((int)threadIdx.y) * 60) + (((int)threadIdx.x) * 4)) + 21))] = placeholder[((((((((int)blockIdx.y) * 57600) + (((int)threadIdx.y) * 1440)) + (k_outer * 20)) + (((int)threadIdx.x) * 4)) + 481))];
    placeholder_shared[((((((int)threadIdx.y) * 60) + (((int)threadIdx.x) * 4)) + 22))] = placeholder[((((((((int)blockIdx.y) * 57600) + (((int)threadIdx.y) * 1440)) + (k_outer * 20)) + (((int)threadIdx.x) * 4)) + 482))];
    placeholder_shared[((((((int)threadIdx.y) * 60) + (((int)threadIdx.x) * 4)) + 23))] = placeholder[((((((((int)blockIdx.y) * 57600) + (((int)threadIdx.y) * 1440)) + (k_outer * 20)) + (((int)threadIdx.x) * 4)) + 483))];
    placeholder_shared[((((((int)threadIdx.y) * 60) + (((int)threadIdx.x) * 4)) + 40))] = placeholder[((((((((int)blockIdx.y) * 57600) + (((int)threadIdx.y) * 1440)) + (k_outer * 20)) + (((int)threadIdx.x) * 4)) + 960))];
    placeholder_shared[((((((int)threadIdx.y) * 60) + (((int)threadIdx.x) * 4)) + 41))] = placeholder[((((((((int)blockIdx.y) * 57600) + (((int)threadIdx.y) * 1440)) + (k_outer * 20)) + (((int)threadIdx.x) * 4)) + 961))];
    placeholder_shared[((((((int)threadIdx.y) * 60) + (((int)threadIdx.x) * 4)) + 42))] = placeholder[((((((((int)blockIdx.y) * 57600) + (((int)threadIdx.y) * 1440)) + (k_outer * 20)) + (((int)threadIdx.x) * 4)) + 962))];
    placeholder_shared[((((((int)threadIdx.y) * 60) + (((int)threadIdx.x) * 4)) + 43))] = placeholder[((((((((int)blockIdx.y) * 57600) + (((int)threadIdx.y) * 1440)) + (k_outer * 20)) + (((int)threadIdx.x) * 4)) + 963))];
    if (((int)threadIdx.y) < 15) {
      placeholder_d_shared[(((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 4)))] = placeholder1[(((((((int)blockIdx.x) * 7200) + (((int)threadIdx.y) * 480)) + (k_outer * 20)) + (((int)threadIdx.x) * 4)))];
    }
    if (((int)threadIdx.y) < 15) {
      placeholder_d_shared[((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 4)) + 1))] = placeholder1[((((((((int)blockIdx.x) * 7200) + (((int)threadIdx.y) * 480)) + (k_outer * 20)) + (((int)threadIdx.x) * 4)) + 1))];
    }
    if (((int)threadIdx.y) < 15) {
      placeholder_d_shared[((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 4)) + 2))] = placeholder1[((((((((int)blockIdx.x) * 7200) + (((int)threadIdx.y) * 480)) + (k_outer * 20)) + (((int)threadIdx.x) * 4)) + 2))];
    }
    if (((int)threadIdx.y) < 15) {
      placeholder_d_shared[((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 4)) + 3))] = placeholder1[((((((((int)blockIdx.x) * 7200) + (((int)threadIdx.y) * 480)) + (k_outer * 20)) + (((int)threadIdx.x) * 4)) + 3))];
    }
    __syncthreads();
    for (int k_inner = 0; k_inner < 20; ++k_inner) {
      placeholder_shared_local[(0)] = placeholder_shared[(((((int)threadIdx.y) * 60) + k_inner))];
      placeholder_shared_local[(1)] = placeholder_shared[((((((int)threadIdx.y) * 60) + k_inner) + 20))];
      placeholder_shared_local[(2)] = placeholder_shared[((((((int)threadIdx.y) * 60) + k_inner) + 40))];
      placeholder_d_shared_local[(0)] = placeholder_d_shared[(((((int)threadIdx.x) * 60) + k_inner))];
      placeholder_d_shared_local[(1)] = placeholder_d_shared[((((((int)threadIdx.x) * 60) + k_inner) + 20))];
      placeholder_d_shared_local[(2)] = placeholder_d_shared[((((((int)threadIdx.x) * 60) + k_inner) + 40))];
      compute_local[(0)] = __ocml_fma_f32(placeholder_shared_local[(0)], placeholder_d_shared_local[(0)], compute_local[(0)]);
      compute_local[(1)] = __ocml_fma_f32(placeholder_shared_local[(0)], placeholder_d_shared_local[(1)], compute_local[(1)]);
      compute_local[(2)] = __ocml_fma_f32(placeholder_shared_local[(0)], placeholder_d_shared_local[(2)], compute_local[(2)]);
      compute_local[(3)] = __ocml_fma_f32(placeholder_shared_local[(1)], placeholder_d_shared_local[(0)], compute_local[(3)]);
      compute_local[(4)] = __ocml_fma_f32(placeholder_shared_local[(1)], placeholder_d_shared_local[(1)], compute_local[(4)]);
      compute_local[(5)] = __ocml_fma_f32(placeholder_shared_local[(1)], placeholder_d_shared_local[(2)], compute_local[(5)]);
      compute_local[(6)] = __ocml_fma_f32(placeholder_shared_local[(2)], placeholder_d_shared_local[(0)], compute_local[(6)]);
      compute_local[(7)] = __ocml_fma_f32(placeholder_shared_local[(2)], placeholder_d_shared_local[(1)], compute_local[(7)]);
      compute_local[(8)] = __ocml_fma_f32(placeholder_shared_local[(2)], placeholder_d_shared_local[(2)], compute_local[(8)]);
    }
  }
  compute[(((((((int)blockIdx.y) * 57600) + (((int)threadIdx.y) * 1440)) + (((int)blockIdx.x) * 15)) + (((int)threadIdx.x) * 3)))] = compute_local[(0)];
  compute[((((((((int)blockIdx.y) * 57600) + (((int)threadIdx.y) * 1440)) + (((int)blockIdx.x) * 15)) + (((int)threadIdx.x) * 3)) + 1))] = compute_local[(1)];
  compute[((((((((int)blockIdx.y) * 57600) + (((int)threadIdx.y) * 1440)) + (((int)blockIdx.x) * 15)) + (((int)threadIdx.x) * 3)) + 2))] = compute_local[(2)];
  compute[((((((((int)blockIdx.y) * 57600) + (((int)threadIdx.y) * 1440)) + (((int)blockIdx.x) * 15)) + (((int)threadIdx.x) * 3)) + 480))] = compute_local[(3)];
  compute[((((((((int)blockIdx.y) * 57600) + (((int)threadIdx.y) * 1440)) + (((int)blockIdx.x) * 15)) + (((int)threadIdx.x) * 3)) + 481))] = compute_local[(4)];
  compute[((((((((int)blockIdx.y) * 57600) + (((int)threadIdx.y) * 1440)) + (((int)blockIdx.x) * 15)) + (((int)threadIdx.x) * 3)) + 482))] = compute_local[(5)];
  compute[((((((((int)blockIdx.y) * 57600) + (((int)threadIdx.y) * 1440)) + (((int)blockIdx.x) * 15)) + (((int)threadIdx.x) * 3)) + 960))] = compute_local[(6)];
  compute[((((((((int)blockIdx.y) * 57600) + (((int)threadIdx.y) * 1440)) + (((int)blockIdx.x) * 15)) + (((int)threadIdx.x) * 3)) + 961))] = compute_local[(7)];
  compute[((((((((int)blockIdx.y) * 57600) + (((int)threadIdx.y) * 1440)) + (((int)blockIdx.x) * 15)) + (((int)threadIdx.x) * 3)) + 962))] = compute_local[(8)];
}

__device__ void fused_reshape_5_kernel0_device(float* __restrict__ T_reshape, float* __restrict__ placeholder){
  for (int ax0_ax1_fused_ax2_fused_outer = 0; ax0_ax1_fused_ax2_fused_outer < 43; ++ax0_ax1_fused_ax2_fused_outer) {
    if ((((ax0_ax1_fused_ax2_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) < 2764800) {
      T_reshape[((((ax0_ax1_fused_ax2_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)))] = placeholder[((((ax0_ax1_fused_ax2_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)))];
    }
  }
}

__device__ void fused_reshape_transpose_copy_reshape_1_kernel0_device(float* __restrict__ T_reshape, float* __restrict__ placeholder){
  for (int ax0_ax1_fused_ax2_fused_outer = 0; ax0_ax1_fused_ax2_fused_outer < 4; ++ax0_ax1_fused_ax2_fused_outer) {
    if ((((ax0_ax1_fused_ax2_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) < 230400) {
      T_reshape[((((ax0_ax1_fused_ax2_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)))] = placeholder[(((((((((ax0_ax1_fused_ax2_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) % 480) / 40) * 19200) + (((((ax0_ax1_fused_ax2_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) / 480) * 40)) + ((((ax0_ax1_fused_ax2_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) % 40)))];
    }
  }
}

__device__ void fused_reshape_add_reshape_transpose_reshape_transpose_kernel0_device(float* __restrict__ T_transpose, float* __restrict__ placeholder, float* __restrict__ placeholder1){
  for (int ax0_ax1_fused_ax2_fused_outer = 0; ax0_ax1_fused_ax2_fused_outer < 4; ++ax0_ax1_fused_ax2_fused_outer) {
    if ((((ax0_ax1_fused_ax2_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) < 230400) {
      T_transpose[((((ax0_ax1_fused_ax2_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)))] = (placeholder[(((((((ax0_ax1_fused_ax2_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) % 480) * 480) + ((((ax0_ax1_fused_ax2_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) / 480)))] + placeholder1[(((((ax0_ax1_fused_ax2_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) % 480))]);
    }
  }
}

__device__ void fused_reshape_add_reshape_transpose_transpose_reshape_transpose_kernel0_device(float* __restrict__ T_transpose, float* __restrict__ placeholder, float* __restrict__ placeholder1){
  for (int ax0_ax1_fused_ax2_fused_outer = 0; ax0_ax1_fused_ax2_fused_outer < 4; ++ax0_ax1_fused_ax2_fused_outer) {
    if ((((ax0_ax1_fused_ax2_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) < 230400) {
      T_transpose[((((ax0_ax1_fused_ax2_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)))] = (placeholder[(((((((((ax0_ax1_fused_ax2_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) % 19200) / 40) * 480) + (((((ax0_ax1_fused_ax2_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) / 19200) * 40)) + ((((ax0_ax1_fused_ax2_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) % 40)))] + placeholder1[((((((ax0_ax1_fused_ax2_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) % 19200) / 40))]);
    }
  }
}

__device__ void fused_reshape_add_reshape_transpose_divide_reshape_kernel0_device(float* __restrict__ T_reshape, float* __restrict__ placeholder, float* __restrict__ placeholder1){
  for (int ax0_ax1_fused_ax2_fused_outer = 0; ax0_ax1_fused_ax2_fused_outer < 4; ++ax0_ax1_fused_ax2_fused_outer) {
    if ((((ax0_ax1_fused_ax2_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) < 230400) {
      T_reshape[((((ax0_ax1_fused_ax2_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)))] = ((placeholder[(((((((((ax0_ax1_fused_ax2_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) % 19200) / 40) * 480) + (((((ax0_ax1_fused_ax2_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) / 19200) * 40)) + ((((ax0_ax1_fused_ax2_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) % 40)))] + placeholder1[((((((ax0_ax1_fused_ax2_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) % 19200) / 40))]) * 1.581139e-01f);
    }
  }
}

__device__ void fused_full_equal_reshape_kernel0_device(signed char* __restrict__ T_reshape){
  if (((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) < 480) {
    T_reshape[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)))] = (signed char)0;
  }
}

__device__ void fused_cast_take_broadcast_to_like_cast_take_add_1_kernel0_device(float* __restrict__ T_add, float* __restrict__ placeholder, long* __restrict__ placeholder1, float* __restrict__ placeholder2, long* __restrict__ placeholder3){
  for (int ax0_ax1_fused_ax2_fused_outer = 0; ax0_ax1_fused_ax2_fused_outer < 4; ++ax0_ax1_fused_ax2_fused_outer) {
    if ((((ax0_ax1_fused_ax2_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) < 230400) {
      T_add[((((ax0_ax1_fused_ax2_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)))] = (placeholder[(((min(max(0, ((int)placeholder1[(((((ax0_ax1_fused_ax2_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) / 480))])), 30521) * 480) + ((((ax0_ax1_fused_ax2_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) % 480)))] + placeholder2[(((min(max(0, ((int)placeholder3[(((((ax0_ax1_fused_ax2_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) / 480))])), 1023) * 480) + ((((ax0_ax1_fused_ax2_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) % 480)))]);
    }
  }
}

__device__ void fused_nn_softmax_1_kernel2_device(float* __restrict__ T_softmax_maxelem, float* __restrict__ T_softmax_exp){
  if (((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) < 5760) {
    T_softmax_maxelem[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)))] = 0.000000e+00f;
  }
  for (int k = 0; k < 480; ++k) {
    if (((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) < 5760) {
      T_softmax_maxelem[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)))] = (T_softmax_maxelem[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)))] + T_softmax_exp[((((((int)blockIdx.x) * 122880) + (((int)threadIdx.x) * 480)) + k))]);
    }
  }
}

__device__ void fused_mean_1_kernel1_device(float* __restrict__ T_divide, float* __restrict__ placeholder_red){
  if (((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) < 480) {
    T_divide[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)))] = (placeholder_red[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)))] * 2.083333e-03f);
  }
}

__device__ void fused_reshape_4_kernel0_device(float* __restrict__ T_reshape, float* __restrict__ placeholder){
  for (int ax0_ax1_fused_ax2_fused_outer = 0; ax0_ax1_fused_ax2_fused_outer < 4; ++ax0_ax1_fused_ax2_fused_outer) {
    if ((((ax0_ax1_fused_ax2_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) < 230400) {
      T_reshape[((((ax0_ax1_fused_ax2_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)))] = placeholder[((((ax0_ax1_fused_ax2_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)))];
    }
  }
}

__device__ void fused_variance_1_kernel0_device(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_multiply_red){
  float T_multiply_red_rf[1];
  __shared__ float red_buf0[1024];
  T_multiply_red_rf[(0)] = 0.000000e+00f;
  for (int k2_outer = 0; k2_outer < 15; ++k2_outer) {
    T_multiply_red_rf[(0)] = __ocml_fma_f32((placeholder[(((((((int)blockIdx.x) * 15360) + (((int)threadIdx.y) * 480)) + (k2_outer * 32)) + ((int)threadIdx.x)))] - placeholder1[(((((int)blockIdx.x) * 32) + ((int)threadIdx.y)))]), (placeholder[(((((((int)blockIdx.x) * 15360) + (((int)threadIdx.y) * 480)) + (k2_outer * 32)) + ((int)threadIdx.x)))] - placeholder1[(((((int)blockIdx.x) * 32) + ((int)threadIdx.y)))]), T_multiply_red_rf[(0)]);
  }
  __syncthreads();
  ((volatile float*)red_buf0)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)))] = T_multiply_red_rf[(0)];
  __syncthreads();
  if (((int)threadIdx.x) < 16) {
    ((volatile float*)red_buf0)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)))] = (((volatile float*)red_buf0)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)))] + ((volatile float*)red_buf0)[((((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 16))]);
    ((volatile float*)red_buf0)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)))] = (((volatile float*)red_buf0)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)))] + ((volatile float*)red_buf0)[((((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 8))]);
    ((volatile float*)red_buf0)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)))] = (((volatile float*)red_buf0)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)))] + ((volatile float*)red_buf0)[((((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 4))]);
    ((volatile float*)red_buf0)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)))] = (((volatile float*)red_buf0)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)))] + ((volatile float*)red_buf0)[((((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 2))]);
    ((volatile float*)red_buf0)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)))] = (((volatile float*)red_buf0)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)))] + ((volatile float*)red_buf0)[((((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1))]);
  }
  __syncthreads();
  if (((int)threadIdx.x) == 0) {
    T_multiply_red[(((((int)blockIdx.x) * 32) + ((int)threadIdx.y)))] = ((volatile float*)red_buf0)[((((int)threadIdx.y) * 32))];
  }
}

__device__ void fused_variance_1_kernel1_device(float* __restrict__ T_divide, float* __restrict__ T_multiply_red){
  if (((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) < 480) {
    T_divide[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)))] = (T_multiply_red[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)))] * 2.083333e-03f);
  }
}

__device__ void fused_reshape_cast_broadcast_to_like_where_kernel0_device(float* __restrict__ T_where, signed char* __restrict__ placeholder, float* __restrict__ placeholder1){
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer < 43; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) < 2764800) {
      T_where[((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)))] = ((((int)((bool)placeholder[(((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) % 480))])) != 0) ? -__int_as_float(0x7f800000) : placeholder1[((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)))]);
    }
  }
}

__device__ void fused_nn_softmax_1_kernel3_device(float* __restrict__ T_softmax_norm, float* __restrict__ T_softmax_exp, float* __restrict__ T_softmax_maxelem){
  for (int i0_i1_fused_i2_fused_i3_fused_outer = 0; i0_i1_fused_i2_fused_i3_fused_outer < 43; ++i0_i1_fused_i2_fused_i3_fused_outer) {
    if ((((i0_i1_fused_i2_fused_i3_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) < 2764800) {
      T_softmax_norm[((((i0_i1_fused_i2_fused_i3_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)))] = (T_softmax_exp[((((i0_i1_fused_i2_fused_i3_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)))] / T_softmax_maxelem[(((((i0_i1_fused_i2_fused_i3_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) / 480))]);
    }
  }
}


extern "C" __global__  __attribute__((amdgpu_num_vgpr(25))) __attribute__((amdgpu_num_sgpr(30))) void fused_nn_softmax_1_kernel0_device_wrapper(float* __restrict__ T_softmax_maxelem, float* __restrict__ placeholder) {
    // Force the compiler to use all the index
    if (threadIdx.x + threadIdx.y * 256 + threadIdx.z * 1 * 256 >= 256 * 1 * 1) return;
    // if (blockIdx.x + blockIdx.y * 23 + blockIdx.z * 1 * 23 >= 23 * 1 * 1) return;
    fused_nn_softmax_1_kernel0_device((float* __restrict__)T_softmax_maxelem,(float* __restrict__)placeholder);
    asm volatile(";; end_flag"); // jump back to the caller
}

extern "C" __global__  __attribute__((amdgpu_num_vgpr(25))) __attribute__((amdgpu_num_sgpr(30))) void fused_reshape_add_add_kernel0_device_wrapper(float* __restrict__ T_add, float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ placeholder2) {
    // Force the compiler to use all the index
    if (threadIdx.x + threadIdx.y * 256 + threadIdx.z * 1 * 256 >= 256 * 1 * 1) return;
    // if (blockIdx.x + blockIdx.y * 256 + blockIdx.z * 1 * 256 >= 256 * 1 * 1) return;
    fused_reshape_add_add_kernel0_device((float* __restrict__)T_add,(float* __restrict__)placeholder,(float* __restrict__)placeholder1,(float* __restrict__)placeholder2);
    asm volatile(";; end_flag"); // jump back to the caller
}

extern "C" __global__  __attribute__((amdgpu_num_vgpr(61))) __attribute__((amdgpu_num_sgpr(30))) void fused_nn_batch_matmul_4_kernel0_device_wrapper(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ compute) {
    // Force the compiler to use all the index
    if (threadIdx.x + threadIdx.y * 5 + threadIdx.z * 24 * 5 >= 5 * 24 * 1) return;
    // if (blockIdx.x + blockIdx.y * 2 + blockIdx.z * 10 * 2 >= 2 * 10 * 12) return;
    fused_nn_batch_matmul_4_kernel0_device((float* __restrict__)placeholder,(float* __restrict__)placeholder1,(float* __restrict__)compute);
    asm volatile(";; end_flag"); // jump back to the caller
}

extern "C" __global__  __attribute__((amdgpu_num_vgpr(61))) __attribute__((amdgpu_num_sgpr(34))) void fused_nn_batch_matmul_5_kernel0_device_wrapper(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ compute) {
    // Force the compiler to use all the index
    if (threadIdx.x + threadIdx.y * 16 + threadIdx.z * 16 * 16 >= 16 * 16 * 1) return;
    // if (blockIdx.x + blockIdx.y * 10 + blockIdx.z * 6 * 10 >= 10 * 6 * 12) return;
    fused_nn_batch_matmul_5_kernel0_device((float* __restrict__)placeholder,(float* __restrict__)placeholder1,(float* __restrict__)compute);
    asm volatile(";; end_flag"); // jump back to the caller
}

extern "C" __global__  __attribute__((amdgpu_num_vgpr(25))) __attribute__((amdgpu_num_sgpr(30))) void fused_nn_softmax_1_kernel1_device_wrapper(float* __restrict__ T_softmax_exp, float* __restrict__ placeholder, float* __restrict__ T_softmax_maxelem) {
    // Force the compiler to use all the index
    if (threadIdx.x + threadIdx.y * 256 + threadIdx.z * 1 * 256 >= 256 * 1 * 1) return;
    // if (blockIdx.x + blockIdx.y * 256 + blockIdx.z * 1 * 256 >= 256 * 1 * 1) return;
    fused_nn_softmax_1_kernel1_device((float* __restrict__)T_softmax_exp,(float* __restrict__)placeholder,(float* __restrict__)T_softmax_maxelem);
    asm volatile(";; end_flag"); // jump back to the caller
}

extern "C" __global__  __attribute__((amdgpu_num_vgpr(25))) __attribute__((amdgpu_num_sgpr(30))) void fused_reshape_add_multiply_erf_multiply_add_multiply_reshape_kernel0_device_wrapper(float* __restrict__ T_reshape, float* __restrict__ placeholder, float* __restrict__ placeholder1) {
    // Force the compiler to use all the index
    if (threadIdx.x + threadIdx.y * 256 + threadIdx.z * 1 * 256 >= 256 * 1 * 1) return;
    // if (blockIdx.x + blockIdx.y * 256 + blockIdx.z * 1 * 256 >= 256 * 1 * 1) return;
    fused_reshape_add_multiply_erf_multiply_add_multiply_reshape_kernel0_device((float* __restrict__)T_reshape,(float* __restrict__)placeholder,(float* __restrict__)placeholder1);
    asm volatile(";; end_flag"); // jump back to the caller
}

extern "C" __global__  __attribute__((amdgpu_num_vgpr(25))) __attribute__((amdgpu_num_sgpr(30))) void fused_mean_1_kernel0_device_wrapper(float* __restrict__ placeholder, float* __restrict__ placeholder_red) {
    // Force the compiler to use all the index
    if (threadIdx.x + threadIdx.y * 32 + threadIdx.z * 32 * 32 >= 32 * 32 * 1) return;
    // if (blockIdx.x + blockIdx.y * 15 + blockIdx.z * 1 * 15 >= 15 * 1 * 1) return;
    fused_mean_1_kernel0_device((float* __restrict__)placeholder,(float* __restrict__)placeholder_red);
    asm volatile(";; end_flag"); // jump back to the caller
}

extern "C" __global__  __attribute__((amdgpu_num_vgpr(25))) __attribute__((amdgpu_num_sgpr(30))) void fused_subtract_add_sqrt_divide_multiply_add_1_kernel0_device_wrapper(float* __restrict__ T_add, float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ placeholder2, float* __restrict__ placeholder3, float* __restrict__ placeholder4) {
    // Force the compiler to use all the index
    if (threadIdx.x + threadIdx.y * 256 + threadIdx.z * 1 * 256 >= 256 * 1 * 1) return;
    // if (blockIdx.x + blockIdx.y * 256 + blockIdx.z * 1 * 256 >= 256 * 1 * 1) return;
    fused_subtract_add_sqrt_divide_multiply_add_1_kernel0_device((float* __restrict__)T_add,(float* __restrict__)placeholder,(float* __restrict__)placeholder1,(float* __restrict__)placeholder2,(float* __restrict__)placeholder3,(float* __restrict__)placeholder4);
    asm volatile(";; end_flag"); // jump back to the caller
}

extern "C" __global__  __attribute__((amdgpu_num_vgpr(61))) __attribute__((amdgpu_num_sgpr(30))) void fused_nn_batch_matmul_3_kernel0_device_wrapper(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ compute) {
    // Force the compiler to use all the index
    if (threadIdx.x + threadIdx.y * 5 + threadIdx.z * 40 * 5 >= 5 * 40 * 1) return;
    // if (blockIdx.x + blockIdx.y * 32 + blockIdx.z * 4 * 32 >= 32 * 4 * 1) return;
    fused_nn_batch_matmul_3_kernel0_device((float* __restrict__)placeholder,(float* __restrict__)placeholder1,(float* __restrict__)compute);
    asm volatile(";; end_flag"); // jump back to the caller
}

extern "C" __global__  __attribute__((amdgpu_num_vgpr(25))) __attribute__((amdgpu_num_sgpr(30))) void fused_reshape_5_kernel0_device_wrapper(float* __restrict__ T_reshape, float* __restrict__ placeholder) {
    // Force the compiler to use all the index
    if (threadIdx.x + threadIdx.y * 256 + threadIdx.z * 1 * 256 >= 256 * 1 * 1) return;
    // if (blockIdx.x + blockIdx.y * 256 + blockIdx.z * 1 * 256 >= 256 * 1 * 1) return;
    fused_reshape_5_kernel0_device((float* __restrict__)T_reshape,(float* __restrict__)placeholder);
    asm volatile(";; end_flag"); // jump back to the caller
}

extern "C" __global__  __attribute__((amdgpu_num_vgpr(25))) __attribute__((amdgpu_num_sgpr(30))) void fused_reshape_transpose_copy_reshape_1_kernel0_device_wrapper(float* __restrict__ T_reshape, float* __restrict__ placeholder) {
    // Force the compiler to use all the index
    if (threadIdx.x + threadIdx.y * 256 + threadIdx.z * 1 * 256 >= 256 * 1 * 1) return;
    // if (blockIdx.x + blockIdx.y * 256 + blockIdx.z * 1 * 256 >= 256 * 1 * 1) return;
    fused_reshape_transpose_copy_reshape_1_kernel0_device((float* __restrict__)T_reshape,(float* __restrict__)placeholder);
    asm volatile(";; end_flag"); // jump back to the caller
}

extern "C" __global__  __attribute__((amdgpu_num_vgpr(25))) __attribute__((amdgpu_num_sgpr(30))) void fused_reshape_add_reshape_transpose_reshape_transpose_kernel0_device_wrapper(float* __restrict__ T_transpose, float* __restrict__ placeholder, float* __restrict__ placeholder1) {
    // Force the compiler to use all the index
    if (threadIdx.x + threadIdx.y * 256 + threadIdx.z * 1 * 256 >= 256 * 1 * 1) return;
    // if (blockIdx.x + blockIdx.y * 256 + blockIdx.z * 1 * 256 >= 256 * 1 * 1) return;
    fused_reshape_add_reshape_transpose_reshape_transpose_kernel0_device((float* __restrict__)T_transpose,(float* __restrict__)placeholder,(float* __restrict__)placeholder1);
    asm volatile(";; end_flag"); // jump back to the caller
}

extern "C" __global__  __attribute__((amdgpu_num_vgpr(25))) __attribute__((amdgpu_num_sgpr(30))) void fused_reshape_add_reshape_transpose_transpose_reshape_transpose_kernel0_device_wrapper(float* __restrict__ T_transpose, float* __restrict__ placeholder, float* __restrict__ placeholder1) {
    // Force the compiler to use all the index
    if (threadIdx.x + threadIdx.y * 256 + threadIdx.z * 1 * 256 >= 256 * 1 * 1) return;
    // if (blockIdx.x + blockIdx.y * 256 + blockIdx.z * 1 * 256 >= 256 * 1 * 1) return;
    fused_reshape_add_reshape_transpose_transpose_reshape_transpose_kernel0_device((float* __restrict__)T_transpose,(float* __restrict__)placeholder,(float* __restrict__)placeholder1);
    asm volatile(";; end_flag"); // jump back to the caller
}

extern "C" __global__  __attribute__((amdgpu_num_vgpr(25))) __attribute__((amdgpu_num_sgpr(30))) void fused_reshape_add_reshape_transpose_divide_reshape_kernel0_device_wrapper(float* __restrict__ T_reshape, float* __restrict__ placeholder, float* __restrict__ placeholder1) {
    // Force the compiler to use all the index
    if (threadIdx.x + threadIdx.y * 256 + threadIdx.z * 1 * 256 >= 256 * 1 * 1) return;
    // if (blockIdx.x + blockIdx.y * 256 + blockIdx.z * 1 * 256 >= 256 * 1 * 1) return;
    fused_reshape_add_reshape_transpose_divide_reshape_kernel0_device((float* __restrict__)T_reshape,(float* __restrict__)placeholder,(float* __restrict__)placeholder1);
    asm volatile(";; end_flag"); // jump back to the caller
}

extern "C" __global__  __attribute__((amdgpu_num_vgpr(25))) __attribute__((amdgpu_num_sgpr(30))) void fused_full_equal_reshape_kernel0_device_wrapper(signed char* __restrict__ T_reshape) {
    // Force the compiler to use all the index
    if (threadIdx.x + threadIdx.y * 256 + threadIdx.z * 1 * 256 >= 256 * 1 * 1) return;
    // if (blockIdx.x + blockIdx.y * 2 + blockIdx.z * 1 * 2 >= 2 * 1 * 1) return;
    fused_full_equal_reshape_kernel0_device((signed char* __restrict__)T_reshape);
    asm volatile(";; end_flag"); // jump back to the caller
}

extern "C" __global__  __attribute__((amdgpu_num_vgpr(25))) __attribute__((amdgpu_num_sgpr(30))) void fused_cast_take_broadcast_to_like_cast_take_add_1_kernel0_device_wrapper(float* __restrict__ T_add, float* __restrict__ placeholder, long* __restrict__ placeholder1, float* __restrict__ placeholder2, long* __restrict__ placeholder3) {
    // Force the compiler to use all the index
    if (threadIdx.x + threadIdx.y * 256 + threadIdx.z * 1 * 256 >= 256 * 1 * 1) return;
    // if (blockIdx.x + blockIdx.y * 256 + blockIdx.z * 1 * 256 >= 256 * 1 * 1) return;
    fused_cast_take_broadcast_to_like_cast_take_add_1_kernel0_device((float* __restrict__)T_add,(float* __restrict__)placeholder,(long* __restrict__)placeholder1,(float* __restrict__)placeholder2,(long* __restrict__)placeholder3);
    asm volatile(";; end_flag"); // jump back to the caller
}

extern "C" __global__  __attribute__((amdgpu_num_vgpr(25))) __attribute__((amdgpu_num_sgpr(30))) void fused_nn_softmax_1_kernel2_device_wrapper(float* __restrict__ T_softmax_maxelem, float* __restrict__ T_softmax_exp) {
    // Force the compiler to use all the index
    if (threadIdx.x + threadIdx.y * 256 + threadIdx.z * 1 * 256 >= 256 * 1 * 1) return;
    // if (blockIdx.x + blockIdx.y * 23 + blockIdx.z * 1 * 23 >= 23 * 1 * 1) return;
    fused_nn_softmax_1_kernel2_device((float* __restrict__)T_softmax_maxelem,(float* __restrict__)T_softmax_exp);
    asm volatile(";; end_flag"); // jump back to the caller
}

extern "C" __global__  __attribute__((amdgpu_num_vgpr(25))) __attribute__((amdgpu_num_sgpr(30))) void fused_mean_1_kernel1_device_wrapper(float* __restrict__ T_divide, float* __restrict__ placeholder_red) {
    // Force the compiler to use all the index
    if (threadIdx.x + threadIdx.y * 256 + threadIdx.z * 1 * 256 >= 256 * 1 * 1) return;
    // if (blockIdx.x + blockIdx.y * 2 + blockIdx.z * 1 * 2 >= 2 * 1 * 1) return;
    fused_mean_1_kernel1_device((float* __restrict__)T_divide,(float* __restrict__)placeholder_red);
    asm volatile(";; end_flag"); // jump back to the caller
}

extern "C" __global__  __attribute__((amdgpu_num_vgpr(25))) __attribute__((amdgpu_num_sgpr(30))) void fused_reshape_4_kernel0_device_wrapper(float* __restrict__ T_reshape, float* __restrict__ placeholder) {
    // Force the compiler to use all the index
    if (threadIdx.x + threadIdx.y * 256 + threadIdx.z * 1 * 256 >= 256 * 1 * 1) return;
    // if (blockIdx.x + blockIdx.y * 256 + blockIdx.z * 1 * 256 >= 256 * 1 * 1) return;
    fused_reshape_4_kernel0_device((float* __restrict__)T_reshape,(float* __restrict__)placeholder);
    asm volatile(";; end_flag"); // jump back to the caller
}

extern "C" __global__  __attribute__((amdgpu_num_vgpr(25))) __attribute__((amdgpu_num_sgpr(30))) void fused_variance_1_kernel0_device_wrapper(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_multiply_red) {
    // Force the compiler to use all the index
    if (threadIdx.x + threadIdx.y * 32 + threadIdx.z * 32 * 32 >= 32 * 32 * 1) return;
    // if (blockIdx.x + blockIdx.y * 15 + blockIdx.z * 1 * 15 >= 15 * 1 * 1) return;
    fused_variance_1_kernel0_device((float* __restrict__)placeholder,(float* __restrict__)placeholder1,(float* __restrict__)T_multiply_red);
    asm volatile(";; end_flag"); // jump back to the caller
}

extern "C" __global__  __attribute__((amdgpu_num_vgpr(25))) __attribute__((amdgpu_num_sgpr(30))) void fused_variance_1_kernel1_device_wrapper(float* __restrict__ T_divide, float* __restrict__ T_multiply_red) {
    // Force the compiler to use all the index
    if (threadIdx.x + threadIdx.y * 256 + threadIdx.z * 1 * 256 >= 256 * 1 * 1) return;
    // if (blockIdx.x + blockIdx.y * 2 + blockIdx.z * 1 * 2 >= 2 * 1 * 1) return;
    fused_variance_1_kernel1_device((float* __restrict__)T_divide,(float* __restrict__)T_multiply_red);
    asm volatile(";; end_flag"); // jump back to the caller
}

extern "C" __global__  __attribute__((amdgpu_num_vgpr(25))) __attribute__((amdgpu_num_sgpr(30))) void fused_reshape_cast_broadcast_to_like_where_kernel0_device_wrapper(float* __restrict__ T_where, signed char* __restrict__ placeholder, float* __restrict__ placeholder1) {
    // Force the compiler to use all the index
    if (threadIdx.x + threadIdx.y * 256 + threadIdx.z * 1 * 256 >= 256 * 1 * 1) return;
    // if (blockIdx.x + blockIdx.y * 256 + blockIdx.z * 1 * 256 >= 256 * 1 * 1) return;
    fused_reshape_cast_broadcast_to_like_where_kernel0_device((float* __restrict__)T_where,(signed char* __restrict__)placeholder,(float* __restrict__)placeholder1);
    asm volatile(";; end_flag"); // jump back to the caller
}

extern "C" __global__  __attribute__((amdgpu_num_vgpr(25))) __attribute__((amdgpu_num_sgpr(30))) void fused_nn_softmax_1_kernel3_device_wrapper(float* __restrict__ T_softmax_norm, float* __restrict__ T_softmax_exp, float* __restrict__ T_softmax_maxelem) {
    // Force the compiler to use all the index
    if (threadIdx.x + threadIdx.y * 256 + threadIdx.z * 1 * 256 >= 256 * 1 * 1) return;
    // if (blockIdx.x + blockIdx.y * 256 + blockIdx.z * 1 * 256 >= 256 * 1 * 1) return;
    fused_nn_softmax_1_kernel3_device((float* __restrict__)T_softmax_norm,(float* __restrict__)T_softmax_exp,(float* __restrict__)T_softmax_maxelem);
    asm volatile(";; end_flag"); // jump back to the caller
}

extern "C" __global__  void fused_nn_softmax_1_kernel0(
    void* func_l, int layers_l, int task_num_l, int task_offset_l, float** param_l,
    void* func_r, int layers_r, int task_num_r, int task_offset_r, float** param_r,
    int cu_partition) {
    asm volatile(";; caller_flag");
    return;
}

extern "C" __global__  void fused_reshape_add_add_kernel0(
    void* func_l, int layers_l, int task_num_l, int task_offset_l, float** param_l,
    void* func_r, int layers_r, int task_num_r, int task_offset_r, float** param_r,
    int cu_partition) {
    asm volatile(";; caller_flag");
    return;
}

extern "C" __global__  void fused_nn_batch_matmul_4_kernel0(
    void* func_l, int layers_l, int task_num_l, int task_offset_l, float** param_l,
    void* func_r, int layers_r, int task_num_r, int task_offset_r, float** param_r,
    int cu_partition) {
    asm volatile(";; caller_flag");
    return;
}

extern "C" __global__  void fused_nn_batch_matmul_5_kernel0(
    void* func_l, int layers_l, int task_num_l, int task_offset_l, float** param_l,
    void* func_r, int layers_r, int task_num_r, int task_offset_r, float** param_r,
    int cu_partition) {
    asm volatile(";; caller_flag");
    return;
}

extern "C" __global__  void fused_nn_softmax_1_kernel1(
    void* func_l, int layers_l, int task_num_l, int task_offset_l, float** param_l,
    void* func_r, int layers_r, int task_num_r, int task_offset_r, float** param_r,
    int cu_partition) {
    asm volatile(";; caller_flag");
    return;
}

extern "C" __global__  void fused_reshape_add_multiply_erf_multiply_add_multiply_reshape_kernel0(
    void* func_l, int layers_l, int task_num_l, int task_offset_l, float** param_l,
    void* func_r, int layers_r, int task_num_r, int task_offset_r, float** param_r,
    int cu_partition) {
    asm volatile(";; caller_flag");
    return;
}

extern "C" __global__  void fused_mean_1_kernel0(
    void* func_l, int layers_l, int task_num_l, int task_offset_l, float** param_l,
    void* func_r, int layers_r, int task_num_r, int task_offset_r, float** param_r,
    int cu_partition) {
    asm volatile(";; caller_flag");
    return;
}

extern "C" __global__  void fused_subtract_add_sqrt_divide_multiply_add_1_kernel0(
    void* func_l, int layers_l, int task_num_l, int task_offset_l, float** param_l,
    void* func_r, int layers_r, int task_num_r, int task_offset_r, float** param_r,
    int cu_partition) {
    asm volatile(";; caller_flag");
    return;
}

extern "C" __global__  void fused_nn_batch_matmul_3_kernel0(
    void* func_l, int layers_l, int task_num_l, int task_offset_l, float** param_l,
    void* func_r, int layers_r, int task_num_r, int task_offset_r, float** param_r,
    int cu_partition) {
    asm volatile(";; caller_flag");
    return;
}

extern "C" __global__  void fused_reshape_5_kernel0(
    void* func_l, int layers_l, int task_num_l, int task_offset_l, float** param_l,
    void* func_r, int layers_r, int task_num_r, int task_offset_r, float** param_r,
    int cu_partition) {
    asm volatile(";; caller_flag");
    return;
}

extern "C" __global__  void fused_reshape_transpose_copy_reshape_1_kernel0(
    void* func_l, int layers_l, int task_num_l, int task_offset_l, float** param_l,
    void* func_r, int layers_r, int task_num_r, int task_offset_r, float** param_r,
    int cu_partition) {
    asm volatile(";; caller_flag");
    return;
}

extern "C" __global__  void fused_reshape_add_reshape_transpose_reshape_transpose_kernel0(
    void* func_l, int layers_l, int task_num_l, int task_offset_l, float** param_l,
    void* func_r, int layers_r, int task_num_r, int task_offset_r, float** param_r,
    int cu_partition) {
    asm volatile(";; caller_flag");
    return;
}

extern "C" __global__  void fused_reshape_add_reshape_transpose_transpose_reshape_transpose_kernel0(
    void* func_l, int layers_l, int task_num_l, int task_offset_l, float** param_l,
    void* func_r, int layers_r, int task_num_r, int task_offset_r, float** param_r,
    int cu_partition) {
    asm volatile(";; caller_flag");
    return;
}

extern "C" __global__  void fused_reshape_add_reshape_transpose_divide_reshape_kernel0(
    void* func_l, int layers_l, int task_num_l, int task_offset_l, float** param_l,
    void* func_r, int layers_r, int task_num_r, int task_offset_r, float** param_r,
    int cu_partition) {
    asm volatile(";; caller_flag");
    return;
}

extern "C" __global__  void fused_full_equal_reshape_kernel0(
    void* func_l, int layers_l, int task_num_l, int task_offset_l, float** param_l,
    void* func_r, int layers_r, int task_num_r, int task_offset_r, float** param_r,
    int cu_partition) {
    asm volatile(";; caller_flag");
    return;
}

extern "C" __global__  void fused_cast_take_broadcast_to_like_cast_take_add_1_kernel0(
    void* func_l, int layers_l, int task_num_l, int task_offset_l, float** param_l,
    void* func_r, int layers_r, int task_num_r, int task_offset_r, float** param_r,
    int cu_partition) {
    asm volatile(";; caller_flag");
    return;
}

extern "C" __global__  void fused_nn_softmax_1_kernel2(
    void* func_l, int layers_l, int task_num_l, int task_offset_l, float** param_l,
    void* func_r, int layers_r, int task_num_r, int task_offset_r, float** param_r,
    int cu_partition) {
    asm volatile(";; caller_flag");
    return;
}

extern "C" __global__  void fused_mean_1_kernel1(
    void* func_l, int layers_l, int task_num_l, int task_offset_l, float** param_l,
    void* func_r, int layers_r, int task_num_r, int task_offset_r, float** param_r,
    int cu_partition) {
    asm volatile(";; caller_flag");
    return;
}

extern "C" __global__  void fused_reshape_4_kernel0(
    void* func_l, int layers_l, int task_num_l, int task_offset_l, float** param_l,
    void* func_r, int layers_r, int task_num_r, int task_offset_r, float** param_r,
    int cu_partition) {
    asm volatile(";; caller_flag");
    return;
}

extern "C" __global__  void fused_variance_1_kernel0(
    void* func_l, int layers_l, int task_num_l, int task_offset_l, float** param_l,
    void* func_r, int layers_r, int task_num_r, int task_offset_r, float** param_r,
    int cu_partition) {
    asm volatile(";; caller_flag");
    return;
}

extern "C" __global__  void fused_variance_1_kernel1(
    void* func_l, int layers_l, int task_num_l, int task_offset_l, float** param_l,
    void* func_r, int layers_r, int task_num_r, int task_offset_r, float** param_r,
    int cu_partition) {
    asm volatile(";; caller_flag");
    return;
}

extern "C" __global__  void fused_reshape_cast_broadcast_to_like_where_kernel0(
    void* func_l, int layers_l, int task_num_l, int task_offset_l, float** param_l,
    void* func_r, int layers_r, int task_num_r, int task_offset_r, float** param_r,
    int cu_partition) {
    asm volatile(";; caller_flag");
    return;
}

extern "C" __global__  void fused_nn_softmax_1_kernel3(
    void* func_l, int layers_l, int task_num_l, int task_offset_l, float** param_l,
    void* func_r, int layers_r, int task_num_r, int task_offset_r, float** param_r,
    int cu_partition) {
    asm volatile(";; caller_flag");
    return;
}

extern "C" __device__ __noinline__ dim3 get_3d_idx_256_1_1(int idx) {
  dim3 dim(256, 1, 1);
  dim3 result;
  result.x = idx % dim.x;
  result.y = idx / dim.x % dim.y;
  result.z = idx / (dim.x * dim.y);
  return result;
}

extern "C" __device__ __noinline__ dim3 get_3d_idx_32_4_1(int idx) {
  dim3 dim(32, 4, 1);
  dim3 result;
  result.x = idx % dim.x;
  result.y = idx / dim.x % dim.y;
  result.z = idx / (dim.x * dim.y);
  return result;
}

extern "C" __device__ __noinline__ dim3 get_3d_idx_23_1_1(int idx) {
  dim3 dim(23, 1, 1);
  dim3 result;
  result.x = idx % dim.x;
  result.y = idx / dim.x % dim.y;
  result.z = idx / (dim.x * dim.y);
  return result;
}

extern "C" __device__ __noinline__ dim3 get_3d_idx_2_1_1(int idx) {
  dim3 dim(2, 1, 1);
  dim3 result;
  result.x = idx % dim.x;
  result.y = idx / dim.x % dim.y;
  result.z = idx / (dim.x * dim.y);
  return result;
}

extern "C" __device__ __noinline__ dim3 get_3d_idx_32_32_1(int idx) {
  dim3 dim(32, 32, 1);
  dim3 result;
  result.x = idx % dim.x;
  result.y = idx / dim.x % dim.y;
  result.z = idx / (dim.x * dim.y);
  return result;
}

extern "C" __device__ __noinline__ dim3 get_3d_idx_16_16_1(int idx) {
  dim3 dim(16, 16, 1);
  dim3 result;
  result.x = idx % dim.x;
  result.y = idx / dim.x % dim.y;
  result.z = idx / (dim.x * dim.y);
  return result;
}

extern "C" __device__ __noinline__ dim3 get_3d_idx_2_10_12(int idx) {
  dim3 dim(2, 10, 12);
  dim3 result;
  result.x = idx % dim.x;
  result.y = idx / dim.x % dim.y;
  result.z = idx / (dim.x * dim.y);
  return result;
}

extern "C" __device__ __noinline__ dim3 get_3d_idx_5_24_1(int idx) {
  dim3 dim(5, 24, 1);
  dim3 result;
  result.x = idx % dim.x;
  result.y = idx / dim.x % dim.y;
  result.z = idx / (dim.x * dim.y);
  return result;
}

extern "C" __device__ __noinline__ dim3 get_3d_idx_15_1_1(int idx) {
  dim3 dim(15, 1, 1);
  dim3 result;
  result.x = idx % dim.x;
  result.y = idx / dim.x % dim.y;
  result.z = idx / (dim.x * dim.y);
  return result;
}

extern "C" __device__ __noinline__ dim3 get_3d_idx_10_6_12(int idx) {
  dim3 dim(10, 6, 12);
  dim3 result;
  result.x = idx % dim.x;
  result.y = idx / dim.x % dim.y;
  result.z = idx / (dim.x * dim.y);
  return result;
}

extern "C" __device__ __noinline__ dim3 get_3d_idx_5_40_1(int idx) {
  dim3 dim(5, 40, 1);
  dim3 result;
  result.x = idx % dim.x;
  result.y = idx / dim.x % dim.y;
  result.z = idx / (dim.x * dim.y);
  return result;
}

__global__ void get_3d_idx_caller(int* buf) {
    dim3 task_idx;

    task_idx = get_3d_idx_256_1_1(threadIdx.x);
    buf[task_idx.x] = task_idx.x;
    buf[task_idx.y] = task_idx.y;
    buf[task_idx.z] = task_idx.z;

    task_idx = get_3d_idx_32_4_1(threadIdx.x);
    buf[task_idx.x] = task_idx.x;
    buf[task_idx.y] = task_idx.y;
    buf[task_idx.z] = task_idx.z;

    task_idx = get_3d_idx_23_1_1(threadIdx.x);
    buf[task_idx.x] = task_idx.x;
    buf[task_idx.y] = task_idx.y;
    buf[task_idx.z] = task_idx.z;

    task_idx = get_3d_idx_2_1_1(threadIdx.x);
    buf[task_idx.x] = task_idx.x;
    buf[task_idx.y] = task_idx.y;
    buf[task_idx.z] = task_idx.z;

    task_idx = get_3d_idx_32_32_1(threadIdx.x);
    buf[task_idx.x] = task_idx.x;
    buf[task_idx.y] = task_idx.y;
    buf[task_idx.z] = task_idx.z;

    task_idx = get_3d_idx_16_16_1(threadIdx.x);
    buf[task_idx.x] = task_idx.x;
    buf[task_idx.y] = task_idx.y;
    buf[task_idx.z] = task_idx.z;

    task_idx = get_3d_idx_2_10_12(threadIdx.x);
    buf[task_idx.x] = task_idx.x;
    buf[task_idx.y] = task_idx.y;
    buf[task_idx.z] = task_idx.z;

    task_idx = get_3d_idx_5_24_1(threadIdx.x);
    buf[task_idx.x] = task_idx.x;
    buf[task_idx.y] = task_idx.y;
    buf[task_idx.z] = task_idx.z;

    task_idx = get_3d_idx_15_1_1(threadIdx.x);
    buf[task_idx.x] = task_idx.x;
    buf[task_idx.y] = task_idx.y;
    buf[task_idx.z] = task_idx.z;

    task_idx = get_3d_idx_10_6_12(threadIdx.x);
    buf[task_idx.x] = task_idx.x;
    buf[task_idx.y] = task_idx.y;
    buf[task_idx.z] = task_idx.z;

    task_idx = get_3d_idx_5_40_1(threadIdx.x);
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
