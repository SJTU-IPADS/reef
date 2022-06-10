#pragma once
#include <hip/hip_runtime.h>
#include "reef/util/common.h"

#define GPUInit hipInit
#define GPUDeviceGet hipDeviceGet
#define GPUModuleLoad hipModuleLoad
#define GPUModuleGetFunction hipModuleGetFunction
#define GPUMalloc hipMalloc
#define GPUMemcpyHtoD hipMemcpyHtoD
#define GPUMemcpyDtoH hipMemcpyDtoH
#define GPUModuleLaunchKernel hipModuleLaunchKernel
#define GPUStreamDefault hipStreamDefault
#define GPUStreamSynchronize hipStreamSynchronize
#define GPUDeviceSynchronize hipDeviceSynchronize
#define GPUStreamCreate hipStreamCreate
#define GPUStreamQuery hipStreamQuery
#define GPUStatusOK hipSuccess
#define GPUFree hipFree
#define GPUWriteValue32Async hipStreamWriteValue32
#define GPUClearHostQueue hipStreamClearQueue
#define GPUResetCU hipResetWavefronts
#define GPUMemset hipMemset

#define GPU_RETURN_STATUS(cmd) \
{\
    hipError_t error = cmd;\
    if (error != hipSuccess) {\
        LOG(ERROR) << "hip error: " << hipGetErrorString(error) << "at " << __FILE__ << ":" << __LINE__; \
        return Status::Fail;\
    }\
}

#define ASSERT_GPU_ERROR(cmd) \
{\
    hipError_t error = cmd;\
    if (error != hipSuccess) {\
        LOG(ERROR) << "hip error: " << hipGetErrorString(error) << "at " << __FILE__ << ":" << __LINE__; \
        exit(EXIT_FAILURE);\
    }\
}


namespace reef {
namespace executor {

typedef hipDeviceptr_t GPUDevicePtr_t;
typedef hipFunction_t GPUFunction_t;
typedef hipDevice_t GPUDevice_t;
typedef hipModule_t GPUModule_t;
typedef hipError_t GPUError_t;
typedef hipStream_t GPUStream_t;

typedef unsigned long long int GPUFunctionPtr_t;

bool GPUStreamEmpty(GPUStream_t s);

class GPUConfig {
public:
    static uint32_t get_num_cus();
    static Status get_kernel_address(const char* name, GPUModule_t mod, GPUFunctionPtr_t& ret);

    struct KernelResource {
        int shared_memory;
        int vgprs;
        int sgprs;
        int stack_size;   
    };

    static KernelResource max_resource(const KernelResource& kr1, const KernelResource& kr2);

    static Status get_kernel_resource(GPUFunction_t func, KernelResource& ret);

    static int calculate_occupancy(const KernelResource& resource, dim3 thread_idx);
};

} // namespace executor
} // namespace reef