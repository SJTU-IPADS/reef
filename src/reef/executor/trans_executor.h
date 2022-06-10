#pragma once 
#include "reef/executor/executor_base.h"



namespace reef {
namespace executor {


// TransExecutor is used for both real-time tasks and best-effort tasks.
// Instead of using the raw GPU code from DL compiler, TransExecutor executes
// transformed GPU code that support dynamic kernel padding.
// There are mainly two transformations:
// 1. kernel args
//   The original kernel looks like:
//
//   __global__ void foo(float* a, float* b) { ... }
//
//   The transformed kernel looks like:
//
//   __global__ void foo(   
//     void* func_l, int layers_l, int task_num_l, int task_offset_l, float** param_l,
//     void* func_r, int layers_r, int task_num_r, int task_offset_r, float** param_r,
//     int cu_partition) { ... }
//
//   The param `param_l` and `param_r` should consist the original kernel arg `float* a, flaot* b`.
//
//   The execution of the transformed kernel must follow the `persistent thread` style.
//
//   TODO: currently, dynamic kernel padding only support 2 kernels with float params. 
//
// 2. proxy kernel
//   The new transformed kernels can be called by two ways: 
//     (1) directly launch the new kernel with new params (usually for test).
//     (2) through proxy kernel
//   
//   Currently, proxy kernel has the same args with transformed kernels.

class TransExecutor : public ExecutorBase {
public:
    TransExecutor();
    virtual ~TransExecutor();
    virtual Status load_model_from_GPU_module(const char* json_file_path, GPUModule_t module) override;

public:

    class KernelArg {
    public:
        GPUFunction_t kernel;
        GPUFunctionPtr_t funcion_pointer;
        dim3 task_dim;
        dim3 thread_dim;
        // GPUDeviceptr_t task_slots;
        int block_num;
        int block_offset;
        int cu_lower;
        int cu_upper;
        GPUDevicePtr_t args;

        int min_occupancy; // This is the minimal required occupancy for real-time task.
        GPUConfig::KernelResource resource;
        KernelProfile profile;    
    };

protected:
    Status init_rt_executor(const char* json_file_path, GPUModule_t module);

    virtual Status launch_kernel(int kernel_offset, GPUStream_t stream) override;
    
    GPUFunction_t get_proxy_kernel(const GPUConfig::KernelResource& kr);
protected:
    std::vector<KernelArg> trans_args;
    GPUDevicePtr_t func_args_base_ptr;

    std::vector<GPUFunction_t> proxy_kernels;
    std::vector<GPUFunction_t> proxy_kernels_nostack;
    int max_stack_size;

    virtual std::string REEF_PROXY_KERNEL_PREFIX() const {
        return "merge_framework_";
    } 
    virtual std::string REEF_PROXY_KERNEL_NOSTACK_PREFIX() const {
        return "merge_framework_nostack_";
    }

};

} // namespace executor
} // namespace reef