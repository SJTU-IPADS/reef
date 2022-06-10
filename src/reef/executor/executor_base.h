#pragma once
#include "reef/executor/model.h"
#include "reef/util/common.h"


#ifdef __REEF_HIP_GPU__
#include "reef/executor/hip/hip_impl.h"
#endif
#ifdef __REEF_CUDA_GPU__
#include "reef/executor/cuda/cuda_impl.h"
#endif

namespace reef {
namespace executor {

class ExecutorBase {
public:
    ExecutorBase();
    virtual ~ExecutorBase();

    Status load_model_from_file(const char* json_file_path, const char* co_file_path);

    virtual Status load_model_from_GPU_module(const char* json_file_path, GPUModule_t module);

    Status load_param_from_file(const char* param_file_path);

    Status set_input(const std::string& key, const std::vector<float>& value);

    Status set_input(int idx, const void* value, size_t len);

    Status set_input(const std::string& key, const void* value, size_t len);

    Status get_data_size(const std::string& key, size_t &size);

    Status get_output(std::vector<float>& out);

    Status get_output(void* out, size_t len);

    Status get_data(int idx, void* out, size_t len);

    Status execute(GPUStream_t stream = GPUStreamDefault); 

    Status execute_to(int idx, GPUStream_t stream = GPUStreamDefault);

    Status execute_kernel(int idx, GPUStream_t stream = GPUStreamDefault);

    size_t num_kernels() const;

    void set_stream(GPUStream_t stream);

    GPUStream_t stream() const;
    
    std::unique_ptr<Model> model;
protected:
    Status init_executor_base(const char* json_file_path, GPUModule_t module);

    virtual Status launch_kernel(int kernel_offset, GPUStream_t stream);

    Status find_storage_idx(const std::string& name, size_t &idx);

protected:

    std::vector<GPUDevicePtr_t> storage;
    std::unordered_map<std::string, GPUFunction_t> kernels;
    std::vector<std::vector<GPUDevicePtr_t*>> raw_args;

    GPUModule_t base_mod;
    GPUStream_t s;
};


} // namespace executor
} // namespace reef