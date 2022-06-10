#pragma once
#include "reef/executor/trans_executor.h"


namespace reef {
namespace server {
    class REEFScheduler;
} // namespace server
namespace executor {

// HybridExecutor contains two version of the model
//   (1) transformed version, which is used to perform dynamic kernel padding
//   (2) preemptable version, which is used to perform reset-based preemption (for best-effort tasks).
//
// The transformed version is inherited from TransExecutor.
// 
// The preemptable version adds preemption flag based on the raw model.
class HybridExecutor : public TransExecutor {

friend class server::REEFScheduler;

public:
    HybridExecutor();
    virtual ~HybridExecutor();
    Status load_hybrid_model_from_file(
        const char* json_file_path,
        const char* profile_file_path,
        const char* trans_co_file_path,
        const char* preempt_co_file_path);

    Status load_hybrid_model_from_GPU_module(
        const char* json_file_path,
        const char* profile_file_path,
        GPUModule_t trans_module,
        GPUModule_t preempt_module
    );

    Status execute_preemptale(GPUStream_t stream = GPUStreamDefault);

    Status set_preempt_flag(GPUDevicePtr_t flag);

    Status reset_task_slot_async(int kernel_offset, GPUStream_t stream);
    
    Status get_reset_kernel_idx(int start_inx, int& ret);

    void reset_task_slots(hipStream_t stream);

    void copy_be_kernel_offset(hipStream_t stream);

    int get_be_kernel_offset(int begin);
protected:
    Status init_hybrid_executor(
        const char* json_file_path,
        const char* profile_file_path,
        GPUModule_t trans_module,
        GPUModule_t preempt_module
    );

    Status launch_preempt_kernel(int kernel_offset, GPUStream_t stream);

protected:
    GPUModule_t preempt_mod;
    GPUModule_t trans_mod;

    std::vector<GPUFunction_t> preempt_kernels;
    std::vector<std::vector<GPUDevicePtr_t*>> preempt_args;
    GPUDevicePtr_t preempt_flag;
    GPUDevicePtr_t task_slot_base; // TODO: remove this 

    int* task_slots_host_empty;
    int* task_slots_host;
    
    std::shared_ptr<ModelProfile> model_profile;
}; 

} // namespace executor
} // namespace reef 



