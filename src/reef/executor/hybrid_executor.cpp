#include "reef/executor/hybrid_executor.h"

namespace reef {
namespace executor {

HybridExecutor::HybridExecutor() {

}

HybridExecutor::~HybridExecutor() {

}

Status HybridExecutor::load_hybrid_model_from_file(
    const char* json_file_path,
    const char* profile_file_path,
    const char* trans_co_file_path,
    const char* preempt_co_file_path) 
{
    GPUModule_t trans_module, preempt_module;
    LOG(INFO) << std::string(trans_co_file_path);
    GPU_RETURN_STATUS(GPUModuleLoad(&trans_module, trans_co_file_path));
    GPU_RETURN_STATUS(GPUModuleLoad(&preempt_module, preempt_co_file_path));

    return load_hybrid_model_from_GPU_module(
        json_file_path,
        profile_file_path,
        trans_module,
        preempt_module
    );
}

Status HybridExecutor::load_hybrid_model_from_GPU_module(
    const char* json_file_path,
    const char* profile_file_path,
    GPUModule_t trans_module,
    GPUModule_t preempt_module)
{
    // 1. Init transformed module
    Status ret = load_model_from_GPU_module(json_file_path, trans_module);
    if (ret != Status::Succ) return ret;
    
    // 2. Init preemptable module
    return init_hybrid_executor(
                json_file_path,
                profile_file_path,
                trans_module,
                preempt_module
            );
}

Status HybridExecutor::init_hybrid_executor(
    const char* json_file_path,
    const char* profile_file_path,
    GPUModule_t trans_module,
    GPUModule_t preempt_module) 
{
    // TODO: load profile
    preempt_mod = preempt_module;
    trans_mod = trans_module;

    // 1. load preemptable kernels
    size_t num_kernels = model->kernels.size();
    preempt_kernels.resize(num_kernels);
    for (size_t i = 0; i < num_kernels; i++) {
        GPU_RETURN_STATUS(GPUModuleGetFunction(
            &preempt_kernels[i], preempt_mod, model->kernels[i].name.c_str()
        ));
    }

    // 2. allocate preempt flag
    GPU_RETURN_STATUS(GPUMalloc((GPUDevicePtr_t*)&preempt_flag, 4));
    int value = 0;
    GPU_RETURN_STATUS(GPUMemcpyHtoD(preempt_flag, &value, 4));
    // TODO: remove this
    GPU_RETURN_STATUS(GPUMalloc((GPUDevicePtr_t*)&task_slot_base, num_kernels*4));
    GPU_RETURN_STATUS(GPUMemset(task_slot_base, 0, num_kernels*4));

    // 3. prepare preemptable kernel args
    preempt_args.resize(num_kernels);
    for (int i = 0; i < num_kernels; i++) {
        auto &kernel_args = preempt_args[i];
        kernel_args.push_back(&preempt_flag);
        kernel_args.push_back(&task_slot_base);   
        auto &origin_args = raw_args[i];
        for (int j = 0; j < origin_args.size(); j++) {
            kernel_args.push_back(origin_args[j]);
        }
    }

    // 4. load profiles
    model_profile.reset(ModelProfile::from_json(profile_file_path));
    for (int i = 0; i < num_kernels; i++) {
        auto& kernel_arg = trans_args[i];
        kernel_arg.profile = model_profile->kernel_latency[model->kernels[i].name];
    }

    // 5. prepare task slots for wait-based preemption
    GPU_RETURN_STATUS(hipHostMalloc((void**)&task_slots_host, num_kernels * sizeof(int), hipHostMallocDefault));
    GPU_RETURN_STATUS(hipHostMalloc((void**)&task_slots_host_empty, num_kernels * sizeof(int), hipHostMallocDefault));
    memset(task_slots_host, 0, num_kernels * sizeof(int));
    memset(task_slots_host_empty, 0, num_kernels * sizeof(int));
    return Status::Succ;
}


Status HybridExecutor::set_preempt_flag(GPUDevicePtr_t flag) {
    GPU_RETURN_STATUS(GPUFree(preempt_flag)); // TODO: avoid double free
    preempt_flag = flag;
    return Status::Succ;
}

Status HybridExecutor::execute_preemptale(GPUStream_t stream) {
    for (int i = 0; i < this->model->kernels.size(); i++) {
        Status ret = launch_preempt_kernel(i, stream);
        if (ret != Status::Succ) return ret;
    }
    GPU_RETURN_STATUS(GPUStreamSynchronize(stream));
    return Status::Succ;
}

Status HybridExecutor::reset_task_slot_async(int kernel_offset, GPUStream_t stream) {
    GPU_RETURN_STATUS(GPUWriteValue32Async(
        stream, (GPUDevicePtr_t)((char*)task_slot_base + kernel_offset * 4), 0, 0
    ));
    return Status::Succ;
}

Status HybridExecutor::get_reset_kernel_idx(int start_inx, int& ret) {
    return Status::Succ; // TODO:
}

void HybridExecutor::reset_task_slots(hipStream_t stream) {
    ASSERT_GPU_ERROR(hipMemcpyHtoDAsync(task_slot_base, task_slots_host_empty, 4 * this->num_kernels(), stream));
}

void HybridExecutor::copy_be_kernel_offset( hipStream_t stream) {
    ASSERT_GPU_ERROR(hipMemcpyDtoHAsync(task_slots_host, task_slot_base, 4 * this->num_kernels(), stream));
}

int HybridExecutor::get_be_kernel_offset(int begin) {
    // TODO: binary search
    for (int i = begin; i < this->num_kernels(); i++) {
        int finished_num = task_slots_host[i];
        int required_num = trans_args[i].block_num;

        if (finished_num < required_num) return i;
    }
    // for (int i = begin; i > 0)
    return this->num_kernels();
}

Status HybridExecutor::launch_preempt_kernel(int kernel_offset, GPUStream_t stream) {
    KernelArg &kernel_arg = trans_args[kernel_offset];
    GPUFunction_t func = preempt_kernels[kernel_offset];
    // LOG(INFO) << "launch " << kernel_offset;
    GPUDevicePtr_t task_slot = (GPUDevicePtr_t)((char*)task_slot_base + kernel_offset * 4);
    this->preempt_args[kernel_offset][1] = &task_slot; // TODO:
    GPU_RETURN_STATUS(GPUModuleLaunchKernel(func,
        kernel_arg.task_dim.x, kernel_arg.task_dim.y, kernel_arg.task_dim.z,
        kernel_arg.thread_dim.x, kernel_arg.thread_dim.y, kernel_arg.thread_dim.z,
        0, stream, (void**)(this->preempt_args[kernel_offset].data()), 0
    ));
    return Status::Succ;
}

} // namespace executor
} // namespace reef