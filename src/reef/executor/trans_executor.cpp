#include "reef/executor/trans_executor.h"

namespace reef {
namespace executor {

TransExecutor::TransExecutor() {}
TransExecutor::~TransExecutor() {}

Status TransExecutor::load_model_from_GPU_module(const char* json_file_path, GPUModule_t module) {
    Status ret = init_executor_base(json_file_path, module);
    if (ret != Status::Succ) return ret;
    return init_rt_executor(json_file_path, module);
}


Status TransExecutor::init_rt_executor(const char* json_file_path, GPUModule_t module) {
    size_t num_kernel_calls = model->kernels.size();
    int num_cus = GPUConfig::get_num_cus();
    trans_args.resize(num_kernel_calls);

    bool need_load_kernels = true; // TODO: move to class config

    // 1. fullfil the trans_args, which will be used to launch transformed kernels
    for (size_t i = 0; i < num_kernel_calls; i++) {
        KernelArg &kernel_arg = trans_args[i];
        std::string& kernel_name = model->kernels[i].name;

        uint32_t *launch_params = model->kernels[i].launch_params;
        kernel_arg.task_dim = dim3(launch_params[0],launch_params[1], launch_params[2]);
        kernel_arg.thread_dim = dim3(launch_params[3],launch_params[4], launch_params[5]);
        kernel_arg.block_num = launch_params[0] * launch_params[1] * launch_params[2];
        kernel_arg.block_offset = 0;
        kernel_arg.cu_lower = 0;
        kernel_arg.cu_upper = GPUConfig::get_num_cus();

        if (need_load_kernels) {
            RETURN_STATUS(
                GPUConfig::get_kernel_address(
                    kernel_name.c_str(), module, kernel_arg.funcion_pointer
                )
            );
            kernel_arg.kernel = kernels[kernel_name];
            RETURN_STATUS(
                GPUConfig::get_kernel_resource(
                    kernel_arg.kernel,
                    kernel_arg.resource
                );
            )
        }
    }

    // 2. prepare REAL kernel params (model params)
    size_t num_total_kernel_args = 0;
    size_t func_args_ptr_buffer_size = 0;
    for (size_t i = 0; i < num_kernel_calls; i++) {
        num_total_kernel_args += raw_args[i].size();
    }

    func_args_ptr_buffer_size = align_up(num_total_kernel_args * sizeof(float *), (size_t)4096);

    GPU_RETURN_STATUS(
        GPUMalloc((GPUDevicePtr_t*)&func_args_base_ptr, func_args_ptr_buffer_size)
    );
    size_t func_args_offset = 0;
    
    std::vector<GPUDevicePtr_t> host_args(num_total_kernel_args);

    for (size_t i = 0; i < num_kernel_calls; i++) {
        KernelArg &kernel_arg = trans_args[i];
        kernel_arg.args = (GPUDevicePtr_t)(
            (size_t)func_args_base_ptr + func_args_offset * sizeof(float*)
        );
        for (size_t arg_idx : model->kernels[i].args) {
            host_args[func_args_offset] = storage[arg_idx];
            func_args_offset ++;
        }
    }

    GPU_RETURN_STATUS(
        GPUMemcpyHtoD((GPUDevicePtr_t)func_args_base_ptr, (void*)host_args.data(), num_total_kernel_args * sizeof(float*))
    );

    // 3. calculate num_layers
    if (need_load_kernels) {
        for (size_t i = 0; i < num_kernel_calls; i++) {
            KernelArg &kernel_arg = trans_args[i];
            KernelInfo &info = model->kernels[i];
            std::string &kernel_name = info.name;
            GPUFunction_t func = kernels[kernel_name];

            int max_layers = GPUConfig::calculate_occupancy(
                kernel_arg.resource,
                kernel_arg.thread_dim
            );
            int num_layers = align_up(kernel_arg.block_num, num_cus) / num_cus;
            if (num_layers > max_layers) num_layers = max_layers;
            kernel_arg.min_occupancy = num_layers;
        }
    }

    // 4. prepare proxy kernels
    proxy_kernels.resize(10);
    proxy_kernels_nostack.resize(10);
    for (int i = 1; i <= 10; i++) {
        {
            std::stringstream kernel_name;
            kernel_name << REEF_PROXY_KERNEL_PREFIX() << i;

            GPUFunction_t proxy_kernel;
            GPU_RETURN_STATUS(GPUModuleGetFunction(
                &proxy_kernel, module, kernel_name.str().c_str())
            );
            proxy_kernels[i-1] = proxy_kernel;
        }

        {
            std::stringstream kernel_name;
            kernel_name << REEF_PROXY_KERNEL_NOSTACK_PREFIX() << i;

            GPUFunction_t proxy_kernel;
            GPU_RETURN_STATUS(GPUModuleGetFunction(
                &proxy_kernel, module, kernel_name.str().c_str()));
            proxy_kernels_nostack[i-1] = proxy_kernel;
        }
    }
    GPUConfig::KernelResource kr;
    RETURN_STATUS(GPUConfig::get_kernel_resource(proxy_kernels[0], kr));
    max_stack_size = kr.stack_size; // TODO: move to GPU interface
    return Status::Succ;
}

Status TransExecutor::launch_kernel(int kernel_offset, GPUStream_t stream) {
    std::string& func_name = this->model->kernels[kernel_offset].name;
    GPUFunction_t func = this->kernels[func_name];
    int num_cus = GPUConfig::get_num_cus();
    uint32_t *launch_params = this->model->kernels[kernel_offset].launch_params;

    KernelArg &kernel_arg = this->trans_args[kernel_offset];
    int logical_layers = align_up(kernel_arg.block_num, num_cus) / num_cus;
    int cu_partition = align_up(kernel_arg.block_num, logical_layers) / logical_layers;
    void* placeholder = nullptr;

    void *arg[] = {
        &placeholder,
        &(kernel_arg.min_occupancy),
        &(kernel_arg.block_num),
        &(kernel_arg.block_offset),
        &(kernel_arg.args),

        // These args are not actually used.
        &placeholder,
        &(kernel_arg.min_occupancy),
        &(kernel_arg.block_num),
        &(kernel_arg.block_offset),
        &(kernel_arg.args),

        
        &(kernel_arg.cu_upper),
    };
    // assert(this->model->shared_memory.find(func_name) != this->model->shared_memory.end());
    // std::cout << "shared: " << this->base_executor->model->shared_memory[func_name] << std::endl;
    // unsigned int logical_work_groups = launch_params[0] * launch_params[1] * launch_params[2];
    // unsigned int num_layers = align_up(logical_work_groups, (unsigned int) GPU_NUM_CU) / GPU_NUM_CU;
    // unsigned int physical_work_groups = num_layers * GPU_NUM_CU; // align_up(logical_work_groups, num_layers) / num_layers;
    
    // std::cout << func_name << std::endl;
    GPU_RETURN_STATUS(GPUModuleLaunchKernel(func,
        num_cus * kernel_arg.min_occupancy, 1, 1,
        launch_params[3] * launch_params[4] * launch_params[5], 1, 1,
        128, stream, arg, 0
    ));        
    return Status::Succ;
}

GPUFunction_t TransExecutor::get_proxy_kernel(const GPUConfig::KernelResource& kr) {
    // FIXME: move to GPU interface
    static int sgpr_bound[] = {    
        102, 102, 102, 102, 102,
        102, 102, 102, 88, 80
    };
    
    static int vgpr_bound[] = {
        256, 128, 84, 64, 48,
        40, 36, 32, 28, 28
    };
    int sgpr_idx = 0, vgpr_idx = 0;
    int occupancy = 10;
    for (int i = 1; i < 10; i++) {
        if (kr.vgprs > vgpr_bound[i] || kr.sgprs > sgpr_bound[i]) {
            assert(i > 0);
            occupancy = i;
            break;        
        }
    }
    if (kr.stack_size > 0)
        return proxy_kernels[occupancy - 1];
    else
        return proxy_kernels_nostack[occupancy - 1];
}

} // namespace executor
} // namespace reef