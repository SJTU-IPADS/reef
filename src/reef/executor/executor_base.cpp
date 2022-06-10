#include "reef/executor/executor_base.h"
#include <glog/logging.h>

namespace reef {
namespace executor{

ExecutorBase::ExecutorBase() {

}

ExecutorBase::~ExecutorBase() {
    // TODO: free GPU memory
}

Status ExecutorBase::load_model_from_file(
    const char* json_file_path, 
    const char* co_file_path)
{
    GPU_RETURN_STATUS(GPUInit(0));
    // CUcontext ctx;
    GPUDevice_t device;
    GPU_RETURN_STATUS(GPUDeviceGet(&device, 0));
    // GPU_RETURN_STATUS(cuCtxCreate(&ctx, 0, device));
    GPUModule_t mod;
    GPU_RETURN_STATUS(GPUModuleLoad(&mod, co_file_path));
    return this->load_model_from_GPU_module(json_file_path, mod);
}

Status ExecutorBase::load_model_from_GPU_module(
    const char* json_file_path, GPUModule_t mod) {
    return init_executor_base(json_file_path, mod);
}


Status ExecutorBase::init_executor_base(
    const char* json_file_path,
    GPUModule_t mod)
{
    base_mod = mod;

    // 1. load json model file
    model.reset(Model::from_json(json_file_path));
    if (model.get() == nullptr) RETURN_STATUS(Status::NotFound);

    // 2. load hip kernels
    for (KernelInfo &kernel_info : model->kernels) {
        GPUFunction_t kernel;
        GPU_RETURN_STATUS(
            GPUModuleGetFunction(&kernel, mod, kernel_info.name.c_str())
        );
        kernels.emplace(kernel_info.name, kernel);
    }

    // 3. allocate device storage
    for (StorageInfo &storage_info : model->storage) {
        size_t stype_size = Model::get_stype_size(storage_info.stype);
        size_t storage_size = stype_size * storage_info.size;
        GPUDevicePtr_t device_ptr;
        std::vector<char> temp;
        temp.resize(storage_size, 0);
        GPU_RETURN_STATUS(GPUMalloc((GPUDevicePtr_t*)&device_ptr, storage_size));
        GPU_RETURN_STATUS(GPUMemcpyHtoD(device_ptr, temp.data(), storage_size));
        storage.push_back(device_ptr);
    }

    // 4. map args to storage
    raw_args.reserve(model->kernels.size());
    for (KernelInfo &kernel_info : model->kernels) {
        std::vector<GPUDevicePtr_t*> kernel_arg;
        for (size_t arg_idx : kernel_info.args) {
            // assert(arg_idx < storage.size());
            kernel_arg.push_back(&storage[arg_idx]);
        }
        raw_args.push_back(kernel_arg);
    }

    LOG(INFO) << "create base model stream";
    GPU_RETURN_STATUS(hipStreamCreateWithWindowSize(&s, 16));
    return Status::Succ;
}

Status ExecutorBase::load_param_from_file(
    const char* param_file_path) {
    std::unique_ptr<ModelParam> params(ModelParamParser::parse_from_file(param_file_path));
    for (size_t i = 0; i < storage.size(); i++) {
        StorageInfo& storage_info = this->model->storage[i];
        if (params->find(storage_info.name) == params->end()) 
            continue;
        auto &array = params->at(storage_info.name);
        GPU_RETURN_STATUS(GPUMemcpyHtoD(
            (GPUDevicePtr_t)storage[i], array.data(), 
            array.size() * sizeof(float))); 
    }
    return Status::Succ;
}

Status ExecutorBase::get_data_size(const std::string& key, size_t &size) {
    size_t input_storage_idx;
    if (find_storage_idx(key, input_storage_idx) != Status::Succ) RETURN_STATUS(Status::NotFound);   
    StorageInfo& storage_info = this->model->storage[input_storage_idx];
    size = Model::get_stype_size(storage_info.stype) * storage_info.size;
    return Status::Succ;
}

Status ExecutorBase::set_input(
    const std::string& key, const std::vector<float>& value) {
    return set_input(key, (void*)value.data(), value.size() * sizeof(float));
}

Status ExecutorBase::set_input(const std::string& key, const void* value, size_t len) {
    size_t input_storage_idx;
    if (find_storage_idx(key, input_storage_idx) != Status::Succ) RETURN_STATUS(Status::NotFound);
    StorageInfo& storage_info = this->model->storage[input_storage_idx];
    size_t storage_size = Model::get_stype_size(storage_info.stype) * storage_info.size;
    if (len < storage_size) RETURN_STATUS(Status::OutOfRange);
    GPU_RETURN_STATUS(GPUMemcpyHtoD(
        (GPUDevicePtr_t)this->storage[input_storage_idx], (void*)value, 
        storage_size)
    );
    return Status::Succ;
}

Status ExecutorBase::set_input(int idx, const void* value, size_t len) {
    if (idx >= storage.size()) RETURN_STATUS(Status::OutOfRange);
    StorageInfo& storage_info = this->model->storage[idx];
    size_t storage_size = Model::get_stype_size(storage_info.stype) * storage_info.size;
    if (len < storage_size) RETURN_STATUS(Status::OutOfRange);
    GPU_RETURN_STATUS(GPUMemcpyHtoD(
        (GPUDevicePtr_t)this->storage[idx], (void*)value, 
        storage_size)
    );
    return Status::Succ;
}

Status ExecutorBase::get_output(std::vector<float>& out) {
    size_t input_storage_idx;
    if (find_storage_idx("output", input_storage_idx) != Status::Succ) RETURN_STATUS(Status::NotFound);
    StorageInfo& storage_info = this->model->storage[input_storage_idx];
    if (Model::get_stype_size(storage_info.stype) != sizeof(float)) RETURN_STATUS(Status::Fail);
    out.resize(storage_info.size);
    return get_data(input_storage_idx, (void*)out.data(), storage_info.size * sizeof(float));
}

Status ExecutorBase::get_output(void* out, size_t len) {
    size_t input_storage_idx;
    if (find_storage_idx("output", input_storage_idx) != Status::Succ) RETURN_STATUS(Status::NotFound);
    StorageInfo& storage_info = this->model->storage[input_storage_idx];
    size_t storage_size = Model::get_stype_size(storage_info.stype) * storage_info.size;
    if (len < storage_size) RETURN_STATUS(Status::Fail);
    return get_data(input_storage_idx, out, len);
}

Status ExecutorBase::get_data(int idx, void* out, size_t len) {
    if (idx >= this->storage.size()) RETURN_STATUS(Status::OutOfRange);
    StorageInfo& storage_info = this->model->storage[idx];
    size_t storage_size = Model::get_stype_size(storage_info.stype) * storage_info.size;
    if (len < storage_size) RETURN_STATUS(Status::Fail);
    GPU_RETURN_STATUS(GPUMemcpyDtoH(
        out, (GPUDevicePtr_t)this->storage[idx], storage_size
    ));
    return Status::Succ;
}

Status ExecutorBase::find_storage_idx(const std::string& name, size_t& idx) {
    // TODO: O(n) -> O(1)
    for (size_t i = 0; i < this->storage.size(); i++) {
        StorageInfo& storage_info = this->model->storage[i];
        if (storage_info.name == name) {
            idx = i;
            return Status::Succ;
        }
    }
    RETURN_STATUS(Status::NotFound);
    return Status::NotFound; // otherwise, the compiler thinks no return value.
}

size_t ExecutorBase::num_kernels() const {
    return model->kernels.size();
}


void ExecutorBase::set_stream(GPUStream_t stream) {
    s = stream;
}


GPUStream_t ExecutorBase::stream() const {
    return s;
}

Status ExecutorBase::execute(GPUStream_t stream) {
    execute_to(num_kernels());
    return Status::Succ;
}

Status ExecutorBase::execute_to(int idx, GPUStream_t stream) {
    for (int i = 0; i < idx; i++) {
        RETURN_STATUS(launch_kernel(i, stream));
    }  
    GPU_RETURN_STATUS(GPUStreamSynchronize(stream));
    return Status::Succ;
}

Status ExecutorBase::execute_kernel(int idx, GPUStream_t stream) {
    if (idx >= num_kernels()) RETURN_STATUS(Status::OutOfRange);
    RETURN_STATUS(launch_kernel(idx, stream));
    GPU_RETURN_STATUS(GPUStreamSynchronize(stream));
    return Status::Succ;
}

Status ExecutorBase::launch_kernel(int kernel_offset, GPUStream_t stream) {
    int i = kernel_offset;
    std::string& func_name = this->model->kernels[i].name;
    GPUFunction_t func = this->kernels[func_name];
    uint32_t *launch_params = this->model->kernels[i].launch_params;
    // std::cout << func_name << std::endl;
    GPU_RETURN_STATUS(GPUModuleLaunchKernel(func,
        launch_params[0], launch_params[1], launch_params[2],
        launch_params[3], launch_params[4], launch_params[5],
        0, stream, (void **)this->raw_args[i].data(), 0
    ));
    return Status::Succ;
}

} // namespace executor
} // namespace reef