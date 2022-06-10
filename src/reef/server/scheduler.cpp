#include "reef/server/scheduler.h"
#include "reef/util/common.h"

#define ENABLE_TASK_CV 

namespace reef {
namespace server {

REEFScheduler::REEFScheduler(ScheduleMode _mode) : 
    mode(_mode),
    model_pool(this->model_pool_capacity),
    model_pool_size(0),
    task_idx_pool(0),
    be_queues(max_num_be_queues),
    be_queue_cnt(0),
    _shutdown(false),
    preempted(false),
    wait_sync(false),
    preempt_count(0),
    preempt_latency_sum(0),
    kernel_sel_latency_sum(0),
    kernel_sel_count(0)
{
    ASSERT_GPU_ERROR(GPUStreamCreate(&execute_stream));
    ASSERT_GPU_ERROR(GPUStreamCreate(&preempt_stream));
    ASSERT_GPU_ERROR(GPUMalloc((void**)&preempt_flag, 4));
    ASSERT_GPU_ERROR(GPUWriteValue32Async(preempt_stream, preempt_flag, 0, 0));
    ASSERT_GPU_ERROR(GPUStreamSynchronize(preempt_stream));
    if (mode == WaitPreempt) {
        be_stream_device_queue_cap = 1024;
    } else {
        be_stream_device_queue_cap = 2;
    }
}

REEFScheduler::~REEFScheduler() {
    
}


Status REEFScheduler::create_task_queue(std::shared_ptr<TaskQueue>& ret, bool rt) {
    executor::GPUStream_t stream;
    if (rt) {
        LOG(INFO) << "create rt stream";
        GPU_RETURN_STATUS(hipStreamCreateWithWindowSize(&stream, 1024));
    } else {
        LOG(INFO) << "create be stream";
        GPU_RETURN_STATUS(hipStreamCreateWithWindowSize(
            &stream, be_stream_device_queue_cap
        ));
    }
    ret = std::make_shared<TaskQueue>();
    ret->stream = stream;
    return Status::Succ;
}

Status REEFScheduler::load_model(
    const std::string& model_dir,
    const std::string& model_name,
    ModelID& mid
) {
    std::string rt_co_path = model_dir + "/" + model_name + ".trans.co";
    std::string be_co_path = model_dir + "/" + model_name + ".be.co";
    std::string json_path = model_dir + "/" + model_name + ".json";
    std::string profile_path = model_dir + "/" + model_name + ".profile.json";
    
    return load_model(rt_co_path, be_co_path, json_path, profile_path, "", mid);
}

Status REEFScheduler::load_model(
    const std::string& rt_co_path,
    const std::string& be_co_path,
    const std::string& json_path,
    const std::string& profile_path,
    const std::string& param_path,
    ModelID& mid
) {
    std::shared_ptr<Model> model(new Model);
    RETURN_STATUS(model->executor.load_hybrid_model_from_file(
        json_path.c_str(),
        profile_path.c_str(),
        rt_co_path.c_str(),
        be_co_path.c_str()
    ));
    model->qid = rt_queue_id; // rt queue as default
    if (param_path.size() > 0) {
        RETURN_STATUS(model->executor.load_param_from_file(param_path.c_str()));
    }
    auto idx = model_pool_size.fetch_add(1);
    if (idx >= model_pool_capacity) {
        LOG(ERROR) << "model pool is full";
        RETURN_STATUS(Status::Fail);
    }
    model->executor.set_preempt_flag(preempt_flag);
    model_pool[idx] = std::move(model);
    LOG(INFO) << "load model from " << json_path << ", idx: " << idx;
    mid = idx;
    return Status::Succ;
}

Status REEFScheduler::create_queue(
    const TaskQueueType& qtp,
    QueueID& qid
) {
    if (qtp == TaskQueueType::RealTimeQueue) {
        qid = rt_queue_id;
        if (rt_queue.get() == nullptr) {
            assert(create_task_queue(rt_queue, true) == Status::Succ);
        }
        return Status::Succ;
    }
    std::shared_ptr<TaskQueue> q;
    RETURN_STATUS(create_task_queue(q, false));
    {
        // writer lock
        std::unique_lock<std::mutex> lock(be_queues_mtx);
        auto idx = be_queue_cnt;
        if (idx >= max_num_be_queues) RETURN_STATUS(Status::Full);
        be_queues[idx] = std::move(q);
        be_queue_cnt++;
        qid = idx;
    }
    return Status::Succ;
}

Status REEFScheduler::bind_model_queue(
    const QueueID& qid,
    const ModelID& mid
) {
    if (model_pool_size.load() <= mid) RETURN_STATUS(Status::OutOfRange);
    if (be_queue_cnt <= qid && qid != rt_queue_id) RETURN_STATUS(Status::OutOfRange);
    model_pool[mid]->qid = qid;
    return Status::Succ;
}

Status REEFScheduler::new_task(
    const ModelID& mid,
    TaskID& tid
) {
    if (model_pool_size.load() <= mid) RETURN_STATUS(Status::OutOfRange);
    auto &model = model_pool[mid];
    std::shared_ptr<Task> task(new Task);
    task->model = model;
    task->id = task_idx_pool.fetch_add(1);
    task->qid = model->qid;
    task->block_offset = 0;
    task->kernel_offset = 0;
    task->launch_offset = 0;
    task->state = TaskState::Init;
    task->submit = std::chrono::system_clock::now();
    task->preempted = false;
    task->padding = false;
    task->padding_to_finish = false;
    if (model->qid == rt_queue_id) {
        rt_queue->task_queue.push(task);
    } else {
        be_queues[model->qid]->task_queue.push(task);
    }
    tid = task->id;
    {
        std::unique_lock<std::mutex> lock(task_cnt_mtx);
        if (task_cnt == 0) 
            task_cnt_cv.notify_all();
        task_cnt++;
    }
    {
        std::unique_lock<std::mutex> lock(task_pool_mtx);
        task_pool.insert({tid, task});
    }
    return Status::Succ;
}

Status REEFScheduler::get_task(TaskID tid, std::shared_ptr<Task>& t) {
    std::shared_ptr<Task> task;
    {
        std::unique_lock<std::mutex> lock(task_pool_mtx);
        auto res = task_pool.find(tid);
        if (res == task_pool.end()) RETURN_STATUS(Status::NotFound);
        task = res->second;
    }
    t = task;
    return Status::Succ;
}

Status REEFScheduler::wait_task(TaskID tid) {
    std::shared_ptr<Task> task;
    RETURN_STATUS(get_task(tid, task));
#ifdef ENABLE_TASK_CV
    {
        std::unique_lock<std::mutex> lock(task->mtx);
        while (task->state != TaskState::Finish) {
            task->cv.wait(lock);
        }
    }
#else
    while (task->state != TaskState::Finish) {
        usleep(10);
    }
#endif
    // {
    //     std::unique_lock<std::mutex> lock(task_pool_mtx);
    //     auto res = task_pool.find(tid);
    //     if (res == task_pool.end()) RETURN_STATUS(Status::NotFound);
    //     task_pool.erase(res);
    // }
    return Status::Succ;
}

void REEFScheduler::set_wait_sync(bool value) {
    wait_sync = value;
}

Status REEFScheduler::get_data_size(ModelID mid, const std::string& name, size_t& size) {
    if (mid >= model_pool_size.load()) RETURN_STATUS(Status::OutOfRange);
    auto &model = model_pool[mid];
    return model->executor.get_data_size(name, size);
}


Status REEFScheduler::set_input(ModelID mid, const void* data, size_t len, const std::string& name) {
    if (mid >= model_pool_size.load()) RETURN_STATUS(Status::OutOfRange);
    auto &model = model_pool[mid];
    return model->executor.set_input(name, data, len);
}

Status REEFScheduler::get_output(ModelID mid, void* data, size_t len, const std::string& name) {
   if (mid >= model_pool_size.load()) RETURN_STATUS(Status::OutOfRange);
   auto &model = model_pool[mid];
   return model->executor.get_output(data, len);
}

Status REEFScheduler::run() {
    if (scheduler.get() != nullptr) RETURN_STATUS(Status::Fail);
    if (rt_queue.get() == nullptr) {
        assert(create_task_queue(rt_queue, true) == Status::Succ);
    }
    scheduler.reset(new std::thread([this]{
        // make sure the real_time queue is created for convenience.
        while (true) {
            this->loop_body();
            if (this->_shutdown.load()) return;
        }
    }));
    return Status::Succ;
}

Status REEFScheduler::shutdown() {
    _shutdown.store(true);
    scheduler->join();
    return Status::Succ;
}

REEFScheduler::ScheduleMode REEFScheduler::sche_mode() const {
    return mode;
}


void REEFScheduler::set_be_stream_cap(int value) {
    be_stream_device_queue_cap = value;
}

void REEFScheduler::loop_body() {
    // Real-time Mode:
    rtmode:
    while (true) {
        if (rt_queue->task_queue.empty()) goto bemode;

        preempt_be_tasks();

        auto rt_task = rt_queue->task_queue.front();
        rt_queue->task_queue.pop();
        execute_rt_task(rt_task);
    }


    // Best-effort Mode:
    bemode:
    auto be_queue_num = be_queue_cnt;
    for (int i = 0; i < be_queue_cnt; i++) {
        auto &be_queue = be_queues[i];
        while (!be_queue->task_queue.empty()) {
            if (!rt_queue->task_queue.empty()) {
                goto rtmode;
            }
            auto be_task = be_queue->task_queue.front();
            execute_be_task(be_task, be_queue);
            if (!rt_queue->task_queue.empty()) {
                goto rtmode;
            }
            if (be_task->state == TaskState::Finish) {
                be_queue->task_queue.pop();
#ifdef ENABLE_TASK_CV
                {
                    std::unique_lock<std::mutex> lock(be_task->mtx);
                    be_task->cv.notify_all();
                }
#endif
                continue;
            }
            break;
        }
    }
}

void REEFScheduler::preempt_reset() {
    // step 1: reset device queue
    // actually, this step should be the second one,
    // but we can overlap this with the host queue reset.
    ASSERT_GPU_ERROR(GPUWriteValue32Async(preempt_stream, preempt_flag, 1, 0));
    auto num_be_queues = be_queue_cnt;
    
    // step 2: reset host queue
    for (int i = 0; i < num_be_queues; i++) {
        uint64_t temp;
        ASSERT_GPU_ERROR(GPUClearHostQueue(&temp, be_queues[i]->stream));
        if (!be_queues[i]->task_queue.empty()) {
            auto task = be_queues[i]->task_queue.front();
            if (task->state == TaskState::Executing) {
                LOG(INFO) << task->kernel_offset << ", " << task->launch_offset;
                task->kernel_offset = std::max(
                    task->launch_offset - (int)temp - be_stream_device_queue_cap,
                    task->kernel_offset
                );
                LOG(INFO) << "new kernel_offset " << task->kernel_offset;
                task->state = TaskState::Waiting;
                task->preempted = true;
            }
        }
    }

    // step 3: reset CUs
    for (int i = 0; i < be_stream_device_queue_cap + 1; i++)
        ASSERT_GPU_ERROR(GPUResetCU());

    GPUStreamSynchronize(preempt_stream);
    if (wait_sync) {
        // GPUDeviceSynchronize();
        for (int i = 0; i < num_be_queues; i++) {
            while (GPUStreamQuery(be_queues[i]->stream) != GPUStatusOK) {
            // if (GPUStreamQuery(preempt_stream) != GPUStatusOK) {
                ASSERT_GPU_ERROR(GPUResetCU());
                sched_yield();
            }
        }
    }
}

void REEFScheduler::preempt_wait() {
    auto start = std::chrono::system_clock::now();
    int value = 1;
    // ASSERT_GPU_ERROR(hipStreamWriteValue32(preempt_stream, preempt_flag, 1, 0));

    ASSERT_GPU_ERROR(hipMemcpyHtoDAsync(preempt_flag, &value, 4, preempt_stream));
    ASSERT_GPU_ERROR(hipStreamSynchronize(preempt_stream));
    auto set_flag = std::chrono::system_clock::now();
    ASSERT_GPU_ERROR(hipDeviceSynchronize());
    auto end = std::chrono::system_clock::now();
    auto set_flag_duration = std::chrono::duration_cast<std::chrono::microseconds>(set_flag-start).count();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
    for (int i = 0; i < be_queue_cnt; i++) {
        auto &be_queue = be_queues[i]->task_queue;
        if (!be_queue.empty()) {
            auto be_task = be_queue.front();
            be_task->model->executor.copy_be_kernel_offset(preempt_stream);
            be_task->model->executor.reset_task_slots(preempt_stream);
            ASSERT_GPU_ERROR(hipStreamSynchronize(preempt_stream));
            auto kernel_offset = be_task->model->executor.get_be_kernel_offset(be_task->kernel_offset);
            be_task->kernel_offset = be_task->kernel_offset < kernel_offset ? kernel_offset : be_task->kernel_offset;
        }
    }
    // LOG(INFO) << "preempt latency: " << duration << "us, set flag: " << set_flag_duration;
}

void REEFScheduler::preempt_be_tasks() {
    if (preempted) return;
    preempted = true;
    LOG(INFO) << "preempt";
    auto start = std::chrono::system_clock::now();
    switch (this->mode) {
    case ScheduleMode::NoPreempt: {
        ASSERT_GPU_ERROR(GPUDeviceSynchronize());
        for (int i = 0; i < be_queue_cnt; i++) {
            auto &be_queue = be_queues[i]->task_queue;
            if (!be_queue.empty()) {
                auto be_task = be_queue.front();
                if (be_task->state == TaskState::Executing) {
                    be_task->kernel_offset = be_task->launch_offset;
                }
            }
        }
        break;
    }
    case ScheduleMode::Reset:
    case ScheduleMode::REEF: {
        preempt_reset();
        break;
    }
    case ScheduleMode::WaitPreempt: {
        preempt_wait();
        break;
    }
    default:
        break;
    }

    auto end = std::chrono::system_clock::now();
    preempt_count++;
    auto latency = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
    preempt_latency_sum += latency;
    LOG(INFO) << "preempt latency: " << latency << " us";

    for (int i = 0; i < be_queue_cnt; i++) {
        auto &be_queue = be_queues[i]->task_queue;
        if (!be_queue.empty()) {
            auto be_task = be_queue.front();
            if (be_task->state == TaskState::Executing) {
//                 if (be_task->kernel_offset >= be_task->model->executor.num_kernels()) {
//                     be_task->end = std::chrono::system_clock::now();
// #ifdef ENABLE_TASK_CV
//                     {
//                         std::unique_lock<std::mutex> lock(be_task->mtx);
//                         be_task->cv.notify_all();
//                     }
// #endif
//                     be_task->state = TaskState::Finish;
//                     be_queue.pop();
//                 } else {
//                     LOG(INFO) << be_task->kernel_offset;
                    be_task->state = TaskState::Waiting;
            }
        }
    
    }
}

int64_t REEFScheduler::avg_preempt_latency() const {
    if (preempt_count == 0) return 0;
    return preempt_latency_sum / preempt_count;
}

int64_t REEFScheduler::avg_kernel_sel_latency() const {
    if (kernel_sel_count == 0) return 0;
    return kernel_sel_latency_sum / kernel_sel_count;
}

void REEFScheduler::reset_preempt_flag_async() {
    int preempt_value_false = 0;
    hipStreamWriteValue32(preempt_stream, preempt_flag, 0, 0);
}

void REEFScheduler::execute_rt_task(std::shared_ptr<Task> &task) {
    LOG(INFO) << "start rt task";
    task->state = TaskState::Executing;
    task->start = std::chrono::system_clock::now();
    // auto &exe = task->model->executor;
    // exe.execute(rt_queue->stream);
    if (mode == ScheduleMode::REEF) {
        ASSERT_GPU_ERROR(hipStreamWriteValue32(rt_queue->stream, preempt_flag, 0, 0));
        dynamic_kernel_padding(task);
        ASSERT_GPU_ERROR(GPUStreamSynchronize(rt_queue->stream));
    }
    else {
        task->model->executor.execute(rt_queue->stream);
    }
    task->end = std::chrono::system_clock::now();
    task->state = TaskState::Finish;
#ifdef ENABLE_TASK_CV
    {
        std::unique_lock<std::mutex> lock(task->mtx);
        task->cv.notify_all();
    }
#endif
    LOG(INFO) << "rt task finish";
    return;
}

executor::GPUFunction_t REEFScheduler::get_proxy_kernel(
    const executor::GPUConfig::KernelResource& resource, 
    executor::HybridExecutor* rt_executor,
    executor::HybridExecutor* be_executor
) {   
    // TODO: GPU independent implementation?
    executor::HybridExecutor* target_executor = rt_executor;
    if (be_executor != nullptr && be_executor->max_stack_size > rt_executor->max_stack_size)
        target_executor = be_executor;
    
    return target_executor->get_proxy_kernel(resource);
}

void REEFScheduler::dynamic_kernel_padding(std::shared_ptr<Task>& rt_task) {
    auto &rt_executor = rt_task->model->executor;
    auto &rt_model = rt_executor.model;
    std::vector<std::shared_ptr<Task>> complete_be_tasks;
    auto start = std::chrono::system_clock::now();

    auto current_time_point = start;

    for (int i = 0; i < rt_executor.num_kernels(); i++) {
        auto sel_start = std::chrono::system_clock::now();
        executor::KernelInfo &rt_kernel = rt_model->kernels[i];
        executor::TransExecutor::KernelArg &rt_args = rt_executor.trans_args[i];
        bool can_merge = false;
        // Occupancy is the physical GPU occupancy, which means the number of simultanous 
        // wavefront per CU.
        int required_occupancy = rt_args.min_occupancy;
        int rt_occupancy = required_occupancy;
        int be_occupancy = 0;

        int rt_latency = rt_args.profile.total_latency;

        int final_occupancy = required_occupancy;
        int final_be_block_num = 0;
        executor::GPUConfig::KernelResource final_kernel_resource = rt_args.resource;
        // int final_vgprs = rt_args.vgprs;
        // int final_sgprs = rt_args.sgprs;
        // int final_shared_memory = rt_args.shared_memory;
        // int final_stack_size = rt_args.stack_size;
        int final_block_size = rt_args.thread_dim.x * rt_args.thread_dim.y * rt_args.thread_dim.z;
        

        executor::GPUFunctionPtr_t rt_func = rt_args.funcion_pointer;
        executor::GPUFunctionPtr_t be_func = rt_func; // nullptr;

        int logical_layers = 
            align_up<int>(rt_args.block_num, executor::GPUConfig::get_num_cus()) / executor::GPUConfig::get_num_cus();
        int cu_partition = executor::GPUConfig::get_num_cus(); 

        executor::TransExecutor::KernelArg *final_be_args = &rt_args;
        executor::KernelInfo *final_be_info = nullptr;
        Task* final_be_task = nullptr;
        int final_be_block_offset = 0;
        auto num_be_queues = be_queue_cnt;
        for (int queue_idx = 0; queue_idx < num_be_queues; queue_idx++) {
            auto &be_tqueue = be_queues[queue_idx];
            auto &be_queue = be_tqueue->task_queue;
            if (be_queue.empty()) continue;
            auto &be_task = be_queue.front();
            auto &be_executor = be_task->model->executor;
            auto &be_model = be_executor.model;
            auto be_kernel_offset = be_task->kernel_offset;
            executor::KernelInfo &be_kernel = be_model->kernels[be_kernel_offset];
            executor::TransExecutor::KernelArg &be_args = be_executor.trans_args[be_kernel_offset];


            bool occupancy_can_merge = false;
            bool latency_can_merge = false;


            // 1. calculate the occupancy of the merged kernel
            int merged_block_size = std::max(
                rt_args.thread_dim.x * rt_args.thread_dim.y * rt_args.thread_dim.z,
                be_args.thread_dim.x * be_args.thread_dim.y * be_args.thread_dim.z
            );
            executor::GPUConfig::KernelResource merged_resource = 
                executor::GPUConfig::max_resource(rt_args.resource, be_args.resource);

            int merged_occupancy = 
                executor::GPUConfig::calculate_occupancy(merged_resource, merged_block_size);

            occupancy_can_merge = merged_occupancy >= required_occupancy;

            if (!occupancy_can_merge) continue;

            // 2. use latency profile to decide the new occupancy
            be_occupancy = 0;
            for (int j = merged_occupancy - 1; j >= 0; j--) {
                if (be_args.profile.latency.size() <= j) continue;
                if (be_args.profile.latency[j] <= rt_latency) {
                    be_occupancy = j + 1;
                    break;
                }
            }

            latency_can_merge = be_occupancy > 0;
            if (!latency_can_merge) continue;

            // 3. the two kernel can be merged, now calculate the best-effort kernel's arg (e.g. num of blocks, logical occupancy ...)
            cu_partition = align_up(rt_args.block_num, logical_layers) / logical_layers;

            final_occupancy = std::max(required_occupancy, be_occupancy);
            // final_vgprs = merged_vgprs;
            // final_sgprs = merged_sgprs;
            // final_shared_memory = merged_shared_memory;
            final_kernel_resource = merged_resource;
            final_block_size = merged_block_size;
            // final_stack_size = std::max(final_stack_size, be_args.stack_size);
            be_func = be_args.funcion_pointer;

            int be_latency_per_iter = be_args.profile.latency[be_occupancy - 1];
            int be_iters = rt_latency / be_latency_per_iter;
            final_be_block_num = (executor::GPUConfig::get_num_cus() - cu_partition) * be_iters * be_occupancy;
            final_be_block_num = 
                final_be_block_num + be_task->block_offset > be_args.block_num ?
                    be_args.block_num - be_task->block_offset :
                    final_be_block_num;
            final_be_args = &be_args;
            final_be_block_offset = be_task->block_offset;
            final_be_info = &be_kernel;

            final_be_task = be_task.get();
            break;
        }
        auto sel_end = std::chrono::system_clock::now();
        kernel_sel_count++;
        kernel_sel_latency_sum += (sel_end-sel_start).count();
        
        if (final_be_task == nullptr) {
            // if the rt kernel cannot be padded with be kernels,
            // launch the original kernel.
            rt_executor.launch_preempt_kernel(i, rt_queue->stream);
        } else {
            // 4. get proxy kernel
            hipFunction_t proxy_kernel = get_proxy_kernel(
                final_kernel_resource,
                &rt_executor,
                final_be_task == nullptr ? nullptr : &final_be_task->model->executor
            );

            // cu_partition = 60; // for debug
            // 5. launch kernel
            void* kernel_args[] = {
                &rt_func,
                &rt_occupancy,
                &rt_args.block_num,
                &rt_args.block_offset,
                &rt_args.args,

                &be_func,
                &be_occupancy,
                &final_be_block_num,
                &final_be_block_offset,
                &final_be_args->args,

                &cu_partition,
            };

            if (final_be_task != nullptr && final_be_task->start.time_since_epoch().count() == 0)
                final_be_task->start = current_time_point; 

            // auto sel_end = std::chrono::system_clock::now();
            // num_rt_kernels ++;
            // selection_time += (sel_end - sel_start);
            ASSERT_GPU_ERROR(GPUModuleLaunchKernel(
                proxy_kernel,
                final_occupancy * executor::GPUConfig::get_num_cus(), 1, 1,
                final_block_size, 1, 1,
                final_kernel_resource.shared_memory, rt_queue->stream,
                kernel_args, 0
            ));
        }

        current_time_point = current_time_point + std::chrono::microseconds(rt_args.profile.total_latency);

        // 6. update the best-effort kernel's progress
        if (final_be_task != nullptr) {
            final_be_task->padding = true;
            final_be_task->block_offset += final_be_block_num;
            if (final_be_task->block_offset >= final_be_args->block_num) {
                final_be_task->block_offset = 0;
                final_be_task->kernel_offset ++;
                if (final_be_task->kernel_offset >= final_be_task->model->executor.num_kernels()) {
                    // Task is finished.
                    final_be_task->end = current_time_point;
                    final_be_task->kernel_offset = 0;
                    final_be_task->padding_to_finish = true;
                    complete_be_tasks.push_back(
                        be_queues[final_be_task->qid]->task_queue.front()
                    );
                    be_queues[final_be_task->qid]->task_queue.pop();
                }
            }
        }
    }

    // notify completed best-effort tasks
    for (auto task : complete_be_tasks) {
#ifdef ENABLE_TASK_CV
        std::unique_lock<std::mutex> lock(task->mtx);
        task->cv.notify_all();
#endif
        task->state = TaskState::Finish;
    }
}

void REEFScheduler::execute_be_task(std::shared_ptr<Task>& task, std::shared_ptr<TaskQueue>& tqueue) {
    switch (task->state) {
    case TaskState::Finish:
        return;
    case TaskState::Executing: {
        // check if the task is finished
        bool finished = executor::GPUStreamEmpty(tqueue->stream);
        if (finished) {
            // TODO: use GPUEvent timestamp
            // LOG(INFO) << "be task finished";
            task->end = std::chrono::system_clock::now();
            task->state = TaskState::Finish;
            task->kernel_offset = 0;
            task->block_offset = 0;
        }
        return;
    }
    case TaskState::Init:
    case TaskState::Waiting: {
        int num_kernels = task->model->executor.num_kernels();
        if (task->state == TaskState::Init) {
            task->start = std::chrono::system_clock::now();
        }
        if (preempted) {
            reset_preempt_flag_async();
            preempted = false;
            ASSERT_GPU_ERROR(GPUStreamSynchronize(preempt_stream));
            // LOG(INFO) << "reset preempt flag";
        }
        int num_launched = 0;
        auto& exe = task->model->executor;
        if (task->kernel_offset >= num_kernels) {
            task->kernel_offset = 0;
            task->block_offset = 0;
            task->end = std::chrono::system_clock::now();
            // LOG(INFO) << "best-effort task done " << be_task->id;
            task->state = TaskState::Finish;
            return;
        }
        task->state = TaskState::Executing;
        for (int i = task->kernel_offset; i < exe.num_kernels(); i++) {
            assert(exe.launch_preempt_kernel(i, tqueue->stream) == Status::Succ);
            task->launch_offset = i;
            if (!rt_queue->task_queue.empty()) {
                if (mode == ScheduleMode::REEF || mode == ScheduleMode::Reset) {
                    // LOG(INFO) << "preempt during launch";
                    return; // preempt
                }   
            }
        }
        // LOG(INFO) << "launch be task";
    }
    }

    return;
}

std::vector<std::chrono::system_clock::time_point> 
    REEFScheduler::Task::get_timestamp() const {
    return std::vector<std::chrono::system_clock::time_point>({
        submit, start, end
    });
}


bool REEFScheduler::Task::is_preempted() const {
    return preempted;
}

bool REEFScheduler::Task::is_padded() const {
    return padding;
}

bool REEFScheduler::Task::is_padded_to_complete() const {
    return padding_to_finish;
}


} // namespace server
} // namespace reef