#pragma once

#include <string>
#include <mutex>
#include <condition_variable>
#include <queue>

#include "reef/util/threadsafe_queue.h"
#include "reef/util/common.h"
#include "reef/executor/hybrid_executor.h"

namespace reef {
namespace server {

class REEFScheduler {
    class Model;
public:
    typedef uint32_t ModelID;
    typedef uint32_t QueueID;
    typedef uint32_t TaskID;
    
    enum ScheduleMode {
        NoPreempt, // no preemption
        MultiStream, // multiple GPU streams
        WaitPreempt, // wait-based preemption
        REEF,
        Reset // reset-based preemption without DKP
    };

    enum TaskQueueType {
        RealTimeQueue,
        BestEffortQueue,
    };

    enum TaskState {
        Init,
        Waiting,
        Executing,
        Finish    
    };

    struct Task {
        friend REEFScheduler;
    private:
        std::shared_ptr<Model> model;
        QueueID qid;
        TaskID id;
        volatile TaskState state;
        int launch_offset; // the kernel idx that has been launched to host queue
        int kernel_offset; // the kernel idx that has been executed
        int block_offset; // for DKP
        std::mutex mtx;
        std::condition_variable cv;
        std::chrono::system_clock::time_point submit; // when this task is created
        std::chrono::system_clock::time_point start; // when this task is scheduled
        std::chrono::system_clock::time_point end; // when this task is completed
        bool preempted;
        bool padding;
        bool padding_to_finish;
    public:
        bool is_preempted() const;
        bool is_padded() const;
        bool is_padded_to_complete() const;
        std::vector<std::chrono::system_clock::time_point> get_timestamp() const;
    };

public:
    REEFScheduler(ScheduleMode _mode = ScheduleMode::REEF);
    ~REEFScheduler();

    Status load_model(
        const std::string& model_dir,
        const std::string& model_name,
        ModelID& mid
    );

    Status load_model(
        const std::string& rt_co_path,
        const std::string& be_co_path,
        const std::string& json_path,
        const std::string& profile_path,
        const std::string& param_path,
        ModelID& mid
    );

    Status create_queue(
        const TaskQueueType& qtp,
        QueueID& qid
    );

    Status bind_model_queue(
        const QueueID& qid,
        const ModelID& mid
    );

    Status get_data_size(ModelID mid, const std::string& name, size_t& size);

    Status set_input(ModelID mid, const void* data, size_t len, const std::string& name="data");

    Status get_output(ModelID mid, void* data, size_t len, const std::string& name="output");

    Status new_task(
        const ModelID& mid,
        TaskID& tid
    );

    Status wait_task(
        TaskID tid
    );

    Status get_task(
        TaskID tid,
        std::shared_ptr<Task>& t
    );

    ScheduleMode sche_mode() const;

    void set_wait_sync(bool value);

    void set_be_stream_cap(int value);
    Status run();
    Status shutdown();

    int64_t avg_preempt_latency() const;
    
    int64_t avg_kernel_sel_latency() const;
private:
    ScheduleMode mode;
    const size_t model_pool_capacity = 1024;
    std::atomic_uint32_t model_pool_size;
    struct Model {
        executor::HybridExecutor executor;
        QueueID qid;
    };
    std::vector<std::shared_ptr<Model>> model_pool;


    std::atomic_uint32_t task_idx_pool;
    std::unordered_map<TaskID, std::shared_ptr<Task>> task_pool;
    std::mutex task_pool_mtx;

    struct TaskQueue {
        ThreadSafeQueue<std::shared_ptr<Task>> task_queue;
        executor::GPUStream_t stream;
    };

    const size_t max_num_be_queues = 32;
    const QueueID rt_queue_id = 32; // the same with be queue num
    std::mutex be_queues_mtx;
    std::vector<std::shared_ptr<TaskQueue>> be_queues;
    volatile uint32_t be_queue_cnt;
    std::shared_ptr<TaskQueue> rt_queue;
    std::mutex task_cnt_mtx;
    std::condition_variable task_cnt_cv; // To wake up the scheduler
    volatile uint32_t task_cnt;
    bool wait_sync;

    std::unique_ptr<std::thread> scheduler;
    executor::GPUStream_t execute_stream, preempt_stream;
    executor::GPUDevicePtr_t preempt_flag;
    bool preempted;
    int be_stream_device_queue_cap;
    std::atomic_bool _shutdown;

    uint64_t preempt_count;
    uint64_t preempt_latency_sum;

    uint64_t kernel_sel_count;
    uint64_t kernel_sel_latency_sum;
private:
    Status create_task_queue(std::shared_ptr<TaskQueue>& ret, bool rt);
    void loop_body();
    void execute_be_task(std::shared_ptr<Task>& task, std::shared_ptr<TaskQueue>& tqueue);
    void execute_rt_task(std::shared_ptr<Task>& task);
    void preempt_be_tasks();
    void reset_preempt_flag_async();
    void preempt_reset();
    void preempt_wait();
    void dynamic_kernel_padding(std::shared_ptr<Task>& rt_task);
    executor::GPUFunction_t get_proxy_kernel(
        const executor::GPUConfig::KernelResource& resource, 
        executor::HybridExecutor* rt_executor,
        executor::HybridExecutor* be_executor
    );
};


} // namespace server
} // namespace reef