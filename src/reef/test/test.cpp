#include <gtest/gtest.h>
#include <thread>

#include "reef/server/server.h"
#include "reef/client/client.h"
#include "reef/server/scheduler.h"
#include "reef/executor/executor_base.h"
#include "reef/executor/trans_executor.h"
#include "reef/executor/hybrid_executor.h"


using namespace reef;


#define RESOURCE(model, type) RESOURCE_DIR "/" #model "/" #model "." type
#define PARAM_RESOURCE(model) RESOURCE(model, "param")
#define PROFILE_RESOURCE(model) RESOURCE(model, "profile.json")
#define JSON_RESOURCE(model) RESOURCE(model, "json")
#define RAW_MODEL(model) JSON_RESOURCE(model), RESOURCE(model, "raw.co")
#define TRANS_MODEL(model) JSON_RESOURCE(model), RESOURCE(model, "trans.co")
#define HYBRID_MODEL(model) JSON_RESOURCE(model), PROFILE_RESOURCE(model), RESOURCE(model, "trans.co"), RESOURCE(model, "be.co")
#define SCHEDULER_MODEL(model) RESOURCE(model, "trans.co"), RESOURCE(model, "be.co"), JSON_RESOURCE(model), PROFILE_RESOURCE(model)

#define ASSERT_SUCC(expr) ASSERT_TRUE(Status::Succ == expr)

TEST(rpc, connection) {
    server::REEFServer server(DEFAULT_REEF_ADDR);
    server.run();
    client::REEFClient client(DEFAULT_REEF_ADDR);
    client.init(true);
    server.shutdown();
}

TEST(rpc, resnet18_param) {
    server::REEFServer server(DEFAULT_REEF_ADDR);
    server.run();
    client::REEFClient client(DEFAULT_REEF_ADDR);
    client.init(true);
    auto model = client.load_model(RESOURCE_DIR "/resnet18", "resnet18");
    auto input_blob = model->get_input_blob();
    auto output_blob = model->get_output_blob();

    std::vector<float> input(3 * 224 * 224);
    for (size_t i = 0; i < 3*224*224; i++)
        input[i] = 10.0;
    memcpy(input_blob->data(), input.data(), input_blob->size());
    model->load_input();
    model->infer();
    model->get_output();
    std::vector<float> output(1000);
    memcpy(output.data(), output_blob->data(), output_blob->size());
    std::vector<float> ans = {0.0003392257, 0.0014304413, 0.0004299286, 0.0010349639, 0.0020997059,
                        0.0016049921, 0.0010267848, 0.00042607592, 0.0018747754, 0.0024558322};
    for (size_t i = 0; i < ans.size(); i++)
        ASSERT_FLOAT_EQ(ans[i], output[i]);
    server.shutdown();  
}

TEST(parser, parse_model)
{
    std::unique_ptr<executor::Model> m(executor::Model::from_json(JSON_RESOURCE(resnet18)));
    ASSERT_TRUE(m != nullptr);
    ASSERT_TRUE(m->kernels.size() != 0);
    
}


TEST(parser, parse_params)
{
    std::unique_ptr<executor::ModelParam> params(
        executor::ModelParamParser::parse_from_file(
            PARAM_RESOURCE(resnet18)
        )
    );
    ASSERT_TRUE(params != nullptr);
    ASSERT_EQ(params->at("p22").size(), 294912);
}

TEST(parser, parse_profile)
{
    std::unique_ptr<executor::ModelProfile> profile(
        executor::ModelProfile::from_json(PROFILE_RESOURCE(resnet18))
    );
    ASSERT_EQ(profile->kernel_latency.size(), 54);
    for (auto &pair : profile->kernel_latency) {
        ASSERT_GT(pair.second.latency.size(), 0);
        for (auto latency : pair.second.latency) {
            ASSERT_GT(latency, 0.0);
        }
    }
}

TEST(util, shm) 
{
    util::SharedMemory shm1("shm_test", 1024, true);
    util::SharedMemory shm2("shm_test", 1024, false);

    *(int*)shm1.data() = 666;
    ASSERT_EQ(*(int*)shm2.data(), 666);
}

TEST(executor_base, mocked_kernel) {
    executor::ExecutorBase exe;
    ASSERT_SUCC(exe.load_model_from_file(RAW_MODEL(mocked_kernel)));
    std::vector<float> input(8192);
    for (size_t i = 0; i < 8192; i++) input[i] = 3;
    ASSERT_SUCC(exe.set_input("a", input));
    ASSERT_SUCC(exe.set_input("b", input));
    ASSERT_SUCC(exe.set_input("c", input));
    ASSERT_SUCC(exe.execute());

    std::vector<float> output;
    ASSERT_SUCC(exe.get_output(output));
    for (size_t i = 0; i < 8192; i++) 
        ASSERT_FLOAT_EQ(output[i], 18.0);
}

TEST(executor_base, resnet18) 
{
    executor::ExecutorBase exe;
    ASSERT_SUCC(exe.load_model_from_file(RAW_MODEL(resnet18)));
    ASSERT_SUCC(exe.execute());
}

// TEST(executor_base, resnet34) 
// {
//     executor::ExecutorBase exe;
//     ASSERT_SUCC(exe.load_model_from_file(RAW_MODEL(resnet34)));
//     ASSERT_SUCC(exe.execute());
// }

void assert_equal_vec(int idx, size_t size, executor::ExecutorBase& e1, executor::ExecutorBase& e2) {
    std::vector<float> o1, o2;
    o1.resize(size);
    o2.resize(size);
    e1.get_data(idx, o1.data(), size*4);
    e2.get_data(idx, o2.data(), size*4);
    for (size_t i = 0; i < size; i++) {
        if (o1[i] != o2[i]) {
            // std::cerr << "i: " << i << std::endl;
            std::cout << "***i: " << i << ", data: " << o1[i] << ", " << o2[i] << std::endl;
        } else {
            // std::cout << "i: " << i << ", data: " << o1[i] << std::endl;
        }
        ASSERT_FLOAT_EQ(o1[i], o2[i]);
    }
}

void assert_storage(executor::ExecutorBase& e1, executor::ExecutorBase& e2) {
    for (int i = 0; i < e2.model->storage.size() - 2; i++) {
        std::vector<float> o1, o2;
        auto s = e1.model->storage[i];
        o1.resize(s.size);
        o2.resize(s.size);
        e1.get_data(i, o1.data(), s.size * 4);
        e2.get_data(i, o2.data(), s.size * 4);
        for (int j = 0; j < s.size; j++) {
            if (o1[j] != o2[j]) {
                // EXPECT_FLOAT_EQ(o1[j], o2[j]);
                // printf("storage: %d, o1[%d]=%16f, o2[%d]=%16f\n", i, j, o1[j], j, o2[j]);
                // assert(0);
            }
            
            ASSERT_FLOAT_EQ(o1[j], o2[j]);
        }
    }
}

// TEST(test, test)
// {
//     executor::ExecutorBase exe;
//     ASSERT_SUCC(exe.load_model_from_file(RAW_MODEL(resnet18)));
//     ASSERT_SUCC(exe.load_param_from_file(PARAM_RESOURCE(resnet18)));

//     executor::ExecutorBase exe2;
//     ASSERT_SUCC(exe2.load_model_from_file(RAW_MODEL(resnet18)));
//     ASSERT_SUCC(exe2.load_param_from_file(PARAM_RESOURCE(resnet18)));
    
//     std::vector<float> input(802816), data(3*224*224);

//     for (size_t i = 0; i < 802816; i++)
//         input[i] = 2.0;
//     for (size_t i = 0; i < 3*224*224; i++)
//         data[i] = 10.0;
//     ASSERT_SUCC(exe.set_input("data", data));
//     ASSERT_SUCC(exe2.set_input("data", data));
//     ASSERT_SUCC(exe.set_input(2, input.data(), 200704*4));
//     // ASSERT_SUCC(exe.set_input(3, input.data(), 9408*4));
//     // ASSERT_SUCC(exe.set_input(4, input.data(), 64*4));
//     ASSERT_SUCC(exe2.set_input(2, input.data(), 200704*4));
//     // ASSERT_SUCC(exe2.set_input(3, input.data(), 9408*4));
//     // ASSERT_SUCC(exe2.set_input(4, input.data(), 64*4));
//     // 2 3 4 -> 5
//     // 200704
//     // 9408
//     // 64
//     // 802816
//     // exe.execute_kernel(2);
//     // exe2.execute_kernel(2);
//     // std::vector<float> data3(9408);
//     // exe.get_data(3, data3.data(), data3.size()*4);
//     // for (int i = 42; i <=48; i++)
//     //     printf("data3[%d]: %f\n", i, data3[i]);
//     // for (int i = 91; i <=97; i++)
//     //     printf("data3[%d]: %f\n", i, data3[i]);
//     // for (int i = 140; i <=146; i++)
//     //     printf("data3[%d]: %f\n", i, data3[i]);
//     for (int i = 0; i < 57; i++) {
//         exe.execute_kernel(i);
//         exe2.execute_kernel(i);
//         std::cout << "kernel " << i << std::endl;
//         assert_storage(exe, exe2);
//     }
//     // exe.execute_to(9);
//     // exe2.execute_to(9);
    

//     // // assert_equal_vec(3, 9408, exe, exe2);
//     // assert_equal_vec(2, 200704, exe, exe2);
// }

TEST(executor_base, resnet18_param) 
{
    executor::ExecutorBase exe;
    ASSERT_SUCC(exe.load_model_from_file(RAW_MODEL(resnet18)));
    ASSERT_SUCC(exe.load_param_from_file(PARAM_RESOURCE(resnet18)));
    std::vector<float> input(3 * 224 * 224);
    for (size_t i = 0; i < 3*224*224; i++)
        input[i] = 10.0;
    ASSERT_SUCC(exe.set_input("data", input));
    ASSERT_SUCC(exe.execute());

    // executor::ExecutorBase exe2;
    // ASSERT_SUCC(exe2.load_model_from_file(RAW_MODEL(resnet18)));
    // ASSERT_SUCC(exe2.load_param_from_file(PARAM_RESOURCE(resnet18)));
    
    
    std::vector<float> output;
    ASSERT_SUCC(exe.get_output(output));
    std::vector<float> ans = {0.0003392257, 0.0014304413, 0.0004299286, 0.0010349639, 0.0020997059,
                           0.0016049921, 0.0010267848, 0.00042607592, 0.0018747754, 0.0024558322};
    for (size_t i = 0; i < ans.size(); i++)
        ASSERT_FLOAT_EQ(ans[i], output[i]);
}

TEST(rt_executor, mocked_kernel)
{
    executor::TransExecutor exe;
    ASSERT_SUCC(exe.load_model_from_file(TRANS_MODEL(mocked_kernel)));
    std::vector<float> input(8192);
    for (size_t i = 0; i < 8192; i++) input[i] = 3;
    ASSERT_SUCC(exe.set_input("a", input));
    ASSERT_SUCC(exe.set_input("b", input));
    ASSERT_SUCC(exe.set_input("c", input));
    ASSERT_SUCC(exe.execute());

    std::vector<float> output;
    ASSERT_SUCC(exe.get_output(output));
    for (size_t i = 0; i < 8192; i++) 
        ASSERT_FLOAT_EQ(output[i], 18.0);
}

TEST(rt_executor, resnet18) 
{
    executor::TransExecutor exe;
    ASSERT_SUCC(exe.load_model_from_file(TRANS_MODEL(resnet18)));
    ASSERT_SUCC(exe.execute());
}

// TEST(rt_executor, resnet34) 
// {
//     executor::TransExecutor exe;
//     ASSERT_SUCC(exe.load_model_from_file(TRANS_MODEL(resnet34)));
//     ASSERT_SUCC(exe.execute());
// }

TEST(rt_executor, resnet18_param) 
{
    executor::TransExecutor exe;
    ASSERT_SUCC(exe.load_model_from_file(TRANS_MODEL(resnet18)));
    ASSERT_SUCC(exe.load_param_from_file(PARAM_RESOURCE(resnet18)));
    std::vector<float> input(3 * 224 * 224);
    for (size_t i = 0; i < 3*224*224; i++)
        input[i] = 10.0;
    ASSERT_SUCC(exe.set_input("data", input));
    ASSERT_SUCC(exe.execute());

    std::vector<float> output;
    ASSERT_SUCC(exe.get_output(output));
    std::vector<float> ans = {0.0003392257, 0.0014304413, 0.0004299286, 0.0010349639, 0.0020997059,
                           0.0016049921, 0.0010267848, 0.00042607592, 0.0018747754, 0.0024558322};
    for (size_t i = 0; i < ans.size(); i++)
        ASSERT_FLOAT_EQ(ans[i], output[i]);
}

TEST(hybrid_executor, mocked_kernel)
{
    executor::HybridExecutor exe;
    ASSERT_SUCC(exe.load_hybrid_model_from_file(HYBRID_MODEL(mocked_kernel)));
    std::vector<float> input(8192);
    for (size_t i = 0; i < 8192; i++) input[i] = 3;
    ASSERT_SUCC(exe.set_input("a", input));
    ASSERT_SUCC(exe.set_input("b", input));
    ASSERT_SUCC(exe.set_input("c", input));
    ASSERT_SUCC(exe.execute_preemptale());

    std::vector<float> output;
    ASSERT_SUCC(exe.get_output(output));
    for (size_t i = 0; i < 8192; i++) 
        ASSERT_FLOAT_EQ(output[i], 18.0);
}

TEST(hybrid_executor, resnet18) 
{
    executor::HybridExecutor exe;
    ASSERT_SUCC(exe.load_hybrid_model_from_file(HYBRID_MODEL(resnet18)));
    ASSERT_SUCC(exe.execute_preemptale());
}

// TEST(hybrid_executor, resnet34) 
// {
//     executor::HybridExecutor exe;
//     ASSERT_SUCC(exe.load_hybrid_model_from_file(HYBRID_MODEL(resnet34)));
//     ASSERT_SUCC(exe.execute_preemptale());
// }

TEST(hybrid_executor, resnet18_param) 
{
    executor::HybridExecutor exe;
    ASSERT_SUCC(exe.load_hybrid_model_from_file(HYBRID_MODEL(resnet18)));
    ASSERT_SUCC(exe.load_param_from_file(PARAM_RESOURCE(resnet18)));
    std::vector<float> input(3 * 224 * 224);
    for (size_t i = 0; i < 3*224*224; i++)
        input[i] = 10.0;
    ASSERT_SUCC(exe.set_input("data", input));
    ASSERT_SUCC(exe.execute_preemptale());

    std::vector<float> output;
    ASSERT_SUCC(exe.get_output(output));
    std::vector<float> ans = {0.0003392257, 0.0014304413, 0.0004299286, 0.0010349639, 0.0020997059,
                           0.0016049921, 0.0010267848, 0.00042607592, 0.0018747754, 0.0024558322};
    for (size_t i = 0; i < ans.size(); i++)
        ASSERT_FLOAT_EQ(ans[i], output[i]);
}

TEST(scheduler, mocked_kernel_rt)
{
    server::REEFScheduler sche;
    server::REEFScheduler::ModelID mid;
    server::REEFScheduler::QueueID qid;
    server::REEFScheduler::TaskID tid;
    std::vector<float> input(8192);
    for (size_t i = 0; i < 8192; i++) input[i] = 3;
    ASSERT_SUCC(sche.run());
    ASSERT_SUCC(sche.load_model(SCHEDULER_MODEL(mocked_kernel), "", mid));
    ASSERT_SUCC(sche.create_queue(server::REEFScheduler::RealTimeQueue, qid));
    ASSERT_SUCC(sche.bind_model_queue(qid, mid));
    ASSERT_SUCC(sche.set_input(mid, (void*)input.data(), input.size() * sizeof(float), "a"));
    ASSERT_SUCC(sche.set_input(mid, (void*)input.data(), input.size() * sizeof(float), "b"));
    ASSERT_SUCC(sche.set_input(mid, (void*)input.data(), input.size() * sizeof(float), "c"));
    ASSERT_SUCC(sche.new_task(mid, tid));
    ASSERT_SUCC(sche.wait_task(tid));
    std::vector<float> output;
    output.resize(8192);
    ASSERT_SUCC(sche.get_output(mid, (void*)output.data(), output.size() * sizeof(float)));
    for (size_t i = 0; i < 8192; i++) 
        ASSERT_FLOAT_EQ(output[i], 18.0);
    ASSERT_SUCC(sche.shutdown());
}

TEST(scheduler, resnet18_param_rt)
{
    server::REEFScheduler sche;
    server::REEFScheduler::ModelID mid;
    server::REEFScheduler::QueueID qid;
    server::REEFScheduler::TaskID tid;
    std::vector<float> input(3 * 224 * 224);
    for (size_t i = 0; i < 3*224*224; i++)
        input[i] = 10.0;
    ASSERT_SUCC(sche.run());
    ASSERT_SUCC(sche.load_model(SCHEDULER_MODEL(resnet18), PARAM_RESOURCE(resnet18), mid));
    ASSERT_SUCC(sche.create_queue(server::REEFScheduler::RealTimeQueue, qid));
    ASSERT_SUCC(sche.bind_model_queue(qid, mid));
    ASSERT_SUCC(sche.set_input(mid, (void*)input.data(), input.size() * sizeof(float), "data"));
    ASSERT_SUCC(sche.new_task(mid, tid));
    ASSERT_SUCC(sche.wait_task(tid));
    std::vector<float> output;
    output.resize(1000);
    ASSERT_SUCC(sche.get_output(mid, (void*)output.data(), output.size() * sizeof(float)));
    std::vector<float> ans = {0.0003392257, 0.0014304413, 0.0004299286, 0.0010349639, 0.0020997059,
                           0.0016049921, 0.0010267848, 0.00042607592, 0.0018747754, 0.0024558322};
    for (size_t i = 0; i < ans.size(); i++)
        ASSERT_FLOAT_EQ(ans[i], output[i]);
    ASSERT_SUCC(sche.shutdown());
}

TEST(scheduler, mocked_kernel_be)
{
    server::REEFScheduler sche;
    server::REEFScheduler::ModelID mid;
    server::REEFScheduler::QueueID qid;
    server::REEFScheduler::TaskID tid;
    std::vector<float> input(8192);
    for (size_t i = 0; i < 8192; i++) input[i] = 3;
    ASSERT_SUCC(sche.run());
    ASSERT_SUCC(sche.load_model(SCHEDULER_MODEL(mocked_kernel), "", mid));
    ASSERT_SUCC(sche.create_queue(server::REEFScheduler::BestEffortQueue, qid));
    ASSERT_SUCC(sche.bind_model_queue(qid, mid));
    ASSERT_SUCC(sche.set_input(mid, (void*)input.data(), input.size() * sizeof(float), "a"));
    ASSERT_SUCC(sche.set_input(mid, (void*)input.data(), input.size() * sizeof(float), "b"));
    ASSERT_SUCC(sche.set_input(mid, (void*)input.data(), input.size() * sizeof(float), "c"));
    ASSERT_SUCC(sche.new_task(mid, tid));
    ASSERT_SUCC(sche.wait_task(tid));
    std::vector<float> output;
    output.resize(8192);
    ASSERT_SUCC(sche.get_output(mid, (void*)output.data(), output.size() * sizeof(float)));
    for (size_t i = 0; i < 8192; i++) 
        ASSERT_FLOAT_EQ(output[i], 18.0);
    ASSERT_SUCC(sche.shutdown());
}

TEST(scheduler, resnet18_param_be)
{
    server::REEFScheduler sche;
    server::REEFScheduler::ModelID mid;
    server::REEFScheduler::QueueID qid;
    server::REEFScheduler::TaskID tid;
    std::vector<float> input(3 * 224 * 224);
    for (size_t i = 0; i < 3*224*224; i++)
        input[i] = 10.0;
    ASSERT_SUCC(sche.run());
    ASSERT_SUCC(sche.load_model(SCHEDULER_MODEL(resnet18), PARAM_RESOURCE(resnet18), mid));
    ASSERT_SUCC(sche.create_queue(server::REEFScheduler::BestEffortQueue, qid));
    ASSERT_SUCC(sche.bind_model_queue(qid, mid));
    ASSERT_SUCC(sche.set_input(mid, (void*)input.data(), input.size() * sizeof(float), "data"));
    ASSERT_SUCC(sche.new_task(mid, tid));
    ASSERT_SUCC(sche.wait_task(tid));
    std::vector<float> output;
    output.resize(1000);
    ASSERT_SUCC(sche.get_output(mid, (void*)output.data(), output.size() * sizeof(float)));
    std::vector<float> ans = {0.0003392257, 0.0014304413, 0.0004299286, 0.0010349639, 0.0020997059,
                           0.0016049921, 0.0010267848, 0.00042607592, 0.0018747754, 0.0024558322};
    for (size_t i = 0; i < ans.size(); i++)
        ASSERT_FLOAT_EQ(ans[i], output[i]);
    ASSERT_SUCC(sche.shutdown());
}

TEST(scheduler, resnet18_param_preempt)
{
    server::REEFScheduler sche;
    server::REEFScheduler::ModelID rt_mid, be_mid;
    server::REEFScheduler::QueueID rt_qid, be_qid;
    server::REEFScheduler::TaskID rt_tid, be_tid;
    std::vector<float> input(3 * 224 * 224);
    for (size_t i = 0; i < 3*224*224; i++)
        input[i] = 10.0;
    ASSERT_SUCC(sche.run());
    ASSERT_SUCC(sche.load_model(SCHEDULER_MODEL(resnet18), PARAM_RESOURCE(resnet18), rt_mid));
    ASSERT_SUCC(sche.load_model(SCHEDULER_MODEL(resnet18), PARAM_RESOURCE(resnet18), be_mid));
    ASSERT_SUCC(sche.create_queue(server::REEFScheduler::RealTimeQueue, rt_qid));
    ASSERT_SUCC(sche.create_queue(server::REEFScheduler::BestEffortQueue, be_qid));
    ASSERT_SUCC(sche.bind_model_queue(rt_qid, rt_mid));
    ASSERT_SUCC(sche.bind_model_queue(be_qid, be_mid));
    ASSERT_SUCC(sche.set_input(rt_mid, (void*)input.data(), input.size() * sizeof(float), "data"));
    ASSERT_SUCC(sche.set_input(be_mid, (void*)input.data(), input.size() * sizeof(float), "data"));
    ASSERT_SUCC(sche.new_task(be_mid, be_tid)); 
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    ASSERT_SUCC(sche.new_task(rt_mid, rt_tid)); // RT is submitted latter than BE
    std::shared_ptr<server::REEFScheduler::Task> rt_task, be_task;
    ASSERT_SUCC(sche.get_task(rt_tid, rt_task));
    ASSERT_SUCC(sche.get_task(be_tid, be_task));
    ASSERT_SUCC(sche.wait_task(be_mid));
    ASSERT_SUCC(sche.wait_task(rt_mid));
    std::vector<float> output;
    output.resize(1000);
    ASSERT_SUCC(sche.get_output(be_mid, (void*)output.data(), output.size() * sizeof(float)));
    std::vector<float> ans = {0.0003392257, 0.0014304413, 0.0004299286, 0.0010349639, 0.0020997059,
                           0.0016049921, 0.0010267848, 0.00042607592, 0.0018747754, 0.0024558322};
    for (size_t i = 0; i < ans.size(); i++)
        ASSERT_FLOAT_EQ(ans[i], output[i]);
    auto rt_ts = rt_task->get_timestamp();
    auto be_ts = be_task->get_timestamp();
    ASSERT_GT(rt_ts[0], be_ts[0]);
    ASSERT_GT(rt_ts[1], be_ts[1]);
    ASSERT_GT(be_ts[2], rt_ts[2]);
    ASSERT(be_task->is_padded());
    ASSERT(be_task->is_preempted());
    ASSERT_SUCC(sche.shutdown());
}

TEST(scheduler, mocked_kernel)
{
    server::REEFScheduler sche;
    server::REEFScheduler::ModelID rt_mid, be_mid;
    server::REEFScheduler::QueueID rt_qid, be_qid;
    server::REEFScheduler::TaskID rt_tid, be_tid;
    std::vector<float> input(3 * 224 * 224);
    for (size_t i = 0; i < 3*224*224; i++)
        input[i] = 10.0;
    std::vector<float> be_input(8192);
    for (size_t i = 0; i < 8192; i++) be_input[i] = 3;

    ASSERT_SUCC(sche.load_model(SCHEDULER_MODEL(resnet18), PARAM_RESOURCE(resnet18), rt_mid));
    ASSERT_SUCC(sche.load_model(SCHEDULER_MODEL(mocked_kernel), "", be_mid));
    ASSERT_SUCC(sche.create_queue(server::REEFScheduler::RealTimeQueue, rt_qid));
    ASSERT_SUCC(sche.create_queue(server::REEFScheduler::BestEffortQueue, be_qid));
    ASSERT_SUCC(sche.bind_model_queue(rt_qid, rt_mid));
    ASSERT_SUCC(sche.bind_model_queue(be_qid, be_mid));
    ASSERT_SUCC(sche.set_input(rt_mid, (void*)input.data(), input.size() * sizeof(float), "data"));
    ASSERT_SUCC(sche.set_input(be_mid, (void*)be_input.data(), be_input.size() * sizeof(float), "a"));
    ASSERT_SUCC(sche.set_input(be_mid, (void*)be_input.data(), be_input.size() * sizeof(float), "b"));
    ASSERT_SUCC(sche.set_input(be_mid, (void*)be_input.data(), be_input.size() * sizeof(float), "c"));
    ASSERT_SUCC(sche.new_task(be_mid, be_tid)); 
    ASSERT_SUCC(sche.new_task(rt_mid, rt_tid)); 
    ASSERT_SUCC(sche.run());
    std::shared_ptr<server::REEFScheduler::Task> rt_task, be_task;
    ASSERT_SUCC(sche.get_task(rt_tid, rt_task));
    ASSERT_SUCC(sche.get_task(be_tid, be_task));
    ASSERT_SUCC(sche.wait_task(be_mid));
    ASSERT_SUCC(sche.wait_task(rt_mid));
    std::vector<float> output;
    output.resize(1000);
    ASSERT_SUCC(sche.get_output(rt_mid, (void*)output.data(), output.size() * sizeof(float)));
    std::vector<float> ans = {0.0003392257, 0.0014304413, 0.0004299286, 0.0010349639, 0.0020997059,
                           0.0016049921, 0.0010267848, 0.00042607592, 0.0018747754, 0.0024558322};
    for (size_t i = 0; i < ans.size(); i++)
        ASSERT_FLOAT_EQ(ans[i], output[i]);
    std::vector<float> be_output;
    be_output.resize(8192);
    ASSERT_SUCC(sche.get_output(be_mid, (void*)be_output.data(), be_output.size() * sizeof(float)));
    for (size_t i = 0; i < 8192; i++) 
        ASSERT_FLOAT_EQ(be_output[i], 18.0);
    ASSERT(be_task->is_padded());
    ASSERT(be_task->is_padded_to_complete());
    ASSERT_SUCC(sche.shutdown());
}