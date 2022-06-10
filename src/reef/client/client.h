#pragma once

#include "reef/rpc/reef.grpc.pb.h"
#include "reef/util/common.h"
#include "reef/util/shared_memory.h"

#include <glog/logging.h>

#include <string>
#include <memory>
#include <vector>

namespace reef {
namespace client {

using reef::rpc::REEFService;

class TaskHandle {
public:
    int32_t tid;
    std::chrono::system_clock::time_point submit, finish;
};

// ModelHandle can be used to submit inference task.
class ModelHandle {
public:
    ModelHandle(
        const std::shared_ptr<REEFService::Stub>& rpc_client,
        int32_t _mid, 
        const std::string& dir,
        const std::string& name
    );
    // submit an inference task. wait for completion.
    TaskHandle infer();

    // submit an asynchronous inference task.
    TaskHandle infer_async();

    // get the poniter of input shared memory.
    std::shared_ptr<util::SharedMemory> get_input_blob(const std::string& name = "data");

    // get the poniter of output shared memory.
    std::shared_ptr<util::SharedMemory> get_output_blob(const std::string& name = "output");

    // load model input in REEF server. wait for completion.
    void load_input();

    // load model output in REEF server. wait for completion.
    void get_output();

    // TODO: FIXME
    void get_blob();
    void set_blob();

    int32_t get_mid() const;
private:
    std::shared_ptr<REEFService::Stub> rpc_client;
    int32_t mid;
    std::string dir;
    std::string name;
    std::string input_blob_key, output_blob_key;
    std::shared_ptr<util::SharedMemory> input_blob;
    std::shared_ptr<util::SharedMemory> output_blob;

private:
    std::shared_ptr<util::SharedMemory> register_blob(const std::string& name, std::string& key);
    
};


// REEFClient is used to estabilish connection to REEF server
// and load models into the server.
class REEFClient {
public:
    REEFClient(const std::string &server_addr);
    // initialize the client
    // Each client should be configured with a priority.
    // The real-time clients will share a RT task queue.
    // Each best-effort client will have its own BE task queue.
    bool init(bool real_time = false);

    // load a DNN model (in REEF server).
    std::shared_ptr<ModelHandle> load_model(
        const std::string& model_dir,
        const std::string& name
    );

private:
    std::shared_ptr<REEFService::Stub> rpc_client;
    std::mutex models_mtx;
    std::vector<std::shared_ptr<ModelHandle>> models;
    int32_t qid;
};


} // namespace client
} // namespace reef