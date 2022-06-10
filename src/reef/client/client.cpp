#include "reef/client/client.h"

#include <grpcpp/grpcpp.h>

namespace reef {
namespace client {

REEFClient::REEFClient(const std::string &server_addr) : rpc_client(nullptr) {
    LOG(INFO) << "Create REEFClient to " << server_addr;

    rpc_client = REEFService::NewStub(
                    grpc::CreateChannel(server_addr, grpc::InsecureChannelCredentials())
                 );

    ASSERT_MSG(rpc_client.get() != nullptr, "cannot create rpc client");
    LOG(INFO) << "Create REEFClient succeeds";
}

bool REEFClient::init(bool real_time) {
    // set client (task queue) priority
    grpc::ClientContext ctx;
    reef::rpc::SetPriorityRequest request;
    reef::rpc::SetPriorityReply reply;
    request.set_rt(real_time);
    auto status = rpc_client->SetPriority(&ctx, request, &reply);
    ASSERT_MSG(status.ok(), status.error_message());
    ASSERT(reply.succ());
    qid = reply.qid();
    return true;
}

std::shared_ptr<ModelHandle> REEFClient::load_model(
    const std::string& model_dir,
    const std::string& name
) {
    grpc::ClientContext ctx;
    reef::rpc::LoadModelRequest request;
    reef::rpc::LoadModelReply reply;   
    LOG(INFO) << "Loading model " << name;
    request.set_dir(model_dir);
    request.set_name(name);
    request.set_qid(qid);
    auto status = rpc_client->LoadModel(&ctx, request, &reply);
    ASSERT_MSG(status.ok(), status.error_message());
    ASSERT(reply.succ());    
    std::shared_ptr<ModelHandle> model = 
        std::make_shared<ModelHandle>(
            rpc_client, reply.mid(), model_dir, name
        );
    {
        std::unique_lock<std::mutex> lock(models_mtx);
        models.push_back(model);
    }
    return model;
}

ModelHandle::ModelHandle(
    const std::shared_ptr<REEFService::Stub>& _rpc_client,
    int32_t _mid,
    const std::string& _dir, 
    const std::string& _name
) : rpc_client(_rpc_client), mid(_mid), dir(_dir), name(_name) {

}

// submit an inference task. wait for completion.
TaskHandle ModelHandle::infer() {
    grpc::ClientContext ctx;
    reef::rpc::InferRequest request;
    reef::rpc::InferReply reply; 
    request.set_mid(mid);
    TaskHandle t;
    t.submit = std::chrono::system_clock::now();
    auto status = rpc_client->Infer(&ctx, request, &reply);
    ASSERT_MSG(status.ok(), status.error_message());
    ASSERT(reply.succ());
    t.finish = std::chrono::system_clock::now();
    t.tid = reply.tid();    
    return t;
}

// submit an asynchronous inference task.
TaskHandle ModelHandle::infer_async() {
    return TaskHandle();
}

// get the poniter of input shared memory.
std::shared_ptr<util::SharedMemory> ModelHandle::get_input_blob(const std::string& name) {
    if (input_blob.get() == nullptr) 
        input_blob = register_blob(name, input_blob_key);
    return input_blob;
}

// get the poniter of output shared memory.
std::shared_ptr<util::SharedMemory> ModelHandle::get_output_blob(const std::string& name) {
    if (output_blob.get() == nullptr) {
        output_blob = register_blob(name, output_blob_key);
    }
    return output_blob;
}

std::shared_ptr<util::SharedMemory> ModelHandle::register_blob(const std::string& name, std::string& key) {
    grpc::ClientContext ctx;
    reef::rpc::RegisterBlobRequest request;
    reef::rpc::RegisterBlobReply reply; 

    request.set_mid(mid);
    request.set_name(name);
    auto status = rpc_client->RegisterBlob(&ctx, request, &reply);
    ASSERT_MSG(status.ok(), status.error_message());
    ASSERT(reply.succ());    
    
    std::shared_ptr<util::SharedMemory> shm =
        std::make_shared<util::SharedMemory>(reply.key(), reply.size());
    key = reply.key();
    return shm;
}

// load model input in REEF server. wait for completion.
void ModelHandle::load_input() {
    ASSERT(input_blob.get() != nullptr);
    grpc::ClientContext ctx;
    reef::rpc::SetBlobRequest request;
    reef::rpc::SetBlobReply reply; 
    request.set_key(input_blob_key);
    auto status = rpc_client->SetBlob(&ctx, request, &reply);
    ASSERT_MSG(status.ok(), status.error_message());
    ASSERT(reply.succ());    
}

// load model output in REEF server. wait for completion.
void ModelHandle::get_output() {
    ASSERT(output_blob.get() != nullptr);
    grpc::ClientContext ctx;
    reef::rpc::GetBlobRequest request;
    reef::rpc::GetBlobReply reply; 
    request.set_key(output_blob_key);
    auto status = rpc_client->GetBlob(&ctx, request, &reply);
    ASSERT_MSG(status.ok(), status.error_message());
    ASSERT(reply.succ());    
}

int32_t ModelHandle::get_mid() const {
    return this->mid;
}

} // namespace client
} // namespace reef