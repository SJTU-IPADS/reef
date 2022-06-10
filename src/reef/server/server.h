#pragma once

#include "reef/util/common.h"
#include "reef/util/shared_memory.h"
#include "reef/rpc/reef.grpc.pb.h"
#include "reef/server/scheduler.h"
#include <grpcpp/grpcpp.h>

namespace reef {
namespace server {


class REEFServer final : public reef::rpc::REEFService::Service {
public:
    REEFServer(const std::string& addr);
    virtual ~REEFServer() {}
    void run();

    void wait();

    void shutdown();
    
    REEFScheduler* get_scheduler() const {
        return scheduler.get();
    }
    
private: 
    // RPC handles
    grpc::Status SetPriority(
        grpc::ServerContext *context,
        const reef::rpc::SetPriorityRequest *request,
        reef::rpc::SetPriorityReply *reply
    ) override;

    grpc::Status LoadModel(
        grpc::ServerContext *context,
        const reef::rpc::LoadModelRequest *request,
        reef::rpc::LoadModelReply *reply
    ) override;

    grpc::Status RegisterBlob(
        grpc::ServerContext *context,
        const reef::rpc::RegisterBlobRequest *request,
        reef::rpc::RegisterBlobReply *reply
    ) override;
    
    grpc::Status GetBlob(
        grpc::ServerContext *context,
        const reef::rpc::GetBlobRequest *request,
        reef::rpc::GetBlobReply *reply
    ) override;

    grpc::Status SetBlob(
        grpc::ServerContext *context,
        const reef::rpc::SetBlobRequest *request,
        reef::rpc::SetBlobReply *reply
    ) override;

    grpc::Status Infer(
        grpc::ServerContext *context,
        const reef::rpc::InferRequest *request,
        reef::rpc::InferReply *reply
    ) override;

private:
    std::string server_addr;
    std::unique_ptr<grpc::Server> rpc_server;
    std::unique_ptr<REEFScheduler> scheduler;
    std::mutex shm_mtx;
    struct SharedMemoryInfo {
        std::string name;
        std::shared_ptr<util::SharedMemory> shm;
        REEFScheduler::ModelID mid;
    };
    std::unordered_map<std::string, SharedMemoryInfo> shms;
};

} // namespace server 
} // namespace reef