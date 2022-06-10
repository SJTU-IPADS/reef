#include "reef/server/server.h"
#include <grpcpp/grpcpp.h>
#include <glog/logging.h>

namespace reef {
namespace server {

REEFServer::REEFServer(const std::string& addr) 
    : server_addr(addr), rpc_server(nullptr)
{
    scheduler.reset(new REEFScheduler());
}

void REEFServer::run() {
    grpc::ServerBuilder builder;
    builder.AddListeningPort(server_addr, grpc::InsecureServerCredentials());
    builder.RegisterService(this);

    rpc_server = builder.BuildAndStart();
    scheduler->run();
}

void REEFServer::wait() {
    ASSERT(rpc_server.get() != nullptr);
    rpc_server->Wait();
}

void REEFServer::shutdown() {
    ASSERT(rpc_server.get() != nullptr); 
    rpc_server->Shutdown();
    scheduler->shutdown();
}

grpc::Status REEFServer::SetPriority(
    grpc::ServerContext *context,
    const reef::rpc::SetPriorityRequest *request,
    reef::rpc::SetPriorityReply *reply
) {
    LOG(INFO) << "new client, real_time: " << request->rt();
    // create queue
    REEFScheduler::QueueID qid;
    Status s = scheduler->create_queue(
        request->rt() ? 
            REEFScheduler::TaskQueueType::RealTimeQueue
            : REEFScheduler::TaskQueueType::BestEffortQueue, 
        qid
    );
    if (s != Status::Succ)
        reply->set_succ(false);
    else {
        reply->set_succ(true);
        reply->set_qid(qid);
    }
    return grpc::Status::OK;
}

grpc::Status REEFServer::LoadModel(
    grpc::ServerContext *context,
    const reef::rpc::LoadModelRequest *request,
    reef::rpc::LoadModelReply *reply
) {
    LOG(INFO) << "load model: " << request->name() << ", qid: " << request->qid();
    std::string prefix = request->dir() + "/" + request->name();
    std::string param_file = prefix + ".param";
    if (access(param_file.c_str(), F_OK) == -1) {
        param_file = "";
        LOG(INFO) << request->name() << " no param file";
    }

    REEFScheduler::ModelID mid;
    Status s = scheduler->load_model(
        prefix + ".trans.co",
        prefix + ".be.co",
        prefix + ".json",
        prefix + ".profile.json",
        param_file, // TODO: load param
        mid
    );
    if (s != Status::Succ) {
        reply->set_succ(false);
        return grpc::Status::OK;
    } else {
        reply->set_mid(mid);
    }
    s = scheduler->bind_model_queue(request->qid(), mid);
    if (s != Status::Succ) {
        reply->set_succ(false); // TODO: unload model
        return grpc::Status::OK;
    } else {
        reply->set_succ(true);
    }
    return grpc::Status::OK;
}

grpc::Status REEFServer::RegisterBlob(
    grpc::ServerContext *context,
    const reef::rpc::RegisterBlobRequest *request,
    reef::rpc::RegisterBlobReply *reply
) {
    reply->set_succ(false);
    size_t size;
    auto s = scheduler->get_data_size(request->mid(), request->name(), size);
    if (s != Status::Succ) return grpc::Status::OK;
    std::string key = std::to_string(request->mid()) + "_" + request->name();
    reply->set_key(key);
    reply->set_size(size);
    reply->set_succ(true);
    {
        std::unique_lock<std::mutex> lock(shm_mtx);
        auto iter = shms.find(key);
        if (iter == shms.end()) {
            auto shm = std::make_shared<util::SharedMemory>(key, size, true);
            SharedMemoryInfo shminfo;
            shminfo.name = request->name();
            shminfo.mid = request->mid();
            shminfo.shm = shm;
            shms.insert({key, shminfo});
        }
    }
    return grpc::Status::OK;
}

grpc::Status REEFServer::GetBlob(
    grpc::ServerContext *context,
    const reef::rpc::GetBlobRequest *request,
    reef::rpc::GetBlobReply *reply
) {
    SharedMemoryInfo shminfo;
    {
        std::unique_lock<std::mutex> lock(shm_mtx);
        auto iter = shms.find(request->key());
        if (iter == shms.end()) {
            reply->set_succ(false);
            return grpc::Status::OK;
        }
        shminfo = iter->second;
    }
    auto s = scheduler->get_output(shminfo.mid, shminfo.shm->data(), shminfo.shm->size(), shminfo.name);
    if (s != Status::Succ) {
        reply->set_succ(false);
    } else {
        reply->set_succ(true);
    }
    return grpc::Status::OK;
}

grpc::Status REEFServer::SetBlob(
    grpc::ServerContext *context,
    const reef::rpc::SetBlobRequest *request,
    reef::rpc::SetBlobReply *reply
) {
    SharedMemoryInfo shminfo;
    {
        std::unique_lock<std::mutex> lock(shm_mtx);
        auto iter = shms.find(request->key());
        if (iter == shms.end()) {
            reply->set_succ(false);
            return grpc::Status::OK;
        }
        shminfo = iter->second;
    }
    auto s = scheduler->set_input(shminfo.mid, shminfo.shm->data(), shminfo.shm->size(), shminfo.name);
    if (s != Status::Succ) {
        reply->set_succ(false);
    } else {
        reply->set_succ(true);
    }
    return grpc::Status::OK;
}

grpc::Status REEFServer::Infer(
    grpc::ServerContext *context,
    const reef::rpc::InferRequest *request,
    reef::rpc::InferReply *reply
) {
    REEFScheduler::TaskID tid;
    auto s = scheduler->new_task(request->mid(), tid);
    if (s != Status::Succ) {
        reply->set_succ(false);
    } else {
        s = scheduler->wait_task(tid);
        reply->set_succ(true);
        if (s != Status::Succ)
            reply->set_succ(false);
        reply->set_tid(tid);
    }
    return grpc::Status::OK;
}


} // namespace server
} // namespace reef