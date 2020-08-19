#include <iostream>
#include <memory>
#include <string>

#include <grpc++/server.h>  
#include <grpc++/server_builder.h>  
#include <grpc++/server_context.h> 


#include "tensor.grpc.pb.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

using tensor::HelloRequest;
using tensor::HelloReply;
using tensor::GetSizeRequest;
using tensor::GetSizeReply;
using tensor::GetRequest;
using tensor::GetReply;
using tensor::SetRequest;
using tensor::SetReply;
using tensor::DelRequest;
using tensor::DelReply;
using tensor::PushRequest;
using tensor::PushReply;
using tensor::PopRequest;
using tensor::PopReply;

using tensor::TensorService;

typedef std::map<std::string, tensor::Tensor> TensorBuffer;
typedef std::pair<std::string, tensor::Tensor> TensorPair;

class TensorServiceImpl final : public TensorService::Service {
public:
  TensorServiceImpl(TensorBuffer *buffer) : buffer_(buffer) {}

  // Ping, Say Hello
  Status Hello(ServerContext* context, const HelloRequest* request, HelloReply* response) override;
  // Get tensor size
  Status GetSize(ServerContext* context, const GetSizeRequest* request, GetSizeReply* response) override;
  // Get/Set/Del tensor
  Status Get(ServerContext* context, const GetRequest* request, GetReply* response) override;
  Status Set(ServerContext* context, const SetRequest* request, SetReply* response) override;
  Status Del(ServerContext* context, const DelRequest* request, DelReply* response) override;
  // Add tensor
  Status LPush(ServerContext* context, const PushRequest* request, PushReply* response) override;
  Status RPush(ServerContext* context, const PushRequest* request, PushReply* response) override;
  // Delete tensor slice
  Status LPop(ServerContext* context, const PopRequest* request, PopReply* response) override;
  Status RPop(ServerContext* context, const PopRequest* request, PopReply* response) override;

private:
  std::unique_ptr<TensorBuffer> buffer_;
};

Status TensorServiceImpl::Hello(ServerContext* context, const HelloRequest* request, HelloReply* response) {
  std::string prefix("Hello ");
  response->set_message(prefix + request->name());
  return Status::OK;
}

Status TensorServiceImpl::GetSize(ServerContext* context, const GetSizeRequest* request, GetSizeReply* response) {
  TensorBuffer::iterator it = buffer_->find(request->id());
  if(it == buffer_->end()) {
    std::cout << "NOT Found" << std::endl;
    response->clear_size();
    return Status::OK;
  }
  // Found
  // response->set_allocated_size(it->second.size());
  return Status::OK;
}

Status TensorServiceImpl::Get(ServerContext* context, const GetRequest* request, GetReply* response) {
  return Status::OK;
}

Status TensorServiceImpl::Set(ServerContext* context, const SetRequest* request, SetReply* response) {
  buffer_->insert(TensorPair(request->id(), request->tensor()));
  return Status::OK;
}

Status TensorServiceImpl::Del(ServerContext* context, const DelRequest* request, DelReply* response) {
  buffer_->erase(request->id());
  response->set_message("OK");
  return Status::OK;
}

Status TensorServiceImpl::LPush(ServerContext* context, const PushRequest* request, PushReply* response) {
  return Status::OK;
}

Status TensorServiceImpl::RPush(ServerContext* context, const PushRequest* request, PushReply* response) {
  return Status::OK;
}

Status TensorServiceImpl::LPop(ServerContext* context, const PopRequest* request, PopReply* response) {
  return Status::OK;
}

Status TensorServiceImpl::RPop(ServerContext* context, const PopRequest* request, PopReply* response) {
  return Status::OK;
}

// // Logic and data behind the server's behavior.
// class TensorServiceImpl final : public TensorService::Service {
//   Status Hello(ServerContext* context, const HelloRequest* request, HelloReply* reply) override {
//     std::string prefix("Hello ");
//     reply->set_message(prefix + request->name());
//     return Status::OK;
//   }
// };

void RunServer(std::string endpoint) {
  TensorBuffer buffer;

  TensorServiceImpl service(&buffer);

  ServerBuilder builder;
  // Listen on the given address without any authentication mechanism.
  builder.AddListeningPort(endpoint, grpc::InsecureServerCredentials());
  // Register "service" as the instance through which we'll communicate with
  // clients. In this case it corresponds to an *synchronous* service.
  builder.RegisterService(&service);
  // Finally assemble the server.
  std::unique_ptr<Server> server(builder.BuildAndStart());
  std::cout << "Server listening on " << endpoint << std::endl;

  // Wait for the server to shutdown. Note that some other thread must be
  // responsible for shutting down the server for this call to ever return.
  server->Wait();
}

int main(int argc, char** argv) {
  RunServer("0.0.0.0:50051");

  return 0;
}

