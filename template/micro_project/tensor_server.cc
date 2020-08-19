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
using tensor::GetTensorRequest;
using tensor::GetTensorReply;
using tensor::SetTensorRequest;
using tensor::SetTensorReply;
using tensor::DelTensorRequest;
using tensor::DelTensorReply;

using tensor::TensorService;

typedef std::map<std::string, tensor::Tensor> TensorBuffer;
typedef std::pair<std::string, tensor::Tensor> TensorPair;

class TensorServiceImpl final : public TensorService::Service {
public:
  // TensorServiceImpl() {}

  // Ping, Say Hello
  Status Hello(ServerContext* context, const HelloRequest* request, HelloReply* response) override;

  // Get/Set/Del tensor
  Status Get(ServerContext* context, const GetTensorRequest* request, GetTensorReply* response) override;
  Status Set(ServerContext* context, const SetTensorRequest* request, SetTensorReply* response) override;
  Status Del(ServerContext* context, const DelTensorRequest* request, DelTensorReply* response) override;

private:
  TensorBuffer m_buffer;
};

Status TensorServiceImpl::Hello(ServerContext* context, const HelloRequest* request, HelloReply* response) {
  std::string prefix("Hello ");
  response->set_message(prefix + request->name());
  return Status::OK;
}

Status TensorServiceImpl::Get(ServerContext* context, const GetTensorRequest* request, GetTensorReply* response) {
  return Status::OK;
}

Status TensorServiceImpl::Set(ServerContext* context, const SetTensorRequest* request, SetTensorReply* response) {
  m_buffer[request->id()] = request->tensor();
  return Status::OK;
}

Status TensorServiceImpl::Del(ServerContext* context, const DelTensorRequest* request, DelTensorReply* response) {
  m_buffer.erase(request->id());
  response->set_message("OK");
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
  TensorServiceImpl service;

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

