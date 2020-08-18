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

// using tensor::HelloRequest;
// using tensor::HelloReply;
using tensor::TensorService;

// Logic and data behind the server's behavior.
class TensorServiceImpl final : public TensorService::Service {
  // Status SayHello(ServerContext* context, const HelloRequest* request,
  //                 HelloReply* reply) override {
  //   std::string prefix("Hello ");
  //   reply->set_message(prefix + request->name());
  //   return Status::OK;
  // }
};

void RunServer() {
  std::string server_address("0.0.0.0:50051");
  TensorServiceImpl service;

  ServerBuilder builder;
  // Listen on the given address without any authentication mechanism.
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  // Register "service" as the instance through which we'll communicate with
  // clients. In this case it corresponds to an *synchronous* service.
  builder.RegisterService(&service);
  // Finally assemble the server.
  std::unique_ptr<Server> server(builder.BuildAndStart());
  std::cout << "Server listening on " << server_address << std::endl;

  // Wait for the server to shutdown. Note that some other thread must be
  // responsible for shutting down the server for this call to ever return.
  server->Wait();
}

int main(int argc, char** argv) {
  RunServer();

  return 0;
}

