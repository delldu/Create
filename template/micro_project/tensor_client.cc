#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>
#include <string>

#include <grpc++/grpc++.h>  

#include "tensor.grpc.pb.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using tensor::HelloRequest;
using tensor::HelloReply;
using tensor::TensorService;

std::string tensor_uuid() {
  int i;
  std::ostringstream os;
  // cat /proc/sys/kernel/random/uuid
  std::ifstream file("/proc/sys/kernel/random/uuid");
  os << file.rdbuf();
  file.close();

  return os.str();
}

class TensorServiceClient {
 public:
  TensorServiceClient(std::shared_ptr<Channel> channel)
      : stub_(TensorService::NewStub(channel)) {}

  // Assembles the client's payload, sends it and presents the response back
  // from the server.
  std::string Hello(const std::string& user) {
    // Data we are sending to the server.
    HelloRequest request;
    request.set_name(user);

    // Container for the data we expect from the server.
    HelloReply reply;

    // Context for the client. It could be used to convey extra information to
    // the server and/or tweak certain RPC behaviors.
    ClientContext context;

    // The actual RPC.
    Status status = stub_->Hello(&context, request, &reply);

    // Act upon its status.
    if (status.ok()) {
      return reply.message();
    } else {
      std::cout << status.error_code() << ": " << status.error_message()
                << std::endl;
      return "RPC failed";
    }
  }

 private:
  std::unique_ptr<TensorService::Stub> stub_;
};

int main(int argc, char** argv) {
  // Instantiate the client. It requires a channel, out of which the actual RPCs
  // are created. This channel models a connection to an endpoint (in this case,
  // localhost at port 50051). We indicate that the channel isn't authenticated
  // (use of InsecureChannelCredentials()).

  // TensorServiceClient connect(grpc::CreateChannel("localhost:50051", grpc::InsecureChannelCredentials()));
  // std::string user("world");
  // std::string reply = connect.Hello(user);
  // std::cout << "TensorService received: " << reply << std::endl;

  // tensor::TensorSize size;
  // size.set_n(10);
  // size.set_c(20);
  // size.set_h(30);
  // size.set_w(40);

  // std::cout << "size: " << size.n() << "x" << size.c() << "x" << size.h() << "x" << size.w() << std::endl;

  std::string uuid = tensor_uuid();
  std::cout << "uuid: " << uuid << std::endl;

  return 0;
}

