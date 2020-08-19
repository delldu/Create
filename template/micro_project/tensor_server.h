#ifndef TENSOR_SERVER_H
#define TENSOR_SERVER_H

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

// Tensor
using tensor::TensorService;
using tensor::HelloRequest;
using tensor::HelloReply;
using tensor::GetTensorRequest;
using tensor::GetTensorReply;
using tensor::SetTensorRequest;
using tensor::SetTensorReply;
using tensor::DelTensorRequest;
using tensor::DelTensorReply;

// Image clean
using tensor::ImageCleanService;
using tensor::ImageCleanRequest;
using tensor::ImageCleanReply;

typedef std::map<std::string, tensor::Tensor> TensorBuffer;
// typedef std::pair<std::string, tensor::Tensor> TensorPair;

class TensorServiceImpl final : public TensorService::Service {
public:
  // Ping, Say Hello
  Status Hello(ServerContext* context, const HelloRequest* request, HelloReply* response) override;

  // Get/Set/Del tensor
  Status Get(ServerContext* context, const GetTensorRequest* request, GetTensorReply* response) override;
  Status Set(ServerContext* context, const SetTensorRequest* request, SetTensorReply* response) override;
  Status Del(ServerContext* context, const DelTensorRequest* request, DelTensorReply* response) override;

  TensorBuffer* BufferAddress() { return &m_buffer; }

~TensorServiceImpl() {
  m_buffer.clear();
}

private:
  TensorBuffer m_buffer;
};

#endif // TENSOR_SERVER_H