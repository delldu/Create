#ifndef TENSOR_CLIENT_H
#define TENSOR_CLIENT_H

#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#include <grpc++/grpc++.h>

#include "tensor.grpc.pb.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;

using tensor::TensorService;
using tensor::HelloReply;
using tensor::HelloRequest;
using tensor::SetTensorRequest;
using tensor::SetTensorReply;
using tensor::GetTensorRequest;
using tensor::GetTensorReply;
using tensor::DelTensorRequest;
using tensor::DelTensorReply;


using tensor::ImageCleanService;
using tensor::ImageCleanRequest;
using tensor::ImageCleanReply;

class ImageCleanServiceClient {
public:
    ImageCleanServiceClient(std::shared_ptr<Channel> channel)
        : tensor_service_stub_(TensorService::NewStub(channel))
        , image_clean_service_stub_(ImageCleanService::NewStub(channel))
    {
    }

    std::string Hello(const std::string& user);
	std::string SetTensor(const std::string& id, tensor::Tensor& tensor);
	std::string GetTensor(const std::string& id, tensor::Tensor& tensor);
	std::string DelTensor(const std::string& id);

private:
    std::unique_ptr<TensorService::Stub> tensor_service_stub_;
    std::unique_ptr<ImageCleanService::Stub> image_clean_service_stub_;
};

#endif // TENSOR_CLIENT_H
