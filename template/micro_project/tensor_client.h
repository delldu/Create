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
using tensor::ChkTensorRequest;
using tensor::ChkTensorReply;

using tensor::ImageCleanService;
using tensor::ImageCleanRequest;
using tensor::ImageCleanReply;

class TensorServiceClient {
public:
    TensorServiceClient(std::shared_ptr<Channel> channel)
    {
        m_tensor_service_stub = TensorService::NewStub(channel);
    }

    std::string Hello(const std::string& user);
	bool SetTensor(const std::string& id, const tensor::Tensor& tensor);
	bool GetTensor(const std::string& id, tensor::Tensor& tensor);
	bool DelTensor(const std::string& id);
    bool ChkTensor(const std::string& id);

    ~TensorServiceClient() {
    }

private:
    std::unique_ptr<TensorService::Stub> m_tensor_service_stub;
};

class ImageCleanServiceClient : public TensorServiceClient {
public:
    ImageCleanServiceClient(std::shared_ptr<Channel> channel) : TensorServiceClient(channel) {
        m_image_clean_service_stub = ImageCleanService::NewStub(channel);
    }

    ~ImageCleanServiceClient() {
    }
private:
    std::unique_ptr<ImageCleanService::Stub> m_image_clean_service_stub;
};


#endif // TENSOR_CLIENT_H
