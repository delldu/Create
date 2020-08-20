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
using grpc::StatusCode;

// Tensor
using tensor::DelTensorReply;
using tensor::DelTensorRequest;
using tensor::GetTensorReply;
using tensor::GetTensorRequest;
using tensor::HelloReply;
using tensor::HelloRequest;
using tensor::SetTensorReply;
using tensor::SetTensorRequest;
using tensor::ChkTensorReply;
using tensor::ChkTensorRequest;
using tensor::TensorService;

// Image clean
using tensor::ImageCleanReply;
using tensor::ImageCleanRequest;
using tensor::ImageCleanService;

typedef std::map<std::string, tensor::Tensor> TensorBuffer;
// typedef std::pair<std::string, tensor::Tensor> TensorPair;

class TensorServiceImpl final : public TensorService::Service {
public:
    // Ping, Say Hello
    Status Hello(ServerContext* context, const HelloRequest* request, HelloReply* response) override;

    // Get/Set/Delete/Check tensor
    Status GetTensor(ServerContext* context, const GetTensorRequest* request, GetTensorReply* response) override;
    Status SetTensor(ServerContext* context, const SetTensorRequest* request, SetTensorReply* response) override;
    Status DelTensor(ServerContext* context, const DelTensorRequest* request, DelTensorReply* response) override;
    Status ChkTensor(ServerContext* context, const ChkTensorRequest* request, ChkTensorReply* response) override;

    TensorBuffer* BufferAddress() { return &m_buffer; }

    void DebugTensorBuffer(const std::string& prompt) {
    	TensorBuffer::iterator it;

    	std::cout << " ------ " << prompt << " ----------- " << std::endl;
    	for (it = m_buffer.begin(); it != m_buffer.end(); it++) {
    		std::cout << it->first << ": " << it->second.n() << "x" << it->second.c() << "x" << it->second.h() << "x" << it->second.w() << ", ";
    		std::cout << it->second.data().substr(0, 10) << std::endl;
    	}
    }

    ~TensorServiceImpl()
    {
        m_buffer.clear();
    }

private:
    TensorBuffer m_buffer;
};


class ImageCleanServiceImpl final : public ImageCleanService::Service {
public:
    ImageCleanServiceImpl(TensorBuffer* bufferaddr)
        : m_buffer_ptr(bufferaddr)
    {
        // Load model ...
    }
    Status ImageClean(ServerContext* context, const ImageCleanRequest* request, ImageCleanReply* response) override;

    ~ImageCleanServiceImpl() {
        // Release model ...
    }

private:
    TensorBuffer* m_buffer_ptr;
};


#endif // TENSOR_SERVER_H
