#include "tensor_server.h"

Status TensorServiceImpl::Hello(ServerContext* context, const HelloRequest* request, HelloReply* response)
{
    std::string prefix("Hello ");
    response->set_message(prefix + request->name());
    return Status::OK;
}

Status TensorServiceImpl::GetTensor(ServerContext* context, const GetTensorRequest* request, GetTensorReply* response)
{
    TensorBuffer::iterator it = m_buffer.find(request->id());

    if (it == m_buffer.end()) {
        response->clear_tensor();
        return Status(StatusCode::NOT_FOUND, "Tensor not found.");
    }

    response->mutable_tensor()->CopyFrom(it->second);
    return Status::OK;
}

Status TensorServiceImpl::SetTensor(ServerContext* context, const SetTensorRequest* request, SetTensorReply* response)
{
    m_buffer[request->id()] = request->tensor();
    return Status::OK;
}

Status TensorServiceImpl::DelTensor(ServerContext* context, const DelTensorRequest* request, DelTensorReply* response)
{
    m_buffer.erase(request->id());
    return Status::OK;
}

Status TensorServiceImpl::CheckID(ServerContext* context, const CheckIDRequest* request, CheckIDReply* response)
{
    TensorBuffer::iterator it = m_buffer.find(request->id());
    return (it == m_buffer.end()) ? (Status(StatusCode::NOT_FOUND, "Tensor not found.")) : (Status::OK);
}

Status ImageCleanServiceImpl::ImageClean(ServerContext* context, const ImageCleanRequest* request, ImageCleanReply* response)
{
    return Status::OK;
}

void StartImageCleanServer(std::string endpoint)
{
    TensorServiceImpl tensor_service;
    ImageCleanServiceImpl image_clean_service(tensor_service.BufferAddress());

    ServerBuilder builder;
    builder.SetMaxReceiveMessageSize(32 * 1024 * 1024);
    builder.SetMaxSendMessageSize(32 * 1024 * 1024);
    builder.AddListeningPort(endpoint, grpc::InsecureServerCredentials());

    builder.RegisterService(&tensor_service);
    builder.RegisterService(&image_clean_service);

    std::unique_ptr<Server> server(builder.BuildAndStart());
    std::cout << "Server listening on " << endpoint << std::endl;

    server->Wait();
}

int main(int argc, char** argv)
{
    StartImageCleanServer("0.0.0.0:50051");

    return 0;
}
