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

    if(it == m_buffer.end()) {
        response->clear_tensor();
        response->set_message("NOK");
        return Status(StatusCode::NOT_FOUND, "Tensor not found.");
    }

    response->set_allocated_tensor(&(it->second));
    response->set_message("OK");
    return Status::OK;
}

Status TensorServiceImpl::SetTensor(ServerContext* context, const SetTensorRequest* request, SetTensorReply* response)
{
    m_buffer[request->id()] = request->tensor();
    response->set_message("OK");
    return Status::OK;
}

Status TensorServiceImpl::DelTensor(ServerContext* context, const DelTensorRequest* request, DelTensorReply* response)
{
    m_buffer.erase(request->id());
    response->set_message("OK");
    return Status::OK;
}

class ImageCleanServiceImpl final : public ImageCleanService::Service {
public:
    ImageCleanServiceImpl(TensorBuffer* bufferaddr)
        : m_buffer_ptr(bufferaddr)
    {
    }
    Status ImageClean(ServerContext* context, const ImageCleanRequest* request, ImageCleanReply* response) override;

private:
    TensorBuffer* m_buffer_ptr;
};

Status ImageCleanServiceImpl::ImageClean(ServerContext* context, const ImageCleanRequest* request, ImageCleanReply* response)
{
    return Status::OK;
}

void StartImageCleanServer(std::string endpoint)
{
    TensorServiceImpl tensor_service;
    ImageCleanServiceImpl image_clean_service(tensor_service.BufferAddress());

    ServerBuilder builder;
    // Listen on the given address without any authentication mechanism.
    builder.AddListeningPort(endpoint, grpc::InsecureServerCredentials());
    // Register "tensor_service" as the instance through which we'll communicate with
    // clients. In this case it corresponds to an *synchronous* service.

    builder.RegisterService(&tensor_service);
    builder.RegisterService(&image_clean_service);

    // Finally assemble the server.
    std::unique_ptr<Server> server(builder.BuildAndStart());
    std::cout << "Server listening on " << endpoint << std::endl;

    // Wait for the server to shutdown. Note that some other thread must be
    // responsible for shutting down the server for this call to ever return.
    server->Wait();
}

int main(int argc, char** argv)
{
    StartImageCleanServer("0.0.0.0:50051");

    return 0;
}
