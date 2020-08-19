#include "tensor_client.h"

std::string UUID()
{
    std::ostringstream os;
    // cat /proc/sys/kernel/random/uuid
    std::ifstream file("/proc/sys/kernel/random/uuid");
    os << file.rdbuf();
    file.close();

    return os.str();
}

std::string ImageCleanServiceClient::Hello(const std::string& user)
{
    // Data we are sending to the server.
    HelloRequest request;
    request.set_name(user);

    // Container for the data we expect from the server.
    HelloReply reply;

    // Context for the client. It could be used to convey extra information to
    // the server and/or tweak certain RPC behaviors.
    ClientContext context;

    // The actual RPC.
    Status status = tensor_service_stub_->Hello(&context, request, &reply);

    // Act upon its status.
    if (! status.ok()) {
        std::cout << status.error_code() << ": " << status.error_message() << std::endl;
        return "Hello RPC Failed";
    }

    return reply.message();
}

std::string ImageCleanServiceClient::SetTensor(const std::string& id, tensor::Tensor& tensor)
{
    // Data we are sending to the server.
    SetTensorRequest request;
    request.set_id(id);
    request.set_allocated_tensor(&tensor);

    // Container for the data we expect from the server.
    SetTensorReply reply;

    // Context for the client. It could be used to convey extra information to
    // the server and/or tweak certain RPC behaviors.
    ClientContext context;

    // The actual RPC.
    Status status = tensor_service_stub_->SetTensor(&context, request, &reply);

    // Act upon its status.
    if (! status.ok()) {
        std::cout << status.error_code() << ": " << status.error_message() << std::endl;
        return "SetTensor RPC Failed.";
    }

    return reply.message();
}

std::string ImageCleanServiceClient::GetTensor(const std::string& id, tensor::Tensor& tensor)
{
    // Data we are sending to the server.
    GetTensorRequest request;
    request.set_id(id);

    // Container for the data we expect from the server.
    GetTensorReply reply;

    // Context for the client. It could be used to convey extra information to
    // the server and/or tweak certain RPC behaviors.
    ClientContext context;

    // The actual RPC.
    Status status = tensor_service_stub_->GetTensor(&context, request, &reply);

    // Act upon its status.
    if (! status.ok()) {
        std::cout << status.error_code() << ": " << status.error_message() << std::endl;
        return "GetTensor RPC Failed.";
    }
    // Save tensor
    tensor = reply.tensor();

    return reply.message();
}

std::string ImageCleanServiceClient::DelTensor(const std::string& id)
{
    // Data we are sending to the server.
    DelTensorRequest request;
    request.set_id(id);

    // Container for the data we expect from the server.
    DelTensorReply reply;

    // Context for the client. It could be used to convey extra information to
    // the server and/or tweak certain RPC behaviors.
    ClientContext context;

    // The actual RPC.
    Status status = tensor_service_stub_->DelTensor(&context, request, &reply);

    // Act upon its status.
    if (! status.ok()) {
        std::cout << status.error_code() << ": " << status.error_message() << std::endl;
        return "DelTensor RPC Failed.";
    }

    return reply.message();
}


int main(int argc, char** argv)
{
    // Instantiate the client. It requires a channel, out of which the actual RPCs
    // are created. This channel models a connection to an endpoint (in this case,
    // localhost at port 50051). We indicate that the channel isn't authenticated
    // (use of InsecureChannelCredentials()).

    std::string uuid = UUID();
    std::cout << "uuid: " << uuid << std::endl;

    ImageCleanServiceClient connect(grpc::CreateChannel("localhost:50051", grpc::InsecureChannelCredentials()));
    std::string user("world");
    std::string reply = connect.Hello(user);
    std::cout << "TensorService received: " << reply << std::endl;

    // tensor::TensorSize size;
    // size.set_n(10);
    // size.set_c(20);
    // size.set_h(30);
    // size.set_w(40);

    // std::cout << "size: " << size.n() << "x" << size.c() << "x" << size.h() << "x" << size.w() << std::endl;

    return 0;
}
