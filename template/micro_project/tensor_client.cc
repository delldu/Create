#include "tensor_client.h"

#include <random>

#define DebugErrorMessage(tag, status)                                         \
    do {                                                                       \
        std::cout << tag << ": " << __FILE__ << "," << __LINE__ << std::endl;  \
        std::cout << "Error Code: " << status.error_code() << std::endl;       \
        std::cout << "Error Message: " << status.error_message() << std::endl; \
    } while (0)

std::string UUID()
{
    std::ostringstream os;
    // cat /proc/sys/kernel/random/uuid
    std::ifstream file("/proc/sys/kernel/random/uuid");
    os << file.rdbuf();
    file.close();

    int len = os.str().size();
    return os.str().substr(0, len - 1); // Remove 0x0a
}

// Create random Tensor
tensor::Tensor CreateTensor()
{
    std::default_random_engine e(time(0));
    std::uniform_int_distribution<unsigned> u(32, 128);
    tensor::Tensor tensor;

    int n = 1; // u(e) % 20 + 10;
    int c = 3; // u(e) % 3 + 1;
    int h = 2048; // u(e) % 767 + 1;
    int w = 4096; // u(e) % 1023 + 1;

    tensor.set_n(n);
    tensor.set_c(c);
    tensor.set_h(h);
    tensor.set_w(w);

    std::string d(n * c * h * w, ' ');
    for (int i = 0; i < n * c * h * w; i++) {
        d[i] = u(e);
    }

    tensor.set_data(d);

    return tensor;
}

bool SameTensor(const tensor::Tensor& t1, const tensor::Tensor& t2)
{
    if (t1.n() != t2.n() || t1.c() != t2.c() || t1.h() != t2.h() || t1.w() != t2.w())
        return false;

    std::cout << "t1.data: " << t1.data().substr(0, 10) << std::endl;
    std::cout << "t2.data: " << t2.data().substr(0, 10) << std::endl;

    return (t1.data() == t2.data());
}

std::string TensorServiceClient::Hello(const std::string& user)
{
    HelloRequest request;
    HelloReply reply;
    ClientContext context;

    request.set_name(user);
    Status status = m_tensor_service_stub->Hello(&context, request, &reply);

    return (status.ok()) ? reply.message() : "Hello RPC Failed";
}

bool TensorServiceClient::SetTensor(const std::string& id, const tensor::Tensor& tensor)
{
    SetTensorRequest request;
    SetTensorReply reply;
    ClientContext context;

    request.set_id(id);
    request.mutable_tensor()->CopyFrom(tensor);
    Status status = m_tensor_service_stub->SetTensor(&context, request, &reply);

    return status.ok();
}

bool TensorServiceClient::GetTensor(const std::string& id, tensor::Tensor& tensor)
{
    GetTensorRequest request;
    GetTensorReply reply;
    ClientContext context;

    request.set_id(id);
    Status status = m_tensor_service_stub->GetTensor(&context, request, &reply);

    if (status.ok()) {
        // Save tensor
        tensor = reply.tensor();
        return true;
    }

    return false;
}

bool TensorServiceClient::DelTensor(const std::string& id)
{
    DelTensorRequest request;
    DelTensorReply reply;
    ClientContext context;

    request.set_id(id);
    Status status = m_tensor_service_stub->DelTensor(&context, request, &reply);

    return status.ok();
}

bool TensorServiceClient::CheckID(const std::string& id)
{
    CheckIDRequest request;
    CheckIDReply reply;
    ClientContext context;

    request.set_id(id);
    Status status = m_tensor_service_stub->CheckID(&context, request, &reply);

    return status.ok();
}

int main(int argc, char** argv)
{
    grpc::ChannelArguments channel_args;
    channel_args.SetInt(GRPC_ARG_MAX_RECEIVE_MESSAGE_LENGTH, 32 * 1024 * 1024);
    channel_args.SetInt(GRPC_ARG_MAX_SEND_MESSAGE_LENGTH, 32 * 1024 * 1024);
    std::shared_ptr<Channel> channel = grpc::CreateCustomChannel("localhost:50051", grpc::InsecureChannelCredentials(), channel_args);

    // TensorServiceClient connect(grpc::CreateChannel("localhost:50051", grpc::InsecureChannelCredentials()));
    TensorServiceClient connect(channel);
    auto tensor = CreateTensor();
    std::string uuid = UUID();
    std::cout << "uuid: " << uuid << std::endl;

    for (int i = 0; i < 1000; i++) {
        std::string user("world");
        std::string reply = connect.Hello(user);
        std::cout << "Hello received: " << reply << std::endl;

        bool ok = connect.SetTensor(uuid, tensor);
        std::cout << "--- SetTensor: -----" << ok << ", expected true" << std::endl;

        tensor::Tensor b;
        ok = connect.GetTensor(uuid, b);
        std::cout << "--- GetTensor: -----" << ok << ", expected true" << std::endl;

        // if (SameTensor(tensor, b)) {
        //   std::cout << "tensor == b" << std::endl;
        // } else {
        //   std::cout << "tensor != b" << std::endl;
        // }

        ok = connect.CheckID(uuid);
        std::cout << "--- CheckID: -----" << ok << ", expected true" << std::endl;

        ok = connect.DelTensor(uuid);
        std::cout << "--- DelTensor " << uuid << ": -----" << ok << ", expected true" << std::endl;

        ok = connect.CheckID("12345");
        std::cout << "--- CheckID 12345: -----" << ok << ", expected false" << std::endl;

        // tensor::TensorSize size;
        // size.set_n(10);
        // size.set_c(20);
        // size.set_h(30);
        // size.set_w(40);

        // std::cout << "size: " << size.n() << "x" << size.c() << "x" << size.h() << "x" << size.w() << std::endl;
    }

    return 0;
}
