#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <atomic>

#include <grpcpp/grpcpp.h>
#include "slm.grpc.pb.h"
#include "model.h"

using namespace std;

static const int NUM_REPLICAS = 4;

class SlmServiceImpl final : public slm::SlmService::Service {
public:
    SlmServiceImpl() : nextReplica(0) {
        cout << "Loading " << NUM_REPLICAS << " model replicas..." << endl;

        for (int i = 0; i < NUM_REPLICAS; i++) {
            models.push_back(make_unique<tinyLM>());
            models.back()->loadPretrained("weights");
        }

        cout << "All replicas loaded." << endl;
    }

    grpc::Status generate(
        grpc::ServerContext*,
        const slm::GenerateRequest* request,
        slm::GenerateResponse* response) override
    {
        const string& prompt = request->prompt();
        int maxNew = request->maxnew();

        int id = nextReplica.fetch_add(1) % NUM_REPLICAS;

        string output = models[id]->generate(prompt, maxNew);
        response->set_text(output);

        return grpc::Status::OK;
    }

private:
    vector<unique_ptr<tinyLM>> models;
    atomic<int> nextReplica;
};

void runGrpcServer() {
    string address = "0.0.0.0:50051";

    SlmServiceImpl service;

    grpc::ServerBuilder builder;
    builder.AddListeningPort(address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);

    unique_ptr<grpc::Server> server(builder.BuildAndStart());
    cout << "SLM gRPC server listening on " << address << endl;

    server->Wait();
}
