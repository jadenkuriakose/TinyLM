#include <iostream>
#include <string>
#include <cstdlib>
#include "model.h"

using namespace std;

void runGrpcServer();

int main(int argc, char** argv) {
    srand(42);

    if (argc >= 2 && string(argv[1]) == "--grpc") {
        runGrpcServer();
        return 0;
    }

    if (argc < 3) {
        cout << "./slm \"prompt\" maxNew\n";
        cout << "./slm --grpc\n";
        return 1;
    }

    tinyLM model;
    model.loadPretrained("weights");

    string prompt = argv[1];
    int maxNew = atoi(argv[2]);

    cout << "prompt: " << prompt << endl;
    cout << "output: " << model.generate(prompt, maxNew) << endl;
    return 0;
}
