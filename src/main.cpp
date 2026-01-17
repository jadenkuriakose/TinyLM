#include <iostream>
#include <string>
#include <cstdlib>
#include "model.h"

using namespace std;

static void printUsage() {
    cout << "usage:\n";
    cout << "  ./SLM \"prompt\" maxNew\n";
    cout << "  ./SLM --parity\n";
}

int main(int argc, char** argv) {
    srand(42);

    int dim = 128;
    int vocab = 256;
    int maxLen = 256;
    int nLayers = 4;

    string weightsDir = "weights";

    tinyLM model(dim, vocab, maxLen, nLayers);
    model.loadPretrained(weightsDir);

    if (argc >= 2 && string(argv[1]) == "--parity") {
        model.reset();
        tensor logits = model.step((int)'h');

        cout << "cpp logits[:8]: ";
        for (int i = 0; i < 8; i++) cout << logits(0, i) << " ";
        cout << endl;
        return 0;
    }

    if (argc < 3) {
        printUsage();
        return 1;
    }

    string prompt = argv[1];
    int maxNew = atoi(argv[2]);

    cout << "prompt: " << prompt << endl;
    string out = model.generate(prompt, maxNew, 40, 0.9f);
    cout << "output: " << out << endl;

    return 0;
}
