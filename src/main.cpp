#include <iostream>
#include <string>
#include "model.h"

using std::cout;
using std::endl;
using std::string;

int main(int argc, char** argv) {
    // small config so cpu runs quickly
    int vocabSize = 256;   // char-level
    int hiddenDim = 64;
    int numHeads = 4;
    int maxSeqLen = 256;

    tinyLM model(vocabSize, hiddenDim, numHeads, maxSeqLen);

    string prompt = "hello";
    if (argc >= 2) {
        prompt = argv[1];
    }

    int maxNewTokens = 64;
    if (argc >= 3) {
        maxNewTokens = std::atoi(argv[2]);
    }

    auto promptIds = encodeText(prompt);
    auto outIds = generate(model, promptIds, maxNewTokens);
    auto outText = decodeText(outIds);

    cout << "prompt: " << prompt << endl;
    cout << "output: " << outText << endl;

    return 0;
}
