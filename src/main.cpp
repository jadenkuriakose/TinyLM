#include <iostream>
#include "model.h"

using std::cout;
using std::endl;

int main(int argc, char** argv) {
    string prompt = "hello";
    if (argc > 1) prompt = argv[1];

    tinyLM model(64, 256);
    model.loadPretrained("weights");
    string out = model.generate(prompt, 80);

    cout << "prompt: " << prompt << endl;
    cout << "output: " << out << endl;
}
