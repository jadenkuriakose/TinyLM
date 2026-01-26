#include <iostream>
#include <string>
#include <cstdlib>
#include "model.h"

using namespace std;

int main(int argc, char** argv) {
    srand(42);

    if (argc < 3) {
        cout << "./SLM \"prompt\" maxNew\n";
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
