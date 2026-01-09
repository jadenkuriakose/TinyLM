#pragma once
#include <fstream>
#include <string>
using std::string;
using std::ifstream;
using std::runtime_error;
using std::ios;

inline void loadBinary(const string& path, float* dst, int count) {
    ifstream f(path, ios::binary);
    if (!f) {
        throw runtime_error("failed to open weight file: " + path);
    }
    f.read(reinterpret_cast<char*>(dst), count * sizeof(float));
    f.close();
}
