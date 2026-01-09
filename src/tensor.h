#pragma once

#include <vector>

using std::vector;

// simple 2d tensor (row-major)
struct tensor {
    int rows;
    int cols;
    vector<float> data;

    tensor() : rows(0), cols(0) {}
    tensor(int r, int c) : rows(r), cols(c), data(r * c) {}

    inline float& operator()(int i, int j) {
        return data[i * cols + j];
    }

    inline const float& operator()(int i, int j) const {
        return data[i * cols + j];
    }

    void fill(float v) {
        for (auto& x : data) x = v;
    }
};
