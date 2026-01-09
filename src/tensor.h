#pragma once

#include <vector>

using std::vector;

// simple 2d tensor, cpu only
// row-major [rows, cols]
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

    int numel() const { return rows * cols; }

    void fill(float value) {
        for (auto& x : data) x = value;
    }
};
