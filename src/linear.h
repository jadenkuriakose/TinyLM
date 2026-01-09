#pragma once

#include "tensor.h"
#include <random>

using std::mt19937;
using std::normal_distribution;

// linear layer
// y = x * w^t + b
struct linear {
    int inputDim;
    int outputDim;

    tensor weight; // [out, in]
    tensor bias;   // [1, out]

    linear() : inputDim(0), outputDim(0) {}

    linear(int inDim, int outDim)
        : inputDim(inDim),
          outputDim(outDim),
          weight(outDim, inDim),
          bias(1, outDim) {

        mt19937 gen(42);
        normal_distribution<float> dist(0.0f, 0.02f);

        for (auto& x : weight.data) x = dist(gen);
        bias.fill(0.0f);
    }

    tensor forward(const tensor& x) const {
        // x: [n, in]
        // w: [out, in]
        // y: [n, out]
        tensor y(x.rows, outputDim);

        for (int r = 0; r < x.rows; r++) {
            for (int o = 0; o < outputDim; o++) {
                float sum = bias(0, o);
                for (int i = 0; i < inputDim; i++) {
                    sum += x(r, i) * weight(o, i);
                }
                y(r, o) = sum;
            }
        }
        return y;
    }
};
