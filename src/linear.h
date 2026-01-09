#pragma once
#include "tensor.h"
#include "weights.h"
#include <random>

using std::mt19937;
using std::normal_distribution;

struct linear {
    int inDim;
    int outDim;
    tensor weight;
    tensor bias;

    linear() : inDim(0), outDim(0) {}

    linear(int inD, int outD)
        : inDim(inD),
          outDim(outD),
          weight(outD, inD),
          bias(1, outD) {

        mt19937 gen(42);
        normal_distribution<float> dist(0.0f, 0.02f);
        for (auto& x : weight.data) x = dist(gen);
        bias.fill(0.0f);
    }

    tensor forward(const tensor& x) const {
        tensor y(x.rows, outDim);
        for (int r = 0; r < x.rows; r++) {
            for (int o = 0; o < outDim; o++) {
                float sum = bias(0, o);
                for (int i = 0; i < inDim; i++) {
                    sum += x(r, i) * weight(o, i);
                }
                y(r, o) = sum;
            }
        }
        return y;
    }

    //  load pretrained weights 
    void loadWeights(const string& wPath, const string& bPath) {
        loadBinary(wPath, weight.data.data(), weight.data.size());
        loadBinary(bPath, bias.data.data(), bias.data.size());
    }
};
