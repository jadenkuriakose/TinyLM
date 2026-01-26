#pragma once
#include "tensor.h"
#include "weights.h"
#include <random>
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>

using namespace std;

struct linear {
    int inDim;
    int outDim;
    tensor weight;
    tensor bias;

    vector<int8_t> qWeight;
    vector<float> wScale;

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

    void quantizeWeights() {
        qWeight.assign((size_t)outDim * (size_t)inDim, 0);
        wScale.assign((size_t)outDim, 1.0f);

        for (int o = 0; o < outDim; o++) {
            float maxAbs = 0.0f;
            for (int i = 0; i < inDim; i++) {
                float v = weight(o, i);
                float a = fabs(v);
                if (a > maxAbs) maxAbs = a;
            }

            float s = (maxAbs > 0.0f) ? (maxAbs / 127.0f) : 1.0f;
            wScale[(size_t)o] = s;

            float inv = 1.0f / s;
            size_t base = (size_t)o * (size_t)inDim;
            for (int i = 0; i < inDim; i++) {
                float q = weight(o, i) * inv;
                int qi = (int)lrintf(q);
                qi = max(-127, min(127, qi));
                qWeight[base + (size_t)i] = (int8_t)qi;
            }
        }
    }

    tensor forward(const tensor& x) const {
        tensor y(x.rows, outDim);

        if (!qWeight.empty() && (int)wScale.size() == outDim) {
            for (int r = 0; r < x.rows; r++) {
                for (int o = 0; o < outDim; o++) {
                    float sum = bias(0, o);
                    float s = wScale[(size_t)o];
                    size_t base = (size_t)o * (size_t)inDim;
                    for (int i = 0; i < inDim; i++) {
                        sum += x(r, i) * ((float)qWeight[base + (size_t)i] * s);
                    }
                    y(r, o) = sum;
                }
            }
            return y;
        }

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

    void loadWeights(const string& wPath, const string& bPath) {
        loadBinary(wPath, weight.data.data(), (int)weight.data.size());
        loadBinary(bPath, bias.data.data(), (int)bias.data.size());
        quantizeWeights();
    }
};
