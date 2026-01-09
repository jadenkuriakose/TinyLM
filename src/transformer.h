#pragma once
#include "linear.h"
#include <cmath>
#include <vector>

using std::vector;
using std::sqrt;
using std::tanh;
using std::exp;

inline float gelu(float x) {
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + tanh(0.79788456f * (x + 0.044715f * x3)));
}

//  RMSNorm 
struct rmsNorm {
    int dim;
    float eps;
    vector<float> weight;

    rmsNorm() : dim(0), eps(1e-6f) {}
    rmsNorm(int d) : dim(d), eps(1e-6f), weight(d, 1.0f) {}

    tensor forward(const tensor& x) const {
        tensor out(1, dim);
        float meanSq = 0.0f;
        for (int i = 0; i < dim; i++)
            meanSq += x(0, i) * x(0, i);
        meanSq /= dim;

        float scale = 1.0f / sqrt(meanSq + eps);
        for (int i = 0; i < dim; i++)
            out(0, i) = x(0, i) * scale * weight[i];
        return out;
    }

    void loadWeights(const string& path) {
        loadBinary(path, weight.data(), dim);
    }
};

//  KV Cache 
struct kvCache {
    int maxLen;
    int dim;
    tensor k;
    tensor v;

    kvCache(int m, int d) : maxLen(m), dim(d), k(m, d), v(m, d) {}

    void write(int pos, const tensor& kt, const tensor& vt) {
        for (int i = 0; i < dim; i++) {
            k(pos, i) = kt(0, i);
            v(pos, i) = vt(0, i);
        }
    }
};

//  Transformer Block 
struct transformerBlock {
    int dim;

    linear qProj;
    linear kProj;
    linear vProj;
    linear oProj;

    linear fc1;
    linear fc2;

    rmsNorm attnNorm;
    rmsNorm mlpNorm;

    transformerBlock(int d)
        : dim(d),
          qProj(d, d),
          kProj(d, d),
          vProj(d, d),
          oProj(d, d),
          fc1(d, 4 * d),
          fc2(4 * d, d),
          attnNorm(d),
          mlpNorm(d) {}

    tensor forwardToken(const tensor& x, int pos, kvCache& cache) {
        tensor xn = attnNorm.forward(x);

        tensor q = qProj.forward(xn);
        tensor k = kProj.forward(xn);
        tensor v = vProj.forward(xn);

        cache.write(pos, k, v);

        tensor attnOut(1, dim);
        for (int s = 0; s <= pos; s++) {
            float score = 0.0f;
            for (int i = 0; i < dim; i++)
                score += q(0, i) * cache.k(s, i);

            for (int i = 0; i < dim; i++)
                attnOut(0, i) += score * cache.v(s, i);
        }

        tensor proj = oProj.forward(attnOut);

        tensor r1(1, dim);
        for (int i = 0; i < dim; i++)
            r1(0, i) = x(0, i) + proj(0, i);

        tensor hn = mlpNorm.forward(r1);
        tensor h = fc1.forward(hn);
        for (auto& v : h.data) v = gelu(v);
        tensor m = fc2.forward(h);

        tensor out(1, dim);
        for (int i = 0; i < dim; i++)
            out(0, i) = r1(0, i) + m(0, i);

        return out;
    }
};
