#pragma once
#include "linear.h"
#include <cmath>
#include <vector>

using namespace std;

inline float gelu(float x) {
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + tanh(0.79788456f * (x + 0.044715f * x3)));
}

struct rmsNorm {
    int dim;
    float eps;
    vector<float> weight;

    rmsNorm() : dim(0), eps(1e-6f) {}
    rmsNorm(int d) : dim(d), eps(1e-6f), weight(d, 1.0f) {}

    tensor forward(const tensor& x) const {
        tensor out(1, dim);
        float meanSq = 0.0f;
        for (int i = 0; i < dim; i++) meanSq += x(0, i) * x(0, i);
        meanSq /= (float)dim;

        float scale = 1.0f / sqrt(meanSq + eps);
        for (int i = 0; i < dim; i++) out(0, i) = x(0, i) * scale * weight[i];
        return out;
    }

    void loadWeights(const string& path) {
        loadBinary(path, weight.data(), dim);
    }
};

struct kvCache {
    int maxLen;
    int dim;
    tensor k;
    tensor v;

    kvCache() : maxLen(0), dim(0) {}
    kvCache(int m, int d) : maxLen(m), dim(d), k(m, d), v(m, d) {}

    void clear() {
        k.fill(0.0f);
        v.fill(0.0f);
    }

    void write(int pos, const tensor& kt, const tensor& vt) {
        for (int i = 0; i < dim; i++) {
            k(pos, i) = kt(0, i);
            v(pos, i) = vt(0, i);
        }
    }
};

struct transformerBlock {
    int dim;

    linear qProj, kProj, vProj, oProj;
    linear fc1, fc2;
    rmsNorm attnNorm, mlpNorm;

    transformerBlock() : dim(0) {}
    transformerBlock(int d)
        : dim(d),
          qProj(d, d), kProj(d, d), vProj(d, d), oProj(d, d),
          fc1(d, 4 * d), fc2(4 * d, d),
          attnNorm(d), mlpNorm(d) {}

    tensor forwardToken(const tensor& x, int pos, kvCache& cache) {
        tensor xNorm = attnNorm.forward(x);

        tensor q = qProj.forward(xNorm);
        tensor k = kProj.forward(xNorm);
        tensor v = vProj.forward(xNorm);
        cache.write(pos, k, v);

        tensor attnOut(1, dim);

        vector<float> scores(pos + 1);
        float maxScore = -1e30f;

        for (int s = 0; s <= pos; s++) {
            float dot = 0.0f;
            for (int i = 0; i < dim; i++) dot += q(0, i) * cache.k(s, i);
            dot /= sqrt((float)dim);
            scores[s] = dot;
            if (dot > maxScore) maxScore = dot;
        }

        float denom = 0.0f;
        for (int s = 0; s <= pos; s++) {
            scores[s] = exp(scores[s] - maxScore);
            denom += scores[s];
        }

        for (int i = 0; i < dim; i++) {
            float acc = 0.0f;
            for (int s = 0; s <= pos; s++) {
                float p = scores[s] / (denom + 1e-9f);
                acc += p * cache.v(s, i);
            }
            attnOut(0, i) = acc;
        }

        tensor attnProj = oProj.forward(attnOut);

        tensor r1(1, dim);
        for (int i = 0; i < dim; i++) r1(0, i) = x(0, i) + attnProj(0, i);

        tensor hNorm = mlpNorm.forward(r1);
        tensor h = fc1.forward(hNorm);
        for (auto& vv : h.data) vv = gelu(vv);
        tensor m = fc2.forward(h);

        tensor out(1, dim);
        for (int i = 0; i < dim; i++) out(0, i) = r1(0, i) + m(0, i);

        return out;
    }
};
