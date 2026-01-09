#pragma once

#include "linear.h"
#include <cmath>
#include <vector>

using std::sqrt;
using std::tanh;
using std::exp;
using std::vector;

inline float gelu(float x) {
    // smooth nonlinearity used in many transformers
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + tanh(0.79788456f * (x + 0.044715f * x3)));
}

// kv cache for one block
// stores concatenated heads in hiddenDim layout: [pos, hiddenDim]
struct kvCache {
    int maxSeqLen;
    int hiddenDim;

    tensor k; // [maxSeqLen, hiddenDim]
    tensor v; // [maxSeqLen, hiddenDim]

    kvCache() : maxSeqLen(0), hiddenDim(0) {}

    kvCache(int maxLen, int dim)
        : maxSeqLen(maxLen),
          hiddenDim(dim),
          k(maxLen, dim),
          v(maxLen, dim) {}

    void writeKv(int pos, const tensor& kToken, const tensor& vToken) {
        // kToken/vToken are [1, hiddenDim]
        for (int i = 0; i < hiddenDim; i++) {
            k(pos, i) = kToken(0, i);
            v(pos, i) = vToken(0, i);
        }
    }
};

// single transformer block with autoregressive forward using kv cache
struct transformerBlock {
    int hiddenDim;
    int numHeads;
    int headDim;

    linear qProj;
    linear kProj;
    linear vProj;
    linear oProj;

    linear fc1;
    linear fc2;

    transformerBlock() : hiddenDim(0), numHeads(0), headDim(0) {}

    transformerBlock(int dim, int heads)
        : hiddenDim(dim),
          numHeads(heads),
          headDim(dim / heads),
          qProj(dim, dim),
          kProj(dim, dim),
          vProj(dim, dim),
          oProj(dim, dim),
          fc1(dim, 4 * dim),
          fc2(4 * dim, dim) {}

    tensor forwardToken(const tensor& xToken, int pos, kvCache& cache) const {
        // xToken: [1, hiddenDim] for the current token only
        // pos: position index in the sequence
        // cache holds past keys/values at [0..pos]

        tensor q = qProj.forward(xToken); // [1, hiddenDim]
        tensor k = kProj.forward(xToken); // [1, hiddenDim]
        tensor v = vProj.forward(xToken); // [1, hiddenDim]

        cache.writeKv(pos, k, v);

        tensor attnOut(1, hiddenDim);

        // causal attention is implicit: we only attend to s <= pos
        // compute per-head softmax weights over past positions
        for (int h = 0; h < numHeads; h++) {

            float maxScore = -1e30f;
            vector<float> scores(pos + 1);

            // scores[s] = (q dot k_s) / sqrt(d)
            for (int s = 0; s <= pos; s++) {
                float dot = 0.0f;
                for (int i = 0; i < headDim; i++) {
                    int idx = h * headDim + i;
                    dot += q(0, idx) * cache.k(s, idx);
                }
                float score = dot / sqrt((float)headDim);
                scores[s] = score;
                if (score > maxScore) maxScore = score;
            }

            // stable softmax
            float denom = 0.0f;
            for (int s = 0; s <= pos; s++) {
                scores[s] = exp(scores[s] - maxScore);
                denom += scores[s];
            }
            float invDenom = 1.0f / (denom + 1e-9f);

            // head output = sum_s p[s] * v_s
            for (int d = 0; d < headDim; d++) {
                float acc = 0.0f;
                for (int s = 0; s <= pos; s++) {
                    float p = scores[s] * invDenom;
                    acc += p * cache.v(s, h * headDim + d);
                }
                attnOut(0, h * headDim + d) = acc;
            }
        }

        // output projection
        tensor attnProj = oProj.forward(attnOut);

        // residual 1: x + attn(x)
        tensor resid1(1, hiddenDim);
        for (int i = 0; i < hiddenDim; i++) {
            resid1(0, i) = xToken(0, i) + attnProj(0, i);
        }

        // mlp: fc1 -> gelu -> fc2
        tensor hidden = fc1.forward(resid1);
        for (auto& val : hidden.data) val = gelu(val);
        tensor mlpOut = fc2.forward(hidden);

        // residual 2: resid1 + mlp(resid1)
        tensor out(1, hiddenDim);
        for (int i = 0; i < hiddenDim; i++) {
            out(0, i) = resid1(0, i) + mlpOut(0, i);
        }

        return out;
    }
};
