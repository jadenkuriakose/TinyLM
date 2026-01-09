#pragma once

#include "transformer.h"
#include <string>
#include <vector>

using std::string;
using std::vector;

// embedding lookup table
// tokenId -> vector in hiddenDim
struct embedding {
    int vocabSize;
    int hiddenDim;
    tensor table; // [vocabSize, hiddenDim]

    embedding() : vocabSize(0), hiddenDim(0) {}

    embedding(int vSize, int dim)
        : vocabSize(vSize),
          hiddenDim(dim),
          table(vSize, dim) {

        mt19937 gen(123);
        normal_distribution<float> dist(0.0f, 0.02f);
        for (auto& x : table.data) x = dist(gen);
    }

    tensor forward(int tokenId) const {
        tensor out(1, hiddenDim);
        int id = tokenId;
        if (id < 0) id = 0;
        if (id >= vocabSize) id = vocabSize - 1;

        for (int i = 0; i < hiddenDim; i++) {
            out(0, i) = table(id, i);
        }
        return out;
    }
};

// minimal 1-block language model
struct tinyLM {
    int vocabSize;
    int hiddenDim;
    int numHeads;
    int maxSeqLen;

    embedding tokEmbed;
    transformerBlock block;
    linear lmHead; // logits = hidden * W^t + b -> [1, vocab]

    kvCache cache;
    int curPos = 0;

    tinyLM(int vSize, int dim, int heads, int maxLen)
         : vocabSize(vSize),
          hiddenDim(dim),
          numHeads(heads),
          maxSeqLen(maxLen),
          tokEmbed(vSize, dim),
          block(dim, heads),
          lmHead(dim, vSize),
          cache(maxLen, dim),
          curPos(0) {}

    void reset() {
        curPos = 0;
        cache.k.fill(0.0f);
        cache.v.fill(0.0f);
    }

    tensor forwardNextLogits(int tokenId) {
        // embed current token, run one step, return logits for next token
        tensor x = tokEmbed.forward(tokenId);
        tensor h = block.forwardToken(x, curPos, cache);
        curPos++;
        return lmHead.forward(h); // [1, vocabSize]
    }
};

// char-level tokenizer so you can see output immediately
// token space is 0..255
inline vector<int> encodeText(const string& text) {
    vector<int> ids;
    ids.reserve((int)text.size());
    for (unsigned char c : text) {
        ids.push_back((int)c);
    }
    return ids;
}

inline string decodeText(const vector<int>& ids) {
    string out;
    out.reserve(ids.size());
    for (int id : ids) {
        unsigned char c = (unsigned char)(id & 255);
        out.push_back((char)c);
    }
    return out;
}

inline int sampleGreedy(const tensor& logits) {
    // logits: [1, vocab]
    int vocab = logits.cols;
    int bestId = 0;
    float bestVal = logits(0, 0);
    for (int i = 1; i < vocab; i++) {
        float v = logits(0, i);
        if (v > bestVal) {
            bestVal = v;
            bestId = i;
        }
    }
    return bestId;
}

inline vector<int> generate(tinyLM& model, const vector<int>& promptIds, int maxNewTokens) {
    model.reset();

    vector<int> out = promptIds;
    out.reserve((int)promptIds.size() + maxNewTokens);

    // warm up cache by feeding prompt tokens
    // we keep logits from the last prompt token to start generation
    tensor logits;
    for (int i = 0; i < (int)promptIds.size(); i++) {
        logits = model.forwardNextLogits(promptIds[i]);
    }

    // if prompt is empty, start from a default token (space)
    if (promptIds.empty()) {
        logits = model.forwardNextLogits((int)' ');
    }

    // generate new tokens
    for (int step = 0; step < maxNewTokens; step++) {
        int nextId = sampleGreedy(logits);
        out.push_back(nextId);

        if (model.curPos >= model.maxSeqLen) break;

        logits = model.forwardNextLogits(nextId);
    }

    return out;
}
