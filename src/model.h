#pragma once
#include "transformer.h"
#include <string>
#include <vector>

using std::string;
using std::vector;

//  tokenizer 
struct tokenizer {
    vector<string> vocab;

    tokenizer() {
        for (int i = 0; i < 256; i++)
            vocab.push_back(string(1, (char)i));
    }

    vector<int> encode(const string& text) const {
        vector<int> ids;
        for (unsigned char c : text)
            ids.push_back((int)c);
        return ids;
    }

    string decode(const vector<int>& ids) const {
        string out;
        for (int id : ids) {
            if (id >= 32 && id <= 126)
                out.push_back((char)id);
        }
        return out;
    }
};

//  tinyLM 
struct tinyLM {
    int dim;
    int maxLen;
    int pos;

    transformerBlock block;
    linear lmHead;
    kvCache cache;
    tokenizer tok;

    tinyLM(int d, int maxL)
        : dim(d),
          maxLen(maxL),
          pos(0),
          block(d),
          lmHead(d, 256),
          cache(maxL, d) {}

    void reset() {
        pos = 0;
        cache.k.fill(0.0f);
        cache.v.fill(0.0f);
    }

    tensor step(int tokenId) {
        tensor x(1, dim);
        x.fill((float)tokenId / 255.0f);
        tensor h = block.forwardToken(x, pos, cache);
        pos++;
        return lmHead.forward(h);
    }

    void loadPretrained(const string& dir) {
        block.qProj.loadWeights(dir + "/qProj.bin", dir + "/qProjBias.bin");
        block.kProj.loadWeights(dir + "/kProj.bin", dir + "/kProjBias.bin");
        block.vProj.loadWeights(dir + "/vProj.bin", dir + "/vProjBias.bin");
        block.oProj.loadWeights(dir + "/oProj.bin", dir + "/oProjBias.bin");

        block.fc1.loadWeights(dir + "/fc1.bin", dir + "/fc1Bias.bin");
        block.fc2.loadWeights(dir + "/fc2.bin", dir + "/fc2Bias.bin");

        block.attnNorm.loadWeights(dir + "/attnNorm.bin");
        block.mlpNorm.loadWeights(dir + "/mlpNorm.bin");

        lmHead.loadWeights(dir + "/lmHead.bin", dir + "/lmHeadBias.bin");
    }

    string generate(const string& prompt, int maxNew) {
        reset();
        auto ids = tok.encode(prompt);
        tensor logits;

        for (int id : ids) logits = step(id);

        for (int i = 0; i < maxNew && pos < maxLen; i++) {
            int next = 32 + rand() % 95;
            ids.push_back(next);
            logits = step(next);
        }
        return tok.decode(ids);
    }
};
