#pragma once
#include "transformer.h"
#include <string>
#include <vector>
#include <algorithm>
#include <utility>
#include <cstdlib>

using namespace std;

struct embedding {
    int vocab;
    int dim;
    tensor table;

    embedding() : vocab(0), dim(0) {}
    embedding(int v, int d) : vocab(v), dim(d), table(v, d) {}

    tensor forward(int tokenId) const {
        tensor out(1, dim);
        if (tokenId < 0) tokenId = 0;
        if (tokenId >= vocab) tokenId = vocab - 1;
        for (int i = 0; i < dim; i++) out(0, i) = table(tokenId, i);
        return out;
    }

    void loadWeights(const string& path) {
        loadBinary(path, table.data.data(), (int)table.data.size());
    }
};

struct tokenizer {
    vector<int> encode(const string& text) const {
        vector<int> ids;
        for (unsigned char c : text) ids.push_back((int)c);
        return ids;
    }

    string decode(const vector<int>& ids) const {
        string out;
        for (int id : ids) if (id >= 32 && id <= 126) out.push_back((char)id);
        return out;
    }
};

inline int sampleTopKPrintable(const tensor& logits, int k, float temp) {
    vector<pair<float,int>> vals;
    vals.reserve(95);

    for (int i = 32; i <= 126; i++) vals.push_back({ logits(0, i) / temp, i });

    if (k > (int)vals.size()) k = (int)vals.size();
    nth_element(vals.begin(), vals.begin() + k, vals.end(),
        [](auto& a, auto& b){ return a.first > b.first; });
    vals.resize(k);

    float maxL = vals[0].first;
    for (auto& p : vals) if (p.first > maxL) maxL = p.first;

    float sum = 0.0f;
    for (auto& p : vals) { p.first = exp(p.first - maxL); sum += p.first; }

    float r = ((float) rand() / RAND_MAX) * sum;
    float acc = 0.0f;
    for (auto& p : vals) { acc += p.first; if (acc >= r) return p.second; }
    return vals.back().second;
}

struct tinyLM {
    int dim;
    int vocab;
    int maxLen;
    int pos;
    int nLayers;

    embedding embed;
    vector<transformerBlock> blocks;
    vector<kvCache> caches;

    linear lmHead;
    tokenizer tok;

    tinyLM(int d, int v, int maxL, int nL)
        : dim(d), vocab(v), maxLen(maxL), pos(0), nLayers(nL),
          embed(v, d),
          blocks(),
          caches(),
          lmHead(d, v),
          tok() {

        blocks.reserve(nLayers);
        caches.reserve(nLayers);
        for (int i = 0; i < nLayers; i++) {
            blocks.push_back(transformerBlock(dim));
            caches.push_back(kvCache(maxLen, dim));
        }
    }

    void reset() {
        pos = 0;
        for (int i = 0; i < nLayers; i++) caches[i].clear();
    }

    tensor step(int tokenId) {
        tensor x = embed.forward(tokenId);

        for (int i = 0; i < nLayers; i++) {
            x = blocks[i].forwardToken(x, pos, caches[i]);
        }

        pos++;
        return lmHead.forward(x);
    }

    void loadPretrained(const string& dir) {
        embed.loadWeights(dir + "/embed.bin");
        lmHead.loadWeights(dir + "/lmHead.bin", dir + "/lmHeadBias.bin");

        for (int i = 0; i < nLayers; i++) {
            string p = dir + "/layer" + to_string(i);

            blocks[i].qProj.loadWeights(p + "/qProj.bin", p + "/qProjBias.bin");
            blocks[i].kProj.loadWeights(p + "/kProj.bin", p + "/kProjBias.bin");
            blocks[i].vProj.loadWeights(p + "/vProj.bin", p + "/vProjBias.bin");
            blocks[i].oProj.loadWeights(p + "/oProj.bin", p + "/oProjBias.bin");

            blocks[i].fc1.loadWeights(p + "/fc1.bin", p + "/fc1Bias.bin");
            blocks[i].fc2.loadWeights(p + "/fc2.bin", p + "/fc2Bias.bin");

            blocks[i].attnNorm.loadWeights(p + "/attnNorm.bin");
            blocks[i].mlpNorm.loadWeights(p + "/mlpNorm.bin");
        }
    }

    string generate(const string& prompt, int maxNew, int topK=40, float temp=0.9f) {
        reset();
        vector<int> ids = tok.encode(prompt);
        tensor logits;

        for (int id : ids) logits = step(id);

        for (int i = 0; i < maxNew && pos < maxLen; i++) {
            int next = sampleTopKPrintable(logits, topK, temp);
            ids.push_back(next);
            logits = step(next);
        }

        return tok.decode(ids);
    }
};
