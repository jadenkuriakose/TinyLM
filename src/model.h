#pragma once
#include "transformer.h"
#include "tokenizer.h"

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <unordered_map>
#include <cstdlib>
#include <cmath>

using namespace std;

struct modelConfig {
    int dim = 192;
    int vocab = 4096;
    int maxLen = 256;
    int numLayers = 16;
    string tokenizerModel = "./tokenizer/tinyLM.model";
};

inline modelConfig loadConfig(const string& path) {
    ifstream f(path);
    if (!f) throw runtime_error("failed to open config: " + path);

    modelConfig c;
    string line;

    while (getline(f, line)) {
        auto eq = line.find('=');
        if (eq == string::npos) continue;

        string key = line.substr(0, eq);
        string val = line.substr(eq + 1);

        if (key == "dim") c.dim = stoi(val);
        else if (key == "vocab") c.vocab = stoi(val);
        else if (key == "maxLen") c.maxLen = stoi(val);
        else if (key == "numLayers") c.numLayers = stoi(val);
        else if (key == "tokenizerModel") c.tokenizerModel = val;
    }

    return c;
}

struct embedding {
    int vocab = 0;
    int dim = 0;
    tensor table;

    embedding() {}
    embedding(int v, int d) : vocab(v), dim(d), table(v, d) {}

    tensor forward(int tokenId) const {
        if (tokenId < 0) tokenId = 0;
        if (tokenId >= vocab) tokenId = vocab - 1;

        tensor out(1, dim);
        for (int i = 0; i < dim; i++) out(0, i) = table(tokenId, i);
        return out;
    }

    void loadWeights(const string& path) {
        loadBinary(path, table.data.data(), (int)table.data.size());
    }
};

inline int sampleNucleus(
    const tensor& logits,
    const vector<int>& recentTokens,
    int vocab
) {
    const float temp = 0.9f;
    const float topP = 0.95f;
    const float repPenalty = 1.25f;
    const int repWindow = 128;
    const int cap = 256;


    unordered_map<int,int> freq;
    int start = max(0, (int)recentTokens.size() - repWindow);
    for (int i = start; i < (int)recentTokens.size(); i++) freq[recentTokens[i]]++;

    vector<pair<float,int>> cand;
    cand.reserve(vocab);

    for (int i = 0; i < vocab; i++) {
        float logit = logits(0, i) / temp;
        auto it = freq.find(i);
        if (it != freq.end()) {
            logit /= pow(repPenalty, (float)it->second);
        }
        cand.push_back({logit, i});
    }

    sort(cand.begin(), cand.end(),
         [](auto& a, auto& b) { return a.first > b.first; });

    float maxLogit = cand[0].first;

    float sum = 0.0f;
    vector<float> probs;
    probs.reserve(min(vocab, cap));

    int limit = min(vocab, cap);
    for (int i = 0; i < limit; i++) {
        float p = exp(cand[i].first - maxLogit);
        probs.push_back(p);
        sum += p;

        if (i >= 8) {
            float cum = sum / (sum + 1e-9f);
            if (cum >= topP) {
                limit = i + 1;
                break;
            }
        }
    }

    float r = ((float)rand() / RAND_MAX) * sum;
    float acc = 0.0f;
    for (int i = 0; i < (int)probs.size(); i++) {
        acc += probs[i];
        if (acc >= r) return cand[i].second;
    }

    return cand[0].second;
}

struct tinyLM {
    int dim = 0;
    int vocab = 0;
    int maxLen = 0;
    int numLayers = 0;
    int pos = 0;

    embedding embed;
    vector<transformerBlock> blocks;
    vector<kvCache> caches;
    linear lmHead;

    tokenizer tok;

    tinyLM() {}

    void init(int d, int v, int maxL, int nL) {
        dim = d;
        vocab = v;
        maxLen = maxL;
        numLayers = nL;
        pos = 0;

        embed = embedding(vocab, dim);
        lmHead = linear(dim, vocab);

        blocks.clear();
        caches.clear();
        blocks.reserve(numLayers);
        caches.reserve(numLayers);

        for (int i = 0; i < numLayers; i++) {
            blocks.push_back(transformerBlock(dim));
            caches.push_back(kvCache(maxLen, dim));
        }
    }

    void reset() {
        pos = 0;
        for (int i = 0; i < numLayers; i++) caches[i].clear();
    }

    tensor step(int tokenId) {
        tensor x = embed.forward(tokenId);

        for (int i = 0; i < numLayers; i++) {
            x = blocks[i].forwardToken(x, pos, caches[i]);
        }

        pos++;
        return lmHead.forward(x);
    }

    void loadPretrained(const string& weightsDir) {
        modelConfig c = loadConfig(weightsDir + "/config.txt");
        init(c.dim, c.vocab, c.maxLen, c.numLayers);

        tok.load(c.tokenizerModel);

        embed.loadWeights(weightsDir + "/embed.bin");
        lmHead.loadWeights(weightsDir + "/lmHead.bin", weightsDir + "/lmHeadBias.bin");

        for (int i = 0; i < numLayers; i++) {
            string p = weightsDir + "/layer" + to_string(i);

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

    string generate(const string& prompt, int maxNew) {
        reset();

        vector<int> ids = tok.encode(prompt);
        vector<int> recentTokens;
        tensor logits;

        for (int id : ids) {
            logits = step(id);
            recentTokens.push_back(id);
        }

        for (int i = 0; i < maxNew && pos < maxLen; i++) {
            int next = sampleNucleus(logits, recentTokens, vocab);
            ids.push_back(next);
            recentTokens.push_back(next);
            logits = step(next);
        }

        return tok.decode(ids);
    }
};
