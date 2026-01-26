#pragma once
#include <string>
#include <vector>
#include <stdexcept>
#include <sentencepiece_processor.h>

using namespace std;

struct tokenizer {
    sentencepiece::SentencePieceProcessor sp;

    void load(const string& modelPath) {
        auto status = sp.Load(modelPath);
        if (!status.ok()) {
            throw runtime_error("failed to load tokenizer: " + modelPath);
        }
    }

    vector<int> encode(const string& text) const {
        vector<int> ids;
        sp.Encode(text, &ids);
        return ids;
    }

    string decode(const vector<int>& ids) const {
        string out;
        sp.Decode(ids, &out);
        return out;
    }

    int vocabSize() const {
        return (int)sp.GetPieceSize();
    }
};
