import os
import math
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
from tqdm import tqdm
from pathlib import Path
from contextlib import nullcontext

dim = 192
maxLen = 256
nLayers = 16

batchSize = 32
seqLen = 128
lr = 3e-4
steps = 8000
logEvery = 50

vocabSize = 4096
modelType = "bpe"
characterCoverage = 0.9995

rootDir = Path(__file__).resolve().parent.parent
dataPath = rootDir / "data" / "train.txt"

tokDir = rootDir / "tokenizer"
tokPrefix = tokDir / "tinyLM"
tokModelPath = Path(str(tokPrefix) + ".model")

outDir = rootDir / "weights"
cacheDir = rootDir / "data"
tokensPath = cacheDir / "tokens.bin"

tokDir.mkdir(parents=True, exist_ok=True)
outDir.mkdir(parents=True, exist_ok=True)
cacheDir.mkdir(parents=True, exist_ok=True)

class rmsNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight

class tinyBlock(nn.Module):
    def __init__(self, dim, maxLen):
        super().__init__()
        self.attnNorm = rmsNorm(dim)
        self.qProj = nn.Linear(dim, dim)
        self.kProj = nn.Linear(dim, dim)
        self.vProj = nn.Linear(dim, dim)
        self.oProj = nn.Linear(dim, dim)

        self.mlpNorm = rmsNorm(dim)
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.fc2 = nn.Linear(4 * dim, dim)
        self.gelu = nn.GELU()

        self.register_buffer("mask", torch.tril(torch.ones(maxLen, maxLen, dtype=torch.bool)))

    def forward(self, x):
        b, t, d = x.shape
        xNorm = self.attnNorm(x)
        q = self.qProj(xNorm)
        k = self.kProj(xNorm)
        v = self.vProj(xNorm)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(d)
        scores = scores.masked_fill(~self.mask[:t, :t], -1e9)

        probs = F.softmax(scores, dim=-1)
        attn = probs @ v
        x = x + self.oProj(attn)

        h = self.mlpNorm(x)
        x = x + self.fc2(self.gelu(self.fc1(h)))
        return x

class tinyLM(nn.Module):
    def __init__(self, vocab, dim, nLayers, maxLen):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList([tinyBlock(dim, maxLen) for _ in range(nLayers)])
        self.lmHead = nn.Linear(dim, vocab)

    def forward(self, idx, targets=None):
        x = self.embed(idx)
        for b in self.blocks:
            x = b(x)
        logits = self.lmHead(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

def dump(t, path):
    t.detach().cpu().numpy().astype("float32").tofile(str(path))

def exportWeights(m, vocab):
    cfg = {
        "dim": dim,
        "maxLen": maxLen,
        "numLayers": nLayers,
        "vocab": int(vocab),
        "tokenizerModel": "./tokenizer/tinyLM.model",
    }

    (outDir / "config.txt").write_text(
        "\n".join([f"{k}={v}" for k, v in cfg.items()]),
        encoding="utf-8"
    )

    dump(m.embed.weight, outDir / "embed.bin")
    dump(m.lmHead.weight, outDir / "lmHead.bin")
    dump(m.lmHead.bias,   outDir / "lmHeadBias.bin")

    for i, b in enumerate(m.blocks):
        layerDir = outDir / f"layer{i}"
        layerDir.mkdir(parents=True, exist_ok=True)

        dump(b.qProj.weight, layerDir / "qProj.bin")
        dump(b.qProj.bias,   layerDir / "qProjBias.bin")
        dump(b.kProj.weight, layerDir / "kProj.bin")
        dump(b.kProj.bias,   layerDir / "kProjBias.bin")
        dump(b.vProj.weight, layerDir / "vProj.bin")
        dump(b.vProj.bias,   layerDir / "vProjBias.bin")
        dump(b.oProj.weight, layerDir / "oProj.bin")
        dump(b.oProj.bias,   layerDir / "oProjBias.bin")
        dump(b.fc1.weight, layerDir / "fc1.bin")
        dump(b.fc1.bias,   layerDir / "fc1Bias.bin")
        dump(b.fc2.weight, layerDir / "fc2.bin")
        dump(b.fc2.bias,   layerDir / "fc2Bias.bin")
        dump(b.attnNorm.weight, layerDir / "attnNorm.bin")
        dump(b.mlpNorm.weight,  layerDir / "mlpNorm.bin")

def trainTokenizerIfNeeded(retrainTok):
    if tokModelPath.exists() and not retrainTok:
        print("tokenizer: found", tokModelPath, flush=True)
        return

    print("tokenizer: training...", flush=True)
    spm.SentencePieceTrainer.train(
        input=str(dataPath),
        model_prefix=str(tokPrefix),
        vocab_size=vocabSize,
        model_type=modelType,
        character_coverage=characterCoverage,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        num_threads=max(1, (os.cpu_count() or 2) - 1),
    )
    print("tokenizer: saved", tokModelPath, flush=True)

def encodeIfNeeded(sp, reencode):
    if tokensPath.exists() and not reencode:
        print("encode: found", tokensPath, flush=True)
        return

    print("encode: streaming to", tokensPath, flush=True)

    if tokensPath.exists():
        tokensPath.unlink()

    out = open(tokensPath, "ab")

    with open(dataPath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            ids = sp.encode(line.strip(), out_type=int)
            np.array(ids, dtype=np.int32).tofile(out)
            if i % 1000 == 0:
                print("encoded lines:", i, flush=True)

    out.close()
    assert tokensPath.exists(), "tokens.bin not created"
    print("encode: done", flush=True)

def sampleBatch(mem, batchSize, seqLen, device):
    n = mem.shape[0]
    ix = np.random.randint(0, n - seqLen - 1, size=(batchSize,))
    x = np.stack([mem[i:i+seqLen] for i in ix], axis=0)
    y = np.stack([mem[i+1:i+seqLen+1] for i in ix], axis=0)
    return (
        torch.from_numpy(x).to(device).long(),
        torch.from_numpy(y).to(device).long()
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrainTok", action="store_true")
    ap.add_argument("--reencode", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    trainTokenizerIfNeeded(args.retrainTok)

    sp = spm.SentencePieceProcessor()
    sp.load(str(tokModelPath))
    vocab = sp.get_piece_size()
    print("vocab:", vocab, flush=True)

    encodeIfNeeded(sp, args.reencode)

    mem = np.memmap(tokensPath, dtype=np.int32, mode="r")
    print("tokens:", mem.shape[0], flush=True)

    device = "cpu"
    if not args.cpu and torch.backends.mps.is_available():
        device = "mps"
    print("device:", device, flush=True)

    model = tinyLM(vocab, dim, nLayers, maxLen).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    pbar = tqdm(total=steps)

    for step in range(1, steps + 1):
        x, y = sampleBatch(mem, batchSize, seqLen, device)
        i, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        pbar.update(1)

    pbar.close()
    exportWeights(model, vocab)
    print("done. weights exported.", flush=True)

if __name__ == "__main__":
    main()
