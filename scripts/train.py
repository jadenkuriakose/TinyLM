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
from pathlib import Path
from tqdm import tqdm

dim = 192
nLayers = 16
maxLen = 256

batchSize = 32
seqLen = 128
lr = 3e-4
steps = 8000

vocabSize = 4096

rootDir = Path(__file__).resolve().parent.parent
dataPath = rootDir / "data" / "train.txt"
tokPath = rootDir / "tokenizer" / "tinyLM.model"
tokensPath = rootDir / "data" / "tokens.bin"
weightsDir = rootDir / "weights"

weightsDir.mkdir(exist_ok=True)
(rootDir / "data").mkdir(exist_ok=True)

# -----------------------------
# Model
# -----------------------------

class rmsNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x / (x.pow(2).mean(-1, keepdim=True).sqrt() + 1e-6) * self.w


class block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.n1 = rmsNorm(d)
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)

        self.n2 = rmsNorm(d)
        self.fc1 = nn.Linear(d, 4 * d)
        self.fc2 = nn.Linear(4 * d, d)

    def forward(self, x):
        b,t,d = x.shape
        h = self.n1(x)

        q = self.q(h)
        k = self.k(h)
        v = self.v(h)

        att = (q @ k.transpose(-2,-1)) / math.sqrt(d)
        mask = torch.tril(torch.ones(t,t,device=x.device))
        att = att.masked_fill(mask==0, -1e9)
        att = att.softmax(-1)

        x = x + self.o(att @ v)

        h = self.n2(x)
        x = x + self.fc2(F.gelu(self.fc1(h)))
        return x


class tinyLM(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList([block(dim) for _ in range(nLayers)])
        self.head = nn.Linear(dim, vocab)

    def forward(self, x, y=None):
        x = self.emb(x)
        for b in self.blocks:
            x = b(x)
        logits = self.head(x)
        loss = None
        if y is not None:
            loss = F.cross_entropy(logits.view(-1,logits.size(-1)), y.view(-1))
        return logits, loss


# -----------------------------
# Token Encoding
# -----------------------------

def buildTokens(reencode):
    if tokensPath.exists() and not reencode:
        print("tokens.bin found")
        return

    print("building tokens.bin")

    sp = spm.SentencePieceProcessor()
    sp.load(str(tokPath))

    if tokensPath.exists():
        tokensPath.unlink()

    out = open(tokensPath,"ab")
    with open(dataPath,"r",encoding="utf-8") as f:
        for i,line in enumerate(f):
            ids = sp.encode(line.strip(), out_type=int)
            np.array(ids,dtype=np.int32).tofile(out)
            if i % 1000 == 0:
                print("encoded", i)
    out.close()

    assert tokensPath.exists()
    print("tokens.bin created")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reencode", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    buildTokens(args.reencode)

    mem = np.memmap(tokensPath, dtype=np.int32, mode="r")
    print("tokens:", mem.shape[0])

    device = "cpu"
    if not args.cpu and torch.backends.mps.is_available():
        device="mps"

    model = tinyLM(vocabSize).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    pbar = tqdm(total=steps)

    for s in range(1,steps+1):
        ix = np.random.randint(0, mem.shape[0]-seqLen-1, size=batchSize)
        x = np.stack([mem[i:i+seqLen] for i in ix])
        y = np.stack([mem[i+1:i+seqLen+1] for i in ix])

        x = torch.from_numpy(x).to(device).long()
        y = torch.from_numpy(y).to(device).long()

        _, loss = model(x,y)
        opt.zero_grad()
        loss.backward()
        opt.step()

        pbar.set_postfix(loss=f"{loss.item():.3f}")
        pbar.update(1)

    pbar.close()

    # save weights
    model.emb.weight.detach().cpu().numpy().tofile(weightsDir/"embed.bin")
    model.head.weight.detach().cpu().numpy().tofile(weightsDir/"lmHead.bin")
    model.head.bias.detach().cpu().numpy().tofile(weightsDir/"lmHeadBias.bin")

    print("training done")

if __name__ == "__main__":
    main()
