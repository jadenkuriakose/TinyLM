import os
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

dim = 128
vocab = 256
maxLen = 256
nLayers = 4

batchSize = 16
seqLen = 256
lr = 3e-4
steps = 20000
logEvery = 100

dataPath = os.path.join(os.path.dirname(__file__), "..", "data", "train.txt")
outDir = os.path.join(os.path.dirname(__file__), "..", "weights")

os.makedirs(outDir, exist_ok=True)

def readBytes(path):
    with open(path, "rb") as f:
        return np.frombuffer(f.read(), dtype=np.uint8)

data = readBytes(dataPath)
if len(data) < seqLen + 2:
    raise RuntimeError("data too small, put more text in data/train.txt")

def getBatch():
    ix = np.random.randint(0, len(data) - seqLen - 1, size=(batchSize,))
    x = np.stack([data[i:i+seqLen] for i in ix], axis=0)
    y = np.stack([data[i+1:i+seqLen+1] for i in ix], axis=0)
    return torch.from_numpy(x).long(), torch.from_numpy(y).long()

class rmsNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight

class tinyBlock(nn.Module):
    def __init__(self, dim):
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

    def forward(self, x):
        b, t, d = x.shape
        xNorm = self.attnNorm(x)

        q = self.qProj(xNorm)
        k = self.kProj(xNorm)
        v = self.vProj(xNorm)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(d)
        mask = torch.tril(torch.ones(t, t, device=x.device))
        scores = scores.masked_fill(mask == 0, -1e9)

        probs = F.softmax(scores, dim=-1)
        attn = probs @ v

        x = x + self.oProj(attn)

        hNorm = self.mlpNorm(x)
        h = self.gelu(self.fc1(hNorm))
        x = x + self.fc2(h)

        return x

class tinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList([tinyBlock(dim) for _ in range(nLayers)])
        self.lmHead = nn.Linear(dim, vocab)

    def forward(self, idx, targets=None):
        x = self.embed(idx)
        for b in self.blocks:
            x = b(x)
        logits = self.lmHead(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, vocab), targets.view(-1))
        return logits, loss

def dump(t, path):
    t.detach().cpu().numpy().astype("float32").tofile(path)

def exportWeights(m):
    dump(m.embed.weight, os.path.join(outDir, "embed.bin"))
    dump(m.lmHead.weight, os.path.join(outDir, "lmHead.bin"))
    dump(m.lmHead.bias,   os.path.join(outDir, "lmHeadBias.bin"))

    for i, b in enumerate(m.blocks):
        layerDir = os.path.join(outDir, f"layer{i}")
        os.makedirs(layerDir, exist_ok=True)

        dump(b.qProj.weight, os.path.join(layerDir, "qProj.bin"))
        dump(b.qProj.bias,   os.path.join(layerDir, "qProjBias.bin"))

        dump(b.kProj.weight, os.path.join(layerDir, "kProj.bin"))
        dump(b.kProj.bias,   os.path.join(layerDir, "kProjBias.bin"))

        dump(b.vProj.weight, os.path.join(layerDir, "vProj.bin"))
        dump(b.vProj.bias,   os.path.join(layerDir, "vProjBias.bin"))

        dump(b.oProj.weight, os.path.join(layerDir, "oProj.bin"))
        dump(b.oProj.bias,   os.path.join(layerDir, "oProjBias.bin"))

        dump(b.fc1.weight, os.path.join(layerDir, "fc1.bin"))
        dump(b.fc1.bias,   os.path.join(layerDir, "fc1Bias.bin"))

        dump(b.fc2.weight, os.path.join(layerDir, "fc2.bin"))
        dump(b.fc2.bias,   os.path.join(layerDir, "fc2Bias.bin"))

        dump(b.attnNorm.weight, os.path.join(layerDir, "attnNorm.bin"))
        dump(b.mlpNorm.weight,  os.path.join(layerDir, "mlpNorm.bin"))

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = tinyLM().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=lr)

t0 = time.time()
for step in range(1, steps + 1):
    x, y = getBatch()
    x = x.to(device)
    y = y.to(device)

    i, loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

    if step % logEvery == 0:
        dt = time.time() - t0
        print(f"step {step}/{steps} loss {loss.item():.4f} time {dt:.1f}s")
        t0 = time.time()

exportWeights(model)
print("training complete, weights exported")
