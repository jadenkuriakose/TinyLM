import torch
import torch.nn as nn
import numpy as np
import os

dim = 128
vocab = 256
maxLen = 256

weightsDir = os.path.join(os.path.dirname(__file__), "..", "weights")

def loadBin(path, shape):
    arr = np.fromfile(path, dtype=np.float32)
    return torch.from_numpy(arr.reshape(shape))

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

    def forward(self, x, cacheK, cacheV, pos):
        xNorm = self.attnNorm(x)

        q = self.qProj(xNorm)
        k = self.kProj(xNorm)
        v = self.vProj(xNorm)

        cacheK[pos] = k
        cacheV[pos] = v

        scores = (cacheK[:pos + 1] @ q.T).squeeze(-1) / np.sqrt(dim)
        probs = torch.softmax(scores, dim=0)
        attn = (probs.unsqueeze(-1) * cacheV[:pos + 1]).sum(dim=0, keepdim=True)

        x = x + self.oProj(attn)

        hNorm = self.mlpNorm(x)
        h = self.gelu(self.fc1(hNorm))
        x = x + self.fc2(h)

        return x

class tinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.block = tinyBlock(dim)
        self.lmHead = nn.Linear(dim, vocab)

    def forward(self, tokenId, cacheK, cacheV, pos):
        x = self.embed(torch.tensor([tokenId]))
        x = self.block(x, cacheK, cacheV, pos)
        return self.lmHead(x)

def loadWeights(m):
    m.block.qProj.weight.data = loadBin(os.path.join(weightsDir, "qProj.bin"), (dim, dim))
    m.block.qProj.bias.data   = loadBin(os.path.join(weightsDir, "qProjBias.bin"), (dim,))

    m.block.kProj.weight.data = loadBin(os.path.join(weightsDir, "kProj.bin"), (dim, dim))
    m.block.kProj.bias.data   = loadBin(os.path.join(weightsDir, "kProjBias.bin"), (dim,))

    m.block.vProj.weight.data = loadBin(os.path.join(weightsDir, "vProj.bin"), (dim, dim))
    m.block.vProj.bias.data   = loadBin(os.path.join(weightsDir, "vProjBias.bin"), (dim,))

    m.block.oProj.weight.data = loadBin(os.path.join(weightsDir, "oProj.bin"), (dim, dim))
    m.block.oProj.bias.data   = loadBin(os.path.join(weightsDir, "oProjBias.bin"), (dim,))

    m.block.fc1.weight.data = loadBin(os.path.join(weightsDir, "fc1.bin"), (4 * dim, dim))
    m.block.fc1.bias.data   = loadBin(os.path.join(weightsDir, "fc1Bias.bin"), (4 * dim,))

    m.block.fc2.weight.data = loadBin(os.path.join(weightsDir, "fc2.bin"), (dim, 4 * dim))
    m.block.fc2.bias.data   = loadBin(os.path.join(weightsDir, "fc2Bias.bin"), (dim,))

    m.block.attnNorm.weight.data = loadBin(os.path.join(weightsDir, "attnNorm.bin"), (dim,))
    m.block.mlpNorm.weight.data  = loadBin(os.path.join(weightsDir, "mlpNorm.bin"), (dim,))

    m.lmHead.weight.data = loadBin(os.path.join(weightsDir, "lmHead.bin"), (vocab, dim))
    m.lmHead.bias.data   = loadBin(os.path.join(weightsDir, "lmHeadBias.bin"), (vocab,))

    if os.path.exists(os.path.join(weightsDir, "embed.bin")):
        m.embed.weight.data = loadBin(os.path.join(weightsDir, "embed.bin"), (vocab, dim))

m = tinyLM().eval()
loadWeights(m)

cacheK = torch.zeros(maxLen, dim)
cacheV = torch.zeros(maxLen, dim)

tokenId = ord("h")
logits = m(tokenId, cacheK, cacheV, 0)

print("pytorch logits[:8]:", logits[0, :8].detach().cpu().numpy())
