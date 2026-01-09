import torch
import torch.nn as nn
import os

dim = 64
vocab = 256
maxLen = 256

exportDir = os.path.join(os.path.dirname(__file__), "..", "weights")
os.makedirs(exportDir, exist_ok=True)

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

        scores = (cacheK[:pos + 1] @ q.T).squeeze(-1)
        attn = (scores.unsqueeze(-1) * cacheV[:pos + 1]).sum(dim=0, keepdim=True)

        x = x + self.oProj(attn)

        hNorm = self.mlpNorm(x)
        h = self.gelu(self.fc1(hNorm))
        x = x + self.fc2(h)

        return x


class tinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = tinyBlock(dim)
        self.lmHead = nn.Linear(dim, vocab)

    def forward(self, tokenId, cacheK, cacheV, pos):
        x = torch.full((1, dim), tokenId / 255.0)
        h = self.block(x, cacheK, cacheV, pos)
        return self.lmHead(h)


model = tinyLM().eval()


def dump(tensor, name):
    tensor.detach().cpu().numpy().astype("float32").tofile(
        os.path.join(exportDir, name)
    )


dump(model.block.qProj.weight, "qProj.bin")
dump(model.block.qProj.bias,   "qProjBias.bin")

dump(model.block.kProj.weight, "kProj.bin")
dump(model.block.kProj.bias,   "kProjBias.bin")

dump(model.block.vProj.weight, "vProj.bin")
dump(model.block.vProj.bias,   "vProjBias.bin")

dump(model.block.oProj.weight, "oProj.bin")
dump(model.block.oProj.bias,   "oProjBias.bin")

dump(model.block.fc1.weight, "fc1.bin")
dump(model.block.fc1.bias,   "fc1Bias.bin")

dump(model.block.fc2.weight, "fc2.bin")
dump(model.block.fc2.bias,   "fc2Bias.bin")

dump(model.block.attnNorm.weight, "attnNorm.bin")
dump(model.block.mlpNorm.weight,  "mlpNorm.bin")

dump(model.lmHead.weight, "lmHead.bin")
dump(model.lmHead.bias,   "lmHeadBias.bin")

print("weights generated successfully")
