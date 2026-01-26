import os
import re
import unicodedata
from datasets import load_dataset

outPath = os.path.join(os.path.dirname(__file__), "..", "data", "train.txt")
os.makedirs(os.path.dirname(outPath), exist_ok=True)

MAX_LINES = 80000

def cleanLine(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)

    s = s.replace("\u201c", "\"").replace("\u201d", "\"")
    s = s.replace("\u2018", "'").replace("\u2019", "'")
    s = s.replace("\u2014", "-").replace("\u2013", "-")
    s = s.replace("\u2026", "...")

    s = re.sub(r"[\u0000-\u001f\u007f-\u009f]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

datasetsToLoad = [
    ("wikitext", "wikitext-2-raw-v1"),
    ("roneneldan/TinyStories", None),
    ("openwebtext", None),
]

lineCount = 0

with open(outPath, "w", encoding="utf-8") as f:
    for name, subset in datasetsToLoad:
        if name == "wikitext":
            ds = load_dataset(name, subset, split="train", streaming=True)
        else:
            ds = load_dataset(name, split="train", streaming=True)

        for ex in ds:
            line = ex["text"]
            line = cleanLine(line)
            if len(line) == 0:
                continue

            if re.match(r"^=+ .* =+$", line):
                continue

            f.write(line + "\n")
            lineCount += 1

            if lineCount >= MAX_LINES:
                break

        if lineCount >= MAX_LINES:
            break

print("saved:", outPath, "lines:", lineCount)
