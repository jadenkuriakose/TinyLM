import os
import re
import unicodedata
from datasets import load_dataset

outPath = os.path.join(os.path.dirname(__file__), "..", "data", "train.txt")
os.makedirs(os.path.dirname(outPath), exist_ok=True)

def cleanLine(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)

    s = s.replace("\u201c", "\"").replace("\u201d", "\"")
    s = s.replace("\u2018", "'").replace("\u2019", "'")
    s = s.replace("\u2014", "-").replace("\u2013", "-")
    s = s.replace("\u2026", "...")

    s = re.sub(r"[\u0000-\u001f\u007f-\u009f]", " ", s)

    s = re.sub(r"\s+", " ", s).strip()
    return s

ds = load_dataset("wikitext", "wikitext-2-raw-v1")
lines = ds["train"]["text"]

with open(outPath, "w", encoding="utf-8") as f:
    for line in lines:
        line = cleanLine(line)
        if len(line) == 0:
            continue

        if re.match(r"^=+ .* =+$", line):
            continue

        f.write(line + "\n")

print("saved:", outPath)
