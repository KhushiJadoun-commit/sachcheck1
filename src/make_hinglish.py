import re
import pandas as pd
from aksharamukha import transliterate

SEED = 42
PER_CLASS = {"train": 1500, "val": 300, "test": 300}
WORDS = 60

df = pd.read_csv("data/processed/all.csv")
df = df[(df["source"] != "hinfakenews_translit") & (df["lang"] != "hinglish")]
hi = df[(df["lang"] == "hi") & (df["source"] == "hinfakenews")]
picked = []
for split, n in PER_CLASS.items():
    for lab in [0, 1]:
        part = hi[(hi["split"] == split) & (hi["label"] == lab)]
        picked.append(part.sample(min(n, len(part)), random_state=SEED))
h = pd.concat(picked).copy()

def to_hinglish(t):
    t = " ".join(str(t).split()[:WORDS])
    r = transliterate.process("Devanagari", "RomanColloquial", t)
    return re.sub(r"\s+", " ", r.lower()).strip()

h["text"] = [to_hinglish(t) for t in h["text"]]
h["lang"] = "hinglish"
h["source"] = "hinfakenews_translit"
df = pd.concat([df, h], ignore_index=True)
df.to_csv("data/processed/all.csv", index=False, encoding="utf-8")
print(df.groupby(["lang", "label"]).size().unstack(fill_value=0))
print("SAMPLES:")
for t in h["text"].head(4):
    print("*", t[:120])
