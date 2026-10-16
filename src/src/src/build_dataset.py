import re
import pandas as pd
from sklearn.model_selection import train_test_split

print("BUILD VERSION 3")

# ---------------- SETTINGS ----------------
HINDI_FAKE_VALUE = 0     # confirmed: BOOL 0 = fake
ENGLISH_NEWS_N = 12000
HINDI_PER_CLASS = 6000
EN_WORDS, HI_WORDS = 120, 100
SEED = 42
# ------------------------------------------

DEV = re.compile(r"[\u0900-\u097F]")
D = r"\u0900-\u097F"   # Devanagari range, used for word-edge checks
HINGLISH_HINTS = {"hai", "nahi", "kya", "ke", "ki", "ka", "ko", "mein", "aur",
                  "sarkar", "bhai", "yeh", "ye", "tha", "hain", "kar", "raha"}

MARATHI = re.compile(r"(?:^|\s)(?:आहे|आहेत|यांनी|आणि|त्यांनी|वाचा|नाही|असून|म्हणाले)(?:\s|$|[.,।])")

# Sentences containing any of these are dropped from ALL Hindi articles (both labels).
LEAK = re.compile(
    r"फर्जी|झूठ|भ्रामक|असत्य|अफवाह|गलत|हकीकत|सच्चाई|सत्यता|पड़ताल|"
    r"फैक्ट|जांच में|दावा|दावे|वायरल|सोशल मीडिया|पोस्ट|मैसेज|संदेश|यूजर|"
    r"फेसबुक|व्हाट्सऐप|व्हाट्सएप|ट्विटर|ट्वीट|यूट्यूब|"
    r"पीआईबी|PIB|विश्वास न्यूज|न्यूजचेकर|बूम|ऑल्ट न्यूज|एंटी फेक|वॉर रूम|"
    r"क्या है सच|सच क्या|असली सच|सत्य|vishvas|vishwas|vishva|"
    r"फेक|सर्च|वेबसाइट|शेयर|हमने|हमें|हमारी|न्यूज|"
    r"वीडियो|तस्वीर|फोटो|"
    rf"(?<![{D}])सच(?![{D}])|"
    r"fact[\s-]?check|fake|false|misleading|hoax|debunk|viral|boomlive|altnews|factly|newschecker|news",
    re.IGNORECASE)

WEEKDAYS = re.compile(r"सोमवार|मंगलवार|बुधवार|गुरुवार|बृहस्पतिवार|शुक्रवार|शनिवार|रविवार")

def clean_text(t):
    t = str(t).replace("_x000D_", " ").replace("\u200c", "").replace("\u200d", "")
    t = re.sub(r"http\S+|www\.\S+", " ", t)
    t = re.sub(r"@\w+", " ", t)
    t = re.sub(r"[-_=]{3,}", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def trunc(t, n):
    return " ".join(t.split()[:n])

def detect_lang(t):
    if DEV.search(t):
        return "hi"
    words = set(re.findall(r"[a-z]+", t.lower()))
    return "hinglish" if len(words & HINGLISH_HINTS) >= 2 else "en"

def strip_leaks(text):
    text = str(text).replace("\u093c", "")          # drop nukta: ज़ -> ज, so spelling styles match
    sents = re.split(r"(?<=[।.!?])\s+|\n+", text)
    kept = [s for s in sents if not LEAK.search(s)]
    return " ".join(kept), len(sents) - len(kept)

def hindi_normalize(t):
    t = re.sub(r"[।.,!?:;\"'‘’“”()\[\]\-–—|]", " ", t)   # punctuation style is an outlet fingerprint
    t = WEEKDAYS.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()

def add_split(df):
    tr, tmp = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=SEED)
    va, te = train_test_split(tmp, test_size=0.5, stratify=tmp["label"], random_state=SEED)
    return pd.concat([tr.assign(split="train"), va.assign(split="val"), te.assign(split="test")])

def finish(df, source):
    df = df[["text", "label"]].copy()
    df["text"] = df["text"].map(clean_text)
    df = df[df["text"].str.split().str.len() >= 8]
    df["lang"] = df["text"].map(detect_lang)
    df["source"] = source
    return df

parts = []

# ---------- 1. Constraint (English tweets); official test file kept as test ----------
lab = {"real": 0, "fake": 1}
for fname, split in [("Constraint_Train.csv", "train"), ("Constraint_Val.csv", "val"),
                     ("english_test_with_labels.csv", "test")]:
    c = pd.read_csv(f"data/raw/{fname}").rename(columns={"tweet": "text"})
    c["label"] = c["label"].map(lab)
    c = finish(c.dropna(subset=["text", "label"]), "constraint")
    c["split"] = split
    parts.append(c)

# ---------- 2. WELFake (original 0=fake,1=real -> flipped) ----------
w = pd.read_csv("data/raw/WELFake_Dataset.csv").dropna(subset=["label"])
w["text"] = w["title"].fillna("").astype(str) + ". " + w["text"].fillna("").astype(str)
w["label"] = 1 - w["label"].astype(int)
w = w.sample(ENGLISH_NEWS_N, random_state=SEED)
w["text"] = w["text"].map(lambda t: trunc(clean_text(t), EN_WORDS))
parts.append(add_split(finish(w, "welfake")))

# ---------- 3. HinFakeNews (Hindi) ----------
h = pd.read_csv("data/raw/hinfakenews.csv").dropna(subset=["CONTENT", "BOOL"])
h["label"] = (h["BOOL"] == HINDI_FAKE_VALUE).astype(int)       # 1 = fake

is_mr = (h["CONTENT"].astype(str).str.contains(MARATHI)
         | h["TITLE"].astype(str).str.contains(MARATHI))
print("Marathi-like rows removed by label (0=real,1=fake):", h[is_mr]["label"].value_counts().to_dict())
h = h[~is_mr].copy()

h["text"] = h["TITLE"].fillna("").astype(str).str.strip() + ". " + h["CONTENT"].astype(str)
res = h["text"].map(strip_leaks)
h["text"] = res.map(lambda r: trunc(hindi_normalize(clean_text(r[0])), HI_WORDS))
h["dropped"] = res.map(lambda r: r[1])
print("Leak sentences removed per article (avg) by label:", h.groupby("label")["dropped"].mean().round(2).to_dict())
h = finish(h, "hinfakenews")
h = h[h["lang"] == "hi"]
print("Hindi articles left after cleaning:", h["label"].value_counts().to_dict(), "\n")
n = min(HINDI_PER_CLASS, h["label"].value_counts().min())
h = h.groupby("label", group_keys=False).sample(n, random_state=SEED)
parts.append(add_split(h))

# ---------- Merge, dedupe (test copies win) ----------
df = pd.concat(parts, ignore_index=True)
df["_o"] = df["split"].map({"test": 0, "val": 1, "train": 2})
df = df.sort_values("_o").drop_duplicates(subset="text", keep="first").drop(columns="_o")
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
df[["text", "label", "lang", "source", "split"]].to_csv(
    "data/processed/all.csv", index=False, encoding="utf-8")

print("Saved data/processed/all.csv", df.shape, "\n")
print("Language x label:\n", df.groupby(["lang", "label"]).size().unstack(fill_value=0), "\n")
print("Source x split:\n", df.groupby(["source", "split"]).size().unstack(fill_value=0), "\n")

hi = df[df["lang"] == "hi"]
print("Median words by label:", hi.assign(w=hi["text"].str.split().str.len()).groupby("label")["w"].median().to_dict())
print("\nSAMPLES (label 1 = fake)")
for lb in [0, 1]:
    print(f"\n--- label {lb} ---")
    for t in hi[hi["label"] == lb]["text"].head(4):
        print("*", t[:160])