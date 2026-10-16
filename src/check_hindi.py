import pandas as pd

df = pd.read_csv("data/raw/hinfakenews.csv")
print(df["BOOL"].value_counts())

for lab in [0, 1]:
    print("\n" + "=" * 70)
    print(f"BOOL = {lab}: 3 samples")
    for _, r in df[df["BOOL"] == lab].sample(3, random_state=1).iterrows():
        print("\nTITLE  :", str(r["TITLE"])[:200])
        print("CONTENT:", str(r["CONTENT"])[:400])

# Leakage check: how often do 'verdict' words appear per label?
words = ["फर्जी", "फर्ज़ी", "झूठ", "भ्रामक", "गलत", "फैक्ट चेक", "फैक्ट-चेक",
         "दावा", "वायरल", "fact", "fake", "false", "claim", "viral"]
print("\n" + "=" * 70)
print("Share of articles containing each word (%), BOOL=0 vs BOOL=1")
rows = []
for w in words:
    has = df["CONTENT"].astype(str).str.contains(w, case=False, regex=False)
    rows.append((w, round(has[df["BOOL"] == 0].mean() * 100, 1),
                    round(has[df["BOOL"] == 1].mean() * 100, 1)))
print(pd.DataFrame(rows, columns=["word", "BOOL=0 %", "BOOL=1 %"]).to_string(index=False))

# Length per label (length can be a shortcut too)
df["words"] = df["CONTENT"].astype(str).str.split().str.len()
print("\nMedian word count:\n", df.groupby("BOOL")["words"].median())