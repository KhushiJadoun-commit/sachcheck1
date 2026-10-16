import pandas as pd, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data/processed/all.csv")
hi = df[df["lang"] == "hi"]
tr, va = hi[hi["split"] == "train"], hi[hi["split"] == "val"]

vec = TfidfVectorizer(token_pattern=r"\S+", min_df=5, max_features=50000)
clf = LogisticRegression(max_iter=1000)
clf.fit(vec.fit_transform(tr["text"]), tr["label"])
print("Hindi-only word model, val accuracy:",
      round(clf.score(vec.transform(va["text"]), va["label"]), 3))

words = np.array(vec.get_feature_names_out())
order = np.argsort(clf.coef_[0])
print("\nTop 30 words for REAL (label 0):\n", " | ".join(words[order[:30]]))
print("\nTop 30 words for FAKE (label 1):\n", " | ".join(words[order[-30:]]))
