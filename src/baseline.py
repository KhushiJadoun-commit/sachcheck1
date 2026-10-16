import time, joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

df = pd.read_csv("data/processed/all.csv")
train = df[df["split"] == "train"]
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), max_features=300000, sublinear_tf=True)),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
])
pipe.fit(train["text"], train["label"])
part = df[df["split"] == "test"].copy()
t0 = time.time()
part["pred"] = pipe.predict(part["text"])
ms = (time.time() - t0) / len(part) * 1000
part["ok"] = (part["label"] == part["pred"]).astype(float)
print("TEST n=%d latency=%.2f ms" % (len(part), ms))
print(classification_report(part["label"], part["pred"], target_names=["real", "fake"]))
print("by lang  :", part.groupby("lang")["ok"].mean().round(3).to_dict())
print("by source:", part.groupby("source")["ok"].mean().round(3).to_dict())
joblib.dump(pipe, "baseline.joblib")
