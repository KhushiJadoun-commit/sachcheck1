import time, joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("data/processed/all.csv")
train = df[df["split"] == "train"]

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                              max_features=300_000, sublinear_tf=True)),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
])
pipe.fit(train["text"], train["label"])

for name in ["val", "test"]:
    part = df[df["split"] == name].copy()
    t0 = time.time()
    part["pred"] = pipe.predict(part["text"])
    ms = (time.time() - t0) / len(part) * 1000
    part["correct"] = (part["label"] == part["pred"]).astype(float)
    print("=" * 60)
    print(f"{name.upper()}  n={len(part)}  latency={ms:.2f} ms/sample")
    print(classification_report(part["label"], part["pred"], target_names=["real", "fake"]))
    print(confusion_matrix(part["label"], part["pred"]))
    print("accuracy by lang  :", part.groupby("lang")["correct"].mean().round(3).to_dict())
    print("accuracy by source:", part.groupby("source")["correct"].mean().round(3).to_dict())

joblib.dump(pipe, "baseline.joblib")