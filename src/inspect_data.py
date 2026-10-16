import glob
import pandas as pd

for f in sorted(glob.glob("data/raw/*.csv")):
    try:
        df = pd.read_csv(f, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(f, encoding="latin-1")
    except Exception as e:
        print("=" * 60)
        print(f, "ERROR:", e)
        continue
    print("=" * 60)
    print(f, df.shape)
    print("Columns:", list(df.columns))
    for c in df.columns:
        if df[c].nunique() <= 10:
            print(f"  {c} values:", df[c].value_counts().to_dict())
    txt = df.select_dtypes("object")
    if txt.shape[1]:
        sample = txt.iloc[:200, 0].astype(str).str.cat()
        hi = sum("\u0900" <= ch <= "\u097F" for ch in sample) / max(len(sample), 1)
        print(f"  Hindi-script share (first text col): {hi:.0%}")
    print(df.head(2).to_string()[:500])