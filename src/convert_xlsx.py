import pandas as pd

df = pd.read_excel("data/raw/HinFakeNews-V1.xlsx")
print(df.shape)
print(list(df.columns))
df.to_csv("data/raw/hinfakenews.csv", index=False, encoding="utf-8")
print("Saved data/raw/hinfakenews.csv")