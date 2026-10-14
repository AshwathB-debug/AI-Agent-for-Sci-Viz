import pandas as pd

pd.read_json("data/runs.jsonl", lines=True).to_csv("runs.csv", index=False)
# pd.read_json("data/feedback.jsonl", lines=True).to_csv("feedback.csv", index=False)