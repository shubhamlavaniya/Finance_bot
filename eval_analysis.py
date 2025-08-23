import pandas as pd

df = pd.read_csv("evaluation/eval_results.csv")

# Group by method to calculate averages
grouped_results = df.groupby("method").agg(
    avg_time_s=('time_s', 'mean'),
    avg_semantic_score=('semantic_score', 'mean'),
    correctness_rate=('correct', lambda x: (x == 'Y').mean())
)

print(grouped_results)