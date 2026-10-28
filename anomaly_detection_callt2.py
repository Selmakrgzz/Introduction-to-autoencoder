from datasets import load_dataset

calit2 = load_dataset("PLAN-Lab/mTSBench", "CalIt2")

df_train = calit2["train"].to_pandas()
df_test = calit2["test"].to_pandas()
