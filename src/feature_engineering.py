import pandas as pd

def create_features(df):
    df["newagt"] = df["age"] - df["tenure"]

    df["creditscore_bin"] = pd.qcut(df["creditscore"], 10, labels=False)
    df["age_bin"] = pd.qcut(df["age"], 8, labels=False)
    df["balance_bin"] = pd.qcut(df["balance"].rank(method="first"), 10, labels=False)
    df["salary_bin"] = pd.qcut(df["estimatedsalary"], 10, labels=False)

    df["monthly_salary"] = df["estimatedsalary"] / 12
    return df
