import pandas as pd
import numpy as np

DATA_DIR = r"C:\Users\NAVEEN\OneDrive\Desktop\ML_Lab_Datasets_and_Programs\datasets"

df = pd.read_csv(DATA_DIR + r"\playtennis.csv")
target = "PlayTennis"

def entropy(col):
    p = col.value_counts(normalize=True)
    return -(p * np.log2(p)).sum()

def info_gain(df, attr):
    total = entropy(df[target])
    weighted = sum(len(s) / len(df) * entropy(s[target]) for _, s in df.groupby(attr))
    return total - weighted

def id3(df, attrs):
    if df[target].nunique() == 1:
        return df[target].iloc[0]
    if not attrs:
        return df[target].mode()[0]
    best = max(attrs, key=lambda a: info_gain(df, a))
    tree = {best: {}}
    for val, subset in df.groupby(best):
        tree[best][val] = id3(subset, [a for a in attrs if a != best])
    return tree

def classify(tree, sample):
    if not isinstance(tree, dict):
        return tree
    attr = next(iter(tree))
    return classify(tree[attr][sample[attr]], sample)

def show(tree, indent=0):
    if not isinstance(tree, dict):
        print(" " * indent + "->", tree); return
    for attr, branches in tree.items():
        for val, sub in branches.items():
            print(" " * indent + f"{attr} = {val}")
            show(sub, indent + 4)

attrs = [c for c in df.columns if c != target]
print(f"Entropy of dataset: {entropy(df[target]):.4f}")
for a in attrs:
    print(f"Information gain({a}) = {info_gain(df, a):.4f}")

tree = id3(df, attrs)
print("\nDecision tree:")
show(tree)

new = {"Outlook": "Sunny", "Temperature": "Cool", "Humidity": "High", "Wind": "Strong"}
print("\nNew sample:", new)
print("Predicted class:", classify(tree, new))
