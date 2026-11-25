import pandas as pd
import numpy as np

DATA_DIR = r"C:\Users\NAVEEN\OneDrive\Desktop\ML_Lab_Datasets_and_Programs\datasets"

data = pd.read_csv(DATA_DIR + r"\enjoysport.csv")
concepts = np.array(data.iloc[:, :-1])
target = np.array(data.iloc[:, -1])
n = concepts.shape[1]

specific_h = concepts[0].copy()
general_h = [["?" for _ in range(n)] for _ in range(n)]
print("Initial S:", list(specific_h))
print("Initial G:", general_h, "\n")

for i, h in enumerate(concepts):
    if target[i] == "Yes":
        for x in range(n):
            if h[x] != specific_h[x]:
                specific_h[x] = "?"
                general_h[x][x] = "?"
    if target[i] == "No":
        for x in range(n):
            if h[x] != specific_h[x]:
                general_h[x][x] = specific_h[x]
            else:
                general_h[x][x] = "?"
    print(f"Step {i+1}\n  S = {list(specific_h)}\n  G = {general_h}")

# remove fully general hypotheses
general_h = [g for g in general_h if g != ["?"] * n]
print("\nFinal Specific hypothesis S:", list(specific_h))
print("Final General hypotheses G:", general_h)
