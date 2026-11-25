import pandas as pd

DATA_DIR = r"C:\Users\NAVEEN\OneDrive\Desktop\ML_Lab_Datasets_and_Programs\datasets"

data = pd.read_csv(DATA_DIR + r"\enjoysport.csv")
print("Training data:\n", data, "\n")

attributes = data.iloc[:, :-1].values
target = data.iloc[:, -1].values

# Initialise h with the first positive example
for i, label in enumerate(target):
    if label == "Yes":
        hypothesis = list(attributes[i])
        break

print("Initial hypothesis:", hypothesis)
for i, row in enumerate(attributes):
    if target[i] == "Yes":                      # FIND-S ignores negative examples
        for j in range(len(hypothesis)):
            if row[j] != hypothesis[j]:
                hypothesis[j] = "?"             # generalise the attribute
    print(f"After example {i+1}: {hypothesis}")

print("\nMost specific hypothesis:", hypothesis)
