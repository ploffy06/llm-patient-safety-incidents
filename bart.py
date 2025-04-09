from transformers import pipeline
import pandas as pd
from helper import incident_types

df = pd.read_csv("dataset.csv")

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

for idx, row in df.iterrows():
    sequence_to_classify = row["description"]
    candidate_labels = incident_types
    label = classifier(sequence_to_classify, candidate_labels)
    print(label)

    if idx == 3:
        break

