import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt

df = pd.read_csv("dataset.csv")
query1 = "Summuraize what contributed to the incident"
query2 = "Categorise the safety issue"

# models
model_name = "deepset/roberta-base-squad2"
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)

for idx, row in df.iterrows():
    context = row["description"]
    report_id = row["id"]

    answer1 = nlp({
        "question": query1,
        "context": context
    })

    answer2 = nlp({
        "question": query2,
        "context": context
    })

    print(f"{idx}: {report_id}")
    print("context: ", context)
    print("Summary: ", answer1["answer"])
    print("Category: ", answer2["answer"])
    print("confidence: ", answer1["score"], answer2["score"])
    print("--------------\n")