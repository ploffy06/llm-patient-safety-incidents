import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
from helper import get_queries

df = pd.read_csv("dataset/dataset.csv")
query1, query2 = get_queries()

# models
model_name = "deepset/roberta-base-squad2"
model = pipeline("question-answering", model=model_name, tokenizer=model_name)

for idx, row in df.iterrows():
    context = row["description"]
    report_id = row["id"]

    answer1 = model({
        "question": query1,
        "context": context
    })

    answer2 = model({
        "question": query2,
        "context": context
    })

    print(f"{idx}: {report_id}")
    print("context: ", context)
    print("summary: ", answer1["answer"])
    print("incident_type: ", answer2["answer"])
    print("confidence: ", answer1["score"], answer2["score"])
    print("--------------\n")