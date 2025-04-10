from teapotai import TeapotAI
import pandas as pd
from helper import get_queries
from transformers import pipeline

df = pd.read_csv("dataset/dataset.csv")
query1, query2 = get_queries()

# teapot_ai = TeapotAI()
teapot = pipeline("question-answering", model="teapotai/teapotllm")


for idx, row in df.iterrows():
    context = row["description"]

    answer = teapot.query(
        query=query1,
        context=context
    )
    print(answer)

    if idx == 2:
        break
