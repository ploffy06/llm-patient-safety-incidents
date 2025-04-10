import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
import spacy

df = pd.read_csv("dataset/dataset.csv")
nlp = spacy.load("en_core_web_md")
targets = list(pd.read_csv("dataset/targets.csv")["cause"])
query = "What caused the incident?"

# models
model_name = "deepset/roberta-base-squad2"
model = pipeline("question-answering", model=model_name, tokenizer=model_name)
similarity_scores = []
confidence_scores = []

for idx, row in df.iterrows():
    context = row["description"]
    report_id = row["id"]
    target = targets[idx]

    prediction = model({
        "question": query,
        "context": context
    })

    target_nlp = nlp(target)
    prediction_nlp = nlp(prediction["answer"])
    similarity = target_nlp.similarity(prediction_nlp) # cosine similarity of embedding vectors
    similarity_scores.append(similarity)
    confidence_scores.append(prediction["score"])

    is_correct = False
    print(f"{idx}: {report_id}")
    print(f"actual: {target}")
    print("predicted cause: ", prediction["answer"])
    print("similarity: ", similarity)
    print("confidence: ", prediction["score"])
    print("--------------\n")

plt.figure(figsize=(5,5))
plt.scatter(confidence_scores, similarity_scores)
plt.title("Similarity vs Confidence")
plt.xlabel("Confidence Score")
plt.ylabel("Similarity Score")
plt.savefig('diagrams/roberta_sim_conf.png')