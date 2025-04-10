from transformers import pipeline
import pandas as pd
from helper import get_incident_types
import matplotlib.pyplot as plt
from sklearn import metrics

df = pd.read_csv("dataset/dataset.csv")
targets = list(pd.read_csv("dataset/incident_types.csv")["incident_type"])


classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
labels = get_incident_types()
predictions = []

for idx, row in df.iterrows():
    sequence_to_classify = row["description"]
    report_id = row["id"]

    pred = classifier(sequence_to_classify, labels)
    predictions.append(pred["labels"][0])
    is_correct = pred["labels"][0] == targets[idx]

    print(f"idx: {idx}, id: {report_id}")
    print(f"actual: {targets[idx]}, is_correct: {is_correct}")
    print(pred["scores"])
    print(pred["labels"])
    print("---------\n")

print("accuracy: ", metrics.accuracy_score(targets, predictions))
cr = metrics.classification_report(targets, predictions, target_names=labels)
print(cr)

confusion_matrix = metrics.confusion_matrix(targets, predictions)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=labels)
plt.figure(figsize=(10,10))
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.title('Confusion Matrix',fontsize=16)
cm_display.plot(xticks_rotation="vertical")
plt.tight_layout()
plt.savefig("diagrams/bart_cm.png")
