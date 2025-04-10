from transformers import pipeline
import pandas as pd
from helper import get_incident_types
import matplotlib.pyplot as plt
from sklearn import metrics

df = pd.read_csv("dataset/dataset.csv")
targets = list(pd.read_csv("dataset/targets.csv")["incident_type"])


classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
labels = get_incident_types()
predictions = []
confidence_true = []
confidence_false = []

for idx, row in df.iterrows():
    sequence_to_classify = row["description"]
    report_id = row["id"]

    pred = classifier(sequence_to_classify, labels)
    predicted_class = pred["labels"][0]
    predictions.append(predicted_class)

    if predicted_class == targets[idx]:
        confidence_true.append(pred["scores"][0])
    else:
        confidence_false.append(pred["scores"][0])

    print(idx, report_id)
    print(f"actual: {targets[idx]}")
    print(pred["labels"])
    print(pred["scores"])
    print("---------\n")

print("accuracy: ", metrics.accuracy_score(targets, predictions))
cr = metrics.classification_report(targets, predictions, target_names=labels)
print(cr)

confusion_matrix = metrics.confusion_matrix(targets, predictions)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=labels)
plt.figure(figsize=(10,10))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
cm_display.plot(xticks_rotation="vertical")
plt.tight_layout()
plt.savefig("diagrams/bart_cm.png")

plt.figure(figsize=(10, 5))
plt.hist(confidence_true, alpha=0.7, label="Correct", color="green", density=True)
plt.hist(confidence_false, alpha=0.7, label="Incorrect", color="red", density=True)
plt.xlabel("Confidence Score")
plt.ylabel("Density")
plt.title("Confidence Score Distribution")
plt.legend()
plt.tight_layout()
plt.savefig("diagrams/bart_confidence_distribution.png")
