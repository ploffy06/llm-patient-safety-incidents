import pandas as pd
import uuid
import matplotlib.pyplot as plt

file = open("dataset/corpus.txt", "r")
lines = file.readlines()
file.close()
reports = [line.replace("\n", "") for line in lines]

file = open("dataset/incident_types.txt", "r")
lines = file.readlines()
file.close
incident_types = [line.replace("\n", "") for line in lines]


ids = [uuid.uuid4() for _ in range(len(reports))]

reports_df = pd.DataFrame(
    {
        "id": ids,
        "description": reports
    }
)

incident_types_df = pd.DataFrame(
    {
        "id": ids,
        "incident_type": incident_types
    }
)

reports_df.to_csv("dataset/dataset.csv")
incident_types_df.to_csv("dataset/incident_types.csv")

# see distribution of word count
word_counts = sorted(list(reports_df["description"].str.count(" ") + 1))

plt.figure(figsize=(10, 5))
plt.hist(word_counts, bins=range(min(word_counts), max(word_counts) + 2), edgecolor='black', align='left')
plt.title('Distribution of Word Count')
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.xticks(range(min(word_counts), max(word_counts) + 1))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('diagrams/word_counts.png')

# see distribution of incident types
incident_types = sorted(incident_types)

plt.figure(figsize=(10,5))
plt.hist(incident_types, bins=len(set(incident_types)), edgecolor='black', alpha=0.7)
plt.title("Incident Type Distribution")
plt.xlabel("Incident Type")
plt.ylabel("Frequency")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('diagrams/incident_types.png')
