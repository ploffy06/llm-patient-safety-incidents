import pandas as pd
import uuid
import matplotlib.pyplot as plt

file = open("corpus.txt", "r")
lines = file.readlines()
file.close()

reports = [line.replace("\n", "") for line in lines]

ids = [uuid.uuid4() for _ in range(len(reports))]

reports_df = pd.DataFrame(
    {
        "id": ids,
        "description": reports
    }
)

reports_df.to_csv("dataset.csv")

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
plt.savefig('word_counts.png')
