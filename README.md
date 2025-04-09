# llm-patient-safety-incidents

## About
Using open-source large language models (LLMs) to summarize patient safety incident reports with two main goals:
- extract key information about what caused the incident and
- categorize the safety issue (e.g. medication errors or patient falls).

## Dataset
The dataset contains 20 incident reports that were generated online with ChatGPT in which the word count of each report ranges between 77-94. I created the dataset first
then manually labelled the incident type.

![word count of reports range from 46-94 words](word_counts.png)


## Libraries Used
- pandas
- uuid
- transformers
- matplotlib

## Optimization

## Models Considered
### [bart-large-cnn](https://huggingface.co/facebook/bart-large-cnn)
This model is bart-large trained on MultiNLI dataset in which the hypothesis is "This text is bout {label}", making it a good zero-shot classifier.
BART is a transformer model with a bidirectional encoder and autoregressive decoder - making it good for text comprehension and generation.


### [roberta-base-squad2](https://huggingface.co/deepset/roberta-base-squad2)
Roberta-base-squad2 is fine-tuned on RoBERTa base using SQuAD2 dataset for extractive answer questioning. RoBERTa us a transformer model
petrained for Masked Language Modelling (MLM). Standford Question Answering Dataset (SQuAD) comprises of contexts, questions and answers crowdworked on wikipedia articles. The answers are a segment of text from the context or otherwise unanswerable.

I chose to trial this model to test how question-answering models may perform for this task as we wanted to specifically find out what causaed the incident. Further, I tried this model on classifying the safety issue.

### [PatientSeek](https://huggingface.co/whyhow-ai/PatientSeek)


## Final Chosen Model

## Evaluation